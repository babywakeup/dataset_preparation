# 联合数据集混合训练：实现规格（供 Codex 编码使用）

## 1. 背景与目标

- 训练使用两个子数据集，通过 `ConcatDataset` 拼接：
  - **AtoB**：输入 A（真实图），target B（生成图）。换色效果好，但 B 是合成的，mask 边缘区域存在不均匀 / 伪影，直接监督会把伪影学进模型。
  - **BtoA**：输入 B，target A（真实图）。target 是真实图，边缘质量可信。
- 目标：**保留 AtoB 对换色（颜色/材质语义映射）的监督，削弱它对 mask 边缘的监督**，并让边缘质量主要由 BtoA 决定。
- 需要实现的功能：
  1. 两个子数据集在样本中带有来源标记 `is_atob`
  2. 训练过程中**随 epoch 改变** AtoB / BtoA 的采样比例（自定义 Sampler，兼容 DDP）
  3. 对 AtoB 样本的 **mask 边缘带降权**（alpha 可随 epoch 调整）
  4. 分阶段训练（阶梯式 schedule）的支持
  5. 可选：严格 batch 内比例的 BatchSampler

## 2. 总体方案

| 手段 | 作用 | 优先级 |
|---|---|---|
| AtoB 边缘带 loss 降权 | 直接避免学到合成边缘伪影 | 高 |
| 采样比例随 epoch 变化（AtoB 占比逐步降低） | 前期学换色，后期以真实图为主修边缘 | 高 |
| 分阶段微调（阶梯 schedule） | 阶段一联合训练，阶段二以 BtoA 为主，低学习率 | 中 |
| mask 外区域一致性 loss | 不依赖 B 的质量的干净监督 | 中 |
| domain 标签作为条件（`<real>` / `<synthetic>`），推理固定 `<real>` | 让模型把合成风格作为可分离因素 | 可选 |
| 对 B 重合成：`B' = mask*B + (1-mask)*A`（mask 羽化） | 从数据源头修复 | 可选，数据工程 |

推荐的两阶段配置：

| 阶段 | 数据配比 (AtoB 概率) | 边缘处理 | 学习率 |
|---|---|---|---|
| 一 | 约 0.5 | AtoB 边缘带降权，alpha≈0.1~0.2 | 正常 |
| 二 | 约 0.15（BtoA 为主，混少量过滤后的干净 AtoB） | alpha 逐步放松或无需降权 | 阶段一的 1/5~1/10，早停 |

阶段二要监控换色效果，下降则早停；可只解冻后几层或用 LoRA，防止灾难性遗忘。

## 3. 实现要求

### 3.1 Dataset：返回来源标记

两个 Dataset 的 `__getitem__` 都返回 dict，且包含 `mask` 与 `is_atob`：

```python
class AtoBDataset(Dataset):
    def __getitem__(self, i):
        ...
        return {"input": a, "target": b, "mask": mask, "is_atob": 1}

class BtoADataset(Dataset):
    def __getitem__(self, i):
        ...
        return {"input": b, "target": a, "mask": mask, "is_atob": 0}

concat = ConcatDataset([ds_atob, ds_btoa])   # 顺序固定：[AtoB, BtoA]，Sampler 依赖此顺序
```

### 3.2 自定义 Sampler：按 epoch 调整混合比例（支持 DDP）

约束与要点：

- 继承 `torch.utils.data.Sampler`，实现 `__iter__`（产出整数索引）和 `__len__`。
- `__iter__` 每个 epoch 在 `for batch in loader` 开始时被调用一次；`set_epoch` 必须在其之前调用。
- 使用独立的 `torch.Generator`，种子 = `seed + epoch`。
- **DDP**：所有 rank 使用**相同 seed** 生成**相同的完整索引列表**，再用 `idx[rank::world_size]` 切分；`num_samples` 必须向下取整到 `world_size` 的倍数，保证各 rank 长度一致（否则梯度同步卡死）。使用 `drop_last=True` 时，每卡样本数最好也是 batch_size 的整数倍。
- 使用了该 Sampler 就**不要再套 `DistributedSampler`**，也不能设 `shuffle=True`。
- sampler 在主进程运行，`num_workers>0` 与 `persistent_workers=True` 不影响 `set_epoch` 生效。

```python
import math
import torch
from torch.utils.data import Sampler


def linear_schedule(p_start, p_end, total_epochs):
    def f(epoch):
        t = min(epoch / max(total_epochs - 1, 1), 1.0)
        return p_start + (p_end - p_start) * t
    return f


def cosine_schedule(p_start, p_end, total_epochs):
    def f(epoch):
        t = min(epoch / max(total_epochs - 1, 1), 1.0)
        return p_end + (p_start - p_end) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def step_schedule(p_stage1, p_stage2, switch_epoch):
    return lambda epoch: p_stage1 if epoch < switch_epoch else p_stage2


class ScheduledMixSampler(Sampler):
    """每个样本以概率 p_atob(epoch) 来自 AtoB，否则来自 BtoA。
    ConcatDataset 顺序为 [AtoB, BtoA]。"""

    def __init__(self, n_atob, n_btoa, num_samples, schedule,
                 seed=0, rank=0, world_size=1):
        self.num_samples = (num_samples // world_size) * world_size  # 各 rank 长度一致
        self.n_atob, self.n_btoa = n_atob, n_btoa
        self.schedule = schedule
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)  # 各 rank 相同
        p = self.schedule(self.epoch)
        pick_a = torch.rand(self.num_samples, generator=g) < p
        idx_a = torch.randint(0, self.n_atob, (self.num_samples,), generator=g)
        idx_b = self.n_atob + torch.randint(0, self.n_btoa, (self.num_samples,), generator=g)
        idx = torch.where(pick_a, idx_a, idx_b)
        return iter(idx[self.rank::self.world_size].tolist())

    def __len__(self):
        return self.num_samples // self.world_size
```

### 3.3 可选：严格 batch 内比例的 BatchSampler

需要"每个 batch 恰好 k 个 AtoB"时使用。通过 `batch_sampler=` 传入，此时 `batch_size`、`shuffle`、`sampler`、`drop_last` 都不能再设置。DDP 下所有 rank 生成相同的完整 batch 列表，再按 `batches[rank::world_size]` 切分，且 batch 数需能被 `world_size` 整除。

```python
class MixedBatchSampler:
    def __init__(self, n_atob, n_btoa, batch_size, k_schedule, num_batches,
                 seed=0, rank=0, world_size=1):
        self.n_atob, self.n_btoa = n_atob, n_btoa
        self.bs = batch_size
        self.k_schedule = k_schedule           # epoch -> 每个 batch 中 AtoB 的个数
        self.num_batches = (num_batches // world_size) * world_size
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        k = int(self.k_schedule(self.epoch))
        batches = []
        for _ in range(self.num_batches):
            a = torch.randint(0, self.n_atob, (k,), generator=g)
            b = self.n_atob + torch.randint(0, self.n_btoa, (self.bs - k,), generator=g)
            batches.append(torch.cat([a, b]).tolist())
        return iter(batches[self.rank::self.world_size])

    def __len__(self):
        return self.num_batches // self.world_size
```

### 3.4 DataLoader 与训练循环

```python
import torch.distributed as dist

rank = dist.get_rank() if dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_initialized() else 1

total_epochs = 30
sampler = ScheduledMixSampler(
    n_atob=len(ds_atob), n_btoa=len(ds_btoa),
    num_samples=len(concat),
    schedule=linear_schedule(0.5, 0.1, total_epochs),
    # 分阶段则用：step_schedule(0.5, 0.15, switch_epoch=20)
    seed=42, rank=rank, world_size=world_size,
)
loader = DataLoader(
    concat, batch_size=16, sampler=sampler,      # batch_size 为每卡
    num_workers=8, pin_memory=True, persistent_workers=True,
)

for epoch in range(total_epochs):
    sampler.set_epoch(epoch)                     # 必须在迭代 loader 之前
    alpha = 0.1 + 0.9 * epoch / total_epochs     # 边缘降权系数随 epoch 放松（可选）
    for batch in loader:
        ...
```

注意：

- DataLoader 构造后不能替换 `sampler`/`batch_sampler` 等属性；动态调整只能修改 sampler 对象内部状态，或在阶段切换时重建 DataLoader。
- 全局 batch = `batch_size × world_size`。
- 所有 rank 的 `set_epoch` 调用时机必须一致。

### 3.5 Loss：AtoB 边缘带降权

用形态学膨胀 − 腐蚀得到 mask 边缘带，只对 AtoB 样本降权，BtoA 样本不受影响。

```python
import torch.nn.functional as F


def make_edge_band(mask, width):
    """mask: [B,1,H,W]，值 0/1（float）。返回边缘带 [B,1,H,W] (0/1)。"""
    k = 2 * width + 1
    dil = F.max_pool2d(mask, k, stride=1, padding=width)
    ero = 1.0 - F.max_pool2d(1.0 - mask, k, stride=1, padding=width)
    return ((dil - ero) > 0).float()


def weighted_loss(pred, target, mask, is_atob, alpha, width_range=(3, 10)):
    # 带宽随机，避免对固定宽度过拟合（分辨率不同请调整）
    width = int(torch.randint(width_range[0], width_range[1] + 1, (1,)))
    band = make_edge_band(mask, width)                       # [B,1,H,W]
    atob = is_atob.view(-1, 1, 1, 1).float()                 # [B,1,1,1]
    # AtoB 样本在边缘带内权重为 alpha，其余为 1
    w = 1.0 - atob * band * (1.0 - alpha)
    loss_map = (pred - target).abs().mean(dim=1, keepdim=True)   # 按需替换为实际 loss
    return (w * loss_map).sum() / w.sum().clamp_min(1e-6)
```

- `alpha` 取 0~0.2（0 即完全忽略 AtoB 边缘带），可随 epoch 逐步增大。
- 若是扩散模型，应把 mask 下采样到 latent 分辨率后，在 latent 空间做同样的降权。
- 可选的 mask 外一致性 loss：对所有样本，`(1 - mask)` 区域强制输出 = 输入，这部分监督不依赖 B 的质量。

## 4. 可选：数据侧修复与过滤

- 重合成：`B' = mask_feathered * B + (1 - mask_feathered) * A`，保证 mask 外区域与 A 像素级一致。
- 边缘高频继承：Poisson blending，或低频颜色取自 B、高频取自 A。
- 过滤：在 mask 外区域计算 B 与 A 的 L1 / LPIPS 差异，差异大的样本丢弃或降权；阶段二用的 AtoB 样本应先经过此过滤。

## 5. 诊断与验收

1. 分别单独训练 / 评估 AtoB 与 BtoA，确认边缘不均来自哪个方向，以及是否集中在头发、半透明、细小结构等边缘类型。
2. 对比阶段一与阶段二：
   - 换色效果几乎不掉、边缘明显改善 → 两种能力可分离，现方案够用。
   - 换色明显下降 → 能力耦合较深，应优先做 B 的重合成（第 4 节）。
3. 检查项：
   - 多卡下各 rank 的 `len(loader)` 一致
   - 一个 batch 内 `is_atob` 比例符合当前 schedule
   - 采样索引范围正确（AtoB 为 `[0, n_atob)`，BtoA 为 `[n_atob, n_atob+n_btoa)`），越界错误只会在 worker 内抛出
   - AtoB 占比很低时，小数据集因 replacement 重复采样，注意是否过拟合

## 6. DataLoader 内部机制（实现时需知）

- 传入 `sampler` 且设置 `batch_size` 时，DataLoader 会自动用 `BatchSampler(sampler, batch_size, drop_last)` 包装；传 `batch_sampler` 时则直接使用它，并要求其他相关参数保持默认。
- 每个 epoch 开始（`for batch in loader`）时对 sampler 调用一次 `iter(...)`；`persistent_workers=True` 时同样会重新调用。
- sampler 始终只在主进程运行，worker 只接收索引、执行 `dataset[i]` 与 `collate_fn`；主进程按任务号重排，所以 batch 顺序与 sampler 产出顺序一致。
- 多进程下存在预取（`prefetch_factor × num_workers`），sampler 迭代会略超前于训练进度；对"按 epoch 变化比例"无影响，但若采样依赖训练时的动态状态（如按 loss 难例采样）会有滞后。
- `__len__` 只影响 `len(loader)`、进度条与调度器，写错不会让训练出错，但需保证各 rank 一致以免 DDP 卡死。
- 传给 `DataLoader` 的 `generator` 与自定义 sampler 无关，自定义 sampler 需自备 `torch.Generator`。
