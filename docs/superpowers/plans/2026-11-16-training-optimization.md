# FLUX.3 训练入口优化方案（2026-11-16）

> **For agentic workers:** 按任务顺序逐项实施；每项先补测试/断言，再改实现，最后跑对应回归。不要一次性重写训练循环。

**目标:** 整理 2026-11-16 对 [train_flux3_recolor_lora.py](../../../train_flux3_recolor_lora.py) 的只读复查结果，形成可执行的优化清单，优先降低启动/每步/验证开销，同时修复已确认的恢复、DDP 与 FSDP 验证正确性问题。

**范围:** 根训练入口及相邻的 `flux3_recolor/{data,modeling,precision,text_encoder,inference}.py`、测试与文档。**不包含**真实大模型训练、图像质量调参、数据/权重改写或新模型架构。

**状态标记:** `[已复现]` 本轮实际运行/计数确认；`[静态]` 代码路径核对确认但未复现；`[子代理报告]` 独立复查报告，本轮未复跑；`[待确认]` 需要用户决定语义；`[未执行]` 需要真实 GPU/FSDP 验收。

**已确认的基线:** `py_compile` 相关模块通过；用真实 `validate_training_resume` 复现了缺 `color_swatch_size` 的恢复失败；实际 `train_history.jsonl` 确认默认配置下方向 loss 键交替出现；本机抽样 PNG 解码约 19ms/张（仅 decode+convert，不含 resize/VAE）。独立复查报告的测试基线为 21 项 + 3 项集成 + 7 项 task embedding 通过、2 项跳过；legacy resume 未覆盖。**未运行真实训练或 GPU/FSDP 验收。**

## 当前实施进度（后续用户决策优先）

- P0-1 第一阶段已接入：使用path-only HF Dataset及hash缓存，训练和验证直接消费保存的记录；Dataset/Sampler初始化不读图片头，统一训练/验证，不再按方向分支统计。这里采用用户后续批准的路径元数据方案，替代原文的inspect扩展建议。
- 用户后续要求删除latent缓存：已移除cache_latents参数、预计算/存储/读取分支，统一在线编码；原P0-1的latent文件去重及其他依赖latent缓存的建议取消。P0-2已接入训练：目标/灰/源沿batch维合并编码，色块单独编码，每microbatch两次；FP32/mode/BN/patchify保留。后续已按用户要求移除recipe和norm替换，固定原始FP32 VAE与Transformer autocast。真实GPU速度和峰值显存尚未测量。
- P0-3验证去重/P1-9恢复目录处理已接入：本进程按step和生成配置去重，失败可重试；恢复进程重新生成，成功后替换旧验证目录。FSDP验证重载与精度问题尚未处理。
- 用户后续要求恢复在线caption编码：当前batch动态窗口保留，维度检查使用accelerator.unwrap_model(transformer).config，避免DDP wrapper直接访问。全部caption预编码方案已撤回。
- 训练loss日志现支持logging_steps（默认10）和tqdm；区间按实际样本数加权，在日志点/最终步归约与写盘，删除每优化步的loss日志归约。逐microbatch非有限检查保留；尚未实测GPU吞吐。
- P1-7/P2-1 原分方向监控建议已由统一loss方案替代，不再补分方向窗口统计。
- 原始审查与基线保留如下，行号与部分旧入口描述已过时；本节不表示其他P0/P1/P2事项已完成。

## 优先级总表

| ID | 优先级 | 类型 | 位置 | 问题 | 状态 |
| --- | --- | --- | --- | --- | --- |
| P0-1 | P0 | 启动性能 | `train_flux3_recolor_lora.py:699`, `:219`, `:300`；`flux3_recolor/data.py:55` | 同一批图片在 inspect、shape probe、dataset 初始化、latent cache 中被完整解码多遍 | `[已复现]` 计数/耗时抽样 |
| P0-2 | P0 | 每步性能 | `train_flux3_recolor_lora.py:894-897`, `:300-308` | 原4次，现目标/灰/源合并、色块独立，每microstep 2次；latent缓存已删除 | `已实现，GPU性能待测` |
| P0-3 | P0 | 验证性能 | `train_flux3_recolor_lora.py:685`, `:959-976`, `:617-624` | step/epoch/final重复已去重；验证仍为主进程，FSDP仍重载完整transformer | `部分完成` |
| P0-4 | P0 | DDP 正确性 | `train_flux3_recolor_lora.py:793`, `:854`, `:900` | 在线caption已恢复；维度检查通过unwrap_model读取原模型 | `已实现，CPU DDP+caption验收` |
| P1-1 | P1 | DDP 健壮性 | `train_flux3_recolor_lora.py:838-843`, `:594` | save hook 内 main-only 写失败不广播，其他 rank 可卡在 `wait_for_everyone` 至超时 | `[静态]` |
| P1-2 | P1 | FSDP 正确性 | `train_flux3_recolor_lora.py:618` vs `:467` | FSDP 验证代理按 bf16 加载，训练 transformer 保持 FP32，破坏 precision 一致性 | `[静态]`，需 FSDP 验收 |
| P1-3 | P1 | FSDP 正确性 | `train_flux3_recolor_lora.py:622` | 验证代理忽略 `set_peft_model_state_dict` 的 missing/unexpected 结果，键不匹配会静默用随机 LoRA | `[静态]` |
| P1-4 | P1 | 恢复兼容 | `train_flux3_recolor_lora.py:336-345` | 缺 `color_swatch_size` / `preprocessing.color_swatch_size` 的更老 v2 checkpoint 无法恢复；shim 覆盖不完整 | `[已复现]` |
| P1-5 | P1 | API 健壮性 | `train_flux3_recolor_lora.py:691`, `:911`, `:802` | `main()` 接受 Namespace，但混用 `getattr` 与直接属性；task embedding+caption_json 会在模型加载后才失败 | `[静态]` |
| P1-6 | P1 | 配置一致性 | `train_flux3_recolor_lora.py:134`, `:177`, `:759-771` | `prompt_augmentation` 被显式 prompt/caption 静默忽略但仍持久化 true，恢复同有效配置会不一致 | `[静态]` |
| P1-7 | P1 | 监控正确性 | `train_flux3_recolor_lora.py:948` | 默认 batch=1/accum=1 时每步只记录一个方向，`loss_b_to_a`/`loss_a_to_b` 曲线稀疏交替 | `[已复现]` 实际 history |
| P1-8 | P1 | 恢复语义 | `train_flux3_recolor_lora.py:935`, `:965` | checkpoint 恰在 epoch 边界时 state 已推进到下一 epoch；保存后、epoch 验证前崩溃会跳过该 epoch 验证 | `[静态]` 低严重度 |
| P1-9 | P1 | 恢复/验证 | `train_flux3_recolor_lora.py:654`, `:959-962` | 恢复后重新出图，成功后替换旧validation目录；生成失败保留旧图 | `已实现` |
| P2-1 | P2 | 热循环清理 | `train_flux3_recolor_lora.py:913-949` | 每 microbatch 重建方向 mask/H2D，每步额外 all-reduce+`.item()`；`group_loss` 与 direction_stats 重复 | `[静态]` |
| P2-2 | P2 | 维护性 | `train_flux3_recolor_lora.py:576-578`, `:841-849` | task embedding 保存/加载在 export、save hook、load hook 三处重复 | `[静态]` |
| P2-3 | P2 | 维护性 | `train_flux3_recolor_lora.py:106`, `:334`, `:450`, `:623`, `:642`, `:911` | recipe 名称与 `legacy` 字面量未从 `precision.py` 导入；Namespace 路径直接访问新字段 | `[静态]` |
| P2-4 | P2 | 启动小开销 | `train_flux3_recolor_lora.py:799-820` | prompt_augmentation 时固定 prompt 编码后丢弃；默认验证 prompt 再编码一次 | `[静态]` |
| P2-5 | P2 | 采样/IO | `train_flux3_recolor_lora.py:109`, `:275-296`, `:313`, `:878` | sampler 计划每 epoch 重复计算；workers 默认 0；cache tensor 未 pin/non_blocking；VAE 未设 channels_last | `[静态]` |
| P2-6 | P2 | 参数校验 | `train_flux3_recolor_lora.py:82`, `:99`, `:114` | `prodigy_beta3`、`text_encoder_out_layers`、`report_to` 列表/量化互斥校验不足 | `[静态]` |
| P2-7 | P2 | 文件清理 | `train_flux3_recolor_lora.py:588` | 保存失败留下的 `.checkpoint-*` 临时目录没有启动清理 | `[静态]` |
| P2-8 | P2 | 数据校验 | `flux3_recolor/data.py:63` | 只校验 `主颜色B` 纯色；仅训练 `b_to_a` 时第三张 `主颜色A` 不会被检查 | `[静态]`，待确认是否有意 |
| P2-9 | P2 | 死代码/重复 | `train_flux3_recolor_lora.py:207`, `:269`, `:692` | `sampler` 未使用；`preprocessing_config` 在 main 与 dataset 各算一次 | `[静态]` |
| P2-10 | P2 | 显存/启动 | `train_flux3_recolor_lora.py:488-493`, `:799`, `:895-897` | DDP 每个 rank 都加载/编码 Qwen 固定 prompt；transformer 与 text encoder 同时占显存；`--offload` 时 VAE 每 microstep CPU↔GPU | `[静态]` |
| P2-11 | P2 | 恢复配置 | `train_flux3_recolor_lora.py:509-518`, `:759-771` | `text_encoder_learning_rate` 在未启用 train_text_encoder 时静默无效但仍持久化；启用时 `None` 与显式等于 LoRA LR 的有效配置也被当成不同值 | `[静态]` |
| P2-12 | P2 | 数据/校验 | `train_flux3_recolor_lora.py:208-212`, `:696-700` | `caption_json` 覆盖检查使用全部 names（含 validation），但验证生成不读 caption；错误信息却只说“训练文件名” | `[静态]` |

## P0：启动与每步性能

### P0-1 消除启动阶段重复解码

**问题:** `inspect_dataset` 本身是设计上的全量解码（用于解码错误、尺寸、纯色和像素指纹），但之后又发生了多轮重复解码：

1. `inspect()` 构造一次性 `DreamBoothDataset(args, report['names'])`，只为得到 `shapes`，重新解码全部 train+validation 样本（`train_flux3_recolor_lora.py:699`）。
2. 正式 `DreamBoothDataset.__init__` 又为全部训练记录解码一遍（`:219`）。
3. `cache_latents` 再逐样本解码一遍（`:300`）。
4. 每个 epoch 的 `__getitem__` 仍按需解码（无图像缓存）。

**计数（当前 `test` 1776 组、双向、约 10% 验证）:**

- `inspect_dataset`: `1776 × 5 = 8,880` 次完整解码（双向去重后的 5 个目录）。
- `inspect()` probe: `1776 × 2 × 4 = 14,208`。
- 训练 dataset shapes: 约 `1598 × 2 × 4 = 12,784`。
- `cache_latents`: 约 `3196 × 4 = 12,784`。
- 主进程合计约 `48,656` 次；按本机 19ms/张估算约 15 分钟量级（不含 resize/VAE），其他 rank 还会重复 dataset/cache 部分。

**建议实现:**

- 扩展 `inspect_dataset` 返回值：为每个 `name` 记录各文件夹的 oriented `(w,h)`（它已经在 `load_rgb` 后计算过），或返回可直接用于 `shapes` 的几何元数据。
- `DreamBoothDataset` 增加 `shapes` 注入/尺寸表路径，用尺寸元数据和 policy 数学计算 bucket/输出尺寸，不再调用 `prepare()` 解码。
- 删除 `inspect()` 中的完整 `DreamBoothDataset` probe，替换为基于尺寸表的轻量几何校验（含极端宽高比、16px 网格检查）。
- `cache_latents` 以文件路径（而不是 direction 展开后的 sample_id）去重；双向共享的 `涂灰图`、`原图A`、`生成图B` 只编码一次；`--repeats` 同样复用。
- 可选：将 `image.tobytes()`（约 2.4MB/张）替换为原始文件字节哈希；这会改变 fingerprint 语义，必须连同 checkpoint/version 一起讨论，默认**不做**。

**验收:**

- 新增测试 monkeypatch `flux3_recolor.data.load_rgb` 或 `DreamBoothDataset.prepare` 计数，验证 1776 组双向 + cache 的启动解码次数显著下降（目标 ≤ 原 50%）。
- `test_flux3_recolor_data`、`test_flux3_reference_training` 的双向/resume/split 测试全部通过；数据集 fingerprint 与 split 不变。

### P0-2 VAE 场景批量编码

**原问题（改造前）:** 训练循环对 `pixel_values` 和三个 `cond_pixel_values` 分别调用 `encode_latents`（`train_flux3_recolor_lora.py:894-897`），共 4 次 VAE forward；`cache_latents` 同样逐张（`:300-308`）。

**建议实现:**

- 训练中把目标、涂灰图、源图三张场景图沿batch维拼接成一次 `vae.encode`（它们经 `prepare_with_policy` 保证同尺寸），色块单独一次；每 microstep 从 4 次降到 2 次。
- latent缓存已按用户要求删除，不实施缓存分组。场景batch为原来的3倍，需关注FP32高分辨率下的峰值显存。
- 不改变 BN/patchify/latent 数值路径；编码结果需与逐张编码在容差内一致。

**验收:**

- 新增 CPU 小 VAE 测试：批量编码 vs 逐张编码，`torch.testing.assert_close` 使用明确且可解释的容差（batch 卷积可能带来浮点顺序差异，不能直接要求逐位相等）。
- 在线训练、梯度累积及checkpoint恢复回归通过；不再验收已删除的latent缓存。

### P0-3 验证去重与开销控制

**问题:**

- 默认每 100 步在主进程验证 4×2=8 张、28 步；其他 rank 在 `main_process_action` broadcast 上等待（`train_flux3_recolor_lora.py:685`）。
- 最后一步可能同时触发 `validation-step-N`、epoch 验证和 `final-N`（`:959-976`），同 prompt 下重复生成。
- FSDP 每次验证重新加载完整 transformer（`:617-624`），重建 adapters、precision recipe 和 pipeline。
- 验证默认可能开启 CFG，每步两遍 forward；没有 `--validation_guidance_scale`。

**建议实现:**

- 最后一步（`state['global_step'] >= args.max_train_steps`）跳过 periodic/epoch 验证，或让 final 验证复用刚生成的同 prompt 图片；保留 `--skip_final_inference` 语义。
- 增加 `--validation_guidance_scale`，periodic 验证可用 1 或更低步数；不要改变 final 默认输出。
- 非 FSDP：评估按 rank 分摊验证样本（每 rank 模型已同步），主进程汇总 manifest/tracker；这是中等改动，可先只做去重和参数。
- FSDP：缓存验证代理模型，每次只 gather/更新 LoRA；不要每个 validation step 从磁盘重载基座。与 P1-2/P1-3 一起改。

**验收:**

- 新增小模型测试：达到 max steps 时只生成一组最终验证（或明确 final 复用），periodic/epoch 仍按配置工作。
- `test_flux3_reference_training` 的 validation/TensorBoard 测试、`test_flux3_text_encoder` 的 validation 计数测试通过。
- FSDP 验证改动至少加 CPU/模拟测试；真实 FSDP 仍标 `[未执行]`。

### P0-4 DDP/FSDP + caption_json 的在线文本编码

**问题:** `compute_text_embeddings` 是在 `main()` 内定义的闭包，引用外层局部变量 `transformer`（`:793`）。`main()` 在 `:854` 执行：

```python
transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(...)
```

这会重绑定同一个闭包变量。使用 `--caption_json` 时 `fixed_embeddings=None` 且 `prompt_embeddings` 为空，训练每个 microbatch 都会走到 `:900` 的在线分支，调用 `compute_text_embeddings(batch['prompts'])`；此时 `transformer` 已经是 DDP/FSDP wrapper，而函数末尾访问 `transformer.config.joint_attention_dim`。PyTorch 2.11 的 `DistributedDataParallel` 没有把任意属性转发到 `module` 的 `__getattr__`，因此会报 `AttributeError: 'DistributedDataParallel' object has no attribute 'config'`。现有两 rank CPU DDP 测试没有使用 `--caption_json`，所以这个受支持的组合没有被覆盖。

**建议实现:**

- 在 `accelerator.prepare` 前捕获 `joint_attention_dim = transformer.config.joint_attention_dim`，闭包只使用标量；或使用 `components['transformer'].config` / `accelerator.unwrap_model(transformer).config`。
- 同样检查 `compute_text_embeddings` 内是否还有其他对 `transformer` 的属性访问（目前只有这一处）。
- 增加两 rank CPU DDP + `caption_json` 的实际训练测试（至少 1 步），并覆盖 FSDP 分支的等价逻辑（FSDP 至少做单元/静态断言，真实硬件标 `[未执行]`）。

## P1：正确性与健壮性

### P1-1 save hook 单侧失败广播

**问题:** `save_model_hook` 中 `save_adapter` / `save_text_encoder` 只在主进程执行（`train_flux3_recolor_lora.py:838-843`），没有像 `export_lora` 那样包 `main_process_action`。主进程写失败时，其他 rank 完成 no-op hook 后会卡在 `accelerator.wait_for_everyone()`（`:594`）直到 NCCL/gloo 超时。

**建议实现:**

- 将主进程写盘动作包进 `main_process_action(accelerator, save)`，异常广播到所有 rank 后再进入 wait。
- `weights.clear()` 放入 `finally`，避免异常路径留下引用。
- 同时覆盖已有的 `save_adapter` 路径和新增的 task embedding 路径。

**验收:**

- 两 rank CPU DDP 测试（opt-in）：patch `save_text_encoder` 在主进程抛异常，断言两个 rank 都在短超时内失败，不挂起；现有 DDP 测试仍通过。

### P1-2 FSDP 验证代理 dtype 与训练一致

**问题:** FSDP 训练 transformer 在 `:467` 以 FP32 加载并交给 FSDP；验证代理在 `:618` 以 `torch_dtype=dtype`（bf16/fp16）加载，再应用 precision recipe。这样验证代理的基座权重和 NativeLayerNorm/GroupNorm 的原生 dtype 与训练模型不同，`fp32_vae_native_norm_v1` 的训练/验证一致性保证被破坏。

**建议实现:**

- FSDP 验证代理按训练 master dtype 加载：native recipe 下使用 `torch.float32`，`apply_precision_recipe(..., compute_dtype=dtype)` 控制计算上下文；legacy 保持原行为。
- 若内存不允许，明确记录“FSDP 验证代理精度不等于训练”，不要继续宣称一致。

**验收:**

- 小模型/CPU 测试断言 native recipe + FSDP 分支代理基座为 FP32；真实 FSDP 验证标 `[未执行]`。

### P1-3 检查 FSDP 验证 LoRA 加载结果

**问题:** `make_validation_pipeline` 调用 `set_peft_model_state_dict` 后忽略返回值（`:622`）；键不匹配时高斯初始化的 LoRA 会保留，验证图和 manifest 都来自随机 adapter，且没有报错。

**建议实现:**

- 复用 `load_adapter_state` 的 missing/unexpected 检查逻辑；验证代理加载后立即拒绝不匹配。
- 增加测试：构造缺失/多余 key 的状态，断言报错。

### P1-4 旧 checkpoint swatch 字段迁移

**问题:** `validate_training_resume` 没有为缺失的 `color_swatch_size` 和 `preprocessing.color_swatch_size` 做迁移。使用真实函数复现：缺字段时报 `恢复配置不一致: color_swatch_size`。现 `outputs` 里的 v1/v3/v4/v5 checkpoint 都带 128 字段；缺 `precision_recipe`/task 字段在 `--precision_recipe legacy` 下可正常迁移（已用真实函数验证 PASS）。因此影响的是比色块尺寸改动更老的 v2 checkpoint。

**决策点 `[待确认]`:** 是否需要恢复这批更老 checkpoint？

- 若不需要：保留失败，但把错误改成明确说明“旧 swatch policy 无法用新 128 预处理恢复”，并在文档/tests 中固定。
- 若需要：增加显式的 legacy swatch policy 选项（例如 `--color_swatch_size legacy` 或配置迁移），不要简单 `setdefault(128)` 导致训练输入悄悄改变；新行为必须使用新输出目录并写清 config 字段。

**验收:**

- 合成 saved config 的单元测试覆盖“缺 swatch 字段”的预期报错/迁移；现有 checkpoint 恢复测试不回归。

### P1-5 Namespace 入口默认值归一化

**问题:** `main()` 支持 `argparse.Namespace`（`:691`），但 `args.precision_recipe` 在 `:911` 直接访问，`args.training_prompts`/`args.color_swatch_size` 等也依赖 parse_args；只传旧字段的调用者会在模型加载/文本编码后才崩溃，`train_text_encoder + caption_json` 还会把 `fixed_embeddings=None` 传给 `install_text_encoder`（`:802`）。

**建议实现:**

- 在 `main()` 入口对 Namespace 做一次与 `parse_args` 等价的默认值/组合校验；或只接受 list/CLI，裸 Namespace 明确拒绝。
- 提前校验 `train_text_encoder` 与 `caption_json` 的互斥，报错发生在加载模型之前。

**验收:**

- 单元测试：Namespace 缺新字段时得到明确错误或补齐默认；task+caption_json 在加载模型前失败。

### P1-6 prompt_augmentation 持久化一致性

**问题:** 显式 `--prompt`/`--caption_json` 会静默关闭 augmentation 的实际效果（`:134`），但 `args.prompt_augmentation` 仍为 true 并写入 config；用同一有效配置但不带 inert flag 恢复会报配置不一致。`--train_text_encoder --prompt_augmentation --prompt custom` 也被拒绝（`:177`），尽管变体列表不会使用。

**建议实现:**

- parse 后把“实际生效的 augmentation”写回 `args.prompt_augmentation`，或在显式 prompt/caption 同时出现时直接报错/警告并禁止组合。
- 更新 `validate_training_resume` 的兼容说明；增加恢复测试。

### P1-7 方向 loss 记录

**问题:** `logs.update({f'loss_{direction}': ... if reduced_directions[i,1] > 0})`（`:948`）只为当前 accumulation group 中出现的方向写 key。实际 `outputs/flux3_recolor_v3/train_history.jsonl` 已确认默认 batch=1/accum=1 时 `loss_b_to_a`/`loss_a_to_b` 交替出现，两条曲线各自稀疏。

**建议实现:**

- 维护窗口/累计分方向均值，或在缺失方向写 `NaN` 并在文档说明；不要让消费者把“没出现”误读为 0。
- 若希望每步都有两方向，需要 batch/accum 覆盖两方向；这是训练配置语义，不要偷偷改默认。

**验收:**

- 小模型训练 4 步，断言 history 中方向字段行为符合新约定；保留现有 `loss_*` 兼容性。

### P1-8 epoch 边界 checkpoint 验证缺口

**问题:** 最后一个 local batch 时 `state['epoch']` 先推进到 `epoch+1`（`:935`），checkpoint 在 epoch 验证前保存（`:958`）；若保存后、验证前/中崩溃，恢复从下一 epoch 开始，`state['epoch'] == epoch+1` 的 epoch 验证条件（`:965`）不再匹配，跳过该 epoch 验证。

**建议实现:**

- 在 state 中记录 `last_validated_epoch`/`validation_pending`，恢复时补做；或在 checkpoint 前先标记待验证，验证成功后清除。
- 注意现有“先提交 checkpoint 再验证”的设计是为了验证失败可恢复，不能简单把 checkpoint 移到验证之后。

**验收:**

- 模拟 checkpoint 写入后验证失败并恢复，断言 epoch 验证会补做且不会重复。

### P1-9 恢复重放与已存在验证目录冲突

**问题:** `log_validation` 的目标目录名是确定性的 `validation-step-N`（`:654`），`generate_validation` 内部调用 `prepare_output` 并拒绝非空目录（`flux3_recolor/inference.py:81`）。如果 `--validation_steps` 小于 `--checkpointing_steps`，或崩溃发生在两次 checkpoint 之间：

- 例：`--checkpointing_steps 100 --validation_steps 40`，step 120 已写 `validation-step-120`，step 125 崩溃；最近有效 checkpoint 是 `checkpoint-100`。
- 从 checkpoint-100 恢复会重放 101-125；到 step 120 再次调用验证时，目标目录已存在且非空，`prepare_output` 抛 `输出目录非空，拒绝覆盖`，导致本来可恢复的训练中止。

**建议实现:**

- 在 `training_state.json` 记录已完成验证的 step/label，恢复时跳过或复用已完成验证；或在验证输出目录写入 manifest 后允许同一步的幂等覆盖（先写临时目录再原子替换）。
- 不要无条件允许覆盖任意已有目录；必须校验目录内容/配置属于本训练入口，避免误删用户文件。
- 增加测试：模拟 checkpoint-100、validation-step-120、崩溃后 resume，断言恢复成功且不会重复生成/覆盖无关产物。

## P2：清理与小优化

### P2-1 方向统计热循环

- 用每个 batch 的方向索引（或预先计算的 per-sample direction id）替代每方向 Python 比较 + `torch.tensor` H2D（`:917-920`）。
- 每个 optimizer step 的额外 `accelerator.reduce(direction_stats)` + `.item()`（`:938-949`）可合并进已有日志路径；`report_to=none` 仍需 JSONL，但可减少 reduction 次数。
- `group_loss` 可由 `reduced_directions[:,0].sum()/[:,1].sum()` 推导（`:880`），避免两个累加器漂移。先确认数值等价，再删除重复状态。

### P2-2 task embedding 存取去重

- 将 `export_lora`（`:576-578`）、save hook（`:841-843`）、load hook（`:847-849`）的 lazy import/`unwrap_model`/save/load 收敛为共享 helper，确保最终导出和 checkpoint 恢复不会漏掉 `text_encoder.safetensors`。
- 增加一个测试：只改 helper 契约时，最终目录和 checkpoint 都能被独立推理读取。

### P2-3 recipe 名称与 legacy 字面量

- 从 `flux3_recolor.precision` 导入 recipe 常量用于 argparse choices 和比较；移除 `:334/:450/:623/:642` 的重复 `'legacy'` 字面量。
- 增加 `get_precision_recipe(args)` 统一处理 Namespace/缺字段；`:911` 改用它。

### P2-4 重复 Qwen 编码

- `--prompt_augmentation` 时 `fixed_embeddings` 在 `:799` 编码后于 `:811` 丢弃；可先判断 `len(args.training_prompts) == 1` 再编码。
- 默认验证 prompt 与 `fixed_embeddings`/`encoded_prompts` 相同，可复用已有 embedding，避免 `:818-820` 的重复 Qwen forward。
- 注意不要复用不同窗口长度/不同 prompt 的 embedding；只做语义完全相同且窗口兼容的复用。

### P2-5 sampler / DataLoader / 内存

- `BucketBatchSampler.batches()` 每 epoch 被 `__len__`、`__iter__` 和 `:878` 的 `plan` 重复计算；缓存到 epoch 结束，注意 `set_epoch` 后失效。
- `--dataloader_num_workers` 默认 0（`:109`）；不 cache 时建议文档推荐 4+，或在显存/内存允许时改默认。
- `cache_latents` 的 CPU tensor 可 pin；`cached_latents_for_batch` 的 `.to()` 可加 `non_blocking=True`（`:313`）。
- native recipe 下 VAE 输入已 channels_last（`flux3_recolor/modeling.py:70`），可 benchmark `vae.to(memory_format=torch.channels_last)`；无收益则不保留。

### P2-6 参数校验

- `--prodigy_beta3` 做 finite 校验；`--text_encoder_out_layers` 校验非空、整数、在 hidden_states 范围内；`--report_to` 支持逗号分隔列表（Accelerate 接受列表）。
- bitsandbytes 配置同时 `load_in_4bit` 和 `load_in_8bit` 时提前拒绝；现有组合校验已覆盖大部分。

### P2-7 checkpoint 临时目录清理

- 启动/保存前清理 `.checkpoint-*` 临时目录；只清理本入口命名模式且不是完成 checkpoint 的目录，避免误删。

### P2-8 色块纯色校验范围

- `flux3_recolor/data.py:63` 只检查 `CONDITION_FOLDERS2[2]`（主颜色B）。只训练 `b_to_a` 时第三张是主颜色A，不会被纯色检查。
- `[待确认]` 若主颜色A也应为纯色，改为检查所有 `swatch_folders`；若有意不检查，补文档说明原因。

### P2-9 死代码/重复

- 删除 `sampler = range(len(dataset))`（`:269`）。
- `preprocessing_config` 在 `main`（`:692`）和 `DreamBoothDataset.__init__`（`:207`）各算一次；可只算一次传入。

### P2-10 DDP 文本编码器与 VAE offload

- 固定 prompt 且无 caption 时，每个 rank 都会加载 Qwen text encoder、编码相同 prompt，然后删除；可以考虑 rank0 编码后 broadcast embedding，其他 rank 不加载 text encoder。需处理 `--fsdp_text_encoder` 和 caption_json 在线编码路径。
- `load_training_components` 先把 transformer 移到 GPU，再加载 text encoder（`:488-493`），固定 prompt 下两者同时占显存；可先编码文本、删除 encoder，再移动 transformer，或依赖 `--offload`。
- `--offload` 时 VAE 每个 microstep 都 `to(device)`/`to('cpu')`（`:895-897`），可能比不 offload 更慢；显存允许时优先 `--cache_latents`，文档中给出对照建议。
- 验收：两 rank DDP 测试固定 prompt embedding 广播一致；不改变 caption_json 路径；显存/速度收益需真实 GPU 测量，标 `[未执行]`。

### P2-11 task embedding 学习率参数一致性

- 未启用 `--train_text_encoder` 时，`--text_encoder_learning_rate` 仍被 parse 接受并写入 config（`:759-771`），但 `create_optimizer` 只有在存在 `recolor_text_encoder` 模块时才消费它（`:509-518`），因此是静默 no-op；恢复时如果不带这个 flag，会因 saved 有值、current 为 None 而被 `validate_training_resume` 拒绝，尽管优化轨迹逐位相同。反之，用户传了这个 flag 但没有 `--train_text_encoder` 时也不会收到任何提示。
- 启用 `--train_text_encoder` 时，`create_optimizer` 用 `args.text_encoder_learning_rate or args.learning_rate`，但 config 原样保存 `None`；恢复时显式传一个等于 LoRA LR 的值也会被判为配置不一致，尽管有效学习率相同。
- 建议：未启用 train_text_encoder 时直接拒绝该 flag（或至少警告并从 config 中排除）；启用时在 config 中保存归一化后的实际值，或在 `validate_training_resume` 比较有效值。
- 验收：parse/校验测试覆盖 no-op 组合；恢复测试覆盖 None 与显式相等值。

### P2-12 caption_json 覆盖范围与错误信息

- `DreamBoothDataset.__init__` 要求 `caption_json` 覆盖传入的全部 `names`（`:208-212`）；`inspect()` 用 `report['names']`（train+validation）构造它（`:696-700`），所以即使验证生成不读 caption，validation 文件名也必须有 caption。
- 错误信息写的是“必须覆盖所有训练文件名”，与实际的 all-names 检查不一致；用户只给训练集 caption 时会在模型加载前失败。
- 建议：只对 train split 要求 caption 覆盖（validation 不需要）；`inspect()` 中用于几何校验的 dataset 增加 `require_captions=False` 或直接使用尺寸表（与 P0-1 一起做）。错误信息改为准确描述。
- 验收：train 6 个文件 + validation 1 个文件、caption 只覆盖 train 时训练可启动；缺 train caption 仍报错。

## 实施顺序建议

0. **批次 0（正确性 hotfix）:** P0-4（DDP/FSDP + `caption_json` 在线编码）。
1. **批次 A（低风险性能）:** P0-1、P0-2、P2-5 的 sampler 缓存。
2. **批次 B（验证）:** P0-3 去重/参数；P1-2、P1-3、P1-8、P1-9（FSDP 验证、恢复与验证目录）。
3. **批次 C（健壮性）:** P1-1、P1-4、P1-5、P1-6、P1-7。
4. **批次 D（清理）:** P2-1、P2-2、P2-3、P2-4、P2-6、P2-7、P2-8、P2-9、P2-10、P2-11、P2-12。

每批完成后更新 `docs/ITERATION_NOTES.md`，记录实际改动、验证命令、结果和未执行项；架构事实变化才更新 `docs/PROJECT_CONTEXT.md`。

## 回归命令

CPU 基础回归：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_flux3_recolor_data test_flux3_recolor_training test_flux3_recolor_inference \
  test_flux3_reference_trainer test_flux3_reference_training -v
```

数学/精度/双向/task embedding：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_flux3_math_audit test_flux3_precision_recipe test_flux3_bidirectional \
  test_flux3_text_encoder -q
```

两 rank CPU DDP（需 Gloo/socket）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/tests" CUDA_VISIBLE_DEVICES='' \
RUN_FLUX3_DDP_TESTS=1 ../.venv/bin/python -m unittest test_flux3_reference_distributed -v
```

单卡 CUDA（仅显式授权时）：

```bash
CUDA_VISIBLE_DEVICES=1 RUN_FLUX3_CUDA_TESTS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_cuda_validation -v
```

新增测试应优先使用随机小型仓库/CPU，不加载真实基础权重；真实 GPU/FSDP 验收结果必须单独标注，不用 CPU 结果替代。

## 边界与开放问题

- **不改训练数学**: P0/P1 不能改变 flow matching、sigma 采样、loss 权重、方向归一化或默认精度 recipe，除非该项明确要求（P1-2 是验证代理一致性，不是训练数学）。
- **不改已有产物**: 不覆盖现有 checkpoint、adapter、数据集图片或 fingerprint；需要行为变化时使用新 output dir。
- **fingerprint 语义**: 若把像素指纹改为文件字节哈希，必须作为版本化配置变更并拒绝旧 checkpoint 恢复；默认保持现状。
- **旧 checkpoint**: 是否需要支持缺 `color_swatch_size` 的更老 v2 checkpoint？这是 P1-4 的唯一决策点。
- **final validation prompt 优先级**: 当前 `--validation_prompt` 同时存在时 final 仍使用它（`:974`）；需确认是否符合预期。若是 `--final_validation_prompt` 应优先，改为 `final_validation_prompt or validation_prompt or instance_prompt` 并补测试。
- **验证分摊**: 非 FSDP 按 rank 分摊验证可显著缩短 wall time，但需要合并 manifest/tracker，且要避免长验证触发 NCCL 超时；可先做去重和减少默认开销。
- **真实质量**: 上述代码优化不证明换色质量改善；色块控制/印花保持仍是独立实验问题。若要做训练目标优化，先做小样本过拟合与 LR 1e-5 对照，再考虑 mask 加权 loss。
