# Mask 内自动颜色分区

`AutoColorRegionsNode` 根据原图和 Mask 自动确定颜色区域数量，不需要输入“有几种主色”。只在 Mask 内计算颜色，不运行 SAM / FLUX，也不加载模型权重。

## 页面使用

1. 重启 Studio 并刷新页面。当前虚拟环境已安装新增的分色依赖；其他环境需安装 [Studio 依赖](requirements.txt)。
2. 左侧点击 **⑪ Mask 内自动颜色分区**，进入单节点练习；这会替换当前画布，先保存已有工作流。
3. 上传 `image` 原图和 `mask`，或在现有画布添加节点并连线。两张图片必须尺寸一致。
4. 保持 `detail=1`，点击节点测试按钮。右侧显示自动得到的区域数、代表色、面积比例，以及各区域的 Mask / 原色透明抠图。
5. 想合并近似颜色可设 `detail=0`；想区分更接近的颜色可设 `detail=2`。这个参数不是颜色数量。
6. 若要使用或保存某一区域，添加 **选择颜色区域** 节点，连接 `regions`，再填写 `region_id`。

典型连接：原图和 SAM3 的 `object_mask` → 自动分色 → `regions` → 选择颜色区域 → `mask` / `rgba` → 保存图片。

独立使用不需要填写全局 Sample 输入或 JSONL。只有连接了 SAM3 等上游模型节点时，测试才会执行那些上游节点。

## 节点输入与输出

### AutoColorRegionsNode

| 方向 | 名称 | 节点端口类型 | Python 类型及含义 |
| --- | --- | --- | --- |
| 输入 | `image` | `RGB_IMAGE` | `PIL.Image.Image`，RGB 原图，尺寸 `(W, H)` |
| 输入 | `mask` | `MASK` | `PIL.Image.Image`，同尺寸 `1` / `L` 图片，灰度大于 127 为有效前景 |
| 输入 | `detail` | `INT` | `int`，0 粗 / 1 中 / 2 细，默认 1 |
| 输出 | `regions` | `COLOR_REGIONS` | `ColorRegionSet`，原图副本、标签图和区域元数据，不是普通 JSON |
| 输出 | `overlay` | `RGB_IMAGE` | `PIL.Image.Image`，原尺寸 RGB 伪彩色叠加图，仅用来辨认分区 |
| 输出 | `palette` | `RGB_IMAGE` | `PIL.Image.Image`，原尺寸 RGB 主色图，每个区域填充原图平均 sRGB |
| 输出 | `labels` | `LABEL_MAP` | `PIL.Image.Image`，原尺寸 `L` 整数标签图；0 为背景，1～N 为区域编号 |

`overlay` 和 `palette` 在 Mask 外保留原图。`labels` 的值是编号，**不是黑白 Mask**，数值较小所以看起来很黑。它可以直接接保存图片节点保存为 PNG，但不能直接当作 `MASK` 连线。

### SelectColorRegionNode

| 方向 | 名称 | 节点端口类型 | Python 类型及含义 |
| --- | --- | --- | --- |
| 输入 | `regions` | `COLOR_REGIONS` | `ColorRegionSet`，连接自动分色的同名输出 |
| 输入 | `region_id` | `INT` | `int`，从 1 开始，默认 1；必须在本图范围内 |
| 输出 | `mask` | `MASK` | `PIL.Image.Image`，全尺寸 `L` 模式，所选区域 255，其他位置 0 |
| 输出 | `rgba` | `RGBA_IMAGE` | `PIL.Image.Image`，全尺寸 RGBA，保留原 RGB，区域外 Alpha 为 0 |

右侧区域卡片的图片按区域包围框裁剪，并缩小到最长边不超过 256，适合检查，不是全尺寸导出。通过选择节点取得的输出始终保留原尺寸和坐标。

## 自动数量是如何得到的

1. 提取有效 Mask 内的全部像素，转换到标准 CIE Lab。
2. 使用 `(0.5 × L, a, b)` 特征，减弱亮度差异的影响，但不完全忽略明暗。
3. RGB 每通道使用 5 位量化网格，按像素数量加权；网格多于 32 时，先用固定随机种子的加权 KMeans 形成最多 32 个细分中心。
4. 使用完全链接层次聚类，合并颜色相近的中心；粗 / 中 / 细的距离阈值分别为 24 / 15 / 8。
5. 将每个前景像素分配到最近的细分中心及其合并类别，按面积从大到小重新编号。

没有单独的“小区域删除”步骤；同色区域不要求空间连通。量化、32 个细分中心的上限和距离合并仍可能合并细微颜色，因此不能保证所有稀有颜色都能单独成类。

这些阈值是加权 Lab 空间的距离，不是标准未经加权的 ΔE；完全链接约束的是细分中心间的距离，不是每个原始像素到其他像素的最大色差。

## Python 函数接口

实现位于 [颜色分区模块](../object_inpainting/color_regions.py)。所有新增核心函数和节点 `execute()` 均有类型标注，以及 `Args` / `Returns` 说明，包含数组形状、值域和图片模式。

| 函数 | 输入 | 返回 |
| --- | --- | --- |
| `weighted_lab()` | `(N, 3)` 的 `NDArray[uint8]` RGB | `(N, 3)` 的 `NDArray[float64]` 加权 Lab |
| `color_bins()` | RGB 像素、对应的加权 Lab | `(B, 3)` 的中心与 `(B,)` 的像素计数，均为 `NDArray[float64]` |
| `nearest_centers()` | `(N, 3)` 特征与非空 `(K, 3)` 中心 | `(N,)` 的 `NDArray[intp]`，索引 0～K−1 |
| `cluster_colors()` | RGB、加权 Lab、精细度 `int` | `(N,)` 的 `NDArray[intp]` 类别索引，尚未按面积重排 |
| `segment_colors()` | RGB 原图、同尺寸 Mask、精细度 | `ColorRegionSet` |
| `render_color_regions()` | `ColorRegionSet` | `ColorRegionOutputs`，包含四个分色输出的 `TypedDict` |
| `select_color_region()` | `ColorRegionSet`、区域编号 `int` | `SelectedRegionOutputs`，包含 `mask` 和 `rgba` 的 `TypedDict` |

`ColorRegion` 元数据包括 `id: int`、`pixels: int`、`fraction: float`、`color: tuple[int, int, int]` 和 `bbox: tuple[int, int, int, int]`。包围框右下边界不包含在区域内；面积比例的分母是有效 Mask 面积，不是整图面积。

Web 的 `serialize_color_regions()` 接收 `ColorRegionSet`，返回包含 `kind`、`count`、原图 `size` 和卡片 `items` 的 JSON 字典。卡片提供 `id`、`pixels`、`fraction`、`color`，以及 PNG data URL 格式的 `mask` / `rgba` 缩略图。运行时区域对象仍供下游节点使用，不从预览 JSON 反推全尺寸数据。

## 使用边界

- 这是颜色聚类，不是“红色 / 蓝色 / 衣领 / 袖口”的语义识别；阴影、印花、反光和渐变可能影响类别数。
- 最终类别数自动得到，但最多为 32；`detail` 不保证对应某个固定数量。
- 编号按**本图面积**排列：1 是本图最大颜色区域。下一张的 1 可能完全不同，不应作为跨图稳定颜色身份。
- 批量流程固定选择编号时，某张图片可能没有该编号，会报错；可使用逐张任务的“出错后继续”。
- 输入错误、尺寸不一致、精细度不在 0/1/2、有效 Mask 为空或选择编号不存在时，会返回明确错误，不静默生成空结果。
