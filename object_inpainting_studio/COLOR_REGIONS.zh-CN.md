# Mask 内自动颜色分区

`AutoColorRegionsNode` 根据原图和 Mask 自动确定颜色区域数量，不需要输入“有几种主色”。只在 Mask 内计算颜色，不运行 SAM / FLUX，也不加载模型权重。

> 界面已统一显示名：`image` 显示为 `rgb_image`、`regions` 为 `color_regions`、`images` 为 `image_list`、`labels` 为 `label_map`。本文接口字段和工作流 JSON 保持旧名称；悬停端口可查看内部字段，旧流程无需重连。详见 [端口命名对照](USAGE.zh-CN.md#端口统一命名与连线提示)。

## 页面使用

1. 重启 Studio 并刷新页面。当前虚拟环境已安装新增的分色依赖；其他环境需安装 [Studio 依赖](requirements.txt)。
2. 左侧点击 **⑪ Mask 内自动颜色分区**，进入单节点练习；这会替换当前画布，先保存已有工作流。
3. 上传 `image` 原图和 `mask`，或在现有画布添加节点并连线。两张图片必须尺寸一致。
4. 保持 `detail=1`，点击节点测试按钮。右侧显示自动得到的区域数、代表色、面积比例，以及各区域的 Mask / 原色透明抠图。
5. 想合并近似颜色可设 `detail=0`；想区分更接近的颜色可设 `detail=2`。这个参数不是颜色数量。
6. 若要使用或保存某一区域，添加 **选择颜色区域** 节点，连接 `regions`，再填写 `region_id`。
7. 若只要每种颜色的 **32×32 纯色图**，添加 **提取区域颜色 · 32×32**，连接 `regions`，`indices` 填 `[0,1]`，再将 `images` 接保存节点的同名输入。具体步骤见下文。

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

### ColorRegionSwatchesNode：独立 32×32 颜色图

连接顺序：`AutoColorRegionsNode.regions` → `ColorRegionSwatchesNode.regions`；各个 `imgN` 可以分别连接 `ColorSimilarityNode.image1 / image2` 或 `SaveImagesNode` 的独立保存项。新节点不再返回图片列表。

1. 在左侧“添加节点”中选择 **提取区域颜色 · 32×32**，不要切换单节点练习替换已有画布。
2. 连接自动分色的 `regions`。`indices` 默认 `[0]`，即面积最大的一种颜色，对应一个输出 `img1`；填 `[0,2,1]` 并按 Enter，画布立即出现三个独立输出：`img1 → index=0`、`img2 → index=2`、`img3 → index=1`。数量取决于列表长度，不固定为四项或八项。
3. 点击测试。每个输出都是 **一张 RGB 32×32 纯色图**，不是列表、合成色卡或原图裁剪。每张填充对应区域的平均 RGB，而不是 `overlay` 的伪彩色。
4. 将所需的 `imgN` 分别接到 **保存图片 / 动态保存项** 节点的图片项，可为每项填写不同子目录名；或直接接到色差评估节点。保存目录 `dir` 填好后，未启用条件重试时点击 **仅保存当前张 · 不循环**；启用条件时须走条件运行，通过才保存。
5. 保存子目录由保存项名称决定，留空使用保存项自己的 `imgN`。同一轮文件名相同，均为 PNG。`filename` 留空时生成新编号；指定组名时，已有文件会报错而非覆盖。

| 方向 | 名称 | 节点端口类型 | 含义 |
| --- | --- | --- | --- |
| 输入 | `regions` | `COLOR_REGIONS` | 自动分色的运行时区域集合 |
| 输入 | `indices` | `INT_LIST` | 直接填写非空非负整数列表，0-based 面积排名；例如 `[0,1]`。输出拓扑需在执行前确定，不支持连线或 context |
| 输出 | `img1`～`imgN` | `RGB_IMAGE` | 第 N 个列表位置的独立 `PIL.Image.Image` RGB 32×32 色块；可单独连线 |
| 输出 | `colors` | `JSON_RECORD` | 所选索引、1-based 区域 ID、平均 RGB、Mask 内面积比例和色块尺寸；此元数据不自动写文件 |

注意：
- 此处 **索引从 0 开始**；与“选择颜色区域”的 `region_id` 从 1 开始不同。
- `[1,0]` 按第二大、第一大的顺序输出；`imgN` 表示**输出位置**，不是区域 ID。重复索引会输出独立的重复色块。
- 缩短列表会移除多余输出并断开对应下游输入 / 工作流输出；存续端口的连线保留。重排列表会改变这些端口的颜色含义，不会自动重连。运行、任务状态待确认或计划检查中禁止改输出数量；无效输入不会改图。
- 下载 / 导入保留索引和独立端口连线。色块节点只提供 `imgN` 与 `colors`；旧 `images` 输出及自动兼容已删除。旧工作流必须将 `swatches.images` 改接为各个 `imgN`，例如 `swatches.img1 → save.img1`、`swatches.img2 → save.img2`；不会自动转换旧列表连线。
- 列表必须非空，只接受整数，不接受布尔值、小数或负数。任何索引超出当前图范围都会明确报错，不会悄悄少保存。
- 每张图的区域数量和排名独立；批量中的单色图无法选择 `[0,1]`。可改用 `[0]`，或用逐张任务的“出错后继续”记录失败并处理下一张。
- 色块生成只用 CPU、不会写盘；只有执行保存节点才写入文件。连接的上游若包含模型，测试仍会执行必要的上游模型。

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
