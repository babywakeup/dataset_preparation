# FLUX.2 衣服随机换色：从真实 target 生成 source

`FluxRandomRecolorNode` 真实调用现有的 `Flux2KleinPipeline`，默认使用本地 FLUX.2-klein-base-9B。source 的换色像素来自模型生成，不是程序填色；节点内部使用涂灰原图作为位置参考，并输出 `masked_source` 涂灰辅助图。

> 界面已按真实用途统一命名：输入 `target` 显示 `rgb_image · 目标原图`，输入 `mask` 显示 `masked_rgb_image · 涂灰原图，不是黑白 Mask`。GrayMaskNode 的 `source` 输出也显示 `masked_rgb_image`，直接连接同名端口即可。换色输出 `source` 显示 `rgb_image · 换色结果，未涂灰`，输出 `masked_source` 显示 `masked_rgb_image`，输出二值 `mask` 才显示 `mask`。下文 Python / JSON 接口仍使用原字段，已有流程无需改名。

## 怎么使用

1. 重启 Studio，刷新浏览器，保存已有工作流。
2. 左侧点击 **⑫ FLUX2 衣服随机换色** 进入单节点练习；这会替换画布。也可从节点目录添加到现有流程。
3. 将真实原图连接到 `target`，将 **GrayMaskNode.source** 直接连接到 `mask`。这里的 `mask` 输入现在是 RGB 涂灰原图，不再接受黑白 Mask。两条分支必须使用完全相同的原图。
4. 保持 `seed=-1`、`num_inference_steps=28`、`guidance_scale=4`，点击测试。`preserve_unmasked=1` 默认开启 Mask 合成；设为 `0` 则直接输出模型生成图。
5. 右侧检查 `source`，确认轮廓、印花、褶皱和纹理没有明显变形。
6. 若要保存本次候选，把输出的 `actual_seed` 填回输入 `seed`，再连接保存节点执行。测试与保存是不同执行，随机模式会重新选色和生成。

固定种子只在相同模型、输入、参数和运行环境下便于复现，不保证跨硬件逐像素一致。

## 输入接口

| 名称 | 端口类型 / Python 类型 | 含义 |
| --- | --- | --- |
| `target` | `RGB_IMAGE` / `PIL.Image.Image` | 真实 RGB 原图，宽高至少 16，不会被节点修改 |
| `mask` | `RGB_IMAGE` / `PIL.Image.Image` | GrayMaskNode.source 的同尺寸 RGB 涂灰原图；编辑区为 RGB(128,128,128)，其余与 target 完全相同 |
| `seed` | `INT` / `int` | -1 每次随机；0～2**32-1 固定种子，默认 -1 |
| `num_inference_steps` | `INT` / `int` | 正整数，base 模型默认 28 步 |
| `guidance_scale` | `FLOAT` / `float` | 有限非负引导强度，默认 4.0 |
| `preserve_unmasked` | `INT` / `int` | 1（默认）合成保留 Mask 外原图；0 关闭合成，只裁剪生成图回原尺寸。仅接受整数 0/1 |

## 输出接口

| 名称 | 端口类型 / Python 类型 | 含义 |
| --- | --- | --- |
| `source` | `RGB_IMAGE` / `PIL.Image.Image` | 原尺寸随机换色结果；默认 Mask 外逐像素复用 target，关闭合成时使用整张模型生成图 |
| `masked_source` | `RGB_IMAGE` / `PIL.Image.Image` | source 的编辑区填 128 灰色，其余不变 |
| `mask` | `MASK` / `PIL.Image.Image` | 从涂灰差异恢复的原尺寸二值 L Mask，值为 0/255，见灰色歧义限制 |
| `sampled_color` | `RGB_IMAGE` / `PIL.Image.Image` | 128×128 候选色块，描述此次生成 source 的目标色 |
| `color_name` | `STRING` / `str` | 候选色的英文名称，不是实际生成颜色的测量值 |
| `actual_seed` | `INT` / `int` | 解析随机模式后实际使用的种子，可填回 seed |
| `generation_prompt` | `STRING` / `str` | 本次生成 source 的英文指令 |

所有新函数均有输入、返回类型标注和 `Args` / `Returns` 文档，见 [节点实现](../object_inpainting/nodes/flux_random_recolor.py)。

## 生成过程与限制

- 从 48 种常用服装颜色中随机选择候选色，优先排除与编辑区中位 RGB 距离小于 80 的颜色；不是在整个 RGB 空间均匀采样，也不保证实际生成色与原色存在指定色差。
- 色表包括黑白灰 8 色、米棕 8 色、红粉 8 色、橙黄 8 色、绿 6 色、蓝 6 色、紫 4 色，覆盖米白、燕麦、卡其、驼色、酒红、豆沙粉、军绿、牛仔蓝、薰衣草紫等常见配色；每色绑定固定英文名和 RGB 色块。银灰 / 香槟仅指布料颜色，不要求金属材质。
- 无需修改工作流、端口或种子参数。固定 seed 在同一版本及相同输入下可复现候选色；从旧 16 色表升级后，同一 seed 可能选择不同颜色。色表只约束目标参考色，不证明真实模型能准确匹配；仍需用输出预览或色差节点验收。
- 原图、编辑区涂灰的原图、候选色块依次作为三张参考图，提示模型只改衣服颜色，并从第一张原图保留姿势、纹理、印花和光影。
- 第二张参考图直接使用输入的 `mask` RGB 涂灰图，连接 `GrayMaskNode.source`。黑白 Mask 只接在上游 GrayMaskNode.edit_mask；不再兼容直接送给本节点。
- 节点比较两张图恢复内部编辑 Mask：与 target 不同、且变为精确 RGB(128,128,128) 的像素为编辑区。其他变化会报错，禁止上传经过 JPEG 压缩、缩放、调色或软边混色的涂灰图；推荐直接连线或无损 PNG。
- 原图本来就是精确 RGB(128,128,128) 的位置，涂灰前后没有差异，因此无法恢复是否属于原 Mask；恢复的 Mask 不包含它，不擅自填洞或把全部灰色当成编辑区。默认合成会保留这些原像素，关闭合成时则由模型决定。若整个编辑区原本就是该灰色，会因无法恢复任何区域而报错。
- 这是参考图编辑，不是原生硬约束的 inpaint 模型。Mask 内的几何和纹理保留依赖模型，必须质检。
- 为适配宽高 16 的倍数，只在右侧和下侧补边，生成后裁回原尺寸，不对原图进行缩放。
- `preserve_unmasked=1`（默认）：最后使用二值 Mask 将生成结果合成回原图，Mask 外严格不变。硬边合成可能产生接缝；应优先使用精确的衣服 Mask，不能靠扩张到背景来保证自然。
- `preserve_unmasked=0`：不执行 Mask 合成，`source` 直接使用裁回原尺寸的模型生成图；Mask 外可能变化。两种模式的提示词、参考图、种子与推理参数保持相同，只改变后处理。`masked_source` 始终由实际输出的 `source` 在恢复的 Mask 内涂灰；原图 `target` 不变。旧工作流未指定此参数时仍默认开启合成。
- 不希望改变的印花、皮肤或配饰应排除在 Mask 外。非连通的多个衣服区域可以共用一个 Mask，但会被要求换成同一候选色。
- 节点不自动写文件。模型失败会报错，不会回退成假生成结果。

连接示例：原图同时接 `GrayMaskNode.target` 与 `FluxRandomRecolorNode.target`；分割得到的黑白衣服 Mask 接 `GrayMaskNode.edit_mask`；`GrayMaskNode.source` 接 `FluxRandomRecolorNode.mask`。已有工作流中直接连接黑白 Mask 的旧连线必须删除重连。

模型加载复用现有缓存、推理锁及 `OBJECT_INPAINTING_FLUX_MODEL_DIR` / `OBJECT_INPAINTING_FLUX_DEVICE` 设置，不自动下载权重。默认 base-9B，不是四步蒸馏模型。

## 用于反向训练时不要接反颜色

例：真实 target 是酒红色外套，本节点随机生成蓝色 source。

- 训练输入：蓝色 source + 酒红色配色条件 + masked_source。
- 训练答案：原始酒红色 target。
- `sampled_color` 此时是蓝色，**不能作为“恢复酒红色”的训练目标配色**。
- `generation_prompt` 是生成蓝色 source 的指令，也不能原样当作恢复 target 的训练指令。

恢复 target 的颜色条件需要从原始 target 提取；多色衣服可结合 [自动分色节点](COLOR_REGIONS.zh-CN.md)，但仍需确认区域与颜色。

保存四张训练图片可连接：`img1=原始 target`、`img2=source`、`img3=mask`、`img4=masked_source`。配色条件与元数据需要另行导出；当前保存图片节点不会自动把种子和提示词写成训练清单。

## 已验证范围

自动测试使用替代模型输出验证三张参考图、真实推理调用边界、种子、尺寸、像素合成、上传 API 和页面操作，不代表通过了真实衣服换色质量评估。实际生成样本需人工检查后才能加入训练集。
