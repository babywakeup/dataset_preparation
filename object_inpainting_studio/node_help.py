NODE_HELP = {
    "RecordInputNode": {
        "title": "样本输入",
        "summary": "整理一条样本的原图路径、物体 Mask 路径和物体名称，本身不读取图片。",
        "inputs": {"record": "右侧样本信息：原图、Mask 路径、物体名称。"},
        "outputs": {
            "target_path": "原图路径。",
            "mask_path": "物体 Mask 路径。",
            "object_name": "物体名称，例如 cup。",
        },
        "test": "先填写卡片里的路径和名称，再测试；图片路径存在时会自动显示缩略图。",
        "parameters": [],
    },
    "LoadTargetNode": {
        "title": "读取原图",
        "summary": "从文件读取完整原图，修正照片方向并转换为 RGB；不会生成或修改物体。",
        "inputs": {"path": "一张本地图片的路径，可直接选择图片上传。"},
        "outputs": {"target": "读取后的原图，保留原始尺寸。"},
        "test": "只需要选一张图片，然后点击测试；不需要 Mask、名称或其他节点。",
        "parameters": [],
    },
    "LoadInstanceMaskNode": {
        "title": "读取物体 Mask",
        "summary": "读取已有物体标注并转成黑白 Mask；不会自动检测或分割物体。",
        "inputs": {
            "path": "Mask 图片路径，白色代表物体、黑色代表背景。",
            "target": "对应原图，用于检查两张图片尺寸一致。",
        },
        "outputs": {"object_mask": "二值物体 Mask，像素值只保留 0 和 255。"},
        "test": "分别为 path 和 target 选择 Mask 与原图；尺寸必须一致，灰度大于 127 的区域变成白色。",
        "parameters": [],
    },
    "SelectObjectNode": {
        "title": "检查物体区域",
        "summary": "检查 Mask 是否为空、物体面积是否合适，并计算包住物体的矩形范围。",
        "inputs": {"object_mask": "白色区域表示待处理物体的 Mask。", "config": "物体面积比例上下限。"},
        "outputs": {"object_mask": "通过检查的物体 Mask。", "bbox": "包围框：[左、上、右、下]，单位为像素。"},
        "test": "选择一张黑白 Mask，测试后查看图片和包围框；太小、太大或全黑会报错。",
        "parameters": ["min_object_fraction", "max_object_fraction"],
    },
    "ExtractReferenceNode": {
        "title": "提取物体参考图",
        "summary": "按 Mask 抠出原图里的物体，放到白色方形背景上，作为参考图。",
        "inputs": {
            "target": "完整原图。",
            "object_mask": "物体 Mask。",
            "bbox": "包围框，可连接检查物体区域节点，或填写 [左, 上, 右, 下]。",
            "config": "参考图尺寸及留白比例。",
        },
        "outputs": {"reference": "白底的物体参考图，不是模型生成结果。"},
        "test": "原图和 Mask 必须对应；连接检查物体区域节点取得 bbox，再检查参考图是否保留完整物体。",
        "parameters": ["reference_resolution", "reference_padding"],
    },
    "ExpandMaskNode": {
        "title": "扩展编辑区域",
        "summary": "把物体 Mask 的白色区域向外扩展，给边缘修复留出空间。",
        "inputs": {
            "object_mask": "原始物体 Mask，可直接上传一张黑白图片。",
            "config": "扩展半径 dilation_radius，单位为像素。",
        },
        "outputs": {"edit_mask": "扩展后的编辑 Mask；白色区域将被涂灰并参与重点训练。"},
        "test": "只选一张 Mask，调节扩展半径后测试；半径为 0 时不扩展。",
        "parameters": ["dilation_radius"],
    },
    "AlignSquareNode": {
        "title": "对齐方形训练图",
        "summary": "用同一个方形裁剪范围处理原图和两个 Mask，再缩放到训练尺寸。",
        "inputs": {
            "target": "原图。",
            "object_mask": "原物体 Mask，用来定位物体。",
            "edit_mask": "扩展后的编辑 Mask。",
            "config": "训练尺寸 resolution。",
        },
        "outputs": {
            "target": "方形训练目标图。",
            "object_mask": "对齐后的物体 Mask。",
            "edit_mask": "对齐后的编辑 Mask。",
        },
        "test": "输入三张对应图片；输出应尺寸一致、物体完整，且 Mask 与原图对齐。",
        "parameters": ["resolution"],
    },
    "GrayMaskNode": {
        "title": "制作灰色遮挡输入",
        "summary": "把编辑 Mask 的白色区域涂成灰色，其他像素保持不变。",
        "inputs": {"target": "目标原图。", "edit_mask": "编辑区域 Mask，白色表示需要涂灰。"},
        "outputs": {"source": "灰色遮挡后的输入图，目标是让模型恢复被遮挡物体。"},
        "test": "直接选择原图和同尺寸编辑 Mask；查看白色区域是否变灰、背景是否不变。",
        "parameters": [],
    },
    "BuildPromptNode": {
        "title": "生成提示词",
        "summary": "把物体名称填入固定英文模板，说明恢复物体并保持其他区域不变。",
        "inputs": {"object_name": "物体名称，例如 cup、bag；推荐英文名称。"},
        "outputs": {"prompt": "训练使用的文字提示词，不会调用语言模型。"},
        "test": "只输入 cup 然后测试，不需要任何图片或连线；输出会直接显示为文本。",
        "parameters": [],
    },
    "AssembleSampleNode": {
        "title": "组合训练样本",
        "summary": "把目标图、遮挡图、编辑 Mask、参考图和提示词放到一个样本中。",
        "inputs": {
            "target": "训练目标图。",
            "source": "灰色遮挡图。",
            "edit_mask": "编辑区域 Mask。",
            "reference": "物体参考图。",
            "prompt": "训练提示词。",
        },
        "outputs": {"sample": "组合样本；预览会同时显示四张图片和提示词。"},
        "test": "先逐个测试上游，再连接五项输入；这一步只组合，不保存文件。",
        "parameters": [],
    },
    "ValidateSampleNode": {
        "title": "检查训练样本",
        "summary": "检查尺寸、Mask、物体覆盖、灰色区域、背景一致性和提示词是否正确。",
        "inputs": {
            "sample": "组合好的训练样本。",
            "object_mask": "对齐后的原物体 Mask。",
            "config": "训练图和参考图尺寸。",
        },
        "outputs": {"sample": "通过检查的样本；不通过时显示具体错误。"},
        "test": "连接组合样本和对齐后的物体 Mask；成功表示基础数据规则通过，不代表训练效果已验证。",
        "parameters": ["resolution", "reference_resolution"],
    },
    "SaveSampleNode": {
        "title": "保存图片与分配数据集",
        "summary": "将样本图片保存到磁盘，分配训练/验证集并生成 JSONL 条目。",
        "inputs": {
            "sample": "已检查的样本。",
            "record": "原始样本信息。",
            "config": "验证集比例和随机种子。",
            "output": "数据集输出目录。",
        },
        "outputs": {"entry": "训练数据条目。", "manifest_path": "JSONL 文件路径。", "split": "train 或 validation。"},
        "test": "这是写文件节点，不能安全预览；先检查上游样本，再使用完整流程导出。",
        "parameters": ["validation_fraction", "seed"],
    },
    "WriteJsonlNode": {
        "title": "写入 JSONL 条目",
        "summary": "向清单文件追加一条样本信息，每行记录图片路径和提示词。",
        "inputs": {"entry": "保存样本节点生成的数据条目。", "path": "JSONL 清单路径。"},
        "outputs": {"entry": "已写入的条目。"},
        "test": "这是写文件节点，预览不会执行；通过完整流程的导出按钮使用。",
        "parameters": [],
    },
    "SavePreviewNode": {
        "title": "保存对照预览图",
        "summary": "将目标图、遮挡图、Mask 和参考图拼成对照图并保存到磁盘。",
        "inputs": {"sample": "组合样本。", "path": "预览图输出路径。"},
        "outputs": {"path": "已保存的预览图路径。"},
        "test": "这是写文件节点，不能安全预览；查看组合样本的预览即可，不必先保存。",
        "parameters": [],
    },
    "CropNode": {
        "title": "裁剪图片",
        "summary": "按照指定矩形截取图片的一部分，不缩放，也不保存文件。",
        "inputs": {
            "image": "待裁剪图片，可直接选图或连接读取原图节点。",
            "bbox": "裁剪框：[左, 上, 右, 下]，使用整数像素坐标。",
        },
        "outputs": {
            "cropped": "裁剪后的图片。",
        },
        "test": "选择图片，填写图片范围内的裁剪框，然后点击测试此节点。",
        "parameters": [],
    },
    "SAM2MaskNode": {
        "title": "SAM2 获取物体 Mask",
        "summary": "直接加载 HF 的 SAM2 tiny 权重，使用 Transformers 官方代码根据物体框分割目标。",
        "inputs": {
            "image": "待分割图片，可直接选择图片，或连接原图/裁剪节点。",
            "bbox": "包住物体的框：[左, 上, 右, 下]，坐标相对于当前输入图片。",
        },
        "outputs": {
            "object_mask": "与输入图片同尺寸的黑白 Mask，白色为分割物体，黑色为背景。",
            "quality": "模型预测质量分数，仅供参考，不是与真实标注比较的准确率。",
        },
        "test": "选图、填写物体框，然后点击测试。首次运行从 facebook/sam2-hiera-tiny 下载到 ../hf_cache；后续复用模型。优先 CUDA，也可使用 CPU。",
        "parameters": [],
    },
    "SAM3TextMaskNode": {
        "title": "SAM3 文本分割物体",
        "summary": "用简短英文文本指定物体，无需填写框；仅加载 ModelScope 下载的本地 SAM3，不联网下载权重。",
        "inputs": {
            "image": "待分割图片，可直接选择图片，或连接读取原图/裁剪节点。",
            "text": "要分割的物体，例如 cup、red bag、truck；不是图片生成指令。",
        },
        "outputs": {
            "object_mask": "与输入图同尺寸的黑白 Mask，白色为选中的单个物体，黑色为背景。",
            "bbox": "从选中 Mask 计算的整数框 [左, 上, 右, 下]，右和下不包含，可直接连接 Crop 节点。",
            "quality": "匹配数量、选中物体的预测置信度；不是实测准确率，不会合并多个物体。",
        },
        "test": (
            "选图、填写简短英文物体名，然后测试。默认读取 ../hf_cache/sam3；"
            "可用 OBJECT_INPAINTING_SAM3_MODEL_DIR 指定模型目录。"
            "置信度阈值为 0.5；匹配多个物体时只保留置信度最高的一个。"
            "优先 CUDA，OBJECT_INPAINTING_SAM3_DEVICE 可指定 cpu 或逻辑 cuda:0。首次加载需要等待。"
        ),
        "parameters": [],
    },
    "MaskToRGBANode": {
        "title": "Mask 转透明抠图",
        "summary": "把 Mask 放入原图的 Alpha 通道，输出透明背景 RGBA；不裁剪、不缩放、不改变 RGB。",
        "inputs": {
            "image": "原始图片，可直接选图，或连接读取原图节点。",
            "mask": "同尺寸黑白或灰度 Mask：255 不透明、0 透明、中间值半透明；可连接 SAM3 的 object_mask。",
        },
        "outputs": {"rgba": "与原图同尺寸的 RGBA 图片，Mask 外透明，不是黑底；棋盘格仅用于预览。"},
        "test": "分别选择原图与同尺寸 Mask，或者连接原图和 SAM3 输出，再点击测试。支持灰度 Alpha，不会二值化或写入文件。",
        "parameters": [],
    },
    "RGBAToWhiteBackgroundNode": {
        "title": "RGBA 转白底参考图",
        "summary": "把带透明通道的 RGBA 图合成到白底，输出干净的 RGB 参考图，适合生成商品图或参考图。",
        "inputs": {
            "rgba": "通常来自前一个 MaskToRGBANode 的输出，透明区域会被替换为白色；保留非透明RGB像素。",
        },
        "outputs": {"rgb": "白底 RGB 图片，透明区域已被合成成白色，可直接送给生成模型当作参考图。"},
        "test": "上传或连接一个 RGBA 图，点击测试此节点；它会把透明像素转成白色，不会把外部区域变成黑色。",
        "parameters": [],
    },
    "FluxKleinReferenceNode": {
        "title": "FLUX.2 Klein base-9B 生成图片",
        "summary": "实际调用 Flux2KleinPipeline，输入提示词和最多 3 张参考图，输出生成图片；全部参考图为空时使用文生图。",
        "inputs": {
            "prompt": "生成用的文本提示词，例如 'red coat on white background, product shot'。",
            "reference_1": "第一张参考图，可为空；为空时忽略。",
            "reference_2": "第二张参考图，可为空；为空时忽略。",
            "reference_3": "第三张参考图，可为空；为空时忽略。",
            "width": "输出宽度，至少 16，必须是 16 的倍数，默认 1024。",
            "height": "输出高度，至少 16，必须是 16 的倍数，默认 1024。",
            "num_inference_steps": "去噪步数：base-9B 默认 28，建议 28～50；不是蒸馏版的 4 步。",
            "guidance_scale": "提示词引导强度：base-9B 默认 4。",
            "seed": "随机种子，默认 0；保持输入和参数不变便于复现。",
        },
        "outputs": {"image": "实际生成的 RGB 图片，可预览或连接下游图像节点，不自动保存。"},
        "test": (
            "填写 prompt，可选连接或上传最多 3 张参考图，然后点击测试。首次加载较慢，模型会缓存复用。"
            "默认从 ../hf_cache/models--black-forest-labs--FLUX.2-klein-base-9B 缓存读取模型；"
            "可用 OBJECT_INPAINTING_FLUX_MODEL_DIR 指定完整本地模型目录，缺失时提示错误而不联网下载。"
            "OBJECT_INPAINTING_FLUX_DEVICE 可指定 cpu 或逻辑 cuda:0；注意 CUDA_VISIBLE_DEVICES 下的编号。"
            "旧流程若使用 prompt/reference_images 等组装输出，需改为连接 image 输出。"
        ),
        "parameters": [],
    },
}

NODE_HELP["SaveImagesNode"] = {
    "title": "保存最多四张图片",
    "summary": "将 img1～img4 分别保存到 dir 下同名子目录；空输入跳过，同一组图片使用相同编号。",
    "inputs": {
        "img1": "第一张图片，保存到 img1 子目录；可连接 RGB、RGBA 或 Mask，也可上传。",
        "img2": "第二张图片，保存到 img2 子目录；可为空，不会重新编号。",
        "img3": "第三张图片，保存到 img3 子目录；可为空。",
        "img4": "第四张图片，保存到 img4 子目录；可为空。",
        "dir": "服务器上的保存根目录，支持绝对路径或相对于服务工作目录的路径。",
    },
    "outputs": {"paths": "本次实际保存的图片路径；PNG 保留原尺寸、RGB、透明通道或灰度 Mask。"},
    "test": "连接或上传至少一张图片，填写 dir，再点击“仅保存当前张 · 不循环”。这只保存当前一组；处理整个文件夹请开启“↻ 逐张”，再点击画布顶部“运行全部 · 写入磁盘”。每次保存使用新编号。",
    "parameters": [],
}

NODE_HELP["FolderImagesNode"] = {
    "title": "文件夹图片循环输入",
    "summary": "按相对文件名排序读取文件夹图片；预览一张，连接到保存节点后可批量逐张执行整条上游流程。",
    "inputs": {
        "dir": "服务器上的图片文件夹路径，不是浏览器本机路径。支持 PNG/JPEG/WebP/BMP/TIFF。",
        "recursive": "0：只读当前文件夹；1：包含子目录。批量时自动排除保存节点的输出目录。",
        "index": "从 0 开始的序号：单次测试选择这一张，批量从这一张处理到最后一张。",
    },
    "outputs": {
        "image": "当前原图 RGB，已纠正 EXIF 方向；连接分割、裁剪或 FLUX 参考图输入。",
        "path": "当前原图的完整路径，也可连接读取图片节点的 path。",
        "filename": "当前原始文件名。结果报告记录原图路径与本轮保存路径的对应关系。",
        "index": "当前图片在文件列表中的序号，从 0 开始。",
        "total": "本次扫描的图片总数。",
    },
    "test": (
        "填写 dir，点击测试预览 index 对应的图片。将 image 连接现有处理链，最后接 SaveImagesNode，"
        "开启文件夹上的“↻ 逐张”，在画布顶部选择保存终点并点击“运行全部 · 写入磁盘”。"
        "每轮独立执行，分支共用当前图片，不是多图参考或 GPU batch。"
        "文件列表在任务开始时固定；坏图记录错误，可选择继续。停止需等待正在执行的节点返回。"
        "一次只运行一个批量任务；不要让其他程序同时更改输入文件。"
    ),
    "parameters": [],
}

NODE_HELP["AutoColorRegionsNode"] = {
    "title": "Mask 内自动颜色分区",
    "summary": "只分析 Mask 内的颜色：加权 Lab 细分后按色差自动合并，不需要填写主色数量，不运行生成模型。",
    "inputs": {
        "image": "RGB_IMAGE：原始 RGB 图片，可上传或连接读取原图节点。",
        "mask": "MASK：与原图同尺寸的黑白或灰度 Mask，>127 为有效前景；背景不参与分色。",
        "detail": "INT：0 粗、1 中（默认）、2 细。控制合并精细度，不是区域数量。",
    },
    "outputs": {
        "regions": "COLOR_REGIONS：区域集合。右侧显示实际代表色、Mask 面积占比、裁剪后的 Mask / 原色抠图；连接选择颜色区域节点可取全尺寸数据。",
        "overlay": "RGB_IMAGE：与原图同尺寸的 45% 伪彩色叠加图，伪彩色只标识区域，不代表真实颜色。",
        "palette": "RGB_IMAGE：各区域填充其实际平均 RGB 的主色图；背景保留原图。",
        "labels": "LABEL_MAP：与原图同尺寸的 L 模式整数标签图，0=背景、1～N=区域。不是二值 Mask；可接保存节点，提取 Mask 请用选择颜色区域。",
    },
    "test": "选择原图和同尺寸 Mask，默认 detail=1，点击测试。CPU 自动分析，最多 32 类，不按面积删除小颜色；区域按面积降序编号，1 是当前图最大区域，不代表跨图片固定颜色。没有语义或颜色名称保证，阴影、渐变可能被拆分。",
    "parameters": [],
}

NODE_HELP["SelectColorRegionNode"] = {
    "title": "选择颜色区域",
    "summary": "从自动分色的区域集合选一个编号，输出全尺寸黑白 Mask 和保留原色的透明抠图。",
    "inputs": {
        "regions": "COLOR_REGIONS：必须连接自动颜色分区节点的 regions 输出，不要输入文字或普通 JSON。",
        "region_id": "INT：当前图的区域编号 1～N，默认 1（面积最大）。批量时编号不对应固定颜色，越界会报错。",
    },
    "outputs": {
        "mask": "MASK：与原图同尺寸的 L 模式 0/255 Mask，可接涂灰、抠图或保存。",
        "rgba": "RGBA_IMAGE：与原图同尺寸，区域内保留原 RGB，区域外透明。",
    },
    "test": "先测试自动分色节点并查看右侧区域卡片，将 regions 连接到这里，填写 region_id 后测试。预览不会写文件，保存请连接 SaveImagesNode。",
    "parameters": [],
}

NODE_HELP["FluxRandomRecolorNode"] = {
    "title": "FLUX.2 衣服随机换色 · 生成 source",
    "summary": "真实调用 FLUX.2 Klein，把原始 target 的指定衣服区域随机换色，生成反向训练的 source；Mask 外逐像素保留原图。",
    "inputs": {
        "target": "RGB_IMAGE：真实原图，生成 source 时不会修改它；须与生成涂灰图使用的原图完全一致，宽高至少 16。",
        "mask": "RGB_IMAGE：直接连接 GrayMaskNode.source 的涂灰原图，不是黑白 Mask。编辑区填 RGB(128,128,128)，其余与 target 完全一致，尺寸一致。",
        "seed": "INT：默认 -1，每次随机；0～2**32-1 固定颜色选择和推理种子。批量保持 -1 可逐张随机。",
        "num_inference_steps": "INT：base-9B 去噪步数，默认 28，必须为正整数。",
        "guidance_scale": "FLOAT：有限非负引导强度，默认 4.0。",
    },
    "outputs": {
        "source": "RGB_IMAGE：原尺寸的随机换色结果，可直接接保存节点；仅白色 Mask 内使用生成像素。",
        "masked_source": "RGB_IMAGE：source 的编辑区涂成 128 灰色，用作训练位置提示图。",
        "mask": "MASK：从原图与涂灰图差异恢复的同尺寸 L 二值编辑区域，只有 0/255；原色恰为 128 灰色的位置无法恢复。",
        "sampled_color": "RGB_IMAGE：128×128 随机候选色块，描述本次 source 颜色，不是恢复原 target 的训练目标色。",
        "color_name": "STRING：本次随机候选颜色的英文名称；实际生成色不保证精确匹配。",
        "actual_seed": "INT：本次实际使用的种子；填回 seed 可在相同环境和参数下复现。",
        "generation_prompt": "STRING：生成 source 的英文编辑提示词，不是反向训练恢复 target 的指令。",
    },
    "test": (
        "连接原图到 target，将 GrayMaskNode.source 直接连接到 mask，再点击测试，会实际加载本地 FLUX.2 Klein base-9B。"
        "复用现有模型缓存、OBJECT_INPAINTING_FLUX_MODEL_DIR 与 OBJECT_INPAINTING_FLUX_DEVICE；不联网下载。"
        "原图、编辑区涂灰的原图和候选色块作为三张参考图；涂灰形式与 GrayMaskNode 一致，不是原生硬约束 inpaint。"
        "仅接受 RGB 涂灰图，旧黑白 Mask 输入及连线不再兼容。上传请用无损 PNG，不接受软边混色或其他图像编辑。"
        "内部比较 target 与涂灰图恢复编辑区域；原图本来就是精确 128 灰色的位置无法辨认，会保守保留。"
        "宽高非 16 倍数时只补右下边缘，生成后裁回，不缩放原图。"
        "Mask 内纹理、姿势和颜色仍需人工质检；硬边合成可能产生接缝。"
        "默认随机时，预览和保存会重新生成；想保存同一结果请把 actual_seed 填回 seed 后再保存。"
        "反向训练的目标配色应从原始 target 提取，不能误用 sampled_color。"
    ),
    "parameters": [],
}

CONFIG_HELP = {
    "resolution": "训练图边长（像素，须为 16 的倍数）",
    "reference_resolution": "参考图边长（像素，须为 16 的倍数）",
    "dilation_radius": "Mask 向外扩展半径（像素）",
    "reference_padding": "参考图留白比例（0～1）",
    "min_object_fraction": "物体最小面积比例",
    "max_object_fraction": "物体最大面积比例",
    "validation_fraction": "验证集比例（0 表示全用于训练）",
    "seed": "训练/验证集分配种子",
}
