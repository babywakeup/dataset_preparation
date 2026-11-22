# 项目架构地图

更新日期：2026-11-20。用途：按需定位代码，不要求每轮全部读取。实现变化时同步修正。

## 独立 Qwen 换色 demo

- [examples/recolor/demo_qwen_recolor_pipeline.py](../examples/recolor/demo_qwen_recolor_pipeline.py)参考官方Qwen-Image-Edit-2511 Quick Start，直接调用QwenImageEditPlusPipeline；从本地../../hf_cache/Qwen-Image-Edit-2511加载BF16，输入原图A/涂灰图/主颜色B，prompt不注入目标颜色名/RGB/色号。默认40步、true_cfg_scale4、guidance_scale1、negative_prompt为空格、CPU Generator seed43，输出qwen_recolor_result.png。
- 当前默认组件CPU offload（GPU1空闲80GB实测）；文件顶部use_sequential_cpu_offload可切换逐层offload；不加载LoRA、不调用FLUX训练/精度recipe或mask合成。独立试验入口，不改变现有训练/Studio。

## 独立 FLUX.3 换色训练

- [离线路径数据集构建](../build_flux3_recolor_dataset.py)：HF Dataset.from_generator分别构建子集，concatenate阶段控制比例，再按source_id（完整文件名主干）拆分。仅存Value(string)绝对路径、目标height/width、condition_sizes及身份/prompt，无direction、Image特性或图片bytes。仅读图片头/EXIF，PNG跳过IDAT取晚EXIF；宽高对应转正后尺寸。默认输出datasets/recolor_paths；NFS清理临时cache前释放Dataset mmap引用并gc。训练入口prepare_path_dataset在输出目录检查后/模型加载前由main_process_action主进程直接import调用，通过dataset_cache.py缓存到输入目录上一级datasets/recolor_cache/SHA256并跨run复用；传入当前prompt/captions/source_groups。hash包含构建版本/文件stat/有效构建配置，排除训练与worker参数；FileLock串行同key写入，临时构建后原子发布，并记录cache_config.json。训练现直接读取缓存train/validation；Dataset只用元数据建立buckets/bucket_idx，__getitem__才读路径并同步几何。validation统一记录按固定位置取样，不按子集扩展；日志只有总loss。

- [可学习任务embedding对照](../train_flux3_recolor_text_encoder.py)复用原训练循环，opt-in train_text_encoder；text_encoder.py提供独立Transformers TextEncoderConfig/TextEncoderModel，config.length控制token长度，FP32[1,L,H]从prompt初始化（默认L为实际编码窗口），forward返回BaseModelOutput。与Transformer分别prepare、共同accumulate和clip，LoRA与encoder统一learning_rate/scale_lr（encoder保留零weight decay），输出保留图传入Transformer，不挂子模块/不改forward。验证pipeline引用训练中的text_encoder，共用validation.pipeline_text_embeddings读取并detach正向，CFG负向保留冻结空文本。标准text_encoder/config.json+model.safetensors与LoRA独立保存，hook原位恢复参数及双模型optimizer；text_encoder_format拒绝旧内嵌训练状态，仅支持新text_encoder目录格式，不再提供旧tensor兼容读取。单卡/DDP支持、FSDP拒绝，caption/随机prompt/dropout/验证文本覆盖拒绝。详见[实验说明](FLUX3_TEXT_ENCODER.zh-CN.md)。

- 数学/预处理核对、真实数据检查及验证边界见[训练核查](FLUX3_TRAINING_AUDIT.zh-CN.md)；`tests/test_flux3_math_audit.py` 提供独立解析loss/梯度与官方pipeline差分检查。loss严格拒绝预测/目标形状不一致，防止静默广播。
- [无梯度对照入口](../tools/flux3/compare_flux3_training_pipeline.py)复用实际训练helper，与官方ModularPipeline同noisy latent/sigma比较并分开报告布局/精度/autocast影响；见[对照说明](FLUX3_PIPELINE_COMPARE.zh-CN.md)。真实GPU已发现当前BF16 VAE/训练autocast与用户官方FP32 VAE/无外层autocast不数值等价，现已新增版本化精度方案统一生产入口；不能把旧CPU受控布局测试解读为真实路径等价。
- [数值复核](../tools/flux3/analyze_flux3_numeric_precision.py)重放保存的输入、分离条件与AMP、对照FP32算术和最终舍入。真实GPU结果支持数值路径敏感性，不能把不allclose直接视为代码遗漏、必须修改训练或印花根因；FP32是算术参考而非质量真值，详见对照说明。
- [逐层定位](../tools/flux3/trace_flux3_tensor_path.py)已将同输入AMP差异定位到Transformer首个txt_mode_blocks.0.norm LayerNorm和VAE encoder/decoder首个GroupNorm；临时统一整个归一化类的AMP策略后阶段输出逐位一致，六组Transformer均差0。源头是入口间归一化输出dtype与VAE精度策略不统一，非归一化公式遗漏；历史曾通过自定义norm固定精度，当前已按用户要求删除recipe/norm替换；现在仅固定模型级Transformer autocast、原始FP32 VAE和调用处autocast关闭。历史逐位等价结果不作为当前GPU验收。细节/边界见对照说明及迭代交接。
- [训练/数据检查入口](../train_flux3_recolor_lora.py) 和 [推理入口](../infer_flux3_recolor.py) 独立于下述数据构建/Studio 路径，详见[换色训练说明](FLUX3_RECOLOR.zh-CN.md)。不改本地 Diffusers 示例，不自动下载模型。
- `data.index_dataset`仅在构建/cache指纹层收集配对，根训练入口不再扫描/配对/拆分。cache format v2按各子集实际使用的文件收集stat，保留单个子集中独立有效的记录，不取跨子集交集。
- 根DreamBoothDataset继承torch Dataset，接受HF路径记录；metadata纯几何分桶（H,W），保留原面积/16px网格和参考比例桶。`flux3_recolor/samplers/`保存普通BucketBatchSampler和ScheduledMixBucketBatchSampler；后者保留每桶样本总数，按epoch调整AtoB/BtoA数量，并补齐整batch供Accelerate分发。方向从sample_id的数字子集前缀与training_directions顺序解析，不增加图片或loss方向分支；分桶组成限制目标比例时日志记录实际比例。失败限次替换仅同bucket/repeat/方向，返回实际ID和prompt；色块独立nearest resize。
- 双向训练默认AtoB比例从0.5线性降到0.25；`--atob_ratio_start/end`可改。采样在每个bucket内按方向抽取，维持epoch总样本量，数量不足时有放回抽样；DDP全局batch计划仅交由Accelerate分发一次。单向仍用普通BucketBatchSampler。训练loss与输入目标无方向条件，source groups仍由缓存切分隔离。新训练身份为sample_id+repeat，配置保存比例用于精确恢复。新旧数据管线checkpoint不可直接互恢复。
- `data.read_path_sample`供训练/验证共用，按路径load_rgb/EXIF转正，异分辨率场景先resize至目标画布，然后共享crop/flip；generate_validation按记录输出数字目录及来源manifest。独立验证推理也加载相同cache，可按新dataset_root重定位路径；旧adapter仍可显式三图推理，旧load_validation_sample仅供独立诊断。

- training_logging.TrainingLogger负责tqdm优化器步进和logging_steps（默认10）的区间loss：本地GPU累积loss分子/样本计数，只有日志点及训练最后一步跨rank归约和读取标量、写tracker/history；均值按实际样本数加权。进度从checkpoint步数恢复，FP16跳步不推进，finally关闭。日志窗口不持久化；finite loss逐microbatch检查保留，checkpoint/验证独立。
- 根 `train_flux3_recolor_lora.py` 是唯一训练实现，贴近参考DreamBooth结构；旧 `flux3_recolor/training.py` 仅转发。原生Accelerate prepare/accumulate、AdamW/Prodigy、LR/warmup、PEFT自定义目标、时间采样/weighting、tracker和offload均有实际入口。VAE统一在线编码：modeling.encode_recolor_latents沿batch维合并目标/涂灰/源图，一次编码后拆回，色块单独编码（每microbatch两次），复用encode_latents的FP32/mode/BN/patchify；不再提供latent缓存或相关几何限制；路径Dataset hash缓存和固定prompt预编码保留，自定义caption按当前batch在线编码，dropout空文本匹配当前窗口；使用accelerator.unwrap_model(transformer).config检查维度，避免DDP包装访问错误。固定prompt和变体保留预编码。
- `modeling.py` 保留三参考L=-1/-2/-3、VAE BN/patchify、flow matching、标准adapter及alpha metadata。普通DDP状态仅含LoRA/optimizer/LR/scaler/每rank RNG与epoch/batch位置；FSDP1使用原生完整/分片状态（含基础模型），另导出LoRA。新状态version=2拒绝旧手动累积checkpoint，恢复要求相同world_size/关键配置，非constant调度不能改变horizon。只清理有本入口标识且步数匹配的完成checkpoint。
- 当前固定精度：load_training_components/load_components直接from_pretrained加载FP32 VAE，所有权重/内部模块/forward保持原生；encode_latents仅对调用局部关闭autocast。precision.py仅set_transformer_autocast和pipeline_forward（临时旁路Accelerate输出FP32转换并恢复），训练/验证/推理的Transformer前向共用上下文，pipeline外层不启用AMP。删除precision_recipe参数/版本分支、自定义norm、VAE标记；latent BN和FP32 loss保留，噪声/标签跟随latent dtype，无重复强制target.float。
- 单卡/DDP验证缓存原生pipeline并共享训练Transformer对象，不复制权重；scheduler deepcopy。验证临时恢复Accelerate包装前forward以避免输出强制FP32影响采样，finally恢复训练forward/train mode和RNG。FSDP主rank仍需汇集LoRA并加载副本，不能单rank直接调用训练分片模型。
- initialization.inherit_model_args从组件manifest/LoRA safetensors元数据继承基模、完整PEFT配置、rank/alpha/dropout/targets，来源learned text_encoder自动启用。--init_from仅继承模型权重，optimizer/scheduler/RNG/data position/step重新开始；新目录保护来源，config保留init_from provenance。resume也读自身LoRA元数据构造结构（含pattern/RSLoRA/FSDP验证副本），从自身checkpoint加载完整状态，不依赖原init目录；LoRA与text_encoder在optimizer前初始化，原参数引用不变。用户显式结构冲突拒绝，基模签名检查；新默认LoRA目标不影响旧adapter结构。
- component_io.py统一checkpoint/最终导出的组件保存：transformer/仅标准Diffusers LoRA+adapter metadata，训练的text_encoder/完整save_pretrained，冻结vae/Qwen不导出；model_config.json记录base_model及组件来源，weights_layout=components_v1拒绝旧目录恢复。inference.load_recolor_pipeline消费manifest定位基础模型（可覆盖路径）、校验签名并加载LoRA/custom encoder，返回原生pipeline及recolor_config/recolor_embeddings；generate_sample可默认使用这些文本条件，CLI共用该加载路径。FSDP内部恢复仍使用原生Accelerate状态。
- `inference.py` 使用本地 FLUX.3 ModularPipeline，移除 prompt upsampler 与内部文本编码块，传入按训练设置预编码的正/负文本；复用三图预处理，输出生成图及对照。基础权重限定非蒸馏 `flow_map=False`，要求完整 Diffusers 格式。
- `config.py` 支持普通模型目录、具体snapshot和带refs/main的HF缓存根目录；按 `modular_model_index.json` 解析tokenizer/processor子目录，无元数据时兼容独立tokenizer或text_encoder内tokenizer_config。官方base的processor实际在text_encoder；不能硬要求tokenizer目录。权重索引会校验分片存在性，缓存/权重始终只读。
- processor直接从解析后的组件路径加载，避免Transformers5.5.4用BFL根config做Mistral tokenizer启发判断；不要给标准Qwen regex盲加Mistral修补。已核对base snapshot的Qwen使用共享输入embedding/lm_head权重，权重文件不单列lm_head不是缺片。
- 训练默认TensorBoard，每100个优化步验证4张固定样本（合并validation记录均匀选取，总共4张），按数字子目录保存。validation.ValidationRuns以step/配置/选中sample_id/seed/推理步数在本进程内去重，成功才记完成；log_validation先统一判定再进入FSDP收集与主进程广播，恢复进程重新验证。tracker图片记录集中validation.log_validation_images；validation_output先写同级临时目录，生成成功后替换该步旧目录；独立推理保持非空输出保护。tracker使用稳定validation/generated、validation/comparison标签及global_step，实际执行final才记test；验证后flush，所有rank同步后继续。report_to=none只写盘，num_validation_images=0关闭验证。
- 单卡Accelerate可能返回无index的torch.device('cuda')；CUDA RNG隔离和量化device_map用cuda_device_index回退到torch.cuda.current_device，不能把None传给fork_rng。同一步checkpoint先于验证提交，避免首次验证失败时丢失该保存步的训练状态。
- 新训练第三条件色块在共享data预处理以NEAREST独立resize到默认128×128（--color_swatch_size），保留RGB且不改原文件；policy记录color_swatch_size，验证/推理复用，训练在线按当前预处理编码。旧推理policy缺该字段时继续原生尺寸；新128预处理不能直接resume旧32配置。
- FSDP1要求use_orig_params=true与统一FP32 Transformer/LoRA master，裁剪向Accelerate传完整参数列表以计算全局norm；可选text encoder FSDP。FSDP只支持普通AdamW，无量化/FP8/8bit/Prodigy组合；bitsandbytes量化只支持单GPU。上述GPU路径与真实显存/质量均未验收，不从CPU小模型结果推断成功。

## 1. 两条使用路径

| 路径 | 入口与职责 |
| --- | --- |
| CLI 数据集构建 | [启动文件](../build_object_inpainting_dataset.py) 仅调用 [cli.main](../object_inpainting/cli.py)；解析 manifest、输出目录、可选 workflow，构造 `PipelineConfig` 后调用 `build_dataset` |
| Studio 交互处理 | [启动脚本](../studio_object_inpainting.sh) → [模块入口](../object_inpainting_studio/__main__.py) → [FastAPI 服务](../object_inpainting_studio/server.py)；提供图编辑、预览、保存、文件夹批处理和数据集导出 |

两条路径共享节点和图执行器，但“保存当前图片 / 文件夹批处理”不等于 CLI 的完整训练数据集导出。不要把当前打开的薄入口当成主要实现。

## 2. CLI 数据流和保护条件

- [records.py](../object_inpainting/records.py)：JSONL 每行要求 `target`、`mask`、`object_name`；相对路径基于 manifest 所在目录解析。规范化路径与去空格后的对象名产生 24 位哈希 ID；重复记录报错。
- [默认工作流](../object_inpainting/workflows/object_reference.json)：加载图像/掩码 → 选对象 → 提取参考图、扩展掩码 → 方形对齐 → 编辑区填灰 → prompt → 组装与校验 → 保存样本、JSONL、预览。
- [config.py](../object_inpainting/config.py)：`PipelineConfig` 管理尺寸、扩张、参考图 padding、对象比例、验证集比例、种子等；修改参数前检查其校验。
- [quality.py](../object_inpainting/quality.py)：`PreparedSample` 与校验；默认训练样本约束包括二值 mask、非编辑区保持 target、编辑区为灰色 128。不要直接套用到所有 Studio 节点输出。
- [pipeline.py](../object_inpainting/pipeline.py)：`prepare_sample` 只请求 `sample`；`build_dataset` 要求 `sample`、`entry`、`split`，有预览时还要求 `preview` 输出。`workflow` 与 `graph` 参数互斥。
- 输出目录必须为空或不存在；先在同级临时目录生成再迁移，拒绝覆盖已有数据集。全部样本被拒绝时仍保留输出诊断后抛错。
- 输出含 train/validation/rejected JSONL、summary、workflow 和可选 previews；样本导出字段与路径规则见 [exporting.py](../object_inpainting/exporting.py)。

## 3. 图和节点扩展

- [graph.py](../object_inpainting/graph.py)：`NodeGraph` / 图执行；工作流由 `nodes` 和 `outputs` 构成，输入可为字面量、`{"context": "config"}` 或 `{"node": "id", "output": "port"}`；有类型与环路检查。
- 节点约定：先在 [node_contracts.py](../object_inpainting/node_contracts.py) 定义 TypedDict 入口 / 出口类；节点以 `INPUT_SCHEMA` / `OUTPUT_SCHEMA` 指定，配合显式 `execute()` 参数和返回注解。[base.py](../object_inpainting/nodes/base.py) 自动生成 `INPUT_TYPES` / `OUTPUT_TYPES`，保留旧式自定义节点的手写字符串表支持。所有 28 个内置节点已接入；新增后仍需在 [节点注册表](../object_inpainting/nodes/__init__.py) 注册。
- 公共类型见 [port_types.py](../object_inpainting/port_types.py) 的 `PortType` / `ImageRole` 及 PIL Annotated 别名，不包装图像。涂灰图 `MaskedRGBImage` 与 Mask `MaskImage` 分开标注；SAM2 框使用 `FloatBoundingBox`。完整 [入口出口类型指南](NODE_TYPES.md) 说明合同、限制及全节点对照。
- 动态保存项是静态合同的显式扩展：[基类](../object_inpainting/nodes/base.py) 提供 `prepare_spec`、`input_types`、`optional_input`、`execute_with_spec` 钩子；其他节点默认仍使用原静态端口及 `execute(**inputs)`。`NodeGraph.input_types[node_id]` 为实例端口真相，用于集合 / 连线 / 运行时校验和 Studio 上传解码。只有保存节点依据 `save_items` 生成任意 `imgN: IMAGE`，不放宽其他节点的未知输入校验。
- 动态输出通过基类 `output_types(spec)` 与 `NodeGraph.output_types[node_id]` 定义；所有实例归一化后再检查连线，返回字段集合及条件标量读取同样使用实例表。[色块节点](../object_inpainting/nodes/color_region_swatches.py) 依据字面量 `indices` 生成 `img1`～`imgN: RGB_IMAGE` 和 `colors`，TypedDict 是默认单口模板。输出数在执行前确定，拒绝 context / 连线索引，不按实际区域数悄悄减口。
- 新节点通常还需对应测试；面向 Studio 时检查 [node_help.py](../object_inpainting_studio/node_help.py)、服务端输出序列化及前端输入/预览支持。
- 节点预览有副作用限制；不要为了预览而绕过保存节点的写盘保护。
- 运行异常统一封装为 `NodeExecutionError(ValueError)`，携带原始失败节点 ID / 类型和错误类型 / 原因，保留异常链；上游错误不重复包装。`GraphExecutionCancelled` 是取消控制信号，不作为节点错误。
- CLI 默认图与用户导入的根目录示例工作流不是同一来源；修改前确认实际使用的是哪份 JSON。

## 4. Studio 定位和易混点

| 需求 | 优先查看 |
| --- | --- |
| API、配置解析、节点预览/保存 | [server.py](../object_inpainting_studio/server.py) |
| 跨请求预览上游复用 | [preview_cache.py](../object_inpainting_studio/preview_cache.py)：页面持有不透明 `preview_session`；只复用祖先，显式目标始终执行，保存操作不缓存；CLI / batch 不接入 |
| 统一端口显示名 / 用途元数据 | [port_names.py](../object_inpainting_studio/port_names.py)，API catalog 的 `port_labels.inputs/outputs[原字段]` 提供 `name`、`role`、`kind`；不修改运行时字段 |
| 节点入口 / 出口类 | catalog 的 `input_contract` / `output_contract` 和卡片 / 右侧说明展示实际 schema 类名；见 [类型指南](NODE_TYPES.md) |
| 后台批处理、取消、结果统计 | [batch.py](../object_inpainting_studio/batch.py) |
| 画布、端口、节点 UI、状态 | [app.js](../object_inpainting_studio/static/app.js)、[index.html](../object_inpainting_studio/static/index.html)、[style.css](../object_inpainting_studio/static/style.css) |
| 节点删除与连线清理 | [node_editing.js](../object_inpainting_studio/static/node_editing.js) 的纯工作流编辑函数；[app.js](../object_inpainting_studio/static/app.js) 的按钮、Delete 快捷键、运行锁和状态清理 |
| 文件夹逐张循环、保存终点 | [loop.js](../object_inpainting_studio/static/loop.js) |
| 文件夹全部末端范围与结果收集 | [batch_scope.py](../object_inpainting_studio/batch_scope.py)：源文件夹下游末端、共享上游、所有结果处理与多保存；普通节点预览仍按目标依赖 |
| 显式逐张结果 / 禁用分支 | 可选 `workflow.batch_execution`：`mode=all_terminals/selected_results`、`result_nodes`、`disabled_nodes`；未配置保持全部末端，禁用传播到下游，指定结果依赖禁用节点时报错。支持直接选择中间节点和无保存结果，所需条件循环仍先验收，达标缓存包括被选择的循环内普通节点 |
| 逐张执行计划与任务层次 | `/api/batch/plan` 返回 `executed_nodes`、`repeated_nodes`、`node_roles`、`terminal_nodes`、`write_nodes`、`stages`；前端只发现输入 / 入口，执行范围不再自行推导。完整请求匹配的计划才显示角色与高亮，计划返回后重画连线。外层 `BatchJobs` 每张建立独立 `current_item`，内层条件执行器管理该图片的重试；任务通过 `execution_phase` 区分 input/retry/results/completed |
| 多条件循环、点击向导与逐轮缓存 | [conditional_loop.py](../object_inpainting_studio/conditional_loop.py)，[multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js)；[loop.js](../object_inpainting_studio/static/loop.js) 提供文件夹工具栏、共享点选与请求；后台复用 [batch.py](../object_inpainting_studio/batch.py) |
| 动态保存项、命名与文件冲突 | [save_images.py](../object_inpainting/nodes/save_images.py)、[app.js](../object_inpainting_studio/static/app.js) 的 `saveItems` / `nodeInfo` 和增删编辑函数 |
| Studio 默认原图命名 | [naming.py](../object_inpainting_studio/naming.py) 在 `node_preview_graph` 构建后的独立图上补全空 filename：唯一上游 FolderImages 连接其动态 filename，否则唯一字面路径 LoadTarget 使用路径 basename；显式 filename 优先，无明确来源保留节点 UUID 后备。仅引用已存在祖先，不增加执行分支，不修改用户工作流。覆盖逐张/条件/单图保存，PNG 扩展名与不覆盖规则由 SaveImages 保证 |
| 操作语义与保存目录 | [中文使用指南](../object_inpainting_studio/USAGE.zh-CN.md) |
| 自动颜色区域 / 随机换色 | [颜色区域指南](../object_inpainting_studio/COLOR_REGIONS.zh-CN.md)、[换色指南](../object_inpainting_studio/RANDOM_RECOLOR.zh-CN.md) |
| 面积排名列表 → 独立 32×32 代表色图 | [color_region_swatches.py](../object_inpainting/nodes/color_region_swatches.py)，`indices=[0,2]` → `img1/index=0`、`img2/index=2`，分别接评分 / 保存图片项；仅提供 imgN/colors，旧 images 已删除 |

- 批处理状态在进程内，不应假设服务重启能续跑。
- 色块列表缩短时前端清理已删除输出的下游输入和工作流输出，保留其余连线；重排按位置改变内容。卡片 / inspector / edges / 预览统一采用 `nodeInfo()` 实例输出；拓扑编辑受运行 / 检查锁保护，保留历史预览与会话。色块旧 images 兼容输出、标记处理和迁移均已删除，旧连线需改接独立 imgN；无自动转换。仓库工作流 (6)/(8)/(11) 已清理旧设置，原列表保存改接独立色卡，保留 images_0 目录名；前端测试检查 JSON 无残留。SaveImages 的通用图片列表输入不受影响。
- 条件循环是 Studio 控制层，不改普通 DAG 为任意有环图。新 `conditional_loop={loops:[...]}` 支持 1～32 个 ID 唯一的循环，每个含 `id/name`、`start_nodes/end_nodes/seed_nodes`、`conditions[{node, output, operator, value}]`、`mode=all/any`、`max_iterations`、`timeout_seconds`、`on_exhausted=fail/skip`。旧 flat 格式继续用 `ConditionalPlan` 原语义，两种根字段禁止混写；`build_conditional_plan()` 分派。普通 `NodeGraph.start/run` 继续拒绝条件元数据，不能绕过验收。暂不支持重叠 / 嵌套 / 反馈。
- `prepare_conditional` 裁剪所选 SaveImages、所有条件及显式 end 的祖先，共享归一化后的图给计划与上下文。`MultiConditionalPlan` 取 `descendants(start) ∩ ancestors(end+conditions)` 为各体，验证 start 可达 end、条件 / 种子在体内、非重叠、与所选保存有数据关系，并按传递依赖拓扑排序（含普通 bridge）。end 是验收根，不是普通下游可读端口的白名单：普通节点和所选保存均可消费任意已验收体内输出，无需追加 end。完整定义仍用于边界 / 直接工作流输出等校验；额外写盘检查仅针对本次执行切片，未选保存分支不执行。布尔只支持 == / !=，数字有限；显式递增种子须非负 INT，逐文件重置。
- `/api/conditional/plan` 只检查结构，不推理或写盘；`/api/conditional` 启动当前图的单图后台任务。`/api/batch/plan` 包含所选文件夹所有下游末端及依赖，支持无条件评分和多个保存分支；工具栏运行入口只用于定位文件夹。启用条件时，所有末端在验收后复用达标缓存执行，拒绝评分上游写盘；`/api/batch/preview` 为异步单文件任务并使用批量过滤后的同一 index。所有 SaveImages 输出目录参与输入扫描排除，后续保存错误清理本张此前新建的 SaveImages 文件，取消仍保留已完成输出。结果保留主入口 `paths`，另含 `paths_by_node` 与全部末端预览（仅当前输入）。单图 / 文件夹 / 条件试跑共用 `BATCH_JOBS` 互斥。计划与上下文使用同一图对象，避免图片路径解码重写后运行旧图。
- `/api/conditional/preview` 接收 `loop_id`，无需保存节点 / 目录。按原始图祖先闭包选本循环和必要上游（含各上游自己的全部 end / conditions），再构图校验，避免无关半成品阻塞；复用 `MultiConditionalPlan` / `_run_multiple` / `BATCH_JOBS`，不新建重试器。计算切片拒绝副作用，不复用普通预览缓存；依赖循环必须先验收。末轮 end / 条件及本体已计算图片可写入 `results[].preview`，失败耗尽也有末轮，超时不补跑，不为展示再次生成。
- 普通 `previewNode()` 自动请求 `/api/node/preview`；后端通过 `required_preview_settings()` 识别目标所需循环，包括测试循环内部节点的情况，裁剪并提交后台条件任务。先验收必要循环，再从达标缓存计算目标，失败不执行普通下游，不会额外生成；无循环依赖时沿用普通预览缓存。前端旧 flat 先迁 loops；直接 API 对带 flat 条件配置的普通节点测试返回 422 防绕过，旧 flat 正式条件保存仍兼容。
- `#visualize` yes / no 默认 yes，请求传布尔 `visualize`。开启时每轮推送 `loop_previews`（按 loop ID 仅存一份最新图片快照）和递增 `visualization_revision`，前端同步画布节点 / 循环卡片 / 任务输出区；评分 `history` 保留各轮，不累计全尺寸图片历史。关闭时后端不序列化图片，不影响生成 / 评分 / 保存计算。中途选 no 立即隐藏前端图片，已在服务器开启的当前任务采集不会实时关闭，下次请求才关闭。刷新可从最近任务恢复图像快照；快照不是当前图配置已验收的证明，每卡切换查看不清除其他卡快照。
- [循环演示节点](../object_inpainting/nodes/loop_demo.py) 是两个无模型 CPU 节点：可选 RGB 原图上添加左上角最多 32×32 标记（不接图时为 32×32 色图），seed 按红 / 绿 / 蓝 / 灰 / 黄 / 青 / 紫七色循环，-1 随机并输出实际 seed；保留每轮 10 秒模拟等待。评分只取标记区域蓝通道均值，原图其余部分不影响验收。面板追加两个独立示例，不替换原流程；seed=0/1、阈值 >=255 分别应 3/2 次通过。文件夹 image 可接入演示节点，filename 直接接保存节点；逐文件重置 seed 并仅保存达标图。实际运行状态以交接为准。
- 每轮新 `ConditionalExecution`，固定输出跨尝试深拷贝且交给下游前隔离；多循环通过后强制执行其全部 end、保留已验收体缓存，后续重试不重跑前序。普通 bridge 只有全部依赖循环已验收才可冻结，不可保留失败轮污染。全部循环通过才从已验收缓存执行唯一所选保存一次；耗尽 / 超时不保存，异常不作为评分失败重试。报告兼容 accepted / iterations / history / stop_reason / skipped，增加每循环结果及 history 的 loop_id / loop_name；进度 current_loop_* 逐文件重置，不累计全尺寸历史图片。
- 条件任务超时 / 取消在节点前后及保存依赖就绪后检查，无法强杀节点内 GPU 调用；进入保存后的取消不抹去已完成文件记录。前端点击 SaveImages 节点运行时自动校验并提交条件保存，无需手动检查计划或选择公用保存目标；底层普通 `/api/node/save-images` 仍拒绝带条件配置的请求，防止绕过。前端复用任务状态待确认锁，配置 / 节点变更使计划失效，轮询不重建聚焦的配置输入。
- [multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js) 管理唯一点击式多循环界面：按 `picking.loopId` 精确绑定 start→end→condition，单标量点标题、多个则点输出；仅保留三项点选 / 重选、condition operator / value、预览及删除动作。循环命名输入、多条件增删 / all-any、高级多起终点、种子 / 次数 / 超时 / 耗尽、公用保存选择和手动检查计划 UI 均已删除。默认安全限制和非负起点 INT seed 递增内置，-1 保持随机。旧 JSON 的名称、多条件 / mode、多边界、种子及重试参数仍兼容保留，不因渲染 / 下载丢数据；明确重选条件会用一个新条件替换旧列表。`normalizeImportedConditional()` 在 restore / render / Add 深拷贝旧 flat 为 loops，仅补缺省，end 由旧体流向保存的来源及条件节点推导，不猜失效引用、不改原对象 / 磁盘文件，下载统一新格式；坏原始值须明确删除。改配置保留任务快照，删最后循环恢复普通保存；运行锁 / 恢复清 picker，轮询不重建聚焦字段。
- 预览缓存按节点规范化 spec、所用 config / record、文件 stat、上游输出版本判定有效性；文件夹字面量输入还扫描图片清单和文件 stat，动态目录输入保守重算。缓存保存 / 取用均深拷贝真实输出以防下游修改污染；失败输出不缓存，已成功祖先可继续复用。进程锁串行保护，LRU 最多 8 会话 × 64 节点，15 分钟空闲过期在下一次请求清理；执行结束（包括失败）后刷新空闲时间，不能用加载模型前的时间判刚生成的结果过期。
- 普通编辑保留 `previewSession`，交由服务端逐节点校验，不得因为修改保存目录 / 名称而全局清空模型缓存。前端 `nodeResults` 保存当前画布各节点最近一次成功输出；`nodeResultStale` 只标记过期可能，不删除预览；`previewHidden` 管理手动隐藏 / 显示。卡片预览独立于参数折叠和 loop 展示。`resetPreviewState()` 仅用于新建 / 导入流程；删除节点只清该节点显示，batch / 试跑重置执行会话但保留历史预览。修订号防止编辑期间异步旧响应覆盖当前状态；纯节点位移不失效。
- 端口命名是展示层协议：`rgb_image`、`masked_rgb_image`、`mask`、`rgba_image`、`reference_rgb_image`、`image_list` 等，配中文用途；`kind` 用于前端推荐高亮，不是图运行时的新约束。特别是换色输入原 `mask` 实为 `masked_rgb_image`，输出原 `source` 实为未涂灰的 `rgb_image`。画布 / widgets / inspector / 预览 / 连线提示使用同一 metadata；DOM `data-port`、`data-field`、图 JSON 和节点 execute 原字段保持稳定，未知端口有回退。输入定位按 `data-port`，不能再用可重复的显示标题查找。
- 保存节点的目录与数据集导出目录是不同配置；“只保存当前张”与“逐张循环”也是不同操作。
- [SAM3 节点](../object_inpainting/nodes/sam3_text_mask.py) 只分割，保留两项默认 0.5 的 float 阈值；原膨胀参数 / 逻辑已删除。[MaskDilationNode](../object_inpainting/nodes/mask_dilation.py) 用现有方形 MaxFilter 独立 CPU 膨胀，int 参数 steps / kernel 默认 1 / 3，输出同尺寸 L Mask、最终非零 bbox 和面积说明，支持二值及灰度，不改原图；空 Mask 的框为全 0。原数据集 `ExpandMaskNode` 及完整示例保持兼容。详见 [分支设置](../object_inpainting_studio/USAGE.zh-CN.md#sam3-mask-太小时的设置)。
- [工作流迁移](../object_inpainting/workflow_migrations.py) 深拷贝旧图，将 SAM3 的旧膨胀字段移到唯一 ID 的独立节点，并重接全部原 object_mask / bbox 消费与图输出；旧默认 steps=0 被保留，迁移幂等、不改调用者。图构造时统一应用；`/api/workflow/validate` 返回归一化 `workflow`，前端导入采用它，避免隐藏新增执行节点。原 SAM3 quality 仍来自原始模型，独立节点另报膨胀面积。
- `SaveImagesNode` 默认四个保存项，可增删到八项或更多；每项元数据 `save_items: [{id, name}]` 与单调递增 `save_next_id` 随节点保存，不是可连线输入。显式列表是端口集合真相，删除项不被旧四口默认值复活；旧图深拷贝补齐元数据，保留目录与随机编号。`name` 决定子目录，连线和 `paths` 键仍是稳定 `imgN`；前端 `nodeInfo` 为卡片 / inspector / 连线提供实例类型与显示名，增删改名清理结果和 loop 计划，运行期间禁止修改。
- 保存节点 `filename: STRING` 默认空，同轮共享 UUID；可连接文件夹原 `filename`，去掉最后一个扩展名后统一 PNG。全部图片先验证编码，再检查名称重复、路径、符号链接子目录和文件冲突；`xb` 独占写入，不覆盖，OSError 回滚本轮文件（空目录可能残留）。固定名重跑 / 输入同主干会报冲突，不自动覆盖或跳过。
- `SaveImagesNode.images` 仍按列表位置保存到 `images_0`、`images_1` 等自动命名子目录，可与动态项同时保存，共享组名。图片列表复用服务端递归序列化和前端 object 预览；`INT_LIST` 字面量由前端按 JSON 解析。色块取现有区域平均 RGB，顺序保持输入列表；索引越界报错，不能将当前图排名当作跨图颜色身份。
- 批处理计划检查通过不代表每张图推理成功；不要把单张模型推理步数误认为图片数量。
- 错误可观测性：节点失败和 HTTP 错误写入终端日志；API 保留字符串 `detail` 并为节点错误添加 `node_error`；批处理失败项 / 停止任务报告也有同结构字段。单节点 / 当前张试跑在画布与右侧预览显示真正失败的节点，批量运行通过每张结果和下载报告查看。
- 节点删除只编辑当前画布，不删除磁盘文件；断开下游端口并移除直接关联图输出。输入恢复默认值或 null，清理布局 / 结果 / loop 计划；预览或批量任务未结束 / 状态未知时禁止删除。编辑控件内的 Delete 不触发节点删除。

## 5. 运行与验证导航

- 在本目录启动 Studio：`bash studio_object_inpainting.sh`。脚本默认用 `../.venv/bin/python`，可通过 `OBJECT_INPAINTING_STUDIO_PYTHON` 覆盖，不要求提前激活虚拟环境；设置 `PYTHONPATH`。HF 缓存默认放在 `../../hf_cache`，允许环境变量覆盖 `HF_HOME`、`HF_HUB_CACHE`（优先于旧名 `HUGGINGFACE_HUB_CACHE`）和 `HF_XET_CACHE`，两个 Hub 变量统一指向同一目录。默认 host/port 见模块入口，不猜测当前是否已有服务。
- CLI 用选定解释器运行入口并传 `--input-manifest` 和 `--output-dir`；其他参数见 [cli.py](../object_inpainting/cli.py)。
- [测试指南](../tests/README.md) 提供 unittest / pytest 命令。在本目录的文档命令为：`PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" ../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`。解释器路径需核实，不能假定环境已装好。
- 单类测试收窄 `-p`；浏览器测试依赖 Playwright Chromium，可通过 `STUDIO_TEST_URL` 指向服务。模型节点测试使用替代输出，不自动证明真实 GPU 推理或图像质量。
- Studio FLUX 默认优先加载项目相对 `../../hf_cache/FLUX.2-klein-base-9B` 普通模型目录，目录不存在时回退 Hub 模型 ID 的离线缓存查找；`OBJECT_INPAINTING_FLUX_MODEL_DIR` 可显式覆盖，缓存根为 `../../hf_cache`。
- Studio、SAM2、SAM3 的依赖分别见 [基础依赖](../object_inpainting_studio/requirements.txt)、[SAM2 依赖](../object_inpainting_studio/requirements-sam2.txt)、[SAM3 依赖](../object_inpainting_studio/requirements-sam3.txt)。模型目录、设备、缓存等先检查对应节点实现及启动脚本；不要默认下载大模型。

上述命令来自源码/现有文档核对，本次笔记整理没有启动服务、运行测试或验证本机模型环境。
