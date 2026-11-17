# 迭代交接笔记

只保留恢复工作需要的信息；稳定架构放 [架构地图](PROJECT_CONTEXT.md)，入口与维护约定见 [AGENTS.md](../AGENTS.md)。

## 当前交接

- 2026-11-17 / 用户授权删除precision_recipe/版本分支/custom norm，并要求VAE回到预训练原始状态：删除CLI/resume shim/apply_recipe/NativeLayerNorm/NativeGroupNorm及VAE标记。训练和独立加载都直接from_pretrained(torch_dtype=float32)加载VAE，内部模块/forward/参数不改；encode_latents仅调用处关闭autocast，pipeline验证无外层AMP，原生decode转换latent到vae.dtype。保留latent BN与FP32 loss，噪声/velocity跟随latent；删重复target.float条件。
- precision.py精简为set_transformer_autocast与pipeline_forward（临时旁路Accelerate输出FP32转换，保留Transformer自身AMP并还原），训练/验证/推理统一计算边界，不触碰模型原生FP32保留模块。LoRA/text encoder主参数策略未改。比较工具移除recipe选项和元数据，验证工具/测试重命名verify_flux3_precision.py、test_flux3_precision.py；文档明确历史旧norm GPU数值不代表当前方案验收。旧recipe状态需新目录训练，无兼容分支。
- 验证：专项先RED后GREEN；18模块CPU106项（104通过、2显式opt-in跳过），含原始VAE state逐元素/模块类型/forward身份、外层AMP编码隔离、BF16 CPU pipeline VAE编码和解码FP32、原生LayerNorm和非零FP32 LoRA梯度、训练恢复/验证/推理；两rank CPU DDP普通caption/独立text encoder同步保存恢复2项通过。独立只读review无重要项，语法/--help/git diff --check通过。未运行真实GPU精度/质量或性能验收，未干预已有GPU任务；修改前源码已临时备份。任务完成；下一步未指定。


- 2026-11-17 / 用户要求参考Transformers logging_steps并用tqdm显示进度：新增--logging_steps正整数默认10、resume可调整；training_logging.TrainingLogger管理主进程tqdm（按成功optimizer step，恢复initial=checkpoint.global_step）及GPU区间loss分子/Python样本数。只在日志点和最后一步reduce两个统计量、读取标量、写tracker/history/进度postfix；loss按各rank实际样本数加权。训练不再每优化步归约loss日志；finite检查逐microbatch保留，FP16跳步丢弃本组统计、不推进进度。finally关闭进度条，checkpoint/验证频率独立；日志窗口不持久化，恢复首段只计本次继续训练样本。
- 验证：新日志unit和集成先RED后GREEN；5模块CPU41项（39通过、2显式opt-in跳过），含不等样本权重、非日志点无归约、尾段、logging1/2最终LoRA逐位一致、区间均值、恢复调整频率和进度起点；两rank CPU DDP普通caption/独立text_encoder保存同步恢复2项通过。首次集成测试mock全局tqdm影响HF权重加载，已改为仅mock本模块tqdm并完整重跑通过；独立review无重要问题，--help/git diff --check通过。文档/架构/原优化计划已更新，未用GPU或测实际吞吐；任务完成，下一步未指定。


- 2026-11-17 / 用户误删除后git恢复到较旧暂存版，授权补齐所有缺失修改：先备份现场，按保留测试/交接/会话恢复dataset_cache、path Dataset/buckets、data/modeling helper、validation、独立text_encoder与新入口、component_io和统一inference；重新移除旧task代码/入口/测试/文档、check_dataset、latent缓存、独立LR和方向loss。保留现有Git暂存状态，不操作其他已暂存修改。
- 复查补齐三项：缓存命中比对manifest并load_from_disk验证；缓存v2严格恢复format_version/config/subdatasets/files-list及原排序/SHA256序列化；read_path_sample默认center_crop=True（训练显式传参）。真实磁盘旧2384bd…缓存的key和manifest完全匹配并成功命中，禁止Image.open时真实数据3196train/356val、1024x768单桶、batch2=1598批，与原结果一致。推理CLI别名/split_json、manifest来源字段及诊断direction参数也已恢复。
- 最终验证：17模块CPU104项（102通过、2显式opt-in跳过），含原缓存schema/命中可读性/默认中心crop的3项新回归；两rank CPU DDP在线caption与独立text_encoder同步/恢复2项通过，缓存规范修复后独立encoder额外复验1项通过。首轮诊断接口遗漏已修；独立review及修复复查无剩余重要项，语法/新入口--help/git diff --check通过。未用GPU/真实模型。恢复源码快照另存flux3-restored-source-1794887254831469431.tar.gz（临时目录）；尚未提交Git。恢复任务完成，下一步未指定。


- 2026-11-17 / 用户要求自定义模型与训练参数统一text_encoder命名，并精简验证结构：模块/类改flux3_recolor.text_encoder.TextEncoderConfig/Model，开关--train_text_encoder，入口train_flux3_recolor_text_encoder.py，测试test_flux3_text_encoder.py，文档FLUX3_TEXT_ENCODER.zh-CN.md；components/result/函数参数/config字段统一text_encoder，manifest type=learned、model_type=recolor_text_encoder，无旧别名。optimizer使用完整encoder.parameters且共享learning_rate。
- 验证：build_pipeline附真实自定义recolor_text_encoder引用，训练make_validation_pipeline传unwrap后的共享模块；validation.pipeline_text_embeddings调用当前forward并no_grad+detach正向、保持冻结负向与dtype/device，渲染函数删除模型类型分支；tracker图片记录提取log_validation_images，固定prompt验证复用已有编码。统一推理loader也保留标准from_pretrained模型供pipeline使用，默认generate_sample读取相同helper。pipeline缓存/precision/RNG/offload/去重/临时目录保护保留。
- 验证结果：更名和live encoder新测试RED→GREEN，8模块CPU56项（54通过、2显式opt-in跳过），两rank CPU DDP自定义encoder/在线caption同步恢复2项通过，rank间和连续vs恢复参数逐位一致。首轮测试发现no_grad expand视图仍requires_grad，已显式detach；旧SimpleNamespace字段访问改为依据components处理，完整回归重跑通过。独立review无重要问题，新入口--help/git diff --check通过；未用GPU或真实大模型。任务完成，下一步未指定。


- 2026-11-17 / 用户要求text encoder config添加length控制token数：TaskEmbeddingConfig以正整数length替代sequence_length，参数形状[1,length,hidden_size]及forward输出遵循该值，from_features默认取prompt实际窗口；save_pretrained/from_pretrained持久化恢复length，无新增CLI。新增非默认length=7、保存重载和非法值测试RED→GREEN；相关7项CPU回归通过，覆盖独立梯度/标准格式/训练恢复/验证推理/组件保存，git diff --check通过。文档/架构已更新；未用GPU，任务完成，下一步未指定。


- 2026-11-17 / 用户批准分组件保存与自动基础权重发现：新增component_io.save_components供checkpoint hook/最终export共用，transformer/仅标准Diffusers LoRA及adapter metadata（参考img2img脚本），可训练TaskEmbeddingModel完整save_pretrained至text_encoder/，冻结vae/Qwen不导出；根model_config.json记录base_model及组件来源。task helper统一text_encoder子目录；weights_layout=components_v1拒绝旧目录恢复。hook保存异常通过main_process_action广播；FSDP内部恢复仍为原生Accelerate状态，组件导出只有增量。
- 加载：新增inference.load_recolor_pipeline，读取manifest基础模型/可覆盖路径并核对签名，加载transformer/LoRA及text_encoder标准模型，CFG负向仍由基础Qwen编码；返回原生pipeline附recolor_config/recolor_embeddings，generate_sample可默认读取文本条件。独立推理CLI共用加载入口；非空输出保护保留。文档/架构/测试路径同步，无旧目录兼容。
- 验证：新component_io专项RED→GREEN，6模块CPU回归44项（42通过、2显式opt-in跳过），包含manifest实际发现基础模型、无冻结组件目录、标准from_pretrained、训练恢复与独立推理；两rank CPU DDP普通caption/独立task双模型同步恢复2项通过，rank间及连续vs恢复权重逐位一致。首轮一处测试遗留旧task_encoder目录，已修并完整重跑；独立review无重要问题，git diff --check通过。未用GPU/真实模型；任务完成，下一步未指定。


- 2026-11-17 / 用户要求删除task_embedding_learning_rate：删除CLI、合法性/Prodigy独立LR限制及resume默认字段，optimizer两个参数组统一使用learning_rate和scale_lr，task encoder保留weight_decay=0。测试命令/实验说明/架构同步，移除flag由原拒绝参数用例覆盖。20项CPU回归通过（标准格式/梯度/训练/恢复/验证/推理/入口），另直接检查scale_lr开关两组LR一致且encoder无weight decay；git diff --check通过，未用GPU。任务完成；下一步未指定。


- 2026-11-17 / 用户明确不要旧兼容、会重新训练：删除task_embedding.py的旧单tensor文件读取分支与FILENAME常量，仅支持task_encoder/config.json+model.safetensors。实验说明/架构已纠正，不再声称支持旧文件推理。现有5项CPU测试通过（标准格式往返/跨Transformer梯度/训练更新/恢复/验证/推理），git diff --check通过；未用GPU。本次仅清理task encoder兼容，不扩大到无关历史恢复规则。任务完成；下一步未指定。


- 2026-11-17 / 用户确认两个独立模型方案并授权实施：task_embedding.py改为Transformers TaskEmbeddingConfig/TaskEmbeddingModel，FP32[1,L,H]从prompt初始化，forward返回BaseModelOutput；删除Transformer子模块与forward monkeypatch。主训练将Transformer/task encoder分别prepare、共同accumulate/clip，encoder输出保留梯度传入Transformer；optimizer分组独立LR保持。固定prompt/在线caption默认路径保持。Qwen只负责初始化和固定负向后释放；验证读取独立encoder当前正向。
- 保存/恢复：task_encoder/config.json+model.safetensors通过save_pretrained/from_pretrained，checkpoint hooks分别保存LoRA和独立encoder并原位加载，保留optimizer引用；main返回task_encoder供验收。task_encoder_format标记拒绝旧内嵌训练checkpoint，新实验用新目录；旧task_embedding.safetensors仍可独立推理。FSDP+task继续明确拒绝，无新增参数。
- 验证：3项新结构专项RED→GREEN（跨Transformer梯度/参数独立/未改forward/标准格式/identity/shape），4模块CPU回归36项（34通过、2显式opt-in跳过）；另两rank独立双模型CPU DDP同步/累积/验证/恢复1项通过，encoder和LoRA均跨rank及连续vs恢复逐位一致。独立只读review无重要问题，git diff --check通过。文档/实验说明/架构更新；当前版本未用GPU或真实大模型，不代表质量/速度改善。任务完成；下一步未指定。


- 2026-11-17 / 用户要求恢复在线text encode，并提议task embedding替换整个text encoder：已撤回text_cache.py及对应测试，删除caption_encoding新标记；自定义caption保留encoder/tokenizer并按当前batch在线编码，dropout空文本动态窗口恢复；维度检查使用accelerator.unwrap_model(transformer).config。fixed/variants/task暂保持原实现。文档已改回在线语义；上一迭代预编码说明由本条取代。
- 当前验证：4模块40项CPU（38通过、2显式opt-in跳过），独立review无重要问题，git diff --check通过；两rank CPU DDP+caption/dropout同步与恢复1项通过，参数一致。task替换尚未实施：已异步询问用户是用可训练[1,L,H]模块替代Qwen接口，还是训练整个Qwen；等待必要语义确认，未作依赖修改。未用GPU；任务未完成，下一步确认并实施task encoder设计。


- 2026-11-17 / 用户要求模仿train_dreambooth_lora_flux3_klein.py文本处理：新增text_cache.cache_caption_embeddings，prepare前按实际训练record的fallback prompt去重，分批编码至CPU；batch>1以全集最大长度向上取80倍数/限制max_sequence_length，batch1保留独立窗口；dropout空文本按窗口提前编码。训练只查询并搬运缓存，所有路径释放text_encoder/tokenizer，维度校验使用包装前整数。固定prompt/变体/task embedding保留，无新增参数。caption_encoding标记阻止旧在线caption checkpoint混用（统一窗口改变语义）；新caption训练需新目录，固定prompt配置不变。CPU缓存内存随唯一caption数增长，各rank一份。
- 验证：2项缓存专项RED→GREEN（共享窗口/去重/CPU/no-grad/batch1）；5模块CPU回归42项（40通过、2显式opt-in跳过）；另两rank CPU DDP+自定义caption+dropout同步/恢复/验证1项通过，rank间和连续vs恢复最终参数逐位一致。独立只读review无重要问题，git diff --check通过。未用GPU/真实模型，未验证真实吞吐和质量；本任务完成，下一步未指定。


- 2026-11-16 / 用户批准精简模块化验证去重与恢复处理：新增validation.py，ValidationRuns仅本进程记录成功渲染的step/config/选中sample_id/seed/推理步数；log_validation统一选样和去重后执行原渲染/FSDP收集/广播。相同step/epoch/final只保留首次触发目录与tracker，不同final prompt仍出图。恢复进程不复用去重状态；同级临时目录完整生成后替换该步旧目录，生成失败保留旧图，发布不是原子替换。独立推理仍拒绝非空目录，无新增CLI参数。
- 验证：4项专项先RED后GREEN（去重/配置变化/新进程/失败重试/目录替换/输入保护）；相关5模块32项（30通过、2显式opt-in跳过），两rank CPU DDP同步/恢复/验证1项通过。真实随机小模型覆盖恢复时旧step目录、相同final省略、不同final保留及task embedding/离线推理；一处测试新增final prompt后恢复漏传同配置，已修fixture并完整重跑通过。独立只读review无重要问题，git diff --check通过。未用GPU或真实模型；任务完成，下一步未指定。


- 2026-11-16 / 用户批准P0-2 VAE批量编码：新增modeling.encode_recolor_latents，将目标/涂灰/源图沿batch维拼接后调用原encode_latents并按B拆回，色块单独编码；训练main每microbatch由4次VAE调用降为2次。FP32 VAE、posterior mode、BN/patchify、条件顺序、offload和flow matching不变，无新增参数或latent缓存。合并后场景瞬时batch为3B，真实GPU速度/峰值显存未测量。
- 验证：专项4项先RED（缺helper）后GREEN，覆盖B=1/2/3调用次数与顺序、非平凡BN、AMP/RNG保持、相同噪声/时间步loss及非零LoRA梯度等价；15模块CPU回归93项（91通过、2显式opt-in跳过）。独立只读review未发现重要问题；另两rank CPU DDP同步/恢复/验证验收1项通过（连续vs恢复及rank间参数逐位一致）。文档/架构/原方案已更新，git diff --check通过。未使用GPU或真实权重；本任务完成，下一步未指定。



- 2026-11-16 / 用户改为删除latent缓存以简化代码：删除cache_latents CLI、预计算函数/cached_latents_for_batch、CPU latent字典及训练读取分支；每个microbatch统一在线encode目标与三参考，VAE精度/flow matching不变。删除缓存专用的随机几何互斥及验证后offload条件，offload开关本身保留；Dataset hash缓存和文本embedding预编码保留。旧saved config的cache_latents字段只在恢复校验中pop，不启用旧功能。
- 缓存专用测试删除，训练/DDP/任务embedding用例改为在线模式，FP32 VAE/AMP/官方pipeline等价测试保留；新增参数缺失与旧配置字段忽略契约RED→GREEN。30项相关CPU单测通过；最终14模块90项（88通过、2显式opt-in跳过），另两rank在线编码DDP参数同步/恢复1项通过。独立只读review确认无残留执行引用、device/dtype/offload与恢复逻辑无重要问题。文档与原方案已标记latent去重取消；未用GPU或真实模型。任务完成；下一项未指定。


- 2026-11-16 / 用户批准P0-1第一阶段实际消费缓存：main删除扫描/配对/拆分，直接加载hash缓存train/validation；DreamBoothDataset继承torch Dataset，初始化纯元数据计算buckets/bucket_idx，Sampler按预建bucket_indices组batch并保留DDP尾batch补齐。__getitem__才按record路径load_rgb/EXIF转正，目标/灰图/源图共享几何，异分辨率引用先对齐目标画布，色块独立resize。删除样本direction、方向mask/归约/分方向loss。repeat、随机prompt及空caption回退保留。
- 验证/推理统一消费validation记录，num_validation_images为总图片数，按固定等距位置覆盖整个合并序列；数字目录+来源manifest，无方向扩展。独立验证推理读取config.dataset_path，支持dataset_root重定位；新config记录dataset_format/condition_order/dataset_root，split为sample_id列表。旧adapter仍可显式三图推理，新旧训练checkpoint不直接混用。Cache格式v2按各子集有效文件取stat，不再取跨子集交集；groups映射须覆盖实际参与记录。
- 实际验证：新增接入测试先RED后GREEN；最终14模块CPU共91项（89通过、2项显式opt-in跳过），另真实两rank CPU DDP同步/恢复1项通过；含缓存连续vs恢复adapter/RNG/LR一致、task embedding更新/验证、独立推理、数学/精度对照。首次worker测试被沙箱IPC拒绝，停止该次后沙箱外完整重跑；独立推理旧配置校验曾拒绝新schema，已修并回归；review发现验证前N条偏首子集，已改等距选择并加测试，复查无剩余重要项。实际test的3196 train/356 validation在禁止Image.open时完成Dataset/Sampler初始化，1024x768单桶、batch2为1598批；读样本场景3x1024x768、色块3x128x128且无direction。
- 范围：第一阶段完成；latent按文件去重和VAE批量编码未做，encode次数/precision recipe/flow matching数学未改。未用GPU或真实模型训练。文档/计划进度已更新，训练/推理--help及git diff --check通过，独立复查确认修复后无剩余重要项；下一项待用户指定。


- 2026-11-16 / 用户要求独立Dataset hash缓存，随后要求回到Claude/DeepSeek优化MD：新增dataset_cache.py，SHA256包含有效构建配置/目录映射/选中文件名/源文件size+mtime_ns及构建版本，排除模型/LR/rank/分辨率/worker/output_dir；路径为输入目录上一级datasets/recolor_cache/hash。每key文件锁、临时构建、cache_config.json写完后原子发布；跨run和resume命中加载，不重复读图片头构建。train helper单独组装config，report记录缓存路径；消费Dataset/bucket尚未切换。
- 13项CPU测试通过（key/文件修改/跨run/并发/失败重试/原构建），独立review无重要项；随机小模型连续vs恢复adapter/RNG/LR精确回归1项通过，git diff --check通过；hash缓存任务完成。已重新读取2026-11-16-training-optimization.md：P0-1部分完成，元数据消费/latent文件去重未做；P0-2/P0-3未做；P0-4的闭包wrapper.config仍静态存在。P1-7/P2-1后续须按用户“合并后不分方向”意图调整为移除方向统计。未实施MD的其他项；当前任务转回MD核对，建议先收尾P0-1再处理DDP caption/VAE批量。


- 2026-11-16 / 用户要求删除不必要的数据检查参数：删除train_flux3_recolor_lora.py的check_dataset CLI、其专用main分支及配置排除项；pretrained_model_name_or_path改argparse required，输出路径/恢复checkpoint初始化不再被外层if包裹。正常训练的自动构建调用仍在，Dataset/bucket消费切换仍未实施。相关unit fixture改为显式模型占位路径，旧数据检查CLI用例改为验证必填模型和拒绝已移除开关；文档更新。验证44项相关CPU回归及1项task embedding参数拒绝测试通过（共45项）；--help无已删除参数、模型路径显示必填，git diff --check通过。未用GPU或真实模型。任务完成，下一项未指定。


- 2026-11-16 / 用户要求训练内直接import调用Dataset构建：新增train入口prepare_path_dataset，在output保护/恢复配置检查后、load_training_components前，通过main_process_action仅主进程import build_dataset生成output_dir/dataset；已有保存目录load_from_disk验证可读后复用。check_dataset指定output时也构建。无新CLI参数；沿用当前directions、seed、validation_fraction、workers及prompt/caption_json/groups_json。builder新增可选prompt/captions/source_groups，确保产物元数据与训练选项一致。
- 范围：仅接自动构建调用，已向用户明确当前Dataset/sampler/loss/验证消费逻辑未重写，生成artifact仍待下一步使用。验证8项构建/复用CPU测试通过；另2项随机小模型缓存连续vs恢复adapter/RNG/LR精确一致及TensorBoard实时验证通过。第一次测试fixture漏check_dataset导致参数退出，已修并重跑；不是实现错误。未使用GPU/真实模型。独立review指出单方向额外样本不应扩大groups_json覆盖要求；已传入当前训练索引的sample_names先过滤后分组，新增复现测试RED→GREEN，复查无剩余项。--help/git diff --check通过，文档已更新。任务完成；下一步未指定。


- 2026-11-16 / 用户批准完成path-only Dataset并说明后续：build_flux3_recolor_dataset.py不再生成uint8像素缓存；schema为sample_id/source_id/target_path/condition_paths/height/width/condition_sizes/prompt。Value(string)绝对路径不嵌入图片bytes；目标/三条件各读自己的EXIF显示尺寸，PNG通过seek跳过压缩chunk并支持晚eXIf，不解码像素。保留在concat阶段控制比例，再按source_id分组拆分；无direction列。新默认datasets/recolor_paths，旧24G的datasets/recolor_uint8未删除。
- 已验证：5项CPU测试通过（禁止像素decode的构建/路径列/独立WH、晚PNG EXIF、比例/来源隔离、坏头过滤、NFS缓存清理）；独立只读review无重要问题。实际CPU4构建命令退出0，3196 train/356 validation、源组1598/178无交叉，3552唯一ID；8880独立路径全部存在，抽样24张图与实际EXIF转正尺寸一致，最终4,439,947 bytes（约4.4MB）。无Image特性、无图片binary。训练入口未改、无GPU/模型调用。原图需保持原路径；仅检查头部，不证明像素payload无损坏。
- 后续未实施：训练load_from_disk；按元数据建立参考式buckets/bucket_idx与BucketBatchSampler；Dataset取样读路径并同步几何/色块resize/归一化，统一loss不再分方向；latent cache/实时验证接入统一记录并回归检查。最终5项CPU测试再次通过，--help/git diff --check通过。当前数据集构建任务已完成；下一步等待用户指定。


- 2026-11-16 / 用户批准先构建HF Dataset，训练入口不动：新增build_flux3_recolor_dataset.py，两个方向各自读取目标/灰图/源图/色块RGB uint8及独立尺寸，按mix_ratio在concatenate阶段重复补足比例，合并后按完整文件名主干source_id分组拆分。最终schema无direction，含sample_id/source_id/target/conditions/prompt/height/width。原始连续HWC uint8以binary缓冲区持久化，unpack_image用np.frombuffer无图片解码；每图尺寸在其struct内，色块不resize。临时子集cache构建后清理，最终save_to_disk，输出目录拒绝覆盖。
- 验证：4项CPU测试通过（raw字节/uint8/独立尺寸/禁止解码读取、2:1拼接/源组隔离、缺失坏图分别过滤/拒绝覆盖、共享文件系统缓存清理），独立review确认无重要问题。曾用Array2D嵌套像素导致实测约1条/秒，已停止自己的构建、改raw binary，测试RED→GREEN；实际4进程达到约50条/秒。沙箱内IPC被拒，沙箱外CPU4完成各1776条生成，最终3196 train/356 validation已落盘，约24G。首次实际构建的save_to_disk全部完成后，临时cache清理因NFS持有mmap报错；已定位并在退出缓存scope前del所有Dataset引用/gc.collect，NFS测试RED→GREEN，4项最终通过。最终数据独立加载验证3552个唯一sample_id、源组1598/178且交集0、24张抽样图片与原始RGB逐像素一致；补齐build_summary并清理残留，无需重建。保存datasets/recolor_uint8并由.gitignore排除，未用GPU/模型、未改训练实现。--help/git diff --check通过。构建任务完成；训练接入待下一项任务。


- 2026-11-16 / 用户要求撤掉上一版过度设计，改普通Dataset：已删除fast/full/retry新增CLI、indexing.py独立替换器、指纹/旧checkpoint迁移、尺寸报告传递及验证替换功能。启动只取完整文件名交集；Dataset只读目标图片头为现有batch sampler算shape，取样才RGB解码。异常仅SampleDataError，打印失败文件，固定额外8次，同Dataset/方向/尺寸/repeat候选并返回实际prompt/缓存ID；正常样本无候选搜索。inspect_dataset仍供独立诊断使用，不进入训练/check_dataset链路；check_dataset现在只看配对/拆分。旧checkpoint不迁移，训练使用新输出目录。
- 验证：新增4项CPU测试通过；首轮相关8模块65项完成（63通过、2跳过），涵盖随机小模型训练/缓存/恢复/实时验证/推理。独立review发现小尺寸几何应纳入跳过、repeats重复浪费重试次数，已修并补测试；最终相关8模块66项完成（64通过、2跳过）；新增4项再次通过，--help确认新增参数已移除，git diff --check通过。复查确认修复无剩余项。未使用GPU/真实模型；无新增参数。任务完成，下一项未指定。

- 2026-11-16 / 用户要求复查 train_flux3_recolor_lora.py 可优化点：只读审查+独立复查，未改代码/未用GPU。已核实：启动重复解码（当前test 1776组双向、开cache时主进程约4.9万次；本机抽样约19ms/张，不含resize/VAE）、训练每microstep 4次VAE encode可合并、验证仅主进程且末step与final重复、FSDP验证每次重载完整transformer；缺color_swatch_size的旧version-2 checkpoint无法resume（真实validate_training_resume复现；现outputs v1/v3/v4/v5均带128字段，影响更老checkpoint，文档已说明新128不能resume旧32）；Namespace入口line911直接访问args.precision_recipe且task_embedding+caption_json会传None；默认batch1/accum1时loss_b_to_a/loss_a_to_b每步只出现一个方向；prompt_augmentation被显式prompt/caption忽略但仍持久化true。补充：checkpoint save hook内main-only的save_adapter/save_task_embedding未走main_process_action，主进程单侧写失败时其他rank会在wait_for_everyone挂起至超时（新增task文件写入加重该风险）。补充2：FSDP验证代理以bf16加载而训练transformer保持FP32，破坏precision训练/验证一致性；set_peft_model_state_dict返回值未检查，键不匹配会静默保留gaussian初始化LoRA；epoch边界checkpoint恢复可能跳过该epoch验证。补充3：`--task_embedding_learning_rate` 未启用train_task_embedding时静默无效但仍持久化，恢复不带该flag会配置不一致。补充4：DDP/FSDP+caption_json在线文本编码在accelerator.prepare后访问wrapper.config会崩；恢复重放未checkpoint步骤时已存在的validation-step-*会被prepare_output拒绝覆盖；caption_json覆盖检查错误地要求validation names。问题与实施顺序已整理到[训练入口优化方案](superpowers/plans/2026-11-16-training-optimization.md)。
- 验证：py_compile相关模块通过；真实validate_training_resume对缺swatch字段复现“恢复配置不一致: color_swatch_size”，缺precision/task字段在legacy当前配置下PASS；独立复查基线21项+3集成+7 task embedding测试通过、legacy resume未覆盖；未跑真实训练/GPU。下一步：用户决定是否实施重复解码/VAE批量/验证去重/旧checkpoint迁移；未指定。

- 2026-11-15 / 用户要求确认可学习embedding实际更新：只读核对当前实现的Parameter注册、训练forward直接使用未detach特征、优化器独立group、backward/step及用户实际outputs/flux3_task_embedding。配置train_task_embedding=true、LoRA LR1e-5、task LR0.001、BF16、accum1。比较完成checkpoint100→200，embedding为FP32[1,240,20480]，4,915,163/4,915,200元素变化（99.99925%），平均绝对差0.00553134、最大0.04365665；转换BF16后仍77.6198%元素不同。checkpoint200的optimizer最后group仅该参数，LR0.001/weight_decay0，step200，exp_avg norm0.00049822且shape匹配，确认实际梯度和更新，不只是代码推断。
- 诊断JSON输出outputs/flux3_task_embedding/task_embedding_update_check.json；未改训练代码或停止/干预运行，未用GPU。该证据不等于色块换色质量改善。git diff --check通过；下一项用户任务未指定。


- 2026-11-15 / 用户要求新增可学习text embedding训练py用于比较（已完成）：新增train_flux3_recolor_task_embedding.py，自动启用train_task_embedding，复用原训练循环；原train_flux3_recolor_lora.py默认固定文本不变。新增flux3_recolor/task_embedding.py，共享FP32[1,L,H]由固定prompt编码初始化，在模型train forward内部扩展到batch，保证DDP reducer追踪；原Transformer/LLM/VAE冻结，联合更新LoRA及feature，可独立task_embedding_learning_rate（默认同LoRA、embedding无weight decay）。任务特征注册在追加子模块，避免顶层FP32 Parameter改变Diffusers model.dtype（RED复现后修复）。不实现色块编码器或Qwen训练。
- 验证/保存：eval不强制替换输入；实时验证只把正条件换成当前detach feature，空文本负条件保留。checkpoint和最终导出增加task_embedding.safetensors，恢复原位copy保留optimizer参数引用；flux3_recolor/inference.py自动识别config并读取保存特征，缺文件提前报错。原固定checkpoint缺新配置字段按false/None兼容；新旧模式不能互相恢复。为保持实验含义拒绝captions/augmentation/dropout/验证prompt覆盖；当前单卡/DDP支持、FSDP提前拒绝。Prodigy独立不同LR由review发现不支持，已增加加载模型前的参数拒绝及factory保护，测试通过。
- 实际验证：初始功能缺模块RED→2项GREEN；dtype回归RED→追加子模块修复；首轮26项CPU测试通过。最终OMP/MKL=1、CUDA_VISIBLE_DEVICES空、RUN_FLUX3_DDP_TESTS=1、PYTHONPATH=.:tests的相关10模块共73项通过（task_embedding及原数据/训练/推理/reference/local_model/math/bidirectional/precision），含真实随机小模型学习更新、正负条件分离、验证与checkpoint特征一致、独立推理加载、连续vs恢复LoRA及feature逐位一致；两rank DDP参数同步/恢复一致。另GPU1 RUN_FLUX3_CUDA_TESTS=1的随机小模型BF16两步训练/实时验证1项通过，原模型BF16、featureFP32且更新。--help及git diff --check通过。未加载真实模型训练或证明换色改善；用户现有训练未触碰。
- 文档FLUX3_TASK_EMBEDDING.zh-CN.md提供A/B命令与独立推理；使用说明/架构地图/tests指南/实施计划已更新。一次独立review除已修Prodigy问题无其他重要项。任务完成；下一项用户任务未指定。


- 2026-11-14 / 用户do it授权新Qwen prompt实际复跑：直接GPU1执行demo_qwen_recolor_pipeline.py，新英文Picture1/2/3 prompt、原三图/seed43/40步/true_cfg4/guidance1/negative空格/组件offload不变，未泄漏目标颜色词/RGB；开始GPU1另有约35GB占用，未触碰其他任务。40步约6分43秒完成，进程退出。新图仍深蓝衣服、构图漂移，目标近白色[243,243,247]，确认该prompt修改未解决当前样本问题，不能声明成功。
- 产物outputs/qwen_swatch_only_demo_prompt_v2_00019_43/{generated.png,previous_root_result.png,condition_1/2/3.png,run_config.json,README.md,demo_used.py}，根qwen_recolor_result.png为新图；保存旧结果和完整输入/参数。实际模型运行+输出肉眼/ROI核对，git diff --check通过。未再自动调参/训练/改数据；下一项用户任务未指定。


- 2026-11-14 / 用户要求改Qwen demo为合适prompt：仅修改demo_qwen_recolor_pipeline.py的正向prompt，改为使用pipeline原生Picture1/2/3标签的英文短指令，优先图3取色→图1衣服底色，图2仅用差异定位；强调原纹理/印花/人物/背景/原始取景与尺寸，不缩放裁切或拼参考图。保持三图、40步、seed43、CFG、offload等不变，不含目标颜色名/RGB/色号。ast语法及无颜色词/Picture编号静态断言、git diff --check通过；本轮未重新执行20分钟模型推理，新prompt效果待验证，不能声明已修复。上一轮失败图片保留，未动训练/数据；下一项任务未指定。


- 2026-11-14 / 用户do it授权Qwen真实运行：GPU1当时空闲约80GB，将demo_qwen_recolor_pipeline.py的use_sequential_cpu_offload改False，按组件offload保持其他参数。直接venv执行成功退出，40步完成，根qwen_recolor_result.png已保存；本地原始BF16、无LoRA、三图原图A/涂灰/主颜色B00019_00、色块128，中文prompt不含目标颜色名/RGB/色号，CPU seed43/true_cfg4/guidance1/negative空格/输出768×1024。
- 实际图像肉眼核对：近白色色块[243,243,247]却输出深蓝衣服，构图明显漂移，当前样本仅色块测试失败，不能宣称Qwen已具备满足任务的效果或所有配置都失败。产物outputs/qwen_swatch_only_demo_00019_43/{generated.png,condition_1/2/3.png,run_config.json,README.md,demo_used.py}；保留root结果。实际日志首次迭代约17分钟，后续4.6秒/步，总去噪20分钟，首次长等待未定位；guidance_scale1被非guidance-distilled模型忽略warning，true_cfg4仍按官方参数生效。进程退出，GPU1显存释放；没有重跑/降低分辨率/改采样或训练。下一项任务未指定。


- 2026-11-14 / 用户已下载Qwen-Image-Edit-2511并要求官方式仅色块demo：新增demo_qwen_recolor_pipeline.py，直接QwenImageEditPlusPipeline.from_pretrained本地BF16；图片原图A/涂灰图/主颜色B、00019_00、色块nearest128，中文prompt不含目标颜色名/RGB/色号。沿用官方40步、true_cfg_scale4、guidance_scale1、negative_prompt为空格、1张输出，CPU Generator43，指定原图宽高；保存qwen_recolor_result.png。默认use_sequential_cpu_offload=True，关闭时组件offload；不改FLUXdemo/训练、不加LoRA或mask外合成。
- 核对官方HF模型卡Quick Start与本地QwenImageEditPlusPipeline.__call__签名，模型目录model_index存在且匹配class。ast语法/无目标颜色词及所有调用参数匹配本地API静态检查通过；当前venv实际import QwenImageEditPlusPipeline通过；git diff --check通过。未加载该大模型或执行真实GPU生成，未证明仅色块换色效果。架构地图补独立入口；下一项任务未指定。


- 2026-11-14 / 用户要求两个image_cond modulation写入默认LoRA：flux3_recolor/modeling.py的LORA_TARGETS保留Attention四项+FFN两项，新增early_stream_modulations.image_cond.linear、single_stream_modulations.image_cond.linear；未加入x_embedders等其他层。未显式传--lora_layers的新训练生效，显式列表仍完整覆盖默认。更新使用说明：新增模块使用新输出目录；旧checkpoint继续需原目标代码/恢复配置，不能简单新增模块恢复。
- 验证：OMP/MKL=1、CUDA_VISIBLE_DEVICES为空、PYTHONPATH=.:tests，unittest test_flux3_recolor_training -q 6项通过，含前反向冻结、adapter保存/重载、推理metadata非默认alpha；另真实随机小Transformer flow loss backward检查两个新增模块LoRA B梯度非零（约5.46e-5和9.03e-4），原base权重仍冻结。CPU沙箱打印CUDA初始化warning但命令退出0，未使用GPU或启动用户训练。git diff --check通过。下一项任务未指定。


- 2026-11-14 / 用户授权demo统一工作流prompt/steps/CFG/seed并实际复测：demo_flux3_recolor_pipeline.py现按workflow的seed=-1、8步、guidance_scale4，两模型共用一次actual_seed与节点choose_recolor的颜色名/128 sampled_color，并直接复用recover_recolor_mask及相同pad/crop/composite。prompt逐字复制节点长指令，含color_name（本轮用户明确要求复现旧流程，例外于仅色块实验，不改训练默认）。修正当前demo中的cfg_scale为本地支持的guidance_scale；保留用户原图A/涂灰输入。输出新增run_config记录与mask/条件图、原始与合成图；旧根结果运行前备份。复跑固定seed输出目录已存在时拒绝覆盖。
- 实际GPU1直接执行demo成功，两模型均8步完成退出：actual_seed3568041805、sky blue、RGB[96,168,208]；FLUX2与FLUX3均生成天蓝衣服，已肉眼核对，合成结果mask外差0。产物outputs/flux_workflow_aligned_demo/3568041805/，含demo_used.py、run_config、raw/composited结果与旧根备份，根flux2/3_recolor_result.png为原始crop输出。AST验证prompt与节点源码相同、8/4/-1与JSON相同；语法/git diff --check通过。复现的是工作流换色流程，不是历史白衣逐像素复现（缺旧actual_seed/色块/环境）；包含颜色名，不能证明仅色块能力。未修改训练/数据，下一项任务未指定。


- 2026-11-14 / 用户问demo的FLUX2为何不复现workflow(11)new训练图：只读核对发现用户已改demo为原图A/涂灰/主颜色B，且删除两模型的steps/CFG显式参数，保留其编辑未覆盖。当前workflow随机换色为8步/CFG4/seed=-1，节点长prompt插color_name、随机palette sampled_color128；当前Flux2 demo固定seed43、短无颜色词prompt、未传steps故本地默认50/CFG4。workflow生成后crop/composite，demo直接保存，不能同条件复现。
- 关键颜色来源差别：workflow的colorregionswatches_2从生成source区域分色估计平均RGB输出32图，不是模型输入sampled_color；当前主颜色B[243,243,247]是结果侧估计，不能唯一恢复原颜色名/seed。当前JSON save.img4连平铺图节点而非randomrecolor.source，未直接保存生成图B，历史保存/改名过程未核实；00019_00 PNG无元数据、test未找到JSON种子记录。只记录当前源码，历史模型env override/节点版本未知，不能保证逐像素复现。详见docs/FLUX2_WORKFLOW_DEMO_COMPARE.zh-CN.md。未改生产代码/demo/数据，未用GPU；下一项用户任务未指定。


- 2026-11-14 / 用户要求demo加入FLUX2推理：demo_flux3_recolor_pipeline.py保留原生FLUX3直接调用，后续del/gc/清CUDA缓存再加载本地FLUX.2-klein-base-9B，使用官方Flux2KleinPipeline和model CPU offload（9B避免与Qwen常驻GPU）。两者共用同一原图/涂灰图/色块、同一无颜色词prompt，CPU Generator seed43、28步、CFG3.5；分别输出flux3_recolor_result.png和flux2_recolor_result.png，不做mask外合成或LoRA。仅改demo；未调用真实GPU推理，本轮验证为ast语法/两次共享输入与CFG静态断言、核对本地Flux2 API及模型目录、git diff --check，未证明本次Flux2实际效果。下一项任务未指定。


- 2026-11-14 / 用户要求直接demo交换条件顺序并再跑：已将demo_flux3_recolor_pipeline.py改为待改色原图（原来的生成图B）/涂灰图/主颜色A色块，prompt同步image1编辑/image2定位/image3颜色，无目标颜色词；保持00019_00、色块128、CPU Generator seed43、28步及其他加载方式。GPU1原始模型直接执行该文件成功退出，未动训练dataset/checkpoint或用户训练。结果仍白衣，未达到黑色色块要求；详见outputs/flux3_demo_source_first/{generated.png,report.json,README.md,demo_used.py}，根flux3_recolor_result.png亦为新图，旧根图片已备份。
- 验证：脚本ast解析、真实GPU 28步推理、输出图肉眼与mask中位RGB检查、git diff --check。旧根图参数未独立核实；此前诊断多用CUDA Generator，本demo用CPU Generator，同数字seed并非同噪声，不能把跨工具差异全归于图片顺序。只确认当前新顺序此次仍失败。下一项用户任务未指定。


- 2026-11-14 / 用户以FLUX.2随机换色节点产物质疑仅色块能力结论：只读核对object_inpainting/nodes/flux_random_recolor.py，当前版本第172行明确将color_name写入prompt（颜色词+色块双条件）；条件顺序为原图/涂灰图/色块，区别于训练入口涂灰/源图/色块，但各自prompt匹配，不能直接视为顺序bug。第197行Image.composite强制复制mask外原像素，不能把外部背景逐像素不变归因于模型能力。底层默认FLUX.2-klein-base-9B，可由OBJECT_INPAINTING_FLUX_MODEL_DIR覆盖。若训练数据由此版本产生，不能作为FLUX2“仅色块提供目标颜色”的证据；历史版本/当时generation_prompt未核实。未改节点/训练/数据，未启动GPU；下一项用户任务未指定。公平模型对照需相同参考图顺序、无颜色词prompt、采样设置并只换第三张色块。


- 2026-11-14 / 用户澄清要官方示例式直接脚本：新增demo_flux3_recolor_pipeline.py，顶层直接ModularPipeline.from_pretrained→load_components→to(cuda)→传三张图和prompt→保存flux3_recolor_result.png，无CLI/训练依赖/LoRA。本地HF cache解析snapshot；仅加载本地四组件并关闭未下载的可选prompt扩写，processor直接指向text_encoder目录；色块128放大，prompt不写目标颜色。保留之前CLI入口，不再作为本次用户所要的简短示例。验证仅ast语法解析/git diff --check，未运行本脚本真实GPU推理；先前组件加载测试不能替代该脚本的完整GPU验收。下一项任务未指定。


- 2026-11-14 / 独立pipeline测试文件（用户要求给单独py自行测试）：新增根test_flux3_recolor_pipeline.py，无训练模块依赖/LoRA/precision补丁，官方ModularPipeline.from_pretrained+load_components本地加载，FP32 VAE、其他按mixed_precision；关闭prompt扩写，默认指令不含目标颜色名/RGB/色号，目标颜色仅由色块提供。默认dataset_root=test、00019_00、b_to_a，也可三个显式图片路径；支持prompt/file、seed、steps、CFG、分辨率/色块尺寸。默认完整原生prompt编码，--offload_text_encoder可使用官方编码步骤预计算后卸载Qwen再加载Transformer。支持HF缓存root到snapshot与processor组件目录，输出PNG/三条件/run_config，拒绝非空目录覆盖。
- 验证：先新增tests/test_flux3_pipeline_standalone.py，缺入口模块RED；实现后OMP/MKL=1、CUDA_VISIBLE_DEVICES空、PYTHONPATH=.:tests的unittest test_flux3_pipeline_standalone -q 1项通过，包含本地随机真实小模型两条路径各实际生成一张图；--help通过，git diff --check通过。本轮未额外加载真实基础权重/GPU或运行用户训练；两路径小模型测试不证明真实模型质量。使用说明FLUX3_RECOLOR.zh-CN.md、tests/README已补命令。下一项用户任务未指定。


- 2026-11-14 / 用户约束纠正与仅色块复测：目标颜色只能由第三张色块提供，训练/验证/推理prompt不加入目标颜色名、RGB或色号；solid black/red文字成功不算本任务成功，后续保持该约束。现有dataset默认prompt本就不含目标颜色，未修改生产配置。
- 已完成原始零LoRA/无精度补丁模型、00019_00/seed43/28步/CFG3.5，仅128色块变化、其余条件与无颜色词的同一短prompt固定：黑色块[27,27,28]得到灰衣median[116,115,112]；白色块[243,243,247]仍灰衣[122,126,124]。图片肉眼核对，两组均不符合目标。此样本/配置下原始模型未正确执行仅色块换色；不能据此证明所有prompt/尺寸均失败，灰图颜色是否占主导未做因果控制。GPU1两次无梯度推理已完成退出，未动用户训练/基础权重。
- 产物outputs/flux3_swatch_only_check/{README.md,report.json,两组PNG,reproduce.py}；断言两prompt完全相同且不含常见颜色词，git diff --check通过，无生产代码变更/单元测试。已给前次显式颜色报告加不适用本任务的纠正。下一项用户任务未指定。


- 2026-11-14 / 原始pipeline能力复验（用户要求确认未训练基础模型）：直接原始Transformer BF16/VAE FP32，零LoRA参数、无生产precision/norm/forward补丁、无外层AMP，ModularPipeline.from_pretrained本地入口；官方文本步骤预计算后移除重复编码块，Qwen与Transformer分阶段加载。00019_00、seed43、28步、CFG3.5三组：原prompt+黑色块输出蓝色median[35,122,207]，与前次禁用LoRA对照逐像素MAE0；明确solid black的另一prompt+同黑色色块输出黑色[19,18,17]；明确red+红色块输出红色[212,42,34]。图片已肉眼核对，人物/背景/logo基本保留，未证明所有细节精确保留或全样本成功。
- 结论：基础模型具备基本换色能力，但当前原prompt在该样本的色块控制失败；不能只归因学习率。明确颜色prompt也缩短/改写原文，未隔离唯一颜色词因果；红色同时更换文字与色块，非冲突优先级实验。产物outputs/flux3_pristine_pipeline_check/{README.md,report.json,三组PNG,配置与reproduce.py}；三次GPU推理已完成退出。最初保留文本编码块时缺少prompt报错，移除已预计算的重复编码块后重跑完成，非生产训练故障。未改dataset prompt/生产代码/基础权重/用户训练，无代码测试；下一步用户任务未指定。


- 2026-11-14 / v3第700步颜色排查（完成只读诊断）：用户指出outputs/flux3_recolor_v3/validation-step-700/b_to_a偏色。四张图条件RGB/顺序/目标色块核对正确，确为fp32_vae_native_norm_v1、LR1e-6、双方向、batch/accum1、warmup100。700步仅见b_to_a357/a_to_b343批，相对3196双向训练记录约0.219轮。LoRA B非零且600/700/800范数增长（0.1905/0.2280/0.2487），排除完全未更新。
- GPU1无梯度实际checkpoint700、00019_00、seed43、28步六组对照：有无LoRA×黑白色块CFG3.5，另LoRA黑白CFG1。独立pipeline黑色CFG3.5逐像素复现保存蓝衣图（MAE/max0），未发现该样本共享权重/输入额外漂移。LoRA/base全图RGB MAE0.4248/255、涂灰区域0.6488/255；黑/白色块仍蓝，区域median[34,123,208]→[41,130,210]；CFG1回白色源衣，仍不服从黑色，单调CFG未解决。已确认颜色控制未学起来/该样本改变量小；低LR和覆盖不足为强嫌疑，未用LR对照证实唯一根因，不能声称全量数据语义均正确。
- 产物outputs/flux3_recolor_v3_color_diagnosis/README.md、六组图片/report/响应指标/四验证颜色统计/adapter范数；为避免用户checkpoint自动轮转保留700步adapter与配置，基础权重未复制。复现脚本在该产物目录，分阶段加载Qwen与Transformer、不启动训练。该GPU过程已退出，用户现有训练未触碰；生产脚本/训练参数未改，本轮无代码测试，实际验证为六次真实模型生成与图片/权重统计。下一步建议小样本过拟合与LR1e-5对照，未实施；下一项用户任务未指定。

- 本轮精度修复与共享验证（2026-11-13，已实施）：用户授权开始修改，并要求原生pipeline直接共享训练模型/最新LoRA。新增flux3_recolor/precision.py；新训练默认precision_recipe=fp32_vae_native_norm_v1：原始FP32 VAE、channels_last且编码AMP关闭、FP32 latent缓存/加噪/velocity标签，前向输入显式compute dtype；LayerNorm/GroupNorm局部AMP关闭，RMSNorm保持。Transformer自身固定compute AMP，使FP32 LoRA主权重的分支训练/独立推理一致，不整体cast或清零LoRA A。保留用户attention-only LORA_TARGETS。
- 验证继续走原生Flux3 modular pipeline；单卡/DDP缓存同一pipeline并引用训练Transformer对象，scheduler独立复制，LoRA更新立即可见。临时绕过Accelerate的forward/autocast/输出FP32包装后生成，finally恢复forward/train mode/RNG；构建pipeline失败也恢复模式。FSDP仍汇集LoRA构建副本，真实多GPU FSDP未验收。包装必须保留forward签名，首轮集成因guider筛掉img_ids失败，现wraps原unbound forward并补签名断言，重跑通过。
- 配置/兼容：训练写precision_recipe，独立推理读取；旧缺失字段按legacy，跨方案恢复严格拒绝。旧checkpoint继续需--precision_recipe legacy；新方案必须新输出目录。既有训练进程不会热更新，本轮未重启/停止用户训练或改已有权重。
- 验证：完整CPU相关测试含两进程Gloo/DDP共73项通过（test_flux3_recolor_data/training/inference、reference_trainer/training/distributed、local_model、math_audit、bidirectional、precision_recipe、pipeline_compare、layer_trace；OMP/MKL=1，CUDA_VISIBLE_DEVICES为空、PYTHONPATH=.:tests，RUN_FLUX3_DDP_TESTS=1）。新增原生VAE编码断言后precision_recipe的6项另行通过；RUN_FLUX3_CUDA_TESTS=1、仅GPU1的随机小模型BF16两步训练/实时pipeline画图/继续训练1项通过；git diff --check通过。沙箱CPU首次DataLoader IPC被拒绝，沙箱外重跑；初轮FSDP加载断言仍预期旧BF16 VAE，按新FP32契约更新后通过。
- 真实GPU1结果：verify_flux3_precision_recipe.py只加载Transformer、限本进程显存20%，六组已保存相同输入/noise/timestep对原生无AMP前向velocity均逐位0；最后一组另测零LoRA、人工非零LoRA、真实Accelerate包装临时绕过均逐位0且恢复成功，非零权重确实改变输出。峰值约11.88GiB，过程结束；报告outputs/flux3_precision_recipe_verified.json。独立FP32 VAE真实00047_00、1024预算、两个方向各目标+三条件共8次，helper/原生encoder与helper内外AMP均逐位0，报告outputs/flux3_precision_vae_verified.json。用户GPU1已有其他训练占约58GiB，未触碰；未加载完整大模型训练，也未做多步图像质量验收，不能声称印花根因已解决。
- 复核/文档：独立只读review无重要问题；明确非零LoRA/Accelerate检查只覆盖最后一组。更新使用说明、架构地图、对照说明、测试指南与实施计划。任务完成；下一项任务未指定。

- 本轮逐层定位（2026-11-13，承接用户要求必须找到具体子环节）：新增trace_flux3_tensor_path.py与tests/test_flux3_layer_trace.py。GPU1实际基础模型、同一份完整Transformer输入/noise/timestep/condition/text/IDs，记录前80个叶子事件及全部模式/联合block的全张量SHA256；首个分歧事件42为txt_mode_blocks.0.norm的post。该LayerNorm输入最大差0，关AMP输出BF16、开AMP输出FP32，局部相对差0.1609%；后续调制路径dtype改变。临时只对全部44个LayerNorm关闭AMP、其余模型维持AMP时，velocity相对差5.894%→逐位0。扩展到原先两个方向×三个seed6组，原相对差0.845%–5.894%，干预后六组均逐位0。首个层不等于该层单独解释全网误差；验证的是整个归一化类策略。
- VAE独立定位：同一BF16 VAE、相同像素/布局下，encoder首个AMP分歧为down_blocks.0.resnets.0.norm1（GroupNorm），输入逐位同，输出BF16→FP32；encoder22个归一化模块关闭AMP后latent相对差2.428%→逐位0。固定相同latent及raw decoder input，decoder首个分歧为mid_block.resnets.0.norm1（GroupNorm），decoder30个归一化模块策略对齐后输出相对差0.2266%→逐位0。官方FP32 VAE与当前BF16 VAE的精度差另从像素.to(dtype)开始（max0.00193012），首conv_in输出max0.0198395；共享helper与原生FP32 encoder逐位0，手动逆BN/unpatchify/decode与原生decoder逐位0。decoder在velocity之后，不会造成velocity误差。
- 根因表述：定位为不同入口的归一化autocast输出dtype及VAE加载精度不一致，未证实归一化公式错误或印花根因。训练VAE编码在AMP外、Transformer在AMP内；训练验证整个pipeline在AMP内；独立推理/用户官方示例无外层AMP，官方VAE另为FP32。此前1.7%–10.8%混合了VAE条件latent变化与Transformer上下文变化，不能全归给同一个LayerNorm。本轮干预只临时包装forward并finally还原，生产训练/推理及既有权重未改。
- 验证：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_layer_trace -q`3项通过，含实际随机小模型三阶段、norm策略还原及大activation指纹索引。诊断工具首轮FP32 linspace在N>2^24时把N-1舍入到N导致越界，先RED复现后改int64索引；不是模型训练错误。真实三阶段trace及6组LayerNorm控制已完成，报告outputs/flux3_tensor_path_trace/README.md与各阶段JSON；GPU1显存已释放0MiB。独立复核无重要代码问题，要求区分“首个分歧层”和“全类归一化干预”；git diff --check通过。下一项用户任务未指定，精度策略统一修复尚未实施。

- 本轮浮点复核（2026-11-13）：用户要求重新考虑差异是否仅浮点并重算。新增analyze_flux3_numeric_precision.py，在GPU1仅重放上一轮6组已保存输入：条件tokens×AMP四格、BF16/FP32同配置重复、同权重值/输入值的FP32算术参考（TF32关闭）及最终BF16输出舍入基线。真实6组重复/保存结果重放均逐位一致；训练/官方MSE相对差0.117%–0.923%、velocity相对L2差1.743%–10.777%。训练BF16+AMP对FP32差0.775%–5.351%、官方BF16无AMP对各自FP32差0.797%–5.326%，量级相近；最终舍入基线0.166%–0.167%，不是多层传播误差上界。
- 解释修正：不逐位相同不能据此认定遗漏步骤或必须修改训练。确定性浮点误差也可重复为0；FP32参考改变全精度路径（含时间步乘1000、归一化），不是恢复未知FP32权重/质量真值，也不能断言关AMP更正确。相近loss不能保证最终图像等效，印花根因未由此证实。独立数学复核无重要问题。生产训练/推理/精度参数均未改，未更新原权重/数据；GPU1过程结束显存0MiB，峰值约23.55GiB。
- 验证：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_pipeline_compare -q` 2项通过，涵盖新分析器的实际CPU重放。最初CPU因sigma逆shift的一舍入位差被过严非条件exact guard拒绝，现核对1e-6接近后固定所有非条件字段做factorial；真实GPU保存输入本已逐位相同，重放误差0，GPU计算不受此修正影响。报告outputs/flux3_numeric_precision/{README.md,numeric_report.json}，说明已补充误差解释边界。下一项任务未指定。

- 本轮无梯度真实pipeline对照（2026-11-13）：用户要求独立py复用训练步骤、无LoRA/grad，与pipeline比同sample；后补官方示例VAE FP32/其他BF16，并手动关闭训练授权GPU1。新增compare_flux3_training_pipeline.py及test_flux3_pipeline_compare.py，复用Dataset/collate/encode_latents/文本helper/flow_matching_loss，fork_rng重放真实noise与时间采样（无mock），对齐noisy latent与sigma；native通过ModularPipeline.from_pretrained本地入口+load_components原始FP32 VAE，关闭prompt扩写/CFG以比同一conditional前向，其余官方块保留。RGB、VAE输入、全部Transformer输入/预测、同标签MSE、一步state.latents、独立精度/布局探针与两种上下文重放均记录。默认不放宽atol1e-5/rtol1e-4，不以hook改布局掩盖差异；支持保存CPU张量，报告完成不等于allclose。
- 本轮结果：真实基础权重GPU1、00047_00、1024预算、两个方向×seeds42/43/44共6例。RGB/noisy latent/timestep/text/RoPE IDs逐位相同，条件latent不相同；当前训练VAE为BF16，官方FP32 VAE/无全局autocast不同。当前BF16验证路径中，channels_last+autocast编码探针可逐位匹配pipeline；官方FP32 VAE同像素/布局、无autocast编码helper后cast到Transformer dtype也逐位匹配官方条件tokens。即使Transformer输入相同，切换autocast仍有数值差；同输入且保持训练上下文时6例预测最大差均0。独立复算训练loss最大差5.96e-8。A→B seed42训练loss0.56145465，对官方预测复算MSE0.56663805，velocity RMSE0.10592640；不能声称真实路径等价，也不能断言差异是印花唯一根因。先前CPU控制测试不足以证明实际BF16等价，本轮已纠正范围。
- 本轮验证：新增工具最初RED缺模块，后补一步latent shape回归RED发现state.images是已解码图而非latent，现使用state.latents。最终 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_pipeline_compare test_flux3_math_audit -q`，8项通过。真实GPU仅显式CUDA_VISIBLE_DEVICES=1、HF/Transformers离线；报告与张量位于outputs/flux3_pipeline_compare_verified/{a_to_b,b_to_a}，根README汇总。对照进程已退出，GPU1显存使用0MiB；独立代码复核无重要问题，git diff --check通过。未改生产训练/验证/推理精度配置、原图或既有checkpoint；统一精度/上下文尚未实施。说明见[无梯度对照](FLUX3_PIPELINE_COMPARE.zh-CN.md)，下一项用户任务未指定。

- 本轮计数诊断（2026-11-13，只读）：用户询问1598 samples为何少于自述1905pairs。当前test五目录逐一统计均1776个PNG，无其他文件；最新outputs/flux3_recolor_v2保存report count=1776、training_record_count=1776，配置training_directions=[b_to_a]、repeats1、batch1/G1、validation_fraction0.1，split=train1598/validation178。因此该日志为1776源样本留出178后的单向训练记录，并非双向已展开；相同split双向应3196条训练/356条验证。1905与当前目录及保存报告不符，未推断文件丢失或过滤原因；未改代码/数据/运行，建议新目录明确选择两个方向，旧进程不会热更新。下一项任务未指定。

- 本轮双向换色（2026-11-13）：用户批准将gray/B/colorA→A与gray/A/colorB→B两个子数据集合并。新训练默认training_directions=[b_to_a,a_to_b]，每源各一条、repeats共同作用；源文件名/来源组先拆分后展开。新增CONDITION_FOLDERS2/TARGET_FOLDER2及集中task_folders/subdataset_config；B色块完整配对/纯色检查、像素参与指纹。Dataset/缓存身份含方向，单b_to_a保留旧身份；配置保存方向/子集映射，旧checkpoint仅在显式选择单b_to_a且其他配置/数据不变时兼容。loss可返回逐样本误差，训练按方向真实样本数归并loss_b_to_a/loss_a_to_b，控制台/JSON/TensorBoard共用；不把混合batch均值复制给两方向。
- 本轮验证/推理：每个留出源样本生成两个方向，PNG及TensorBoard标签含方向；num_validation_images=4表示共8张生成图。单方向保留旧目录；推理--direction选择方向、手动三图新增source_image/color_image中性别名。独立复核发现源文件名generated.png会被递归glob误当图片目录，真实验证先RED复现再加is_file过滤，综合测试固定保留此边界。完整61项CPU回归通过，最终console/help调整后双向6项+实际训练/恢复/出图/反向推理1项共7项再通过；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests RUN_FLUX3_DDP_TESTS=1 ../.venv/bin/python -m unittest test_flux3_reference_distributed -q` 1项通过，实际双CPU rank双向分片无交集、参数同步、逐步验证继续训练、连续与中途恢复adapter逐位一致。worker/Gloo用获准沙箱外IPC/socket；均为本地随机小模型。
- 本轮真实数据检查：`OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python train_flux3_recolor_lora.py --check_dataset --instance_data_dir ./test`通过；当前已1754组源样本，三场景均768×1024，两色块均32×32，所选B色块均纯色。默认split为train1579/validation175，双向记录3158/350（repeats1）。未改原图/旧权重/运行配置，未启动真实GPU训练。文档与实施计划已更新，git diff --check通过；下一项用户任务未指定。

- 本轮训练逻辑审计（2026-11-13）：按用户要求核对Dataset/RGB/同步几何/色块、VAE mode+patchify+BN、三参考L坐标、Qwen padding、flow加噪/velocity/加权MSE、梯度累积、缓存、恢复和官方验证解码。新增6项独立数学/官方pipeline差分回归及1项完整batch/4+2/不足累计尾组梯度对照。核心公式未发现符号/顺序/漏归一化错误；生产代码只补 `prediction.shape == velocity.shape` 硬检查，先RED后修复，拒绝静默广播，不改变正确形状loss。详见 [训练核查](FLUX3_TRAINING_AUDIT.zh-CN.md)。
- 本轮验证：最终 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_recolor_data test_flux3_recolor_training test_flux3_recolor_inference test_flux3_reference_trainer test_flux3_reference_training test_flux3_local_model test_flux3_math_audit -q`，54项通过；DataLoader worker在沙箱内因IPC权限受阻，获准沙箱外CPU重跑通过。FP32/BF16 VAE归一化/逆变换、解析sigma/loss/梯度、零LoRA增量及文本官方编码均通过；未重跑真实权重GPU/多卡FSDP/量化。端到端差分曾因小型VAE输入stride不同出现约1.7e-3条件token差异；权重/像素/eval/dtype相同，仅在测试统一encoder连续布局后全部Transformer输入比较通过，未放宽阈值或改生产布局，不归因为真实印花根因。
- 本轮真实数据只读检查：当前test已是468组（历史运行423组），场景均768×1024、原色块均32×32且纯色。精确灰128代理区域外A/B逐像素一致；不能据此证明衣服内部纹理一致。未改原图/旧运行配置/权重，未启动真实模型训练；条纹/印花质量问题仍待固定样本基础模型与checkpoint对照，不宣称已解决。下一项实施任务未指定。

- 本轮 FLUX Studio 离线加载修复（2026-11-12）：日志确认进入 Diffusers 顶层 download 分支，传入值未被识别为存在的本地目录；已核对普通下载目录及组件文件存在。修正 `flux_klein_reference.py` 的 CACHE_DIR 多向上一层问题；未显式设置模型变量时优先使用项目相对 `../../hf_cache/FLUX.2-klein-base-9B`，目录不存在才回退 Hub ID；显式 `OBJECT_INPAINTING_FLUX_MODEL_DIR` 仍优先，保持离线。路径断言修复前失败；修复后 CPU 一次性检查缓存路径、默认本地目录选择及显式覆盖均通过（替代推理输出），`git diff --check` 通过。未加载真实权重、未启动 GPU 或重启服务，先前可运行时的环境尚未核实。下一步：重启 Studio 后单张验证。

- 本轮 Studio SAM3 路径排查（2026-11-12）：批量失败源于节点默认模型目录多向上取一层；修正 `sam3_text_mask.py`，使默认位置与启动脚本默认缓存根一致，为项目相对 `../../hf_cache/sam3`；保留 `OBJECT_INPAINTING_SAM3_MODEL_DIR` 覆盖。本地已核对目录及 config / safetensors / processor / tokenizer 文件存在，无需重新下载。一次性路径断言修复前失败、修复后通过，`git diff --check` 通过；SAM3 unittest 已启动但停留在首个 API 预览用例，未取得完成结果，不记为通过。未加载真实权重、未启动 GPU 推理或重启服务。下一步：重启 Studio 后先试跑单张，再恢复批量；真实分割效果待验证。

- 本轮纹理幻觉排查（2026-11-12，只读模型/数据）：用户报告单prompt训练新增条纹/图案。查看v3 steps100/200的00047_00对照，B图与A目标均纯色，输出100步出现袖部橙色图案、200步出现大片印花；00012_00原有字样有变形。v3保存配置为rank32/alpha32、lr1e-4、G4、场景面积512²、色块128、详细固定prompt，读取history至288步，未启用prompt增强。故已确认存在纹理保真问题，不能归因随机prompt，也不能声称继续训练/改措辞能解决。源码核对flow加噪、velocity目标、独立参考流与参考脚本对应；当前只用全图velocity MSE，从噪声生成，无显式纹理保真项或输入像素保留硬约束，尚未证明具体根因。建议先固定样本/seed/推理设置做基础模型无LoRA与多个checkpoint对照，再核验训练B/A纹理一致性与mask质量，证据后决定低LR对照或结构保真约束；区域加权不能单独解决衣服内部凭空印花。未改训练参数/代码/权重，未启动GPU实验，根因与修复待验证；下一项实施任务未指定。

- 本轮prompt增强（2026-11-12）：新增可选 `--prompt_augmentation`，6种固定等义英文模板保留三图角色与细节约束；按seed/样本标识起点加epoch轮换，可复现。显式prompt或caption_json优先，默认仍单指令。模板批量提前编码到共同窗口，按样本选embedding，支持caption dropout和latent缓存，训练前释放文本编码器；验证/推理固定默认prompt。配置保存模板列表，恢复检查一致性，旧单prompt配置补默认字段兼容；切换增强需新目录。指南含使用和单变量对照方案。CPU验证：新增Dataset可复现/覆盖测试通过；小模型增强+dropout+latent缓存+逐步验证+续训1项通过；原单prompt缓存精确恢复与Dataset测试共2项通过。未启动真实GPU训练或质量对照，不能断言颜色质量改善；下一项任务未指定。

- 本轮默认prompt更新（2026-11-12）：按用户示例强化只改服装颜色、保留身份/纹理/光照/背景等约束，并按实际顺序明确图1涂灰定位、图2待改色完整B图、图3目标色块；颜色直接匹配图3，无未定义color_name变量。核对Dataset/collate/flow_matching_loss：原图A独立作为加噪目标及velocity监督，不混入三张干净条件图。指南同步角色与默认指令。已运行Python检查默认prompt、CLI显式覆盖、三条件常量及语法，通过；未启动训练或修改原始图片/既有运行配置。下一项任务未指定。

- 本轮dataset色块放大（2026-11-12）：按用户要求，新训练在data读取时将第三条件颜色块以NEAREST独立resize到默认128×128，纯色RGB不变，不受场景max_area影响，不改test原始图片。新增 `--color_swatch_size`（默认128、16..1024的16倍数），根preprocessing_config保存color_swatch_size；默认面积与分桶两条路径均接通，验证/推理共享policy，缓存编码得到相应新尺寸。旧LoRA policy缺该字段时仍按原生尺寸推理；新128训练不能直接resume旧32配置，需新输出目录，不会热改正在运行的训练。
- 本轮验证：先将三项色块/策略/Dataset回归跑RED，再实现。`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_recolor_data test_flux3_reference_trainer test_flux3_recolor_inference test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cosine_caption_dropout_validation_and_inference test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cached_training_resume_matches_uninterrupted_parameters_rng_and_lr -v`，29项通过，涵盖RGB/原对象不变、独立预算、legacy32、Dataset尺寸/配置、缓存和小模型训练/验证/推理。增加实际保存配置及validation/inference的condition_3 PNG均128断言后，单独重跑cosine完整链路1项通过。真实00002_00样本检查场景432×576、色块128×128、RGB仍(18,18,19)，源文件SHA256与32×32尺寸未变。未启动GPU训练/质量对照，未修改旧运行配置或产物；下一项任务未指定。

- 本轮rank/色块尺寸分析（2026-11-12，只读）：用户报告rank32效果欠拟合，随后询问32×32色块改128×128是否改善。核对实际配置：run2为rank16/alpha16，run3为rank32/alpha64，其余保存配置相同；不是单变量rank对照，普通PEFT缩放alpha/r由1变2。共同steps401–500训练loss均值分别0.10711/0.10474，不能据此断言rank32欠拟合；另v2为rank32/alpha32/G4/warmup100，读取时仅96个更新步，也不可直接与500步结果比较。查看相同step500的前三张对照，均能换到近黑色，未据此给出整体质量结论。
- 当前真实FLUX.3 VAE×packing网格为16：32色块=2×2/4tokens，64=4×4/16tokens，128=8×8/64tokens；默认432×576场景每张972tokens，现预处理不主动放大32色块。128可能增强色块条件表示，但纯色信息未增加、改善尚未验证，建议固定其他超参数做尺寸对照，并同步训练/验证/推理及重建缓存，新目录运行。另当前默认4张验证A色块RGB分别(18,18,19)/(24,25,28)/(28,29,30)/(28,28,29)，全部近黑；应增加彩色覆盖再评估颜色条件。以上为分析建议，未修改训练代码/图片/运行配置，未启动实验；下一项实施任务未指定。

- 本轮第100步验证崩溃修复（2026-11-12）：用户真实训练打印至step100后出现NoneType.index，触发点是validation_steps=100而非训练horizon结束。核对本地Accelerate单卡default_device返回torch.device('cuda')且index=None，旧log_validation把[None]交给torch.random.fork_rng，PyTorch get_rng_state访问None.index失败。新增cuda_device_index，在未指定索引时取当前逻辑CUDA卡；验证RNG及bitsandbytes device_map共用。物理GPU1经CUDA_VISIBLE_DEVICES=1后对应逻辑0。原流程验证先于checkpoint导致此次checkpoint-100未保存，现已改为同一步先提交checkpoint再验证。
- 本轮实际输出检查：`outputs/flux3_recolor_smoke`含配置、split、logs和train_history，末两行steps99/100，无checkpoint或训练后LoRA，不能由loss日志恢复已丢失权重；未删除/覆盖/重启用户训练。应保留旧目录，使用新目录从头运行；TensorBoard可用logdir=outputs同时查看旧/新run。
- 本轮验证：两项新增回归先RED（单卡索引解析、验证失败前checkpoint不存在）再修复；CPU运行单卡索引/模拟验证失败可恢复/训练中TB图像/缓存连续恢复4项通过，真实小模型adapter与RNG精确恢复。获准只用GPU1运行 `CUDA_VISIBLE_DEVICES=1 RUN_FLUX3_CUDA_TESTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_cuda_validation -v`，1项通过：随机小型BF16模型两步每步验证、CUDA RNG恢复、checkpoint及TB steps1/2，未加载真实基础权重或改用户产物。CPU对应 `test_flux3_reference_trainer.ReferenceTrainerTests.test_single_cuda_without_explicit_index_uses_current_logical_device` 和 `test_flux3_reference_training.ReferenceTrainingIntegrationTests` 下 `test_checkpoint_is_committed_before_validation_failure_and_can_resume` / `test_tensorboard_validation_images_update_during_training_at_real_steps` / `test_cached_training_resume_matches_uninterrupted_parameters_rng_and_lr`，4项通过。未重跑完整CPU/DDP套件。下一项任务未指定。

- 本轮latest恢复诊断（2026-11-12）：用户按smoke目录加resume=latest运行，收到合并式“旧状态格式不兼容”提示。重新只读检查确认当前工作区连outputs目录都不存在，因此实际原因是恢复路径不存在；没有证据判断旧格式，也未推断目录消失原因。新增 `resolve_training_checkpoint`，明确区分输出目录缺失/没有checkpoint/有候选但状态无效，并在torch导入、数据检查及模型读取前失败。仍不允许显式续训失败后静默新训练；没有检查点应删除resume参数从头开始，有其他位置的检查点则使用正确output_dir。指南明确恢复示例前提。未重建、删除或覆盖训练产物，未启动真实训练。
- 本轮验证：新增2项诊断回归先失败后通过（`test_flux3_reference_trainer.ReferenceTrainerTests.test_latest_missing_directory_is_distinct_from_incompatible_state`、`test_latest_valid_state_and_missing_resume_fail_before_loading_inputs`），实际CLI子进程确认目录不存在时先报恢复路径、不会先读取模型或数据。另运行 `test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cached_training_resume_matches_uninterrupted_parameters_rng_and_lr` 1项通过，真实小模型latest恢复与连续训练的adapter/RNG逐位一致。均用projects Python、PYTHONDONTWRITEBYTECODE=1、PYTHONPATH=.:tests，模型回归CUDA_VISIBLE_DEVICES=''；相关语法及git diff --check通过，未重复完整CPU/DDP套件。下一项任务未指定。

- 本轮训练中实时验证（2026-11-12）：根训练脚本默认report_to改为tensorboard、validation_steps=100、num_validation_images=4。原有训练循环的按步/epoch/最终验证均显式传真实global_step；TensorBoard使用固定 `validation/generated/<样本名>` 与 `validation/comparison/<样本名>` 标签，最终阶段为test；每次验证后flush，不再按step建新tag且默认step=0。W&B也使用固定阶段key和真实step，未实际连接外部服务。仍使用固定留出样本/seed，PNG按step保留，验证完继续训练，不是训练后才单独推理。指南增加TensorBoard启动/Images与Scalars位置和从已有2步smoke恢复的完整命令；短程测试可validation_steps=1。未修改既有训练权重/状态，未启动真实GPU训练或TensorBoard常驻服务。
- 本轮验证：先用真实事件读取回归复现旧实现图像标签/step问题，再修复。`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_tensorboard_validation_images_update_during_training_at_real_steps test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cached_training_resume_matches_uninterrupted_parameters_rng_and_lr test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cosine_caption_dropout_validation_and_inference -v`，3项通过。首项在训练验证返回时、end_training之前读取实际TensorBoard event，确认同一生成/五列对照tag的steps先为{1}再为{1,2}；另两项核对参数/RNG精确恢复、验证后继续训练与推理。未重跑完整CPU/DDP套件；没有把之前GPU独立推理当训练内验证证据。下一项任务未指定。

- 本轮真实验证图片（2026-11-12）：用户希望看到训练效果。读取确认用户已有 `outputs/flux3_recolor_smoke`，training_state记录global_step=2、BF16，LoRA及checkpoint-1/2存在，原num_validation_images=0。未重训、未修改原训练配置或权重；获准仅使用GPU1离线运行 `infer_flux3_recolor_lora.py --lora_dir ./outputs/flux3_recolor_smoke --dataset_root ./test --limit 4 --output_dir ./outputs/flux3_recolor_smoke_validation --device cuda --mixed_precision bf16 --seed 42 --num_inference_steps 28`，exit0。留出样本00002_00/00012_00/00047_00/00049_00，seeds42–45；manifest确认4张，所有24个PNG已实际解码，生成/目标432×576、色块32×32、五列comparison2160×576。报告位于验证目录README.md，按涂灰AB/B图/A色块/目标A/模型输出排列。
- 观察与维护：已查看第一张comparison，目标黑色服装、输出仍为B图橙粉色；不能将2步smoke结果视为已学会换色或收敛。根训练脚本默认num_validation_images由1改为4，显式0仍关闭；指南推荐每100步固定4样本验证。默认参数读取及git diff --check通过，此低影响默认/文档改动未重复运行完整CPU/DDP套件。真实基础权重加载和4张GPU推理本轮已执行，未启动额外训练，未测峰值显存/色差指标。下一项任务未指定；若沿用上轮命令需去掉num_validation_images=0。

- 本轮实际模型缓存核对（2026-11-12）：用户已下载 `../../hf_cache/models--black-forest-labs--flux-3-image-klein-base`，refs/main 为 `a83ce283b1daa2f3d782645ae7b06c8e73af1b41`。已只读核对Transformer两片/VAE一片/text_encoder两片，索引列出的文件全部存在、无incomplete blob，safetensors头偏移与文件长度一致；组件权重约11.08/0.34/8.88GB，非显存测量。Transformer使用默认flow_map=False，Qwen36层/hidden_size2560，默认8层文本维度20480匹配Transformer。尚未加载这些真实权重张量或执行GPU训练。
- 实际布局修复：官方base的processor/tokenizer在text_encoder，modular_model_index有明确subfolder；旧脚本硬要求tokenizer目录会失败。`config.py`现解析HF缓存main→snapshot及本地processor元数据/独立目录回退，并校验索引缺片；训练/推理共用。processor从具体组件目录加载，避免Transformers5.5.4读取BFL根config而误报Mistral regex；已确认原regex逐字等于本地Qwen2标准，未启用Mistral修补，实际Qwen3VLProcessor/chat template离线读取成功、默认指令61 tokens。新增布局回归；`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_local_model test_flux3_recolor_training test_flux3_recolor_inference test_flux3_reference_training.ReferenceTrainingIntegrationTests.test_cosine_caption_dropout_validation_and_inference -v`，13项通过。相关Python语法与git diff --check通过。未改缓存文件，未下载/安装/启动真实训练；下一步真实GPU少步验收待执行，下一项代码任务未指定。
- 额外只读结构核对：使用Accelerate init_empty_weights实例化meta模型，Transformer357个/VAE251个/Qwen713个实际存储张量的名称及shape与本地类匹配。Qwen配置tie_word_embeddings=True，未单独存lm_head.weight是正常共享；在meta模型调用tie_weights后验证其与input embedding是同一Parameter，不能将该键当缺片。VAE配置额外force_upcast属性被本地类忽略；未因此修改缓存或推断真实精度表现。以上不是完整真实权重加载/前反向验证。

- 本轮参考训练框架重整（2026-11-12）：用户批准[新版设计](superpowers/specs/2026-11-12-flux3-reference-trainer-design.md)，已按[计划](superpowers/plans/2026-11-12-flux3-reference-trainer.md)实施。根 `train_flux3_recolor_lora.py` 现承载参考式 parse_args/Dataset/collate/分桶/Accelerator/PEFT/optimizer/LR/cache/prepare/accumulate/保存验证流程；`flux3_recolor/training.py` 仅兼容转发，移除旧手动累积循环。支持原生DDP及累积、多batch/repeats/workers、参考自动比例梯级和显式桶（H,W;H,W及HxW）、同步确定增强、身份缓存、caption JSON/dropout匹配窗口、时间采样/weighting、AdamW/Prodigy/scale_lr/调度warmup、tracker、offload、检查点限额和epoch/batch/RNG/LR恢复。推理共用持久化预处理策略；四目录/三参考固定顺序不变，不改原Diffusers示例。功能对应表和命令见[使用说明](../FLUX3_RECOLOR.zh-CN.md)。
- 本轮验证：同一projects Python3.12.8环境，`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_recolor_data test_flux3_recolor_training test_flux3_recolor_inference test_flux3_reference_trainer test_flux3_reference_training -v`，34项通过，无跳过；包含本地随机FLUX.3/VAE/Qwen/tokenizer全链路、缓存连续/恢复adapter和RNG逐位一致、cosine/warmup、epoch/workers、真实TensorBoard事件、Prodigy恢复。另 `RUN_FLUX3_DDP_TESTS=1` 运行 `test_flux3_reference_distributed -v`，1项通过：真实两rank CPU DDP参数同步、分片、验证后训练、epoch中途checkpoint与尾累积恢复，连续/恢复最终adapter逐位一致。DataLoader IPC及Gloo socket在沙箱内被拒绝；受阻测试已中断，获准在沙箱外重跑后通过，不能将沙箱失败当通过。新版 `--check_dataset --instance_data_dir ./test`（临时报告输出）全量423组通过，381/42拆分、场景均768×1024/色块32×32。未运行无关Studio/浏览器套件。
- 审查/限制：独立审查两项Important均补失败回归后修复：FSDP1包装前Transformer/LoRA master统一FP32，梯度裁剪传完整参数列表触发全局norm。仅CPU接口验证，不宣称真实FSDP运行通过。FSDP1需use_orig_params=true，原生checkpoint包含基础参数，验证需额外未分片模型；量化限单GPU，FP8/量化/8bit/Prodigy与FSDP组合明确拒绝。新训练状态version=2拒绝旧手动循环checkpoint；非constant调度不允许改horizon，恢复保持world_size和关键配置。只删除带本入口有效状态标记及匹配步数的完成checkpoint。16份Python语法/空白、8份文档本地链接、CLI help及git diff --check通过。真实权重/GPU精度/scaler/显存/FSDP/bitsandbytes/FP8/质量未执行，torchao未安装；未下载/安装/连接外部tracker/启动业务服务/自动提交。基础权重现已下载，见本轮缓存核对；真实GPU少步检查待执行，下一项代码任务未指定。

- 本轮 FLUX.3 换色 LoRA（2026-11-11）：用户批准[设计](superpowers/specs/2026-11-11-flux3-recolor-design.md)并要求实施，新增[训练/数据检查](../train_flux3_recolor_lora.py)、[三图推理](../infer_flux3_recolor_lora.py)及 `flux3_recolor/` 独立模块；详见[使用说明](../FLUX3_RECOLOR.zh-CN.md)和[执行计划](superpowers/plans/2026-11-11-flux3-recolor.md)。条件固定为涂灰原图 AB / 生成图 B / 主颜色 A，监督原图 A；三图独立编码 L=-1/-2/-3，目标标准全图 flow matching LoRA，冻结基础组件。默认固定英文指令、面积预算 512²（本数据场景处理后 432×576，色块仍 32×32）、单步 batch=1、梯度累积、约10%留出和可选来源分组。支持本地模型、FP32/FP16/BF16、梯度检查点、标准 adapter、optimizer/scaler/RNG/采样位置恢复及图像对照。只支持非蒸馏 `flow_map=False` 的完整 Diffusers 模型，不自动下载。
- 本轮实际验证：projects Python 3.12.8 环境，Diffusers 0.42.0.dev0/torch 2.11.0/transformers 5.5.4/accelerate 1.15.0/peft 0.21.2；`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_flux3_recolor_data test_flux3_recolor_training test_flux3_recolor_inference -v`，23 项通过。真实随机初始化小型 FLUX.3/VAE/Qwen3-VL/本地 tokenizer 全链路训练→途中及结束验证→LoRA重载推理通过，无下载；恢复与连续训练的 adapter/RNG 逐位一致。全量 `--check_dataset --dataset_root ./test`（另指定临时报告目录）确认 423 组均可解码/配对/预处理，三个场景目录全部 768×1024，A色块全部32×32，拆分381/42。11个Python文件语法、CLI help、`git diff --check` 通过；未运行 Studio/浏览器及无关套件。
- 独立审查发现并修复：途中验证共享 scheduler 会改写训练 sigmas（真实小管线和完整入口均先复现失败，再复制 scheduler 修复）；有效几何报错补样本名。额外回归确认非默认 LoRA alpha metadata 在推理重载保留。未下载/加载真实基础权重，未做真实 GPU 训练、显存测量或质量评估；环境有 CUDA 初始化受限及第三方实验API/弃用提示，CPU测试均通过。下一步用户手动下载兼容基础权重后按说明做真实少步训练；下一项代码任务未指定。未改原 Diffusers 示例、Studio或已有业务修改，未启动服务/自动提交。

- 本轮常用服装换色（2026-11-11）：[随机换色节点](../object_inpainting/nodes/flux_random_recolor.py) 从 16 色扩展到固定 48 色：黑白灰 / 米棕 / 红粉 / 橙黄各 8，绿 / 蓝各 6，紫 4。每色固定英文名与 RGB，不新增随机 HSV 或特殊材质；候选仍优先满足与编辑区中位 RGB 距离 >=80，seed=-1 随机 / 固定 seed 可复现、参考色块和生成合成逻辑不变。旧 seed 升级后可能选到不同颜色。工作流、端口、启动脚本未改，未重启服务。
- 本轮验证：使用用户指定 projects Python 3.12.8 虚拟环境，`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/tests" CUDA_VISIBLE_DEVICES=1` 下运行 `python -m unittest test_flux_random_recolor_node test_flux_klein_reference_node test_node_contracts -q`，33 项通过。新增 4 项覆盖 48 色数量 / 名称 / RGB 唯一有效性、全色可达、种子及全局 RNG 隔离、仅编辑区筛选和最远色回退；生成边界测试确认英文名与整张色块 RGB 一致。Python 语法和编辑器诊断通过；模型为替代输出，未真实模型推理或完整套件。已更新[换色说明](../object_inpainting_studio/RANDOM_RECOLOR.zh-CN.md)。下一步重启已运行的 Studio 后端后使用原工作流即可；下一项任务未指定。

- 本轮色卡旧设置清理（2026-11-11）：核对确认色卡节点 / 前端 / 迁移器此前已删除旧 `images` 输出及兼容逻辑，本轮清除仓库工作流 (8)/(11) 的 `legacy_swatch_images`，并将 (6)/(8)/(11) 的旧色卡列表保存连线改为 `colorregionswatches.img1 → save.img6`；补齐保存项及 next ID，保存目录仍为 `images_0`，其他连线不变。当前打开的下载副本 (11) 同步清理。保存节点通用 `IMAGE_LIST` 输入及其他节点兼容不删；保留旧色卡端口拒绝测试，不新增自动迁移。修正使用指南和测试指南的旧列表说明，新增仓库 JSON 无残留 / 保存元数据回归。
- 本轮验证：系统 PATH 无 Node，改用 VS Code 自带 Node 运行时执行五份前端测试，共 146 项通过（无失败 / 跳过）；PowerShell 实际解析 6 份仓库工作流、5 个色卡节点，核对无旧标记 / 旧连线、三份保存配置及下载副本一致。修改 JSON / JS 编辑器诊断和 `git diff --check` 通过。用户跳过 Python 环境配置，本轮未运行 Python、真实浏览器或模型，不沿用前轮结果；未重启服务。无阻塞，下一项任务未指定。
- 本轮原名保存（2026-11-11）：按用户要求，Studio 保存 `filename` 留空时自动沿用唯一上游文件夹的当前 filename，或唯一 LoadTarget 字面路径的 basename；显式名称/连线优先，无明确来源才用 UUID。新增 `naming.py`，在独立执行图补默认命名，不改用户 JSON；覆盖普通逐张、条件循环、试跑和单图保存。保存仍统一 PNG、保留中文/多点主干、已有文件报冲突不覆盖不追加编号。更新节点帮助、端口说明和指南。
- 本轮验证：`code_judge` Python 3.12.13，前轮相关十二模块 243 项通过（mock demo 等待）；五份 JS 145 项通过；真实 Chromium 三项通过（双保存沿用原名、试跑后运行全部同名冲突及模式往返、单图保存与逐张区分）。新增测试确认当前 index 名称、显式名称优先、原 JSON 不变、条件达标保存与默认同名重跑拒绝、中文多点文件名；旧损坏文件测试改用独立输出目录避免被原名冲突提前拦截。未运行模型/完整浏览器套件，未重启业务服务；无阻塞，下一项任务未指定。
- 本轮逐张设计优化（2026-11-11）：落实 ComfyUI 参考中的显式结果、统一计划、分层调度。新增可选 `workflow.batch_execution`（mode/result_nodes/disabled_nodes），默认全部末端；卡片可“作为结果 / 取消结果”和“禁用 / 启用分支”，工具栏切模式，配置随 JSON 保留。可只选评分或循环内部中间节点，无需保存；禁用传播下游，必要验收依赖禁用时报错。指定结果只选必需循环，旧 flat 兼容；独立审查发现并修复中间结果未保留达标缓存而重跑、无关未完成循环提前校验、旧 flat 出口检查误拦截。删除节点会清理选择 / 禁用引用。
- 后端 `/api/batch/plan` 统一返回执行顺序、重复节点、角色、结果、写盘节点和阶段；前端不再重算执行范围，等计划时显示等待、清除旧高亮，乱序响应隔离，完成后重画连线。逐文件任务增加 current_item（ID/index/total/status）、execution_phase 和 execution_plan；沿用外层按图调度 / 内层条件重试，每张缓存隔离，不一次展开目录。只评分时按钮显示计算结果。未改用户 JSON，未重启业务服务，未执行模型。
- 本轮验证（2026-11-11）：`code_judge` Python 3.12.13，项目/tests 在 `PYTHONPATH`；加载 `test_folder_terminal_branches test_folder_batch_node test_loop_demo test_multiple_conditional_loops test_conditional_loop test_loop_preview test_loop_downstream test_node_contracts test_object_inpainting_studio test_save_images_node test_preview_cache test_node_errors`，仅 mock demo 等待，239 项通过。五份 JS 套件 145 项通过；真实 Chromium 六项通过（指定结果/禁用/不写盘、全部末端双保存、多保存范围、固定/断开分支、条件向导、目录重校验）。相关 JS 语法和 `git diff --check` 通过。独立审查复核三个问题的修复，无剩余发现；使用指南/架构/测试指南已更新。无阻塞，下一步用户重启后端刷新页面，下一项任务未指定。
- 本轮参考设计分析（2026-11-11）：只读核对本地 `参考资料/ComfyUI`，确认 `execution.validate_prompt` 以显式 OUTPUT_NODE/部分执行目标确定根；`nodes_loop.py` 的 StartLoop 支持次数/For/List，EndLoop 分离 output_value、next_iteration_value、每轮 termination 依赖和 accumulate；通过动态展开与外部阻塞释放最终结果。列表普通映射与逐文件完整流程语义不同，不能直接替换现有逐文件调度。建议后续借鉴显式结果节点/实例目标、清晰循环边界、后端统一执行计划，保留当前按文件延迟执行而非为大目录一次展开所有轮次。未修改业务代码、未执行测试或模型；下一项实施任务未指定。
- 本轮全部末端执行（2026-11-10）：按用户明确需求，逐张及试跑改为运行同一文件夹下游所有末端（无条件评分、多保存和其他结果处理），共享上游每张仅执行一次；普通单节点测试范围保持。新增 `batch_scope.py` 负责范围和结果收集，计划/画布/末端预览同步；运行入口不再限制分支，多个保存入口默认选第一个。条件循环验收后用缓存执行末端，未达标不写盘；独立审查发现并修复条件上游提前写盘、多个 SaveImages 后续失败残留，补前置拒绝和本张新文件清理（不删除既有文件，取消保留已完成输出）。所有保存目录校验并从输入扫描排除。未修改用户 JSON，未重启业务服务，未执行真实模型。
- 本轮验证：`code_judge` Python 3.12.13，项目/tests 在 `PYTHONPATH`，mock demo 等待；`test_folder_terminal_branches test_folder_batch_node test_loop_demo test_multiple_conditional_loops test_conditional_loop test_loop_preview test_loop_downstream test_node_contracts test_object_inpainting_studio test_save_images_node test_preview_cache test_node_errors` 共 229 项通过；五份前端 142 项通过；真实 Chromium 四项（全部末端评分/双保存、多保存共同范围、固定与断开分支、条件向导保存）通过。独立审查复查修复，无新严重问题。使用指南/架构/测试指南已更新；无阻塞，下一步用户重启 Studio 后端刷新页面，下一项任务未指定。
- 本轮工作流诊断（2026-11-10）：检查用户工作流 (11)，仅启用文件夹逐张遍历，未包含 `workflow.conditional_loop`；`colorsimilarity` 无下游且 `workflow.outputs` 为空，因此不在保存节点祖先范围，显示“不执行”符合当前执行语义。使用 `code_judge` 调用真实 `node_preview_graph(workflow, 'save')` 核实评分未纳入保存执行图，未执行模型/写出图片，未修改用户 JSON。若需要按色差重试，应添加以 `fluxrandomrecolor_2` 为起点、`colorsimilarity` 为终点及条件 `score <= 用户阈值` 的条件循环；阈值待用户指定。下一项开发任务未指定。
- 本轮逐张显示修复（2026-11-10）：用户工作流的评分为保存分支旁路上的条件终点，前端 `loopScope` 原先只遍历保存祖先，错误标注“不执行”。新增 `batchUpstreamNodes`，与服务端 batch 的条件/终点 roots 对齐，覆盖范围、每张执行角色、步骤和文件夹归属；未修改用户 JSON。新增前端回归核对旁路评分、固定输入及取消条件后的范围。五份 JS 套件 141 项通过，`git diff --check` 通过；使用 `code_judge` 与用户 JSON 的相同接线（仅临时替换输入/输出目录、mock 等待）运行 `/api/batch/preview`，确认评分实际调用一次、score=0 满足原配置 <=10。未跑真实浏览器，未重启服务。用户原配置第二个 demo 无 image 输入，保存 32×32 色图；若要保留原图需补 image 连接，蓝色三轮验收需 >=255。下一项任务未指定。
- 本轮文件夹演示（2026-11-10）：已按用户选定方案给 `LoopDemoImageNode` 增加可选 RGB `image` 输入，复制原图并在左上角覆盖最多 32×32 七色标记；保留原尺寸、其余像素、随机 seed、独立无输入演示与用户 10 秒模拟等待。评分改为仅计算标记蓝通道均值；seed=0 每文件三轮通过。节点帮助、端口说明、使用指南及架构地图已同步。新增真实两文件 API 批处理测试，核对 seed 分别 0/1/2、原图未改、输出文件名和像素、仅保存两张达标图。`code_judge` Python 3.12.13 下加载 `test_loop_demo test_folder_batch_node test_loop_preview test_loop_downstream test_node_contracts test_multiple_conditional_loops`，统一 mock `object_inpainting.nodes.loop_demo.time.sleep` 跳过真实等待，138 项通过；`git diff --check` 通过。未跑完整套件/浏览器，未重启服务；下一步用户重启后端刷新，按使用指南接线测试，下一项开发任务未指定。
- 本轮修复（2026-11-10）：用户新增七色与 `time.sleep(10)` 后，`from time import time` 导致 AttributeError；已复现并改为 `import time`，保留延迟及七色。使用 `code_judge` Python 3.12.13 直接核对 14 个固定 seed、随机 seed 及 15 次 `sleep(10)` 调用（mock 跳过真实等待），通过；`git diff --check` 通过。未重跑完整测试，旧三色周期测试与用户新增七色已不一致；无阻塞，下一项任务未指定。
- 本轮补充（2026-11-10）：`LoopDemoImageNode` 支持 `seed=-1` 每次执行随机生成非负 seed，`value` 返回实际 seed；非负 seed 保留红绿蓝周期。修复未定义 `colors`，同步节点帮助与随机/API 回归测试。已用 `code_judge` Python 3.12.13 执行 `python -m unittest test_loop_demo test_loop_preview test_loop_downstream -q`（项目与 tests 在 `PYTHONPATH`），68 项通过，`git diff --check` 通过。未运行完整套件，无阻塞；下一项任务未指定。
- 更新日期：2026-11-10。
- 本轮目标：按新用户要求，条件循环只需 start / end / condition；普通节点测试自动验收必要循环，逐轮可视化且不额外生成。
- 当前界面：保留三项画布点选 / 重选、condition operator / value、预览与删除；已删除循环命名输入、多条件增删 / all-any、高级多起终点、种子 / 次数 / 超时 / 耗尽、公用保存选择和手动检查计划步骤。安全限制及非负起点 INT seed 递增内置，seed=-1 保持随机；旧高级 JSON 参数兼容保留，不因简化界面丢数据，明确重选条件才替换旧多条件列表。
- 执行与兼容：[previewNode](../object_inpainting_studio/static/app.js) 自动请求 `/api/node/preview`；普通节点依赖循环内部任意输出时（包括测试循环内部节点），后端先验收必要循环，再用达标缓存算目标，无额外生成，无需补 `end_nodes`，失败不跑普通下游。独立循环预览仍走 `/api/conditional/preview`，无需保存节点 / 目录，不写盘，无关半成品不阻塞。前端旧 flat 自动迁为 loops；直接 API 用 flat 测试普通节点返回 422 防绕过，flat 正式保存兼容。保存直接点击相应 SaveImages 节点运行，自动校验，无需手动检查计划。
- 可视化：`#visualize` yes / no 默认 yes。每轮推送 `loop_previews`（每循环仅一份最新快照）及 `visualization_revision`，更新画布节点 / 循环卡片 / 任务输出区；不保留图片历史，`history` 仍记录各轮评分。no 随请求传 `visualize=false`，后端不序列化图片，不影响计算；中途选 no 立即隐藏，但已在服务器开启的本任务采集不会实时关闭，下次请求才关闭。刷新可恢复最近任务图像快照，不代表当前图配置已验收。
- 无模型演示：[RGB 周期图及蓝通道评分](../object_inpainting/nodes/loop_demo.py) 可追加两个独立循环，不替换现有流程；seed=0/1、score>=255 分别 3/2 轮通过。失败演示改阈值为 256，显示末轮及未通过，不要求已删除的次数控件。
- 已实际验证：五个前端文件共 **140 项全部通过**（最终清理后重跑）。已选环境 Python **3.12.13**，循环相关核心 **149 项通过**；扩展至 Studio API、文件夹、缓存、节点错误、图片保存及端口命名后，**222 项通过，无跳过**；具体命令见 [测试指南](../tests/README.md)。inprocess 真实 Chromium **6 项通过**：4 项更新循环测试 + 2 项新增 CPU 下游 / 关闭可视化；Event 阻塞核对真实红 / 绿 / 蓝各轮、下游 255、生成 seed=0/1/2 无重跑，以及阈值 256 的失败末轮。最终清理后另重跑两项关键下游 / 可视化浏览器用例，均通过。
- 回归修正：旧“不含任何图片”断言改为仅保留当前输入最新快照，逐轮评分历史仍无图片；Windows 临时目录短 / 长路径别名按实际文件身份比较，保留像素、模式及路径安全断言，未改变生产路径规则。关闭可视化任务不接收快照，任务输出明确提示本次未采集，避免旧图冒充当前结果。
- 验证边界：未运行完整 Python 或浏览器套件，部分其他模块缺 diffusers；未真实 SAM / FLUX 推理，模型质量未测。测试 agent 已通过工具将 scikit-image、Playwright 及 Chromium 装入已选环境，无模型下载。节点内 GPU 调用不能强杀，不支持重叠 / 嵌套 / 反馈。
- 工作区与服务：保留用户启动脚本及 [模块入口](../object_inpainting_studio/__main__.py) 自有改动，不覆盖；未重启用户业务服务，临时测试服务器已关闭，未修改用户工作流 JSON。
- 收尾检查：相关 Python / JS / HTML / CSS 和测试文件编辑器诊断无错误；三个前端模块语法及 `git diff --check` 通过，五份文档 **124 个本地链接**有效。核对 30 份改动文本并补齐新增测试末尾换行；用户原启动脚本无末尾换行，未擅自修改。不沿用历史结果或宣称完整测试套件通过。
- 下一步：下一项业务任务未指定。使用新功能需刷新 Studio；若后端仍为旧进程，需重启服务。

## 前次循环预览（简短历史）

- 2026-11-10：先实现无保存独立循环预览与 CPU 演示；当时前端 124 项通过，Python / 新浏览器未执行。此为前次状态，已由上述本轮 140 / 222 / 6 项验证记录更新。
- 更早向导曾暴露高级设置、共享保存选择与手动计划检查，普通中间输出曾要求附加 end；均不是当前 UI / 执行规则。

## 前次删除旧循环界面

- 更新日期：2026-11-10。
- 本轮目标：删除旧版本条件循环设置，只保留新的多循环卡片界面。
- 已完成：删除旧启用 checkbox / 兼容入口 / flat 配置 DOM、旧事件、旧 picker / renderer 和无用样式；[loop.js](../object_inpainting_studio/static/loop.js) 保留文件夹与共享请求 / 点选逻辑，[multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js) 为唯一条件配置界面。导入 / 恢复 / 直接赋值旧 flat 时自动一次性深拷贝为 loops，保留原参数、仅补缺省，并从旧体的保存来源及条件推导 end；缺失引用和非法值保留提示修正，不猜保存目标，不改原对象 / 用户文件。坏原始配置需明确删除，不静默取消验收。后端 flat API 兼容保留，不暴露旧设置。
- 实际验证：全部前端 108 项通过，移除过时旧 UI 测试并将有价值的计划 / 网络 / 文件夹等测试转到新控制器；条件共享 21 项、多循环 39 项。真实 Chromium 5 项通过，旧配置迁移用例确认旧 DOM 不存在且条件 / 种子 / 限制保留，新向导仍失败一轮后达标保存，并回归多循环 / 动态输出 / 历史缓存。`node --check`、浏览器测试 Python 语法、相关编辑器诊断、`git diff --check`、文档本地链接和修改文件 EOF 检查均通过。
- 验证边界：后端未修改，本轮未重跑非浏览器 Python（前轮 293 项通过）；未运行完整浏览器套件或真实 SAM / FLUX，未重启服务，保留原有未提交修改。
- 下一步：刷新 Studio，条件区只显示“＋ 添加循环”和新卡片；旧文件直接导入自动转换。下一项业务任务未指定。

## 前次多循环向导

- 更新日期：2026-11-10。
- 本轮目标：用“添加循环 → 点 start → 点 end → 点 condition”替代繁杂单循环启用配置，支持多个独立或串联循环。
- 已完成：[multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js) 提供独立循环卡片、点击式顺序向导、命名删除重选、单标量条件标题直选 / 多标量点端口、颜色角色定位、高级多起终点 / 条件 / 重试选项；主入口不需先启用，旧开关收到兼容入口。起点非负 INT seed 自动递增，-1 保持随机不改用户参数。最终保存独立共享且唯一时自动选择，删最后循环恢复普通保存。旧 flat 仍可运行，添加新循环时有效旧配置保留后迁移，无效旧配置明确提示。
- 后端：[conditional_loop.py](../object_inpainting_studio/conditional_loop.py) 增加 `MultiConditionalPlan` / 工厂，配置 `conditional_loop={loops:[{id,name,start_nodes,end_nodes,seed_nodes,conditions,...}]}`，支持 1～32 区域；body 取 start 后继与 end / 条件祖先交集，完整定义校验逃逸，按含桥接节点的传递依赖排序，拒绝重叠 / 嵌套 / 交叉依赖。各循环通过时强制算完 end 并隔离保留验收缓存；普通 bridge 不缓存未验收依赖。所选保存可读取验收体内中间图，其他普通外部中间输出须声明 end；未选保存分支不执行，执行切片不允许其他副作用。所有循环通过后只保存一次，任意失败 / 超时不写盘；历史和任务进度带循环身份，文件夹每图重置。
- 实际验证：全部非浏览器 Python 293 项（新增多循环 47 项）、全部前端 123 项（新增多循环 29 项）、真实 Chromium 5 项通过。新增浏览器完整验证两循环标题向导、评分即 end、各失败一次后通过、保存真实 PNG 蓝像素、无重复生成、报告、焦点、下载导入和删除。语法 / 诊断 / 差异检查见本轮实际结果；模型测试为 CPU 替代输出，无真实质量结论。
- 用户流程核对：只在内存给工作流 (8) 加入 start=`fluxrandomrecolor_2` / end=`colorsimilarity` / condition=`score <= 10`，实际 prepare_conditional 检查通过，体恰为换色 / 自动分色 / 色块 / 色差四节点；保留 -1 随机种子。未改用户 JSON、未推理或写业务图片。
- 验证边界：未运行完整浏览器套件，未运行真实 SAM / FLUX，未重启服务，保留原有未提交修改；不支持嵌套 / 反馈 / 历史最优 / 持久恢复，取消仍是节点边界。只有一个公用最终保存目标，本次不运行未选分支。
- 下一步：重启 Studio 并刷新，点“＋ 添加循环”后按提示依次点标题，再按需添加下一循环；设阈值、检查计划后运行。下一项业务任务未指定。

## 前次单循环实例点选

- 更新日期：2026-11-10。
- 本轮目标：同类节点多次添加时，可按画布中的具体实例灵活指定循环起点、种子、保存终点和条件输出，避免仅靠编号下拉选择。
- 已完成：[loop.js](../object_inpainting_studio/static/loop.js) 提供画布点选按钮、起点 / 种子连续增删、保存终点 / 标量条件端口单选、Esc 退出、已选项定位 / 移除；下拉显示 ID / 中文名 / 类型。使用实例 ID 和输出名绑定，不按类型或坐标匹配，移动节点不影响配置。角色 / 相关循环体 / 可选节点与端口高亮；验收位置仍是评分输出，保存终点仍须 SaveImages，不扩展任意回边或嵌套语义。
- 交互保护：[app.js](../object_inpainting_studio/static/app.js) 标题 / 端口事件优先消费点选，不拖动标题、不创建连线、不修改输入端口。进入点选清除旧待连线，改选使计划失效但保留会话和历史预览；检查 / 运行 / 状态未知期间禁止配置，恢复 / 禁用时退出模式；Esc 结束模式而不撤销已选。持久化沿用原格式，未修改用户工作流。
- 实际验证：全部非浏览器 Python 246 项、前端 JS 94 项（条件面板 46 项，新增 26 项），真实 Chromium 3 项（新增重复类型实例点选 / 端口 / 定位 / 配置往返，加原达标保存和动态色块用例）通过。浏览器测试语法及相关编辑器诊断通过；模型相关用例仍用 CPU 替代输出，无真实模型质量结论。
- 验证边界：未运行完整浏览器套件，未运行真实 SAM / FLUX，未重启服务，保留原有未提交修改。曾有一条完整回归无法取回输出，随后重跑取得 246 项通过的明确结果。
- 下一步：刷新 Studio，在条件面板点“画布增删循环起点”等，再点击具体节点标题；条件点输出圆点，选好后检查计划再运行。若服务仍是旧代码需重启；下一项业务任务未指定。

## 前次动态独立色块输出

- 更新日期：2026-11-10。
- 本轮目标：色块节点参照动态保存项，按 index 列表提供任意数量的独立图片输出，不再要求连接列表。
- 已完成：[色块节点](../object_inpainting/nodes/color_region_swatches.py) 使用直接填写的 `indices`（默认 `[0]`，0-based 非空非负整数列表）生成 `img1`～`imgN: RGB_IMAGE` 和 `colors`；如 `[0,2,1]` 对应三个可分别连接评分 / 保存的 RGB 32×32 色块。按位置命名，保留顺序 / 重复，越界仍明确失败；输出数量在执行前确定，不支持连线 / context 索引。
- 执行器：[基类](../object_inpainting/nodes/base.py) 增加 `output_types(spec)`，[图执行器](../object_inpainting/graph.py) 保存实例 `output_types`，先完成全部节点归一化再检查连线，使用同一表校验图输出和实际返回集合；[条件循环](../object_inpainting_studio/conditional_loop.py) 同步读取实例输出。TypedDict 保持默认单口模板。当时提供的旧 `images` 列表兼容和迁移现已删除；当前只支持独立 `imgN` / `colors`，旧连线须手动改接，本轮已清理仓库工作流中的残留设置。
- 界面：[app.js](../object_inpainting_studio/static/app.js) 的 `nodeInfo()` 提供实例输出，卡片 / inspector 改索引后即时重绘端口；缩口清除失效下游输入 / 工作流输出，保留存续连线；重排改变其内容。受运行 / 检查 / 状态未知锁保护，无效编辑不改图，不能将输出或 context 接入 indices。普通编辑继续保留预览会话及历史图片。帮助和 [颜色指南](../object_inpainting_studio/COLOR_REGIONS.zh-CN.md) 已改为独立口用法。
- 实际验证：项目 Python 3.12.8 虚拟环境、`PYTHONDONTWRITEBYTECODE=1`、项目 / tests 加入 PYTHONPATH、仅 GPU 1 可见；全部非浏览器 Python 246 项通过（新增动态输出 15 项），前端 68 项通过（新增动态色块 13 项），真实 Chromium 4 项通过（新动态口编辑 / 单图评分保存 / 缩口 + 旧列表 + 条件重试 + 生成后保存缓存）。动态用例使用真实 CPU 分色与 ΔE00 / PNG 像素验证；含模型的回归仍使用替代输出。相关编辑器 / 语法、差异空白和新增文件换行检查通过。
- 验证边界：未运行完整浏览器套件，未加载真实 SAM / FLUX 权重，未重启用户服务，未修改用户流程 / 图片，保留原有未提交修改。新独立 Python `execute()` 接口需按 `imgN` 取图；只有旧工作流图引用提供列表兼容。
- 下一步：重启 Studio 并刷新，在色块节点的 `indices` 填如 `[0,2,1]`，按 Enter 后连接各 `imgN`。下一项业务任务未指定。

## 前次单层条件循环

- 更新日期：2026-11-09。
- 本轮目标：按 ComfyUI 对照建议落地第一阶段单层条件重试，先生成、评分，达标才保存；支持当前单图和文件夹外层遍历。
- 已完成：[条件执行器](../object_inpainting_studio/conditional_loop.py) 实现显式起点 / 相关后继范围、all / any 同轮条件、数字与布尔比较、1～1000 次上限、协作超时、fail / skip、逐轮递增种子、固定输入复用及可变图片隔离。保存消费达标轮缓存，不额外生成；未达标不写盘，节点异常不重试。校验其他副作用、循环输出逃逸和非法条件 / 种子；普通 DAG / CLI 不允许静默跳过条件配置。
- 后台与界面：[服务端](../object_inpainting_studio/server.py) 新增条件计划 / 单图启动，文件夹合并评分分支；[后台任务](../object_inpainting_studio/batch.py) 共用单任务互斥、查询 / 恢复 / 取消，记录每轮评分 / 种子 / 停止原因与跳过数。启用条件后文件夹试跑也为后台任务，沿用过滤后的 index。[loop.js](../object_inpainting_studio/static/loop.js) 条件面板支持多起点 / 种子、条件编辑、检查和运行；[app.js](../object_inpainting_studio/static/app.js) 共用任务提交待确认锁，普通保存防绕过，保留历史预览。工作流下载 / 导入保留配置，轮询不重建聚焦控件。
- 本轮复现与修正：新增回归先发现循环体原地修改 PIL 图片后才拍摄固定缓存，污染后续轮；现固定输出交给下游前隔离。另修正文件夹条件计划必须与解码上下文共享同一图对象，以及条件试跑必须使用批量过滤索引，不能走未过滤单图入口。
- 实际验证：项目 Python 3.12.8 虚拟环境、`PYTHONDONTWRITEBYTECODE=1`、项目 / tests 加入 PYTHONPATH、仅 GPU 1 可见；全部非浏览器 Python 231 项通过（新增条件 26 项），JS 55 项通过（新增条件 20 项），真实 Chromium 4 项通过（条件控件配置 / 达标保存 + 原缓存 / 历史预览 / 图片放大 3 项）。条件浏览器用例以 CPU 图片替代 FLUX，计划无推理 / 写盘，红图失败后蓝图通过，仅存蓝图。Python 语法、签名调用兼容、编辑器诊断、JS 语法与差异空白检查通过。
- 验证边界：未运行完整浏览器套件，未加载真实 SAM / FLUX 权重，未重启用户服务，未修改用户流程 / 图片。保留工作区原有未提交修改。仅单服务进程任务 / 缓存，不提供持久续跑；取消 / 超时不能打断节点内 GPU 调用。暂不实现嵌套条件、上一轮反馈、历史最优、去噪步骤级取消或完整动态子图调度。
- 下一步：重启 Studio 并刷新，展开“条件重试 · 单层循环”，选择生成起点 / 递增种子 / 评分条件 / 保存终点，检查计划后先跑当前张；文件夹沿用试跑 / 运行全部。下一项业务任务未指定。

## 前次预览缓存与常驻预览

- 2026-11-09：跨请求预览保存真实输出，普通编辑保留 `previewSession`，结束 / 失败后刷新空闲 TTL；显式测试目标重跑、保存每次执行。[预览缓存](../object_inpainting_studio/preview_cache.py) 按输入 / 文件 / 祖先版本失效，深拷贝隔离，最多 8 会话 × 64 节点，15 分钟空闲过期。最近成功预览保留，编辑仅标历史状态，手动隐藏 / 显示跨重绘保留。
- 当轮验证：205 项非浏览器 Python、35 项 JS、3 项 Chromium 通过；用户工作流 (6) 以 CPU 模型替代输出验证 SAM / FLUX 各执行一次、保存实际写出 5 张 PNG。未验证真实权重，未重启用户服务。

## 前次动态保存项

- 更新日期：2026-11-07。
- 本轮目标：保存节点不固定限制为八口，改为可增删的动态图片项，并提供逐项名称与可选整组文件名。
- 已完成：[SaveImagesNode](../object_inpainting/nodes/save_images.py) 默认四项，可增加到 8 / 12 或更多；每项非输入元数据 `save_items` 保存稳定 `imgN` ID 和目录名，`save_next_id` 避免删项编号复用。`filename: STRING` 默认空共享 UUID，可连接文件夹原文件名，去掉最后一个扩展名后统一 PNG；`paths` 仍以原 ID 为键。原 `images` 列表保留自动 `images_N` 目录。
- 兼容 / 架构：[Node 基类](../object_inpainting/nodes/base.py) 增加实例端口及执行钩子，[NodeGraph](../object_inpainting/graph.py) 的 `input_types[node_id]` 统一校验集合 / 连线 / 运行时类型，[Studio](../object_inpainting_studio/server.py) 上传按实例端口解码。静态节点保持原规则；旧保存图自动深拷贝补齐四项及默认组名，显式删项不复活，归一化幂等；SAM3 独立膨胀迁移保持兼容。
- Studio：[app.js](../object_inpainting_studio/static/app.js) 卡片提供「＋添加图片」、逐项名称输入和 ×；右侧也能改名，不增加命名连线口。增删改名清理结果 / loop 计划，预览 / 运行 / 状态检查时禁止修改，结束后控件解锁；工作流下载 / 导入保留名称和编号。详见 [使用指南](../object_inpainting_studio/USAGE.zh-CN.md)。
- 安全：图片先验证编码，整组目标先检查重名、非法路径、符号链接子目录和已存在文件；不覆盖、不自动换名。OSError 回滚本轮新增文件，空目录可能残留；固定组名重跑和不同扩展名的同主干输入会报冲突，不自动续跑。
- 实际验证：Python 3.12.8，`PYTHONDONTWRITEBYTECODE=1`、仅 GPU 1 可见、项目 / tests 在 PYTHONPATH；`-m unittest` 显式运行全部 16 个非浏览器模块，187 项通过；`node --test tests/test_port_names.js tests/test_node_editing.js` 29 项通过。保存 16 项和批量新增用例验证实际 CPU 文件写入、12 路 / 模式 / 中文名 / 上传 API / 原文件名 / 冲突 / 回滚，前端 8 个新增用例调用真实函数。
- 静态验证：相关编辑器诊断无错误，保存节点 Pylance 无诊断；新增保存测试 Pylance 语法通过，全部编辑 Python 模块已由实际测试导入。JS 语法与 `git diff --check` 通过；21 个本轮相关文件空白 / 末尾换行及 6 份文档的 78 个本地链接检查通过，补齐了三个前端文件原先缺失的末尾换行。
- 验证边界 / 阻塞：尝试新增 Chromium 端到端用例，启动时缺少 `libnspr4.so`，0 项执行，未证明真实浏览器交互通过；未加载真实 SAM / FLUX 权重。没有安装浏览器系统依赖，也未替用户重启正在使用的 Studio。Git 基线已有大量历史未提交代码 / 文档 / 字节码，本轮未清理或覆盖无关改动。
- 下一步：重启 Studio 并刷新，旧图可重导入；连接更多图片，填写各项保存目录名及可选组名。浏览器系统依赖和真实模型 422 仍是独立事项；下一项业务任务未指定。

## 前次独立 Mask 膨胀

- 2026-11-06：新增 [MaskDilationNode](../object_inpainting/nodes/mask_dilation.py)，默认 steps / kernel 为 1 / 3；SAM3 只保留分割阈值，旧内嵌膨胀图统一迁移到独立节点并重接 Mask / bbox。原数据集 ExpandMaskNode 不变。
- 当轮验证：176 项非浏览器 Python、21 项 JS 通过；CPU 膨胀实际执行，SAM3 使用替代模型，浏览器未执行。架构与建议见 [分支设置](../object_inpainting_studio/USAGE.zh-CN.md#sam3-mask-太小时的设置)。

## 前次 FLUX 精度修复

- 更新日期：2026-11-05。
- 本轮目标：修正 FLUX Klein 加载时被忽略的精度参数。
- 已完成：[FLUX 加载器](../object_inpainting/nodes/flux_klein_reference.py) 将 `dtype` 改为 `torch_dtype`；当前虚拟环境 Diffusers 0.39.0 的实际 `from_pretrained` 源码确认支持后者。新增 CPU FP32 / CUDA BF16、离线参数和缓存行为回归测试。
- 实际验证：新增测试修复前 CPU/CUDA 两个子用例失败，修复后 `unittest discover -s tests -p 'test_flux_klein_reference_node.py' -v` 5 项通过；仅物理 GPU 1 可见，使用替代模型、未加载真实权重。编辑器诊断无错误，`git diff --check` 通过。
- 当前阻塞：用户日志确认权重加载后推理 0 步返回 422，但尚缺响应 `detail`；不能认定精度修复已解决真实推理错误。未改变模型路径、GPU 设置或自动增加 offload。
- 下一步：重启 Studio 后重试，若失败获取 `/api/node/preview` 响应 `detail`。未执行真实推理验证。

## 前次 GPU 设置

- 更新日期：2026-11-05。
- 本轮目标：仅持久化通用 GPU 可见性设置。
- 已完成：新增 [.vscode/settings.json](../.vscode/settings.json)，仅设置 Linux 集成终端的 `CUDA_VISIBLE_DEVICES=1`；未添加模型目录、节点设备或其他参数。
- 实际验证：设置文件与交接文档的编辑器诊断无错误；未新建终端验证环境注入，未执行模型推理。
- 当前阻塞：此前节点与 loop 失败原因尚无响应 `detail`，不能认定已解决。
- 下一项业务任务：未指定。设置保存后需新建集成终端；物理 GPU 1 在受限进程内编号为 `cuda:0`。

## 前次启动脚本配置

- 更新日期：2026-11-04。
- 本轮目标：修正 Studio 启动脚本的 Python 与 HF 缓存设置。
- 已完成：默认解释器改为项目父目录的 `.venv/bin/python`，对应用户指定的激活环境；缓存默认根目录改为 `../../hf_cache`，保留外部缓存变量覆盖并统一 Hub 新旧变量；同步架构地图启动说明。
- 实际验证：`bash -n run_object_inpainting_studio.sh`、`git diff --check` 通过；通过 Bash 拦截 `exec` 验证默认解释器、缓存路径、参数传递、环境变量覆盖和旧 Hub 变量兼容，3 组检查通过。未执行 Python、启动服务或运行模型推理。
- 下一项业务任务：未指定。
- 当前阻塞：无。

## 上轮交接

- 更新日期：2026-10-27。
- 本轮目标：补充根目录 README，将分层笔记、跨对话恢复和迭代维护方法写入项目说明。
- 已完成：新增 [README](../README.md)，链接项目入口与已有文档，说明三层笔记的职责、恢复顺序、可复用提示语和防膨胀原则；保留此前核实的架构地图。
- 业务改动：无；未修改 Python、前端、工作流或依赖。
- 下一项业务任务：用户尚未指定；不要从前一轮泛化摘要推断存在正在进行的 bug/feature。
- 当前阻塞：没有已复现的业务故障；运行环境、真实模型可用性及测试通过情况尚未验证。
- 建笔记前 Git 基线：`de4bd1e`（`first commit`），`git status --short` 无输出；这是历史快照，恢复时需重新检查。
- 验证边界：本轮只核对文档路径、变更范围和格式；未运行业务测试、浏览器测试或模型推理，不宣称它们通过。
- 上轮文档验证：3 份文档、41 个本地链接检查通过，编辑器诊断无错误；这不是本轮 README 的验证记录。
- 本轮文档验证：Perl 检查 4 份文档通过，55 个本地链接可解析且无行尾空白；`git diff --check` 无输出（未跟踪的新文件由前述显式检查覆盖）；Git 状态仅新增协作入口、README 和 docs 目录。未运行业务测试。
- 恢复顺序：读取本节 → 获取新需求 → 检查当前 Git 差异 → 只读架构地图相关章节及源码 → 选择相关测试。

## 关键纠偏

- 之前对话只看到了薄 CLI 入口，摘要不完整；此目录还包含 Studio、文件夹循环、模型节点和浏览器测试，不能限定为“只关注 CLI”。
- “继续当前 bug/fix/feature”此前只是占位描述，并非已确认任务。
- 现有测试文档中的命令是运行入口，不是成功记录；模型替代输出测试与真实模型验证分开记录。

## 后续记录格式

每次迭代替换“当前交接”，仅将必要决策/踩坑追加到下方；不要把整段对话或日志复制进来。

- 日期 / 目标：
- 改动文件与关键行为：
- 决策 / 原因 / 不应破坏的约束：
- 实际验证：命令、解释器/环境、通过/失败/跳过及简短结果（未运行就写未运行）。
- 未解决问题 / 下一步：

## 最近关键迭代

- 2026-11-06：先给 SAM3 增加膨胀及阈值，166 项 Python / 20 项前端测试通过；随后按用户要求将膨胀移到独立节点（见当前交接），原内嵌膨胀不再保留。
- 2026-11-05：全部 26 节点接入 TypedDict 输入输出合同与显式 execute 注解，合同自动生成端口表，Studio 显示类名；158 项 Python、19 项前端函数测试通过。PIL 语义类型不包装图像；仅修正已复现的旧错误报告断言，不改变业务错误逻辑。
- 2026-11-05：统一端口显示名及中文用途，连线时推荐同语义端口但不阻断合法同类型连接；保留原字段。132 项后端测试、18 项前端函数测试通过；非浏览器端到端验证。
- 2026-11-05：新增独立 32×32 代表色色块节点，0-based 排名列表选色，图片列表接保存节点；83 项后端测试与 10 项 JS 回归通过。新增浏览器用例仍因系统库缺失未执行，曾修正此前浏览器测试缩进；不证明真实模型或浏览器通过。
- 2026-11-05：增加安全节点删除；节点 / 右侧按钮和 Delete 快捷键，断开下游并清理结果与 loop，运行中禁删，不删除磁盘文件。10 项 JS 控制函数 / 单元测试与 18 项 Studio 测试通过；浏览器未通过系统依赖启动检查。
- 2026-11-05：补齐节点错误来源、API / 批量结构化错误与终端堆栈，67 项后端测试通过；补装缺少的 scikit-learn，15 项分色测试通过。Playwright / Chromium 已安装，但浏览器仍缺系统库。
- 2026-10-27：新增根目录 README，将跨对话笔记技巧写成人可直接使用的说明；未改变原有三层笔记职责。
- 2026-10-27：建立分层持久笔记。选择简短项目入口 + 按需参考文档，而不是将全部架构设为每轮自动加载的长指令。
