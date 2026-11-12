# 迭代交接笔记

只保留恢复工作需要的信息；稳定架构放 [架构地图](PROJECT_CONTEXT.md)，入口与维护约定见 [AGENTS.md](../AGENTS.md)。

## 当前交接

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
