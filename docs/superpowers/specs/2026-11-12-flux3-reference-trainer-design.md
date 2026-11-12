# FLUX.3 换色训练入口：按参考脚本重整

状态：用户已确认实施（2026-11-12）。

## 目标与已确认事项

用户希望尽量具有 `diffusers/examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py` 的训练功能，明确选择“贴近参考脚本重新整理训练入口”。采用参考脚本的参数组织、Dataset/collate/sampler、Accelerator 初始化、模型加载、PEFT adapter、optimizer/scheduler、DataLoader、预编码、prepare、accumulate、checkpoint、验证及最终保存流程。

保持项目根目录的独立 `train_flux3_recolor_lora.py`，不改 Diffusers 原脚本。输入仍为 `test/经过涂灰的原图AB`、`test/生成图B`、`test/主颜色A`，目标为 `test/原图A`；完整文件名配对及默认381/42留出语义保持。继续只读本地完整 Diffusers 格式、非蒸馏 flow_map=False FLUX.3 权重，不自动下载或执行真实训练。

## 方案比较

- 扩充当前模块化简化 trainer：接口变化少，但继续远离参考组织；用户未选择。
- 参考脚本的完整训练框架加任务适配：用户已选择。代码量会明显增加，但关键流程和参数可逐段对照。
- 运行时导入/monkey patch 原脚本：与其全局 args 和内部缓存耦合，不采用。

## 文件与边界

- `train_flux3_recolor_lora.py` 从薄入口变为主要训练文件，保留参考许可证及来源说明，尽量沿用参考函数名、段落和参数名。
- `flux3_recolor/data.py` 继续负责严格配对、完整数据检查、来源分组拆分和共享场景/色块预处理；与参考 Dataset/分桶采样接合，不引入另一套文件匹配规则。
- `flux3_recolor/modeling.py`、`config.py`、`inference.py` 只保留训练与独立推理真正共用的功能。原 `training.py` 不再保留第二套独立训练循环；可保留调用新入口的兼容转发，不使用运行时补丁。
- 更新相关测试、中文说明、导航及交接，提供单卡和 `accelerate launch` 多卡命令。

## 功能范围

### 必须完整接通

1. Accelerate：ProjectConfiguration、DDP、原生 `accelerator.accumulate`、梯度累积、mixed_precision、prepare(model/optimizer/dataloader/lr_scheduler)、主进程写盘与同步、分布式指标聚合、正确结束 trackers。不能再拒绝所有多进程。
2. LoRA：rank/alpha/dropout、自定义 `lora_layers`、FP32 可训练参数、标准 safetensors 与完整 adapter metadata、正确重载与 resume；基础模型始终冻结。
3. 优化：参考支持的 lr_scheduler/warmup/cycles/power、scale_lr、AdamW beta/epsilon/weight_decay、max_grad_norm；epoch 与 max_train_steps 两种控制。scheduler 按真实优化步推进，FP16 skipped step 不应计作成功更新。
4. 数据：真实 DataLoader、workers、repeats、batch>1、按所有待堆叠图片的尺寸分组；显式/自动宽高比分桶、同步场景 crop/flip，颜色色块独立缩放、不做颜色增强。
5. 时间采样：weighting_scheme、logit_mean/std、mode_scale，遵循参考 density 与 loss weighting，目标依然只在生成图 token 上计算。
6. 显存：gradient_checkpointing、cache_latents、VAE/text_encoder offload、allow_tf32。缓存以样本身份及确定的增强版本读取，包含目标和全部三条件；不能按训练 step 取另一张图的缓存。缓存与增强语义必须明确，不能静默冻结本应逐轮改变的增强。
7. 文本：保留固定 instance_prompt；支持可选本地“完整文件名→caption”的 JSON，caption_dropout 必须保持各样本原窗口长度与 padding position-id 约定。不同文本长度 batch 的 padding 与推理约定保持参考行为。
8. 管理：checkpointing_steps、checkpoints_total_limit、latest/显式 checkpoint resume、优化器/LR scheduler/scaler/RNG/当前 epoch 与 batch 位置恢复。清理只限本轮明确识别的完成 checkpoint，绝不删除任意前缀目录或输入。
9. 记录与验证：report_to（none/tensorboard/wandb 等参考 tracker）、logging_dir、训练 loss/lr、validation_epochs 与可选 validation_steps、validation_prompt/final_validation_prompt、num_validation_images、skip_final_inference。图像结果落盘，同时支持 tracker 图像记录；主进程验证、其他进程等待且下一轮能继续。验证 scheduler 与训练 scheduler 独立。

### 参考中的可选高级能力

保留并适配参考已有的 bitsandbytes 量化加载、8bit Adam、Prodigy、torchao FP8、Accelerate FSDP 与 text-encoder FSDP 路径；仅在依赖/硬件存在且组合受支持时启用。互斥或不支持的组合启动时明确拒绝，不提供无实际行为的旗标。高级能力必须列出“CPU可验证接口/尚未真实硬件验证”的边界，不能把保留参考代码视为已验证。

本次不添加 Hub 上传/自动模型或数据下载、其他任务的 prior-preservation、Studio 接入。参考中的联网数据/发布选项不作为本地换色功能同等性的要求。

## 任务相关数据与模型差异

三条件分别 VAE 编码/patchify/BN，按固定顺序拼接 token，L=-1/-2/-3，目标 L=0。scene 条件和目标保持相同几何，色块单独处理。默认保持现有等比面积预算与16网格处理；显式桶可选择尺寸，需同时满足 FLUX.3 多参考预算。训练与推理记录并复用实际预处理策略，不能训练时改成方图而推理仍沿用旧面积规则。

batch>1 时不能直接堆叠不同尺寸的场景或不同尺寸的色块。采样键至少覆盖目标/两场景/色块的最终尺寸；各进程获得相同数量优化更新，epoch 末尾不足累积长度时 loss 归一化需正确，不静默丢完小桶导致空 loader。

默认指令沿用现有英文指令。新增任务参数优先用 `--instance_data_dir`、`--instance_prompt`；现有 `--dataset_root`、`--prompt` 可作为同义别名，冲突时报错。模型路径仍是 `--pretrained_model_name_or_path`。

## 恢复和兼容性

新训练状态增加格式版本；旧手动累积循环的 checkpoint 不假装与新 DataLoader/原生累积状态兼容。当前没有真实用户训练权重，实施时可以明确拒绝旧训练状态、要求新目录；旧标准 LoRA 和推理配置的读取不应无故损坏。

记录数据指纹、split、bucket/augmentation/cache/text 设置、world_size、batch/accumulation、optimizer/scheduler 及训练 horizon。恢复默认要求相同 world_size 和影响轨迹的配置；只有恒定学习率等可证明不会改变已有轨迹的情形允许延长训练 horizon，其他 scheduler 不能静默延长并声称严格等价。日志/保存频率可调整。

## 验证要求与边界

- CPU 不下载：数据/分桶/多batch、三条件输入、缓存打乱不串样本、caption窗口/dropout、LoRA基础冻结/alpha元数据、不同LR轨迹及恢复、checkpoint保留边界、完整随机小模型CLI训练→验证→重载推理。
- 两进程 CPU Accelerate/DDP：样本分片、原生梯度累积、参数同步、仅主进程产物、checkpoint恢复及验证后继续训练。若环境不支持分布式启动，报告具体阻塞，不能宣称通过。
- 真实 GPU/BF16/FP16、真实 FLUX.3、FSDP、量化/FP8/8bit Adam/Prodigy 与换色质量只在具备对应条件时验证；没有条件时保留未执行标识，不自动安装大型可选依赖或下载权重。
- 为用户提供“与参考功能对应表”，每项标明支持、明确限制及本轮验证证据。

## 实施者可自行决定

内部缓存容器、局部 helper 的拆分、测试夹具和日志实现细节。沿用参考默认值为主；batch/resolution 等资源相关默认仍保守，说明与参考不同之处。实现过程中若发现高级组合与三参考数据合同无法兼容，应明确报错及记录限制，不能静默降级。
