# FLUX.3 三参考图换色 LoRA

独立于 Studio 与数据集构建 CLI。训练入口是 [train_flux3_recolor_lora.py](train_flux3_recolor_lora.py)，推理入口是 [infer_flux3_recolor_lora.py](infer_flux3_recolor_lora.py)。不修改 `diffusers/examples/dreambooth/` 的原示例。

## 数据与任务

默认读取 `./test`，按完整文件名配成四元组：

| 角色 | 目录 | 用途 |
| --- | --- | --- |
| 条件 1 | `经过涂灰的原图AB` | 编辑位置、人物与背景布局 |
| 条件 2 | `生成图B` | 服装结构、纹理和光照 |
| 条件 3 | `主颜色A` | 最终颜色色块 |
| 监督目标 | `原图A` | 还原 A 色服装的结果 |

固定条件顺序对应 FLUX.3 的参考图位置编码 L=-1/-2/-3。目标 latent 加噪，条件 latent 不加噪；只对目标计算标准全图 flow matching velocity loss。冻结 VAE、文本编码器及基础 Transformer，仅优化 attention/FFN 的 LoRA（包括 image_cond 分支）。初版不使用 mask 或区域加权。

默认指令可用 `--prompt` 覆盖，训练配置会保存最终值：

> Use image 1 as the base image. Recolor the gray clothing region using the color swatch in image 3. Use image 2 as a reference for garment structure, texture, and shading. Preserve the person, pose, garment details, and background.

## 环境与本地模型

使用支持 `Flux3Transformer2DModel`、`Flux3AutoBlocks` 和 `Flux3LoraLoaderMixin` 的本地 Diffusers 源码，不能直接假定普通旧版 Diffusers 支持 FLUX.3。本轮 CPU 验证环境是 Python 3.12.8、Diffusers 0.42.0.dev0（本仓库 `diffusers/`）、torch 2.11.0+cu128、transformers 5.5.4、accelerate 1.15.0、peft 0.21.2、safetensors 0.8.0、Pillow 12.3.0、numpy 2.2.6。其他版本未验证。

若当前环境未指向本地源码，可在自己的虚拟环境中手动安装 `python -m pip install -e ./diffusers`，并按该源码的依赖要求配置 PyTorch / Transformers / PEFT / Accelerate。数据检查只需 Python 和 Pillow，不加载 torch 或模型。

权重由你手动下载。脚本只接受完整的 **Diffusers 格式、非蒸馏 FLUX.3 基础模型**，即 Transformer 配置的 `flow_map=False`；flow_map 蒸馏权重的训练目标不同，本入口会拒绝。可以直接提供模型目录、具体snapshot目录，或带 `refs/main` 的HF仓库缓存目录。目录至少含：

```text
your-flux3-model/
  transformer/    # config.json 和完整权重/分片
  vae/            # config.json 和完整权重/分片
  text_encoder/   # Qwen3-VL 配置和完整权重/分片；官方base同时在此放processor/tokenizer
  tokenizer/      # 可选：独立布局时放AutoProcessor配置，不要求官方模型另建此目录
```

不需要下载 prompt upsampler；本任务始终直接使用固定或显式指定的指令。所有模型加载均设置 `local_files_only=True`。VAE 必须对应 16 像素 token 网格及 2×2 latent packing，基础 Transformer 的文本输入维度必须匹配选用的 Qwen 隐藏层。默认隐藏层沿用参考示例 `[4,8,12,16,20,24,28,32]`，不同模型配置需要核对。

已核对用户下载的 `../../hf_cache/models--black-forest-labs--flux-3-image-klein-base`，`refs/main` 对应 `a83ce283b1daa2f3d782645ae7b06c8e73af1b41`。可直接将此缓存目录传给 `--pretrained_model_name_or_path`，入口会解析snapshot；训练与推理按 `modular_model_index.json` 读取processor子目录（此模型为 `text_encoder`），也兼容独立tokenizer目录。启动前检查权重索引中列出的分片是否存在。本轮仅检查本地配置、索引、safetensors头与processor，不代表真实权重加载或GPU训练已经通过。

## 先检查数据

在项目根目录运行（`../.venv/bin/python` 可替换为自己的解释器）：

```bash
../.venv/bin/python train_flux3_recolor_lora.py \
  --check_dataset --dataset_root ./test
```

检查会全量解码图片、应用 EXIF 方向、转 RGB、核对文件名集合与三个场景图的尺寸，并验证有效预处理尺寸。任何缺失、损坏或尺寸不一致都会报具体文件名。默认 seed=42、约 10% 验证集；本轮实际 423 组全部通过，拆为 381/42。可加 `--output_dir ./outputs/recolor_data_check` 保存完整报告、像素指纹和拆分；报告目录应与正式训练输出分开。

默认按完整文件名主干分组，**不能据此确认不同文件名是否来自同一原图**。若有同源衍生样本，使用 `--groups_json groups.json`，JSON 为覆盖所有样本的“完整文件名 → 来源组 ID”映射，例如 `{"00000_00.png":"source_00000","00000_01.png":"source_00000"}`；同组不会跨训练/验证集。两个集合均必须非空。

## 与参考训练脚本的对应关系

主训练实现现位于根目录 `train_flux3_recolor_lora.py`，按参考文件的 parse_args → DreamBoothDataset/collate/BucketBatchSampler → Accelerator → 模型与PEFT → optimizer/LR scheduler → DataLoader/cache → prepare → accumulate → 保存/验证顺序组织。`flux3_recolor/training.py` 只做旧入口转发，不再保留另一套手动累积循环。

| 参考能力 | 本入口 | 验证或限制 |
| --- | --- | --- |
| Accelerate / 原生梯度累积 / DDP | 支持 `accelerate launch`，prepare四项训练组件；尾累积按实际样本数归一化 | CPU单进程与两rank验收；真实多GPU未验证 |
| LoRA | rank/alpha/dropout/lora_layers、冻结基础模型、标准safetensors及metadata | 小模型前反向、非默认alpha与重载已验证 |
| AdamW / scale_lr / 梯度裁剪 | beta/epsilon/weight_decay/max_grad_norm 均接通 | CPU实际优化；FSDP全局裁剪使用完整参数列表 |
| LR scheduler / warmup | constant/constant_with_warmup/linear/cosine/cosine_with_restarts/polynomial，cycles/power | CPU验证constant/cosine/warmup及恢复，scheduler按真实成功更新推进 |
| epoch / step | num_train_epochs 或 max_train_steps | 默认1个epoch；max_train_steps优先 |
| DataLoader / workers / batch / repeats | 支持多batch、重复数据与workers | 尾小桶保留；DDP按完整batch补齐，不混不同尺寸 |
| 宽高比分桶 | 显式桶或参考的11种比例自动桶 | 所有三参考尺寸参与采样键；多参考上限1MP |
| 几何增强 | 同步场景裁剪/翻转，按seed+epoch+sample确定 | 色块独立处理，无颜色增强；默认面积模式中心网格裁剪 |
| latent缓存 | 目标和三条件分别按sample identity缓存到每进程CPU内存 | 打乱、分片和恢复不按step索引；缓存禁止随机翻转，分桶要求center_crop |
| caption / caption dropout | 固定指令或本地caption_json，batch共享最大窗口 | dropout空指令保持原窗口和padding position IDs；自定义caption在线编码 |
| 时间采样 / loss weighting | none/sigma_sqrt/logit_normal/mode/cosmap及参考参数 | 保留FLUX.3 flow matching；只监督目标token |
| gradient_checkpointing / offload / TF32 | 支持；VAE/text_encoder可闲置回CPU | CPU验证offload路径；真实GPU显存/TF32未测 |
| checkpoint / resume / limit | 原生optimizer/LR/scaler/RNG与epoch/batch位置；只清理有本入口标识的完成checkpoint | 新状态version=2；旧手动累积checkpoint不兼容 |
| tracker | none/tensorboard/wandb/comet_ml/all | 显式tracker缺依赖会报错；W&B等外部服务未在验收时连接 |
| 验证 | validation_steps/validation_epochs、validation_prompt/final_validation_prompt、skip_final_inference | 主进程图像落盘/记录，其他rank同步；独立scheduler与隔离训练RNG |
| Prodigy | 可选prodigyopt，beta3/decouple/bias correction/warmup选项接通 | CPU优化与checkpoint恢复已验证；不支持FSDP组合 |
| 8bit Adam | 可选bitsandbytes | 需要CUDA；未真实GPU验证，不支持FSDP组合 |
| bitsandbytes量化DiT | bnb_quantization_config_path，按4bit/8bit配置加载 | 仅单GPU；不支持FSDP/FP8组合；未实测 |
| FP8 | do_fp8_training，参考torchao转换 | 需要CUDA及torchao；不支持量化/FSDP组合；当前未安装torchao、未实测 |
| FSDP1 / text encoder FSDP | 通过Accelerate配置，use_orig_params=true；fsdp_text_encoder可选 | 基础Transformer与LoRA以统一FP32 master包装，计算精度由FSDP控制；只允许普通AdamW，无真实GPU验收 |

本入口面向本地换色任务，不添加Hub上传/远程数据下载、prior preservation、NPU专用开关。默认 `report_to=tensorboard`、每100个优化步验证4张固定留出图；batch=1、面积预算512²。需要安装tensorboard；显式 `--report_to none` 只保留磁盘图像和JSONL，不写tracker。

## 训练

下载兼容基础模型后，将下面路径替换为自己的本地目录：

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/accelerate launch --num_processes 1 \
  train_flux3_recolor_lora.py \
  --pretrained_model_name_or_path /path/to/your-flux3-model \
  --instance_data_dir ./test --output_dir ./outputs/flux3_recolor_v2 \
  --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --rank 16 --lora_alpha 16 --learning_rate 1e-4 \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --max_train_steps 1000 --mixed_precision bf16 \
  --gradient_checkpointing --offload \
  --checkpointing_steps 100 --checkpoints_total_limit 3 \
  --num_validation_images 4 --validation_steps 100 \
  --report_to tensorboard
```

可继续用 `python train_flux3_recolor_lora.py ...` 单进程启动。`--dataset_root` 是 `--instance_data_dir` 的别名，`--prompt` 是 `--instance_prompt` 的别名；同时传入不同值会报错。第一轮建议另开新输出目录，先用少步数检查真实模型，再对小样本集合做过拟合检查。这些示例参数不代表显存或颜色质量已验证。

多GPU DDP例子（选择自己允许使用的设备）：

```bash
CUDA_VISIBLE_DEVICES=0,1 ../.venv/bin/accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  train_flux3_recolor_lora.py \
  --pretrained_model_name_or_path /path/to/your-flux3-model \
  --instance_data_dir ./test --output_dir ./outputs/flux3_recolor_ddp \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 1000 --gradient_checkpointing --offload \
  --cache_latents --report_to tensorboard
```

支持同一节点的DDP，输出与数据须对各rank可见。每卡batch为 `train_batch_size`；通常有效batch为每卡batch×进程数×累积次数，尾桶不足时按实际样本数计算。为使各rank更新次数相同，不能整除world_size时会重复完整batch；不会把其他桶的样本塞进尾batch，极小数据集可能跨rank重复。改变world_size会改变训练轨迹，因此恢复时要求保持。

FSDP使用你自己的Accelerate FSDP1配置：要求 `fsdp_version: 1`、`fsdp_use_orig_params: true`、可用CUDA。通过 `accelerate launch --config_file ...` 启动；如需分片文本编码器可加 `--fsdp_text_encoder`。FSDP2、DeepSpeed及不支持的高级组合会明确拒绝。FSDP检查点使用Accelerate原生完整/分片模型及optimizer格式，**会包含基础模型参数**，另附标准LoRA；其磁盘/内存成本与普通DDP的adapter-only检查点不同。FSDP验证会在主进程构建未分片模型，需额外显存，未做真实硬件测量。

## 尺寸、缓存与文本

默认 `resolution` 是面积预算平方根，保留比例：本数据768×1024在512预算下处理为432×576；1024预算保留原尺寸。每轴中心网格裁剪最多15px，32×32色块保持原尺寸。

`--use_aspect_ratio_buckets` 启用参考比例梯级，自动桶的尺寸/裁剪可能与默认面积模式不同。显式桶沿用参考格式，例如 `--aspect_ratio_buckets '512,768;768,512'`（H,W），也接受 `512x768,768x512`。宽高须为16倍数，每图面积不超过1MP。训练记录策略供推理复用；验证/推理使用确定性中心裁剪。分桶训练不加center_crop时使用按epoch确定的随机裁剪，`--random_flip` 同步翻转目标和两张场景条件。

`--cache_latents` 将每个sample/repeat的目标与三参考latent缓存到本进程CPU内存，不按batch序号绑定。它与随机翻转互斥，分桶时需 `--center_crop`；这样不会悄悄把逐轮随机增强冻结。默认不缓存，自定义caption在线编码；固定指令只编码一次。缓存会复制到每个rank，需预留CPU内存。

`--caption_json captions.json` 的格式是 `{"00000_00.png":"recolor red ...", ...}`，须覆盖数据文件名，空字符串回退到instance_prompt。`--caption_dropout 0.1` 只丢弃文本，不丢条件图；空文本的编码窗口与被丢指令一致。固定prompt的推理文本设置仍从训练配置读取。

## 输出与恢复

新训练使用新的空输出目录，不与数据/模型目录重叠。典型产物：

```text
outputs/flux3_recolor_v2/
  pytorch_lora_weights.safetensors
  recolor_config.json
  dataset_report.json
  split.json
  train_history.jsonl
  training_state.json
  checkpoint-100/
  validation-step-100/
  validation-epoch-1-step-.../
  validation-final-1000/
```

普通DDP检查点只保存LoRA及optimizer/LR/scaler/每rank RNG；FSDP原生状态额外含基础模型。完成后才写入version=2标识并成为latest候选。`checkpoints_total_limit` 只清理本入口格式且步数匹配的完成目录，忽略未知/未完成/符号链接目录。

同一步同时触发checkpoint和验证时，先提交checkpoint，再做验证推理；验证异常也能保留该保存步的训练状态。若旧版本已在保存前退出且目录只有配置/日志/loss，没有checkpoint，则无法从loss恢复模型权重，请保留日志并使用新输出目录重跑。

恢复例子：使用相同基础模型、数据、分组、caption、预处理、world_size、batch/accumulation、优化器和LR配置，附加 `--resume_from_checkpoint latest`（或当前输出目录内的完整checkpoint路径）。num_train_epochs与max_train_steps是总训练horizon，不是额外训练量。

**只有constant调度允许直接增加总训练步数**；cosine/linear等改变horizon会改变轨迹，恢复时会拒绝。日志/保存频率可调整；数据或关键配置改变需要新输出目录。模型权重分片不应被原地替换，模型校验检查本地路径及配置摘要，不逐分片哈希。

旧版本手动累积循环的训练checkpoint不能用于新入口恢复，请使用新目录。旧标准LoRA及旧推理配置仍可用于推理。

默认每个完成epoch和训练结束均使用4张留出样本验证（验证集不足4张时取全部）；`--validation_epochs N` 调频率，`--validation_steps N` 增加按步验证。上面的训练命令每100步保存4张固定样本，方便比较训练进展。`--skip_final_inference`（别名skip_final_validation）只跳过最终推理，`--num_validation_images 0` 关闭全部验证。validation_prompt覆盖中途/最终指令；仅final_validation_prompt覆盖最终指令。验证使用固定seed与独立scheduler，避免改变训练噪声时间表和RNG。

## 训练过程中实时查看验证图

训练循环会在step=100、200、300…调用验证，生成4张固定留出图并继续训练；可以用 `--validation_steps 10` 调整频率，短程检查用1。不是必须训练结束后再单独运行推理脚本。一次验证结束就flush事件，TensorBoard可在训练仍在进行时读取；验证推理期间训练会等待，完成后继续。

在另一个终端启动TensorBoard：

```bash
../.venv/bin/tensorboard --logdir ./outputs/flux3_recolor_smoke/logs --port 6006
```

在TensorBoard页面 `http://localhost:6006` 的 **Images** 查看：

- `validation/generated/<文件名>`：训练中各次验证的生成图，拖动step滑块比较。
- `validation/comparison/<文件名>`：涂灰AB / B图 / A色块 / 目标A / 输出的五列对照，同一文件名使用固定tag和真实global_step。
- `test/generated`、`test/comparison`：与参考脚本相同，最终验证单列为test阶段。
- **Scalars** 中 `loss`、`lr`：训练步数对应的损失和学习率。

若在远程服务器运行，需要将6006端口转发到本地。PNG历史仍保存于 `validation-step-100/`、`validation-step-200/` 等目录。不要传 `--num_validation_images 0`，否则会关闭验证。

只有输出目录中确实保存了本入口的完整checkpoint时，才能使用 `--resume_from_checkpoint latest`。首次训练或输出目录已不存在时，去掉此参数，从头训练；脚本不会把显式续训请求静默改成新训练。

若仍保留此前2步smoke的checkpoint，可按下面命令继续并开启训练中监控（保持原数据和其余关键参数不变）。若没有这些checkpoint，删除命令中的 `--resume_from_checkpoint latest`：

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python train_flux3_recolor_lora.py \
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --instance_data_dir ./test --output_dir ./outputs/flux3_recolor_smoke \
  --resume_from_checkpoint latest --max_train_steps 1000 \
  --mixed_precision bf16 --gradient_checkpointing --offload \
  --num_validation_images 4 --validation_steps 100 --report_to tensorboard
```

## 三图推理

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python infer_flux3_recolor_lora.py \
  --lora_dir ./outputs/flux3_recolor_v2 \
  --gray_image './test/经过涂灰的原图AB/00000_00.png' \
  --generated_b_image './test/生成图B/00000_00.png' \
  --color_a_image './test/主颜色A/00000_00.png' \
  --output_dir ./outputs/recolor_single \
  --seed 42 --num_inference_steps 28
```

默认从训练配置读取基础模型路径、prompt 和预处理参数；模型移到其他位置后可加 `--pretrained_model_name_or_path`，配置须一致。输出生成图、三个条件及对照图；单例对照从左至右为涂灰图/B图/A色块/生成图。可加 `--guidance_scale`（默认 3.5）或 `--prompt`。

生成保存清单中的留出验证集：

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python infer_flux3_recolor_lora.py \
  --lora_dir ./outputs/flux3_recolor_v2 --dataset_root ./test \
  --limit 4 --output_dir ./outputs/recolor_validation
```

默认读取 `lora_dir/split.json`，也可用 `--split_json` 指定。验证每张单独保存，五列对照为涂灰图/B图/A色块/目标A/生成图，另存 seed 与推理步数清单。用相同场景和 seed 替换色块，对比是否按颜色改变，并观察纹理、人物和背景；loss 下降本身不证明换色质量。

## 已验证与未验证

本轮34项离线CPU测试通过，另1项真实两进程CPU DDP验收通过。使用本地随机小型FLUX.3/VAE/Qwen3-VL/tokenizer，不下载权重；覆盖数据合同、三参考数学、LoRA、分桶、cache身份、caption/dropout、LR/warmup、workers、TensorBoard、Prodigy、恢复、主进程验证与标准推理。两进程检查参数同步、epoch中途恢复及尾累积；连续/恢复训练的最终adapter逐位一致。完整命令见[测试指南](tests/README.md)。本轮完整数据检查确认423组、381/42拆分。

用户完成2步smoke训练后，本轮已用该LoRA在GPU1/BF16下完成4张真实留出样本推理（每图28步），结果与五列对照在 `outputs/flux3_recolor_smoke_validation/`。第一张仍沿用B图颜色，不能据此宣称训练已收敛。真实基础权重加载/推理已执行，但没有启动额外训练或测峰值显存；FP16 scaler、FSDP运行、bitsandbytes/8bit Adam、FP8和完整训练质量仍需对应验收。未安装torchao或连接外部tracker。
