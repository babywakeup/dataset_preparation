# FLUX.3 三参考图换色 LoRA

独立于 Studio 与数据集构建 CLI。训练入口是 [train_flux3_recolor_lora.py](train_flux3_recolor_lora.py)，推理入口是 [infer_flux3_recolor_lora.py](infer_flux3_recolor_lora.py)。不修改 `diffusers/examples/dreambooth/` 的原示例。

训练采用固定精度：VAE直接从预训练权重以FP32加载，内部LayerNorm/GroupNorm、权重和forward均不替换；编码调用局部关闭autocast，验证/推理不对整个pipeline启用低精度autocast。Transformer按mixed_precision加载，并在自身前向使用一致的autocast上下文；模型原生保留FP32的模块遵循Diffusers默认行为。latent BN归一化和FP32 loss保留。

验证继续使用原生 FLUX3 modular pipeline。单卡/DDP 缓存 pipeline 并直接引用训练 Transformer，LoRA 更新立即可见，不保存/重载模型；scheduler 独立复制。验证临时绕过 Accelerate 的输出 FP32 包装，结束后恢复 forward、训练模式和 RNG。FSDP 仍需汇集 LoRA 并构建主进程验证副本，不能让其他 rank 等待时单独调用分片模型。

已删除precision_recipe参数、legacy版本分支和自定义norm类。训练配置只使用mixed_precision，不再选择精度recipe；旧recipe训练状态不能直接恢复，请用新目录训练。本轮精度结构简化不代表换色质量已经得到验证。

## 数据与任务

目标颜色仅由第三张色块图提供；训练、验证和推理的 prompt 不注入目标颜色名称、RGB 或色号。显式文字指定颜色的成功不能代替仅色块控制的验证。

默认读取 `./test`，分别按两个子集各自完整的文件名配对构建记录：

| 子数据集 | 条件1：涂灰定位 | 条件2：待改色源图 | 条件3：目标色块 | 监督目标 |
| --- | --- | --- | --- | --- |
| `b_to_a` | `经过涂灰的原图AB` | `生成图B` | `主颜色A` | `原图A` |
| `a_to_b` | `经过涂灰的原图AB` | `原图A` | `主颜色B` | `生成图B` |

新训练默认合并两个子集，同一源样本每个方向各提供一条记录，原始记录数比例1:1。
先构建并按比例合并记录，再按来源组划分训练与验证，同源记录不会跨集合泄漏。
`--repeats` 作用于两个方向；指定 `--max_train_steps` 时仍表示总优化步数，不会自动翻倍。
新增的 `主颜色B` 必须完整配对、可解码、尺寸至少16，并且为纯色色块；取样时检查能否解码及是否纯色。

原命令使用新的输出目录即可启用双向，也可显式写：

```bash
--training_directions b_to_a a_to_b --output_dir ./outputs/flux3_recolor_bidirectional
```

仅保留旧方向用 `--training_directions b_to_a`；仅训练反向用 `--training_directions a_to_b`。
旧单向checkpoint续训需显式选择 `b_to_a`，并保持其他配置/数据不变；不能直接把旧checkpoint切成双向。
`training_directions` 仅用于构建选择；训练与验证读取统一记录，不再根据方向分支。sample_id区分不同记录与repeat，防止latent身份混用。
合成B成为监督意味着其错误纹理也可能被学习，双向训练不保证解决新增印花问题。

固定条件顺序对应 FLUX.3 的参考图位置编码 L=-1/-2/-3。目标 latent 加噪，条件 latent 不加噪；只对目标计算标准全图 flow matching velocity loss。冻结 VAE、文本编码器及基础 Transformer，仅优化 attention/FFN 的 LoRA（包括 image_cond 分支）。初版不使用 mask 或区域加权。

默认指令可用 `--prompt` 覆盖，训练配置会保存最终值：

> Edit image 2, the photograph to be recolored. Image 1 is the corresponding photograph with only the editable clothing region covered in uniform gray. The region replaced by gray in image 1 compared with image 2 marks where to recolor; all other regions must stay unchanged, including naturally gray objects. Image 3 is the desired color swatch. Recolor only the marked clothing region to match image 3. Use image 2, not the gray fill in image 1, to preserve all original clothing details. Preserve the exact person's identity, pose, garment silhouette, seams, buttons, prints, logos, fabric weave, texture, folds, shadows, highlights, camera framing, and background. Change fabric color only, not material or geometry. Do not add objects, text, mask markings, or the swatch to the output. Return only the edited photograph.

此指令按实际三条件顺序描述：图1涂灰、图2待改色源图（B或A）、图3目标色块，不交换角色。两个方向共用同一prompt与可选变体；颜色以色块指定，不使用未提供的color_name变量。新训练默认使用该指令并保存到模型配置；已有模型按自己的配置推理。显式prompt或caption_json仍优先，同文件名的caption用于两个方向，应使用通用角色描述。改变prompt后需使用新训练目录而非直接resume旧配置。

训练统一记录 `loss`，不再记录分方向loss。
`--num_validation_images 4` 表示合并后总共4张生成图；按固定等距位置从整个验证序列选择，不按子集翻倍。
PNG保存在 `validation-step-N/0000/` 等数字目录，manifest记录对应sample_id、source_id和原始路径；TensorBoard图像tag也使用数字序号。

同一训练进程内，同一步且prompt、样本、seed和推理配置相同的step/epoch/final验证只生成一次，保留首次触发的目录与tracker记录。final prompt不同时仍单独生成；与此前验证重复的final不会额外写test图像。恢复训练不复用旧进程的去重记录，会重新出图；先写临时目录，完整生成成功后替换该步旧目录，生成失败保留旧图。独立推理仍拒绝覆盖非空输出目录。
数据集推理读取相同缓存中的validation记录，无需指定方向。
手动提供三张图片时，可用中性别名 `--source_image` / `--color_image`（兼容原 `--generated_b_image` / `--color_a_image`）。

## 训练指令变体

新实验可在原训练命令后添加 `--prompt_augmentation`，并使用新 `--output_dir`。
内置6种英文指令：详细约束、先解释输入、先指定颜色、先指定源图、局部编辑、输出导向。
三图顺序和任务含义完全一致，均要求只改标记服装颜色、匹配色块、保留纹理/人物/背景。
根据seed和样本标识确定起点，每个epoch轮换一种；重复样本可使用不同指令，恢复训练可复现。
指令一起提前编码到共同token长度，训练按样本取缓存，支持caption_dropout；
文本编码器仍可在训练前释放。验证和默认推理继续使用固定默认指令，便于跨步骤比较。
显式 `--prompt` / `--instance_prompt` 或 `--caption_json` 优先于内置变体；不添加开关保持单指令行为。
配置保存实际指令列表，续训检查其一致性。不要向已有单指令checkpoint直接开启此开关。

这项增强旨在减少对固定措辞的依赖，不增加图像或颜色信息；是否改善颜色准确度尚未验证。
建议以相同数据划分、seed、rank/alpha、学习率、有效batch和更新步数对照开关，使用固定彩色验证样本，
分别检查颜色匹配、纹理保留与非编辑区域变化；还可用未参与训练的改写指令检查泛化。

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

## 数据准备

训练入口会自动import并调用数据集构建函数，将路径与尺寸元数据保存到共享的hash缓存，然后加载模型训练；相同数据构建配置会跨训练输出目录复用。无需额外的数据检查开关，正常训练必须传入本地 `--pretrained_model_name_or_path`。

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
| caption / caption dropout | 固定指令或本地caption_json，当前batch共享最大窗口 | 自定义caption在线编码，dropout空文本匹配当前窗口 |
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
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --instance_data_dir ./test --output_dir ./outputs/flux3_recolor_v3 \
  --resolution 1024 --train_batch_size 1 \
  --rank 16 --lora_alpha 16 --learning_rate 1e-6 \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --max_train_steps 1000 --mixed_precision bf16 \
  --checkpointing_steps 100 --checkpoints_total_limit 3 \
  --num_validation_images 4 --validation_steps 100 \
  --training_directions b_to_a a_to_b \
  --report_to tensorboard \
  --gradient_checkpointing --offload
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
  --report_to tensorboard
```

支持同一节点的DDP，输出与数据须对各rank可见。每卡batch为 `train_batch_size`；通常有效batch为每卡batch×进程数×累积次数，尾桶不足时按实际样本数计算。为使各rank更新次数相同，不能整除world_size时会重复完整batch；不会把其他桶的样本塞进尾batch，极小数据集可能跨rank重复。改变world_size会改变训练轨迹，因此恢复时要求保持。

FSDP使用你自己的Accelerate FSDP1配置：要求 `fsdp_version: 1`、`fsdp_use_orig_params: true`、可用CUDA。通过 `accelerate launch --config_file ...` 启动；如需分片文本编码器可加 `--fsdp_text_encoder`。FSDP2、DeepSpeed及不支持的高级组合会明确拒绝。FSDP检查点使用Accelerate原生完整/分片模型及optimizer格式，**会包含基础模型参数**，另附标准LoRA；其磁盘/内存成本与普通DDP的adapter-only检查点不同。FSDP验证会在主进程构建未分片模型，需额外显存，未做真实硬件测量。

## 尺寸、缓存与文本

默认 `resolution` 是面积预算平方根，保留比例：本数据768×1024在512预算下处理为432×576；1024预算保留原尺寸。每轴中心网格裁剪最多15px。第三张颜色条件在dataset读取时独立以最近邻resize到 **128×128**，不受场景面积预算影响，纯色RGB保持不变；`test/主颜色A` 的32×32原始文件不改写。可用 `--color_swatch_size 128` 显式指定（默认128，允许16..1024的16倍数），如需尺寸对照可另开新目录指定32。

色块尺寸写入 `recolor_config.json` 的 `preprocessing.color_swatch_size`，训练、验证、推理共用。128×128对应8×8=64个参考token。新训练需要新输出目录，不能直接恢复旧32×32预处理的checkpoint；缓存随新进程自动重新编码。旧LoRA配置没有该字段时，独立推理继续按原来的原生色块尺寸处理，避免改变旧模型输入约定。这个改动不代表128尺寸的真实训练质量已经验证。

`--use_aspect_ratio_buckets` 启用参考比例梯级，自动桶的尺寸/裁剪可能与默认面积模式不同。显式桶沿用参考格式，例如 `--aspect_ratio_buckets '512,768;768,512'`（H,W），也接受 `512x768,768x512`。宽高须为16倍数，每图面积不超过1MP。训练记录策略供推理复用；验证/推理使用确定性中心裁剪。分桶训练不加center_crop时使用按epoch确定的随机裁剪，`--random_flip` 同步翻转目标和两张场景条件。

训练统一在线编码当前batch的目标图与三张条件图，不预计算、不保留latent字典。固定指令仍只编码一次，自定义caption训练时在线编码并保留文本编码器；维度检查通过accelerator.unwrap_model读取原模型配置，支持DDP包装。路径Dataset的hash缓存保留。旧checkpoint中的cache_latents字段在恢复比较时忽略，但不会恢复或启用latent缓存。

`--caption_json captions.json` 的格式是 `{"00000_00.png":"recolor red ...", ...}`，须覆盖数据文件名，空字符串回退到instance_prompt。`--caption_dropout 0.1` 只丢弃文本，不丢条件图；空文本的编码窗口与被丢指令一致。固定prompt的推理文本设置仍从训练配置读取。

## 输出与恢复

新训练使用新的空输出目录，不与数据/模型目录重叠。典型产物：

```text
outputs/flux3_recolor_v2/
  model_config.json
  transformer/
    pytorch_lora_weights.safetensors
  text_encoder/                  # 仅train_text_encoder时存在
    config.json
    model.safetensors
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

冻结VAE和普通Qwen不导出，默认从基础模型加载。checkpoint与最终产物使用相同组件布局。model_config.json记录base_model及transformer/text_encoder/vae的来源；transformer只保存参考脚本同格式的Diffusers LoRA和adapter metadata，自定义text_encoder完整save_pretrained。新weights_layout不兼容旧目录，请重新训练。

统一加载（输出目录或完成的checkpoint目录均可）：

```python
from flux3_recolor.inference import load_recolor_pipeline, generate_sample
pipe = load_recolor_pipeline("outputs/flux3_recolor_v2", device="cuda", mixed_precision="bf16")
# conditions为[涂灰图, 源图, 色块]的PIL图片列表；默认使用加载时准备好的正/负文本条件。
image, references = generate_sample(pipe, conditions, pipe.recolor_config, seed=42, steps=28)
```

基础模型迁移后可传pretrained_model_name_or_path覆盖位置；普通Diffusers加载LoRA仍可直接使用output_dir/transformer，但它本身不会自动加载VAE或自定义文本模型。

普通DDP检查点保存LoRA及optimizer/LR/scaler/每rank RNG；开启train_text_encoder时另含标准text_encoder目录及其优化器状态；FSDP原生状态额外含基础模型。完成后才写入version=2标识并成为latest候选。`checkpoints_total_limit` 只清理本入口格式且步数匹配的完成目录，忽略未知/未完成/符号链接目录。

同一步同时触发checkpoint和验证时，先提交checkpoint，再做验证推理；验证异常也能保留该保存步的训练状态。若旧版本已在保存前退出且目录只有配置/日志/loss，没有checkpoint，则无法从loss恢复模型权重，请保留日志并使用新输出目录重跑。

恢复例子：使用相同基础模型、数据、分组、caption、预处理、world_size、batch/accumulation、优化器和LR配置，附加 `--resume_from_checkpoint latest`（或当前输出目录内的完整checkpoint路径）。num_train_epochs与max_train_steps是总训练horizon，不是额外训练量。

**只有constant调度允许直接增加总训练步数**；cosine/linear等改变horizon会改变轨迹，恢复时会拒绝。日志/保存频率可调整；数据或关键配置改变需要新输出目录。模型权重分片不应被原地替换，模型校验检查本地路径及配置摘要，不逐分片哈希。

旧版本手动累积循环的训练checkpoint不能用于新入口恢复，请使用新目录。旧标准LoRA及旧推理配置仍可用于推理。

`--logging_steps N` 控制loss、学习率、tracker和train_history.jsonl的记录频率，默认10，按成功的优化器更新步计数（不是microbatch）。例如`--gradient_accumulation_steps 4 --logging_steps 10`，通常累计40个microbatch后记录一次。loss为记录区间内各rank实际样本数加权均值；最后一步补记不足一个区间的尾段。tqdm只在主进程显示，每次成功更新推进一步，恢复时从checkpoint.global_step继续。

可在恢复时调整logging_steps。恢复后的首个区间仅统计本次继续训练的样本，不重算或持久化中断前尚未记录的loss。训练逐microbatch的非有限loss检查保留；checkpoint与验证频率独立，不受logging_steps影响。设置`--logging_steps 1`恢复逐步loss输出。

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
- `test/generated`、`test/comparison`：实际执行的最终验证记录于test阶段；若与同一步先前验证重复则省略。
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

默认从适配器配置的dataset_path读取缓存validation；`--split_json`可按sample_id筛选。验证每张单独保存，五列对照为涂灰图/源图/色块/目标图/生成图，另存seed与推理步数清单。用相同场景和 seed 替换色块，对比是否按颜色改变，并观察纹理、人物和背景；loss 下降本身不证明换色质量。

## 已验证与未验证

本轮34项离线CPU测试通过，另1项真实两进程CPU DDP验收通过。使用本地随机小型FLUX.3/VAE/Qwen3-VL/tokenizer，不下载权重；覆盖数据合同、三参考数学、LoRA、分桶、cache身份、caption/dropout、LR/warmup、workers、TensorBoard、Prodigy、恢复、主进程验证与标准推理。两进程检查参数同步、epoch中途恢复及尾累积；连续/恢复训练的最终adapter逐位一致。完整命令见[测试指南](tests/README.md)。本轮完整数据检查确认423组、381/42拆分。

用户完成2步smoke训练后，本轮已用该LoRA在GPU1/BF16下完成4张真实留出样本推理（每图28步），结果与五列对照在 `outputs/flux3_recolor_smoke_validation/`。第一张仍沿用B图颜色，不能据此宣称训练已收敛。真实基础权重加载/推理已执行，但没有启动额外训练或测峰值显存；FP16 scaler、FSDP运行、bitsandbytes/8bit Adam、FP8和完整训练质量仍需对应验收。未安装torchao或连接外部tracker。


## 单独测试原始 pipeline

[test_flux3_recolor_pipeline.py](test_flux3_recolor_pipeline.py) 不依赖训练入口，不加载LoRA/精度补丁；默认使用官方ModularPipeline本地工厂及组件加载器，VAE FP32、其他模型按mixed_precision。关闭prompt扩写以保持输入指令不变，默认prompt不包含目标颜色名称/RGB/色号。

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python test_flux3_recolor_pipeline.py \
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --dataset_root ./test --sample 00019_00.png --direction b_to_a \
  --offload_text_encoder --output_dir ./outputs/pipeline_test_00019
```

`--offload_text_encoder` 使用官方文本步骤预计算后释放Qwen，再加载Transformer；省略该开关时直接传prompt给完整pipeline。默认1024²面积预算、色块128、seed43、28步、CFG3.5。`--prompt`或`--prompt_file`可覆盖指令；`--color_swatch_size`、`--seed`、`--guidance_scale`、`--num_inference_steps`均可调整。也可同时给`--gray_image`、`--source_image`、`--swatch_image`测试任意三张图（覆盖dataset样本路径）。输出generated.png、condition_1/2/3.png和run_config.json，输出目录须为空。CPU小模型测试请使用`--device cpu --mixed_precision no`。


默认 LoRA 目标现包含 Attention（to_q/to_k/to_v/to_out.0）、FFN（ff.linear_in/ff.linear_out）和参考图调制层 `early_stream_modulations.image_cond.linear`、`single_stream_modulations.image_cond.linear`。`--lora_layers` 会完整覆盖默认列表。新增默认模块用于新训练，请使用新的输出目录。旧checkpoint不能直接增加模块恢复；继续旧训练需保持原代码的目标列表及原恢复配置。


可学习文本条件的独立对照入口为 [train_flux3_recolor_text_encoder.py](train_flux3_recolor_text_encoder.py)，使用共享任务embedding＋LoRA；原入口默认仍为固定文本。命令、保存/恢复和限制见[实验说明](FLUX3_TEXT_ENCODER.zh-CN.md)。


### Dataset 读取

训练直接load_from_disk读取hash缓存的train/validation，不再重新配对、拆分或读取图片头。DreamBoothDataset初始化只根据保存的height/width建立buckets、bucket_idx和bucket_indices；BucketBatchSampler按这些桶索引组batch。实际取样才读取路径，目标/灰图/源图共享裁剪和翻转，色块独立resize；若场景原始分辨率不同，先对齐目标画布。明确的读图错误会打印并固定额外尝试最多8个同bucket、同repeat的其他唯一记录，不判断方向。程序错误直接抛出。

没有额外的数据检查模式或重试参数。目标与三参考在训练时在线VAE编码。已删除前一版的指纹/尺寸报告兼容和验证替换逻辑；旧版本 checkpoint 不做兼容迁移，请使用新输出目录开始训练。


### 预先构建路径 Dataset

训练入口会在输出目录检查之后、加载模型之前，直接import并调用 `build_flux3_recolor_dataset.build_dataset`，由主进程通过 `flux3_recolor/dataset_cache.py` 保存到 `instance_data_dir` 上一级的 `datasets/recolor_cache/<SHA256>`；相同数据构建输入会跨输出目录和恢复训练复用，具体路径写入dataset_report.json的path_dataset。无需单独执行构建命令。prompt、caption_json和groups_json会传给构建函数，产物内容与对应训练选项一致。各子集独立收集完整配对，不再被训练入口二次取交集。groups_json应覆盖所有参与构建的源文件名。

如需单独检查，也可以运行独立入口（不加载模型）：

```bash
../.venv/bin/python build_flux3_recolor_dataset.py --dataset_root ./test --output_dir ./datasets/recolor_paths --num_proc 4
```

分别构建两个子集，在concatenate阶段按比例合并，再按source_id分组划分train/validation。默认1:1；`--mix_ratio 2 1`在拼接时通过重复补足比例并保留原记录；只构建一个子集可用`--training_directions b_to_a`。这些方向参数仅用于构建，最终数据无direction列。

保存字段：

| 字段 | 内容 |
|---|---|
| sample_id / source_id | 样本身份 / 完整文件名主干对应的来源组 |
| target_path | 目标图片的绝对路径字符串 |
| condition_paths | 涂灰图、源图、色块的三个绝对路径字符串 |
| height / width | 目标图转正后的高/宽，供bucket分配 |
| condition_sizes | 三张条件图各自的height/width |
| prompt | 固定换色指令，不提供颜色名 |

图片路径使用Value("string")，不是Image特性，因此save_to_disk不会嵌入图片内容。构建只读取图片头与EXIF，保存显示方向对应的尺寸，不做RGB解码、resize或tensor转换。PNG跳过压缩数据chunk读取EXIF，包含IDAT后的方向信息。缺配对或无法读取图片头的记录打印后跳过；可读头部不代表压缩像素完整，像素解码错误留待训练取样处理。

```python
from datasets import load_from_disk

dataset = load_from_disk("./datasets/recolor_paths")
row = dataset["train"][0]
print(row["target_path"], row["height"], row["width"])
```

合并后按source_id拆分，同源两个子集及重复记录留在同一侧。source_id为完整文件名主干，不自动识别不同文件名的同源衍生图。原图必须保留在记录的路径，文件内容改变后需重新构建尺寸元数据。已有输出目录拒绝覆盖。

训练入口现在已实际消费该缓存：初始化仅计算bucket，__getitem__读取路径并同步变换，验证也使用保存的记录。split.json保存train/validation的sample_id列表；recolor_config.json记录dataset_path、dataset_root、dataset_format和统一condition_order。新的数据顺序/身份与旧管线不同，旧checkpoint不能直接resume，请用新output_dir开始。latent缓存已删除，不再计划文件去重；目标、涂灰图和源图沿batch维合并编码，色块单独编码，每个microbatch从4次VAE调用降为2次。仍使用FP32 VAE、posterior mode、原BN归一化和patchify；场景编码batch增至3倍，峰值显存可能增加，实际速度需实机测量。旧datasets/recolor_uint8约24G缓存保留。


hash输入为构建版本、输入目录、方向/目录映射、选中文件名、混合比例、划分seed/比例、prompt、caption和来源分组内容，以及源文件大小/mtime_ns。模型、LoRA rank、学习率、分辨率、repeats及worker数不影响当前路径数据集内容，不进入hash。命中只检查文件状态并加载保存的Dataset，不再执行构建或读取图片头；训练Dataset与Sampler初始化也不读取图片头。源文件若保留大小和mtime但修改内容，无法被这一轻量校验识别。

每个hash使用独立文件锁，构建写入临时目录，完成Dataset与cache_config.json后原子发布；并发相同配置只构建一次，失败不会留下可命中的半成品。构建版本/具体配置/源文件状态可在cache_config.json中查看。无新增CLI参数；构建算法或schema改变时需更新CACHE_FORMAT_VERSION。
