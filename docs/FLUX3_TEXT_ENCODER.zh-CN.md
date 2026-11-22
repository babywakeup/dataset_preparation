# FLUX.3 可训练 text encoder 对照

新入口：[train_flux3_recolor_text_encoder.py](../train_flux3_recolor_text_encoder.py)。固定文本对照仍使用原来的 [train_flux3_recolor_lora.py](../train_flux3_recolor_lora.py)。两者共享训练循环、数据拆分、三图预处理、flow loss、LoRA、Accelerate 累积和验证。

新入口使用独立的 Transformers `TextEncoderConfig` / `TextEncoderModel`，将原 prompt 的编码初始化为共享 FP32 `[1,L,H]` 参数，并联合 LoRA 更新；每个 batch 扩展为 `[B,L,H]`。`L` 由text_encoder/config.json的length字段决定（正整数），从prompt初始化时默认取其实际编码窗口，`H` 为 Transformer 的 `joint_attention_dim`。基础Qwen、VAE和基础Transformer权重保持冻结，自定义text_encoder参与更新；Qwen只做初始化和负条件编码，训练批次不重新编码固定 prompt。所有样本共用这一组参数，不是每张图片独立记忆的特征表，也没有新增色块编码器。目标颜色仍只来自第三张图。

当前训练数据顺序沿用原入口：涂灰定位图、待改色源图、目标色块；不是近期 demo 的原图在前顺序。所有样本的正条件特征可更新，但仍需要 LoRA 学会读取对应色块。

## 新实验训练

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/accelerate launch --num_processes 1 \
  train_flux3_recolor_text_encoder.py \
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --instance_data_dir ./test \
  --output_dir ./outputs/flux3_text_encoder \
  --resolution 1024 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --rank 16 --lora_alpha 16 --learning_rate 1e-5 \
  --max_train_steps 1000 --mixed_precision bf16 --gradient_checkpointing \
  --num_validation_images 4 --validation_steps 100 --report_to tensorboard
```

task encoder 与 LoRA 统一使用 `--learning_rate`，并共享scheduler与scale_lr规则。embedding参数组保留weight_decay=0，不再提供独立学习率参数。

固定文本对照：把脚本名换为 `train_flux3_recolor_lora.py`，使用另一个新输出目录；其余数据、seed、rank、LoRA目标模块、学习率、累积和训练步数保持一致。原入口默认不启用此功能；也可显式传 `--train_text_encoder` 使用同一路径。不要沿用历史模型配置不同的运行作为严格对照。

## 验证、恢复和推理

验证继续使用原生 pipeline，共享训练 Transformer 的 LoRA；每次读取当前更新后的 embedding 作为正条件，空文本负条件保留冻结编码，避免 CFG 两分支都被替换成同一特征。model_config.json记录基础模型位置与各组件来源。导出/每个完成 checkpoint 同时包含：

- `transformer/pytorch_lora_weights.safetensors`：LoRA。
- `text_encoder/config.json`、`text_encoder/model.safetensors`：标准 Transformers 配置和 FP32 特征权重。
- `recolor_config.json`、训练状态等原有配置。

新一轮训练可直接使用主入口加`--init_from 上一轮输出或checkpoint目录`，自动继承text_encoder的完整配置与权重（包含length），不再从prompt重新初始化；LoRA同时继承，新优化器/step=0。使用新输出目录，可更换数据与学习率。该参数与resume互斥；之后resume只依赖新运行自己的checkpoint。

恢复使用同一个新入口并加 `--resume_from_checkpoint latest`，embedding 原位载入以保留 optimizer 参数引用。原固定文本 checkpoint 与本实验不能互相恢复。新的text_encoder_format及weights_layout标记会拒绝旧版内嵌embedding训练状态和旧目录结构；两个独立模型改变了分布式/优化器保存结构，请新建训练目录。仅支持新的text_encoder目录格式，不读取旧单tensor文件。单独复制 LoRA 文件不包含任务 embedding；部署时必须保留配置和text_encoder目录。

原推理入口会自动识别并读取学习后的正条件：

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python infer_flux3_recolor.py \
  --lora_dir ./outputs/flux3_text_encoder \
  --dataset_root ./test --limit 4 \
  --output_dir ./outputs/flux3_text_encoder_inference \
  --device cuda --mixed_precision bf16
```

为了保持“固定任务特征”的实验定义，不支持 caption_json、prompt_augmentation、caption_dropout、validation_prompt/final_validation_prompt 覆盖；可用 instance_prompt 指定初始化文本。本入口目前支持单进程和 DDP，拒绝 FSDP，避免遗漏特征参数的分片保存。

## 已验证范围

当前独立模型版本使用CPU随机真实小模型检查梯度经过Transformer回传到encoder、标准save_pretrained/from_pretrained、原位恢复、训练更新、验证当前特征与独立推理。两模型分别传给accelerator.prepare，accumulate同时覆盖两者，梯度裁剪覆盖LoRA和encoder；不再修改Transformer.forward。当前版本4模块CPU回归36项（34通过、2跳过），另两rank独立双模型CPU DDP同步/恢复1项通过。未对当前版本执行GPU/真实大模型质量或性能验收。

独立加载：

```python
from flux3_recolor.text_encoder import TextEncoderModel
encoder = TextEncoderModel.from_pretrained("outputs/flux3_text_encoder/text_encoder")
features = encoder(batch_size=2).last_hidden_state
```

指定新模型token长度的示例：

```python
from flux3_recolor.text_encoder import TextEncoderConfig, TextEncoderModel
encoder = TextEncoderModel(TextEncoderConfig(length=128, hidden_size=20480))
features = encoder(batch_size=2).last_hidden_state  # [2, 128, 20480]
```

length随save_pretrained保存，from_pretrained据此恢复参数形状；不能只修改已有权重目录的length而不调整权重形状。训练入口默认仍由初始化prompt得到length。

验证pipeline直接引用训练中的text encoder（与Transformer一起共享对象），通过validation.pipeline_text_embeddings读取当前正条件并显式detach，冻结空文本负条件不变；图片tracker记录集中于validation.log_validation_images，训练渲染函数不再区分task类型。

独立三图片推理脚本（同时加载 checkpoint 的 LoRA 和自定义 text encoder）：

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python examples/recolor/demo_flux3_recolor_trained_pipeline.py \
  --model_dir ./outputs/flux3_task_embeddingnew5/checkpoint-600 \
  --gray_image './test/经过涂灰的原图AB/00019_00.png' \
  --source_image './test/生成图B/00019_00.png' \
  --swatch_image './test/主颜色A/00019_00.png' \
  --output_dir ./outputs/flux3_trained_demo \
  --num_inference_steps 28 --seed 42
```

基础模型路径默认从保存的配置读取；迁移模型后可用 `--pretrained_model_name_or_path` 覆盖。
条件顺序与训练一致：涂灰图、源图、目标色块。自定义 text encoder 提供正向特征，不需要输入 prompt 或颜色名称；基础 Qwen 仅用于空文本负向条件。输出生成图 `generated.png`、四列对照图 `comparison.png`（涂灰图、源图、色块、结果）、预处理后的条件图和运行参数。输出目录须为空。
