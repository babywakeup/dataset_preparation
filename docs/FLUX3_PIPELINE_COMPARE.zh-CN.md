# 无梯度训练 / 官方 Pipeline 对照

入口：[compare_flux3_training_pipeline.py](../compare_flux3_training_pipeline.py)。仅加载基础模型，不加LoRA，不创建优化器、不backward、不更新参数。复用实际训练的Dataset/collate/VAE编码/文本编码/flow_matching_loss，而不是另写一套相似的loss。

## 对齐的量

比较同一张目标图编码后、相同随机噪声和相同sigma下的一次conditional velocity预测；不能把训练标量loss直接与多步CFG生成的最终照片比较。

- fork_rng重放实际helper的噪声/时间采样顺序，不替换其采样函数。
- pipeline收到相同noisy latent，输入逆shift的sigma，再由其正常scheduler执行一次shift，避免重复shift。
- 捕获两条实际路径的RGB像素、VAE输入像素、Transformer输入、预测、同标签MSE及一步Euler后的latent。
- 使用原始路径与布局，不先通过hook统一VAE布局来让对照通过。
- 独立重放相同Transformer输入，并分别使用训练autocast和pipeline上下文，分开判断输入差异与执行精度差异。
- 额外的布局/autocast探针单独记录，不替换主对照中的任何输入或预测。

## 官方示例的精度

用户提供的官方代码是 `ModularPipeline.from_pretrained`，随后 `load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})`，没有外围autocast。脚本默认采用这组pipeline精度，并从原始权重重新加载FP32 VAE；不能把训练中的BF16 VAE简单upcast后称为原始FP32 VAE。

两条路径共享同一份基础Transformer/文本权重；pipeline仍使用官方RGB、VAE、文本、位置编码、scheduler和denoise块。为确保比较同一个prompt与一个conditional前向，关闭prompt upsampler和CFG。完整官方示例可扩写prompt并运行CFG，因此这里不声称比较了其完整默认生成过程，也不下载额外的prompt upsampler模型。

## 运行

只在GPU1空闲时运行，输出目录必须为空或不存在：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../.venv/bin/python -u compare_flux3_training_pipeline.py \
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --instance_data_dir ./test --sample 00047_00.png --direction a_to_b \
  --resolution 1024 --mixed_precision bf16 --device cuda \
  --seeds 42 43 44 --output_dir ./outputs/flux3_compare_new --save_tensors
```

`--direction b_to_a`检查另一方向。复现旧legacy训练内验证的BF16 VAE及外层autocast，用 `--pipeline_vae_dtype same --pipeline_autocast training`。这两个选项只调整诊断pipeline，不修改生产训练配置。

`report.json`保留shape、dtype、stride、逐位相等、指定阈值allclose、最大误差、RMSE和相对L2。默认阈值atol=1e-5/rtol=1e-4，没有为BF16自动放宽。程序完成不等于所有字段allclose；必须查看报告。`--save_tensors`额外保存CPU张量以便复核。

注意：本地FLUX.3 decoder即使output_type=latent仍可能执行VAE解码；一步latent的比较读取state.latents，而非state.images。该输出是一步估计，不是正式多步生成质量评估。

## 已核实的发现

真实GPU1初轮（当前训练BF16 VAE，对当前验证BF16 VAE+autocast）中，RGB、noisy latent、时间步、文本和位置编码一致，条件latent不逐位一致。相同Transformer输入且相同autocast重放时预测误差为0；条件编码探针表明channels_last+autocast才能匹配该pipeline的条件tokens。

切换为用户官方精度（FP32 VAE、BF16 Transformer、不包全局autocast）后，也不能声称等价：除了条件编码精度不同，相同Transformer输入在训练autocast与官方不启用autocast的上下文中也有数值差异。官方FP32 VAE以相同像素/布局独立通过共享编码helper计算，cast到Transformer dtype后与官方条件tokens逐位一致。这定位了精度与执行上下文差异，不代表loss符号/标签/shape错误，也不能据此断言已找到新增印花的唯一根因。

之前的CPU控制测试对齐了encoder布局，只证明受控的数学路径；它不足以证明真实BF16执行路径完全等价。本次实际路径测试保留这些差异。该诊断阶段未切换生产配置；后续修复已提供fp32_vae_native_norm_v1方案，见下方。

## 浮点误差的再次计算

入口：[analyze_flux3_numeric_precision.py](../analyze_flux3_numeric_precision.py)。重放上次保存的相同输入，独立改变条件tokens和AMP形成四格对照；比较同配置重复结果，再将同一份可表示权重和输入转为FP32计算（关闭TF32），并以最终一次BF16舍入作为参考。没有重新生成噪声/文本/目标，也不更新任何权重。

真实GPU1六组重算：训练/官方诊断MSE相对差0.117%–0.923%，velocity相对L2差1.743%–10.777%；训练BF16+AMP与同输入FP32参考差0.775%–5.351%，官方BF16无AMP与各自FP32参考差0.797%–5.326%，两者量级相近。最终单次输出BF16舍入约0.166%。所有同配置BF16/FP32重复以及保存预测重放均逐位一致。

必须区分：浮点误差可以确定性地复现；最后一次舍入不是多层误差的上界；FP32参考改变整个精度路径，包括时间步乘法和dtype相关归一化，不是单独矩阵乘误差或图像质量真值。这些结果支持存在数值精度路径敏感性，但不能把“不allclose”直接定性为漏做预处理或loss错误，也不能单凭loss接近保证最终图像等效或解释印花成因。无需仅因为这项差异就认定必须修改训练逻辑。

完整表格在 `outputs/flux3_numeric_precision/README.md`。重跑需新目录：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../.venv/bin/python -u analyze_flux3_numeric_precision.py \
  --pretrained_model_name_or_path ../../hf_cache/models--black-forest-labs--flux-3-image-klein-base \
  --report_root ./outputs/flux3_pipeline_compare_verified \
  --output_dir ./outputs/flux3_numeric_new --device cuda --compute_dtype bf16
```

## 定位到具体归一化子层

后续按用户要求增加 [trace_flux3_tensor_path.py](../trace_flux3_tensor_path.py)，对完整相同的Transformer输入做前后事件指纹，而非只比较最终velocity。真实GPU首个分歧为 `txt_mode_blocks.0.norm`：输入完全一样，LayerNorm在AMP内输出FP32，在AMP外输出BF16，随后调制乘加也使用不同dtype。仅对全部44个LayerNorm禁用AMP、其余网络仍用AMP时，6组同输入velocity从0.845%–5.894%相对差降到逐位相等。这验证的是归一化类策略，不是只有第一个LayerNorm单独造成全部误差。

VAE同输入独立实验：encoder首个AMP分歧为 `down_blocks.0.resnets.0.norm1` GroupNorm，decoder为 `mid_block.resnets.0.norm1` GroupNorm；对各自全部归一化模块统一策略后，encoder的2.428% latent相对差、decoder的0.2266%输出相对差均降为逐位0。decoder用固定相同latent隔离，且decoder在velocity之后，不能解释velocity误差。另一个独立不一致是原始FP32 VAE权重/输入与BF16 VAE的加载精度，首差开始于像素cast和encoder.conv_in；精度/布局一致时编码helper与官方原生编码块、解码手算与官方原生解码块均逐位相等。

具体实现入口：训练在autocast外编码VAE，在autocast内运行Transformer；训练内验证将整个pipeline包在autocast内；独立推理没有外层autocast，用户官方示例另指定FP32 VAE。归一化公式一致，但这些入口的输出dtype策略不统一。此前总1.7%–10.8%还包含条件latent改变，因此不是严格相同完整Transformer输入的AMP误差。

报告 `outputs/flux3_tensor_path_trace/README.md` 附各阶段指纹/局部输入输出误差/6组干预结果。干预只在诊断模型实例临时包装forward并还原；该诊断阶段未改生产代码和checkpoint，未证明是可见印花的根因。


当前已按用户要求删除precision_recipe、legacy分支与自定义norm替换，改为原始FP32 VAE+Transformer自身autocast。上述归一化干预和旧recipe结果属于历史诊断，不能视为当前实现的GPU验收。当前诊断入口不再接受precision_recipe参数。

历史生产修复（现已撤回）：当时默认 `fp32_vae_native_norm_v1`，FP32 VAE/缓存/加噪/标签、显式Transformer输入dtype、局部原生norm、模型内部固定compute AMP；共享原生pipeline验证临时绕过Accelerate输出FP32包装。当时诊断脚本提供legacy/recipe选择，现已删除该接口。新策略必须使用新训练目录，旧checkpoint不能直接切换恢复。

真实Transformer复测工具为 `verify_flux3_precision.py`，只加载Transformer并重放已保存输入，限制显存使用。`outputs/flux3_precision_recipe_verified.json` 六组base同输入对照均逐位0；零/非零LoRA及真实Accelerate临时绕过只验证最后一组输入，均逐位0。非零LoRA为人工设置的测试权重，不代表训练质量结论。小模型CUDA两步训练/原生pipeline画图/继续训练另行验收。
