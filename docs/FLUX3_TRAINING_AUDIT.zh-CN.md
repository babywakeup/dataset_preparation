# FLUX.3 换色训练核查

核查日期：2026-11-13。范围为当前独立训练入口、共享数据/数学模块、本地安装的 Diffusers FLUX.3 pipeline，以及用户指定的 DreamBooth 参考脚本。没有改写原始图片、已有 checkpoint 或运行配置，没有启动真实模型训练。

## 数据与张量路径

| 环节 | 核对结果 |
| --- | --- |
| 配对 | 按完整文件名读取四元组；目标 A 独立于三张条件，缺图、解码失败及场景尺寸不一致会报错 |
| 图片 | EXIF 方向校正、RGB、CHW 浮点、像素从 0..255 映射为 -1..1，与官方多参考 processor 比较像素一致 |
| 几何 | 目标、涂灰图、B 图使用相同尺寸/缩放/裁剪；分桶随机裁剪和翻转同步，色块独立处理 |
| 色块 | 新配置默认 NEAREST 放大至 128×128；旧配置缺尺寸字段时保留原生尺寸；训练、验证和推理共享 policy |
| VAE | 冻结且 eval，使用 posterior.mode；先 2×2 patchify，再按 VAE 自带 BN 的 mean/std 标准化，不套用其他模型的 scaling_factor |
| token | BCHW→BNC；场景像素网格为16。三张条件独立编码后依次拼接，L=-1/-2/-3；待生成目标 L=0；H/W RoPE 缩放0.5 |
| 文本 | Qwen chat template、指定隐藏层拼接、80 token 倍数窗口、padding position_id=1，与本地官方编码实现比较 |
| 训练状态 | 只训练 LoRA，A高斯/B零使初始增量为零；基础 Transformer/VAE/文本编码器冻结 |
| 验证 | 模型切 eval、恢复原训练状态和 RNG；噪声 scheduler 使用副本，推理不会改写1000步训练噪声表；解码使用官方 unpack/逆BN/unpatchify |

缓存以样本身份索引；随机几何与 latent 缓存的冲突会被拒绝。实际分辨率由配置决定，不由原图尺寸直接决定。例如768×1024场景在512²面积预算下经过网格裁剪为432×576，对应27×36个token；128色块对应8×8个token。

## Loss 的独立数值核对

设正确目标 latent 为 x₀，高斯噪声为 ε，从训练 scheduler 按样本索引取得 σ：

```
zσ = (1 - σ) x₀ + σ ε
timestep = σ
velocity_target = ε - x₀
loss = mean(weight(σ) * (prediction.float() - velocity_target.float())²)
```

三张条件保持干净，不参与加噪，不作为额外输出监督。Transformer 内部对目标时间步缩放；入口不应再乘1000。上述目标是 velocity，而非直接预测颜色图、干净图或纯噪声。

新增 `tests/test_flux3_math_audit.py` 使用独立推导的 shift=6.93 sigma 表、固定噪声与时间步，核对加噪输入、条件序列、目标符号、五种 weighting/sampling 选项下的解析 loss 与梯度：完美预测为零，固定残差对应解析加权 MSE。另检查 FP32/BF16 的 VAE 标准化、官方解码逆变换，以及初始 LoRA 对基础预测无影响。

梯度累积另用确定性逐样本目标捕获裁剪前梯度，对比完整6样本、4+2微批次、2+2+2且累计步数为4的不足尾组。此项验证累计缩放，不作为真实模型质量测试。

发现并补充一项防护：预测与目标必须具有完全相同的形状，否则立即报错，避免 PyTorch 静默广播产生貌似有效的 loss。没有发现当前真实 Transformer 已触发该错误的证据；正确形状的 loss 公式不变。

## 真实数据检查

该次数学审计读取 `test` 共468组，目标/两张场景条件均768×1024，原始色块均32×32且所有像素同色。以涂灰图中精确 RGB=(128,128,128) 的区域作为检查代理，468组 A/B 在区域外均逐像素相同。后续双向扩展的数据已增多，最新检查见迭代交接，不把这项468组检查外推到新增数据。

这不能证明服装内部的纹理、字样、褶皱或几何完全一致，也不是高质量 mask 的证明；自然灰色与边界处理可能影响代理。早先运行记录为423组，不能把本次468组的数据事实套用到旧运行；继续训练时仍由已保存 fingerprint/split 检查一致性。

## 结论与边界

后续真实GPU无梯度对照发现当前训练与用户官方示例的VAE精度/autocast上下文存在实际数值差异，详见[无梯度对照](FLUX3_PIPELINE_COMPARE.zh-CN.md)。以下数学核查不应解读为两种真实运行配置完全等价；生产精度统一仍未实施。

最终回归：以下命令54项全部通过（随机小模型、CPU；DataLoader worker项在允许本地IPC的环境执行），`git diff --check`通过：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests \
  ../.venv/bin/python -m unittest test_flux3_recolor_data test_flux3_recolor_training \
  test_flux3_recolor_inference test_flux3_reference_trainer test_flux3_reference_training \
  test_flux3_local_model test_flux3_math_audit -q
```

已核对的核心公式与参考脚本一致；不能据此保证真实生成不新增条纹或印花。目前使用全图 velocity MSE，没有单独的颜色/纹理保真项，也没有复制输入像素的硬约束。用户已有验证图中的印花幻觉仍是待解决的质量问题，不能用单元测试通过或较低训练 loss 代替验证。

测试使用本地随机小型模型，不代表真实大模型 CUDA/BF16、量化、FP8 或多卡 FSDP 的本轮验收。

实际forward差分测试曾观察到：输入像素、VAE权重、eval状态和dtype均相同，输入stride不同时，小模型原始VAE输出最大差异约9.38e-4、条件token约1.70e-3；单独关闭oneDNN未消除。仅在测试的encoder前统一连续布局后，像素、原始编码和全部Transformer输入检查通过，说明此差异与物理布局相关，而非漏做BN/错误token顺序。测试明确控制该变量，没有放宽阈值掩盖差异，也没有改写生产VAE布局。此观察不能证明真实大模型的同等误差大小，更不能证明它导致印花幻觉。
