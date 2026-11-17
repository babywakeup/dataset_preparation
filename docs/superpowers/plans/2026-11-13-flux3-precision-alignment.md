# FLUX.3 训练 / pipeline 精度统一

用户已授权在当前工作区实施；保留已有未提交改动，不提交或启动完整训练。

1. RED→GREEN：原生归一化策略、临时去除 Accelerate forward 包装、pipeline 对象/参数共享。
2. 训练和新适配器推理统一 FP32 VAE；FP32 加噪/标签，前向显式 compute dtype；缓存保留精度。
3. 单卡/DDP 缓存共享模型的原生 pipeline，验证恢复 forward / train mode / RNG；FSDP 保留汇集副本。
4. 保存 precision_recipe；旧配置按 legacy 解释，禁止跨方案恢复。
5. CPU 集成 / 数学测试和 GPU1 同输入对照；更新使用说明与交接。

决策：只修改 LayerNorm/GroupNorm 的局部 autocast 策略，不改 RMSNorm、不整体 cast 模型、不清零 LoRA A。
验证边界：算术一致性不能证明条纹问题已解决。


完成记录：1–5均完成。首次精度/共享测试RED后GREEN；补充非零LoRA测试RED后接入模型内部AMP，集成发现forward签名丢失后修复wraps并重跑。CPU+两rank DDP73项通过，新增VAE原生断言后6项单独通过，CUDA随机小模型2步训练/验证1项通过。真实六组base velocity同输入差0，最后一组零/非零LoRA与Accelerate临时绕过差0；真实8次FP32VAE编码差0。独立review无重要问题。原有checkpoint未改，质量根因未证明；FSDP真实多GPU未执行。无提交/重启用户训练。
