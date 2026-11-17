# FLUX.3 可学习任务embedding对照

用户在讨论共享[B,L,H]可训练特征后授权新增训练文件。当前工作区已有未提交内容，原地保留，不提交或重启真实训练。

设计：新train_flux3_recolor_task_embedding.py默认启用共享[1,L,H] FP32参数，由现有固定prompt编码初始化；批次扩展发生在Transformer.forward内，保证DDP追踪。冻结原始模型，联合更新LoRA和该参数。参考图路径/loss/拆分/随机种子不变。验证eval时传当前参数的detach副本，仅替换正条件，CFG负条件保留原文本编码。保存独立task_embedding.safetensors，checkpoint加载原位copy保留optimizer引用，独立推理读取该文件。

1. RED→GREEN：参数梯度、batch共享/位置、仅训练forward替换、保存/恢复。
2. 共享训练入口增加opt-in参数/集成点，新入口封装；原入口默认行为不变。固定对照直接用原训练入口；可设置task_embedding_learning_rate，默认与LoRA相同。
3. checkpoint最终导出/验证/独立推理集成；拒绝caption/augmentation/dropout/验证文本覆盖与FSDP，避免混淆实验或不完整保存。
4. CPU随机真实小模型训练更新/连续vs恢复/两次验证及独立推理；两rank DDP验证参数同步；回归原流程。无真实大模型训练。
5. 文档、交接与一次独立review。

边界：常量任务embedding本身不编码每样本颜色，仍需LoRA读取三张图；不实现色块编码器、不替换Qwen多模态embedding。目标颜色只由色块提供。

实施发现：顶层FP32 Parameter会改变Diffusers model.dtype，RED复现后改为追加子模块保存weight参数，确保模型原dtype不变。

完成：全部任务完成。首轮26项通过，最终73项CPU回归（含两rank DDP）通过；独立GPU1 BF16随机小模型2步训练/验证1项通过。连续与恢复的LoRA/embedding逐位一致，验证与推理读取保存特征。独立review发现Prodigy不同参数组LR不支持，增加提前拒绝和回归后通过；无其他重要问题。未提交或启动真实训练。
