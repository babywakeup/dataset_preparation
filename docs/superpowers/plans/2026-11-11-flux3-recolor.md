# FLUX.3 Recolor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 新增可使用本地权重的三参考图换色 LoRA 训练及推理入口。

**Architecture:** 独立 `flux3_recolor/` 模块承载数据合同、FLUX.3 编码与训练逻辑，两个薄 CLI 提供训练/数据检查及推理。依据本地 Diffusers FLUX.3 示例实现，不导入或修改其训练脚本。

**Tech Stack:** Python、Pillow、PyTorch、Accelerate、PEFT、本地 Diffusers FLUX.3、Transformers Qwen3-VL。

**Spec:** `docs/superpowers/specs/2026-11-11-flux3-recolor-design.md`（用户已确认实施）。

## Global Constraints

- 固定条件顺序：经过涂灰的原图AB、生成图B、主颜色A；目标原图A。
- 独立脚本，不修改 Studio 或 Diffusers 原示例，不自动下载权重。
- 单步 batch=1，支持梯度累积；冻结基础组件，只训练 LoRA。
- 默认约 10% 留出验证；不做颜色增强，保留纵横比，同步场景几何处理。
- 无权重阶段的测试不能证明真实模型质量或显存需求。

## Review Focus

- 配对缺失/损坏/方向和尺寸不一致：启动前失败，包含具体文件名。
- 三图尺寸不同：分开编码，L=-1/-2/-3，目标 loss 不混入条件。
- 多参考管线二次缩放：预处理幂等且满足 1MP 与 16 像素网格。
- 恢复参数/数据变化：拒绝不兼容恢复，正确恢复 optimizer、RNG、步数和数据位置。
- 输出目录已存在或与输入重叠：拒绝意外覆盖；本地路径缺少组件时在加载前报错。

### Task 1: 数据与预处理

**Files:** `flux3_recolor/__init__.py`, `flux3_recolor/data.py`, `tests/test_flux3_recolor_data.py`。

**Interfaces:** `inspect_dataset(root) -> dict`（全量解码、哈希与尺寸）；`split_samples(names, fraction, seed, groups=None) -> dict`；`prepare_images(target, conditions, max_area) -> tuple`；`RecolorDataset(root, names, max_area)`。

- [x] 写真实临时图片测试：缺配对、损坏、尺寸、RGB/EXIF、分组拆分、预处理比例/网格/色块和一致变换。
- [x] 运行新测试确认缺实现失败。
- [x] 实现纯 Pillow 数据检查及预处理，延迟引入 torch 的 Dataset tensor 转换。
- [x] 运行 `../.venv/bin/python -m unittest discover -s tests -p 'test_flux3_recolor_data.py' -v`，期望全通过。

### Task 2: FLUX.3 核心与训练入口

**Files:** `flux3_recolor/modeling.py`, `flux3_recolor/training.py`, `flux3_recolor/config.py`, `train_flux3_recolor_lora.py`, `tests/test_flux3_recolor_training.py`。

**Interfaces:** Task 1 的数据检查、拆分和 Dataset；`encode_latents(vae, pixels)`、`reference_tokens(latents)`、`flow_matching_loss(model, target, references, embeddings, text_ids, scheduler)`；CLI 支持 `--check_dataset`、本地模型、rank/lr/steps/seed/accumulation/precision/checkpoint/resume 等。

- [x] 写 CPU 小张量测试：真实 FLUX.3 tiny Transformer 前反向、条件流位置编码、只训练 adapter、目标 loss、配置恢复校验和数据清单。
- [x] 运行测试确认缺实现失败。
- [x] 依据本地示例的 patchify/BN/text embeddings/shift=6.93 实现数学与 LoRA 训练，保存标准 safetensors 和配置，checkpoint 仅存 LoRA 及训练状态。
- [x] 实现确定性按 epoch shuffle 和 resume 的数据位置恢复，支持梯度累积最后不足一个 batch 的处理。
- [x] 运行相关 CPU 测试与 CLI help/check，期望通过；权重路径均 local_files_only。

### Task 3: 推理、验证与交接

**Files:** `flux3_recolor/inference.py`, `infer_flux3_recolor_lora.py`, `FLUX3_RECOLOR.zh-CN.md`, `tests/test_flux3_recolor_inference.py`；更新测试指南、架构地图及交接。

**Interfaces:** 读取 Task 2 输出的配置/权重；`generate_sample(pipeline, conditions, config, seed, steps)` 使用与训练相同的预处理；支持显式三图或验证清单。

- [x] 写验证产物、输出防覆盖、配置加载和多图预处理边界测试，运行确认缺实现失败。
- [x] 实现 ModularPipeline 三条件生成和训练后可选验证，保存目标/条件/生成对照，不重新下载权重。
- [x] 文档提供数据检查、训练、恢复与推理的可复制命令和依赖边界。
- [x] 运行三份相关测试、423 样本全量数据检查、Python 编译及 `git diff --check`；期望 CPU 验证通过。
- [x] 按 executing-plans 技能进行一次独立最终审查，修复重要问题并重跑相关验证，记录真实模型未执行。

## 执行记录

- 用户已明确要求“实施”，直接在当前会话执行，不再增加计划审批；现有源码和数据含未提交内容，保持原工作区，新增文件不触碰已有业务修改。未自动提交或创建工作树。
- Pre-flight: Task 1 的 names/尺寸/预处理由 Task 2 和 3 共用；Task 2 保存的配置由 Task 3 读取。接口一致。

- Task 1: complete — 数据合同/全量解码/共享预处理/分组拆分与检查CLI已验证。
- Task 2: complete — 真实小型FLUX.3前反向、LoRA与alpha元数据、连续/恢复逐位等价已验证。
- Task 3: complete — 本地随机小型FLUX.3/VAE/Qwen3-VL/tokenizer仓库全链路训练/验证/推理及23项CPU测试通过。
- Review: 训练验证共享scheduler的Important及几何错误缺文件名的Minor均先复现再修复；最终相关测试23项通过。未下载真实权重、未进行GPU训练。
- Ruling: 使用预编码正/负文本并移除推理文本/upsampler块，复用训练的文本编码逻辑，避免指令被重写和重复文本编码。
- Ruling: 仅新增文件并修正文档导航，在现有工作区实施，不自动提交/合并/创建PR；已有未提交业务改动保持。
