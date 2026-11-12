# FLUX.3 Reference Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 用参考脚本的组织及完整训练框架替换简化入口，适配本地三参考换色数据。

**Architecture:** 根脚本拥有 parse_args、DreamBoothDataset/collate/BucketBatchSampler、Accelerator/model/optimizer/scheduler 初始化、train/validate/save 流程；共用模块只承载数据合同、数学、配置与推理。所有可选高级路径都需有实际行为及明确组合限制。

**Tech Stack:** 本地 Diffusers FLUX.3、PyTorch、Accelerate、PEFT、Pillow、Qwen3-VL；可选 bitsandbytes/Prodigy/torchao/FSDP。

**Spec:** `docs/superpowers/specs/2026-11-12-flux3-reference-trainer-design.md`（用户已确认）。

## Global Constraints

- 完整文件名四目录配对，条件顺序固定，目标只在生成流计算loss，基础权重只读本地。
- 原 Diffusers 示例及现有无关业务改动保持；旧手动累积checkpoint明确拒绝，新状态记录world_size与horizon。
- batch所有参考尺寸匹配；缓存按身份绑定确定增强版本；推理复用预处理配置。
- 单卡/CPU验证不代表真实GPU、FSDP、量化或质量通过；不自动下载/安装可选依赖。

## Review Focus

- 分桶小尾batch和DDP补齐不能丢空loader或产生不同更新数。
- 缓存打乱/分片/增强及caption窗口不能错配或改变语义。
- checkpoint保存/验证同步不能死锁，恢复不能重跑数据或错误跳过优化步。
- 非constant horizon变化必须拒绝，LR/scaler/optimizer与原生累积完整恢复。
- checkpoint保留只清理有本入口格式标识的完整真实目录；输入/旧目录不可误删。

### Task 1: Reference-style entry/data/text/math

**Files:** `train_flux3_recolor_lora.py`, `flux3_recolor/data.py`, `modeling.py`, `tests/test_flux3_reference_trainer.py`。

**Interfaces:** `parse_args(argv)`, `DreamBoothDataset(args,names)`, `BucketBatchSampler(dataset,batch_size,seed)`, `collate_fn`, `get_qwen3vl_prompt_embeds(...,pad_to_length)`, `flow_matching_loss(...,weighting_scheme,...)`。

- [x] 先写参数别名冲突、batch>1/小桶、身份缓存、同步增强、caption窗口与weighting测试，观察缺实现失败。
- [x] 实现主脚本参数与Dataset/采样器，记录可复现epoch增强，扩展三参考数学和文本padding。
- [x] 验证对应CPU用例通过。

### Task 2: Accelerate full loop/optional features/checkpoint

**Files:** 根训练脚本、`flux3_recolor/config.py`、`training.py`兼容转发、相关训练测试及小模型fixture。

**Interfaces:** `main(args_or_argv)`, `create_optimizer`, `make_validation_pipeline`, `save_checkpoint`；真实DataLoader经accelerator.prepare，epoch sampler由set_epoch确定。

- [x] 先写原生累积/小尾LR、完整小仓库cache/dropout/customLoRA/保存恢复与非constant拒绝的测试，运行RED。
- [x] 依参考接通optimizer/scheduler、日志、缓存/offload、原生accumulate、分布式save/resume/验证；高级路径依赖和组合前置检查，不提供空旗标。
- [x] 运行本地随机小模型入口训练/验证/恢复/推理；运行2进程CPU DDP，验证数据分片、参数同步、产物与恢复。

### Task 3: Inference compatibility/docs/review

**Files:** `flux3_recolor/inference.py`、相关测试、中文说明、测试指南、架构/交接。

- [x] 先写显式桶训练/推理预处理一致性、旧state拒绝、checkpoint保留边界回归，运行RED。
- [x] 实现新状态/推理配置兼容；移除旧训练循环，更新旧训练测试至新接口与功能对应表。
- [x] 运行所有相关测试、全量test数据检查、语法/格式/链接检查；独立审查修复Important，再针对新变更验证。

## 执行记录

- 用户已批准设计；在当前已有数据/源码及未提交改动的工作区实施，不再请求计划审批、不创建另一套缺少未跟踪依赖的工作树，不自动提交/发布。
- Pre-flight: Dataset的sample_id/shape_key供cache与DDP sampler消费；保存config的preprocessing供推理复用；train状态与旧手动循环不兼容，以新版本明确界定。

- Task 1: complete — 原参数别名、批量分桶/自动比例梯级、确定增强、身份缓存和三参考数学验证通过。
- Task 2: complete — 真实小模型完整入口、LR/原生累积/恢复、worker/TensorBoard/Prodigy测试通过；2rank CPU DDP中途恢复与尾累积逐位等价。
- Task 3: complete — 共享预处理/旧配置推理/旧state拒绝/完成checkpoint保留边界通过；34项CPU + 1项2rank验收通过。
- Review: 两项FSDP Important（mixed master dtype与global clipping dispatch）先补失败用例再修复；CPU接口回归通过，真实GPU/FSDP仍未执行。
- Ruling: FSDP1沿用Accelerate原生checkpoint（含base），另导出标准LoRA；避免用DDP hook错误抹除FSDP模型状态。文档明确磁盘和内存成本。
- Ruling: 分桶保留尾batch，按完整batch补齐world_size；loss按当前累积组真实global样本数校正Accelerate除法和DDP平均，含尾组。
- Ruling: cache禁止随机flip及随机bucket crop，自定义caption在线编码共享batch窗口；不按step缓存，不静默冻结随机增强。
- Verification: 沙箱socket/IPC拒绝不算代码通过，获准沙箱外运行完整worker与DDP验收后通过；未自动下载/安装/上传或提交。
