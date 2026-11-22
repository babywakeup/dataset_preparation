# FLUX.3 batch samplers

- `bucket.py`: 固定方向的尺寸分桶 batch sampler；保留同一 bucket 的小批次，并按进程数补齐 batch 数。
- `direction_mix.py`: 双向训练使用的分桶混合 sampler；按 epoch 线性调整 AtoB 比例，同时保持 bucket 尺寸兼容和每个 bucket 的样本总量。

训练入口从 `sample_id` 的数字子集前缀解析方向，再结合 `training_directions` 顺序生成 `direction_ids`。Sampler 为所有进程生成相同的全局 batch 计划，由 Accelerate 负责分发。
