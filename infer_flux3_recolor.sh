#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 默认测试一组三参考图；输出目录必须为空。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "${PYTHON_BIN:-../.venv/bin/python}" infer_flux3_recolor.py \
  --lora_dir "${MODEL_DIR:-./outputs/flux3_task_embeddingnew5/checkpoint-600}" \
  --output_dir "${OUTPUT_DIR:-./outputs/flux3_inference}" \
  --gray_image './test/经过涂灰的原图AB/00019_00.png' \
  --source_image './test/生成图B/00019_00.png' \
  --color_image './test/主颜色A/00019_00.png' \
  --mixed_precision bf16 \
  --num_inference_steps 28 \
  --seed 42 \
  "$@"
