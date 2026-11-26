#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 可通过环境变量选择 GPU、Python 和训练权重目录。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "${PYTHON_BIN:-../.venv/bin/python}" gradio_flux3_recolor.py \
  --model_dir "${MODEL_DIR:-./outputs/flux3_task_embeddingnew11/checkpoint-100}" \
  --server_name 127.0.0.1 \
  --server_port 7860 \
  "$@"
