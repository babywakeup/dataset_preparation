#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${OBJECT_INPAINTING_STUDIO_PYTHON:-$SCRIPT_DIR/../.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable is not executable: $PYTHON" >&2
    echo "Set OBJECT_INPAINTING_STUDIO_PYTHON to the desired Python interpreter." >&2
    exit 2
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$(cd -- "$SCRIPT_DIR/../.." && pwd)/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
exec "$PYTHON" -m object_inpainting_studio "$@"




# export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
# export ANTHROPIC_AUTH_TOKEN=sk-f2553c4fa67c46a390ca7aa4a184483a
# export ANTHROPIC_MODEL=deepseek-flash[1m]
# export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-flash[1m]
# export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-flash[1m]
# export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-flash
# export CLAUDE_CODE_SUBAGENT_MODEL=deepseek-flash
# export CLAUDE_CODE_EFFORT_LEVEL=max
# export CLAUDE_CODE_AUTO_COMPACT_WINDOW=786432