#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${OBJECT_INPAINTING_STUDIO_PYTHON:-$SCRIPT_DIR/../../.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable is not executable: $PYTHON" >&2
    echo "Set OBJECT_INPAINTING_STUDIO_PYTHON to the desired Python interpreter." >&2
    exit 2
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$(cd -- "$SCRIPT_DIR/../../.." && pwd)/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_HOME/xet"
exec "$PYTHON" -m object_inpainting_studio "$@"