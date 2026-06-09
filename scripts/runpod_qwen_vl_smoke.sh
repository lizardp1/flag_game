#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "/workspace" && -w "/workspace" ]]; then
    export HF_HOME="/workspace/.cache/huggingface"
  else
    export HF_HOME="$ROOT/.cache/huggingface"
  fi
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib}"
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

if [[ "${SKIP_OPEN_MODEL_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements-open-models.txt
fi

python scripts/run_qwen_vl_smoke.py "$@"
