#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export BACKEND="ollama"
export MODELS="${MODELS:-llava:7b}"
export COLOR_SET="${COLOR_SET:-flag_core}"
export OUT="${OUT:-runs/ollama_llava_visual_flag_core}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"

"$ROOT/scripts/run_qwen_visual_model_sweep.sh" "$@"
