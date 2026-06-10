#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"
cd "$ROOT"

OUT="${OUT:-runs/qwen_visual_model_sweep}"
SUITE="${SUITE:-colors}"
COLOR_SET="${COLOR_SET:-flag_core}"
COLORS="${COLORS:-}"
PIXEL_SIZES="${PIXEL_SIZES:-24x16,48x32,75x150,150x100,300x200}"
MODELS="${MODELS:-Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-32B-Instruct}"
BACKEND="${BACKEND:-auto}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  "$ROOT/scripts/runpod_model_preflight.sh"
fi

MODEL_ARGS=()
for model in ${MODELS}; do
  MODEL_ARGS+=(--model-id "${model}")
done

COLOR_ARGS=(--color-set "${COLOR_SET}")
if [[ -n "$COLORS" ]]; then
  COLOR_ARGS=(--colors "${COLORS}")
fi

python scripts/run_qwen_visual_perception_tests.py \
  --backend "${BACKEND}" \
  "${MODEL_ARGS[@]}" \
  --suite "${SUITE}" \
  "${COLOR_ARGS[@]}" \
  --pixel-sizes "${PIXEL_SIZES}" \
  --out "${OUT}" \
  "$@"
