#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-runs/qwen_visual_model_sweep}"
SUITE="${SUITE:-colors}"
COLORS="${COLORS:-red,blue,green,orange,purple,yellow}"
PIXEL_SIZES="${PIXEL_SIZES:-24x16,48x32,75x150,150x100,300x200}"
MODELS="${MODELS:-Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-32B-Instruct}"
BACKEND="${BACKEND:-qwen}"

MODEL_ARGS=()
for model in ${MODELS}; do
  MODEL_ARGS+=(--model-id "${model}")
done

python scripts/run_qwen_visual_perception_tests.py \
  --backend "${BACKEND}" \
  "${MODEL_ARGS[@]}" \
  --suite "${SUITE}" \
  --colors "${COLORS}" \
  --pixel-sizes "${PIXEL_SIZES}" \
  --out "${OUT}" \
  "$@"
