#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
CONFIG="${CONFIG:-configs/open_models/llava_1_6_mistral_7b_pairwise_smoke.yaml}"
OUT_ROOT="${OUT_ROOT:-runs/llava_mech_interp}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-8}"
PROBE_WORKERS="${PROBE_WORKERS:-1}"
SEED_WORKERS="${SEED_WORKERS:-1}"
GEOMETRY_FEATURE="${GEOMETRY_FEATURE:-last_prompt_token}"
EXCLUDE_SAME_CROP="${EXCLUDE_SAME_CROP:-1}"
RUN_VISUAL="${RUN_VISUAL:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_BATCH="${RUN_BATCH:-1}"
RUN_T0_PROBES="${RUN_T0_PROBES:-1}"
RUN_LINEAR_PROBES="${RUN_LINEAR_PROBES:-1}"

export NND_TRANSFORMERS_VLM_FAMILY="${NND_TRANSFORMERS_VLM_FAMILY:-llava_next}"
export NND_TRANSFORMERS_DTYPE="${NND_TRANSFORMERS_DTYPE:-bfloat16}"
export NND_TRANSFORMERS_DEVICE_MAP="${NND_TRANSFORMERS_DEVICE_MAP:-auto}"
export NND_TRANSFORMERS_ATTN_IMPLEMENTATION="${NND_TRANSFORMERS_ATTN_IMPLEMENTATION:-auto}"

mkdir -p "$OUT_ROOT"

echo "LLaVA mech-interp pipeline"
echo "  model: $MODEL"
echo "  config: $CONFIG"
echo "  out: $OUT_ROOT"
echo "  seeds: $START_SEED .. $((START_SEED + NUM_SEEDS - 1))"
echo "  dtype: $NND_TRANSFORMERS_DTYPE"

if [[ "$RUN_VISUAL" == "1" ]]; then
  python scripts/run_qwen_visual_perception_tests.py \
    --backend llava \
    --model-id "$MODEL" \
    --suite all \
    --color-set flag_core \
    --out "$OUT_ROOT/visual_perception"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  python -m nnd.cli run \
    --config "$CONFIG" \
    --out "$OUT_ROOT/pairwise_smoke" \
    --backend transformers_vlm \
    --override "model=$MODEL"

  python scripts/analyze_activation_geometry.py \
    --runs "$OUT_ROOT/pairwise_smoke" \
    --feature "$GEOMETRY_FEATURE" \
    --out "$OUT_ROOT/pairwise_smoke/activation_geometry_$GEOMETRY_FEATURE" \
    $(if [[ "$EXCLUDE_SAME_CROP" == "1" ]]; then echo "--exclude-same-crop"; fi)
fi

if [[ "$RUN_BATCH" == "1" ]]; then
  python -m nnd.cli batch \
    --config "$CONFIG" \
    --out "$OUT_ROOT/geometry_batch" \
    --backend transformers_vlm \
    --start-seed "$START_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --probe-workers "$PROBE_WORKERS" \
    --seed-workers "$SEED_WORKERS" \
    --override "model=$MODEL" \
    --override "activation_capture.scope=all_probes"

  python scripts/analyze_activation_geometry.py \
    --runs "$OUT_ROOT/geometry_batch" \
    --feature "$GEOMETRY_FEATURE" \
    --out "$OUT_ROOT/geometry_batch/activation_geometry_$GEOMETRY_FEATURE" \
    $(if [[ "$EXCLUDE_SAME_CROP" == "1" ]]; then echo "--exclude-same-crop"; fi)
fi

if [[ "$RUN_T0_PROBES" == "1" ]]; then
  python -m nnd.cli batch \
    --config "$CONFIG" \
    --out "$OUT_ROOT/activation_t0_batch" \
    --backend transformers_vlm \
    --start-seed "$START_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --probe-workers "$PROBE_WORKERS" \
    --seed-workers "$SEED_WORKERS" \
    --override "model=$MODEL" \
    --override "T=0" \
    --override "activation_capture.scope=all_probes"
fi

if [[ "$RUN_LINEAR_PROBES" == "1" && "$RUN_T0_PROBES" == "1" ]]; then
  python scripts/analyze_activation_linear_probe.py \
    --runs "$OUT_ROOT/activation_t0_batch" \
    --target truth_country \
    --feature last_prompt_token \
    --out "$OUT_ROOT/activation_t0_batch/linear_probe_truth_country"

  python scripts/analyze_activation_linear_probe.py \
    --runs "$OUT_ROOT/activation_t0_batch" \
    --target informativeness_label \
    --feature mean_prompt \
    --out "$OUT_ROOT/activation_t0_batch/linear_probe_informativeness"
fi

echo
echo "LLaVA pipeline complete."
echo "Key outputs:"
echo "  $OUT_ROOT/visual_perception/color_group_summary.csv"
echo "  $OUT_ROOT/pairwise_smoke/activation_geometry_$GEOMETRY_FEATURE"
echo "  $OUT_ROOT/geometry_batch/activation_geometry_$GEOMETRY_FEATURE"
