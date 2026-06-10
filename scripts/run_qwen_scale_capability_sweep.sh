#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml}"
OUT_ROOT="${OUT_ROOT:-runs/qwen_scale_capability}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-8}"
N_AGENTS="${N_AGENTS:-8}"
T_STEPS="${T_STEPS:-0}"
PROBE_WORKERS="${PROBE_WORKERS:-1}"
SEED_WORKERS="${SEED_WORKERS:-1}"
GEOMETRY_FEATURE="${GEOMETRY_FEATURE:-last_prompt_token}"
RUN_GEOMETRY="${RUN_GEOMETRY:-1}"

if [[ -z "${MODELS:-}" ]]; then
  MODELS="Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct"
fi

mkdir -p "$OUT_ROOT"

echo "Qwen scale capability sweep"
echo "  config: $CONFIG"
echo "  out: $OUT_ROOT"
echo "  models: $MODELS"
echo "  seeds: $START_SEED .. $((START_SEED + NUM_SEEDS - 1))"
echo "  N=$N_AGENTS T=$T_STEPS"

for model in $MODELS; do
  slug="$(python - "$model" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", sys.argv[1]).strip("_"))
PY
)"
  model_out="$OUT_ROOT/$slug"
  echo
  echo "=== Running $model ==="
  python -m nnd.cli batch \
    --config "$CONFIG" \
    --out "$model_out" \
    --backend transformers_vlm \
    --start-seed "$START_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --probe-workers "$PROBE_WORKERS" \
    --seed-workers "$SEED_WORKERS" \
    --override "model=$model" \
    --override "N=$N_AGENTS" \
    --override "T=$T_STEPS" \
    --override "probe_every=1" \
    --override "activation_capture.scope=all_probes"

  if [[ "$RUN_GEOMETRY" == "1" ]]; then
    python scripts/analyze_activation_geometry.py \
      --runs "$model_out" \
      --feature "$GEOMETRY_FEATURE" \
      --out "$model_out/activation_geometry_$GEOMETRY_FEATURE"
  fi
done

python scripts/summarize_model_scale_comparison.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/scale_summary"

echo
echo "Scale comparison complete."
echo "Summary:"
echo "  $OUT_ROOT/scale_summary/model_scale_metrics.csv"
echo "  $OUT_ROOT/scale_summary/initial_probe_country_distribution.csv"
echo "  $OUT_ROOT/scale_summary/initial_probe_agent_rows.csv"
