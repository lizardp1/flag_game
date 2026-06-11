#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
CONFIG="${CONFIG:-configs/open_models/llava_1_6_mistral_7b_visual_report.yaml}"
OUT="${OUT:-runs/llava_visual_report_run}"
SEED="${SEED:-0}"
TRUTH_COUNTRY="${TRUTH_COUNTRY:-Czech Republic}"
COUNTRY_POOL="${COUNTRY_POOL:-stripe_plus_real_triangle_28}"
N="${N:-5}"
T="${T:-8}"
H="${H:-4}"
TILE="${TILE:-6x4}"
MAX_TOKENS="${MAX_TOKENS:-180}"
TEMPERATURE="${TEMPERATURE:-0.0}"

export NND_TRANSFORMERS_VLM_FAMILY="${NND_TRANSFORMERS_VLM_FAMILY:-llava_next}"
export NND_TRANSFORMERS_DTYPE="${NND_TRANSFORMERS_DTYPE:-bfloat16}"
export NND_TRANSFORMERS_DEVICE_MAP="${NND_TRANSFORMERS_DEVICE_MAP:-auto}"
export NND_TRANSFORMERS_ATTN_IMPLEMENTATION="${NND_TRANSFORMERS_ATTN_IMPLEMENTATION:-auto}"

tile_width="${TILE%x*}"
tile_height="${TILE#*x}"
if [[ "$tile_width" == "$TILE" || -z "$tile_width" || -z "$tile_height" ]]; then
  echo "TILE must look like WIDTHxHEIGHT, got: $TILE" >&2
  exit 2
fi

echo "LLaVA visual report run"
echo "  model: $MODEL"
echo "  out:   $OUT"
echo "  seed:  $SEED"
echo "  N/T/H: $N/$T/$H"
echo "  truth: $TRUTH_COUNTRY"
echo "  pool:  $COUNTRY_POOL"
echo "  tile:  $TILE"

python -m nnd.cli run \
  --config "$CONFIG" \
  --out "$OUT" \
  --seed "$SEED" \
  --backend transformers_vlm \
  --override "model=$MODEL" \
  --override "country_pool=$COUNTRY_POOL" \
  --override "fixed_truth_country=$TRUTH_COUNTRY" \
  --override "N=$N" \
  --override "T=$T" \
  --override "H=$H" \
  --override "tile_width=$tile_width" \
  --override "tile_height=$tile_height" \
  --override "temperature=$TEMPERATURE" \
  --override "max_tokens=$MAX_TOKENS" \
  --override "output.save_crop_images=true" \
  --override "output.make_plots=true" \
  --override "activation_capture.enabled=false"

python scripts/make_flag_game_run_card.py --run "$OUT"

echo
echo "Visual report run complete."
echo "Key outputs:"
echo "  $OUT/visual_report.html"
echo "  $OUT/plots/run_card.png"
echo "  $OUT/plots/country_share_trajectories.png"
echo "  $OUT/plots/run_overview.png"
echo "  $OUT/artifacts/truth_flag.png"
echo "  $OUT/artifacts/agent_00_crop.png"
