#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Sweep social-susceptibility alpha across the full N=8 composition grid.
#
# Defaults:
#   alphas: 0.0, 0.25, 0.5, 0.75, 1.0
#   composition grid: 0..8 gpt-5.4 agents, with the remainder gpt-4o
#   seeds: appends 5 more seeds by default (6..10)
#
# Example:
#   scripts/run_flag_broadcast_alpha_mix_count_sweep.sh
#
# Override examples:
#   ALPHAS=0.25,0.5,0.75 NUM_SEEDS=3 scripts/run_flag_broadcast_alpha_mix_count_sweep.sh
#   OUT_BASE=results/my_alpha_sweep SEED_WORKERS=2 AGENT_WORKERS=4 scripts/run_flag_broadcast_alpha_mix_count_sweep.sh

N_VALUE="${N_VALUE:-8}"
START_SEED="${START_SEED:-6}"
NUM_SEEDS="${NUM_SEEDS:-5}"
ROUNDS="${ROUNDS:-10}"
INTERACTION_M="${INTERACTION_M:-3}"
ALPHAS="${ALPHAS:-0.0,0.25,0.5,0.75,1.0}"
PRESTIGE_COUNTS="${PRESTIGE_COUNTS:-}"

CONFIG="${CONFIG:-configs/flag_game_broadcast/stripe_expanded_highres_m3_v1.yaml}"
if [[ -z "${OUT_BASE:-}" ]]; then
  if [[ "$INTERACTION_M" == "3" ]]; then
    OUT_BASE="results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5"
  else
    OUT_BASE="results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5_m${INTERACTION_M}"
  fi
fi

SEED_WORKERS="${SEED_WORKERS:-1}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
COMPARISON_MODEL="${COMPARISON_MODEL:-gpt-4o}"
PRESTIGE_MODEL="${PRESTIGE_MODEL:-gpt-5.4}"
COUNTRY_POOL="${COUNTRY_POOL:-stripe_plus_real_triangle_28}"

if ! [[ "$N_VALUE" =~ ^[0-9]+$ ]]; then
  echo "N_VALUE must be a non-negative integer, got: ${N_VALUE}"
  exit 1
fi
if (( N_VALUE < 1 )); then
  echo "N_VALUE must be >= 1"
  exit 1
fi

if ! [[ "$NUM_SEEDS" =~ ^[0-9]+$ ]]; then
  echo "NUM_SEEDS must be a non-negative integer, got: ${NUM_SEEDS}"
  exit 1
fi
if (( NUM_SEEDS < 1 )); then
  echo "NUM_SEEDS must be >= 1"
  exit 1
fi

COUNT_ARGS=()
if [[ -n "$PRESTIGE_COUNTS" ]]; then
  IFS=',' read -r -a counts_list <<< "$PRESTIGE_COUNTS"
else
  counts_list=()
  for ((count = 0; count <= N_VALUE; count++)); do
    counts_list+=("$count")
  done
fi

for count in "${counts_list[@]}"; do
  count="$(printf '%s' "$count" | xargs)"
  [[ -z "$count" ]] && continue
  if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "Each prestige count must be a non-negative integer, got: ${count}"
    exit 1
  fi
  if (( count > N_VALUE )); then
    echo "Each prestige count must be <= N_VALUE=${N_VALUE}, got: ${count}"
    exit 1
  fi
  COUNT_ARGS+=(--prestige-count "$count")
done

IFS=',' read -r -a alpha_list <<< "$ALPHAS"

echo "Running broadcast flag-game alpha x composition sweep:"
echo "  N=${N_VALUE}"
echo "  comparison model=${COMPARISON_MODEL}"
echo "  prestige model=${PRESTIGE_MODEL}"
echo "  prestige counts=${PRESTIGE_COUNTS:-0..${N_VALUE}}"
echo "  alphas=${ALPHAS}"
echo "  rounds=${ROUNDS}"
echo "  interaction_m=${INTERACTION_M}"
echo "  seeds=${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
echo "  country_pool=${COUNTRY_POOL}"
echo "  output=${OUT_BASE}"
echo

for raw_alpha in "${alpha_list[@]}"; do
  alpha="$(printf '%s' "$raw_alpha" | xargs)"
  [[ -z "$alpha" ]] && continue
  alpha_slug="alpha_${alpha//./_}"
  alpha_out="${OUT_BASE}/${alpha_slug}"

  echo "Running alpha=${alpha}"
  echo "  output=${alpha_out}"

  PYTHONPATH=. python -m nnd.flag_game_broadcast.cli mix-sweep \
    --config "$CONFIG" \
    --out "$alpha_out" \
    --start-seed "$START_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --backend openai \
    --seed-workers "$SEED_WORKERS" \
    --agent-workers "$AGENT_WORKERS" \
    --comparison-model "$COMPARISON_MODEL" \
    --prestige-model "$PRESTIGE_MODEL" \
    "${COUNT_ARGS[@]}" \
    --skip-pure-controls \
    --override "N=${N_VALUE}" \
    --override "rounds=${ROUNDS}" \
    --override "country_pool=${COUNTRY_POOL}" \
    --override "social_susceptibility=${alpha}" \
    --override "prompt_social_susceptibility=true" \
    --override H=8 \
    --override "interaction_m=${INTERACTION_M}" \
    --override tile_width=6 \
    --override tile_height=4 \
    --override render_scale=25 \
    --override image_detail=high \
    --override output.make_plots=true

  echo
done

echo "Broadcast alpha x composition sweep finished."
echo "Output root: ${OUT_BASE}"
