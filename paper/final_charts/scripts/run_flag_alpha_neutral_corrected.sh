#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Export your API key first, then rerun this script."
  exit 1
fi

# Corrected neutral-alpha rerun for real observation-overlap regimes.
#
# This writes to a new sibling stage and leaves the legacy 06_alpha_neutral
# untouched:
#   results/flag_game/proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5/06_alpha_neutral_corrected
#
# Why this exists:
#   the legacy 06_alpha_neutral completed, but old crop sampling collapsed
#   nominal rho=0.3,0.6,0.9 to roughly the same realized overlap.

CONFIG="${CONFIG:-configs/flag_game/stripe_expanded_highres_m3_v1.yaml}"
LIVE_ROOT="${LIVE_ROOT:-results/flag_game/proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5}"
OUT_STAGE="${OUT_STAGE:-${LIVE_ROOT}/06_alpha_neutral_corrected}"

START_SEED="${START_SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-7}"
SEED_WORKERS="${SEED_WORKERS:-2}"
PROBE_WORKERS="${PROBE_WORKERS:-4}"

MODELS="${MODELS:-gpt-4o,gpt-5.4}"
OVERLAPS="${OVERLAPS:-0.0,0.3,0.6,0.9}"

NEUTRAL_N="${NEUTRAL_N:-16}"
NEUTRAL_ROUNDS="${NEUTRAL_ROUNDS:-24}"

COMMON_OVERRIDES=(
  --override H=8
  --override country_pool=stripe_expanded_24
  --override tile_width=6
  --override tile_height=4
  --override render_scale=25
  --override image_detail=high
  --override early_stop_probe_window=5
  --override output.make_plots=true
  --override interaction_m=3
  --override social_susceptibility=0.5
  --override prompt_social_susceptibility=false
)

csv_to_array() {
  local raw="$1"
  local old_ifs="$IFS"
  IFS=','
  # shellcheck disable=SC2206
  CSV_TO_ARRAY_RESULT=($raw)
  IFS="$old_ifs"
}

trim() {
  printf '%s' "$1" | xargs
}

model_stage_dir() {
  local model="$1"
  case "$model" in
    gpt-4o)
      printf 'gpt4o\n'
      ;;
    gpt-5.4)
      printf 'gpt54\n'
      ;;
    *)
      printf '%s\n' "$model" | tr '.' '_' | tr '-' '_'
      ;;
  esac
}

batch_run_complete() {
  local out_dir="$1"
  local summary="${out_dir}/batch_summary.csv"
  if [[ ! -f "$summary" ]]; then
    return 1
  fi
  local rows
  rows="$(( $(wc -l < "$summary") - 1 ))"
  [[ "$rows" -ge "$NUM_SEEDS" ]]
}

run_neutral_condition() {
  local model="$1"
  local overlap="$2"
  local model_dir="$3"

  local t=$((NEUTRAL_ROUNDS * NEUTRAL_N))
  local probe_every=$((NEUTRAL_N / 2))
  if (( probe_every < 1 )); then
    probe_every=1
  fi

  local out_dir="${OUT_STAGE}/${model_dir}/overlap_${overlap}"
  if batch_run_complete "$out_dir"; then
    echo "Skipping completed corrected neutral-alpha condition: model=${model}, overlap=${overlap}"
    return
  fi

  echo "Running corrected neutral-alpha condition: model=${model}, overlap=${overlap}, rounds=${NEUTRAL_ROUNDS}"
  echo "  output=${out_dir}"

  PYTHONPATH=. python -m nnd.flag_game.cli batch \
    --config "$CONFIG" \
    --out "$out_dir" \
    --start-seed "$START_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --backend openai \
    --seed-workers "$SEED_WORKERS" \
    --probe-workers "$PROBE_WORKERS" \
    "${COMMON_OVERRIDES[@]}" \
    --override "model=${model}" \
    --override "N=${NEUTRAL_N}" \
    --override "T=${t}" \
    --override "probe_every=${probe_every}" \
    --override "observation_overlap=${overlap}"
}

main() {
  local -a models overlaps
  csv_to_array "$MODELS"; models=("${CSV_TO_ARRAY_RESULT[@]}")
  csv_to_array "$OVERLAPS"; overlaps=("${CSV_TO_ARRAY_RESULT[@]}")

  echo "Corrected Flag Game neutral-alpha rerunner"
  echo "  output stage: ${OUT_STAGE}"
  echo "  models: ${MODELS}"
  echo "  overlaps: ${OVERLAPS}"
  echo "  N=${NEUTRAL_N}, rounds=${NEUTRAL_ROUNDS}, tile=6x4, render_scale=25, high, m=3"
  echo "  seeds: ${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
  echo "  prompt_social_susceptibility=false, social_susceptibility=0.5"
  echo

  local model overlap model_dir
  for model in "${models[@]}"; do
    model="$(trim "$model")"
    [[ -z "$model" ]] && continue
    model_dir="$(model_stage_dir "$model")"
    for overlap in "${overlaps[@]}"; do
      overlap="$(trim "$overlap")"
      [[ -z "$overlap" ]] && continue
      run_neutral_condition "$model" "$overlap" "$model_dir"
    done
  done

  echo
  echo "Corrected neutral-alpha rerunner finished."
  echo "Output stage: ${OUT_STAGE}"
}

main "$@"
