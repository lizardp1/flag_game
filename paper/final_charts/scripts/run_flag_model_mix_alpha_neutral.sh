#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Export your API key first, then rerun this script."
  exit 1
fi

# Neutral-alpha rerun of the modern 04_model_mix panel.
#
# This intentionally writes a sibling stage instead of touching the completed
# prompted-alpha 04_model_mix folders:
#   04_model_mix_alpha_neutral/N4,N8,N16,N32
#
# Neutral alpha means:
#   social_susceptibility=0.5
#   prompt_social_susceptibility=false

CONFIG="${CONFIG:-configs/flag_game/stripe_expanded_highres_m3_unprompted_v1.yaml}"
OUT_BASE="${OUT_BASE:-results/flag_game/proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5}"
OUT_STAGE="${OUT_STAGE:-${OUT_BASE}/04_model_mix_alpha_neutral}"
COUNTRY_POOL="${COUNTRY_POOL:-stripe_expanded_24}"
PROMPT_STYLE="${PROMPT_STYLE:-closed_country_list}"

# This script now defaults to the five-seed append block for the completed
# seven-seed neutral-alpha run. Override START_SEED/NUM_SEEDS to rerun a
# different block.
START_SEED="${START_SEED:-8}"
NUM_SEEDS="${NUM_SEEDS:-5}"
SEED_WORKERS="${SEED_WORKERS:-2}"

N_VALUES="${N_VALUES:-4,8,16,32}"
BOOSTED_AGENT_ID="${BOOSTED_AGENT_ID:-0}"
BASELINE_MODEL="${BASELINE_MODEL:-gpt-4o}"
BOOSTED_MODEL="${BOOSTED_MODEL:-gpt-5.4}"
INCLUDE_MIXED_CONDITION="${INCLUDE_MIXED_CONDITION:-true}"

DEFAULT_ROUNDS="${DEFAULT_ROUNDS:-20}"
N16_ROUNDS="${N16_ROUNDS:-24}"
N32_ROUNDS="${N32_ROUNDS:-32}"
N64_ROUNDS="${N64_ROUNDS:-32}"
N128_ROUNDS="${N128_ROUNDS:-32}"

COMMON_OVERRIDES=(
  --override H=8
  --override "country_pool=${COUNTRY_POOL}"
  --override "prompt_style=${PROMPT_STYLE}"
  --override tile_width=6
  --override tile_height=4
  --override render_scale=25
  --override image_detail=high
  --override early_stop_probe_window=5
  --override output.make_plots=true
  --override social_susceptibility=0.5
  --override prompt_social_susceptibility=false
  --override observation_overlap=null
  --override interaction_m=3
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

rounds_for_n() {
  local n="$1"
  case "$n" in
    4|8)
      printf '%s\n' "$DEFAULT_ROUNDS"
      ;;
    16)
      printf '%s\n' "$N16_ROUNDS"
      ;;
    32)
      printf '%s\n' "$N32_ROUNDS"
      ;;
    64)
      printf '%s\n' "$N64_ROUNDS"
      ;;
    128)
      printf '%s\n' "$N128_ROUNDS"
      ;;
    *)
      printf '%s\n' "$DEFAULT_ROUNDS"
      ;;
  esac
}

model_mix_run_complete() {
  local out_dir="$1"
  python - <<PY
from pathlib import Path
import csv
import re

path = Path("${out_dir}") / "model_mix_condition_results.csv"
expected_seeds = set(range(int("${START_SEED}"), int("${START_SEED}") + int("${NUM_SEEDS}")))
baseline_slug = re.sub(r"[^A-Za-z0-9]+", "_", "${BASELINE_MODEL}").strip("_").lower()
boosted_slug = re.sub(r"[^A-Za-z0-9]+", "_", "${BOOSTED_MODEL}").strip("_").lower()
expected_conditions = {
    f"all_{baseline_slug}",
    f"all_{boosted_slug}",
}
if "${INCLUDE_MIXED_CONDITION}".strip().lower() not in {"0", "false", "no", "off"}:
    expected_conditions.add(f"mix_{baseline_slug}_plus_1_{boosted_slug}")
if not path.exists():
    raise SystemExit(1)
with path.open() as handle:
    rows = list(csv.DictReader(handle))
present = {condition: set() for condition in expected_conditions}
for row in rows:
    condition = row.get("condition_name")
    seed = row.get("seed")
    if condition in present and seed not in (None, ""):
        present[condition].add(int(float(seed)))
complete = all(expected_seeds.issubset(seeds) for seeds in present.values())
raise SystemExit(0 if complete else 1)
PY
}

assert_existing_spec_matches() {
  local out_dir="$1"
  local n="$2"
  local t="$3"
  python - <<PY
from pathlib import Path
import json

path = Path("${out_dir}") / "model_mix_spec.json"
if not path.exists():
    raise SystemExit(0)

raw_spec = json.loads(path.read_text())
specs = raw_spec.get("runs", [raw_spec]) if isinstance(raw_spec, dict) else [raw_spec]
expected = {
    "baseline_model": "${BASELINE_MODEL}",
    "boosted_model": "${BOOSTED_MODEL}",
    "boosted_agent_ids": [int("${BOOSTED_AGENT_ID}")],
    "N": int("${n}"),
    "T": int("${t}"),
    "H": 8,
    "interaction_m": 3,
    "social_susceptibility": 0.5,
    "prompt_social_susceptibility": False,
    "prompt_style": "${PROMPT_STYLE}",
    "country_pool": "${COUNTRY_POOL}",
    "tile_width": 6,
    "tile_height": 4,
    "observation_overlap": None,
    "render_scale": 25,
    "speaker_weights": None,
    "include_pure_controls": True,
    "include_mixed_condition": "${INCLUDE_MIXED_CONDITION}".strip().lower() not in {"0", "false", "no", "off"},
}

def mismatches(spec):
    problems = []
    for key, value in expected.items():
        existing = spec.get(key)
        if key == "prompt_style" and existing is None and value == "closed_country_list":
            continue
        if key == "include_mixed_condition" and existing is None and value is True:
            continue
        if existing != value:
            problems.append(f"{key}: existing={spec.get(key)!r}, expected={value!r}")
    return problems

all_problems = [mismatches(spec) for spec in specs if isinstance(spec, dict)]
if not all_problems or all(all_problems):
    print(f"Existing model-mix spec at {path} does not match this neutral-alpha run:")
    problems = all_problems[0] if all_problems else ["spec file is not a JSON object"]
    for problem in problems:
        print(f"  - {problem}")
    print("Use a different OUT_STAGE or move the mismatched partial output aside before rerunning.")
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_model_mix_alpha_neutral_for_n() {
  local n="$1"
  local rounds
  rounds="$(rounds_for_n "$n")"
  local t=$((rounds * n))
  local probe_every=$((n / 2))
  if (( probe_every < 1 )); then
    probe_every=1
  fi

  local out_dir="${OUT_STAGE}/N${n}"
  assert_existing_spec_matches "$out_dir" "$n" "$t"

  if model_mix_run_complete "$out_dir"; then
    echo "Skipping completed neutral-alpha model-mix condition: ${out_dir}"
    return
  fi

  echo "Running neutral-alpha model-mix: N=${n}, rounds=${rounds}, seeds ${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
  echo "  output=${out_dir}"
  echo "  models=${BASELINE_MODEL}, ${BOOSTED_MODEL}"

  if [[ "${INCLUDE_MIXED_CONDITION}" =~ ^(0|false|False|no|NO|off|OFF)$ ]]; then
    echo "  mixed condition: skipped"
    PYTHONPATH=. python -m nnd.flag_game.cli model-mix \
      --config "$CONFIG" \
      --out "$out_dir" \
      --start-seed "$START_SEED" \
      --num-seeds "$NUM_SEEDS" \
      --backend openai \
      --seed-workers "$SEED_WORKERS" \
      --baseline-model "$BASELINE_MODEL" \
      --boosted-model "$BOOSTED_MODEL" \
      --boosted-agent-id "$BOOSTED_AGENT_ID" \
      --skip-mixed-condition \
      "${COMMON_OVERRIDES[@]}" \
      --override "N=${n}" \
      --override "T=${t}" \
      --override "probe_every=${probe_every}"
  else
    echo "  mixed condition: agent ${BOOSTED_AGENT_ID} as ${BOOSTED_MODEL}"
    PYTHONPATH=. python -m nnd.flag_game.cli model-mix \
      --config "$CONFIG" \
      --out "$out_dir" \
      --start-seed "$START_SEED" \
      --num-seeds "$NUM_SEEDS" \
      --backend openai \
      --seed-workers "$SEED_WORKERS" \
      --baseline-model "$BASELINE_MODEL" \
      --boosted-model "$BOOSTED_MODEL" \
      --boosted-agent-id "$BOOSTED_AGENT_ID" \
      "${COMMON_OVERRIDES[@]}" \
      --override "N=${n}" \
      --override "T=${t}" \
      --override "probe_every=${probe_every}"
  fi
}

main() {
  local -a n_values
  csv_to_array "$N_VALUES"
  n_values=("${CSV_TO_ARRAY_RESULT[@]}")

  echo "Modern Flag Game neutral-alpha model-mix runner"
  echo "  output stage: ${OUT_STAGE}"
  echo "  config: ${CONFIG}"
  echo "  seeds: ${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
  echo "  N values: ${N_VALUES}"
  echo "  tile/render: 6x4, render_scale=25, high, m=3"
  echo "  alpha condition: neutral alpha (prompt_social_susceptibility=false, social_susceptibility=0.5)"
  echo "  country_pool: ${COUNTRY_POOL}"
  echo "  prompt_style: ${PROMPT_STYLE}"
  echo "  include_mixed_condition: ${INCLUDE_MIXED_CONDITION}"
  echo

  local n
  for n in "${n_values[@]}"; do
    n="$(trim "$n")"
    [[ -z "$n" ]] && continue
    run_model_mix_alpha_neutral_for_n "$n"
  done

  echo
  echo "Neutral-alpha model-mix runner finished."
  echo "Output stage: ${OUT_STAGE}"
}

main "$@"
