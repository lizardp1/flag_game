#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Append 50 more same-seed N=8 protocol-comparison trials for the data behind
# flag_game_protocol_side_by_side_N8.
#
# Defaults append seeds 0010..0059 to the existing 10-seed result root. The
# preflight refuses to run if any requested seed already appears in the result
# tree or summary CSVs, so existing data is not overwritten accidentally.

RESULT_ROOT="${RESULT_ROOT:-results/flag_game_side_by_side/protocol_side_by_side_N8_m3_full_10seed}"
START_SEED="${START_SEED:-10}"
NUM_SEEDS="${NUM_SEEDS:-50}"
COUNTRY_POOL="${COUNTRY_POOL:-stripe_plus_real_triangle_28}"
BACKEND="${BACKEND:-openai}"
COMPARISON_MODEL="${COMPARISON_MODEL:-gpt-4o}"
PRESTIGE_MODEL="${PRESTIGE_MODEL:-gpt-5.4}"
N_VALUE="${N_VALUE:-8}"
ROUNDS="${ROUNDS:-10}"
INTERACTION_M="${INTERACTION_M:-3}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
SEED_WORKERS="${SEED_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"
REFRESH_PAPER_EXPORTS="${REFRESH_PAPER_EXPORTS:-1}"
SUMMARY_START_SEED="${SUMMARY_START_SEED:-0}"

is_truthy() {
  [[ "$1" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

if ! [[ "$START_SEED" =~ ^[0-9]+$ ]]; then
  echo "START_SEED must be a non-negative integer, got: ${START_SEED}"
  exit 1
fi
if ! [[ "$NUM_SEEDS" =~ ^[0-9]+$ ]]; then
  echo "NUM_SEEDS must be a positive integer, got: ${NUM_SEEDS}"
  exit 1
fi
if (( NUM_SEEDS < 1 )); then
  echo "NUM_SEEDS must be >= 1"
  exit 1
fi
if ! [[ "$SUMMARY_START_SEED" =~ ^[0-9]+$ ]]; then
  echo "SUMMARY_START_SEED must be a non-negative integer, got: ${SUMMARY_START_SEED}"
  exit 1
fi
if [[ "$N_VALUE" != "8" ]]; then
  echo "This append script is for flag_game_protocol_side_by_side_N8; N_VALUE must be 8."
  exit 1
fi
if [[ "$INTERACTION_M" != "3" ]]; then
  echo "This append script is for the m=3 protocol comparison; INTERACTION_M must be 3."
  exit 1
fi
if [[ "$BACKEND" == "openai" && -z "${OPENAI_API_KEY:-}" ]] && ! is_truthy "$DRY_RUN"; then
  echo "OPENAI_API_KEY is not set."
  echo "Export your API key first, or run with DRY_RUN=1 to inspect the command."
  exit 1
fi

python - "$RESULT_ROOT" "$START_SEED" "$NUM_SEEDS" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
start_seed = int(sys.argv[2])
num_seeds = int(sys.argv[3])
target_seeds = set(range(start_seed, start_seed + num_seeds))

def parse_seed(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None

def seed_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None

conflicts: list[str] = []

if result_root.exists():
    for seed_dir in result_root.glob("full/*/*/seed_*"):
        seed = seed_from_path(seed_dir)
        if seed in target_seeds:
            conflicts.append(str(seed_dir))

csv_paths = [
    result_root / "paired_summary_all_pools.csv",
    result_root / "full" / "paired_summary.csv",
    result_root / "full" / "org" / "role_mix_condition_results.csv",
    result_root / "full" / "broadcast" / "broadcast_role_slot_condition_results.csv",
    result_root / "full" / "pairwise" / "pairwise_role_slot_condition_results.csv",
    result_root / "side_by_side_seed_metrics.csv",
]
for csv_path in csv_paths:
    if not csv_path.exists():
        continue
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seed = parse_seed(row.get("seed"))
            if seed in target_seeds:
                conflicts.append(f"{csv_path}: seed {seed}")

if conflicts:
    print("Refusing to run because requested append seeds already exist:")
    for item in conflicts[:80]:
        print(f"- {item}")
    if len(conflicts) > 80:
        print(f"- ... {len(conflicts) - 80} more")
    sys.exit(2)

end_seed = start_seed + num_seeds - 1
print(
    "Preflight OK: no existing directories or CSV rows found for "
    f"seeds {start_seed:04d}..{end_seed:04d} under {result_root}."
)
PY

END_SEED=$((START_SEED + NUM_SEEDS - 1))
SUMMARY_NUM_SEEDS=$((END_SEED - SUMMARY_START_SEED + 1))
if (( SUMMARY_NUM_SEEDS < NUM_SEEDS )); then
  echo "SUMMARY_START_SEED must be <= START_SEED; got ${SUMMARY_START_SEED} > ${START_SEED}."
  exit 1
fi
echo "Appending N=8 protocol side-by-side seeds:"
echo "  result root: ${RESULT_ROOT}"
echo "  seeds: $(printf '%04d' "$START_SEED")..$(printf '%04d' "$END_SEED")"
echo "  paired summary rebuild: $(printf '%04d' "$SUMMARY_START_SEED")..$(printf '%04d' "$END_SEED")"
echo "  country pool: ${COUNTRY_POOL}"
echo "  backend: ${BACKEND}"
echo "  models: ${COMPARISON_MODEL}, ${PRESTIGE_MODEL}"
echo "  agent workers: ${AGENT_WORKERS}"
echo "  seed workers: ${SEED_WORKERS}"
echo "  refresh paper exports: ${REFRESH_PAPER_EXPORTS}"
echo

if is_truthy "$DRY_RUN"; then
  echo "DRY_RUN=1, so no experiments were launched."
  echo "Command that would run:"
  printf '  PYTHONPATH=. python scripts/run_flag_protocol_side_by_side.py'
  printf ' --out %q' "$RESULT_ROOT"
  printf ' --pool %q' "$COUNTRY_POOL"
  printf ' --start-seed %q' "$START_SEED"
  printf ' --num-seeds %q' "$NUM_SEEDS"
  printf ' --backend %q' "$BACKEND"
  printf ' --comparison-model %q' "$COMPARISON_MODEL"
  printf ' --prestige-model %q' "$PRESTIGE_MODEL"
  printf ' --n %q' "$N_VALUE"
  printf ' --rounds %q' "$ROUNDS"
  printf ' --interaction-m %q' "$INTERACTION_M"
  printf ' --agent-workers %q' "$AGENT_WORKERS"
  printf ' --seed-workers %q' "$SEED_WORKERS"
  printf ' --no-make-visuals\n'
  echo "Then it would rebuild paired summaries without rerunning trials:"
  printf '  PYTHONPATH=. python scripts/run_flag_protocol_side_by_side.py'
  printf ' --out %q' "$RESULT_ROOT"
  printf ' --pool %q' "$COUNTRY_POOL"
  printf ' --start-seed %q' "$SUMMARY_START_SEED"
  printf ' --num-seeds %q' "$SUMMARY_NUM_SEEDS"
  printf ' --backend %q' "$BACKEND"
  printf ' --comparison-model %q' "$COMPARISON_MODEL"
  printf ' --prestige-model %q' "$PRESTIGE_MODEL"
  printf ' --n %q' "$N_VALUE"
  printf ' --rounds %q' "$ROUNDS"
  printf ' --interaction-m %q' "$INTERACTION_M"
  printf ' --agent-workers %q' "$AGENT_WORKERS"
  printf ' --seed-workers %q' "$SEED_WORKERS"
  printf ' --no-run --make-visuals\n'
  exit 0
fi

PYTHONPATH=. python scripts/run_flag_protocol_side_by_side.py \
  --out "$RESULT_ROOT" \
  --pool "$COUNTRY_POOL" \
  --start-seed "$START_SEED" \
  --num-seeds "$NUM_SEEDS" \
  --backend "$BACKEND" \
  --comparison-model "$COMPARISON_MODEL" \
  --prestige-model "$PRESTIGE_MODEL" \
  --n "$N_VALUE" \
  --rounds "$ROUNDS" \
  --interaction-m "$INTERACTION_M" \
  --agent-workers "$AGENT_WORKERS" \
  --seed-workers "$SEED_WORKERS" \
  --no-make-visuals

PYTHONPATH=. python scripts/run_flag_protocol_side_by_side.py \
  --out "$RESULT_ROOT" \
  --pool "$COUNTRY_POOL" \
  --start-seed "$SUMMARY_START_SEED" \
  --num-seeds "$SUMMARY_NUM_SEEDS" \
  --backend "$BACKEND" \
  --comparison-model "$COMPARISON_MODEL" \
  --prestige-model "$PRESTIGE_MODEL" \
  --n "$N_VALUE" \
  --rounds "$ROUNDS" \
  --interaction-m "$INTERACTION_M" \
  --agent-workers "$AGENT_WORKERS" \
  --seed-workers "$SEED_WORKERS" \
  --no-run \
  --make-visuals

python poster/make_flag_protocol_side_by_side_visuals.py \
  --result-root "$RESULT_ROOT/full" \
  --output-dir "$RESULT_ROOT" \
  --out-stem "side_by_side" \
  --skip-prelim

if is_truthy "$REFRESH_PAPER_EXPORTS"; then
  python paper/make_flag_protocol_side_by_side_visuals.py
fi

echo
echo "Done. Appended seeds $(printf '%04d' "$START_SEED")..$(printf '%04d' "$END_SEED") without overwriting existing seeds."
