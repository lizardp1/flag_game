#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Append the next 10 seeds for the exact data slice used by
# flag_game_broadcast_m3_diversity_main_with_empirical_metrics:
#   N=8, m=3, prompted social evidence uptake alpha=1.0,
#   and every GPT-5.4 composition count 0..8.
#
# The preflight below refuses to run if any requested seed already appears in
# a target seed directory or existing sweep CSV metadata.

CONFIG="${CONFIG:-configs/flag_game_broadcast/stripe_expanded_highres_m3_v1.yaml}"
OUT_BASE="${OUT_BASE:-results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N8_seeds5}"
START_SEED="${START_SEED:-11}"
NUM_SEEDS="${NUM_SEEDS:-10}"
N_VALUE="${N_VALUE:-8}"
ROUNDS="${ROUNDS:-10}"
INTERACTION_M="${INTERACTION_M:-3}"
ALPHA="${ALPHA:-1.0}"
PRESTIGE_COUNTS="${PRESTIGE_COUNTS:-0,1,2,3,4,5,6,7,8}"
SEED_WORKERS="${SEED_WORKERS:-1}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
COMPARISON_MODEL="${COMPARISON_MODEL:-gpt-4o}"
PRESTIGE_MODEL="${PRESTIGE_MODEL:-gpt-5.4}"
COUNTRY_POOL="${COUNTRY_POOL:-stripe_plus_real_triangle_28}"
REFRESH_EXPORTS="${REFRESH_EXPORTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "$START_SEED" =~ ^[0-9]+$ ]]; then
  echo "START_SEED must be a non-negative integer, got: ${START_SEED}"
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
if [[ "$N_VALUE" != "8" ]]; then
  echo "This script is for the N=8 diversity chart; N_VALUE must be 8."
  exit 1
fi
if [[ "$INTERACTION_M" != "3" ]]; then
  echo "This script is for the m=3 diversity chart; INTERACTION_M must be 3."
  exit 1
fi
if [[ "$ALPHA" != "1.0" && "$ALPHA" != "1" ]]; then
  echo "This script is for the alpha=1.0 diversity chart; ALPHA must be 1.0."
  exit 1
fi
ALPHA="1.0"

python - "$OUT_BASE" "$ALPHA" "$START_SEED" "$NUM_SEEDS" "$N_VALUE" "$PRESTIGE_COUNTS" "$COMPARISON_MODEL" "$PRESTIGE_MODEL" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
alpha = sys.argv[2]
start_seed = int(sys.argv[3])
num_seeds = int(sys.argv[4])
n_value = int(sys.argv[5])
counts = [int(part.strip()) for part in sys.argv[6].split(",") if part.strip()]
comparison_model = sys.argv[7]
prestige_model = sys.argv[8]

if sorted(counts) != list(range(n_value + 1)):
    raise SystemExit(
        f"PRESTIGE_COUNTS must cover every chart composition count 0..{n_value}; got {counts}"
    )

def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

def condition_name(count: int) -> str:
    if count == 0:
        return f"all_{slug(comparison_model)}"
    if count == n_value:
        return f"all_{slug(prestige_model)}"
    return f"mix_{slug(comparison_model)}_plus_{count}_{slug(prestige_model)}"

def parse_seed(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None

alpha_slug = f"alpha_{alpha.replace('.', '_')}"
alpha_root = out_base / alpha_slug
target_seeds = set(range(start_seed, start_seed + num_seeds))
target_conditions = {condition_name(count) for count in counts}

conflicts: list[str] = []
for count in counts:
    name = condition_name(count)
    condition_dir = alpha_root / name
    for seed in sorted(target_seeds):
        seed_dir = condition_dir / f"seed_{seed:04d}"
        if seed_dir.exists():
            conflicts.append(str(seed_dir))

    batch_summary = condition_dir / "batch_summary.csv"
    if batch_summary.exists():
        with batch_summary.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row_seed = parse_seed(row.get("seed"))
                if row_seed in target_seeds:
                    conflicts.append(f"{batch_summary}: seed {row_seed}")

mix_results = alpha_root / "mix_condition_results.csv"
if mix_results.exists():
    with mix_results.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row_seed = parse_seed(row.get("seed"))
            if row.get("condition_name") in target_conditions and row_seed in target_seeds:
                conflicts.append(f"{mix_results}: {row.get('condition_name')} seed {row_seed}")

if conflicts:
    print("Refusing to run because target seeds already exist:")
    for item in conflicts[:50]:
        print(f"- {item}")
    if len(conflicts) > 50:
        print(f"- ... {len(conflicts) - 50} more")
    sys.exit(2)

print(
    "Preflight OK: no existing target seed directories or CSV rows found for "
    f"{len(target_conditions)} conditions x {len(target_seeds)} seeds under {alpha_root}."
)
PY

echo "Appending broadcast m=3 diversity chart seeds:"
echo "  output=${OUT_BASE}"
echo "  alpha=${ALPHA}"
echo "  N=${N_VALUE}"
echo "  interaction_m=${INTERACTION_M}"
echo "  composition counts=${PRESTIGE_COUNTS}"
echo "  seeds=${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
echo "  comparison model=${COMPARISON_MODEL}"
echo "  prestige model=${PRESTIGE_MODEL}"
echo "  refresh exports=${REFRESH_EXPORTS}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, so no experiments were launched."
  exit 0
fi

ALPHAS="$ALPHA" \
START_SEED="$START_SEED" \
NUM_SEEDS="$NUM_SEEDS" \
N_VALUE="$N_VALUE" \
ROUNDS="$ROUNDS" \
INTERACTION_M="$INTERACTION_M" \
PRESTIGE_COUNTS="$PRESTIGE_COUNTS" \
CONFIG="$CONFIG" \
OUT_BASE="$OUT_BASE" \
SEED_WORKERS="$SEED_WORKERS" \
AGENT_WORKERS="$AGENT_WORKERS" \
COMPARISON_MODEL="$COMPARISON_MODEL" \
PRESTIGE_MODEL="$PRESTIGE_MODEL" \
COUNTRY_POOL="$COUNTRY_POOL" \
scripts/run_flag_broadcast_alpha_mix_count_sweep.sh

if [[ "$REFRESH_EXPORTS" == "1" ]]; then
  echo
  echo "Refreshing broadcast aggregate CSVs and paper figures..."
  python poster/make_flag_broadcast_alpha_mix_visuals.py \
    --result-root "$OUT_BASE" \
    --out-stem "flag_broadcast_alpha_mix_N8_m3"
  python paper/make_flag_broadcast_visuals.py
fi

echo
echo "Done. Appended seeds ${START_SEED}..$((START_SEED + NUM_SEEDS - 1)) without overwriting existing seeds."
