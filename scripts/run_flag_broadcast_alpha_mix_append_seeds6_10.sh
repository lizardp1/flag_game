#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Append five more seeds to the existing broadcast alpha x composition sweeps.
# This keeps the original folder names even though the combined dataset becomes
# ten seeds after the append.

START_SEED="${START_SEED:-6}"
NUM_SEEDS="${NUM_SEEDS:-5}"
N_VALUE="${N_VALUE:-8}"

echo "Appending broadcast alpha-mix seeds ${START_SEED}..$((START_SEED + NUM_SEEDS - 1))"
echo

echo "m=3 -> results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5"
START_SEED="$START_SEED" \
NUM_SEEDS="$NUM_SEEDS" \
N_VALUE="$N_VALUE" \
INTERACTION_M=3 \
OUT_BASE="results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5" \
scripts/run_flag_broadcast_alpha_mix_count_sweep.sh

echo
echo "m=1 -> results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5_m1"
START_SEED="$START_SEED" \
NUM_SEEDS="$NUM_SEEDS" \
N_VALUE="$N_VALUE" \
INTERACTION_M=1 \
OUT_BASE="results/flag_game_broadcast/alpha_mix_count_sweep_triangle28_N${N_VALUE}_seeds5_m1" \
scripts/run_flag_broadcast_alpha_mix_count_sweep.sh

echo
echo "Broadcast alpha-mix append finished."
