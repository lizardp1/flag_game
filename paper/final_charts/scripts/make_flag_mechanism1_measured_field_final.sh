#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/flag_game/proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5/04_model_mix_alpha_neutral}"
PROBES="${PROBES:-$ROOT/results/flag_game/empirical_crop_field_probes/france_gpt4o_B50_crops96/results.csv}"
STEM="${STEM:-flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003}"
CONDITION="${CONDITION:-all_gpt_4o}"
MODEL="${MODEL:-gpt-4o}"
PANEL_TARGET="${PANEL_TARGET:-France}"
RIVAL="${RIVAL-Peru}"
SIM_TARGET="${SIM_TARGET-France}"
PHASE_TRIALS="${PHASE_TRIALS:-600}"
J_STAR="${J_STAR:-1.785}"
J_MSG="${J_MSG:-$J_STAR}"
ALPHA="${ALPHA:-0.20}"
BETA="${BETA:-0.827}"
KAPPA="${KAPPA:-12}"
H_TRUTH="${H_TRUTH:-1.374}"
H_RIVAL="${H_RIVAL:--1.066}"
SIGMA_AMBIGUOUS="${SIGMA_AMBIGUOUS:-0.408}"
MIXTURE_MODE="${MIXTURE_MODE:-empirical_anchor}"
MAX_INFORMATIVE_MASS="${MAX_INFORMATIVE_MASS:-0.98}"
EMPIRICAL_ANCHOR_TRUTH_MASS="${EMPIRICAL_ANCHOR_TRUTH_MASS:-0.566}"
EMPIRICAL_ANCHOR_RIVAL_MASS="${EMPIRICAL_ANCHOR_RIVAL_MASS:-0.333}"
EMPIRICAL_ANCHOR_TRUTH_TAU="${EMPIRICAL_ANCHOR_TRUTH_TAU:-22.0}"
EMPIRICAL_ANCHOR_RIVAL_TAU="${EMPIRICAL_ANCHOR_RIVAL_TAU:-28.138}"
EMPIRICAL_ANCHOR_RIVAL_POWER="${EMPIRICAL_ANCHOR_RIVAL_POWER:-1.512}"
CROP_SET_MODE="${CROP_SET_MODE:-exact}"

args=(
  python3 "$ROOT/paper/make_flag_game_mechanism1_measured_field_final.py"
  --probe-input "$PROBES"
  --mechanism-run-root "$RUN_ROOT"
  --mechanism-condition "$CONDITION"
  --model "$MODEL"
  --panel-target-country "$PANEL_TARGET"
  --rival-country "$RIVAL"
  --simulation-target-country "$SIM_TARGET"
  --phase-trials "$PHASE_TRIALS"
  --j-star "$J_STAR"
  --j-msg "$J_MSG"
  --alpha "$ALPHA"
  --beta "$BETA"
  --kappa "$KAPPA"
  --h-truth "$H_TRUTH"
  --h-rival "$H_RIVAL"
  --sigma-ambiguous "$SIGMA_AMBIGUOUS"
  --mixture-mode "$MIXTURE_MODE"
  --max-informative-mass "$MAX_INFORMATIVE_MASS"
  --empirical-anchor-truth-mass "$EMPIRICAL_ANCHOR_TRUTH_MASS"
  --empirical-anchor-rival-mass "$EMPIRICAL_ANCHOR_RIVAL_MASS"
  --empirical-anchor-truth-tau "$EMPIRICAL_ANCHOR_TRUTH_TAU"
  --empirical-anchor-rival-tau "$EMPIRICAL_ANCHOR_RIVAL_TAU"
  --empirical-anchor-rival-power "$EMPIRICAL_ANCHOR_RIVAL_POWER"
  --crop-set-mode "$CROP_SET_MODE"
  --out-stem "$STEM"
)

if [[ "${ALLOW_MISSING:-0}" == "1" ]]; then
  args+=(--allow-missing-mechanism-crops)
fi

SEED_FILTER="${SEED_FILTER:-seed_0003}"
if [[ -n "${SEED_FILTER:-}" ]]; then
  args+=(--seed-filter "$SEED_FILTER")
fi

if [[ -n "${MAX_SEEDS_PER_N:-}" ]]; then
  args+=(--max-seeds-per-n "$MAX_SEEDS_PER_N")
fi

exec "${args[@]}"
