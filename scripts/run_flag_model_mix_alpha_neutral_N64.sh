#!/usr/bin/env bash
set -euo pipefail

# Closed-list/prior-country neutral-alpha N=64 extension.
# Matches the current proposal-facing 04_model_mix_alpha_neutral N sweep:
# seeds 1..10, pure controls plus the mixed one-boosted-agent condition,
# 6x4 crops, scale 25, high detail, m=3, closed country list.

N_VALUES="${N_VALUES:-64}" \
START_SEED="${START_SEED:-1}" \
NUM_SEEDS="${NUM_SEEDS:-10}" \
N64_ROUNDS="${N64_ROUNDS:-32}" \
COUNTRY_POOL="${COUNTRY_POOL:-stripe_expanded_24}" \
PROMPT_STYLE="${PROMPT_STYLE:-closed_country_list}" \
INCLUDE_MIXED_CONDITION="${INCLUDE_MIXED_CONDITION:-true}" \
bash scripts/run_flag_model_mix_alpha_neutral.sh
