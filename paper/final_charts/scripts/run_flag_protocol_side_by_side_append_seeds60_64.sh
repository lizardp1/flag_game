#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Append the next 5 N=8 protocol side-by-side seeds:
#   0060..0064
#
# This delegates to the safer append wrapper, which refuses to run if any
# requested seed already exists and then rebuilds paired summaries from seed 0
# through the new end seed.

START_SEED="${START_SEED:-60}" \
NUM_SEEDS="${NUM_SEEDS:-5}" \
SUMMARY_START_SEED="${SUMMARY_START_SEED:-0}" \
"${SCRIPT_DIR}/run_flag_protocol_side_by_side_append_seeds10_59.sh"
