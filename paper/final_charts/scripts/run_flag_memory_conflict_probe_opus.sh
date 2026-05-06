#!/usr/bin/env bash
set -euo pipefail

# Run the full private-vs-social-evidence memory-conflict probe on Claude Opus.
#
# Defaults match the existing GPT/Claude grid:
#   - model: Claude Opus
#   - m: 1 and 3
#   - target countries: Czech Republic, Peru, Guinea, Bahamas
#   - private evidence: strong and weak crops
#   - social evidence: agrees and contradicts where possible
#   - social-evidence entries: 0 through 8 out of 8 memories
#   - replicates: 3 order/list shuffles per cell
#
# Useful overrides:
#   DRY_RUN=1      write trial_plan.csv only
#   LIMIT=20       run only the first 20 planned trials for a smoke test
#   FORCE=1        rerun trials already present in results.jsonl
#   REPORT_ONLY=1  rebuild summaries/plots from existing results
#   REPLICATES=5   increase replicates per cell
#   OPUS_MODEL=... override the Anthropic model id

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set."
  echo "Export your API key first, then rerun this script."
  exit 1
fi

set --
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  set -- "$@" --dry-run
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  set -- "$@" --force
fi
if [[ "${REPORT_ONLY:-0}" == "1" ]]; then
  set -- "$@" --report-only
fi
if [[ -n "${LIMIT:-}" ]]; then
  set -- "$@" --limit "$LIMIT"
fi

python scripts/run_flag_memory_conflict_probe.py \
  --out "${OUT:-results/flag_game/memory_conflict_probe_opus47}" \
  --backend anthropic \
  --model "${OPUS_MODEL:-claude-opus-4-7}" \
  --m 1 \
  --m 3 \
  --truth-country "Czech Republic" \
  --truth-country "Peru" \
  --truth-country "Guinea" \
  --truth-country "Bahamas" \
  --crop-condition diagnostic_true \
  --crop-condition ambiguous_true \
  --lure-relation compatible \
  --lure-relation incompatible \
  --memory-counts "0,1,2,3,4,5,6,7,8" \
  --h 8 \
  --replicates "${REPLICATES:-3}" \
  --seed "${SEED:-0}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-1.0}" \
  --max-tokens "${MAX_TOKENS:-300}" \
  --image-detail "${IMAGE_DETAIL:-high}" \
  "$@"
