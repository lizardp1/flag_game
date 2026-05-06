#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/flag_game/proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5/04_model_mix_alpha_neutral}"
CONDITION="${CONDITION:-all_gpt_4o}"
MODEL="${MODEL:-gpt-4o}"
B="${B:-50}"
TEMPERATURE="${TEMPERATURE:-0.2}"
OUT="${OUT:-$ROOT/results/flag_game/empirical_crop_field_probes/mechanism1_final_${MODEL//[^A-Za-z0-9_]/_}_B${B}/results.csv}"

manifests=()
while IFS= read -r manifest; do
  manifests+=("$manifest")
done < <(find "$RUN_ROOT" -path "*/$CONDITION/seed_*/trial_manifest.json" | sort)
if [[ "${#manifests[@]}" -eq 0 ]]; then
  echo "No trial manifests found under $RUN_ROOT for condition $CONDITION" >&2
  exit 1
fi

targets="$(
  python3 - "$RUN_ROOT" "$CONDITION" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
condition = sys.argv[2]
targets = set()
for summary_path in root.glob(f"N*/{condition}/seed_*/summary.json"):
    with summary_path.open() as handle:
        summary = json.load(handle)
    target = summary.get("truth_country")
    if target:
        targets.add(str(target))
if not targets:
    for manifest_path in root.glob(f"N*/{condition}/seed_*/trial_manifest.json"):
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        target = manifest.get("truth_country") or manifest.get("target_country")
        if target:
            targets.add(str(target))
print(",".join(sorted(targets)))
PY
)"

if [[ -z "$targets" ]]; then
  echo "Could not infer truth-country targets from $RUN_ROOT" >&2
  exit 1
fi

args=(
  python3 "$ROOT/scripts/run_flag_empirical_crop_field_probes.py"
  --target "$targets"
  --models "$MODEL"
  --B "$B"
  --country-pool stripe_expanded_24
  --tile-width 6
  --tile-height 4
  --render-scale 25
  --temperature "$TEMPERATURE"
  --out "$OUT"
)

for manifest in "${manifests[@]}"; do
  args+=(--crop-manifest "$manifest")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

exec "${args[@]}"
