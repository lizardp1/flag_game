#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib}"
export NND_SKIP_PLOTS=1
mkdir -p "$MPLCONFIGDIR"

OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/flag_game_smoke.XXXXXX")}"

python -m nnd.cli run \
  --config configs/flag_game/stripe_easy_v1.yaml \
  --out "$OUT" \
  --seed 0 \
  --backend scripted \
  --override N=2 \
  --override T=0 \
  --override H=0 \
  --override output.save_crop_images=false \
  --override output.make_plots=false

python - <<PY
import json
from pathlib import Path

out = Path("$OUT")
summary = json.loads((out / "summary.json").read_text())
manifest = json.loads((out / "trial_manifest.json").read_text())
print("Smoke test complete.")
print(f"Output: {out}")
print(f"Truth country: {summary['truth_country']}")
print(f"Final outcome: {summary['final_outcome']}")
print(f"Agents: {len(manifest['agent_models'])}")
PY
