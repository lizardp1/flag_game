#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"

if [[ "${SKIP_OPEN_MODEL_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements-open-models.txt
fi

python scripts/run_qwen_vl_smoke.py "$@"
