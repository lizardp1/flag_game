#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements-runpod.txt

mkdir -p .matplotlib
python -m nnd.cli --help >/dev/null
python -m nnd.flag_game.cli --help >/dev/null
python -m nnd.flag_game_broadcast.cli --help >/dev/null
python -m nnd.flag_game_org.cli --help >/dev/null

cat <<'MSG'
RunPod bootstrap complete.

Try the no-network smoke test:
  ./scripts/runpod_smoke_test.sh

For paper chart redraws:
  cd paper/final_charts
  ./scripts/rebuild_final_charts.sh
MSG
