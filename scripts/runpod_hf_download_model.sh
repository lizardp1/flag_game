#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"
cd "$ROOT"

MODEL_ID="${1:-${MODEL_ID:-}}"
MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-1}"

if [[ -z "$MODEL_ID" ]]; then
  cat >&2 <<'MSG'
Usage:
  ./scripts/runpod_hf_download_model.sh <huggingface-model-id>

Example:
  ./scripts/runpod_hf_download_model.sh llava-hf/llava-v1.6-mistral-7b-hf
MSG
  exit 2
fi

python - "$MODEL_ID" "$MAX_WORKERS" <<'PY'
from __future__ import annotations

import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
max_workers = int(sys.argv[2])
print(f"Downloading {model_id} with max_workers={max_workers}")
path = snapshot_download(repo_id=model_id, max_workers=max_workers)
print(f"Cached {model_id} at {path}")
PY
