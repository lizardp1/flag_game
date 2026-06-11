#!/usr/bin/env bash
# Shared RunPod cache defaults for large model downloads.
#
# Hugging Face and temporary download files should live on /workspace when it is
# available. The default /tmp filesystem on many RunPod templates is too small
# for even 4B VLM checkpoints.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d /workspace && -w /workspace ]]; then
  CACHE_ROOT="${NND_RUNPOD_CACHE_ROOT:-/workspace/.cache}"
else
  CACHE_ROOT="${NND_RUNPOD_CACHE_ROOT:-$ROOT/.cache}"
fi

is_tmp_path() {
  case "${1:-}" in
    /tmp/*|/var/tmp/*|/private/tmp/*) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${HF_HOME:-}" ]] || is_tmp_path "${HF_HOME:-}"; then
  export HF_HOME="$CACHE_ROOT/huggingface"
fi
if [[ -z "${HF_HUB_CACHE:-}" ]] || is_tmp_path "${HF_HUB_CACHE:-}"; then
  export HF_HUB_CACHE="$HF_HOME/hub"
fi
if [[ -z "${TRANSFORMERS_CACHE:-}" ]] || is_tmp_path "${TRANSFORMERS_CACHE:-}"; then
  export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
fi
if [[ -z "${HF_ASSETS_CACHE:-}" ]] || is_tmp_path "${HF_ASSETS_CACHE:-}"; then
  export HF_ASSETS_CACHE="$HF_HOME/assets"
fi
if [[ -z "${HF_XET_CACHE:-}" ]] || is_tmp_path "${HF_XET_CACHE:-}"; then
  export HF_XET_CACHE="$HF_HOME/xet"
fi
if [[ -z "${TMPDIR:-}" ]] || is_tmp_path "${TMPDIR:-}"; then
  export TMPDIR="$CACHE_ROOT/tmp"
fi
if [[ -z "${MPLCONFIGDIR:-}" ]] || is_tmp_path "${MPLCONFIGDIR:-}"; then
  export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
fi

export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-2}"
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="${HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY:-1}"

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$HF_XET_CACHE" \
  "$TMPDIR" \
  "$MPLCONFIGDIR"

echo "RunPod cache env:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HF_XET_CACHE=$HF_XET_CACHE"
echo "  TMPDIR=$TMPDIR"
