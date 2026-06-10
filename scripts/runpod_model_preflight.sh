#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"

MIN_WORKSPACE_GB="${MIN_WORKSPACE_GB:-12}"
MIN_RAM_GB="${MIN_RAM_GB:-18}"
MIN_GPU_FREE_GB="${MIN_GPU_FREE_GB:-10}"

gb_from_kb() {
  awk -v kb="$1" 'BEGIN { printf "%.1f", kb / 1024 / 1024 }'
}

workspace_free_gb="$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
if [[ -z "$workspace_free_gb" ]]; then
  workspace_free_gb="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
fi

mem_available_kb="$(awk '/MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)"
mem_available_gb="$(gb_from_kb "$mem_available_kb")"

gpu_free_mb=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_free_mb="$(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | head -1 \
      | tr -dc '0-9' || true
  )"
fi
gpu_free_gb="0"
if [[ -n "$gpu_free_mb" ]]; then
  gpu_free_gb="$(awk -v mb="$gpu_free_mb" 'BEGIN { printf "%.1f", mb / 1024 }')"
fi

echo "RunPod model preflight:"
echo "  workspace free: ${workspace_free_gb} GB"
echo "  RAM available:  ${mem_available_gb} GB"
echo "  GPU free:       ${gpu_free_gb} GB"
echo "  HF_HOME:        ${HF_HOME}"
echo "  HF_HUB_CACHE:   ${HF_HUB_CACHE}"
echo "  TMPDIR:         ${TMPDIR}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
fi

failed=0
if (( workspace_free_gb < MIN_WORKSPACE_GB )); then
  echo "ERROR: workspace free space is below ${MIN_WORKSPACE_GB} GB." >&2
  failed=1
fi
if awk -v actual="$mem_available_gb" -v min="$MIN_RAM_GB" 'BEGIN { exit !(actual < min) }'; then
  echo "ERROR: available RAM is below ${MIN_RAM_GB} GB." >&2
  failed=1
fi
if awk -v actual="$gpu_free_gb" -v min="$MIN_GPU_FREE_GB" 'BEGIN { exit !(actual < min) }'; then
  echo "ERROR: free GPU memory is below ${MIN_GPU_FREE_GB} GB." >&2
  failed=1
fi

if (( failed )); then
  cat >&2 <<'MSG'

Suggested cleanup:
  rm -rf /tmp/nnd_matplotlib_cache/huggingface
  du -h -d 2 /workspace/.cache/huggingface 2>/dev/null | sort -h | tail -40

If a 4B VLM still cannot fit, try Qwen/Qwen3-VL-2B-Instruct first.
MSG
  exit 1
fi

echo "Preflight passed."
