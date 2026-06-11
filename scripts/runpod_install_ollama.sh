#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama_models}"
mkdir -p "$OLLAMA_MODELS" runs

if ! command -v zstd >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y zstd
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y zstd
  elif command -v yum >/dev/null 2>&1; then
    yum install -y zstd
  else
    echo "ERROR: zstd is required, but no supported package manager was found." >&2
    exit 1
  fi
fi

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve > runs/ollama_server.log 2>&1 &
fi

echo "Ollama ready."
echo "  OLLAMA_MODELS=$OLLAMA_MODELS"
echo "  Server log: runs/ollama_server.log"
