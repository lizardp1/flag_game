#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must be run from inside a git clone of flag_game." >&2
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
TARGET_BRANCH="${BRANCH:-$CURRENT_BRANCH}"

if [[ -z "$TARGET_BRANCH" ]]; then
  echo "Could not infer a branch. Set BRANCH=<branch-name> and rerun." >&2
  exit 1
fi

git fetch origin "$TARGET_BRANCH"
if [[ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]]; then
  git checkout "$TARGET_BRANCH"
fi
git pull --ff-only origin "$TARGET_BRANCH"

./scripts/runpod_bootstrap.sh
./scripts/runpod_smoke_test.sh
