#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAV_ENV="${CAV_ENV:-CAV}"

# Initialize conda without replacing the current shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "[CAV] conda not found"
    exit 1
fi

conda activate "${CAV_ENV}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

echo "[CAV] using conda env: ${CAV_ENV}"

python - <<'PY'
import sys

print("python:", sys.executable)

try:
    import cav_rl
    print("cav_rl:", cav_rl.__file__)
except Exception as e:
    print("missing cav_rl:", e)

try:
    import verl
    print("verl:", verl.__file__)
except Exception as e:
    print("missing verl:", e)
PY