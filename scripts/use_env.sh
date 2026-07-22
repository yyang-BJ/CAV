#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAV_ENV="${CAV_ENV:-CAV}"

# Initialize conda
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "[CAV] conda not found"
    exit 1
fi

# Activate environment
conda activate "${CAV_ENV}"

# CAV source path
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# If a local verl checkout exists, prioritize it
if [[ -d "${PROJECT_ROOT}/src/cav_rl/verl" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src/cav_rl/verl:${PYTHONPATH}"
fi

cd "${PROJECT_ROOT}"

echo "[CAV] using conda env: ${CAV_ENV}"
echo "[CAV] project root: ${PROJECT_ROOT}"

python - <<'PY'
import sys

print("python:", sys.executable)

try:
    import torch
    print(
        f"torch: {torch.__version__}, "
        f"cuda={torch.cuda.is_available()}, "
        f"gpus={torch.cuda.device_count()}"
    )
except Exception as e:
    print("missing torch:", e)
    sys.exit(1)

try:
    import cav_rl
    print("cav_rl:", cav_rl.__file__)
except Exception as e:
    print("missing cav_rl:", e)
    sys.exit(1)

try:
    import verl
    print("verl:", verl.__file__)
except Exception as e:
    print("missing verl:", e)
    sys.exit(1)
PY