#!/usr/bin/env bash
# Use the current conda environment for CAV.
# The environment should already include torch, transformers, datasets, ray, verl, vllm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Require an active conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[CAV] No active conda environment detected." >&2
  echo "Activate the CAV environment first:" >&2
  echo "  conda activate CAV" >&2
  exit 1
fi

# CAV source path
PYTHONPATH_PARTS=("${PROJECT_ROOT}/src")

# Prefer T3 veRL if available
T3_VERL_ROOT="${T3_VERL_ROOT:-${PROJECT_ROOT}/../T3/verl}"
if [[ -f "${T3_VERL_ROOT}/verl/__init__.py" ]]; then
  PYTHONPATH_PARTS+=("${T3_VERL_ROOT}")
fi

# Preserve existing PYTHONPATH
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi

export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

cd "${PROJECT_ROOT}"

echo "[CAV] using conda environment: ${CONDA_PREFIX}"
echo "[CAV] python: $(which python)"

python - <<'PY'
import sys
import cav_rl
import torch
import verl

print(f"cav_rl={getattr(cav_rl, '__version__', '?')} python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
print(f"verl={getattr(verl, '__version__', '?')} @ {verl.__file__}")
PY