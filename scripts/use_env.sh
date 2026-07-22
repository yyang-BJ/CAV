#!/usr/bin/env bash
# Use the self-contained CAV virtual environment.
# The environment should include torch, transformers, datasets, ray, verl, vllm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "[CAV] No active Python environment detected." >&2
  echo "Activate the CAV environment first:" >&2
  echo "  source .venv/bin/activate" >&2
  exit 1
fi

PYTHONPATH_PARTS=()

# IMPORTANT:
# Always use the veRL version bundled with this CAV repository.
# Avoid accidentally importing another project's/system verl.
CAV_VERL_ROOT="${PROJECT_ROOT}/third_party/verl"

if [[ -f "${CAV_VERL_ROOT}/verl/__init__.py" ]]; then
  PYTHONPATH_PARTS+=("${CAV_VERL_ROOT}")
else
  echo "[CAV] WARNING: local veRL not found at ${CAV_VERL_ROOT}" >&2
fi

# CAV source
PYTHONPATH_PARTS+=("${PROJECT_ROOT}/src")

# Preserve existing PYTHONPATH last
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi

export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

cd "${PROJECT_ROOT}"

echo "[CAV] python environment:"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "  venv: ${VIRTUAL_ENV}"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "  conda: ${CONDA_PREFIX}"
fi

echo "[CAV] python: $(which python)"
echo "[CAV] PYTHONPATH=${PYTHONPATH}"

python - <<'PY'
import sys
import torch
import cav_rl
import verl

print(f"cav_rl={getattr(cav_rl, '__version__', '?')} python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
print(f"verl={getattr(verl, '__version__', '?')} @ {verl.__file__}")

try:
    from verl.workers.fsdp_workers import CriticWorker
    print("CriticWorker import: OK")
except Exception as e:
    print("CriticWorker import FAILED:", e)
PY