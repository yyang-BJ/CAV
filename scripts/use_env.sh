#!/usr/bin/env bash
# Reuse the existing GPU-ready environment from T3/.venv.
# It already includes torch/cu126, transformers, datasets, peft, ray, verl, vllm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAV_VENV="${CAV_VENV:-${PROJECT_ROOT}/../T3/.venv}"

if [[ ! -x "${CAV_VENV}/bin/python" ]]; then
  echo "Expected venv not found: ${CAV_VENV}" >&2
  echo "Set CAV_VENV to another Python environment if needed." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${CAV_VENV}/bin/activate"

# CAV's Ray PPO backend targets the newer veRL API (constants_ppo / cav_gae hooks).
# The shared T3/.venv may have an older editable HiPER-agent verl installed; prefer
# T3's veRL package on PYTHONPATH when present so imports resolve correctly.
T3_VERL_ROOT="${T3_VERL_ROOT:-${PROJECT_ROOT}/../T3/verl}"
PYTHONPATH_PARTS=("${PROJECT_ROOT}/src")
if [[ -f "${T3_VERL_ROOT}/verl/__init__.py" ]]; then
  PYTHONPATH_PARTS+=("${T3_VERL_ROOT}")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

cd "${PROJECT_ROOT}"

echo "[CAV] using ${CAV_VENV}"
python - <<'PY'
import cav_rl, torch, verl
print(f"cav_rl={cav_rl.__version__} python={__import__('sys').version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
print(f"verl={getattr(verl, '__version__', '?')} @ {verl.__file__}")
PY
