#!/usr/bin/env bash
# =============================================================================
# CAV 训练环境一键安装（自包含，不依赖 T3 / 系统里已有的 verl）
# 适用于 PPO / GRPO / GRPO-correct / CAV-RL
#
# 用法（在 CAV 仓库根目录，或任意位置传入路径）:
#   bash install_env.sh
#   bash install_env.sh /path/to/CAV
#
# 可选环境变量:
#   PYTHON_BIN=python3.11
#   VERL_REPO=https://github.com/volcengine/verl.git
#   VERL_REF=v0.5.0
#   SKIP_FLASH_ATTN=1
#   SKIP_VERIFY=1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 若脚本在 CAV/share/ 下，默认安装到上一级；否则要求传入 CAV 根目录或在 CAV 根执行。
if [[ -n "${1:-}" ]]; then
  CAV_ROOT="$(cd "$1" && pwd)"
elif [[ -f "${SCRIPT_DIR}/../pyproject.toml" ]]; then
  CAV_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
  CAV_ROOT="${SCRIPT_DIR}"
elif [[ -f "$(pwd)/pyproject.toml" ]]; then
  CAV_ROOT="$(pwd)"
else
  echo "[install] ERROR: 找不到 CAV 仓库根目录（需含 pyproject.toml）。" >&2
  echo "  用法: bash install_env.sh /path/to/CAV" >&2
  exit 1
fi

VENV="${CAV_ROOT}/.venv"
VERL_DIR="${CAV_ROOT}/third_party/verl"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VERL_REPO="${VERL_REPO:-https://github.com/volcengine/verl.git}"
VERL_REF="${VERL_REF:-v0.5.0}"

echo "[install] CAV_ROOT = ${CAV_ROOT}"
echo "[install] VENV     = ${VENV}"
echo "[install] VERL     = ${VERL_DIR}  (${VERL_REPO} @ ${VERL_REF})"

if [[ ! -f "${CAV_ROOT}/pyproject.toml" ]]; then
  echo "[install] ERROR: ${CAV_ROOT} 不是 CAV 仓库（缺少 pyproject.toml）" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
    echo "[install] WARN: 未找到 python3.11，改用 ${PYTHON_BIN}"
  else
    echo "[install] ERROR: 需要 Python 3.11（推荐）或 python3" >&2
    exit 1
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[install] ERROR: 需要 git（用于克隆 veRL）" >&2
  exit 1
fi

echo "[install] Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" -V 2>&1))"

mkdir -p "${CAV_ROOT}/third_party"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[install] 创建虚拟环境 ..."
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel

echo "[install] 安装 PyTorch (CUDA 12.6 wheels) ..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "[install] 安装 Ray / vLLM / Transformers 等 ..."
pip install \
  vllm==0.10.1 \
  ray==2.50.0 \
  transformers==4.55.4 \
  tokenizers==0.21.4 \
  accelerate==1.10.1 \
  peft==0.19.1 \
  datasets==5.0.0 \
  trl==0.9.6 \
  tensordict==0.9.1 \
  xformers==0.0.31 \
  wandb==0.28.0 \
  hydra-core==1.3.3 \
  omegaconf==2.3.1 \
  safetensors==0.8.0 \
  sentencepiece==0.2.1 \
  tiktoken==0.13.0 \
  einops==0.8.2 \
  "numpy==1.26.4" \
  pandas==3.0.3 \
  pyarrow==24.0.0 \
  PyYAML==6.0.3 \
  tqdm==4.68.3

if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  echo "[install] 安装 flash_attn（失败可忽略）..."
  pip install flash_attn==2.8.0.post2 --no-build-isolation || \
    echo "[install] WARN: flash_attn 安装失败，已跳过（多数训练仍可继续）"
fi

# 始终使用本仓库 third_party/verl，避免误用系统 / 其它项目里的旧 verl
if [[ ! -f "${VERL_DIR}/verl/__init__.py" ]]; then
  echo "[install] 克隆 veRL -> third_party/verl ..."
  rm -rf "${VERL_DIR}"
  git clone --depth 1 --branch "${VERL_REF}" "${VERL_REPO}" "${VERL_DIR}"
else
  echo "[install] 已存在 ${VERL_DIR}，跳过克隆"
fi

echo "[install] editable 安装 verl + cav-rl ..."
pip install -e "${VERL_DIR}"
pip install -e "${CAV_ROOT}"

if [[ "${SKIP_VERIFY:-0}" != "1" ]]; then
  echo "[install] 校验导入路径（verl 必须来自本 CAV 目录）..."
  cd "${CAV_ROOT}"
  # shellcheck disable=SC1091
  if [[ -f "${CAV_ROOT}/scripts/use_env.sh" ]]; then
    # 不启用任何 legacy 回退
    unset ALLOW_LEGACY_T3 || true
    # shellcheck disable=SC1091
    source "${CAV_ROOT}/scripts/use_env.sh"
  else
    export PYTHONPATH="${CAV_ROOT}/src:${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  fi
  python - <<PY
import os, sys
import cav_rl, torch, verl
print(f"cav_rl={cav_rl.__version__}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
print(f"verl={getattr(verl, '__version__', '?')}")
print(f"verl.__file__={verl.__file__}")
root = os.path.realpath("${CAV_ROOT}")
vf = os.path.realpath(verl.__file__)
if root not in vf and ".venv" not in vf:
    print("ERROR: verl 似乎不是从本 CAV 环境加载的，请检查 PYTHONPATH / 其它环境是否污染。", file=sys.stderr)
    sys.exit(1)
print("OK: 环境安装完成，verl 来自本 CAV 环境。")
PY
fi

cat <<EOF

============================================================
安装完成。

每次使用前先激活:
  cd ${CAV_ROOT}
  source scripts/use_env.sh

请确认打印的 verl 路径在本仓库下（.venv 或 third_party/verl），
不要用系统 / conda 里其它 verl。

训练前请自行设置数据和模型路径，例如:
  export DATA_DIR=/path/to/gsm8k_baseline
  export BASE_MODEL=/path/to/your/model_or_ckpt
============================================================
EOF
