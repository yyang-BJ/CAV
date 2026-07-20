#!/usr/bin/env bash
# Train plain CoT SFT for the PPO baseline, then merge LoRA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-configs/sft_gsm8k_baseline.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-gsm8k-baseline}"
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-gsm8k-baseline-merged}"
BASE_MODEL="${BASE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-3B-Instruct}"

echo "[baseline-SFT] train with ${CONFIG}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python3 scripts/train_sft.py --config "${CONFIG}"

echo "[baseline-SFT] merge LoRA ${ADAPTER_DIR} -> ${MERGED_DIR}"
python3 scripts/merge_sft_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --output_dir "${MERGED_DIR}"

echo "[baseline-SFT] done. For PPO baseline:"
echo "  INIT_MODEL=sft SFT_MODEL=${MERGED_DIR} bash scripts/train_baseline_gsm8k.sh"
