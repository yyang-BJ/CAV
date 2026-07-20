#!/usr/bin/env bash
# Train format-strengthened CAV SFT, then merge LoRA into a full checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-configs/sft_gsm8k_format.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-v2}"
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-v2-merged}"
BASE_MODEL="${BASE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-3B-Instruct}"

echo "[CAV-SFT] train with ${CONFIG}"
# Single GPU: DataParallel with tiny batches was ~60s/step.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python3 scripts/train_sft.py --config "${CONFIG}"

echo "[CAV-SFT] merge LoRA ${ADAPTER_DIR} -> ${MERGED_DIR}"
python3 scripts/merge_sft_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --output_dir "${MERGED_DIR}"

echo "[CAV-SFT] optional format eval (greedy)"
python3 scripts/eval_sft_format.py \
  --model_path "${MERGED_DIR}" \
  --num_samples "${EVAL_SAMPLES:-64}" \
  --budget_actions 0,16,32,64,128 || true

echo "[CAV-SFT] done. For PPO: export BASE_MODEL=${MERGED_DIR}"
