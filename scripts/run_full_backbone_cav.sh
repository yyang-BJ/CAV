#!/usr/bin/env bash
# Full-data CAV PPO on Qwen2.5-3B-Instruct backbone, wandb online, nohup-friendly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/use_env.sh"

mkdir -p outputs/CAV-GSM8K
TS="$(date +%Y%m%d-%H%M%S)"
EXP="${EXPERIMENT_NAME:-qwen2.5-3b-cav-ppo-backbone-full-${TS}}"
echo "${EXP}" | tee outputs/CAV-GSM8K/.latest_full_exp

export DATA_DIR="${DATA_DIR:-/home/dataset-assist-0/ZX/dataset/gsm8k_cav}"
export BASE_MODEL="${BASE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-3B-Instruct}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export TEST_FREQ="${TEST_FREQ:-6}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export NUM_GPUS="${NUM_GPUS:-2}"
export EXPERIMENT_NAME="${EXP}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TARGET_EXPECTED_TOKENS="${TARGET_EXPECTED_TOKENS:-96.0}"
export B_START="${B_START:-none}"

echo "[launch] EXP=${EXP}"
echo "[launch] DATA_DIR=${DATA_DIR}"
echo "[launch] BASE_MODEL=${BASE_MODEL}"
echo "[launch] STEPS=${TOTAL_TRAINING_STEPS} WANDB_MODE=${WANDB_MODE}"

exec bash "${SCRIPT_DIR}/train_cav_gsm8k.sh"
