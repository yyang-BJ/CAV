#!/usr/bin/env bash
# Smoke GRPO on GSM8K baseline split with Qwen2.5-1.5B-Instruct.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/use_env.sh"

SMOKE_DIR="${DATA_DIR:-/home/dataset-assist-0/ZX/dataset/gsm8k_baseline_smoke}"
if [[ ! -f "${SMOKE_DIR}/train.parquet" ]]; then
    echo "[smoke-grpo] building smoke split -> ${SMOKE_DIR}"
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${SCRIPT_DIR}/make_smoke_split_baseline.py" \
        --out_dir "${SMOKE_DIR}" \
        --n_train "${SMOKE_N_TRAIN:-256}" \
        --n_val "${SMOKE_N_VAL:-64}"
fi

mkdir -p outputs/Baseline-GSM8K-GRPO
TS="$(date +%Y%m%d-%H%M%S)"
EXP="${EXPERIMENT_NAME:-qwen2.5-1.5b-grpo-smoke-${TS}}"
echo "${EXP}" | tee outputs/Baseline-GSM8K-GRPO/.latest_smoke_exp

export DATA_DIR="${SMOKE_DIR}"
export BASE_MODEL="${BASE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-1.5B-Instruct}"
export INIT_MODEL="${INIT_MODEL:-backbone}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export NUM_GPUS="${NUM_GPUS:-2}"
export EXPERIMENT_NAME="${EXP}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"

echo "[smoke-grpo] EXP=${EXP}"
echo "[smoke-grpo] DATA_DIR=${DATA_DIR}"
echo "[smoke-grpo] BASE_MODEL=${BASE_MODEL}"
echo "[smoke-grpo] steps=${TOTAL_TRAINING_STEPS} train_bs=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N}"

exec bash "${SCRIPT_DIR}/train_grpo_gsm8k.sh"
