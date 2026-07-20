#!/usr/bin/env bash
set -euo pipefail
cd /home/dataset-assist-0/ZX/CAV
# shellcheck disable=SC1091
source scripts/use_env.sh
ray stop --force >/dev/null 2>&1 || true

mkdir -p outputs/Baseline-GSM8K-GRPO-Correct
TS=$(date +%Y%m%d-%H%M%S)
EXP="qwen2.5-1.5b-grpo-correct-v2-${TS}"
echo "$EXP" | tee outputs/Baseline-GSM8K-GRPO-Correct/.latest_full_exp
STAGE1=/home/dataset-assist-0/ZX/CAV/outputs/Baseline-GSM8K-GRPO/qwen2.5-1.5b-grpo-full-20260719-200211/global_step_180/actor/hf_merged
LOG="outputs/Baseline-GSM8K-GRPO-Correct/${EXP}.nohup.log"

nohup env \
  EXPERIMENT_NAME="$EXP" \
  INIT_MODEL=stage1 \
  STAGE1_MODEL="$STAGE1" \
  BASE_MODEL="$STAGE1" \
  WANDB_MODE=online \
  DATA_DIR=/home/dataset-assist-0/ZX/dataset/gsm8k_baseline \
  TRAIN_BATCH_SIZE=32 \
  VAL_BATCH_SIZE=64 \
  PPO_MINI_BATCH_SIZE=32 \
  ROLLOUT_N=4 \
  MAX_RESPONSE_LENGTH=512 \
  ROLLOUT_TEMPERATURE=0.7 \
  LENGTH_BONUS_WEIGHT=0.03 \
  WRONG_REWARD=0.0 \
  LENGTH_SCORE_MIN=0.0 \
  LENGTH_SCORE_MAX=1.0 \
  KL_LOSS_COEF=0.005 \
  TOTAL_TRAINING_STEPS=50 \
  TEST_FREQ=5 \
  SAVE_FREQ=25 \
  GPU_MEMORY_UTILIZATION=0.48 \
  VAL_BEFORE_TRAIN=true \
  bash scripts/train_grpo_correct_gsm8k.sh \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "EXP=$EXP"
echo "LOG=$LOG"
