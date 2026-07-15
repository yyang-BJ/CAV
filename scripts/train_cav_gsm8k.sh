#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VERL_CONFIG_PATH="${VERL_CONFIG_PATH:-${PROJECT_ROOT}/../T3-main/verl/verl/trainer/config}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/gsm8k}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
NUM_GPUS="${NUM_GPUS:-8}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

project_name="CAV-GSM8K"
experiment_name="${EXPERIMENT_NAME:-qwen2.5-3b-cav-ppo}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"
mkdir -p "${default_local_dir}"

python3 -m cav_rl.verl.main_cav_ppo \
    --config-path "${VERL_CONFIG_PATH}" \
    --config-name ppo_trainer \
    +cav.budget_actions='[0,16,32,64,128]' \
    +cav.correct_reward=1.0 \
    +cav.wrong_reward=0.0 \
    +cav.computation_price=0.0005 \
    +cav.actual_token_price=0.0001 \
    +cav.format_penalty=0.1 \
    +cav.missing_stop_penalty=0.2 \
    +cav.invalid_budget_penalty=0.1 \
    +cav.target_expected_budget=128.0 \
    +cav.lambda_c=0.0005 \
    data_name="GSM8K-CAV" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    data.truncation=right \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    critic.model.path="${BASE_MODEL}" \
    critic.model.trust_remote_code=true \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=true \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.adv_estimator=cav_gae \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=false \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq=20 \
    trainer.save_freq=100 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir="${default_local_dir}" \
    2>&1 | tee "${default_local_dir}.log"
