#!/usr/bin/env bash
# Plain PPO baseline on GSM8K CoT (outcome reward only, standard GAE).
#
# Init model:
#   INIT_MODEL=backbone  -> raw Instruct (default)
#   INIT_MODEL=sft       -> CoT SFT merged checkpoint
# Or set BASE_MODEL=/path/to/checkpoint explicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

T3_VERL_ROOT="${T3_VERL_ROOT:-${PROJECT_ROOT}/../T3/verl}"
if [[ -f "${T3_VERL_ROOT}/verl/__init__.py" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src:${T3_VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

if ! python3 -c "import cav_rl, verl" >/dev/null 2>&1; then
    echo "[baseline] missing cav_rl/verl. Activate env first:" >&2
    echo "  source ${PROJECT_ROOT}/scripts/use_env.sh" >&2
    exit 1
fi

if [[ -z "${VERL_CONFIG_PATH:-}" ]]; then
    VERL_CONFIG_PATH="$(python3 - <<'PY'
from pathlib import Path
import verl.trainer
print(Path(verl.trainer.__file__).resolve().parent / "config")
PY
)"
fi

DATA_DIR="${DATA_DIR:-/home/dataset-assist-0/ZX/dataset/gsm8k_baseline}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"
DATA_NAME="${DATA_NAME:-GSM8K-Baseline}"

BACKBONE_MODEL="${BACKBONE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-3B-Instruct}"
SFT_MODEL="${SFT_MODEL:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-gsm8k-baseline-merged}"
INIT_MODEL="${INIT_MODEL:-backbone}"  # backbone | sft

if [[ -z "${BASE_MODEL:-}" ]]; then
    case "${INIT_MODEL}" in
        backbone|instruct|raw)
            BASE_MODEL="${BACKBONE_MODEL}"
            ;;
        sft|cot-sft)
            BASE_MODEL="${SFT_MODEL}"
            ;;
        *)
            echo "[baseline] unknown INIT_MODEL=${INIT_MODEL} (use backbone|sft or set BASE_MODEL)" >&2
            exit 1
            ;;
    esac
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
NUM_GPUS="${NUM_GPUS:-2}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[baseline] missing train file: ${TRAIN_FILE}" >&2
    echo "[baseline] run: PYTHONPATH=src python3 scripts/preprocess_gsm8k_baseline.py" >&2
    exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[baseline] missing model dir: ${BASE_MODEL}" >&2
    echo "[baseline] INIT_MODEL=${INIT_MODEL}. For sft, train with scripts/train_sft_baseline.sh first." >&2
    exit 1
fi

unset VLLM_ATTENTION_BACKEND || true
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export RAY_object_store_memory="${RAY_object_store_memory:-1000000000}"

project_name="Baseline-GSM8K"
experiment_name="${EXPERIMENT_NAME:-qwen2.5-3b-ppo-${INIT_MODEL}}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"
mkdir -p "${default_local_dir}"

echo "[baseline] INIT_MODEL=${INIT_MODEL} BASE_MODEL=${BASE_MODEL}"
echo "[baseline] DATA_DIR=${DATA_DIR} -> ${default_local_dir}"

python3 -m cav_rl.verl.main_baseline_ppo \
    --config-path "${VERL_CONFIG_PATH}" \
    --config-name ppo_trainer \
    +baseline.correct_reward=1.0 \
    +baseline.wrong_reward=0.0 \
    +baseline.format_score=0.0 \
    +baseline.extract_method=flexible \
    +cav.hierarchical_rollout=false \
    ++data_name="${DATA_NAME}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length=1024 \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-768}" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-false}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE:-true}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}" \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    critic.model.path="${BASE_MODEL}" \
    critic.enable=true \
    critic.model.trust_remote_code=true \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=true \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.model.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    critic.model.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-false}" \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=false \
    early_cut=false \
    +trunc_strength=0 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq="${TEST_FREQ:-6}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-100}" \
    trainer.default_local_dir="${default_local_dir}" \
    +trainer.dump_val_cases="${DUMP_VAL_CASES:-true}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR:-${default_local_dir}/val_cases}" \
    2>&1 | tee "${default_local_dir}.log"
