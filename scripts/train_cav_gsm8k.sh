#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${PROJECT_ROOT}"

# Locate a source checkout of veRL when it is not installed as a package. The
# sibling layout works for AutoDL (/root/autodl-tmp/{CAV,verl}); T3_VERL_ROOT
# remains supported for existing deployments.
VERL_ROOT="${VERL_ROOT:-${T3_VERL_ROOT:-}}"
if [[ -z "${VERL_ROOT}" ]]; then
    for candidate in "${PROJECT_ROOT}/../verl" "${PROJECT_ROOT}/../T3/verl"; do
        if [[ -f "${candidate}/verl/__init__.py" ]]; then
            VERL_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
PYTHONPATH_PARTS=("${PROJECT_ROOT}/src")
if [[ -n "${VERL_ROOT}" && -f "${VERL_ROOT}/verl/__init__.py" ]]; then
    PYTHONPATH_PARTS+=("${VERL_ROOT}")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_PARTS+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

if ! "${PYTHON_BIN}" -c "import cav_rl, verl" >/dev/null 2>&1; then
    echo "[CAV] missing cav_rl/verl in current Python. Activate env first:" >&2
    echo "  source ${PROJECT_ROOT}/scripts/use_env.sh" >&2
    echo "[CAV] or set PYTHON_BIN and VERL_ROOT explicitly." >&2
    exit 1
fi

if [[ -z "${VERL_CONFIG_PATH:-}" ]]; then
    VERL_CONFIG_PATH="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import verl.trainer
print(Path(verl.trainer.__file__).resolve().parent / "config")
PY
)"
fi

abspath_from_project() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
    esac
}

DATA_DIR="$(abspath_from_project "${DATA_DIR:-data/gsm8k}")"
TRAIN_FILE="$(abspath_from_project "${TRAIN_FILE:-${DATA_DIR}/train.parquet}")"
VAL_FILE="$(abspath_from_project "${VAL_FILE:-${DATA_DIR}/test.parquet}")"
DATA_NAME="${DATA_NAME:-GSM8K-CAV}"
BASE_MODEL="$(abspath_from_project "${BASE_MODEL:-${MODEL_PATH:-outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-v2-merged}}")"
OUTPUT_ROOT="$(abspath_from_project "${OUTPUT_ROOT:-outputs}")"
VERL_CONFIG_PATH="$(abspath_from_project "${VERL_CONFIG_PATH}")"
NUM_GPUS="${NUM_GPUS:-$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
if [[ -z "${USE_REMOVE_PADDING:-}" ]]; then
    if "${PYTHON_BIN}" -c "import flash_attn" >/dev/null 2>&1; then
        USE_REMOVE_PADDING=true
    else
        USE_REMOVE_PADDING=false
    fi
fi
# Align schedule with T3 MovieRec (2xA100 Qwen2.5-3B). Keep train batch at 64 by
# default: CAV hierarchical rollout does multiple vLLM calls per step, so 128
# (T3 one-shot) is riskier here. Override TRAIN_BATCH_SIZE=128 if memory allows.

# Fail fast with a clear message if paths are wrong.
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[CAV] missing train file: ${TRAIN_FILE}" >&2
    echo "[CAV] generate it with: PYTHONPATH=src ${PYTHON_BIN} scripts/preprocess_gsm8k.py --local_dir ${DATA_DIR}" >&2
    echo "[CAV] or export DATA_DIR/TRAIN_FILE to an existing parquet dataset." >&2
    exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
    echo "[CAV] missing validation file: ${VAL_FILE}" >&2
    echo "[CAV] export DATA_DIR/VAL_FILE to an existing parquet dataset." >&2
    exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[CAV] missing model dir: ${BASE_MODEL}" >&2
    echo "[CAV] export BASE_MODEL to the merged SFT checkpoint." >&2
    exit 1
fi
if [[ ! -d "${VERL_CONFIG_PATH}" ]]; then
    echo "[CAV] missing veRL config directory: ${VERL_CONFIG_PATH}" >&2
    echo "[CAV] export VERL_CONFIG_PATH to the directory containing ppo_trainer.yaml." >&2
    exit 1
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "[CAV] no CUDA GPUs detected; set NUM_GPUS only after making GPUs visible." >&2
    exit 1
fi

# Prefer FlashAttention / vLLM V1; XFORMERS forces V0 fallback on recent vLLM.
unset VLLM_ATTENTION_BACKEND || true
# Avoid hard-failing when wandb is not logged in; console logging still works.
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
# Small /dev/shm on this machine; keep Ray object store modest.
export RAY_object_store_memory="${RAY_object_store_memory:-1000000000}"

project_name="CAV-GSM8K"
experiment_name="${EXPERIMENT_NAME:-qwen2.5-3b-cav-ppo}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"
mkdir -p "${default_local_dir}"

echo "[CAV] project=${PROJECT_ROOT}"
echo "[CAV] python=${PYTHON_BIN} verl=${VERL_ROOT:-installed-package}"
echo "[CAV] data=${TRAIN_FILE} val=${VAL_FILE}"
echo "[CAV] model=${BASE_MODEL} output=${default_local_dir}"
echo "[CAV] gpus=${NUM_GPUS} remove_padding=${USE_REMOVE_PADDING}"

"${PYTHON_BIN}" -m cav_rl.verl.main_cav_ppo \
    --config-path "${VERL_CONFIG_PATH}" \
    --config-name ppo_trainer \
    +cav.budget_actions='[0,16,32,64,128]' \
    +cav.correct_reward=1.0 \
    +cav.wrong_reward=0.0 \
    +cav.format_penalty=0.5 \
    +cav.invalid_action_penalty=0.5 \
    +cav.missing_stop_penalty=0.2 \
    +cav.invalid_budget_penalty=0.1 \
    +cav.correctness_requires_valid_format=true \
    +cav.target_expected_tokens="${TARGET_EXPECTED_TOKENS:-96.0}" \
    +cav.b_start="${B_START:-none}" \
    +cav.b_anneal_ratio="${B_ANNEAL_RATIO:-0.7}" \
    +cav.lambda_scale_start_ratio="${LAMBDA_SCALE_START_RATIO:-0.1}" \
    +cav.lambda_scale_end_ratio="${LAMBDA_SCALE_END_RATIO:-0.4}" \
    +cav.lambda_c="${LAMBDA_C:-0.0005}" \
    +cav.dual_lr="${DUAL_LR:-0.00001}" \
    +cav.min_lambda_c="${MIN_LAMBDA_C:-0.0}" \
    +cav.max_lambda_c="${MAX_LAMBDA_C:-0.02}" \
    +cav.dual_update="${DUAL_UPDATE:-true}" \
    +cav.hierarchical_rollout="${HIERARCHICAL_ROLLOUT:-true}" \
    +cav.max_macro_steps="${MAX_MACRO_STEPS:-6}" \
    +cav.budget_max_tokens="${BUDGET_MAX_TOKENS:-64}" \
    +cav.answer_max_tokens="${ANSWER_MAX_TOKENS:-96}" \
    +cav.parse_fail_keep_tokens="${PARSE_FAIL_KEEP_TOKENS:-32}" \
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
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.hybrid_engine=true \
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
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.3}" \
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
    critic.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.model.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    critic.model.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-false}" \
    algorithm.adv_estimator=cav_gae \
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
