#!/usr/bin/env bash
# GRPO-correct on GSM8K: Right-short > Right-long >> Wrong (correct-only length bonus).
# Keeps plain baseline GRPO (scripts/train_grpo_gsm8k.sh) unchanged.
#
# Init model:
#   INIT_MODEL=stage1   -> finished plain GRPO checkpoint (default)
#   INIT_MODEL=backbone -> raw Instruct
#   INIT_MODEL=sft      -> CoT SFT merged checkpoint
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
    echo "[grpo-correct] missing cav_rl/verl. Activate env first:" >&2
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
DATA_NAME="${DATA_NAME:-GSM8K-GRPO-Correct}"

BACKBONE_MODEL="${BACKBONE_MODEL:-/home/dataset-assist-0/ZX/models/Qwen2.5-1.5B-Instruct}"
SFT_MODEL="${SFT_MODEL:-${PROJECT_ROOT}/outputs/sft-qwen2.5-1.5b-gsm8k-baseline-merged}"
# Prefer merged HF weights; raw FSDP `huggingface/` only has tokenizer/config.
STAGE1_MODEL_DEFAULT="${PROJECT_ROOT}/outputs/Baseline-GSM8K-GRPO/qwen2.5-1.5b-grpo-full-20260719-200211/global_step_180/actor/hf_merged"
STAGE1_MODEL="${STAGE1_MODEL:-${STAGE1_MODEL_DEFAULT}}"
INIT_MODEL="${INIT_MODEL:-stage1}"  # stage1 | backbone | sft

if [[ -z "${BASE_MODEL:-}" ]]; then
    case "${INIT_MODEL}" in
        stage1|grpo1|phase1)
            BASE_MODEL="${STAGE1_MODEL}"
            ;;
        backbone|instruct|raw)
            BASE_MODEL="${BACKBONE_MODEL}"
            ;;
        sft|cot-sft)
            BASE_MODEL="${SFT_MODEL}"
            ;;
        *)
            echo "[grpo-correct] unknown INIT_MODEL=${INIT_MODEL} (use stage1|backbone|sft or set BASE_MODEL)" >&2
            exit 1
            ;;
    esac
fi

# Fail early if stage1 path has no weights (common FSDP export pitfall).
if [[ ! -f "${BASE_MODEL}/model.safetensors" && ! -f "${BASE_MODEL}/pytorch_model.bin" ]]; then
    shopt -s nullglob
    shards=("${BASE_MODEL}"/model-*.safetensors "${BASE_MODEL}"/pytorch_model-*.bin)
    shopt -u nullglob
    if [[ ${#shards[@]} -eq 0 ]]; then
        echo "[grpo-correct] BASE_MODEL has no HF weights: ${BASE_MODEL}" >&2
        echo "[grpo-correct] merge FSDP actor shards first, e.g.:" >&2
        echo "  python -m verl.model_merger merge --backend fsdp --local_dir .../global_step_180/actor --target_dir .../hf_merged" >&2
        exit 1
    fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
NUM_GPUS="${NUM_GPUS:-2}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
ROLLOUT_N="${ROLLOUT_N:-4}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[grpo-correct] missing train file: ${TRAIN_FILE}" >&2
    exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[grpo-correct] missing model dir: ${BASE_MODEL}" >&2
    echo "[grpo-correct] INIT_MODEL=${INIT_MODEL}. For stage1, set STAGE1_MODEL=.../actor/huggingface" >&2
    exit 1
fi

unset VLLM_ATTENTION_BACKEND || true
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export RAY_object_store_memory="${RAY_object_store_memory:-1000000000}"

project_name="Baseline-GSM8K-GRPO-Correct"
experiment_name="${EXPERIMENT_NAME:-qwen2.5-1.5b-grpo-correct-${INIT_MODEL}}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"
mkdir -p "${default_local_dir}"

echo "[grpo-correct] INIT_MODEL=${INIT_MODEL} BASE_MODEL=${BASE_MODEL}"
echo "[grpo-correct] DATA_DIR=${DATA_DIR} rollout.n=${ROLLOUT_N} -> ${default_local_dir}"

# Quote negative Hydra overrides so they are not parsed as CLI flags.
CORRECT_REWARD="${CORRECT_REWARD:-1.0}"
WRONG_REWARD="${WRONG_REWARD:--0.5}"
UNPARSABLE_REWARD="${UNPARSABLE_REWARD:--1.0}"
STRICT_FORMAT_BONUS="${STRICT_FORMAT_BONUS:-0.2}"
MISSING_FORMAT_PENALTY="${MISSING_FORMAT_PENALTY:--0.2}"
LENGTH_BONUS_WEIGHT="${LENGTH_BONUS_WEIGHT:-0.01}"
LENGTH_FLOOR="${LENGTH_FLOOR:-270}"
LENGTH_FLOOR_SOFTNESS="${LENGTH_FLOOR_SOFTNESS:-40}"
LENGTH_EXCESS_REF="${LENGTH_EXCESS_REF:-0}"
LENGTH_SCORE_MAX="${LENGTH_SCORE_MAX:-1.0}"
LENGTH_SCORE_MIN="${LENGTH_SCORE_MIN:-0.0}"
LENGTH_SCORE_NEUTRAL="${LENGTH_SCORE_NEUTRAL:-0.0}"
MIN_CORRECT_FOR_LENGTH_RANKING="${MIN_CORRECT_FOR_LENGTH_RANKING:-2}"
EXTRACT_METHOD="${EXTRACT_METHOD:-flexible}"

python3 -m cav_rl.verl.main_baseline_grpo_correct \
    --config-path "${VERL_CONFIG_PATH}" \
    --config-name ppo_trainer \
    "+grpo_correct.correct_reward=${CORRECT_REWARD}" \
    "+grpo_correct.wrong_reward=${WRONG_REWARD}" \
    "+grpo_correct.unparsable_reward=${UNPARSABLE_REWARD}" \
    "+grpo_correct.strict_format_bonus=${STRICT_FORMAT_BONUS}" \
    "+grpo_correct.missing_format_penalty=${MISSING_FORMAT_PENALTY}" \
    "+grpo_correct.length_bonus_weight=${LENGTH_BONUS_WEIGHT}" \
    "+grpo_correct.length_floor=${LENGTH_FLOOR}" \
    "+grpo_correct.length_floor_softness=${LENGTH_FLOOR_SOFTNESS}" \
    "+grpo_correct.length_excess_ref=${LENGTH_EXCESS_REF}" \
    "+grpo_correct.length_score_max=${LENGTH_SCORE_MAX}" \
    "+grpo_correct.length_score_min=${LENGTH_SCORE_MIN}" \
    "+grpo_correct.length_score_neutral=${LENGTH_SCORE_NEUTRAL}" \
    "+grpo_correct.min_correct_for_length_ranking=${MIN_CORRECT_FOR_LENGTH_RANKING}" \
    "+grpo_correct.extract_method=${EXTRACT_METHOD}" \
    +cav.hierarchical_rollout=false \
    ++data_name="${DATA_NAME}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length=1024 \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
    data.truncation=right \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.03}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0.0}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-1}" \
    actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO:-0.2}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-false}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE:-true}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.48}" \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFFLOAD:-false}" \
    critic.enable=false \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO:-true}" \
    early_cut=false \
    +trunc_strength=0 \
    trainer.critic_warmup=0 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-true}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-50}" \
    trainer.default_local_dir="${default_local_dir}" \
    +trainer.dump_val_cases="${DUMP_VAL_CASES:-true}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR:-${default_local_dir}/val_cases}" \
    2>&1 | tee "${default_local_dir}.log"
