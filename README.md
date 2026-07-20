# CAV-RL: Computation Allocation Value for Adaptive LLM Reasoning

This is an independent research-code project for **Computation Allocation Value (CAV)** training. It implements the optimized proposal in which an LLM learns how much computation to allocate before each reasoning segment.

Default backbone: `Qwen/Qwen2.5-3B-Instruct` (local path via env / script args).

Default dataset: `openai/gsm8k` (`main`; preprocess scripts write parquet under a local data dir).

## Method

The policy is a single autoregressive LLM that emits structured blocks:

```text
<budget>64</budget>
<reason>one budget-limited reasoning segment</reason>
...
<budget>0</budget>
<answer>final numeric answer only</answer>
```

At macro step `k`, the model samples a budget action `b_k` from a discrete action set:

```text
[0, 16, 32, 64, 128]
```

`b_k > 0` means continue reasoning with that maximum budget. `b_k = 0` means stop reasoning and answer.

**Rollout is hierarchical** (default): at each macro state `H_k` the trainer first generates a budget block, then a reason or answer segment conditioned on `b_k`, then advances `H_{k+1}`. This is implemented in `cav_rl/verl/hierarchical_rollout.py` (local reference: `generate_macro_completion`). Set `cav.hierarchical_rollout=false` to fall back to one-shot `generate_sequences` + post-parse.

The actual computation cost uses consumed reasoning tokens. The main trajectory
reward is gated so a correct answer stays non-negative:

```text
l_k = number of tokens generated inside <reason>...</reason>
C(T) = sum_k l_k
R'(tau) = R_answer / (1 + lambda_c * C(T))
```

In the token/macro reward tensor, the gated term is placed on the stop/answer
macro (`b_k = 0`). Positive-budget macros no longer receive per-step
`-lambda_c * l_k`. Format / missing-stop / invalid-budget penalties remain
additive extras on the terminal anchor.

The CAV advantage uses variable-length TD/GAE over macro steps:

```text
delta_k = r_k + gamma^{l_k} * V(H_{k+1}) - V(H_k)
A_k = GAE(delta, gamma^{l_k}, lambda_gae)
```

`lambda_c` is updated by projected dual ascent on the expected budget constraint (default on):

```text
lambda <- [lambda + eta * (E[C] - B)]_+
```

with `B = cav.target_expected_tokens`. See `docs/method_alignment.md` for a proposal↔code map.

## Project Layout

```text
configs/
  sft_gsm8k.yaml              # optional format-alignment SFT config
  cav_ppo_gsm8k.yaml          # local debug PPO config
  verl_cav_overrides.yaml     # CAV-specific Hydra override reference
docs/
  method_alignment.md         # revised proposal ↔ code
scripts/
  preprocess_gsm8k.py         # writes parquet data
  preprocess_gsm8k_baseline.py
  make_smoke_split_baseline.py
  train_sft.py                # optional CAV format SFT
  train_cav_gsm8k.sh          # Ray CAV PPO launcher
  train_baseline_gsm8k.sh     # plain PPO baseline
  train_grpo_gsm8k.sh         # plain GRPO baseline
  train_grpo_correct_gsm8k.sh # GRPO-correct (correct-only length bonus)
  test_grpo_correct_reward.py # unit tests for GRPO-correct reward
  run_smoke_grpo.sh           # 1.5B GRPO smoke launcher
src/cav_rl/
  config.py                   # local configs
  prompts.py                  # budget/reason/answer prompt
  parsing.py                  # CAV block parser and token span masks
  rewards.py                  # local reward utilities
  lambda_dual.py              # dual lambda_c controller
  cav.py                      # local CAV/GAE target code
  rollout.py                  # local hierarchical generate_macro_completion
  ppo.py                      # local debug PPO loop
  verl/
    main_cav_ppo.py           # Ray CAV PPO entry point
    main_baseline_ppo.py      # plain PPO baseline entry
    main_baseline_grpo.py     # plain GRPO baseline entry
    main_baseline_grpo_correct.py  # GRPO-correct entry
    grpo_correct_reward.py    # correct-only rank length reward
    hierarchical_rollout.py   # VeRL hierarchical macro rollout
    single_turn.py            # fit loop (gen + dual update)
    reward.py                 # reward manager and structural masks
    baseline_reward.py        # outcome-only GSM8K reward
    advantage.py              # cav_gae advantage registration
    masks.py                  # response text to token/macro masks
```

## Environment

The default environment file is aligned with the provided machine:

```text
Ubuntu 20.04
CUDA 12.1
torch 2.1.2
2 x NVIDIA A100-SXM4-80GB
```

Create the environment:

```bash
cd E:/BJDXYJY_intern/paper/RL/AAAI/code/CAV
conda env create -f environment.yml
conda activate cav-rl
```

Or install this project into an already prepared Ray/LLM-RL environment:

```bash
pip install -e .
```

## Data Preparation

Generate parquet files for the default GSM8K setup:

```bash
python scripts/preprocess_gsm8k.py \
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --local_dir data/gsm8k \
  --budget_actions 0,16,32,64,128
```

You can change the dataset source:

```bash
python scripts/preprocess_gsm8k.py \
  --dataset_name YOUR_DATASET \
  --dataset_config YOUR_CONFIG \
  --local_dir data/your_dataset
```

The current preprocessor expects GSM8K-style rows with `question` and `answer` fields, where the final answer can be extracted from `#### answer`.

## Optional SFT

Run format alignment before PPO. Prefer budget-fitted targets and **right padding**
(causal LM SFT). A previous fast run used left padding with `batch>1`, which
misaligned labels and collapsed format learning — that bug is fixed.

```bash
# recommended
bash scripts/train_sft_format.sh
# writes:
#   outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-v2
#   outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-v2-merged
```

## Plain PPO Baseline (no CAV)

Outcome-only CoT PPO with standard GAE. Supports Instruct backbone or CoT SFT init.

```bash
# 1) build CoT parquet (once)
PYTHONPATH=src python3 scripts/preprocess_gsm8k_baseline.py

# 2a) PPO from raw Instruct
INIT_MODEL=backbone bash scripts/train_baseline_gsm8k.sh

# 2b) optional CoT SFT, then PPO from SFT
bash scripts/train_sft_baseline.sh
INIT_MODEL=sft bash scripts/train_baseline_gsm8k.sh
```

## Plain GRPO Baseline (no CAV)

Outcome-only CoT GRPO (group-relative advantage, no critic). Default backbone is
`Qwen2.5-1.5B-Instruct`. Reuses the same baseline parquet / reward as PPO.

```bash
# 1) build CoT parquet (once; shared with PPO baseline)
PYTHONPATH=src python3 scripts/preprocess_gsm8k_baseline.py

# 2) smoke (auto-builds a 256/64 split under gsm8k_baseline_smoke)
bash scripts/run_smoke_grpo.sh

# 3) full-data GRPO (override DATA_DIR / BASE_MODEL as needed)
DATA_DIR=/home/dataset-assist-0/ZX/dataset/gsm8k_baseline \
BASE_MODEL=/home/dataset-assist-0/ZX/models/Qwen2.5-1.5B-Instruct \
bash scripts/train_grpo_gsm8k.sh
```

## GRPO-Correct (length bonus on correct only)

Separate from plain baseline GRPO. Reward order: **Right-short > Right-long >> Wrong**.
Length ranking is computed only among correct samples that share the same `uid`
(`rollout.n` group). Wrong answers get a fixed negative reward (no short-wrong bonus).

```bash
# unit test (no GPU)
PYTHONPATH=src python3 scripts/test_grpo_correct_reward.py

# train (defaults: init from stage-1 GRPO ckpt, n=4, phase-1 reward coeffs)
source scripts/use_env.sh
bash scripts/train_grpo_correct_gsm8k.sh

# or point BASE_MODEL explicitly
BASE_MODEL=/path/to/stage1/actor/huggingface \
bash scripts/train_grpo_correct_gsm8k.sh
```

Key files: `src/cav_rl/verl/grpo_correct_reward.py`,
`src/cav_rl/verl/main_baseline_grpo_correct.py`,
`scripts/train_grpo_correct_gsm8k.sh`.

## Ray PPO Training (CAV)

Default launch for 2 x A100:

```bash
bash scripts/train_cav_gsm8k.sh
```

Common overrides:

```bash
export BASE_MODEL=/path/to/Qwen2.5-3B-Instruct
export DATA_DIR=/path/to/parquet_dir
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/test.parquet
export DATA_NAME=My-CAV-Data
export OUTPUT_ROOT=/path/to/outputs
export NUM_GPUS=2
bash scripts/train_cav_gsm8k.sh
```

If your Ray PPO package stores its Hydra configs outside the installed Python package, set:

```bash
export VERL_CONFIG_PATH=/path/to/verl/trainer/config
```

If unset, the script attempts to locate the config directory from the installed `verl` Python package.

## Key Hyperparameters

```text
cav.budget_actions=[0,16,32,64,128]
cav.hierarchical_rollout=true
cav.max_macro_steps=6
cav.lambda_c=0.0005              # initial dual price
cav.dual_update=true
cav.dual_lr=0.01
cav.target_expected_tokens=32.0  # B in the dual constraint
cav.min_lambda_c=0.0
cav.max_lambda_c=0.02
algorithm.adv_estimator=cav_gae
algorithm.gamma=1.0
algorithm.lam=0.95
actor lr=1e-6
critic lr=1e-5
KL loss coef=0.001
total_training_steps=100          # shorter run; override with TOTAL_TRAINING_STEPS
test_freq=6                       # validate every 6 train steps (and on the last step)
train_batch_size=64               # safer than T3's 128 under hierarchical rollout
```

`cav.lambda_c` prices actual reasoning tokens `l_k`, not allocated budgets `b_k`. With `dual_update=true` it is updated online from batch-mean `C`.

Rollout sampling defaults to `temperature=0.3` (`ROLLOUT_TEMPERATURE`). On the current SFT checkpoint, `temperature=1.0` often yields invalid CAV tags under vLLM; prefer `0.2–0.5` for early RL.

Schedule defaults: `100` steps, validate every `6` train steps (and once on the final step). `train_batch_size` stays `64` because hierarchical generation is heavier than T3's one-shot rollout. Try `TRAIN_BATCH_SIZE=128` only if VRAM allows.

## CAV Logged Metrics

Ray PPO training steps log:

```text
cav/accuracy
cav/actual_reason_tokens/mean|max|min
cav/allocated_budget/mean|max|min
cav/format_valid_rate
cav/stop_rate
cav/lambda_c
cav/dual_gap                  # E[C] - B (when dual update is on)
```

Validation focuses on the effectiveness–efficiency trade-off:

```text
val/accuracy                  # pure task correctness
val/actual_reason_tokens      # mean actual reasoning tokens
val/actual_reason_tokens_max|min
val/allocated_budget          # mean declared budgets
val/format_valid_rate
val/stop_rate
val/lambda_c
val/reward_mean               # average CAV reward (not accuracy)
val/counts_taken              # generation turns (veRL default)
```

