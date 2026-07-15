# CAV-RL: Computation Allocation Value for Adaptive LLM Reasoning

This repository implements the coding version of the proposal **Learning When to Stop Thinking** after the optimization notes. It treats reasoning as a computation-allocation problem rather than a fixed-horizon generation problem.

The default backbone is:

```text
Qwen/Qwen2.5-3B-Instruct
```

The default dataset is:

```text
openai/gsm8k, config=main
```

No model or dataset is bundled. The code is prepared for the target backbone, dataset, and veRL/Ray training environment, but it does not download anything by itself unless you run the scripts.

## Method

CAV uses one autoregressive LLM policy with a HiPER-style structured interface:

```text
<budget>64</budget>
<reason>one budget-limited reasoning segment</reason>
...
<budget>0</budget>
<answer>final numeric answer only</answer>
```

At macro step `k`, the high-level allocation policy samples a budget action `b_k` from a discrete set such as:

```text
[0, 16, 32, 64, 128]
```

`b_k > 0` allocates computation to the reasoning executor. `b_k = 0` stops reasoning and emits the answer.

The allocation learning signal follows the optimized proposal:

```text
CAV(H_k, b_k, lambda_c)
  = E[V(H_{k+1}) | H_k, b_k] - V(H_k) - lambda_c * b_k
```

In sampled rollouts we use the TD/GAE estimator:

```text
delta_k = r_k + gamma * V(H_{k+1}) - V(H_k)
A_k = delta_k + gamma * lambda_gae * A_{k+1}
```

where `r_k` includes computation cost, token cost, format penalties, and final answer correctness.

## Relation to HiPER, T3, and EnvRL

The implementation follows their engineering style rather than copying their code:

- **HiPER**: structured single-LLM policy, field masks, and hierarchical credit assignment.
- **T3**: veRL recipe organization, parquet preprocessing, `verl.trainer.main_ppo`-style launch scripts, FSDP/vLLM/Ray assumptions.
- **EnvRL/GiGPO**: reward and advantage logic kept modular so new estimators can be plugged into veRL.

## Project Layout

```text
configs/
  sft_gsm8k.yaml              # optional format-alignment SFT
  cav_ppo_gsm8k.yaml          # lightweight local PPO/debug config
  verl_cav_overrides.yaml     # CAV-specific Hydra override reference
scripts/
  preprocess_gsm8k.py         # writes veRL-compatible parquet
  train_sft.py                # optional CAV format SFT
  train_cav_gsm8k.sh          # Ray/veRL PPO launcher
src/cav_rl/
  config.py                   # local configs
  prompts.py                  # budget/reason/answer prompt
  parsing.py                  # block parser and token span masks
  rewards.py                  # local reward utilities
  cav.py                      # local CAV/GAE target code
  ppo.py                      # local debug PPO loop
  verl/
    main_cav_ppo.py           # veRL RayPPOTrainer entry point
    reward.py                 # veRL reward manager and structural masks
    advantage.py              # cav_gae registration and dispatcher patch
    masks.py                  # response text to token/macro masks
```

## Environment

Create the environment:

```bash
cd E:/BJDXYJY_intern/paper/RL/AAAI/code/CAV
conda env create -f environment.yml
conda activate cav-rl
```

Or install into an existing veRL environment:

```bash
pip install -e .
```

For large-scale runs, install this package in the same environment as the veRL/T3/EnvRL fork you plan to use.

## Data Preparation

Generate veRL-compatible GSM8K parquet files:

```bash
python scripts/preprocess_gsm8k.py \
  --local_dir data/gsm8k \
  --budget_actions 0,16,32,64,128
```

The output rows contain:

```text
prompt
answer
data_source
ability
reward_model
extra_info
```

This matches the schema expected by veRL's `RLHFDataset`.

## Optional SFT

Run a format-alignment stage before PPO:

```bash
python scripts/train_sft.py --config configs/sft_gsm8k.yaml
```

This teaches Qwen2.5-3B-Instruct to emit the CAV blocks before RL.

## veRL/Ray PPO Training

The main large-scale path is:

```bash
bash scripts/train_cav_gsm8k.sh
```

Useful overrides:

```bash
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export NUM_GPUS=8
export DATA_DIR=E:/BJDXYJY_intern/paper/RL/AAAI/code/CAV/data/gsm8k
export VERL_CONFIG_PATH=E:/BJDXYJY_intern/paper/RL/AAAI/code/T3-main/verl/verl/trainer/config
```

The launcher instantiates veRL's `RayPPOTrainer`, actor/rollout worker, critic worker, optional ref worker, and resource pool. CAV customizes only:

- reward manager: parses `<budget>/<reason>/<answer>`, grades GSM8K, applies computation cost;
- masks: creates budget/executor token masks;
- advantage: registers `algorithm.adv_estimator=cav_gae`.

## Key Hyperparameters

```text
cav.budget_actions=[0,16,32,64,128]
cav.lambda_c=0.0005
cav.actual_token_price=0.0001
algorithm.adv_estimator=cav_gae
algorithm.gamma=1.0
algorithm.lam=0.95
actor lr=1e-6
critic lr=1e-5
KL loss coef=0.001
```

## Notes

`src/cav_rl/ppo.py` is kept as a local debug path. The intended scalable path is `src/cav_rl/verl/main_cav_ppo.py` plus `scripts/train_cav_gsm8k.sh`.

