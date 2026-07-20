from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import set_seed

from cav_rl.config import load_cav_ppo_config
from cav_rl.data import load_gsm8k_examples
from cav_rl.modeling import CAVCritic, load_actor, load_tokenizer, maybe_apply_lora
from cav_rl.ppo import collect_rollouts, ppo_update, save_checkpoint
from cav_rl.rewards import build_lambda_controller


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Standalone CAV PPO training (no external project required).")
    parser.add_argument("--config", default="configs/cav_ppo_gsm8k.yaml")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args(argv)


def _maybe_load_sft_adapter(actor, adapter_or_model: str | None):
    if not adapter_or_model:
        return actor
    from peft import PeftModel

    path = Path(adapter_or_model)
    if path.exists():
        return PeftModel.from_pretrained(actor, str(path))
    return load_actor(adapter_or_model)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_cav_ppo_config(args.config)
    set_seed(config.seed)
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(config.model_name_or_path)
    actor = load_actor(config.model_name_or_path, torch_dtype=dtype)
    actor = _maybe_load_sft_adapter(actor, config.sft_adapter_or_model)
    actor = maybe_apply_lora(actor, config.lora)
    actor.to(device)

    critic_name = config.critic_model_name_or_path or config.model_name_or_path
    critic = CAVCritic(critic_name, torch_dtype=dtype, pooling=config.critic.pooling)
    critic.to(device)

    train_examples = load_gsm8k_examples(
        config.dataset_name,
        config.dataset_config,
        split="train",
        max_samples=config.max_train_samples,
    )
    if not train_examples:
        raise RuntimeError("No training examples loaded. Check dataset name/network access.")

    actor_optimizer = torch.optim.AdamW(
        (p for p in actor.parameters() if p.requires_grad),
        lr=config.actor_learning_rate,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        weight_decay=config.weight_decay,
    )
    lambda_ctrl = build_lambda_controller(config.reward)

    print(
        f"[CAV] standalone PPO | device={device} | train={len(train_examples)} | "
        f"iters={config.num_iterations} | model={config.model_name_or_path}"
    )

    for step in range(1, config.num_iterations + 1):
        samples = collect_rollouts(actor, tokenizer, train_examples, config, rng, lambda_ctrl.value)
        stats = ppo_update(
            actor,
            critic,
            tokenizer,
            samples,
            actor_optimizer,
            critic_optimizer,
            config,
            lambda_ctrl.value,
        )
        lambda_ctrl.update(stats.mean_budget)

        if step % args.log_every == 0:
            print(
                f"step={step:04d} "
                f"actor={stats.actor_loss:.4f} critic={stats.critic_loss:.4f} "
                f"reward={stats.mean_reward:.4f} acc={stats.mean_correct:.3f} "
                f"budget={stats.mean_budget:.1f} reason_tokens={stats.mean_reason_tokens:.1f} "
                f"lambda_c={lambda_ctrl.value:.6f}"
            )

        if step % args.save_every == 0 or step == config.num_iterations:
            save_checkpoint(actor, critic, tokenizer, output_dir, step)
            print(f"[CAV] saved checkpoint-{step} -> {output_dir}")


if __name__ == "__main__":
    main()
