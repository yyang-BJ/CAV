from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .cav import attach_cav_targets_and_advantages
from .config import CAVPPOConfig
from .data import MathExample
from .logprobs import sum_masked_logprobs, token_logprobs_from_logits
from .parsing import field_token_mask
from .rewards import LambdaController
from .rollout import CAVRolloutSample, attach_old_logprobs, build_rollout_sample


@dataclass
class PPOStats:
    actor_loss: float
    critic_loss: float
    mean_reward: float
    mean_correct: float
    mean_budget: float
    mean_reason_tokens: float
    lambda_c: float


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _current_field_logprob(actor, tokenizer, sample: CAVRolloutSample, field) -> torch.Tensor:
    device = _device(actor)
    text = sample.prompt + sample.completion
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    output = actor(**encoded, use_cache=False)
    token_logprobs = token_logprobs_from_logits(output.logits, encoded["input_ids"])[0]
    prompt_len = len(tokenizer(sample.prompt, add_special_tokens=False)["input_ids"])
    completion_logprobs = token_logprobs[prompt_len : prompt_len + sample.completion_token_count]
    mask = field_token_mask(tokenizer, sample.completion, field.span)
    return sum_masked_logprobs(completion_logprobs, mask)


def _ppo_clip_loss(
    current_logprob: torch.Tensor,
    old_logprob: float,
    advantage: float,
    clip_range: float,
    kl_coef: float,
) -> torch.Tensor:
    old = current_logprob.new_tensor(old_logprob)
    adv = current_logprob.new_tensor(advantage)
    ratio = torch.exp(current_logprob - old)
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    approx_kl = 0.5 * torch.square(current_logprob - old)
    return -torch.min(ratio * adv, clipped * adv) + kl_coef * approx_kl


def _critic_loss(critic, tokenizer, samples: list[CAVRolloutSample]) -> torch.Tensor:
    macros = [macro for sample in samples for macro in sample.macros]
    if not macros:
        return next(critic.parameters()).new_tensor(0.0)
    device = _device(critic)

    high_encoded = tokenizer(
        [macro.high_prefix for macro in macros],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    low_encoded = tokenizer(
        [macro.low_prefix for macro in macros],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    high_values = critic(high_encoded["input_ids"], high_encoded["attention_mask"]).v_high
    low_values = critic(low_encoded["input_ids"], low_encoded["attention_mask"]).v_low
    high_targets = torch.tensor([macro.high_target for macro in macros], device=device)
    low_targets = torch.tensor([macro.low_target for macro in macros], device=device)
    return F.mse_loss(high_values, high_targets) + F.mse_loss(low_values, low_targets)


def collect_rollouts(
    actor,
    tokenizer,
    examples: list[MathExample],
    config: CAVPPOConfig,
    rng: random.Random,
    lambda_c: float,
) -> list[CAVRolloutSample]:
    generation_kwargs = {
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "do_sample": config.generation.do_sample,
    }
    batch = rng.sample(examples, k=min(config.rollout_batch_size, len(examples)))
    samples = [
        build_rollout_sample(
            actor,
            tokenizer,
            example,
            config.max_completion_length,
            config.budget_actions,
            config.reward,
            lambda_c,
            generation_kwargs,
            max_macro_steps=config.max_macro_steps,
        )
        for example in batch
    ]

    for sample in samples:
        device = _device(actor)
        encoded = tokenizer(sample.prompt + sample.completion, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = actor(**encoded, use_cache=False).logits
            logprobs = token_logprobs_from_logits(logits, encoded["input_ids"])[0]
        prompt_len = len(tokenizer(sample.prompt, add_special_tokens=False)["input_ids"])
        comp_logprobs = logprobs[prompt_len : prompt_len + sample.completion_token_count].detach().cpu()
        attach_old_logprobs(sample, tokenizer, comp_logprobs)
    return samples


def ppo_update(
    actor,
    critic,
    tokenizer,
    samples: list[CAVRolloutSample],
    actor_optimizer,
    critic_optimizer,
    config: CAVPPOConfig,
    lambda_c: float,
    lambda_controller: LambdaController | None = None,
) -> PPOStats:
    actor.train()
    critic.train()
    rng = random.Random(config.seed)
    actor_losses = []
    critic_losses = []

    for _ in range(config.ppo_epochs):
        attach_cav_targets_and_advantages(samples, critic, tokenizer, config.gamma, config.gae_lambda)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), config.mini_batch_size):
            mini = shuffled[start : start + config.mini_batch_size]
            field_losses = []
            for sample in mini:
                for field in sample.fields:
                    if field.span is None:
                        continue
                    current = _current_field_logprob(actor, tokenizer, sample, field)
                    field_losses.append(
                        _ppo_clip_loss(current, field.old_logprob, field.advantage, config.clip_range, config.kl_coef)
                    )
            actor_loss = torch.stack(field_losses).mean() if field_losses else next(actor.parameters()).new_tensor(0.0)
            critic_loss = _critic_loss(critic, tokenizer, mini)
            loss = actor_loss + config.value_coef * critic_loss

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            actor_optimizer.step()
            critic_optimizer.step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))

    rewards = [sample.reward.total for sample in samples]
    correct = [float(sample.reward.is_correct) for sample in samples]
    budgets = [sample.reward.allocated_budget for sample in samples]
    reason_tokens = [sample.reward.actual_reason_tokens for sample in samples]
    mean_reason_tokens = sum(reason_tokens) / max(len(reason_tokens), 1)
    if lambda_controller is not None:
        lambda_c = lambda_controller.update(mean_reason_tokens)
    return PPOStats(
        actor_loss=sum(actor_losses) / max(len(actor_losses), 1),
        critic_loss=sum(critic_losses) / max(len(critic_losses), 1),
        mean_reward=sum(rewards) / max(len(rewards), 1),
        mean_correct=sum(correct) / max(len(correct), 1),
        mean_budget=sum(budgets) / max(len(budgets), 1),
        mean_reason_tokens=mean_reason_tokens,
        lambda_c=lambda_c,
    )


def save_checkpoint(actor, critic, tokenizer, output_dir: str | Path, step: int) -> None:
    path = Path(output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(path / "actor")
    critic.backbone.save_pretrained(path / "critic_backbone")
    torch.save(
        {
            "v_high_head": critic.v_high_head.state_dict(),
            "v_low_head": critic.v_low_head.state_dict(),
        },
        path / "critic_value_heads.pt",
    )
    tokenizer.save_pretrained(path / "tokenizer")
