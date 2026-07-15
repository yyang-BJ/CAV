from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import RewardConfig
from .data import MathExample
from .parsing import FieldSpan, ParsedCompletion, field_token_mask, parse_completion
from .prompts import build_chat_prompt
from .rewards import RewardBreakdown, compute_reward


@dataclass
class FieldRecord:
    name: str
    macro_index: int | None
    old_logprob: float
    advantage: float = 0.0
    target: float = 0.0
    span: FieldSpan | None = None


@dataclass
class MacroRecord:
    macro_index: int
    budget: int
    high_prefix: str
    low_prefix: str
    reward: float
    high_advantage: float = 0.0
    low_advantage: float = 0.0
    high_target: float = 0.0
    low_target: float = 0.0


@dataclass
class CAVRolloutSample:
    question: str
    gold_answer: str
    prompt: str
    completion: str
    parsed: ParsedCompletion
    reward: RewardBreakdown
    completion_token_count: int
    fields: list[FieldRecord] = field(default_factory=list)
    macros: list[MacroRecord] = field(default_factory=list)


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int, generation_kwargs: dict) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_rollout_sample(
    model,
    tokenizer,
    example: MathExample,
    max_new_tokens: int,
    allowed_budgets: list[int],
    reward_config: RewardConfig,
    lambda_c: float,
    generation_kwargs: dict,
) -> CAVRolloutSample:
    prompt = build_chat_prompt(tokenizer, example.question, allowed_budgets, add_generation_prompt=True)
    completion = generate_completion(model, tokenizer, prompt, max_new_tokens, generation_kwargs)
    parsed = parse_completion(completion, set(allowed_budgets), tokenizer=tokenizer)
    completion_token_count = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
    reward = compute_reward(parsed, example.answer, reward_config, lambda_c=lambda_c)
    sample = CAVRolloutSample(
        question=example.question,
        gold_answer=example.answer,
        prompt=prompt,
        completion=completion,
        parsed=parsed,
        reward=reward,
        completion_token_count=completion_token_count,
    )
    attach_macro_records(sample)
    return sample


def attach_macro_records(sample: CAVRolloutSample) -> None:
    sample.macros.clear()
    for idx, decision in enumerate(sample.parsed.decisions):
        high_prefix = sample.prompt + sample.completion[: decision.budget_span.start]
        low_prefix = sample.prompt + sample.completion[: decision.budget_span.end]
        reward = sample.reward.macro_rewards[idx] if idx < len(sample.reward.macro_rewards) else 0.0
        sample.macros.append(
            MacroRecord(
                macro_index=decision.macro_index,
                budget=decision.budget,
                high_prefix=high_prefix,
                low_prefix=low_prefix,
                reward=reward,
            )
        )


def attach_old_logprobs(sample: CAVRolloutSample, tokenizer, completion_logprobs: torch.Tensor) -> None:
    sample.fields.clear()
    for span in sample.parsed.fields:
        mask = field_token_mask(tokenizer, sample.completion, span)
        n = min(len(mask), completion_logprobs.numel())
        if n == 0:
            old_logprob = 0.0
        else:
            mask_tensor = torch.tensor(mask[:n], dtype=torch.bool)
            old_logprob = float(completion_logprobs[:n][mask_tensor].sum().item())
        sample.fields.append(
            FieldRecord(
                name=span.name,
                macro_index=span.macro_index,
                old_logprob=old_logprob,
                span=span,
            )
        )

