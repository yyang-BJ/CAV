from __future__ import annotations

from dataclasses import dataclass, field
import re

import torch

from .config import RewardConfig
from .data import MathExample
from .parsing import BUDGET_RE, FieldSpan, ParsedCompletion, field_token_mask, parse_completion
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
    reason_token_count: int
    duration: int
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


def _truncate_after_tag(text: str, tag: str) -> str:
    match = re.search(re.escape(tag), text, flags=re.IGNORECASE)
    if match is None:
        return text
    return text[: match.end()]


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int, generation_kwargs: dict) -> str:
    if max_new_tokens <= 0:
        return ""
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


def generate_macro_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    allowed_budgets: list[int],
    max_macro_steps: int,
    generation_kwargs: dict,
) -> str:
    """Generate one high-level CAV macro action at a time.

    Each macro step first samples a budget block from pi(b_k | H_k). If the
    sampled budget is positive, the next call samples only that step's reasoning
    segment from pi(z_k | H_k, b_k). A zero budget samples the final answer and
    terminates. This keeps rollout states aligned with H_k instead of parsing a
    single monolithic completion after the fact.

    Parse-fail budget pieces are not appended (do not enter the trajectory).
    Missing ``</reason>`` is soft-closed after stripping dangling open tags.
    """
    completion = ""
    allowed_set = {int(b) for b in allowed_budgets}
    allowed_positive = [budget for budget in allowed_budgets if budget > 0]
    max_reason_budget = max(allowed_positive, default=max_new_tokens)

    for _ in range(max_macro_steps):
        remaining = max_new_tokens - len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if remaining <= 0:
            break

        budget_text = generate_completion(
            model,
            tokenizer,
            prompt + completion,
            min(64, remaining),
            generation_kwargs,
        )
        budget_text = _truncate_after_tag(budget_text, "</budget>")
        budget_match = BUDGET_RE.search(budget_text)
        if budget_match is None:
            # Keep a short bad span for negative learning; do not continue macros.
            keep = 32
            ids = tokenizer(budget_text, add_special_tokens=False)["input_ids"][:keep]
            completion += tokenizer.decode(ids, skip_special_tokens=True)
            break
        budget = int(budget_match.group(1))
        if budget not in allowed_set:
            keep = 32
            ids = tokenizer(budget_text, add_special_tokens=False)["input_ids"][:keep]
            completion += tokenizer.decode(ids, skip_special_tokens=True)
            break
        completion += budget_text

        remaining = max_new_tokens - len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if remaining <= 0:
            break

        if budget <= 0:
            answer_text = generate_completion(
                model,
                tokenizer,
                prompt + completion,
                min(96, remaining),
                generation_kwargs,
            )
            completion += _truncate_after_tag(answer_text, "</answer>")
            break

        payload_budget = min(max(budget, 1), max_reason_budget)
        reason_text = generate_completion(
            model,
            tokenizer,
            prompt + completion,
            min(payload_budget + 16, remaining),
            generation_kwargs,
        )
        reason_text = _truncate_after_tag(reason_text, "</reason>")
        if "</reason>" not in reason_text.lower():
            reason_text = re.sub(r"<[^>\n]*$", "", reason_text)
            # Soft-close so the next macro state stays well-formed.
            body_ids = tokenizer(reason_text, add_special_tokens=False)["input_ids"][:payload_budget]
            reason_text = tokenizer.decode(body_ids, skip_special_tokens=True) + "</reason>\n"
        completion += reason_text

    return completion


def build_rollout_sample(
    model,
    tokenizer,
    example: MathExample,
    max_new_tokens: int,
    allowed_budgets: list[int],
    reward_config: RewardConfig,
    lambda_c: float,
    generation_kwargs: dict,
    max_macro_steps: int = 6,
) -> CAVRolloutSample:
    prompt = build_chat_prompt(tokenizer, example.question, allowed_budgets, add_generation_prompt=True)
    completion = generate_macro_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        allowed_budgets,
        max_macro_steps,
        generation_kwargs,
    )
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
                reason_token_count=decision.reason_token_count,
                duration=decision.reason_token_count if decision.budget > 0 else 0,
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
