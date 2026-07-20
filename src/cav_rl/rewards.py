from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from .config import RewardConfig
from .lambda_dual import DualLambdaConfig, LambdaController
from .parsing import ParsedCompletion

# Re-export for local PPO / callers that import from rewards.
__all__ = ["LambdaController", "RewardBreakdown", "compute_reward", "grade_answer"]


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


@dataclass
class RewardBreakdown:
    total: float
    correctness: float
    computation_cost: float
    format_penalty: float
    missing_stop_penalty: float
    invalid_budget_penalty: float
    overflow_budget_penalty: float
    allocated_budget: int
    actual_reason_tokens: int
    is_correct: bool
    predicted_answer: str | None
    gold_answer: str
    macro_rewards: list[float]


def build_lambda_controller(config: RewardConfig, total_training_steps: int = 100) -> LambdaController:
    return LambdaController(
        DualLambdaConfig(
            initial_lambda_c=config.initial_lambda_c,
            target_expected_tokens=config.target_expected_tokens,
            dual_lr=config.dual_lr,
            min_lambda_c=config.min_lambda_c,
            max_lambda_c=config.max_lambda_c,
            enabled=True,
            b_start=getattr(config, "b_start", None),
            b_anneal_ratio=float(getattr(config, "b_anneal_ratio", 0.7)),
            lambda_scale_start_ratio=float(getattr(config, "lambda_scale_start_ratio", 0.1)),
            lambda_scale_end_ratio=float(getattr(config, "lambda_scale_end_ratio", 0.4)),
            total_training_steps=int(total_training_steps),
        )
    )


def _last_number(text: str | None) -> str | None:
    if not text:
        return None
    matches = NUMBER_RE.findall(text)
    if not matches:
        return text.strip()
    return matches[-1].replace(",", "")


def _decimal_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return a.strip().lower() == b.strip().lower()


def grade_answer(prediction: str | None, gold: str) -> bool:
    return _decimal_equal(_last_number(prediction), _last_number(gold))


def compute_reward(
    parsed: ParsedCompletion,
    gold_answer: str,
    config: RewardConfig,
    lambda_c: float,
) -> RewardBreakdown:
    """Main term: R' = R_answer / (1 + λ C) with C = Σ reason tokens."""
    is_correct = grade_answer(parsed.answer, gold_answer)
    grant_correctness = parsed.valid_format or not getattr(
        config, "correctness_requires_valid_format", True
    )
    if grant_correctness and is_correct:
        answer_reward = config.correct_reward
    else:
        answer_reward = config.wrong_reward
    allocated_budget = parsed.total_allocated_budget
    actual_reason_tokens = sum(decision.reason_token_count for decision in parsed.decisions)
    cost_factor = max(0.0, float(lambda_c)) * max(0, int(actual_reason_tokens))
    gated = float(answer_reward) / (1.0 + cost_factor)
    # Keep field as λC for logging; the applied main reward is ``gated``.
    computation_cost = cost_factor
    format_penalty = 0.0 if parsed.valid_format else config.format_invalid_penalty()
    missing_stop_penalty = 0.0 if parsed.has_stop else config.missing_stop_penalty
    invalid_budget_count = sum(1 for err in parsed.errors if "not in allowed set" in err)
    overflow_count = sum(1 for decision in parsed.decisions if decision.overflow)
    invalid_budget_penalty = config.invalid_budget_penalty * invalid_budget_count
    overflow_budget_penalty = config.overflow_budget_penalty * overflow_count

    macro_rewards = []
    for decision in parsed.decisions:
        reward = 0.0
        if decision.budget > 0:
            if decision.overflow:
                reward -= config.overflow_budget_penalty
        else:
            reward += gated
        macro_rewards.append(reward)

    if macro_rewards:
        macro_rewards[-1] -= format_penalty + missing_stop_penalty + invalid_budget_penalty
    else:
        macro_rewards.append(-(format_penalty + missing_stop_penalty + invalid_budget_penalty))

    total = (
        gated
        - format_penalty
        - missing_stop_penalty
        - invalid_budget_penalty
        - overflow_budget_penalty
    )
    return RewardBreakdown(
        total=total,
        correctness=gated,
        computation_cost=computation_cost,
        format_penalty=format_penalty,
        missing_stop_penalty=missing_stop_penalty,
        invalid_budget_penalty=invalid_budget_penalty,
        overflow_budget_penalty=overflow_budget_penalty,
        allocated_budget=allocated_budget,
        actual_reason_tokens=actual_reason_tokens,
        is_correct=is_correct,
        predicted_answer=parsed.answer,
        gold_answer=gold_answer,
        macro_rewards=macro_rewards,
    )
