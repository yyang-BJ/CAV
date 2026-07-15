from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from .config import RewardConfig
from .parsing import ParsedCompletion


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


@dataclass
class RewardBreakdown:
    total: float
    correctness: float
    computation_cost: float
    actual_token_cost: float
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


class LambdaController:
    """Dual variable for the expected computation budget constraint."""

    def __init__(self, config: RewardConfig):
        self.value = float(config.initial_lambda_c)
        self.target_expected_budget = float(config.target_expected_budget)
        self.lr = float(config.dual_lr)
        self.min_value = float(config.min_lambda_c)
        self.max_value = float(config.max_lambda_c)

    def update(self, observed_budget: float) -> float:
        self.value += self.lr * (observed_budget - self.target_expected_budget)
        self.value = min(max(self.value, self.min_value), self.max_value)
        return self.value


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
    is_correct = grade_answer(parsed.answer, gold_answer)
    correctness = config.correct_reward if is_correct else config.wrong_reward
    allocated_budget = parsed.total_allocated_budget
    actual_reason_tokens = sum(decision.reason_token_count for decision in parsed.decisions)
    computation_cost = lambda_c * allocated_budget
    actual_token_cost = config.actual_token_price * actual_reason_tokens
    format_penalty = 0.0 if parsed.valid_format else config.format_penalty
    missing_stop_penalty = 0.0 if parsed.has_stop else config.missing_stop_penalty
    invalid_budget_count = sum(1 for err in parsed.errors if "not in allowed set" in err)
    overflow_count = sum(1 for decision in parsed.decisions if decision.overflow)
    invalid_budget_penalty = config.invalid_budget_penalty * invalid_budget_count
    overflow_budget_penalty = config.overflow_budget_penalty * overflow_count

    macro_rewards = []
    for decision in parsed.decisions:
        reward = 0.0
        if decision.budget > 0:
            reward -= lambda_c * decision.budget
            reward -= config.actual_token_price * decision.reason_token_count
            if decision.overflow:
                reward -= config.overflow_budget_penalty
        else:
            reward += correctness
        macro_rewards.append(reward)

    if macro_rewards:
        macro_rewards[-1] -= format_penalty + missing_stop_penalty + invalid_budget_penalty
    else:
        macro_rewards.append(-(format_penalty + missing_stop_penalty + invalid_budget_penalty))

    total = (
        correctness
        - computation_cost
        - actual_token_cost
        - format_penalty
        - missing_stop_penalty
        - invalid_budget_penalty
        - overflow_budget_penalty
    )
    return RewardBreakdown(
        total=total,
        correctness=correctness,
        computation_cost=computation_cost,
        actual_token_cost=actual_token_cost,
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

