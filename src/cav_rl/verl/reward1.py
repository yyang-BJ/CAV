from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

import numpy as np
import torch

from .masks import build_cav_masks


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


@dataclass
class CAVRewardConfig:
    correct_reward: float = 1.0
    wrong_reward: float = 0.0
    # Structural format penalty (skeleton / tags / stop). Overflow is separate.
    format_penalty: float = 0.5
    invalid_action_penalty: float | None = 0.5
    # Kept for config compatibility; folded into structural format in reward1.
    missing_stop_penalty: float = 0.2
    invalid_budget_penalty: float = 0.1
    # Per overflowing reason segment (l_k > b_k).
    overflow_budget_penalty: float = 0.05
    # Guarantee: correct trajectory R >= wrong_reward + margin (after caps).
    correct_overflow_margin: float = 0.05
    # Unused for R_ans in reward1 (answer match is independent of format).
    # Kept so main_cav_ppo / older callers can still pass the flag.
    correctness_requires_valid_format: bool = True
    target_expected_tokens: float = 128.0
    lambda_c: float = 0.0005
    dual_lr: float = 0.01
    min_lambda_c: float = 0.0
    max_lambda_c: float = 0.02
    dual_update: bool = True
    debug_print_responses: bool = False

    def format_invalid_penalty(self) -> float:
        if self.invalid_action_penalty is not None:
            return float(self.invalid_action_penalty)
        return float(self.format_penalty)


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


def gsm8k_score(prediction: str | None, gold: str) -> float:
    return 1.0 if _decimal_equal(_last_number(prediction), _last_number(gold)) else 0.0


def gated_answer_reward(answer_reward: float, lambda_c: float, actual_reason_tokens: float) -> float:
    """Trajectory main reward: R' = R_answer / (1 + λ C), with C = Σ l_k."""
    cost = max(0.0, float(lambda_c)) * max(0.0, float(actual_reason_tokens))
    return float(answer_reward) / (1.0 + cost)


def _is_overflow_error(err: str) -> bool:
    return "over budget" in err


def structural_format_invalid(parse_failed: bool, errors: list[str]) -> bool:
    """Skeleton / tags / termination only — overflow is not a format failure here."""
    if parse_failed:
        return True
    return any(not _is_overflow_error(e) for e in errors)


def overflow_segment_count(errors: list[str]) -> int:
    return sum(1 for e in errors if _is_overflow_error(e))


def compose_cav_reward(
    *,
    answer_reward: float,
    lambda_c: float,
    actual_reason_tokens: float,
    format_penalty: float,
    overflow_raw: float,
    wrong_reward: float,
    correct_margin: float,
) -> tuple[float, float, float, float]:
    """Return (total_delta, main, format_pen, overflow_pen).

    Ranking guarantee (only when structural format is clean):
        correct + overflow  =>  R >= wrong_reward + correct_margin
    by capping overflow. Format penalties are never shrunk.
    """
    main = gated_answer_reward(answer_reward, lambda_c, actual_reason_tokens)
    format_pen = max(0.0, float(format_penalty))
    overflow_raw = max(0.0, float(overflow_raw))
    wrong = float(wrong_reward)
    margin = max(0.0, float(correct_margin))

    if float(answer_reward) > wrong and format_pen <= 0.0:
        max_overflow = max(0.0, main - wrong - margin)
        overflow_pen = min(overflow_raw, max_overflow)
    else:
        overflow_pen = overflow_raw

    total = main - format_pen - overflow_pen
    return total, main, format_pen, overflow_pen


class CAVRewardManager:
    """veRL reward: R = R_ans/(1+λC) - P_format - P_overflow.

    - R_ans: GSM8K match 0/1 (independent of format/overflow; parse_fail → 0)
    - P_format: structural skeleton/tags/stop only
    - P_overflow: per-segment l_k > b_k, capped so correct ≫ wrong
    """

    def __init__(
        self,
        tokenizer,
        allowed_budgets: list[int],
        reward_config: CAVRewardConfig | None = None,
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.allowed_budgets = allowed_budgets
        self.reward_config = reward_config or CAVRewardConfig()
        self.num_examine = num_examine

    def __call__(self, data):
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        budget_mask = torch.zeros_like(responses, dtype=torch.float32)
        reason_mask = torch.zeros_like(responses, dtype=torch.float32)
        executor_mask = torch.zeros_like(responses, dtype=torch.float32)
        budget_values = torch.zeros_like(responses, dtype=torch.float32)
        macro_ids = torch.full_like(responses, -1, dtype=torch.long)
        final_rewards = []
        allocated_budgets = []
        actual_reason_tokens = []
        accuracies = []
        valid_formats = []
        has_stops = []
        overflow_counts = []

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            valid_response_ids = item.batch["responses"][:valid_response_len]
            response_text = self.tokenizer.decode(valid_response_ids.int(), skip_special_tokens=True)
            gold = item.non_tensor_batch["answer"]

            parse_failed = bool(item.non_tensor_batch.get("cav_rollout_parse_failed", False))
            fail_start_raw = item.non_tensor_batch.get("cav_parse_fail_span_start", -1)
            try:
                fail_start = int(fail_start_raw)
            except (TypeError, ValueError):
                fail_start = -1
            if parse_failed and fail_start < 0:
                fail_start = 0
            if not parse_failed:
                fail_start = -1

            if i < 2 and bool(getattr(self.reward_config, "debug_print_responses", False)):
                pmask = item.batch["attention_mask"][:prompt_len]
                prompt_text = self.tokenizer.decode(prompt_ids[pmask > 0].int(), skip_special_tokens=False)
                print(
                    f"[CAV-reward1] i={i} resp_len={valid_response_len} parse_failed={parse_failed} "
                    f"fail_start={fail_start} prompt_tail={prompt_text[-80:]!r} text={response_text[:240]!r}",
                    flush=True,
                )

            prefix_len = fail_start if parse_failed else valid_response_len
            prefix_len = max(0, min(prefix_len, valid_response_len))
            if prefix_len > 0:
                prefix_ids = item.batch["responses"][:prefix_len]
                prefix_text = self.tokenizer.decode(prefix_ids.int(), skip_special_tokens=True)
                masks = build_cav_masks(self.tokenizer, prefix_text, prefix_len, self.allowed_budgets)
            else:
                masks = build_cav_masks(self.tokenizer, "", 0, self.allowed_budgets)

            score = gsm8k_score(masks.answer, gold)
            # Pure answer match; format/overflow are penalties only.
            if (not parse_failed) and score > 0:
                answer_reward = self.reward_config.correct_reward
            else:
                answer_reward = self.reward_config.wrong_reward

            struct_bad = structural_format_invalid(parse_failed, list(masks.errors))
            format_pen = self.reward_config.format_invalid_penalty() if struct_bad else 0.0
            n_overflow = overflow_segment_count(list(masks.errors))
            overflow_raw = float(self.reward_config.overflow_budget_penalty) * float(n_overflow)

            total_delta, main_reward, format_pen_applied, overflow_pen = compose_cav_reward(
                answer_reward=answer_reward,
                lambda_c=self.reward_config.lambda_c,
                actual_reason_tokens=float(masks.actual_reason_tokens),
                format_penalty=format_pen,
                overflow_raw=overflow_raw,
                wrong_reward=self.reward_config.wrong_reward,
                correct_margin=self.reward_config.correct_overflow_margin,
            )

            if valid_response_len > 0:
                final_anchor = valid_response_len - 1
                placed_main = False
                if prefix_len > 0 and not parse_failed:
                    valid_macro_ids = masks.macro_ids[:prefix_len]
                    if (valid_macro_ids >= 0).any():
                        macro_count = int(valid_macro_ids[valid_macro_ids >= 0].max().item()) + 1
                        for macro_idx in range(macro_count):
                            macro_mask = valid_macro_ids == macro_idx
                            if not macro_mask.any():
                                continue
                            budget_positions = torch.nonzero(
                                macro_mask & (masks.budget_mask[:prefix_len] > 0),
                                as_tuple=False,
                            ).flatten()
                            anchor = int(budget_positions[-1].item()) if budget_positions.numel() else int(
                                torch.nonzero(macro_mask, as_tuple=False).flatten()[0].item()
                            )
                            final_anchor = anchor
                            budget_value = float(masks.budget_values[:prefix_len][macro_mask].max().item())
                            if budget_value <= 0:
                                reward_tensor[i, anchor] += total_delta
                                placed_main = True
                                break
                    if not placed_main:
                        reward_tensor[i, final_anchor] += total_delta
                        placed_main = True
                elif parse_failed:
                    fail_macro = 0
                    if prefix_len > 0 and (masks.macro_ids[:prefix_len] >= 0).any():
                        fail_macro = int(
                            masks.macro_ids[:prefix_len][masks.macro_ids[:prefix_len] >= 0].max().item()
                        ) + 1
                    fail_from = prefix_len
                    macro_ids[i, fail_from:valid_response_len] = fail_macro
                    budget_mask[i, fail_from:valid_response_len] = 1.0
                    reward_tensor[i, valid_response_len - 1] += total_delta
                    placed_main = True
                else:
                    reward_tensor[i, final_anchor] += total_delta

            if prefix_len > 0:
                budget_mask[i, :prefix_len] = masks.budget_mask
                reason_mask[i, :prefix_len] = masks.reason_mask
                executor_mask[i, :prefix_len] = masks.executor_mask
                budget_values[i, :prefix_len] = masks.budget_values
                macro_ids[i, :prefix_len] = masks.macro_ids

            final_rewards.append(float(reward_tensor[i, :valid_response_len].sum().item()))
            allocated_budgets.append(float(masks.allocated_budget))
            actual_reason_tokens.append(float(masks.actual_reason_tokens))
            accuracies.append(0.0 if parse_failed else float(score))
            # Structural validity (overflow alone does not invalidate).
            valid_formats.append(0.0 if struct_bad else 1.0)
            has_stops.append(0.0 if parse_failed else (1.0 if masks.has_stop else 0.0))
            overflow_counts.append(float(n_overflow))

        data.batch["cav_budget_mask"] = budget_mask
        data.batch["cav_reason_mask"] = reason_mask
        data.batch["cav_executor_mask"] = executor_mask
        data.batch["cav_budget_values"] = budget_values
        data.batch["cav_macro_ids"] = macro_ids
        data.non_tensor_batch["cav_allocated_budget"] = np.array(allocated_budgets, dtype=np.float32)
        data.non_tensor_batch["cav_actual_reason_tokens"] = np.array(actual_reason_tokens, dtype=np.float32)
        data.non_tensor_batch["cav_accuracy"] = np.array(accuracies, dtype=np.float32)
        data.non_tensor_batch["cav_valid_format"] = np.array(valid_formats, dtype=np.float32)
        data.non_tensor_batch["cav_has_stop"] = np.array(has_stops, dtype=np.float32)
        data.non_tensor_batch["cav_overflow_count"] = np.array(overflow_counts, dtype=np.float32)
        data.non_tensor_batch["cav_lambda_c"] = np.full(len(data), self.reward_config.lambda_c, dtype=np.float32)
        return reward_tensor, np.array(final_rewards, dtype=np.float32)


class CAVValidationRewardManager(CAVRewardManager):
    """Validation reward manager that also aggregates effectiveness/efficiency stats."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._accuracies: list[float] = []
        self._actual_reason_tokens: list[float] = []
        self._allocated_budgets: list[float] = []
        self._valid_formats: list[float] = []
        self._has_stops: list[float] = []
        self._overflow_counts: list[float] = []

    def __call__(self, data):
        reward_tensor, _ = super().__call__(data)
        self._accuracies.extend(data.non_tensor_batch["cav_accuracy"].tolist())
        self._actual_reason_tokens.extend(data.non_tensor_batch["cav_actual_reason_tokens"].tolist())
        self._allocated_budgets.extend(data.non_tensor_batch["cav_allocated_budget"].tolist())
        self._valid_formats.extend(data.non_tensor_batch["cav_valid_format"].tolist())
        self._has_stops.extend(data.non_tensor_batch["cav_has_stop"].tolist())
        self._overflow_counts.extend(data.non_tensor_batch["cav_overflow_count"].tolist())
        return reward_tensor

    def pop_metrics(self) -> dict[str, float]:
        """Return validation metrics focused on effectiveness vs efficiency."""
        if not self._accuracies:
            return {}

        accuracies = np.asarray(self._accuracies, dtype=np.float64)
        reason_tokens = np.asarray(self._actual_reason_tokens, dtype=np.float64)
        allocated = np.asarray(self._allocated_budgets, dtype=np.float64)
        valid_formats = np.asarray(self._valid_formats, dtype=np.float64)
        has_stops = np.asarray(self._has_stops, dtype=np.float64)
        overflow_counts = np.asarray(self._overflow_counts, dtype=np.float64)

        metrics = {
            "val/accuracy": float(accuracies.mean()),
            "val/actual_reason_tokens": float(reason_tokens.mean()),
            "val/actual_reason_tokens_max": float(reason_tokens.max()),
            "val/actual_reason_tokens_min": float(reason_tokens.min()),
            "val/allocated_budget": float(allocated.mean()),
            "val/format_valid_rate": float(valid_formats.mean()),
            "val/stop_rate": float(has_stops.mean()),
            "val/overflow_rate": float((overflow_counts > 0).mean()),
            "val/overflow_count_mean": float(overflow_counts.mean()),
            "val/lambda_c": float(self.reward_config.lambda_c),
        }
        self.reset_stats()
        return metrics
