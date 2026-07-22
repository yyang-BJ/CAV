from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import dataclasses
import numpy as np
import torch

from .masks import build_cav_masks


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


@dataclass
class CAVRewardConfig:
    correct_reward: float = 1.0
    wrong_reward: float = 0.0
    # HiPER-style invalid-action penalty for malformed CAV completions.
    # ``invalid_action_penalty`` wins when set; otherwise ``format_penalty`` is used.
    format_penalty: float = 0.5
    invalid_action_penalty: float | None = 0.5
    missing_stop_penalty: float = 0.2
    invalid_budget_penalty: float = 0.1
    # Only grant answer correctness when parse_completion reports valid_format.
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


class CAVRewardManager:
    """veRL-compatible reward manager.

    It decodes each response, parses CAV blocks, writes structure masks into
    `data.batch`, and returns token-level rewards. The main outcome term is the
    gated reward ``R_answer / (1 + λ C)`` with ``C = Σ l_k`` (actual reason
    tokens), placed on the stop/answer macro anchor. Format / stop / illegal
    budget penalties remain additive extras.
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
                    f"[CAV-reward] i={i} resp_len={valid_response_len} parse_failed={parse_failed} "
                    f"fail_start={fail_start} prompt_tail={prompt_text[-80:]!r} text={response_text[:240]!r}",
                    flush=True,
                )

            # Parse only the legal prefix when rollout aborted on a bad budget span.
            prefix_len = fail_start if parse_failed else valid_response_len
            prefix_len = max(0, min(prefix_len, valid_response_len))
            if prefix_len > 0:
                prefix_ids = item.batch["responses"][:prefix_len]
                prefix_text = self.tokenizer.decode(prefix_ids.int(), skip_special_tokens=True)
                masks = build_cav_masks(self.tokenizer, prefix_text, prefix_len, self.allowed_budgets)
            else:
                masks = build_cav_masks(self.tokenizer, "", 0, self.allowed_budgets)

            score = gsm8k_score(masks.answer, gold)
            # Parse-fail aborts before a legal stop/answer; never grant correctness.
            grant_correctness = (
                (not parse_failed)
                and (masks.valid_format or not self.reward_config.correctness_requires_valid_format)
            )
            if grant_correctness and score > 0:
                answer_reward = self.reward_config.correct_reward
            else:
                answer_reward = self.reward_config.wrong_reward
            # C = Σ l_k over reason bodies; main term never goes negative when answer_reward ≥ 0.
            main_reward = gated_answer_reward(
                answer_reward,
                self.reward_config.lambda_c,
                float(masks.actual_reason_tokens),
            )

            if valid_response_len > 0:
                final_anchor = valid_response_len - 1
                if prefix_len > 0:
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
                            # Place gated trajectory reward on the stop/answer macro only.
                            if budget_value <= 0 and not parse_failed:
                                reward_tensor[i, anchor] += main_reward
                                break
                    elif not parse_failed:
                        reward_tensor[i, final_anchor] += main_reward

                if parse_failed:
                    # Synthetic terminal macro over the short bad budget span.
                    fail_macro = 0
                    if prefix_len > 0 and (masks.macro_ids[:prefix_len] >= 0).any():
                        fail_macro = int(masks.macro_ids[:prefix_len][masks.macro_ids[:prefix_len] >= 0].max().item()) + 1
                    fail_from = prefix_len
                    macro_ids[i, fail_from:valid_response_len] = fail_macro
                    budget_mask[i, fail_from:valid_response_len] = 1.0
                    reward_tensor[i, valid_response_len - 1] -= self.reward_config.format_invalid_penalty()
                else:
                    if not masks.valid_format:
                        reward_tensor[i, final_anchor] -= self.reward_config.format_invalid_penalty()
                    if not masks.has_stop:
                        reward_tensor[i, final_anchor] -= self.reward_config.missing_stop_penalty
                    invalid_budget_count = sum(1 for err in masks.errors if "not in allowed set" in err)
                    reward_tensor[i, final_anchor] -= self.reward_config.invalid_budget_penalty * invalid_budget_count

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
            valid_formats.append(0.0 if parse_failed else (1.0 if masks.valid_format else 0.0))
            has_stops.append(0.0 if parse_failed else (1.0 if masks.has_stop else 0.0))

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
        data.non_tensor_batch["cav_lambda_c"] = np.full(len(data), self.reward_config.lambda_c, dtype=np.float32)
        return reward_tensor, np.array(final_rewards, dtype=np.float32)

class CAVVeRLRewardManager(CAVRewardManager):
    """Adapter for newer veRL reward manager API."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        **kwargs,
    ):
        allowed_budgets = list(
            config.cav.get(
                "budget_actions",
                [0, 16, 32, 64, 128],
            )
        )

        reward_cfg_kwargs = {}
        if "reward_config" in kwargs:
            reward_cfg_kwargs = kwargs["reward_config"]

        valid_fields = {
            f.name for f in dataclasses.fields(CAVRewardConfig)
        }

        reward_cfg = CAVRewardConfig(
            **{
                k: v
                for k, v in reward_cfg_kwargs.items()
                if k in valid_fields
            }
        )

        super().__init__(
            tokenizer=tokenizer,
            allowed_budgets=allowed_budgets,
            reward_config=reward_cfg,
        )
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

    def __call__(self, data):
        reward_tensor, _ = super().__call__(data)
        self._accuracies.extend(data.non_tensor_batch["cav_accuracy"].tolist())
        self._actual_reason_tokens.extend(data.non_tensor_batch["cav_actual_reason_tokens"].tolist())
        self._allocated_budgets.extend(data.non_tensor_batch["cav_allocated_budget"].tolist())
        self._valid_formats.extend(data.non_tensor_batch["cav_valid_format"].tolist())
        self._has_stops.extend(data.non_tensor_batch["cav_has_stop"].tolist())
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

        metrics = {
            # Effectiveness
            "val/accuracy": float(accuracies.mean()),
            # Efficiency
            "val/actual_reason_tokens": float(reason_tokens.mean()),
            "val/actual_reason_tokens_max": float(reason_tokens.max()),
            "val/actual_reason_tokens_min": float(reason_tokens.min()),
            "val/allocated_budget": float(allocated.mean()),
            # Diagnostics (optional but useful)
            "val/format_valid_rate": float(valid_formats.mean()),
            "val/stop_rate": float(has_stops.mean()),
            "val/lambda_c": float(self.reward_config.lambda_c),
        }
        self.reset_stats()
        return metrics
