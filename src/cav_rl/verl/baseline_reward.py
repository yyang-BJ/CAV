"""Outcome-only GSM8K reward for the plain PPO baseline (no CAV structure)."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch

from cav_rl.verl.reward import _decimal_equal, _last_number


STRICT_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)", re.MULTILINE)


@dataclass
class BaselineRewardConfig:
    correct_reward: float = 1.0
    wrong_reward: float = 0.0
    # Optional tiny format bonus when #### is present (default off / 0).
    format_score: float = 0.0
    extract_method: str = "flexible"  # strict | flexible
    debug_print_responses: bool = False


def extract_baseline_answer(text: str, method: str = "flexible") -> str | None:
    if not text:
        return None
    if method == "strict":
        matches = STRICT_ANSWER_RE.findall(text)
        if not matches:
            return None
        return matches[-1].replace(",", "")
    # flexible: prefer ####, else last number
    matches = STRICT_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return _last_number(text)


def baseline_gsm8k_score(prediction_text: str, gold: str, method: str = "flexible") -> float:
    pred = extract_baseline_answer(prediction_text, method=method)
    return 1.0 if _decimal_equal(pred, _last_number(gold) or gold) else 0.0


class BaselineRewardManager:
    """Place outcome reward on the last response token; track length for efficiency."""

    def __init__(
        self,
        tokenizer,
        reward_config: BaselineRewardConfig | None = None,
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_config = reward_config or BaselineRewardConfig()
        self.num_examine = num_examine

    def __call__(self, data):
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        final_rewards = []
        accuracies = []
        response_lens = []
        has_strict_format = []

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            valid_response_ids = item.batch["responses"][:valid_response_len]
            response_text = self.tokenizer.decode(valid_response_ids.int(), skip_special_tokens=True)
            gold = item.non_tensor_batch["answer"]

            if i < 2 and bool(self.reward_config.debug_print_responses):
                print(
                    f"[baseline-reward] i={i} resp_len={valid_response_len} text={response_text[:240]!r}",
                    flush=True,
                )

            score = baseline_gsm8k_score(
                response_text,
                gold,
                method=self.reward_config.extract_method,
            )
            has_fmt = 1.0 if STRICT_ANSWER_RE.search(response_text or "") else 0.0
            if score > 0:
                reward = self.reward_config.correct_reward
            else:
                reward = self.reward_config.wrong_reward
                if has_fmt > 0 and self.reward_config.format_score:
                    reward += float(self.reward_config.format_score)

            if valid_response_len > 0:
                reward_tensor[i, valid_response_len - 1] += reward

            final_rewards.append(float(reward_tensor[i, :valid_response_len].sum().item()) if valid_response_len else 0.0)
            accuracies.append(float(score))
            response_lens.append(float(valid_response_len))
            has_strict_format.append(has_fmt)

        data.non_tensor_batch["baseline_accuracy"] = np.array(accuracies, dtype=np.float32)
        data.non_tensor_batch["baseline_response_length"] = np.array(response_lens, dtype=np.float32)
        data.non_tensor_batch["baseline_has_strict_format"] = np.array(has_strict_format, dtype=np.float32)
        # Alias names used by shared validate printing when present.
        data.non_tensor_batch["cav_accuracy"] = data.non_tensor_batch["baseline_accuracy"]
        data.non_tensor_batch["cav_actual_reason_tokens"] = data.non_tensor_batch["baseline_response_length"]
        return reward_tensor, np.array(final_rewards, dtype=np.float32)


class BaselineValidationRewardManager(BaselineRewardManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._accuracies: list[float] = []
        self._response_lens: list[float] = []
        self._strict_formats: list[float] = []

    def __call__(self, data):
        reward_tensor, _ = super().__call__(data)
        self._accuracies.extend(data.non_tensor_batch["baseline_accuracy"].tolist())
        self._response_lens.extend(data.non_tensor_batch["baseline_response_length"].tolist())
        self._strict_formats.extend(data.non_tensor_batch["baseline_has_strict_format"].tolist())
        return reward_tensor

    def pop_metrics(self) -> dict[str, float]:
        if not self._accuracies:
            return {}
        from cav_rl.verl.baseline_metrics import _length_distribution_metrics

        acc = np.asarray(self._accuracies, dtype=np.float64)
        lens = np.asarray(self._response_lens, dtype=np.float64)
        fmt = np.asarray(self._strict_formats, dtype=np.float64)

        metrics: dict[str, float] = {
            "val/accuracy": float(acc.mean()),
            "val/strict_format_rate": float(fmt.mean()),
        }
        # Full val set (diagnostics only).
        metrics.update(_length_distribution_metrics(lens, prefix="val/response_length"))

        correct_mask = acc > 0.5
        format_mask = fmt > 0.5
        # Primary B-calibration set: correct answer AND #### format.
        ok_mask = correct_mask & format_mask
        metrics.update(
            _length_distribution_metrics(lens[ok_mask], prefix="val/response_length_correct_format")
        )
        metrics["val/correct_format_rate"] = float(ok_mask.mean()) if ok_mask.size else 0.0
        metrics["val/correct_format_count"] = float(ok_mask.sum())

        # Secondary splits for attribution.
        metrics.update(
            _length_distribution_metrics(lens[correct_mask], prefix="val/response_length_correct")
        )
        metrics.update(
            _length_distribution_metrics(lens[~correct_mask], prefix="val/response_length_wrong")
        )

        # Aliases used for B estimation / dashboards → correct+format subset.
        if ok_mask.any():
            primary_mean = metrics["val/response_length_correct_format/mean"]
            metrics["val/token_cost_correct_format"] = primary_mean
            metrics["val/actual_reason_tokens"] = primary_mean
            metrics["val/response_length"] = primary_mean
            metrics["val/response_length_max"] = metrics["val/response_length_correct_format/max"]
            metrics["val/response_length_min"] = metrics["val/response_length_correct_format/min"]
        else:
            metrics["val/token_cost_correct_format"] = 0.0
            metrics["val/actual_reason_tokens"] = 0.0
            metrics["val/response_length"] = 0.0
            metrics["val/response_length_max"] = 0.0
            metrics["val/response_length_min"] = 0.0

        self.reset_stats()
        return metrics
