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
    computation_price: float = 0.0005
    actual_token_price: float = 0.0001
    format_penalty: float = 0.1
    missing_stop_penalty: float = 0.2
    invalid_budget_penalty: float = 0.1
    target_expected_budget: float = 128.0
    lambda_c: float = 0.0005


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


class CAVRewardManager:
    """veRL-compatible reward manager.

    It decodes each response, parses CAV blocks, writes structure masks into
    `data.batch`, and returns token-level rewards. Outcome reward is placed on
    the last valid response token, while budget/token/format costs are placed on
    the corresponding structural fields when possible.
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
        executor_mask = torch.zeros_like(responses, dtype=torch.float32)
        budget_values = torch.zeros_like(responses, dtype=torch.float32)
        macro_ids = torch.full_like(responses, -1, dtype=torch.long)
        final_rewards = []
        allocated_budgets = []

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            valid_response_ids = item.batch["responses"][:valid_response_len]
            response_text = self.tokenizer.decode(valid_response_ids.int(), skip_special_tokens=True)
            gold = item.non_tensor_batch["answer"]

            masks = build_cav_masks(self.tokenizer, response_text, valid_response_len, self.allowed_budgets)
            score = gsm8k_score(masks.answer, gold)
            correctness = self.reward_config.correct_reward if score > 0 else self.reward_config.wrong_reward

            budget_cost = self.reward_config.lambda_c * masks.budget_values[:valid_response_len]
            token_cost = self.reward_config.actual_token_price * masks.executor_mask[:valid_response_len]
            reward_tensor[i, :valid_response_len] -= budget_cost + token_cost

            if valid_response_len > 0:
                reward_tensor[i, valid_response_len - 1] += correctness
                if not masks.valid_format:
                    reward_tensor[i, valid_response_len - 1] -= self.reward_config.format_penalty
                if not masks.has_stop:
                    reward_tensor[i, valid_response_len - 1] -= self.reward_config.missing_stop_penalty
                invalid_budget_count = sum(1 for err in masks.errors if "not in allowed set" in err)
                reward_tensor[i, valid_response_len - 1] -= self.reward_config.invalid_budget_penalty * invalid_budget_count

            budget_mask[i, :valid_response_len] = masks.budget_mask
            executor_mask[i, :valid_response_len] = masks.executor_mask
            budget_values[i, :valid_response_len] = masks.budget_values
            macro_ids[i, :valid_response_len] = masks.macro_ids
            final_rewards.append(float(reward_tensor[i, :valid_response_len].sum().item()))
            allocated_budgets.append(float(masks.allocated_budget))

        data.batch["cav_budget_mask"] = budget_mask
        data.batch["cav_executor_mask"] = executor_mask
        data.batch["cav_budget_values"] = budget_values
        data.batch["cav_macro_ids"] = macro_ids
        data.non_tensor_batch["cav_allocated_budget"] = np.array(allocated_budgets, dtype=np.float32)
        return reward_tensor, np.array(final_rewards, dtype=np.float32)


class CAVValidationRewardManager(CAVRewardManager):
    def __call__(self, data):
        reward_tensor, _ = super().__call__(data)
        return reward_tensor

