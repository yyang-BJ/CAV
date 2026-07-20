"""GRPO-correct reward: outcome + strict-format bonus + correct-format length rank.

Policy:
  - correctness is primary;
  - ``####`` strict format gets an explicit bonus / missing-format penalty;
  - length ranking only among correct+strict-format samples with length >= floor;
  - wrong / unparsable never get a short-length bonus.

Baseline GRPO reward in ``baseline_reward.py`` is left unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from cav_rl.verl.baseline_reward import STRICT_ANSWER_RE
from cav_rl.verl.reward import NUMBER_RE, _decimal_equal, _last_number


@dataclass
class GrpoCorrectRewardConfig:
    """Correctness + strict-format reward with floor-gated length ranking."""

    correct_reward: float = 1.0
    wrong_reward: float = -0.5
    unparsable_reward: float = -1.0
    strict_format_bonus: float = 0.2
    missing_format_penalty: float = -0.2
    length_bonus_weight: float = 0.01
    # Only correct+strict-format responses with length >= floor enter ranking.
    length_floor: float = 270.0
    length_floor_softness: float = 40.0  # unused; CLI compat
    length_excess_ref: float = 0.0  # unused; CLI compat
    length_score_max: float = 1.0
    length_score_min: float = 0.0
    length_score_neutral: float = 0.0
    min_correct_for_length_ranking: int = 2
    extract_method: str = "flexible"
    debug_print_responses: bool = False


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "")
    return _last_number(cleaned) or cleaned or None


def extract_final_answer(completion: str, method: str = "flexible") -> str | None:
    """Extract a numeric final answer; return None when no number is present.

    Note: upstream ``_last_number`` falls back to the raw string when no digits
    exist; for GRPO-correct we treat that as unparsable.
    """
    if not completion:
        return None
    if method == "strict":
        matches = STRICT_ANSWER_RE.findall(completion)
        if not matches:
            return None
        return matches[-1].replace(",", "")
    # flexible: #### preferred, else last number literal (no raw-text fallback).
    matches = STRICT_ANSWER_RE.findall(completion)
    if matches:
        return matches[-1].replace(",", "")
    num_matches = NUMBER_RE.findall(completion)
    if not num_matches:
        return None
    return num_matches[-1].replace(",", "")


def rank_lengths_ascending(lengths: Sequence[float]) -> list[float]:
    """Shortest -> rank 0, longest -> rank n-1. Equal lengths share the average rank."""
    n = len(lengths)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: float(lengths[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and float(lengths[order[j + 1]]) == float(lengths[order[i]]):
            j += 1
        avg = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def compute_rank_based_length_scores(
    lengths: Sequence[float],
    *,
    length_score_max: float = 1.0,
    length_score_min: float = 0.0,
    length_score_neutral: float = 0.0,
) -> list[float]:
    """Map group lengths to scores in [min, max]; shortest→max, longest→min.

    Singleton → neutral.
    """
    n = len(lengths)
    if n <= 1:
        return [float(length_score_neutral) for _ in lengths]
    ranks = rank_lengths_ascending(lengths)
    denom = float(n - 1)
    scores = []
    for rank in ranks:
        score = length_score_max - (length_score_max - length_score_min) * (rank / denom)
        scores.append(float(score))
    return scores


def reward_group(
    completions: Sequence[str],
    gold_answer: str,
    token_lens: Sequence[float],
    cfg: GrpoCorrectRewardConfig,
) -> tuple[list[float], list[dict]]:
    """Score one GRPO group (same question / uid)."""
    infos: list[dict] = []
    gold_norm = normalize_answer(gold_answer)

    for completion, token_len in zip(completions, token_lens):
        pred = extract_final_answer(completion, method=cfg.extract_method)
        parsable = pred is not None
        pred_norm = normalize_answer(pred) if parsable else None
        correct = bool(parsable and gold_norm is not None and _decimal_equal(pred_norm, gold_norm))
        has_strict = bool(STRICT_ANSWER_RE.search(completion or ""))
        infos.append(
            {
                "completion": completion,
                "pred_answer": pred,
                "parsable": parsable,
                "correct": correct,
                "token_len": float(token_len),
                "has_strict_format": has_strict,
            }
        )

    # Length rank only among correct + strict-format + floor-eligible.
    length_scores = [cfg.length_score_neutral for _ in infos]
    floor = float(cfg.length_floor)
    eligible = [
        i
        for i, info in enumerate(infos)
        if info["correct"] and info["has_strict_format"] and info["token_len"] >= floor
    ]
    for i, info in enumerate(infos):
        if info["correct"] and info["has_strict_format"] and info["token_len"] < floor:
            length_scores[i] = float(cfg.length_score_min)

    if len(eligible) >= cfg.min_correct_for_length_ranking:
        eligible_lengths = [infos[i]["token_len"] for i in eligible]
        eligible_scores = compute_rank_based_length_scores(
            eligible_lengths,
            length_score_max=cfg.length_score_max,
            length_score_min=cfg.length_score_min,
            length_score_neutral=cfg.length_score_neutral,
        )
        for local_idx, global_idx in enumerate(eligible):
            length_scores[global_idx] = eligible_scores[local_idx]
    elif len(eligible) == 1:
        length_scores[eligible[0]] = float(cfg.length_score_max)

    rewards: list[float] = []
    for i, info in enumerate(infos):
        strict = bool(info["has_strict_format"])
        if info["correct"] and strict:
            reward = (
                cfg.correct_reward
                + cfg.strict_format_bonus
                + cfg.length_bonus_weight * length_scores[i]
            )
        elif info["correct"] and not strict:
            reward = cfg.correct_reward + cfg.missing_format_penalty
        elif info["parsable"] and strict:
            reward = cfg.wrong_reward + cfg.strict_format_bonus
        elif info["parsable"] and not strict:
            reward = cfg.wrong_reward + cfg.missing_format_penalty
        else:
            reward = cfg.unparsable_reward
        info["length_score"] = float(length_scores[i])
        info["reward"] = float(reward)
        rewards.append(float(reward))
    return rewards, infos


class GrpoCorrectRewardManager:
    """veRL reward manager with uid-grouped correct-only length ranking."""

    def __init__(
        self,
        tokenizer,
        reward_config: GrpoCorrectRewardConfig | None = None,
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_config = reward_config or GrpoCorrectRewardConfig()
        self.num_examine = num_examine

    def __call__(self, data):
        responses = data.batch["responses"]
        n = int(responses.shape[0])
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        completions: list[str] = []
        token_lens: list[float] = []
        golds: list[str] = []
        for i in range(n):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            valid_response_ids = item.batch["responses"][:valid_response_len]
            response_text = self.tokenizer.decode(valid_response_ids.int(), skip_special_tokens=True)
            gold = str(item.non_tensor_batch.get("answer", "") or "")
            completions.append(response_text)
            token_lens.append(float(valid_response_len))
            golds.append(gold)
            if i < 2 and bool(self.reward_config.debug_print_responses):
                print(
                    f"[grpo-correct-reward] i={i} resp_len={valid_response_len} "
                    f"text={response_text[:240]!r}",
                    flush=True,
                )

        uids = data.non_tensor_batch.get("uid")
        groups: dict[str, list[int]] = defaultdict(list)
        if uids is None:
            for i in range(n):
                groups[str(i)].append(i)
        else:
            for i in range(n):
                groups[str(uids[i])].append(i)

        rewards = [0.0] * n
        accuracies = [0.0] * n
        parsables = [0.0] * n
        strict_fmts = [0.0] * n
        length_scores = [self.reward_config.length_score_neutral] * n
        length_bonus_applied = [0.0] * n
        groups_ge2_correct = 0
        num_groups = 0

        for group_indices in groups.values():
            num_groups += 1
            group_completions = [completions[i] for i in group_indices]
            group_lens = [token_lens[i] for i in group_indices]
            # Same gold within a GRPO group; take first.
            group_gold = golds[group_indices[0]]
            group_rewards, infos = reward_group(
                group_completions,
                group_gold,
                group_lens,
                self.reward_config,
            )
            n_correct = sum(1 for info in infos if info["correct"])
            n_eligible = sum(
                1
                for info in infos
                if info["correct"]
                and info["has_strict_format"]
                and info["token_len"] >= self.reward_config.length_floor
            )
            if n_correct >= self.reward_config.min_correct_for_length_ranking:
                groups_ge2_correct += 1
            for local_idx, global_idx in enumerate(group_indices):
                info = infos[local_idx]
                rewards[global_idx] = group_rewards[local_idx]
                accuracies[global_idx] = 1.0 if info["correct"] else 0.0
                parsables[global_idx] = 1.0 if info["parsable"] else 0.0
                strict_fmts[global_idx] = 1.0 if info["has_strict_format"] else 0.0
                length_scores[global_idx] = float(info["length_score"])
                if (
                    info["correct"]
                    and info["has_strict_format"]
                    and (
                        n_eligible >= self.reward_config.min_correct_for_length_ranking
                        or (
                            n_eligible == 1
                            and info["token_len"] >= self.reward_config.length_floor
                        )
                    )
                ):
                    length_bonus_applied[global_idx] = 1.0

        for i in range(n):
            valid_len = int(token_lens[i])
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] += float(rewards[i])

        data.non_tensor_batch["baseline_accuracy"] = np.array(accuracies, dtype=np.float32)
        data.non_tensor_batch["baseline_response_length"] = np.array(token_lens, dtype=np.float32)
        data.non_tensor_batch["baseline_has_strict_format"] = np.array(strict_fmts, dtype=np.float32)
        data.non_tensor_batch["cav_accuracy"] = data.non_tensor_batch["baseline_accuracy"]
        data.non_tensor_batch["cav_actual_reason_tokens"] = data.non_tensor_batch["baseline_response_length"]

        data.non_tensor_batch["grpo_correct_reward"] = np.array(rewards, dtype=np.float32)
        data.non_tensor_batch["grpo_correct_parsable"] = np.array(parsables, dtype=np.float32)
        data.non_tensor_batch["grpo_correct_length_score"] = np.array(length_scores, dtype=np.float32)
        data.non_tensor_batch["grpo_correct_length_bonus_applied"] = np.array(
            length_bonus_applied, dtype=np.float32
        )
        data.non_tensor_batch["grpo_correct_unparsable"] = np.array(
            [1.0 - p for p in parsables], dtype=np.float32
        )
        # Scalar diagnostics for the current batch (broadcast for metric helpers).
        data.non_tensor_batch["grpo_correct_frac_groups_ge2_correct"] = np.array(
            [float(groups_ge2_correct / max(num_groups, 1))] * n, dtype=np.float32
        )

        return reward_tensor, np.array(rewards, dtype=np.float32)


class GrpoCorrectValidationRewardManager(GrpoCorrectRewardManager):
    """Validation: no group length ranking effect when n=1; emit requested val/* keys."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._accuracies: list[float] = []
        self._response_lens: list[float] = []
        self._strict_formats: list[float] = []
        self._parsables: list[float] = []
        self._rewards: list[float] = []

    def __call__(self, data):
        reward_tensor, final_rewards = super().__call__(data)
        self._accuracies.extend(data.non_tensor_batch["baseline_accuracy"].tolist())
        self._response_lens.extend(data.non_tensor_batch["baseline_response_length"].tolist())
        self._strict_formats.extend(data.non_tensor_batch["baseline_has_strict_format"].tolist())
        self._parsables.extend(data.non_tensor_batch["grpo_correct_parsable"].tolist())
        self._rewards.extend(final_rewards.tolist())
        return reward_tensor

    def pop_metrics(self) -> dict[str, float]:
        if not self._accuracies:
            return {}
        acc = np.asarray(self._accuracies, dtype=np.float64)
        lens = np.asarray(self._response_lens, dtype=np.float64)
        fmt = np.asarray(self._strict_formats, dtype=np.float64)
        parsable = np.asarray(self._parsables, dtype=np.float64)
        correct_mask = acc > 0.5
        wrong_mask = ~correct_mask

        metrics: dict[str, float] = {
            "val/accuracy": float(acc.mean()),
            "val/reward_mean": float(np.asarray(self._rewards, dtype=np.float64).mean()),
            "val/avg_completion_tokens": float(lens.mean()) if lens.size else 0.0,
            "val/format_success_rate": float(fmt.mean()) if fmt.size else 0.0,
            "val/strict_format_rate": float(fmt.mean()) if fmt.size else 0.0,
            "val/unparsable_rate": float(1.0 - parsable.mean()) if parsable.size else 0.0,
            "val/parsable_rate": float(parsable.mean()) if parsable.size else 0.0,
        }
        if correct_mask.any():
            metrics["val/correct_avg_tokens"] = float(lens[correct_mask].mean())
        else:
            metrics["val/correct_avg_tokens"] = 0.0
        if wrong_mask.any():
            metrics["val/wrong_avg_tokens"] = float(lens[wrong_mask].mean())
        else:
            metrics["val/wrong_avg_tokens"] = 0.0

        # Keep aliases used by existing dashboards.
        metrics["val/response_length/mean"] = metrics["val/avg_completion_tokens"]
        metrics["val/response_length/count"] = float(lens.size)
        if lens.size:
            metrics["val/response_length/min"] = float(lens.min())
            metrics["val/response_length/max"] = float(lens.max())
            metrics["val/response_length/median"] = float(np.median(lens))

        self.reset_stats()
        return metrics


def compute_grpo_correct_train_metrics(batch) -> dict[str, float]:
    """Extra train-batch diagnostics for the metric patch."""
    non_tensor = getattr(batch, "non_tensor_batch", None) or {}
    metrics: dict[str, float] = {}

    rewards = non_tensor.get("grpo_correct_reward")
    if rewards is not None:
        arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if arr.size:
            metrics["grpo_correct/reward_mean"] = float(arr.mean())

    length_scores = non_tensor.get("grpo_correct_length_score")
    if length_scores is not None:
        arr = np.asarray(length_scores, dtype=np.float64).reshape(-1)
        if arr.size:
            metrics["grpo_correct/length_score_mean"] = float(arr.mean())

    applied = non_tensor.get("grpo_correct_length_bonus_applied")
    if applied is not None:
        arr = np.asarray(applied, dtype=np.float64).reshape(-1)
        if arr.size:
            metrics["grpo_correct/length_bonus_applied_rate"] = float(arr.mean())

    frac = non_tensor.get("grpo_correct_frac_groups_ge2_correct")
    if frac is not None:
        arr = np.asarray(frac, dtype=np.float64).reshape(-1)
        if arr.size:
            metrics["grpo_correct/frac_groups_ge2_correct"] = float(arr[0])

    unp = non_tensor.get("grpo_correct_unparsable")
    if unp is not None:
        arr = np.asarray(unp, dtype=np.float64).reshape(-1)
        if arr.size:
            metrics["grpo_correct/unparsable_rate"] = float(arr.mean())

    return metrics
