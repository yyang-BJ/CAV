from __future__ import annotations

from dataclasses import dataclass
import random
import re

from datasets import load_dataset

from .parsing import parse_completion
from .prompts import (
    build_baseline_chat_prompt,
    build_baseline_sft_completion,
    build_chat_prompt,
    build_sft_completion,
)


FINAL_ANSWER_RE = re.compile(r"####\s*(.+)\s*$")

FORMAT_TAG_STRINGS = (
    "<budget>",
    "</budget>",
    "<reason>",
    "</reason>",
    "<answer>",
    "</answer>",
)


@dataclass
class MathExample:
    question: str
    rationale: str
    answer: str


def normalize_gsm8k_answer(answer: str) -> str:
    return answer.strip().replace(",", "")


def split_gsm8k_solution(solution: str) -> tuple[str, str]:
    match = FINAL_ANSWER_RE.search(solution.strip())
    if not match:
        lines = [line.strip() for line in solution.splitlines() if line.strip()]
        fallback = lines[-1] if lines else solution.strip()
        return solution.strip(), normalize_gsm8k_answer(fallback)
    rationale = solution[: match.start()].strip()
    final = normalize_gsm8k_answer(match.group(1))
    return rationale, final


def load_gsm8k_examples(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "train",
    max_samples: int | None = None,
) -> list[MathExample]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    examples = []
    for row in dataset:
        rationale, answer = split_gsm8k_solution(row["answer"])
        examples.append(MathExample(question=row["question"], rationale=rationale, answer=answer))
    return examples


def choose_sft_mode(
    rng: random.Random,
    rationale: str,
    *,
    direct_prob: float,
    multi_prob: float,
    min_multi_chars: int = 160,
) -> str:
    u = rng.random()
    if u < direct_prob:
        return "direct"
    if u < direct_prob + multi_prob and len((rationale or "").strip()) >= min_multi_chars:
        return "multi"
    return "single"


def build_validated_sft_completion(
    example: MathExample,
    tokenizer,
    budget_actions: list[int],
    *,
    reason_budget: int | None,
    mode: str,
) -> str:
    """Build an SFT target and repair until parse_completion reports valid_format."""
    allowed = set(int(b) for b in budget_actions)
    completion = build_sft_completion(
        example.rationale,
        example.answer,
        reason_budget,
        tokenizer=tokenizer,
        allowed_budgets=list(budget_actions),
        mode=mode,
    )
    parsed = parse_completion(completion, allowed, tokenizer=tokenizer)
    if parsed.valid_format:
        return completion

    # Fallback 1: force single fitted reason.
    completion = build_sft_completion(
        example.rationale,
        example.answer,
        reason_budget,
        tokenizer=tokenizer,
        allowed_budgets=list(budget_actions),
        mode="single",
    )
    parsed = parse_completion(completion, allowed, tokenizer=tokenizer)
    if parsed.valid_format:
        return completion

    # Fallback 2: direct answer only (always structurally valid if answer non-empty).
    completion = build_sft_completion(
        "",
        example.answer,
        reason_budget,
        tokenizer=tokenizer,
        allowed_budgets=list(budget_actions),
        mode="direct",
    )
    return completion


def format_token_positions(tokenizer, input_ids: list[int], prompt_len: int) -> list[int]:
    """Return completion-token indices that belong to CAV structural tags."""
    if prompt_len >= len(input_ids):
        return []
    completion_ids = input_ids[prompt_len:]
    tag_id_seqs = [
        tokenizer(tag, add_special_tokens=False)["input_ids"]
        for tag in FORMAT_TAG_STRINGS
    ]
    hit = [False] * len(completion_ids)
    for tag_ids in tag_id_seqs:
        if not tag_ids:
            continue
        n = len(tag_ids)
        for start in range(0, len(completion_ids) - n + 1):
            if completion_ids[start : start + n] == tag_ids:
                for j in range(start, start + n):
                    hit[j] = True
    return [prompt_len + i for i, flag in enumerate(hit) if flag]


class CAVSFTDataset:
    def __init__(
        self,
        examples: list[MathExample],
        tokenizer,
        max_length: int,
        budget_actions: list[int],
        reason_budget: int | None = 64,
        *,
        seed: int = 42,
        fit_budget: bool = True,
        direct_answer_prob: float = 0.0,
        multi_macro_prob: float = 0.0,
        format_token_weight: float = 1.0,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.budget_actions = budget_actions
        self.reason_budget = reason_budget
        self.seed = seed
        self.fit_budget = fit_budget
        self.direct_answer_prob = float(direct_answer_prob)
        self.multi_macro_prob = float(multi_macro_prob)
        self.format_token_weight = float(format_token_weight)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        prompt = build_chat_prompt(self.tokenizer, example.question, self.budget_actions, add_generation_prompt=True)
        rng = random.Random(self.seed + idx)
        if self.fit_budget:
            mode = choose_sft_mode(
                rng,
                example.rationale,
                direct_prob=self.direct_answer_prob,
                multi_prob=self.multi_macro_prob,
            )
            completion = build_validated_sft_completion(
                example,
                self.tokenizer,
                self.budget_actions,
                reason_budget=self.reason_budget,
                mode=mode,
            )
        else:
            completion = build_sft_completion(example.rationale, example.answer, self.reason_budget or 64)

        full_text = prompt + completion
        encoded = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels

        if self.format_token_weight > 1.0:
            weights = [0.0 if token == -100 else 1.0 for token in labels]
            for pos in format_token_positions(self.tokenizer, list(encoded["input_ids"]), prompt_len):
                if pos < len(weights) and labels[pos] != -100:
                    weights[pos] = self.format_token_weight
            encoded["loss_weights"] = weights
        return encoded


class BaselineSFTDataset:
    """Plain CoT SFT: rationale + #### answer (no CAV tags)."""

    def __init__(
        self,
        examples: list[MathExample],
        tokenizer,
        max_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        prompt = build_baseline_chat_prompt(self.tokenizer, example.question, add_generation_prompt=True)
        completion = build_baseline_sft_completion(example.rationale, example.answer)
        full_text = prompt + completion
        encoded = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        return encoded


def make_sft_collator(tokenizer):
    """Pad a batch for causal LM SFT.

    Training must use *right* padding so ``labels`` stay aligned with ``input_ids``.
    ``load_tokenizer`` defaults to left padding for generation; SFT overrides that.
    """

    def collate(features: list[dict]) -> dict:
        import torch

        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError(
                f"SFT collator requires tokenizer.padding_side='right', got {tokenizer.padding_side!r}. "
                "Left padding misaligns labels and destroys format learning when batch_size>1."
            )

        labels = [feature.pop("labels") for feature in features]
        loss_weights = [feature.pop("loss_weights", None) for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_len - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if any(weight is not None for weight in loss_weights):
            padded_weights = []
            for label, weight in zip(labels, loss_weights):
                if weight is None:
                    weight = [0.0 if token == -100 else 1.0 for token in label]
                padded_weights.append(list(weight) + [0.0] * (max_len - len(weight)))
            batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch

    return collate
