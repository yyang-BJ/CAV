from __future__ import annotations

from dataclasses import dataclass
import re

from datasets import load_dataset

from .prompts import build_chat_prompt, build_sft_completion


FINAL_ANSWER_RE = re.compile(r"####\s*(.+)\s*$")


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


class CAVSFTDataset:
    def __init__(self, examples: list[MathExample], tokenizer, max_length: int, budget_actions: list[int], reason_budget: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.budget_actions = budget_actions
        self.reason_budget = reason_budget

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        prompt = build_chat_prompt(self.tokenizer, example.question, self.budget_actions, add_generation_prompt=True)
        completion = build_sft_completion(example.rationale, example.answer, self.reason_budget)
        full_text = prompt + completion
        encoded = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        encoded["labels"] = labels
        return encoded


def make_sft_collator(tokenizer):
    def collate(features: list[dict]) -> dict:
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)
        import torch

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return collate

