from __future__ import annotations

from dataclasses import dataclass

import torch

from cav_rl.parsing import FieldSpan, parse_completion


@dataclass
class CAVMaskResult:
    budget_mask: torch.Tensor
    executor_mask: torch.Tensor
    macro_ids: torch.Tensor
    budget_values: torch.Tensor
    allocated_budget: int
    actual_reason_tokens: int
    answer: str | None
    valid_format: bool
    has_stop: bool
    errors: list[str]


def _char_to_token_mask(tokenizer, text: str, span: FieldSpan, response_len: int) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = torch.zeros(response_len, dtype=torch.float32)
    for idx, (start, end) in enumerate(encoded["offset_mapping"][:response_len]):
        if start < span.end and end > span.start:
            mask[idx] = 1.0
    return mask


def build_cav_masks(
    tokenizer,
    response_text: str,
    response_len: int,
    allowed_budgets: list[int],
) -> CAVMaskResult:
    parsed = parse_completion(response_text, set(allowed_budgets), tokenizer=tokenizer)
    budget_mask = torch.zeros(response_len, dtype=torch.float32)
    executor_mask = torch.zeros(response_len, dtype=torch.float32)
    macro_ids = torch.full((response_len,), -1, dtype=torch.long)
    budget_values = torch.zeros(response_len, dtype=torch.float32)

    for field in parsed.fields:
        field_mask = _char_to_token_mask(tokenizer, response_text, field, response_len)
        if field.name == "budget":
            budget_mask += field_mask
        elif field.name in {"reason", "answer"}:
            executor_mask += field_mask
        if field.macro_index is not None:
            macro_ids[field_mask.bool()] = int(field.macro_index)

    decision_by_index = {decision.macro_index: decision for decision in parsed.decisions}
    for token_idx in range(response_len):
        macro_idx = int(macro_ids[token_idx].item())
        if macro_idx >= 0 and macro_idx in decision_by_index:
            budget_values[token_idx] = float(max(decision_by_index[macro_idx].budget, 0))

    return CAVMaskResult(
        budget_mask=budget_mask.clamp_max(1.0),
        executor_mask=executor_mask.clamp_max(1.0),
        macro_ids=macro_ids,
        budget_values=budget_values,
        allocated_budget=parsed.total_allocated_budget,
        actual_reason_tokens=sum(decision.reason_token_count for decision in parsed.decisions),
        answer=parsed.answer,
        valid_format=parsed.valid_format,
        has_stop=parsed.has_stop,
        errors=parsed.errors,
    )

