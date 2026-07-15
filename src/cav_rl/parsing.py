from __future__ import annotations

from dataclasses import dataclass
import re


BUDGET_RE = re.compile(r"<budget>\s*(-?\d+)\s*</budget>", re.IGNORECASE)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


@dataclass
class FieldSpan:
    name: str
    start: int
    end: int
    text: str
    macro_index: int | None = None


@dataclass
class MacroDecision:
    macro_index: int
    budget: int
    budget_span: FieldSpan
    payload_span: FieldSpan | None
    reason_token_count: int = 0
    overflow: bool = False


@dataclass
class ParsedCompletion:
    decisions: list[MacroDecision]
    answer: str | None
    answer_span: FieldSpan | None
    fields: list[FieldSpan]
    valid_format: bool
    errors: list[str]

    @property
    def has_stop(self) -> bool:
        return any(decision.budget == 0 for decision in self.decisions)

    @property
    def positive_budgets(self) -> list[int]:
        return [decision.budget for decision in self.decisions if decision.budget > 0]

    @property
    def total_allocated_budget(self) -> int:
        return sum(self.positive_budgets)


def _next_payload(text: str, budget: int, pos: int, macro_index: int) -> FieldSpan | None:
    if budget > 0:
        match = REASON_RE.search(text, pos)
        if not match:
            return None
        return FieldSpan("reason", match.start(), match.end(), match.group(1).strip(), macro_index)
    match = ANSWER_RE.search(text, pos)
    if not match:
        return None
    return FieldSpan("answer", match.start(), match.end(), match.group(1).strip(), macro_index)


def parse_completion(
    text: str,
    allowed_budgets: set[int],
    tokenizer=None,
) -> ParsedCompletion:
    decisions: list[MacroDecision] = []
    fields: list[FieldSpan] = []
    errors: list[str] = []
    answer = None
    answer_span = None
    pos = 0
    macro_index = 0

    for match in BUDGET_RE.finditer(text):
        if match.start() < pos:
            continue
        budget = int(match.group(1))
        budget_span = FieldSpan("budget", match.start(), match.end(), str(budget), macro_index)
        fields.append(budget_span)
        if budget not in allowed_budgets:
            errors.append(f"budget {budget} at macro step {macro_index} is not in allowed set")

        payload_span = _next_payload(text, budget, match.end(), macro_index)
        if payload_span is None:
            errors.append(f"budget {budget} at macro step {macro_index} has no matching payload block")
            decisions.append(MacroDecision(macro_index, budget, budget_span, None))
            pos = match.end()
        else:
            fields.append(payload_span)
            reason_token_count = 0
            overflow = False
            if payload_span.name == "reason" and tokenizer is not None:
                reason_token_count = len(tokenizer(payload_span.text, add_special_tokens=False)["input_ids"])
                overflow = budget > 0 and reason_token_count > budget
                if overflow:
                    errors.append(
                        f"reason at macro step {macro_index} used {reason_token_count} tokens over budget {budget}"
                    )
            decisions.append(
                MacroDecision(
                    macro_index=macro_index,
                    budget=budget,
                    budget_span=budget_span,
                    payload_span=payload_span,
                    reason_token_count=reason_token_count,
                    overflow=overflow,
                )
            )
            pos = payload_span.end
            if budget == 0:
                answer = payload_span.text
                answer_span = payload_span
                break
        macro_index += 1

    if not decisions:
        errors.append("no <budget> block found")
    if decisions and decisions[-1].budget != 0:
        errors.append("completion did not terminate with <budget>0</budget>")
    if answer is None:
        match = ANSWER_RE.search(text)
        if match:
            answer = match.group(1).strip()
            answer_span = FieldSpan("answer", match.start(), match.end(), answer, None)
            fields.append(answer_span)
        else:
            errors.append("no <answer> block found")

    return ParsedCompletion(
        decisions=decisions,
        answer=answer,
        answer_span=answer_span,
        fields=sorted(fields, key=lambda span: span.start),
        valid_format=not errors,
        errors=errors,
    )


def field_token_mask(tokenizer, completion: str, span: FieldSpan) -> list[bool]:
    encoded = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    mask = []
    for start, end in encoded["offset_mapping"]:
        mask.append(bool(start < span.end and end > span.start))
    return mask

