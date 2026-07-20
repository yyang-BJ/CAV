from __future__ import annotations

import re


# Standard CoT baseline (no CAV budget tags). Used by PPO/SFT baselines.
BASELINE_SYSTEM_PROMPT = """You are a careful mathematical reasoning assistant.
Solve the grade-school math problem with clear step-by-step reasoning.

Mandatory output format:
1. Write the reasoning steps first.
2. End your entire response with exactly one final line in this form:
#### <number>
3. <number> must be the final numeric answer only (digits, optional leading minus / decimal point). No units, words, commas, or extra symbols.
4. Do not put any text after that final #### line.
5. Do not use other answer wrappers (no <answer>, no markdown code fences, no "Answer:" line instead of ####).
"""


def build_baseline_user_prompt(question: str) -> str:
    return (
        "Solve the problem step by step.\n"
        "You must finish with a final line exactly like: #### <number>\n"
        f"Question:\n{question}\n"
    )


def build_baseline_chat_prompt(tokenizer, question: str, add_generation_prompt: bool = True) -> str:
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": build_baseline_user_prompt(question)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return (
        f"System:\n{BASELINE_SYSTEM_PROMPT}\n\n"
        f"User:\n{build_baseline_user_prompt(question)}\n\nAssistant:\n"
    )


def build_baseline_sft_completion(rationale: str, answer: str) -> str:
    """GSM8K-style CoT target: rationale then #### answer."""
    rationale = (rationale or "").strip()
    answer = (answer or "").strip()
    if rationale:
        return f"{rationale}\n#### {answer}"
    return f"#### {answer}"


SYSTEM_PROMPT = """You are a careful mathematical reasoning agent.
You control your own computation budget before each reasoning segment.

Allowed output shapes (copy the tags exactly; no other text outside the tags):

If more reasoning is useful:
<budget>N</budget>
<reason>one concise reasoning segment</reason>

If you are ready to answer:
<budget>0</budget>
<answer>final numeric answer only</answer>

Hard rules (must match the parser):
1. Tags must be spelled exactly: <budget>, </budget>, <reason>, </reason>, <answer>, </answer>.
   Never emit broken tags such as udget, </udget>, banswer, breason, or unclosed <... fragments.
2. Inside <budget>...</budget> put exactly one integer from the Allowed budgets list in the user prompt. Nothing else.
3. After a positive budget N, emit exactly one <reason>...</reason> block. The reason body must use at most N tokens.
4. After budget 0, emit exactly one <answer>...</answer> block, then stop. Do not add more budget/reason blocks.
5. Every completion must end with a budget-0 answer step. Do not stop after a positive-budget reason alone.
6. Do not invent extra tags, markdown fences, or free-form text outside the required blocks.
7. Inside <reason>, write only math steps for the current subgoal. Do not discuss difficulty, budget policy, or meta commentary.

Budget policy (decide BEFORE each <budget> tag):
A. Estimate difficulty from the question only:
   - Easy: 1-2 arithmetic steps, few numbers, no nested conditions.
   - Medium: several steps or one intermediate quantity to track.
   - Hard: multi-hop reasoning, percentages/rates, implicit conditions, or easy to misread.
B. Choose the FIRST positive budget by difficulty:
   - Easy -> 16 or 32. Never start with 128.
   - Medium -> 32 or 64. Prefer not to start with 128.
   - Hard -> start with 32 or 64 only. Do not open with 128; use several short segments instead.
C. Segmented loop (required for Medium/Hard; also fine for Easy if unsure):
   1) Pick a small-to-medium N and write ONE partial step in <reason> (at most N tokens).
   2) Re-read what you wrote. If the final answer is clear and checked, emit budget 0 and <answer>.
   3) If not ready, pick another small-to-medium N and continue with only the NEXT missing step.
   4) Prefer 2-4 short segments over one maximum-budget segment.
D. Use the largest allowed positive budget only if a previous shorter segment was clearly insufficient
   and you still need a long derivation in one block. Default is to avoid the maximum budget.
E. Always prefer the smallest allowed positive budget that can make useful progress; use 0 when ready.
"""


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def build_user_prompt(question: str, budget_actions: list[int]) -> str:
    actions = ", ".join(str(x) for x in sorted(set(budget_actions)))
    return (
        f"Allowed budgets: [{actions}]\n"
        "First judge Easy/Medium/Hard, pick a matching first budget "
        "(Easy: 16/32; Medium: 32/64; Hard: start 32/64, segment; avoid opening with the max budget), "
        "then solve with short reasoning segments and stop with budget 0 when ready.\n"
        f"Question:\n{question}\n"
    )


def build_chat_prompt(tokenizer, question: str, budget_actions: list[int], add_generation_prompt: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, budget_actions)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{build_user_prompt(question, budget_actions)}\n\nAssistant:\n"


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _token_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def fit_reason_to_budget(
    tokenizer,
    rationale: str,
    allowed_budgets: list[int],
    preferred_budget: int | None = None,
) -> tuple[str, int]:
    """Return (reason_text, budget) with tokenized reason length <= budget.

    Chooses the smallest allowed positive budget that fits. ``preferred_budget`` is
    only used as a soft hint when truncating to the maximum budget is required.
    """
    positive = sorted({int(b) for b in allowed_budgets if int(b) > 0})
    if not positive:
        raise ValueError("allowed_budgets must include at least one positive budget")

    rationale = _clean_text(rationale)
    n = _token_len(tokenizer, rationale)
    for budget in positive:
        if n <= budget:
            return rationale, budget

    budget = positive[-1]
    _ = preferred_budget  # API compatibility; smallest-fit policy ignores preference
    return _truncate_to_tokens(tokenizer, rationale, budget), budget


def _split_rationale_chunks(rationale: str, n_chunks: int) -> list[str]:
    text = _clean_text(rationale)
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < n_chunks:
        # Fall back to even character splits when sentence split is too coarse.
        chunk_size = max(1, (len(text) + n_chunks - 1) // n_chunks)
        return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]
    # Greedy pack sentences into n_chunks with roughly equal character mass.
    total = sum(len(p) for p in parts)
    target = total / n_chunks
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        current.append(part)
        current_len += len(part)
        if len(chunks) < n_chunks - 1 and current_len >= target:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append(" ".join(current))
    while len(chunks) < n_chunks and chunks:
        chunks.append(chunks[-1])
    return chunks[:n_chunks]


def build_sft_completion(
    rationale: str,
    final_answer: str,
    reason_budget: int | None = 64,
    *,
    tokenizer=None,
    allowed_budgets: list[int] | None = None,
    mode: str = "single",
) -> str:
    """Build a structurally valid CAV completion for SFT.

    Modes:
    - ``single``: one positive-budget reason block, then stop+answer
    - ``direct``: stop+answer only (no reasoning)
    - ``multi``: two reason blocks (when rationale is long enough), then stop+answer

    When ``tokenizer`` and ``allowed_budgets`` are provided, reason budgets are chosen
    so that tokenized reason content fits (``l_k <= b_k``). Legacy callers that only
    pass ``reason_budget`` keep the old fixed-budget API, but should prefer the fitted path.
    """
    final_answer = _clean_text(final_answer)
    allowed = list(allowed_budgets or ([0, reason_budget] if reason_budget else [0, 16, 32, 64, 128]))
    mode = (mode or "single").lower()

    if mode == "direct" or not _clean_text(rationale):
        return f"<budget>0</budget>\n<answer>{final_answer}</answer>"

    if tokenizer is None:
        # Legacy path: fixed budget, whitespace-collapsed rationale (may overflow).
        rationale = _clean_text(rationale)
        budget = int(reason_budget or 64)
        return (
            f"<budget>{budget}</budget>\n"
            f"<reason>{rationale}</reason>\n"
            "<budget>0</budget>\n"
            f"<answer>{final_answer}</answer>"
        )

    blocks: list[str] = []
    if mode == "multi":
        chunks = _split_rationale_chunks(rationale, 2)
        if len(chunks) >= 2:
            for chunk in chunks[:2]:
                reason, budget = fit_reason_to_budget(tokenizer, chunk, allowed, preferred_budget=reason_budget)
                if not reason:
                    continue
                blocks.append(f"<budget>{budget}</budget>\n<reason>{reason}</reason>")
        if not blocks:
            mode = "single"

    if mode == "single" or not blocks:
        reason, budget = fit_reason_to_budget(tokenizer, rationale, allowed, preferred_budget=reason_budget)
        if reason:
            blocks = [f"<budget>{budget}</budget>\n<reason>{reason}</reason>"]
        else:
            blocks = []

    blocks.append(f"<budget>0</budget>\n<answer>{final_answer}</answer>")
    return "\n".join(blocks)
