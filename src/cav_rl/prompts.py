from __future__ import annotations


SYSTEM_PROMPT = """You are a careful mathematical reasoning agent.
You control your own computation budget before each reasoning segment.

At each macro step, choose one budget from the allowed set shown in the user prompt.

If more reasoning is useful, output exactly:
<budget>N</budget>
<reason>one concise reasoning segment that fits within N tokens</reason>

If you are ready to answer, output exactly:
<budget>0</budget>
<answer>final numeric answer only</answer>

Rules:
- <budget> must contain one integer from the allowed budget set.
- Budget 0 means stop reasoning and answer.
- A positive budget must be followed by one <reason> block.
- Budget 0 must be followed by one <answer> block and no later blocks.
- Prefer the smallest budget that can make useful progress.
"""


def build_user_prompt(question: str, budget_actions: list[int]) -> str:
    actions = ", ".join(str(x) for x in sorted(set(budget_actions)))
    return f"Allowed budgets: [{actions}]\nSolve the problem.\nQuestion:\n{question}\n"


def build_chat_prompt(tokenizer, question: str, budget_actions: list[int], add_generation_prompt: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, budget_actions)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{build_user_prompt(question, budget_actions)}\n\nAssistant:\n"


def build_sft_completion(rationale: str, final_answer: str, reason_budget: int) -> str:
    rationale = " ".join((rationale or "").strip().split())
    final_answer = " ".join((final_answer or "").strip().split())
    if rationale:
        return (
            f"<budget>{reason_budget}</budget>\n"
            f"<reason>{rationale}</reason>\n"
            "<budget>0</budget>\n"
            f"<answer>{final_answer}</answer>"
        )
    return f"<budget>0</budget>\n<answer>{final_answer}</answer>"

