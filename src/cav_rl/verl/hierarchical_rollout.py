"""True hierarchical CAV rollout for VeRL: sample b_k at H_k, then z_k / answer."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict

from cav_rl.parsing import BUDGET_RE
from cav_rl.rollout import _truncate_after_tag


GenerateFn = Callable  # (DataProto) -> DataProto


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def hierarchical_enabled(config) -> bool:
    cav = config.get("cav", {}) if hasattr(config, "get") else getattr(config, "cav", {})
    if cav is None:
        return True
    return _as_bool(cav.get("hierarchical_rollout", True), default=True)


def _strip_left_pad(token_ids: list[int], pad_id: int) -> list[int]:
    idx = 0
    while idx < len(token_ids) and token_ids[idx] == pad_id:
        idx += 1
    return token_ids[idx:]


def _strip_right_pad(token_ids: list[int], pad_id: int, eos_id: int | None) -> list[int]:
    ids = list(token_ids)
    while ids and ids[-1] == pad_id:
        ids.pop()
    if eos_id is not None and eos_id != pad_id and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return ids


def _truncate_ids_after_tag(tokenizer, token_ids: list[int], tag: str) -> list[int]:
    if not token_ids:
        return token_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    truncated = _truncate_after_tag(text, tag)
    if truncated == text:
        return token_ids
    return tokenizer.encode(truncated, add_special_tokens=False)


def _strip_dangling_open_tag(tokenizer, token_ids: list[int]) -> list[int]:
    """Remove a trailing incomplete ``<...`` fragment before soft-closing reason."""
    if not token_ids:
        return token_ids
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    cleaned = re.sub(r"<[^>\n]*$", "", text)
    if cleaned == text:
        return token_ids
    return tokenizer.encode(cleaned, add_special_tokens=False)


def _left_pad_batch(
    sequences: list[list[int]],
    pad_id: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            seq = seq[-max_len:]
        if not seq:
            continue
        input_ids[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, max_len - len(seq) :] = 1
    return input_ids, attention_mask


def _right_pad_batch(sequences: list[list[int]], pad_id: int, max_len: int) -> torch.Tensor:
    batch_size = len(sequences)
    out = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            seq = seq[:max_len]
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def _build_prompt_batch(
    base_prompt_ids: list[list[int]],
    completions: list[list[int]],
    pad_id: int,
    max_model_len: int,
    next_max_tokens: int,
    meta_info: dict,
    non_tensor_batch: dict,
):
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    sequences = [base + comp for base, comp in zip(base_prompt_ids, completions)]
    max_prompt_cap = max(8, max_model_len - max(next_max_tokens, 1))
    max_len = min(max(len(s) for s in sequences), max_prompt_cap)
    input_ids, attention_mask = _left_pad_batch(sequences, pad_id, max_len)
    position_ids = compute_position_id_with_mask(attention_mask)

    nt = {k: v for k, v in non_tensor_batch.items() if k != "raw_prompt_ids"}
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.size(0),
    )
    return DataProto(batch=batch, non_tensor_batch=nt, meta_info=deepcopy(meta_info))


def _parse_budget_from_ids(tokenizer, token_ids: list[int]) -> int | None:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    match = BUDGET_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def generate_hierarchical_sequences(
    gen_batch,
    generate_fn: GenerateFn,
    tokenizer,
    *,
    max_response_length: int,
    max_model_len: int,
    allowed_budgets: list[int],
    max_macro_steps: int = 6,
    budget_max_tokens: int = 64,
    answer_max_tokens: int = 96,
    reason_tag_slack: int = 16,
    parse_fail_keep_tokens: int = 32,
):
    """Driver-side hierarchical generation matching local generate_macro_completion.

    Each macro step samples ``b_k ~ pi(.|H_k)`` then either a reason segment or the
    final answer. Output matches veRL ``generate_sequences`` so reward/GAE/PPO stay unchanged.

    On budget parse-fail: keep prior legal macros, append a short bad span for a
    negative reward signal, and stop (bad tokens never condition later macros).

    ``budget_max_tokens`` defaults to 64 (not 24) because SFT/RL models often emit a short
    preamble before ``<budget>...</budget>``; 24 was truncating before the closing tag.
    """
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    batch_size = len(gen_batch)
    device = gen_batch.batch["input_ids"].device
    keep_fail = max(int(parse_fail_keep_tokens), 1)

    base_prompt_ids = [
        _strip_left_pad(gen_batch.batch["input_ids"][i].tolist(), pad_id) for i in range(batch_size)
    ]
    prompt_tensor = gen_batch.batch["input_ids"].clone()
    prompt_attention = gen_batch.batch["attention_mask"].clone()

    completions: list[list[int]] = [[] for _ in range(batch_size)]
    done = [False] * batch_size
    parse_failed = [False] * batch_size
    parse_fail_span_start = [-1] * batch_size
    allowed_set = {int(b) for b in allowed_budgets}
    allowed_positive = [b for b in allowed_budgets if b > 0]
    max_reason_budget = max(allowed_positive) if allowed_positive else max_response_length

    base_meta = dict(gen_batch.meta_info or {})
    base_meta.setdefault("eos_token_id", eos_id)
    base_meta.setdefault("pad_token_id", pad_id)
    base_meta["recompute_log_prob"] = False
    non_tensor = dict(gen_batch.non_tensor_batch)

    def remaining(i: int) -> int:
        return max_response_length - len(completions[i])

    def append_piece(i: int, piece: list[int]) -> None:
        rem = remaining(i)
        if rem <= 0:
            done[i] = True
            return
        completions[i].extend(piece[:rem])
        if remaining(i) <= 0:
            done[i] = True

    for macro_i in range(max_macro_steps):
        active = [i for i in range(batch_size) if not done[i] and remaining(i) > 0]
        if not active:
            break

        # Phase 1: budget block for all unfinished samples.
        budget_cap = min(budget_max_tokens, max(remaining(i) for i in active))
        budget_cap = max(budget_cap, 1)
        print(
            f"[CAV-hier] macro={macro_i} phase=budget active={len(active)} max_tokens={budget_cap}",
            flush=True,
        )
        budget_batch = _build_prompt_batch(
            base_prompt_ids,
            completions,
            pad_id,
            max_model_len,
            budget_cap,
            {**base_meta, "max_tokens": budget_cap},
            non_tensor,
        )
        budget_out = generate_fn(budget_batch)

        budgets: list[int | None] = [None] * batch_size
        parse_fail_preview = []
        for i in active:
            raw = _strip_right_pad(budget_out.batch["responses"][i].tolist(), pad_id, eos_id)
            piece = _truncate_ids_after_tag(tokenizer, raw, "</budget>")
            budget = _parse_budget_from_ids(tokenizer, piece)
            # Compromise: bad budget must not condition later macros, but keep a short
            # span (+ prior legal macros) so PPO can apply an explicit negative reward.
            if budget is None or int(budget) not in allowed_set:
                preview = tokenizer.decode(piece, skip_special_tokens=True)[:120]
                parse_fail_preview.append(preview)
                parse_fail_span_start[i] = len(completions[i])
                append_piece(i, piece[:keep_fail])
                parse_failed[i] = True
                done[i] = True
                continue
            append_piece(i, piece)
            if done[i]:
                continue
            budgets[i] = int(budget)
        if parse_fail_preview:
            print(
                f"[CAV-hier] macro={macro_i} budget-parse-fail n={len(parse_fail_preview)} "
                f"(keep_span<={keep_fail}, stop continuation) preview={parse_fail_preview[0]!r}",
                flush=True,
            )

        # Phase 2: reason or answer conditioned on b_k.
        payload_active = [i for i in range(batch_size) if not done[i] and budgets[i] is not None]
        if not payload_active:
            continue

        caps = []
        for i in payload_active:
            rem = remaining(i)
            if rem <= 0:
                done[i] = True
                continue
            b = int(budgets[i])
            if b <= 0:
                caps.append(min(answer_max_tokens, rem))
            else:
                payload_budget = min(max(b, 1), max_reason_budget)
                caps.append(min(payload_budget + reason_tag_slack, rem))

        payload_active = [i for i in payload_active if not done[i]]
        if not payload_active or not caps:
            continue

        payload_cap = max(max(caps), 1)
        n_reason = sum(1 for i in payload_active if int(budgets[i]) > 0)
        n_answer = sum(1 for i in payload_active if int(budgets[i]) <= 0)
        print(
            f"[CAV-hier] macro={macro_i} phase=payload reason={n_reason} answer={n_answer} "
            f"max_tokens={payload_cap}",
            flush=True,
        )
        payload_batch = _build_prompt_batch(
            base_prompt_ids,
            completions,
            pad_id,
            max_model_len,
            payload_cap,
            {**base_meta, "max_tokens": payload_cap},
            non_tensor,
        )
        payload_out = generate_fn(payload_batch)

# After reason payload, if </reason> is missing, keep only up to budget tokens of
# content when possible by truncating raw piece to payload_budget (already done via cap).
# Prefer stopping the reason segment at the first newline-run of repeated chars? skip.

        for i in payload_active:
            rem = remaining(i)
            if rem <= 0:
                done[i] = True
                continue
            b = int(budgets[i])
            raw = _strip_right_pad(payload_out.batch["responses"][i].tolist(), pad_id, eos_id)
            if b <= 0:
                piece = _truncate_ids_after_tag(tokenizer, raw[: min(answer_max_tokens, rem)], "</answer>")
                append_piece(i, piece)
                done[i] = True
            else:
                payload_budget = min(max(b, 1), max_reason_budget)
                piece = _truncate_ids_after_tag(
                    tokenizer,
                    raw[: min(payload_budget + reason_tag_slack, rem)],
                    "</reason>",
                )
                piece_text = tokenizer.decode(piece, skip_special_tokens=True)
                if "</reason>" not in piece_text.lower():
                    # Soft-enforce hierarchical contract: drop dangling open tags, then close.
                    body = _strip_dangling_open_tag(tokenizer, piece)
                    if len(body) > payload_budget:
                        body = body[:payload_budget]
                    closer = tokenizer.encode("</reason>\n", add_special_tokens=False)
                    piece = body + closer
                append_piece(i, piece)

    response_ids = _right_pad_batch(completions, pad_id, max_response_length).to(device)
    response_mask = torch.zeros_like(response_ids, dtype=prompt_attention.dtype)
    for i, seq in enumerate(completions):
        n = min(len(seq), max_response_length)
        if n > 0:
            response_mask[i, :n] = 1
        # Prior legal macros + short fail span both remain trainable.

    seq = torch.cat([prompt_tensor.to(device), response_ids], dim=-1)
    attention_mask = torch.cat([prompt_attention.to(device), response_mask], dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)
    out_non_tensor = {k: v for k, v in non_tensor.items() if k != "raw_prompt_ids"}
    out_non_tensor["cav_rollout_parse_failed"] = np.array(parse_failed, dtype=bool)
    out_non_tensor["cav_parse_fail_span_start"] = np.array(parse_fail_span_start, dtype=np.int32)

    batch = TensorDict(
        {
            "prompts": prompt_tensor.to(device),
            "responses": response_ids,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        batch_size=batch_size,
    )
    return DataProto(batch=batch, non_tensor_batch=out_non_tensor, meta_info=dict(base_meta))
