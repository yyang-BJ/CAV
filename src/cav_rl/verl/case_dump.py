"""Dump validation trajectories for error attribution (CAV + baseline)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cav_rl.parsing import parse_completion
from cav_rl.verl.baseline_reward import extract_baseline_answer
from cav_rl.verl.reward import gsm8k_score


def _as_list(values, n: int, default=None):
    if values is None:
        return [default] * n
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 1 and n > 1:
        return [arr.item()] * n
    out = []
    for i in range(n):
        out.append(arr[i] if i < arr.size else default)
    return out


def _as_float_list(values, n: int, default: float = 0.0) -> list[float]:
    if values is None:
        return [default] * n
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(arr[i]) if i < arr.size else default for i in range(n)]


def _as_bool_list(values, n: int, default: bool = False) -> list[bool]:
    if values is None:
        return [default] * n
    arr = np.asarray(values).reshape(-1)
    return [bool(arr[i]) if i < arr.size else default for i in range(n)]


def _question_from_item(non_tensor: dict, i: int) -> str | None:
    extra = non_tensor.get("extra_info")
    if extra is not None:
        try:
            info = extra[i]
            if isinstance(info, dict) and "question" in info:
                return str(info["question"])
        except Exception:
            pass
    return None


def collect_case_records(
    batch,
    tokenizer,
    *,
    mode: str,
    allowed_budgets: list[int] | None = None,
    global_step: int = 0,
) -> list[dict[str, Any]]:
    """Build per-sample records after reward_fn has populated non_tensor fields."""
    responses = batch.batch["responses"]
    prompts = batch.batch["prompts"]
    attention = batch.batch["attention_mask"]
    n = int(responses.shape[0])
    non_tensor = batch.non_tensor_batch
    allowed = set(int(b) for b in (allowed_budgets or [0, 16, 32, 64, 128]))

    answers = _as_list(non_tensor.get("answer"), n, default="")
    data_sources = _as_list(non_tensor.get("data_source"), n, default="unknown")
    uids = _as_list(non_tensor.get("uid"), n, default=None)
    indices = []
    extra = non_tensor.get("extra_info")
    for i in range(n):
        idx = None
        if extra is not None:
            try:
                info = extra[i]
                if isinstance(info, dict):
                    idx = info.get("index")
            except Exception:
                pass
        indices.append(idx)

    records: list[dict[str, Any]] = []
    for i in range(n):
        prompt_len = int(prompts.shape[-1])
        resp_len = int(attention[i, prompt_len:].sum().item())
        resp_ids = responses[i, :resp_len]
        prompt_ids = prompts[i][attention[i, :prompt_len] > 0]
        response_text = tokenizer.decode(resp_ids.int(), skip_special_tokens=True)
        prompt_text = tokenizer.decode(prompt_ids.int(), skip_special_tokens=False)
        gold = str(answers[i] or "")
        question = _question_from_item(non_tensor, i)

        record: dict[str, Any] = {
            "step": int(global_step),
            "mode": mode,
            "index": indices[i],
            "uid": None if uids[i] is None else str(uids[i]),
            "data_source": str(data_sources[i]),
            "question": question,
            "gold_answer": gold,
            "prompt_text": prompt_text,
            "trajectory": response_text,
            "response_length": resp_len,
        }

        if mode == "baseline":
            acc = _as_float_list(non_tensor.get("baseline_accuracy"), n)[i]
            pred = extract_baseline_answer(response_text, method="flexible")
            has_strict = _as_float_list(non_tensor.get("baseline_has_strict_format"), n)[i] > 0.5
            correct = acc > 0.5
            error_type = "ok"
            if resp_len <= 0:
                error_type = "empty_response"
            elif not correct:
                error_type = "wrong_answer" if has_strict else "missing_strict_format"
            record.update(
                {
                    "predicted_answer": pred,
                    "correct": correct,
                    "accuracy": float(acc),
                    "has_strict_format": bool(has_strict),
                    "error_type": error_type,
                }
            )
        else:
            # CAV
            acc = _as_float_list(non_tensor.get("cav_accuracy"), n)[i]
            valid_fmt = _as_float_list(non_tensor.get("cav_valid_format"), n)[i] > 0.5
            has_stop = _as_float_list(non_tensor.get("cav_has_stop"), n)[i] > 0.5
            reason_tokens = _as_float_list(non_tensor.get("cav_actual_reason_tokens"), n)[i]
            allocated = _as_float_list(non_tensor.get("cav_allocated_budget"), n)[i]
            parse_failed = _as_bool_list(non_tensor.get("cav_rollout_parse_failed"), n)[i]
            fail_start = _as_float_list(non_tensor.get("cav_parse_fail_span_start"), n, default=-1.0)[i]

            parsed = parse_completion(response_text, allowed, tokenizer=tokenizer)
            pred = parsed.answer
            # Prefer reward manager score when present; fall back to re-grade.
            correct = acc > 0.5
            if not correct and pred is not None:
                correct = gsm8k_score(pred, gold) > 0

            error_type = "ok"
            errs = parsed.errors or []
            if parse_failed:
                error_type = "rollout_budget_parse_fail"
            elif resp_len <= 0:
                error_type = "empty_response"
            elif any("over budget" in e for e in errs):
                # Overflow is separate from structural format in reward1; still tag here.
                error_type = "format_overflow"
            elif not valid_fmt:
                # Prefer a coarse bucket from parser / structural errors.
                if any("no <budget>" in e for e in errs):
                    error_type = "format_no_budget"
                elif any("no <answer>" in e for e in errs):
                    error_type = "format_no_answer"
                elif any("did not terminate" in e for e in errs):
                    error_type = "format_missing_stop"
                elif any("not in allowed set" in e for e in errs):
                    error_type = "format_illegal_budget"
                else:
                    error_type = "format_invalid"
            elif not has_stop:
                error_type = "missing_stop"
            elif not correct:
                error_type = "wrong_answer"

            record.update(
                {
                    "predicted_answer": pred,
                    "correct": bool(correct),
                    "accuracy": float(acc),
                    "valid_format": bool(valid_fmt),
                    "has_stop": bool(has_stop),
                    "actual_reason_tokens": float(reason_tokens),
                    "allocated_budget": float(allocated),
                    "rollout_parse_failed": bool(parse_failed),
                    "parse_fail_span_start": int(fail_start),
                    "parse_errors": list(parsed.errors),
                    "macro_budgets": [int(d.budget) for d in parsed.decisions],
                    "error_type": error_type,
                }
            )

        records.append(record)
    return records


def summarize_error_types(records: list[dict[str, Any]]) -> dict[str, float]:
    counter = Counter(str(r.get("error_type", "unknown")) for r in records)
    total = max(len(records), 1)
    metrics = {f"val/error_type/{k}": float(v) / total for k, v in sorted(counter.items())}
    metrics["val/error_type/count_total"] = float(len(records))
    metrics["val/error_type/count_bad"] = float(sum(1 for r in records if r.get("error_type") != "ok"))
    return metrics


def dump_case_records(
    records: list[dict[str, Any]],
    dump_dir: str | Path,
    *,
    global_step: int,
    also_bad_only: bool = True,
) -> dict[str, str]:
    """Write all / bad-only JSONL dumps. Returns written paths."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    all_path = dump_dir / f"step_{global_step}_all.jsonl"
    bad_path = dump_dir / f"step_{global_step}_bad.jsonl"
    summary_path = dump_dir / f"step_{global_step}_error_summary.json"

    with all_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    bad = [r for r in records if r.get("error_type") != "ok"]
    paths = {"all": str(all_path)}
    if also_bad_only:
        with bad_path.open("w", encoding="utf-8") as f:
            for rec in bad:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths["bad"] = str(bad_path)

    summary = {
        "step": global_step,
        "n_total": len(records),
        "n_bad": len(bad),
        "error_type_counts": dict(Counter(str(r.get("error_type")) for r in records)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"] = str(summary_path)
    print(
        f"[case-dump] step={global_step} total={len(records)} bad={len(bad)} "
        f"-> {all_path}" + (f" | {bad_path}" if also_bad_only else ""),
        flush=True,
    )
    return paths


def resolve_case_dump_dir(config) -> str | None:
    """Return dump directory, or None to disable."""
    enabled = str(config.trainer.get("dump_val_cases", True)).lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None
    explicit = config.trainer.get("validation_data_dir", None)
    if explicit:
        return str(explicit)
    local_dir = config.trainer.get("default_local_dir", None)
    if local_dir:
        return os.path.join(str(local_dir), "val_cases")
    return "val_cases"
