#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cav_rl.data import load_gsm8k_examples
from cav_rl.parsing import parse_completion
from cav_rl.prompts import build_chat_prompt
from cav_rl.verl.reward import gsm8k_score


def main() -> None:
    model_path = "/home/dataset-assist-0/ZX/CAV/outputs/sft-qwen2.5-3b-cav-gsm8k-fmt-merged"
    budgets = [0, 16, 32, 64, 128]
    allowed = set(budgets)
    n = 32

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    examples = load_gsm8k_examples(split="test", max_samples=n)

    n_valid = n_stop = n_correct = overflow = 0
    for i, ex in enumerate(examples, 1):
        prompt = build_chat_prompt(tok, ex.question, budgets, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        parsed = parse_completion(text, allowed, tokenizer=tok)
        n_valid += int(parsed.valid_format)
        n_stop += int(parsed.has_stop)
        overflow += sum(1 for d in parsed.decisions if d.overflow)
        n_correct += int(gsm8k_score(parsed.answer, ex.answer) > 0)
        if i % 4 == 0 or i == n:
            print(
                f"[{i}/{n}] running_format={n_valid / i:.3f} stop={n_stop / i:.3f} acc={n_correct / i:.3f}",
                flush=True,
            )
            if i == 4:
                print("sample0:", text[:240].replace("\n", " | "), flush=True)

    print(
        f"FINAL format_valid={n_valid / n:.3f} stop_rate={n_stop / n:.3f} "
        f"accuracy={n_correct / n:.3f} overflow_macros={overflow} n={n}",
        flush=True,
    )


if __name__ == "__main__":
    main()
