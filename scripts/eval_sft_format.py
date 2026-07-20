from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cav_rl.data import load_gsm8k_examples
from cav_rl.parsing import parse_completion
from cav_rl.prompts import build_chat_prompt
from cav_rl.verl.reward import gsm8k_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAV SFT format validity under greedy decode")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--budget_actions", default="0,16,32,64,128")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    budgets = [int(x.strip()) for x in args.budget_actions.split(",") if x.strip()]
    allowed = set(budgets)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    examples = load_gsm8k_examples(split=args.split, max_samples=args.num_samples)
    n_valid = 0
    n_stop = 0
    n_correct = 0
    overflow = 0
    for ex in examples:
        prompt = build_chat_prompt(tokenizer, ex.question, budgets, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        parsed = parse_completion(text, allowed, tokenizer=tokenizer)
        n_valid += int(parsed.valid_format)
        n_stop += int(parsed.has_stop)
        overflow += sum(1 for d in parsed.decisions if d.overflow)
        n_correct += int(gsm8k_score(parsed.answer, ex.answer) > 0)

    n = len(examples)
    print(
        f"format_valid={n_valid / n:.3f} stop_rate={n_stop / n:.3f} "
        f"accuracy={n_correct / n:.3f} overflow_macros={overflow} n={n}"
    )


if __name__ == "__main__":
    main()
