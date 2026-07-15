from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from cav_rl.data import split_gsm8k_solution
from cav_rl.prompts import SYSTEM_PROMPT, build_user_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--local_dir", default="data/gsm8k")
    parser.add_argument("--train_output", default="train.parquet")
    parser.add_argument("--val_output", default="test.parquet")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--budget_actions", default="0,16,32,64,128")
    return parser.parse_args()


def convert_split(dataset_name: str, dataset_config: str, split: str, max_samples: int | None, budget_actions: list[int]):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows = []
    for idx, row in enumerate(dataset):
        _, answer = split_gsm8k_solution(row["answer"])
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row["question"], budget_actions)},
        ]
        rows.append(
            {
                "data_source": "gsm8k-cav",
                "prompt": prompt,
                "answer": answer,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "question": row["question"],
                    "budget_actions": budget_actions,
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    budget_actions = [int(item.strip()) for item in args.budget_actions.split(",") if item.strip()]
    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = convert_split(args.dataset_name, args.dataset_config, "train", args.max_train_samples, budget_actions)
    val_rows = convert_split(args.dataset_name, args.dataset_config, "test", args.max_val_samples, budget_actions)

    import pandas as pd

    pd.DataFrame(train_rows).to_parquet(output_dir / args.train_output)
    pd.DataFrame(val_rows).to_parquet(output_dir / args.val_output)
    print(f"wrote {len(train_rows)} train rows to {output_dir / args.train_output}")
    print(f"wrote {len(val_rows)} validation rows to {output_dir / args.val_output}")


if __name__ == "__main__":
    main()

