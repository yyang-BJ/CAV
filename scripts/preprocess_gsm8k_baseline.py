from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from cav_rl.data import split_gsm8k_solution
from cav_rl.prompts import BASELINE_SYSTEM_PROMPT, build_baseline_user_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K for plain CoT PPO baseline")
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--local_dir", default="/home/dataset-assist-0/ZX/dataset/gsm8k_baseline")
    parser.add_argument("--train_output", default="train.parquet")
    parser.add_argument("--val_output", default="test.parquet")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    return parser.parse_args()


def convert_split(dataset_name: str, dataset_config: str, split: str, max_samples: int | None):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows = []
    for idx, row in enumerate(dataset):
        _, answer = split_gsm8k_solution(row["answer"])
        prompt = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": build_baseline_user_prompt(row["question"])},
        ]
        rows.append(
            {
                "data_source": "gsm8k-baseline",
                "prompt": prompt,
                "answer": answer,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": idx,
                    "question": row["question"],
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = convert_split(args.dataset_name, args.dataset_config, "train", args.max_train_samples)
    val_rows = convert_split(args.dataset_name, args.dataset_config, "test", args.max_val_samples)

    import pandas as pd

    pd.DataFrame(train_rows).to_parquet(output_dir / args.train_output)
    pd.DataFrame(val_rows).to_parquet(output_dir / args.val_output)
    print(f"wrote {len(train_rows)} train rows to {output_dir / args.train_output}")
    print(f"wrote {len(val_rows)} validation rows to {output_dir / args.val_output}")


if __name__ == "__main__":
    main()
