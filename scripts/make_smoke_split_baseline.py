#!/usr/bin/env python3
"""Build a small train/val pair from GSM8K baseline parquet for GRPO/PPO smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_train",
        default="/home/dataset-assist-0/ZX/dataset/gsm8k_baseline/train.parquet",
        help="Full baseline train parquet.",
    )
    parser.add_argument(
        "--src_val",
        default="/home/dataset-assist-0/ZX/dataset/gsm8k_baseline/test.parquet",
        help="Full baseline val/test parquet.",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/dataset-assist-0/ZX/dataset/gsm8k_baseline_smoke",
        help="Output directory for smoke train/test parquet files.",
    )
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_val", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output", default="train.parquet")
    parser.add_argument("--val_output", default="test.parquet")
    return parser.parse_args()


def _reindex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    extras = []
    for i, row in out.iterrows():
        info = dict(row["extra_info"]) if isinstance(row["extra_info"], dict) else {}
        src_idx = info.get("index", i)
        info["source_index"] = int(src_idx) if src_idx is not None else int(i)
        info["index"] = int(i)
        extras.append(info)
    out["extra_info"] = extras
    return out


def _sample(src: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(src)
    if n > len(df):
        raise SystemExit(f"need {n} rows but source only has {len(df)}: {src}")
    return _reindex(df.sample(n=n, random_state=seed).reset_index(drop=True))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _sample(Path(args.src_train), args.n_train, args.seed)
    val_df = _sample(Path(args.src_val), args.n_val, args.seed + 1)

    train_path = out_dir / args.train_output
    val_path = out_dir / args.val_output
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    user0 = train_df.iloc[0]["prompt"][1]["content"]
    print(f"wrote {len(train_df)} train -> {train_path}")
    print(f"wrote {len(val_df)} val   -> {val_path}")
    print(f"seed={args.seed}")
    print(f"sample user prompt head: {user0[:120]!r}")
    print(
        "Use with:\n"
        f"  DATA_DIR={out_dir} TOTAL_TRAINING_STEPS=10 TRAIN_BATCH_SIZE=8 \\\n"
        "  bash scripts/run_smoke_grpo.sh"
    )


if __name__ == "__main__":
    main()
