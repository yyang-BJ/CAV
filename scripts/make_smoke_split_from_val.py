#!/usr/bin/env python3
"""Split the CAV validation parquet into a small train/val pair for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_val",
        default="/home/dataset-assist-0/ZX/dataset/gsm8k_cav/test.parquet",
        help="Full validation parquet (already contains the desired prompt).",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/dataset-assist-0/ZX/dataset/gsm8k_cav_smoke",
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
        info["source_val_index"] = int(src_idx) if src_idx is not None else int(i)
        info["index"] = int(i)
        extras.append(info)
    out["extra_info"] = extras
    return out


def main() -> None:
    args = parse_args()
    src = Path(args.src_val)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    n = len(df)
    need = args.n_train + args.n_val
    if need > n:
        raise SystemExit(f"need {need} rows but source only has {n}: {src}")

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_df = _reindex(shuffled.iloc[: args.n_train])
    val_df = _reindex(shuffled.iloc[args.n_train : args.n_train + args.n_val])

    train_path = out_dir / args.train_output
    val_path = out_dir / args.val_output
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    # Sanity: prompt should still carry the new budget policy.
    user0 = train_df.iloc[0]["prompt"][1]["content"]
    if "Easy/Medium/Hard" not in user0:
        print("WARNING: smoke train user prompt missing Easy/Medium/Hard marker")

    print(f"source: {src} ({n} rows)")
    print(f"wrote {len(train_df)} train -> {train_path}")
    print(f"wrote {len(val_df)} val   -> {val_path}")
    print(f"seed={args.seed}")
    print(
        "Use with:\n"
        f"  DATA_DIR={out_dir} TOTAL_TRAINING_STEPS=20 TRAIN_BATCH_SIZE=32 \\"
        "\n  bash scripts/train_cav_gsm8k.sh"
    )


if __name__ == "__main__":
    main()
