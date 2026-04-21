#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_per_accent_cap(df: pd.DataFrame, max_per_accent: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby("accent"):
        n = min(max_per_accent, len(group))
        pieces.append(group.sample(n=n, random_state=seed))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_per_accent_percent(df: pd.DataFrame, percent: float, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in df.groupby("accent"):
        n = max(1, int(round(len(group) * (percent / 100.0))))
        n = min(n, len(group))
        pieces.append(group.sample(n=n, random_state=seed))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create subset manifests from full manifests")
    p.add_argument("--src-dir", type=Path, default=Path("manifests"))
    p.add_argument("--dst-dir", type=Path, default=Path("manifests_subset"))
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--train-max-per-accent", type=int, default=0, help="Per-accent cap for train (0 disables cap mode)")
    p.add_argument("--dev-max-per-accent", type=int, default=0, help="Per-accent cap for dev (0 disables cap mode)")
    p.add_argument("--test-max-per-accent", type=int, default=0, help="Per-accent cap for test (0 disables cap mode)")

    p.add_argument("--train-percent", type=float, default=0.0, help="Per-accent train subset percentage (e.g., 10, 15)")
    p.add_argument("--dev-percent", type=float, default=0.0, help="Per-accent dev subset percentage")
    p.add_argument("--test-percent", type=float, default=0.0, help="Per-accent test subset percentage")
    return p.parse_args()


def subset_split(df: pd.DataFrame, percent: float, cap: int, seed: int) -> pd.DataFrame:
    if percent > 0:
        if percent > 100:
            raise ValueError(f"percent must be <= 100, got {percent}")
        return sample_per_accent_percent(df, percent, seed)
    if cap > 0:
        return sample_per_accent_cap(df, cap, seed)
    return df.copy()


def main() -> int:
    args = parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    fixed_train = pd.read_csv(args.src_dir / "fixed_train_real.tsv", sep="\t")
    fixed_dev = pd.read_csv(args.src_dir / "fixed_dev_real.tsv", sep="\t")
    fixed_test = pd.read_csv(args.src_dir / "fixed_test_real.tsv", sep="\t")

    subset_train = subset_split(fixed_train, args.train_percent, args.train_max_per_accent, args.seed)
    subset_dev = subset_split(fixed_dev, args.dev_percent, args.dev_max_per_accent, args.seed)
    subset_test = subset_split(fixed_test, args.test_percent, args.test_max_per_accent, args.seed)

    subset_train.to_csv(args.dst_dir / "fixed_train_real.tsv", sep="\t", index=False)
    subset_dev.to_csv(args.dst_dir / "fixed_dev_real.tsv", sep="\t", index=False)
    subset_test.to_csv(args.dst_dir / "fixed_test_real.tsv", sep="\t", index=False)

    for name in ["train_baseline.tsv", "train_balanced_real.tsv", "train_synthetic_augmented.tsv", "train_hybrid.tsv"]:
        src_path = args.src_dir / name
        if src_path.exists():
            src = pd.read_csv(src_path, sep="\t")
            sampled = subset_split(src, args.train_percent, args.train_max_per_accent, args.seed)
            sampled.to_csv(args.dst_dir / name, sep="\t", index=False)

    for name in [
        "experiment_setup.csv",
        "selected_accent_split_summary.csv",
        "balanced_real_accent_summary.csv",
        "synthetic_augmented_metadata.csv",
        "hybrid_augmented_metadata.csv",
    ]:
        src_path = args.src_dir / name
        if src_path.exists():
            pd.read_csv(src_path).to_csv(args.dst_dir / name, index=False)

    print(f"Wrote subset manifests to: {args.dst_dir}")
    for file_name in ["fixed_train_real.tsv", "fixed_dev_real.tsv", "fixed_test_real.tsv"]:
        df = pd.read_csv(args.dst_dir / file_name, sep="\t")
        print(f"{file_name}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
