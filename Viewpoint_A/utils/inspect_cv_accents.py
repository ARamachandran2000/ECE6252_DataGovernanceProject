#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accent_experiment.common import load_all_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Common Voice English accent distribution")
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=PROJECT_ROOT / "cv-corpus-25.0-2026-03-09",
        help="Path to Common Voice corpus root",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "accent_distribution.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Show top K accents in console",
    )
    parser.add_argument("--approx-train-rows", type=int, default=0, help="Use only first N train rows (fast approximate mode)")
    parser.add_argument("--approx-dev-rows", type=int, default=0, help="Use only first N dev rows (fast approximate mode)")
    parser.add_argument("--approx-test-rows", type=int, default=0, help="Use only first N test rows (fast approximate mode)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_df = load_all_splits(
        args.cv_root,
        nrows_train=args.approx_train_rows if args.approx_train_rows > 0 else None,
        nrows_dev=args.approx_dev_rows if args.approx_dev_rows > 0 else None,
        nrows_test=args.approx_test_rows if args.approx_test_rows > 0 else None,
    )

    pivot = (
        all_df.pivot_table(
            index="accent",
            columns="split",
            values="sample_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in ["train", "dev", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0

    hours = (
        all_df.groupby("accent", dropna=False)["duration_s"].sum().reset_index().rename(columns={"duration_s": "hours"})
    )
    hours["hours"] = hours["hours"] / 3600.0

    out = pivot.merge(hours, on="accent", how="left")
    out = out[["accent", "train", "dev", "test", "hours"]]
    out = out.sort_values("train", ascending=False).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved accent distribution: {args.out_csv}")
    if args.approx_train_rows > 0 or args.approx_dev_rows > 0 or args.approx_test_rows > 0:
        print(
            "Approximate mode enabled:",
            f"train={args.approx_train_rows or 'all'},",
            f"dev={args.approx_dev_rows or 'all'},",
            f"test={args.approx_test_rows or 'all'}",
        )
    print(out.head(args.top_k).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
