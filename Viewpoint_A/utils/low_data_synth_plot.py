#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot low-data replacement curves")
    p.add_argument("--metrics-csv", type=Path, default=Path("outputs_low_data") / "low_data_metrics.csv")
    p.add_argument("--plots-dir", type=Path, default=Path("plots_low_data"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    if df.empty:
        raise ValueError(f"No rows in metrics file: {args.metrics_csv}")

    dfr = df[df["variant"] == "real_only"].sort_values("level_pct")
    dfs = df[df["variant"] == "real_plus_synth"].sort_values("level_pct")

    # Plot 1: target accent WER vs real %
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dfr["level_pct"], dfr["target_accent_wer"], marker="o", label="real-only")
    ax.plot(dfs["level_pct"], dfs["target_accent_wer"], marker="o", label="real + synthetic")
    ax.set_xlabel("% real data")
    ax.set_ylabel("WER (target accent)")
    ax.set_title("Low-data replacement curve (target accent WER)")
    ax.set_xticks(sorted(df["level_pct"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.plots_dir / "target_accent_wer_curve.png", dpi=170)
    plt.close()

    # Plot 2: macro WER vs real %
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dfr["level_pct"], dfr["macro_wer"], marker="o", label="real-only")
    ax.plot(dfs["level_pct"], dfs["macro_wer"], marker="o", label="real + synthetic")
    ax.set_xlabel("% real data")
    ax.set_ylabel("Macro WER")
    ax.set_title("Low-data replacement curve (macro WER)")
    ax.set_xticks(sorted(df["level_pct"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.plots_dir / "macro_wer_curve.png", dpi=170)
    plt.close()

    print(f"Saved plots in: {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
