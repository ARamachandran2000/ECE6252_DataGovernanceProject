#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accent experiment comparison figures")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.outputs_dir / "summary_metrics.csv"
    accent_path = args.outputs_dir / "accent_comparison_metrics.csv"

    if not summary_path.exists() or not accent_path.exists():
        raise FileNotFoundError("Expected summary_metrics.csv and accent_comparison_metrics.csv in outputs dir.")

    summary = pd.read_csv(summary_path)
    accent = pd.read_csv(accent_path)

    # 1) Per-accent WER grouped bars.
    wer_cols = [c for c in accent.columns if c.endswith("_wer")]
    wer_long = accent.melt(id_vars=["accent"], value_vars=wer_cols, var_name="condition", value_name="wer")
    wer_long["condition"] = wer_long["condition"].str.replace("_wer", "", regex=False)

    pivot = wer_long.pivot(index="accent", columns="condition", values="wer")
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Per-accent WER across conditions")
    ax.set_xlabel("Accent")
    ax.set_ylabel("WER")
    ax.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(args.plots_dir / "per_accent_wer_grouped.png", dpi=160)
    plt.close()

    # 2) Macro vs overall trade-off.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(summary["overall_wer"], summary["macro_wer"], s=90)
    for _, row in summary.iterrows():
        ax.annotate(row["condition"], (row["overall_wer"], row["macro_wer"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Overall WER")
    ax.set_ylabel("Macro WER")
    ax.set_title("Macro vs Overall WER trade-off")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plots_dir / "macro_vs_overall_tradeoff.png", dpi=160)
    plt.close()

    # 3) Worst-group WER/CER.
    worst = summary[["condition", "worst_group_wer", "worst_group_cer"]].set_index("condition")
    ax = worst.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Worst-group error by condition")
    ax.set_ylabel("Error")
    plt.tight_layout()
    plt.savefig(args.plots_dir / "worst_group_error.png", dpi=160)
    plt.close()

    # 4) Majority-minority gap.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["condition"], summary["majority_minority_gap"])
    ax.set_title("Majority-minority WER gap")
    ax.set_ylabel("Gap (majority WER - minority WER)")
    ax.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(args.plots_dir / "majority_minority_gap.png", dpi=160)
    plt.close()

    print(f"Saved plots to: {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
