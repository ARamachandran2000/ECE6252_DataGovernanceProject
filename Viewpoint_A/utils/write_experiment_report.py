#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write markdown summary for accent experiment")
    parser.add_argument("--manifests-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--report-path", type=Path, default=Path("reports") / "accent_experiment_report.md")
    return parser.parse_args()


def _table_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> int:
    args = parse_args()

    setup = pd.read_csv(args.manifests_dir / "experiment_setup.csv")
    split_summary = pd.read_csv(args.manifests_dir / "selected_accent_split_summary.csv")
    balanced_summary = pd.read_csv(args.manifests_dir / "balanced_real_accent_summary.csv")

    summary_metrics = pd.read_csv(args.outputs_dir / "summary_metrics.csv") if (args.outputs_dir / "summary_metrics.csv").exists() else pd.DataFrame()
    accent_metrics = pd.read_csv(args.outputs_dir / "accent_comparison_metrics.csv") if (args.outputs_dir / "accent_comparison_metrics.csv").exists() else pd.DataFrame()

    lines = []
    lines.append("# Whisper Accent Experiment Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Dataset: Common Voice English (cv-corpus-25.0-2026-03-09)")
    lines.append("- Model: openai/whisper-base")
    lines.append("- Conditions: Baseline, Balanced-real, Synthetic/Augmented, Hybrid")
    lines.append("")
    lines.append(_table_md(setup))

    lines.append("")
    lines.append("## Split Counts (Selected Accents)")
    lines.append("")
    lines.append(_table_md(split_summary))

    lines.append("")
    lines.append("## Balanced-real Policy")
    lines.append("")
    lines.append("- This condition uses only authentic Common Voice data.")
    lines.append("- Balancing strategy is logged per accent below.")
    lines.append("")
    lines.append(_table_md(balanced_summary))

    if not summary_metrics.empty:
        lines.append("")
        lines.append("## Condition Comparison")
        lines.append("")
        lines.append(_table_md(summary_metrics))

    if not accent_metrics.empty:
        lines.append("")
        lines.append("## Accent-level WER/CER Comparison")
        lines.append("")
        lines.append(_table_md(accent_metrics))

    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"- Per-accent WER: `{args.plots_dir / 'per_accent_wer_grouped.png'}`")
    lines.append(f"- Macro vs overall: `{args.plots_dir / 'macro_vs_overall_tradeoff.png'}`")
    lines.append(f"- Worst-group error: `{args.plots_dir / 'worst_group_error.png'}`")
    lines.append(f"- Majority-minority gap: `{args.plots_dir / 'majority_minority_gap.png'}`")

    lines.append("")
    lines.append("## Conclusion Template")
    lines.append("")
    lines.append("- Check whether Synthetic/Augmented and Hybrid reduce minority-accent WER vs Baseline.")
    lines.append("- Compare how close they come to Balanced-real while preserving overall performance.")
    lines.append("- Emphasize scalability and privacy benefits without claiming synthetic data universally outperforms real data.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines))
    print(f"Wrote report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
