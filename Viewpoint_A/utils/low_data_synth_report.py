#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write short summary for low-data replacement experiment")
    p.add_argument("--metrics-csv", type=Path, default=Path("outputs_low_data") / "low_data_metrics.csv")
    p.add_argument("--ratio-csv", type=Path, default=Path("outputs_low_data") / "low_data_replacement_ratio.csv")
    p.add_argument("--report-path", type=Path, default=Path("reports") / "low_data_replacement_summary.md")
    p.add_argument("--plots-dir", type=Path, default=Path("plots_low_data"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = pd.read_csv(args.metrics_csv)
    ratio = pd.read_csv(args.ratio_csv) if args.ratio_csv.exists() else pd.DataFrame()

    lines = []
    lines.append("# Low-Data Synthetic Replacement Summary")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(metrics.to_markdown(index=False))

    if not ratio.empty:
        lines.append("")
        lines.append("## Replacement Ratio")
        lines.append("")
        lines.append(ratio.to_markdown(index=False))

    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"- `{args.plots_dir / 'target_accent_wer_curve.png'}`")
    lines.append(f"- `{args.plots_dir / 'macro_wer_curve.png'}`")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines))
    print(f"Wrote report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
