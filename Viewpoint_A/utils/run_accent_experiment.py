#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONDITIONS = [
    ("baseline", "train_baseline.tsv", "whisper_baseline"),
    ("balanced_real", "train_balanced_real.tsv", "whisper_balanced_real"),
    ("synthetic_augmented", "train_synthetic_augmented.tsv", "whisper_synthetic_augmented"),
    ("hybrid", "train_hybrid.tsv", "whisper_hybrid"),
]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4-condition Whisper accent experiment")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifests-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-id", default="openai/whisper-base")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    setup_csv = args.manifests_dir / "experiment_setup.csv"
    if not setup_csv.exists():
        raise FileNotFoundError(f"Missing setup file: {setup_csv}. Run build_experiment_manifests.py first.")
    setup = pd.read_csv(setup_csv).iloc[0]
    majority_accents = str(setup["majority_accents"])

    dev_manifest = args.manifests_dir / "fixed_dev_real.tsv"
    test_manifest = args.manifests_dir / "fixed_test_real.tsv"

    for condition, train_file, out_dir_name in CONDITIONS:
        train_manifest = args.manifests_dir / train_file
        model_dir = args.outputs_dir / out_dir_name

        if not args.skip_train:
            train_cmd = [
                args.python,
                str(PROJECT_ROOT / "utils" / "train_whisper_condition.py"),
                "--train-manifest",
                str(train_manifest),
                "--dev-manifest",
                str(dev_manifest),
                "--output-dir",
                str(model_dir),
                "--model-id",
                args.model_id,
                "--seed",
                str(args.seed),
                "--learning-rate",
                str(args.learning_rate),
                "--epochs",
                str(args.epochs),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--warmup-steps",
                str(args.warmup_steps),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-dev-samples",
                str(args.max_dev_samples),
            ]
            if args.fp16:
                train_cmd.append("--fp16")
            run_cmd(train_cmd)

        if not args.skip_eval:
            eval_cmd = [
                args.python,
                str(PROJECT_ROOT / "utils" / "evaluate_whisper_by_accent.py"),
                "--model-dir",
                str(model_dir),
                "--test-manifest",
                str(test_manifest),
                "--output-dir",
                str(args.outputs_dir),
                "--condition",
                condition,
                "--majority-accents",
                majority_accents,
            ]
            run_cmd(eval_cmd)

    # Build aggregate comparison tables.
    rows = []
    accent_tables = []
    for condition, _, _ in CONDITIONS:
        metrics_json = args.outputs_dir / f"overall_metrics_{condition}.json"
        accent_csv = args.outputs_dir / f"accent_metrics_{condition}.csv"
        if metrics_json.exists():
            rows.append(json.loads(metrics_json.read_text()))
        if accent_csv.exists():
            table = pd.read_csv(accent_csv)
            table = table[["accent", "wer", "cer"]].rename(columns={"wer": f"{condition}_wer", "cer": f"{condition}_cer"})
            accent_tables.append(table)

    if rows:
        pd.DataFrame(rows).to_csv(args.outputs_dir / "summary_metrics.csv", index=False)

    if accent_tables:
        merged = accent_tables[0]
        for t in accent_tables[1:]:
            merged = merged.merge(t, on="accent", how="outer")
        merged.to_csv(args.outputs_dir / "accent_comparison_metrics.csv", index=False)

    print(f"Finished. Outputs in: {args.outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
