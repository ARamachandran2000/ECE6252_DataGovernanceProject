#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run low-data real-only vs real+synth replacement experiment")
    p.add_argument("--python", default="python")
    p.add_argument("--manifests-dir", type=Path, default=PROJECT_ROOT / "manifests_low_data")
    p.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs_low_data")
    p.add_argument("--levels", type=str, default="5,10,20,50,100")
    p.add_argument(
        "--target-accents",
        type=str,
        default="India and South Asia (India, Pakistan, Sri Lanka)|Southern African (South Africa, Zimbabwe, Namibia)",
    )
    p.add_argument("--model-id", default="openai/whisper-base")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-train-samples", type=int, default=256)
    p.add_argument("--max-dev-samples", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def target_wer(accent_csv: Path, target_accents: list[str]) -> float:
    df = pd.read_csv(accent_csv)
    sub = df[df["accent"].isin(target_accents)]
    if sub.empty:
        return float("nan")
    return float(sub["wer"].mean())


def main() -> int:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    target_accents = [x.strip() for x in args.target_accents.split("|") if x.strip()]

    dev_manifest = args.manifests_dir / "fixed_dev_real.tsv"
    test_manifest = args.manifests_dir / "fixed_test_real.tsv"

    rows = []

    for p in levels:
        for variant in ["real_only", "real_plus_synth"]:
            train_manifest = args.manifests_dir / f"train_{variant}_{p}.tsv"
            condition = f"{variant}_{p}"
            model_dir = args.outputs_dir / f"whisper_{condition}"

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
                    "",
                ]
                run_cmd(eval_cmd)

            metrics_json = args.outputs_dir / f"overall_metrics_{condition}.json"
            accent_csv = args.outputs_dir / f"accent_metrics_{condition}.csv"
            if metrics_json.exists() and accent_csv.exists():
                overall = json.loads(metrics_json.read_text())
                row = {
                    "level_pct": p,
                    "variant": variant,
                    "condition": condition,
                    "overall_wer": overall["overall_wer"],
                    "overall_cer": overall["overall_cer"],
                    "macro_wer": overall["macro_wer"],
                    "macro_cer": overall["macro_cer"],
                    "target_accent_wer": target_wer(accent_csv, target_accents),
                }
                rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(["level_pct", "variant"])
    metrics_df.to_csv(args.outputs_dir / "low_data_metrics.csv", index=False)

    # replacement ratio against real_only@100 (target accent WER)
    base = metrics_df[(metrics_df["variant"] == "real_only") & (metrics_df["level_pct"] == 100)]
    if not base.empty:
        base_val = float(base.iloc[0]["target_accent_wer"])
        rep_rows = []
        for _, r in metrics_df.iterrows():
            if pd.isna(r["target_accent_wer"]) or base_val <= 0:
                ratio = float("nan")
            else:
                ratio = base_val / float(r["target_accent_wer"])
            rep_rows.append({
                "condition": r["condition"],
                "level_pct": r["level_pct"],
                "variant": r["variant"],
                "target_accent_wer": r["target_accent_wer"],
                "relative_to_real100": ratio,
            })
        pd.DataFrame(rep_rows).to_csv(args.outputs_dir / "low_data_replacement_ratio.csv", index=False)

    print(f"Saved metrics: {args.outputs_dir / 'low_data_metrics.csv'}")
    print(f"Saved replacement ratio: {args.outputs_dir / 'low_data_replacement_ratio.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
