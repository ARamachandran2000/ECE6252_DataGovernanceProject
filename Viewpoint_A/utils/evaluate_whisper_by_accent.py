#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accent_experiment.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on fixed real test set by accent")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--majority-accents", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.test_manifest, sep="\t")

    majority_accents = [a.strip() for a in args.majority_accents.split("|") if a.strip()]

    has_cuda = torch.cuda.is_available()
    asr = pipeline(
        task="automatic-speech-recognition",
        model=str(args.model_dir),
        tokenizer=str(args.model_dir),
        feature_extractor=str(args.model_dir),
        device=0 if has_cuda else -1,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )

    predictions = []
    for _, row in df.iterrows():
        out = asr(str(row["audio_path"]))
        pred_text = str(out.get("text", "")).strip()
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "accent": row["accent"],
                "reference": str(row["text"]),
                "prediction": pred_text,
            }
        )

    pred_df = pd.DataFrame(predictions)
    by_accent, overall = summarize_metrics(pred_df, majority_accents=majority_accents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_dir / f"predictions_{args.condition}.csv", index=False)
    by_accent.to_csv(args.output_dir / f"accent_metrics_{args.condition}.csv", index=False)

    overall_dict = {
        "condition": args.condition,
        "overall_wer": overall.overall_wer,
        "overall_cer": overall.overall_cer,
        "macro_wer": overall.macro_wer,
        "macro_cer": overall.macro_cer,
        "worst_group_wer": overall.worst_group_wer,
        "worst_group_cer": overall.worst_group_cer,
        "majority_minority_gap": overall.majority_minority_gap_wer,
    }
    with (args.output_dir / f"overall_metrics_{args.condition}.json").open("w") as f:
        json.dump(overall_dict, f, indent=2)

    print(json.dumps(overall_dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
