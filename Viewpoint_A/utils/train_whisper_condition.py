#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accent_experiment.metrics import cer, wer

TARGET_SR = 16_000


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def _prepare_dataset(df: pd.DataFrame, processor: WhisperProcessor) -> Dataset:
    ds = Dataset.from_pandas(df[["sample_id", "audio_path", "text", "accent"]], preserve_index=False)

    def preprocess(batch: dict) -> dict:
        audio = _load_audio(batch["audio_path"])
        features = processor.feature_extractor(audio, sampling_rate=TARGET_SR).input_features[0]
        labels = processor.tokenizer(batch["text"]).input_ids
        return {
            "input_features": features,
            "labels": labels,
            "sample_id": batch["sample_id"],
            "accent": batch["accent"],
        }

    return ds.map(preprocess, remove_columns=ds.column_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a single manifest condition")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/whisper-base")
    parser.add_argument("--seed", type=int, default=17)

    # Keep these fixed across all runs for controlled comparison.
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    train_df = pd.read_csv(args.train_manifest, sep="\t")
    dev_df = pd.read_csv(args.dev_manifest, sep="\t")

    if args.max_train_samples > 0:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if args.max_dev_samples > 0:
        dev_df = dev_df.sample(n=min(args.max_dev_samples, len(dev_df)), random_state=args.seed)

    processor = WhisperProcessor.from_pretrained(args.model_id, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.suppress_tokens = []

    train_ds = _prepare_dataset(train_df, processor)
    dev_ds = _prepare_dataset(dev_df, processor)

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wers = [wer(r.strip(), h.strip()) for r, h in zip(label_text, pred_text)]
        cers = [cer(r.strip(), h.strip()) for r, h in zip(label_text, pred_text)]
        return {"wer": float(np.mean(wers)), "cer": float(np.mean(cers))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "train_manifest": str(args.train_manifest),
        "dev_manifest": str(args.dev_manifest),
        "model_id": args.model_id,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    train_kwargs = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        fp16=args.fp16,
        dataloader_num_workers=2,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=args.seed,
    )
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        train_kwargs["evaluation_strategy"] = "epoch"
    else:
        train_kwargs["eval_strategy"] = "epoch"

    train_args = Seq2SeqTrainingArguments(**train_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    metrics = trainer.evaluate()
    with (args.output_dir / "dev_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model and metrics to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
