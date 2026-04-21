from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .augmentation import apply_augmentation, load_audio, save_audio
from .common import (
    choose_accents,
    filter_by_accents,
    load_all_splits,
    remove_speaker_leakage,
    summarize_counts,
    write_manifest,
)


@dataclass
class BuildConfig:
    cv_root: Path
    manifests_dir: Path
    augmented_dir: Path
    random_seed: int = 17
    total_accents: int = 6
    majority_n: int = 2
    min_dev_samples: int = 40
    min_test_samples: int = 40
    balanced_oversample_cap_factor: float = 2.0
    synthetic_target_ratio: float = 1.0
    hybrid_target_ratio: float = 0.6
    max_aug_per_accent: int = 2000
    approx_train_rows: int = 0
    approx_dev_rows: int = 0
    approx_test_rows: int = 0


def _sample_rows(df: pd.DataFrame, n: int, seed: int, replace: bool = False) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if n >= len(df) and not replace:
        return df.copy()
    return df.sample(n=n, replace=replace, random_state=seed)


def _build_balanced_real(train_df: pd.DataFrame, cfg: BuildConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train_df["accent"].value_counts().to_dict()
    values = list(counts.values())
    if not values:
        return train_df.iloc[0:0].copy(), pd.DataFrame(columns=["accent", "real_count", "target_count", "strategy"])

    target = int(np.median(values))
    rows = []
    summary_rows = []

    for accent, group in train_df.groupby("accent"):
        current = len(group)
        if current > target:
            sampled = _sample_rows(group, target, seed=cfg.random_seed + abs(hash(accent)) % 10000, replace=False)
            strategy = "undersample"
            out_n = target
        elif current < target:
            max_cap = int(current * cfg.balanced_oversample_cap_factor)
            out_n = min(target, max_cap)
            if out_n <= current:
                sampled = group.copy()
                strategy = "keep"
                out_n = current
            else:
                extra = _sample_rows(
                    group,
                    out_n - current,
                    seed=cfg.random_seed + abs(hash(accent)) % 10000,
                    replace=True,
                ).copy()
                extra["sample_id"] = [f"{sid}__dup{i}" for i, sid in enumerate(extra["sample_id"].tolist())]
                sampled = pd.concat([group, extra], ignore_index=True)
                strategy = "oversample_repeat"
        else:
            sampled = group.copy()
            strategy = "keep"
            out_n = current

        rows.append(sampled)
        summary_rows.append(
            {
                "accent": accent,
                "real_count": current,
                "target_count": out_n,
                "strategy": strategy,
            }
        )

    balanced = pd.concat(rows, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("target_count", ascending=False)
    return balanced, summary


def _majority_accents(train_df: pd.DataFrame, n: int) -> list[str]:
    return train_df["accent"].value_counts().head(n).index.tolist()


def _augment_needed(
    base_df: pd.DataFrame,
    target_ratio: float,
    majority_accents: list[str],
    max_aug_per_accent: int,
) -> dict[str, int]:
    counts = base_df["accent"].value_counts().to_dict()
    majority_count = max([counts.get(a, 0) for a in majority_accents], default=0)
    target = int(round(majority_count * target_ratio))

    needed = {}
    for accent, count in counts.items():
        if accent in majority_accents:
            continue
        needed[accent] = min(max_aug_per_accent, max(0, target - count))
    return needed


def _build_augmented_rows(
    base_df: pd.DataFrame,
    condition_name: str,
    needed_by_accent: dict[str, int],
    cfg: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    aug_types = [
        "speed_0.90",
        "speed_1.10",
        "noise_snr24",
        "volume_down_3db",
        "volume_up_3db",
        "reverb_light",
        "time_mask_8pct",
    ]

    generated_rows: list[dict] = []
    meta_rows: list[dict] = []

    for accent, needed in needed_by_accent.items():
        if needed <= 0:
            continue
        pool = base_df[base_df["accent"] == accent].copy().reset_index(drop=True)
        if pool.empty:
            continue

        out_dir = cfg.augmented_dir / condition_name / accent.replace("/", "-").replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(needed):
            source = pool.iloc[i % len(pool)]
            source_path = Path(source["audio_path"])
            if not source_path.exists():
                continue

            aug_type = aug_types[i % len(aug_types)]
            waveform, sr = load_audio(source_path)
            aug_wave = apply_augmentation(waveform, sr, aug_type).clamp(-1.0, 1.0)

            src_id = str(source["sample_id"])
            new_id = f"{src_id}__{condition_name}__aug{i:05d}"
            out_audio = out_dir / f"{new_id}.wav"
            save_audio(out_audio, aug_wave, sr)
            duration_s = float(aug_wave.shape[1]) / float(sr)

            row = source.to_dict()
            row["sample_id"] = new_id
            row["audio_path"] = str(out_audio.resolve())
            row["path"] = out_audio.name
            row["duration_s"] = duration_s
            row["is_synthetic"] = 1
            row["source_sample_id"] = src_id
            row["augmentation_type"] = aug_type
            row["condition"] = condition_name
            generated_rows.append(row)

            meta_rows.append(
                {
                    "condition": condition_name,
                    "source_sample_id": src_id,
                    "generated_sample_id": new_id,
                    "source_accent": accent,
                    "augmentation_type": aug_type,
                    "transcript": str(source["text"]),
                    "audio_path": str(out_audio.resolve()),
                }
            )

    generated_df = pd.DataFrame(generated_rows)
    meta_df = pd.DataFrame(meta_rows)
    return generated_df, meta_df


def _mark_real(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = df.copy()
    out["is_synthetic"] = 0
    out["source_sample_id"] = ""
    out["augmentation_type"] = ""
    out["condition"] = condition
    return out


def build_all_manifests(cfg: BuildConfig) -> dict[str, Path]:
    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
    cfg.augmented_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_all_splits(
        cfg.cv_root,
        nrows_train=cfg.approx_train_rows if cfg.approx_train_rows > 0 else None,
        nrows_dev=cfg.approx_dev_rows if cfg.approx_dev_rows > 0 else None,
        nrows_test=cfg.approx_test_rows if cfg.approx_test_rows > 0 else None,
    )

    train_df = all_df[all_df["split"] == "train"].copy()
    dev_df = all_df[all_df["split"] == "dev"].copy()
    test_df = all_df[all_df["split"] == "test"].copy()

    selected_accents = choose_accents(
        train_df,
        dev_df,
        test_df,
        majority_n=cfg.majority_n,
        total_accents=cfg.total_accents,
        min_dev_samples=cfg.min_dev_samples,
        min_test_samples=cfg.min_test_samples,
    )

    train_df = filter_by_accents(train_df, selected_accents)
    dev_df = filter_by_accents(dev_df, selected_accents)
    test_df = filter_by_accents(test_df, selected_accents)

    fixed_train, fixed_dev, fixed_test, leakage = remove_speaker_leakage(train_df, dev_df, test_df)

    fixed_train = _mark_real(fixed_train, condition="fixed_real")
    fixed_dev = _mark_real(fixed_dev, condition="fixed_real")
    fixed_test = _mark_real(fixed_test, condition="fixed_real")

    # Baseline
    train_baseline = _mark_real(fixed_train, condition="baseline")

    # Balanced real
    train_balanced_real, balanced_summary = _build_balanced_real(fixed_train, cfg)
    train_balanced_real = _mark_real(train_balanced_real, condition="balanced_real")

    majority_accents = _majority_accents(fixed_train, cfg.majority_n)

    # Synthetic / augmented: baseline + synthetic top-up for minorities.
    needed_synth = _augment_needed(
        base_df=fixed_train,
        target_ratio=cfg.synthetic_target_ratio,
        majority_accents=majority_accents,
        max_aug_per_accent=cfg.max_aug_per_accent,
    )
    synth_gen, synth_meta = _build_augmented_rows(
        base_df=fixed_train,
        condition_name="synthetic_augmented",
        needed_by_accent=needed_synth,
        cfg=cfg,
    )
    train_synthetic_augmented = pd.concat(
        [
            _mark_real(fixed_train, condition="synthetic_augmented"),
            synth_gen,
        ],
        ignore_index=True,
    )

    # Hybrid: use balanced real then softer synthetic top-up.
    needed_hybrid = _augment_needed(
        base_df=train_balanced_real,
        target_ratio=cfg.hybrid_target_ratio,
        majority_accents=majority_accents,
        max_aug_per_accent=cfg.max_aug_per_accent,
    )
    hybrid_gen, hybrid_meta = _build_augmented_rows(
        base_df=train_balanced_real,
        condition_name="hybrid",
        needed_by_accent=needed_hybrid,
        cfg=cfg,
    )
    train_hybrid = pd.concat(
        [
            _mark_real(train_balanced_real, condition="hybrid"),
            hybrid_gen,
        ],
        ignore_index=True,
    )

    # Save fixed manifests.
    fixed_train_path = cfg.manifests_dir / "fixed_train_real.tsv"
    fixed_dev_path = cfg.manifests_dir / "fixed_dev_real.tsv"
    fixed_test_path = cfg.manifests_dir / "fixed_test_real.tsv"
    write_manifest(fixed_train, fixed_train_path)
    write_manifest(fixed_dev, fixed_dev_path)
    write_manifest(fixed_test, fixed_test_path)

    # Save condition manifests.
    baseline_path = cfg.manifests_dir / "train_baseline.tsv"
    balanced_path = cfg.manifests_dir / "train_balanced_real.tsv"
    synth_path = cfg.manifests_dir / "train_synthetic_augmented.tsv"
    hybrid_path = cfg.manifests_dir / "train_hybrid.tsv"

    write_manifest(train_baseline, baseline_path)
    write_manifest(train_balanced_real, balanced_path)
    write_manifest(train_synthetic_augmented, synth_path)
    write_manifest(train_hybrid, hybrid_path)

    # Save transparency artifacts.
    summarize_counts(pd.concat([fixed_train, fixed_dev, fixed_test], ignore_index=True)).to_csv(
        cfg.manifests_dir / "selected_accent_split_summary.csv", index=False
    )
    balanced_summary.to_csv(cfg.manifests_dir / "balanced_real_accent_summary.csv", index=False)
    synth_meta.to_csv(cfg.manifests_dir / "synthetic_augmented_metadata.csv", index=False)
    hybrid_meta.to_csv(cfg.manifests_dir / "hybrid_augmented_metadata.csv", index=False)

    # Log leakage and setup metadata.
    setup = pd.DataFrame(
        [
            {
                "random_seed": cfg.random_seed,
                "selected_accents": "|".join(selected_accents),
                "majority_accents": "|".join(majority_accents),
                "dev_removed_for_train_overlap": leakage["dev_removed_for_train_overlap"],
                "test_removed_for_train_overlap": leakage["test_removed_for_train_overlap"],
                "synthetic_target_ratio": cfg.synthetic_target_ratio,
                "hybrid_target_ratio": cfg.hybrid_target_ratio,
                "max_aug_per_accent": cfg.max_aug_per_accent,
            }
        ]
    )
    setup.to_csv(cfg.manifests_dir / "experiment_setup.csv", index=False)

    return {
        "fixed_train_real": fixed_train_path,
        "fixed_dev_real": fixed_dev_path,
        "fixed_test_real": fixed_test_path,
        "train_baseline": baseline_path,
        "train_balanced_real": balanced_path,
        "train_synthetic_augmented": synth_path,
        "train_hybrid": hybrid_path,
    }
