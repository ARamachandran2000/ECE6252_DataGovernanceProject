#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accent_experiment.augmentation import apply_augmentation, load_audio, save_audio
from accent_experiment.common import write_manifest


AUG_TYPES = [
    "speed_0.90",
    "speed_1.10",
    "noise_snr24",
    "volume_down_3db",
    "volume_up_3db",
    "reverb_light",
    "time_mask_8pct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build low-data real-only and real+synthetic manifests by accent")
    p.add_argument("--base-manifests-dir", type=Path, default=PROJECT_ROOT / "manifests_subset")
    p.add_argument("--output-manifests-dir", type=Path, default=PROJECT_ROOT / "manifests_low_data")
    p.add_argument("--augmented-dir", type=Path, default=PROJECT_ROOT / "datasets" / "augmented_low_data")
    p.add_argument(
        "--target-accents",
        type=str,
        default="India and South Asia (India, Pakistan, Sri Lanka)|Southern African (South Africa, Zimbabwe, Namibia)",
        help="Pipe-separated accent labels",
    )
    p.add_argument("--levels", type=str, default="5,10,20,50,100")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-synth-per-accent", type=int, default=300)
    return p.parse_args()


def sample_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = max(1, min(n, len(df)))
    return df.sample(n=n, random_state=seed)


def make_real_subset(
    train_df: pd.DataFrame,
    target_accents: list[str],
    pct: int,
    seed: int,
) -> pd.DataFrame:
    out = []
    for accent, g in train_df.groupby("accent"):
        if accent in target_accents:
            keep_n = int(round(len(g) * pct / 100.0))
            out.append(sample_group(g, keep_n, seed + abs(hash((accent, pct))) % 100000))
        else:
            out.append(g)
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_synth_topup(
    real_subset: pd.DataFrame,
    full_train: pd.DataFrame,
    target_accents: list[str],
    pct: int,
    augmented_dir: Path,
    max_synth_per_accent: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for accent in target_accents:
        full_g = full_train[full_train["accent"] == accent]
        cur_g = real_subset[real_subset["accent"] == accent]
        if full_g.empty or cur_g.empty:
            continue

        need = max(0, len(full_g) - len(cur_g))
        need = min(need, max_synth_per_accent)
        if need == 0:
            continue

        out_dir = augmented_dir / f"pct_{pct}" / accent.replace("/", "-").replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        cur_rows = cur_g.reset_index(drop=True)
        for i in range(need):
            src = cur_rows.iloc[i % len(cur_rows)]
            src_path = Path(str(src["audio_path"]))
            if not src_path.exists():
                continue
            aug_type = AUG_TYPES[i % len(AUG_TYPES)]
            wav, sr = load_audio(src_path)
            aug_wav = apply_augmentation(wav, sr, aug_type).clamp(-1.0, 1.0)

            src_id = str(src["sample_id"])
            sid = f"{src_id}__lowdata_p{pct}__aug{i:05d}"
            out_wav = out_dir / f"{sid}.wav"
            save_audio(out_wav, aug_wav, sr)

            r = src.to_dict()
            r["sample_id"] = sid
            r["audio_path"] = str(out_wav.resolve())
            r["path"] = out_wav.name
            r["duration_s"] = float(aug_wav.shape[1]) / float(sr)
            r["is_synthetic"] = 1
            r["source_sample_id"] = src_id
            r["augmentation_type"] = aug_type
            rows.append(r)

    if not rows:
        return pd.DataFrame(columns=real_subset.columns)
    return pd.DataFrame(rows)


def ensure_manifest_cols(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = df.copy()
    if "is_synthetic" not in out.columns:
        out["is_synthetic"] = 0
    if "source_sample_id" not in out.columns:
        out["source_sample_id"] = ""
    if "augmentation_type" not in out.columns:
        out["augmentation_type"] = ""
    out["condition"] = condition
    return out


def main() -> int:
    args = parse_args()
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    target_accents = [x.strip() for x in args.target_accents.split("|") if x.strip()]

    out_dir = args.output_manifests_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.augmented_dir.mkdir(parents=True, exist_ok=True)

    fixed_train = pd.read_csv(args.base_manifests_dir / "fixed_train_real.tsv", sep="\t")
    fixed_dev = pd.read_csv(args.base_manifests_dir / "fixed_dev_real.tsv", sep="\t")
    fixed_test = pd.read_csv(args.base_manifests_dir / "fixed_test_real.tsv", sep="\t")

    write_manifest(ensure_manifest_cols(fixed_dev, "low_data_fixed"), out_dir / "fixed_dev_real.tsv")
    write_manifest(ensure_manifest_cols(fixed_test, "low_data_fixed"), out_dir / "fixed_test_real.tsv")

    summary_rows = []

    for p in levels:
        real_subset = make_real_subset(fixed_train, target_accents, p, args.seed)
        real_subset = ensure_manifest_cols(real_subset, f"low_data_real_{p}")

        synth_only = generate_synth_topup(
            real_subset=real_subset,
            full_train=fixed_train,
            target_accents=target_accents,
            pct=p,
            augmented_dir=args.augmented_dir,
            max_synth_per_accent=args.max_synth_per_accent,
            seed=args.seed,
        )
        synth_only = ensure_manifest_cols(synth_only, f"low_data_synth_{p}")

        real_plus = pd.concat([real_subset, synth_only], ignore_index=True)
        real_plus = ensure_manifest_cols(real_plus, f"low_data_real_plus_synth_{p}")

        # Step-3 files requested.
        write_manifest(real_subset, out_dir / f"train_real_{p}.tsv")
        # Step-4 files requested (synthetic-only per level).
        write_manifest(synth_only, out_dir / f"train_aug_{p}.tsv")
        # Step-5 files requested.
        write_manifest(real_subset, out_dir / f"train_real_only_{p}.tsv")
        write_manifest(real_plus, out_dir / f"train_real_plus_synth_{p}.tsv")

        for variant, df in [("real_only", real_subset), ("real_plus_synth", real_plus), ("synth_only", synth_only)]:
            counts = df["accent"].value_counts().to_dict()
            summary_rows.append(
                {
                    "level_pct": p,
                    "variant": variant,
                    "rows": len(df),
                    "target_counts": "|".join([f"{a}:{counts.get(a, 0)}" for a in target_accents]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "low_data_manifest_summary.csv", index=False)

    setup = {
        "target_accents": "|".join(target_accents),
        "levels": ",".join(map(str, levels)),
        "seed": args.seed,
        "max_synth_per_accent": args.max_synth_per_accent,
    }
    pd.DataFrame([setup]).to_csv(out_dir / "low_data_setup.csv", index=False)

    print(f"Wrote low-data manifests to: {out_dir}")
    print(f"Summary: {out_dir / 'low_data_manifest_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
