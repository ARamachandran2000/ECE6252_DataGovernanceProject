#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from accent_experiment import BuildConfig, build_all_manifests


def _count_tsv_rows(tsv_path: Path) -> int:
    # Subtract header row; never return negative.
    with tsv_path.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _pct_to_nrows(total_rows: int, pct: float) -> int:
    if pct <= 0:
        return 0
    if pct > 100:
        raise ValueError(f"Percentage must be <= 100, got {pct}")
    if total_rows <= 0:
        return 0
    return max(1, int(round(total_rows * (pct / 100.0))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed and condition-specific manifests for accent experiment")
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=PROJECT_ROOT / "cv-corpus-25.0-2026-03-09",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=PROJECT_ROOT / "manifests",
    )
    parser.add_argument(
        "--augmented-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "augmented",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--total-accents", type=int, default=6)
    parser.add_argument("--majority-n", type=int, default=2)
    parser.add_argument("--min-dev-samples", type=int, default=40)
    parser.add_argument("--min-test-samples", type=int, default=40)
    parser.add_argument("--balanced-oversample-cap-factor", type=float, default=2.0)
    parser.add_argument("--synthetic-target-ratio", type=float, default=1.0)
    parser.add_argument("--hybrid-target-ratio", type=float, default=0.6)
    parser.add_argument("--max-aug-per-accent", type=int, default=2000)
    parser.add_argument("--approx-train-rows", type=int, default=0, help="Use only first N train rows (fast approximate mode)")
    parser.add_argument("--approx-dev-rows", type=int, default=0, help="Use only first N dev rows (fast approximate mode)")
    parser.add_argument("--approx-test-rows", type=int, default=0, help="Use only first N test rows (fast approximate mode)")
    parser.add_argument(
        "--approx-train-pct",
        type=float,
        default=0.0,
        help="Use only the first X percent of original train split rows (e.g., 10 or 15). Ignored if --approx-train-rows is set.",
    )
    parser.add_argument(
        "--approx-dev-pct",
        type=float,
        default=0.0,
        help="Use only the first X percent of original dev split rows. Ignored if --approx-dev-rows is set.",
    )
    parser.add_argument(
        "--approx-test-pct",
        type=float,
        default=0.0,
        help="Use only the first X percent of original test split rows. Ignored if --approx-test-rows is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    approx_train_rows = args.approx_train_rows
    approx_dev_rows = args.approx_dev_rows
    approx_test_rows = args.approx_test_rows

    # Convert percentages to nrows when nrows were not explicitly provided.
    en_root = args.cv_root / "en"
    if approx_train_rows <= 0 and args.approx_train_pct > 0:
        approx_train_rows = _pct_to_nrows(_count_tsv_rows(en_root / "train.tsv"), args.approx_train_pct)
    if approx_dev_rows <= 0 and args.approx_dev_pct > 0:
        approx_dev_rows = _pct_to_nrows(_count_tsv_rows(en_root / "dev.tsv"), args.approx_dev_pct)
    if approx_test_rows <= 0 and args.approx_test_pct > 0:
        approx_test_rows = _pct_to_nrows(_count_tsv_rows(en_root / "test.tsv"), args.approx_test_pct)

    cfg = BuildConfig(
        cv_root=args.cv_root,
        manifests_dir=args.manifests_dir,
        augmented_dir=args.augmented_dir,
        random_seed=args.seed,
        total_accents=args.total_accents,
        majority_n=args.majority_n,
        min_dev_samples=args.min_dev_samples,
        min_test_samples=args.min_test_samples,
        balanced_oversample_cap_factor=args.balanced_oversample_cap_factor,
        synthetic_target_ratio=args.synthetic_target_ratio,
        hybrid_target_ratio=args.hybrid_target_ratio,
        max_aug_per_accent=args.max_aug_per_accent,
        approx_train_rows=approx_train_rows,
        approx_dev_rows=approx_dev_rows,
        approx_test_rows=approx_test_rows,
    )

    outputs = build_all_manifests(cfg)
    if approx_train_rows > 0 or approx_dev_rows > 0 or approx_test_rows > 0:
        print(
            "Approximate mode enabled:",
            f"train={approx_train_rows or 'all'},",
            f"dev={approx_dev_rows or 'all'},",
            f"test={approx_test_rows or 'all'}",
        )
    print("Built manifests:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"Transparency summary: {args.manifests_dir / 'selected_accent_split_summary.csv'}")
    print(f"Balanced-real summary: {args.manifests_dir / 'balanced_real_accent_summary.csv'}")
    print(f"Synthetic metadata: {args.manifests_dir / 'synthetic_augmented_metadata.csv'}")
    print(f"Hybrid metadata: {args.manifests_dir / 'hybrid_augmented_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
