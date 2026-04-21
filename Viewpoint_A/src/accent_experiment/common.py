from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


UNKNOWN_ACCENT = "Unknown"


@dataclass(frozen=True)
class CvPaths:
    root: Path

    @property
    def en_dir(self) -> Path:
        return self.root / "en"

    @property
    def clips_dir(self) -> Path:
        return self.en_dir / "clips"

    @property
    def train_tsv(self) -> Path:
        return self.en_dir / "train.tsv"

    @property
    def dev_tsv(self) -> Path:
        return self.en_dir / "dev.tsv"

    @property
    def test_tsv(self) -> Path:
        return self.en_dir / "test.tsv"

    @property
    def clip_durations_tsv(self) -> Path:
        return self.en_dir / "clip_durations.tsv"


def normalize_accent(raw: str | None) -> str:
    if raw is None:
        return UNKNOWN_ACCENT

    accent = str(raw).strip()
    if not accent:
        return UNKNOWN_ACCENT

    # Common Voice can contain multi-label accents separated by '|'.
    first = accent.split("|")[0].strip()
    return first if first else UNKNOWN_ACCENT


def load_split(
    tsv_path: Path,
    split_name: str,
    clips_dir: Path,
    verify_audio: bool = False,
    nrows: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    expected_cols = {"client_id", "path", "sentence", "accents"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {tsv_path}: {sorted(missing)}")

    df = df.copy()
    df["split"] = split_name
    df["accent"] = df["accents"].apply(normalize_accent)
    clips_prefix = str(clips_dir)
    df["audio_path"] = df["path"].apply(lambda p: f"{clips_prefix}/{p}")
    df["sample_id"] = df["path"].str.replace(".mp3", "", regex=False)
    df["text"] = df["sentence"].astype(str)
    if verify_audio:
        df["has_audio"] = df["audio_path"].apply(lambda p: Path(p).exists())
    else:
        df["has_audio"] = True
    return df


def load_duration_map(duration_tsv: Path) -> dict[str, float]:
    df = pd.read_csv(duration_tsv, sep="\t", dtype={"clip": str, "duration[ms]": float})
    if "clip" not in df.columns or "duration[ms]" not in df.columns:
        return {}
    return {row["clip"]: float(row["duration[ms]"]) / 1000.0 for _, row in df.iterrows()}


def attach_duration_seconds(df: pd.DataFrame, duration_map: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["duration_s"] = out["path"].map(duration_map).fillna(0.0)
    return out


def load_all_splits(
    cv_root: Path,
    verify_audio: bool = False,
    nrows_train: int | None = None,
    nrows_dev: int | None = None,
    nrows_test: int | None = None,
) -> pd.DataFrame:
    paths = CvPaths(root=cv_root)
    train_df = load_split(
        paths.train_tsv,
        "train",
        paths.clips_dir,
        verify_audio=verify_audio,
        nrows=nrows_train,
    )
    dev_df = load_split(
        paths.dev_tsv,
        "dev",
        paths.clips_dir,
        verify_audio=verify_audio,
        nrows=nrows_dev,
    )
    test_df = load_split(
        paths.test_tsv,
        "test",
        paths.clips_dir,
        verify_audio=verify_audio,
        nrows=nrows_test,
    )

    all_df = pd.concat([train_df, dev_df, test_df], axis=0, ignore_index=True)
    all_df = attach_duration_seconds(all_df, load_duration_map(paths.clip_durations_tsv))

    # Keep rows with valid audio files and non-empty transcripts.
    all_df = all_df[all_df["has_audio"]].copy()
    all_df = all_df[all_df["text"].str.strip() != ""].copy()
    return all_df


def choose_accents(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    majority_n: int = 2,
    total_accents: int = 6,
    min_dev_samples: int = 40,
    min_test_samples: int = 40,
) -> list[str]:
    train_counts = train_df["accent"].value_counts()
    dev_counts = dev_df["accent"].value_counts()
    test_counts = test_df["accent"].value_counts()

    eligible = [
        accent
        for accent in train_counts.index.tolist()
        if dev_counts.get(accent, 0) >= min_dev_samples
        and test_counts.get(accent, 0) >= min_test_samples
    ]

    if len(eligible) < total_accents:
        total_accents = len(eligible)

    if total_accents == 0:
        raise ValueError("No accents satisfy minimum dev/test sample thresholds.")

    eligible_sorted = sorted(eligible, key=lambda a: train_counts.get(a, 0), reverse=True)
    majority = eligible_sorted[:majority_n]

    remaining = [a for a in eligible_sorted if a not in majority]
    tail_needed = max(0, total_accents - len(majority))
    minority = list(reversed(remaining))[:tail_needed]

    selected = majority + minority
    # Stable order: majority first by descending resources, then minority by ascending resources.
    return selected


def filter_by_accents(df: pd.DataFrame, accents: Iterable[str]) -> pd.DataFrame:
    allowed = set(accents)
    return df[df["accent"].isin(allowed)].copy()


def remove_speaker_leakage(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_speakers = set(train_df["client_id"].astype(str).tolist())

    dev_overlap = dev_df[dev_df["client_id"].isin(train_speakers)]
    test_overlap = test_df[test_df["client_id"].isin(train_speakers)]

    cleaned_dev = dev_df[~dev_df["client_id"].isin(train_speakers)].copy()
    cleaned_test = test_df[~test_df["client_id"].isin(train_speakers)].copy()

    leakage = {
        "dev_removed_for_train_overlap": int(dev_overlap.shape[0]),
        "test_removed_for_train_overlap": int(test_overlap.shape[0]),
    }
    return train_df.copy(), cleaned_dev, cleaned_test, leakage


def to_manifest(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "audio_path",
        "text",
        "accent",
        "client_id",
        "split",
        "duration_s",
        "is_synthetic",
        "source_sample_id",
        "augmentation_type",
        "condition",
    ]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            if col == "is_synthetic":
                out[col] = 0
            else:
                out[col] = ""
    return out[cols].copy()


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_manifest(df).to_csv(path, sep="\t", index=False)


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "accent"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            speakers=("client_id", pd.Series.nunique),
            hours=("duration_s", lambda x: float(x.sum()) / 3600.0),
        )
        .reset_index()
        .sort_values(["split", "samples"], ascending=[True, False])
    )
    return summary
