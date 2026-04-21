#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable

import edge_tts
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_accent(raw: str | None) -> str:
    if raw is None:
        return "Unknown"
    accent = str(raw).strip()
    if not accent:
        return "Unknown"
    return accent.split("|")[0].strip() or "Unknown"


def load_cv_train(cv_root: Path) -> pd.DataFrame:
    tsv = cv_root / "en" / "train.tsv"
    clips_dir = cv_root / "en" / "clips"
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    df["accent"] = df["accents"].apply(normalize_accent)
    df["audio_path"] = df["path"].apply(lambda p: str(clips_dir / p))
    df["text"] = df["sentence"].astype(str)
    df = df[(df["text"].str.strip() != "") & (df["audio_path"].apply(lambda p: Path(p).exists()))].copy()
    return df


@dataclass
class AccentProfile:
    mean_mel: np.ndarray
    std_mel: np.ndarray
    centroid_mean: float
    centroid_std: float
    pitch_mean: float
    pitch_std: float
    rms_mean: float
    rms_std: float


def load_audio_16k(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    return y.astype(np.float32), sr


def compute_features(y: np.ndarray, sr: int) -> dict[str, np.ndarray | float]:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean_mel = np.mean(log_mel, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    rms = librosa.feature.rms(y=y).flatten()

    # Robust pitch estimation. Keep only voiced values.
    f0 = librosa.yin(y, fmin=65.0, fmax=350.0, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
    else:
        pitch_mean = float(np.mean(f0))
        pitch_std = float(np.std(f0))

    return {
        "mean_mel": mean_mel,
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }


def build_accent_profile(audio_paths: Iterable[Path]) -> AccentProfile:
    mel_vectors: list[np.ndarray] = []
    centroid_means: list[float] = []
    centroid_stds: list[float] = []
    pitch_means: list[float] = []
    pitch_stds: list[float] = []
    rms_means: list[float] = []
    rms_stds: list[float] = []

    for path in audio_paths:
        y, sr = load_audio_16k(path)
        feats = compute_features(y, sr)
        mel_vectors.append(feats["mean_mel"])
        centroid_means.append(feats["centroid_mean"])
        centroid_stds.append(feats["centroid_std"])
        pitch_means.append(feats["pitch_mean"])
        pitch_stds.append(feats["pitch_std"])
        rms_means.append(feats["rms_mean"])
        rms_stds.append(feats["rms_std"])

    mel_stack = np.stack(mel_vectors, axis=0)
    return AccentProfile(
        mean_mel=np.mean(mel_stack, axis=0),
        std_mel=np.std(mel_stack, axis=0) + 1e-6,
        centroid_mean=float(np.mean(centroid_means)),
        centroid_std=float(np.mean(centroid_stds)),
        pitch_mean=float(np.mean(pitch_means)),
        pitch_std=float(np.mean(pitch_stds)),
        rms_mean=float(np.mean(rms_means)),
        rms_std=float(np.mean(rms_stds)),
    )


def distance_to_profile(feats: dict[str, np.ndarray | float], profile: AccentProfile) -> float:
    mel_dist = float(np.mean(np.abs((feats["mean_mel"] - profile.mean_mel) / profile.std_mel)))
    centroid_dist = abs(float(feats["centroid_mean"]) - profile.centroid_mean) / (profile.centroid_std + 1e-6)
    pitch_dist = abs(float(feats["pitch_mean"]) - profile.pitch_mean) / (profile.pitch_std + 1e-6)
    rms_dist = abs(float(feats["rms_mean"]) - profile.rms_mean) / (profile.rms_std + 1e-6)
    return float(mel_dist + 0.35 * centroid_dist + 0.20 * pitch_dist + 0.20 * rms_dist)


async def synthesize_tts(text: str, voice: str, rate: str, pitch: str, volume: str, mp3_out: Path) -> None:
    communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    await communicator.save(str(mp3_out))


def save_mel_plot(wav_path: Path, png_path: Path) -> None:
    y, sr = load_audio_16k(wav_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(9, 3.5))
    librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {wav_path.stem}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic data generation using accent-profile matching")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=project_root / "cv-corpus-25.0-2026-03-09",
        help="Common Voice corpus root",
    )
    parser.add_argument(
        "--target-accent",
        required=True,
        help="Exact accent label to model from Common Voice (e.g., 'India and South Asia (India, Pakistan, Sri Lanka)')",
    )
    parser.add_argument(
        "--voice",
        default="en-IN-NeerjaNeural",
        help="TTS voice used for synthesis",
    )
    parser.add_argument(
        "--num-reference",
        type=int,
        default=120,
        help="How many real accent samples to use when building the mel/acoustic profile",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=80,
        help="How many synthetic candidates to generate before filtering",
    )
    parser.add_argument(
        "--num-keep",
        type=int,
        default=40,
        help="How many best-matching synthetic samples to keep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "datasets" / "synthetic_data_generation",
    )
    parser.add_argument(
        "--save-mel-images",
        action="store_true",
        help="Save mel spectrogram PNGs for kept samples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    accent_slug = slugify(args.target_accent)
    run_dir = args.output_dir / accent_slug
    candidates_dir = run_dir / "candidates"
    kept_dir = run_dir / "kept"
    mel_dir = run_dir / "mel"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    kept_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mel_images:
        mel_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load real accent data and build a target acoustic profile.
    train_df = load_cv_train(args.cv_root)
    accent_df = train_df[train_df["accent"] == args.target_accent].copy()
    if accent_df.empty:
        raise ValueError(f"No rows found for target accent: {args.target_accent}")

    accent_df = accent_df.sample(n=min(args.num_reference, len(accent_df)), random_state=args.seed)
    profile_paths = [Path(p) for p in accent_df["audio_path"].tolist()]
    profile = build_accent_profile(profile_paths)

    # Use real target-accent transcripts as text prompts for synthesis.
    texts = accent_df["text"].tolist()

    rows = []
    for i in range(args.num_candidates):
        text = texts[i % len(texts)]

        rate = random.choice(["-8%", "-4%", "+0%", "+4%", "+8%"])
        pitch = random.choice(["-3Hz", "-1Hz", "+0Hz", "+1Hz", "+3Hz"])
        volume = random.choice(["-2%", "+0%", "+2%"])

        mp3_path = candidates_dir / f"cand_{i:04d}.mp3"
        wav_path = candidates_dir / f"cand_{i:04d}.wav"

        asyncio.run(synthesize_tts(text=text, voice=args.voice, rate=rate, pitch=pitch, volume=volume, mp3_out=mp3_path))

        y, sr = load_audio_16k(mp3_path)
        sf.write(wav_path, y, sr)

        feats = compute_features(y, sr)
        score = distance_to_profile(feats, profile)

        rows.append(
            {
                "candidate_id": f"cand_{i:04d}",
                "text": text,
                "voice": args.voice,
                "rate": rate,
                "pitch": pitch,
                "volume": volume,
                "mp3_path": str(mp3_path),
                "wav_path": str(wav_path),
                "distance_score": score,
            }
        )

    candidates = pd.DataFrame(rows).sort_values("distance_score", ascending=True).reset_index(drop=True)
    candidates.to_csv(run_dir / "candidates_scored.csv", index=False)

    keep_n = min(args.num_keep, len(candidates))
    kept = candidates.head(keep_n).copy()

    # Promote best candidates to final synthetic set.
    final_rows = []
    for rank, row in kept.iterrows():
        src_wav = Path(row["wav_path"])
        out_wav = kept_dir / f"synthetic_{rank:04d}.wav"
        out_txt = kept_dir / f"synthetic_{rank:04d}.txt"

        y, sr = load_audio_16k(src_wav)
        sf.write(out_wav, y, sr)
        out_txt.write_text(str(row["text"]))

        if args.save_mel_images:
            out_mel = mel_dir / f"synthetic_{rank:04d}_mel.png"
            save_mel_plot(out_wav, out_mel)

        final_rows.append(
            {
                "sample_id": f"synthetic_{rank:04d}",
                "audio_path": str(out_wav),
                "text": str(row["text"]),
                "target_accent": args.target_accent,
                "distance_score": float(row["distance_score"]),
                "voice": row["voice"],
                "rate": row["rate"],
                "pitch": row["pitch"],
                "volume": row["volume"],
                "is_synthetic": 1,
                "source_method": "tts+accent_profile_matching",
            }
        )

    manifest = pd.DataFrame(final_rows)
    manifest.to_csv(run_dir / "synthetic_manifest.csv", index=False)

    methodology = (
        "Method: (1) Build target accent acoustic profile from real Common Voice samples using log-mel mean vector, "
        "spectral centroid stats, pitch stats, and RMS stats; "
        "(2) Generate TTS candidates with controlled prosody jitter (rate/pitch/volume); "
        "(3) Score each candidate by distance to accent profile; "
        "(4) Keep top-scoring candidates as synthetic data."
    )
    (run_dir / "methodology.txt").write_text(methodology)

    print(f"Target accent: {args.target_accent}")
    print(f"Reference samples used: {len(profile_paths)}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Candidates kept: {len(manifest)}")
    print(f"Output run dir: {run_dir}")
    print(f"Manifest: {run_dir / 'synthetic_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
