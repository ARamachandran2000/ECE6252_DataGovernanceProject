#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synthetic_data_generation import (  # noqa: E402
    build_accent_profile,
    compute_features,
    distance_to_profile,
    load_audio_16k,
    load_cv_train,
    save_mel_plot,
    synthesize_tts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reselect Southern African synthetic candidates from African English voices only")
    project_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--cv-root",
        type=Path,
        default=project_root / "cv-corpus-25.0-2026-03-09",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=project_root
        / "datasets"
        / "synthetic_data_generation"
        / "southern_african_south_africa_zimbabwe_namibia",
    )
    p.add_argument("--num-reference", type=int, default=120)
    p.add_argument("--num-keep", type=int, default=3)
    p.add_argument("--cands-per-voice", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    target_accent = "Southern African (South Africa, Zimbabwe, Namibia)"
    african_english_voices = [
        "en-ZA-LeahNeural",
        "en-ZA-LukeNeural",
        "en-KE-AsiliaNeural",
        "en-KE-ChilembaNeural",
        "en-NG-AbeoNeural",
        "en-NG-EzinneNeural",
        "en-TZ-ElimuNeural",
        "en-TZ-ImaniNeural",
    ]

    run_dir = args.run_dir
    candidates_dir = run_dir / "candidates"
    kept_dir = run_dir / "kept"
    mel_dir = run_dir / "mel"

    for d in [candidates_dir, kept_dir, mel_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*"):
            if p.is_file():
                p.unlink()

    train_df = load_cv_train(args.cv_root)
    accent_df = train_df[train_df["accent"] == target_accent].copy()
    if accent_df.empty:
        raise ValueError(f"No rows found for target accent: {target_accent}")

    accent_df = accent_df.sample(n=min(args.num_reference, len(accent_df)), random_state=args.seed)
    profile_paths = [Path(p) for p in accent_df["audio_path"].tolist()]
    profile = build_accent_profile(profile_paths)
    texts = accent_df["text"].tolist()

    rows = []
    idx = 0
    for voice in african_english_voices:
        for _ in range(args.cands_per_voice):
            text = texts[idx % len(texts)]
            rate = random.choice(["-8%", "-4%", "+0%", "+4%", "+8%"])
            pitch = random.choice(["-3Hz", "-1Hz", "+0Hz", "+1Hz", "+3Hz"])
            volume = random.choice(["-2%", "+0%", "+2%"])

            cand_id = f"cand_{idx:04d}"
            mp3_path = candidates_dir / f"{cand_id}.mp3"
            wav_path = candidates_dir / f"{cand_id}.wav"

            asyncio.run(synthesize_tts(text=text, voice=voice, rate=rate, pitch=pitch, volume=volume, mp3_out=mp3_path))
            y, sr = load_audio_16k(mp3_path)
            sf.write(wav_path, y, sr)
            score = distance_to_profile(compute_features(y, sr), profile)

            rows.append(
                {
                    "candidate_id": cand_id,
                    "text": text,
                    "voice": voice,
                    "rate": rate,
                    "pitch": pitch,
                    "volume": volume,
                    "mp3_path": str(mp3_path.resolve()),
                    "wav_path": str(wav_path.resolve()),
                    "distance_score": float(score),
                }
            )
            idx += 1

    candidates = pd.DataFrame(rows).sort_values("distance_score", ascending=True).reset_index(drop=True)
    candidates.to_csv(run_dir / "candidates_scored.csv", index=False)

    kept = candidates.head(min(args.num_keep, len(candidates))).copy()
    final_rows = []
    for rank, row in kept.iterrows():
        src_wav = Path(row["wav_path"])
        out_wav = kept_dir / f"synthetic_{rank:04d}.wav"
        out_txt = kept_dir / f"synthetic_{rank:04d}.txt"

        y, sr = load_audio_16k(src_wav)
        sf.write(out_wav, y, sr)
        out_txt.write_text(str(row["text"]))
        save_mel_plot(out_wav, mel_dir / f"synthetic_{rank:04d}_mel.png")

        final_rows.append(
            {
                "sample_id": f"synthetic_{rank:04d}",
                "audio_path": str(out_wav.resolve()),
                "text": str(row["text"]),
                "target_accent": target_accent,
                "distance_score": float(row["distance_score"]),
                "voice": row["voice"],
                "rate": row["rate"],
                "pitch": row["pitch"],
                "volume": row["volume"],
                "is_synthetic": 1,
                "source_method": "tts+accent_profile_matching_african_voices_only",
            }
        )

    pd.DataFrame(final_rows).to_csv(run_dir / "synthetic_manifest.csv", index=False)

    methodology = (
        "Method: (1) Build target Southern African profile from Common Voice real samples using mel/centroid/pitch/RMS stats; "
        "(2) Generate candidates using African English TTS voices only (en-ZA, en-KE, en-NG, en-TZ) with controlled prosody jitter; "
        "(3) Score by distance to the target profile; "
        "(4) Keep top-scoring candidates."
    )
    (run_dir / "methodology.txt").write_text(methodology)

    print(f"Regenerated in: {run_dir}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Candidates kept: {len(final_rows)}")
    print(pd.DataFrame(final_rows)[["sample_id", "voice", "distance_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
