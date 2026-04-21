#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys
import wave

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_setup import WhisperASR


def write_tone_wav(path: Path, sample_rate: int = 16_000, duration_s: float = 1.0) -> None:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm16 = (tone * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper base smoke test")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional WAV file to use for testing. If omitted, a generated tone is used.",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-base",
        help="Hugging Face model id (default: openai/whisper-base)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.audio is not None:
        if not args.audio.exists():
            print(f"Audio file does not exist: {args.audio}")
            return 1
        test_wav = args.audio
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_smoke_"))
        test_wav = tmp_dir / "tone.wav"
        write_tone_wav(test_wav)

    transcriber = WhisperASR(model_id=args.model)
    result = transcriber.transcribe_wav(test_wav)

    if not isinstance(result.text, str):
        print("Smoke test failed: output text is not a string")
        return 1

    print("Smoke test passed.")
    print(f"Model: {args.model}")
    print(f"Input: {test_wav}")
    print(f"Output text: {result.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
