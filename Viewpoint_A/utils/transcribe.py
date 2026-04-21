#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_setup import WhisperASR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file with Whisper base")
    parser.add_argument("audio", type=Path, help="Path to a WAV file")
    parser.add_argument(
        "--model",
        default="openai/whisper-base",
        help="Hugging Face model id (default: openai/whisper-base)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.exists():
        print(f"Audio file does not exist: {args.audio}")
        return 1

    transcriber = WhisperASR(model_id=args.model)
    result = transcriber.transcribe_file(args.audio)

    print("Transcription:")
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
