"""Utilities for loading and running Whisper base for transcription."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import wave

import numpy as np
import torch
from transformers import pipeline

DEFAULT_MODEL_ID = "openai/whisper-base"
TARGET_SAMPLE_RATE = 16_000


@dataclass
class TranscriptionResult:
    text: str
    raw: dict


def _decode_pcm(data: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        # 8-bit PCM WAV is unsigned.
        audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        return (audio - 128.0) / 128.0

    if sample_width == 2:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    if sample_width == 3:
        # 24-bit little-endian PCM.
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        signed = (signed ^ sign_mask) - sign_mask
        return signed.astype(np.float32) / float(1 << 23)

    if sample_width == 4:
        audio = np.frombuffer(data, dtype=np.int32).astype(np.float32)
        return audio / float(1 << 31)

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def load_wav_mono(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        audio_bytes = wav_file.readframes(frame_count)

    audio = _decode_pcm(audio_bytes, sample_width)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_sample_rate:
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        duration_s = audio.shape[0] / sample_rate
        src_t = np.linspace(0.0, duration_s, num=audio.shape[0], endpoint=False)
        dst_len = max(1, int(round(duration_s * target_sample_rate)))
        dst_t = np.linspace(0.0, duration_s, num=dst_len, endpoint=False)
        audio = np.interp(dst_t, src_t, audio).astype(np.float32)

    return np.clip(audio.astype(np.float32), -1.0, 1.0)


class WhisperASR:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipe = self._build_pipeline()

    def _build_pipeline(self):
        project_root = Path(__file__).resolve().parents[1]
        cache_root = Path(
            os.environ.get("WHISPER_CACHE_DIR", str(project_root / ".hf_cache"))
        )
        hub_cache = cache_root / "hub"
        transformers_cache = cache_root / "transformers"
        hub_cache.mkdir(parents=True, exist_ok=True)
        transformers_cache.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

        has_cuda = torch.cuda.is_available()
        device = 0 if has_cuda else -1
        dtype = torch.float16 if has_cuda else torch.float32

        return pipeline(
            task="automatic-speech-recognition",
            model=self.model_id,
            device=device,
            language="en",
            torch_dtype=dtype,
            model_kwargs={"cache_dir": str(cache_root)},
        )

    def transcribe_wav(self, wav_path: str | Path) -> TranscriptionResult:
        audio = load_wav_mono(wav_path, target_sample_rate=TARGET_SAMPLE_RATE)
        output = self._pipe({"array": audio, "sampling_rate": TARGET_SAMPLE_RATE})
        text = str(output.get("text", "")).strip()
        return TranscriptionResult(text=text, raw=output)

    def transcribe_file(self, audio_path: str | Path) -> TranscriptionResult:
        path = Path(audio_path)
        if path.suffix.lower() == ".wav":
            return self.transcribe_wav(path)

        # Let Transformers handle decoding for non-WAV formats (e.g., MP3).
        output = self._pipe(str(path))
        text = str(output.get("text", "")).strip()
        return TranscriptionResult(text=text, raw=output)


# Backward-compatible alias used by scripts in this project.
WhisperTinyEN = WhisperASR
