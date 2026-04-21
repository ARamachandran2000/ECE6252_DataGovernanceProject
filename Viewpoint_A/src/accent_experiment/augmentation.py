from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AugmentedSample:
    source_sample_id: str
    source_accent: str
    augmentation_type: str
    output_audio_path: str


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def _resample_if_needed(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    return torchaudio.functional.resample(waveform, src_sr, dst_sr)


def _augment_speed(waveform: torch.Tensor, sample_rate: int, factor: float) -> torch.Tensor:
    new_sr = max(1000, int(round(sample_rate * factor)))
    # Resample then store back at original sample rate. This changes tempo/pitch naturally.
    return torchaudio.functional.resample(waveform, sample_rate, new_sr)


def _augment_noise(waveform: torch.Tensor, snr_db: float = 24.0) -> torch.Tensor:
    noise = torch.randn_like(waveform)
    signal_power = waveform.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    snr = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    return waveform + scale * noise


def _augment_volume(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    gain = 10 ** (gain_db / 20.0)
    return waveform * gain


def _augment_reverb(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # Small synthetic room impulse response.
    ir_len = int(sample_rate * 0.08)
    t = torch.linspace(0, 1, ir_len)
    decay = torch.exp(-6.0 * t)
    ir = torch.zeros(ir_len)
    ir[0] = 1.0
    ir += 0.35 * decay * torch.randn(ir_len)
    ir = ir / ir.abs().max().clamp(min=1e-6)

    conv = torchaudio.functional.fftconvolve(waveform, ir.view(1, -1))
    return conv[:, : waveform.shape[1]]


def _augment_time_mask(waveform: torch.Tensor, max_fraction: float = 0.08) -> torch.Tensor:
    length = waveform.shape[1]
    mask_len = max(1, int(length * max_fraction))
    if length <= mask_len:
        return waveform
    start = random.randint(0, length - mask_len)
    out = waveform.clone()
    out[:, start : start + mask_len] = 0.0
    return out


def apply_augmentation(waveform: torch.Tensor, sample_rate: int, aug_type: str) -> torch.Tensor:
    if aug_type == "speed_0.90":
        return _augment_speed(waveform, sample_rate, 0.90)
    if aug_type == "speed_1.10":
        return _augment_speed(waveform, sample_rate, 1.10)
    if aug_type == "noise_snr24":
        return _augment_noise(waveform, snr_db=24.0)
    if aug_type == "volume_down_3db":
        return _augment_volume(waveform, gain_db=-3.0)
    if aug_type == "volume_up_3db":
        return _augment_volume(waveform, gain_db=3.0)
    if aug_type == "reverb_light":
        return _augment_reverb(waveform, sample_rate)
    if aug_type == "time_mask_8pct":
        return _augment_time_mask(waveform, max_fraction=0.08)
    raise ValueError(f"Unknown augmentation type: {aug_type}")


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    waveform = _resample_if_needed(waveform, int(sample_rate), TARGET_SAMPLE_RATE)
    waveform = waveform.clamp(-1.0, 1.0)
    return waveform, TARGET_SAMPLE_RATE


def save_audio(audio_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.clamp(-1.0, 1.0).squeeze(0).cpu().numpy()
    sf.write(str(audio_path), wav, sample_rate)
