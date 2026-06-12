# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import soundfile as sf
import torch

from cosmos_framework.inference.sound import load_conditioning_audio


def _write_wav(path: Path, sample_rate: int, channels: int, num_samples: int) -> None:
    if channels > 1:
        data = torch.zeros(num_samples, channels).numpy()
    else:
        data = torch.zeros(num_samples).numpy()
    sf.write(str(path), data, sample_rate)


def test_load_conditioning_audio_resamples_and_pads(tmp_path: Path):
    src = tmp_path / "in.wav"
    _write_wav(src, sample_rate=44100, channels=1, num_samples=44100)  # 1.0s mono @44.1k

    out = load_conditioning_audio(src, sample_rate=48000, audio_channels=2, num_samples=96000)

    assert out.shape == (1, 2, 96000)  # [1, C, N]; stereo, padded to 2.0s @48k
    assert out.dtype == torch.float32


def test_load_conditioning_audio_trims(tmp_path: Path):
    src = tmp_path / "in.wav"
    _write_wav(src, sample_rate=48000, channels=2, num_samples=48000 * 4)  # 4s stereo @48k

    out = load_conditioning_audio(src, sample_rate=48000, audio_channels=2, num_samples=48000 * 2)

    assert out.shape == (1, 2, 48000 * 2)  # trimmed to 2s
