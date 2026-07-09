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


import types

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.sound import inject_sound_into_batch


def _fake_model(sound_latent_t: int, temporal_cf: int = 4):
    sound_tok = types.SimpleNamespace(
        get_latent_num_samples=lambda n: sound_latent_t,
        audio_channels=2,
    )
    vision_tok = types.SimpleNamespace(temporal_compression_factor=temporal_cf)
    return types.SimpleNamespace(tokenizer_sound_gen=sound_tok, tokenizer_vision_gen=vision_tok)


def test_inject_sound_conditions_sound_and_preserves_image():
    model = _fake_model(sound_latent_t=50)
    video = torch.zeros(1, 3, 48, 16, 16)  # [1,3,T,H,W], T=48 -> 12 video latents @cf=4
    audio = torch.zeros(1, 2, 96000)
    batch = {
        "video": [video],
        "sequence_plan": [
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])
        ],
    }

    inject_sound_into_batch(batch, audio, model, condition_sound=True)

    plan = batch["sequence_plan"][0]
    assert plan.has_sound is True
    assert plan.condition_frame_indexes_sound == list(range(50))  # all sound conditioned (ts2v)
    assert plan.condition_frame_indexes_vision == [0]              # image cond preserved


def test_inject_sound_default_generates_sound():
    model = _fake_model(sound_latent_t=50)
    video = torch.zeros(1, 3, 48, 16, 16)
    audio = torch.zeros(1, 2, 96000)
    batch = {
        "video": [video],
        "sequence_plan": [
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
        ],
    }

    inject_sound_into_batch(batch, audio, model)  # default condition_sound=False

    plan = batch["sequence_plan"][0]
    assert plan.condition_frame_indexes_sound == []  # t2vs: sound generated
