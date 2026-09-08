# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for Qwen3-Omni Thinking raw-audio preprocessing."""

import os

import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.configuration_qwen3_omni_audio import (
    Qwen3OmniAudioConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.qwen3_omni_audio import (
    Qwen3OmniAudioModel,
    get_qwen3_omni_audio_output_lengths,
)
from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model
from cosmos_framework.data.generator.processors.qwen3_omni_audio_processor import Qwen3OmniAudioProcessor


class _FakeWhisperFeatureExtractor:
    chunk_length: int = 300
    feature_size: int = 128
    hop_length: int = 160
    n_fft: int = 400
    sampling_rate: int = 16_000

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        assert return_attention_mask is True
        assert padding is True
        assert truncation is False
        feature_lengths = torch.tensor(
            [(waveform.shape[0] + self.hop_length - 1) // self.hop_length for waveform in raw_speech]
        )
        max_frames = int(feature_lengths.max())
        frame_indices = torch.arange(max_frames)
        return {
            "input_features": torch.zeros(len(raw_speech), self.feature_size, max_frames),
            "attention_mask": frame_indices.unsqueeze(0) < feature_lengths.unsqueeze(1),
        }


@pytest.mark.L0
@pytest.mark.CPU
def test_processor_returns_common_abi_and_exact_qwen_lengths() -> None:
    processor = Qwen3OmniAudioProcessor(feature_extractor=_FakeWhisperFeatureExtractor())
    audios = [
        torch.linspace(-0.25, 0.25, 16_000),
        np.linspace(-0.5, 0.5, 8_001, dtype=np.float32),
    ]

    outputs = processor(audios, sampling_rate=16_000)

    assert set(outputs) == {"audio_features", "audio_feature_lengths", "audio_token_lengths"}
    assert outputs["audio_features"].shape == (2, 100, 128)
    assert outputs["audio_features"].dtype == torch.float32
    assert torch.equal(outputs["audio_feature_lengths"], torch.tensor([100, 51]))
    assert torch.equal(outputs["audio_token_lengths"], torch.tensor([13, 7]))


@pytest.mark.L0
@pytest.mark.CPU
def test_default_feature_extractor_matches_thinking_frontend() -> None:
    processor = Qwen3OmniAudioProcessor()

    outputs = processor(np.zeros(320, dtype=np.float32), sampling_rate=16_000)

    assert processor.feature_extractor.sampling_rate == 16_000
    assert processor.feature_extractor.feature_size == 128
    assert processor.feature_extractor.hop_length == 160
    assert processor.feature_extractor.n_fft == 400
    assert processor.feature_extractor.chunk_length == 300
    assert outputs["audio_features"].shape == (1, 2, 128)
    assert torch.equal(outputs["audio_feature_lengths"], torch.tensor([2]))
    assert torch.equal(outputs["audio_token_lengths"], torch.tensor([1]))


@pytest.mark.L0
@pytest.mark.CPU
def test_token_timestamps_follow_released_logical_clock() -> None:
    timestamps = Qwen3OmniAudioProcessor.get_token_timestamps(201)

    assert len(timestamps) == 27
    assert timestamps == pytest.approx([token_index / 13 for token_index in range(27)])

    feature_lengths = torch.tensor([99, 100, 101, 199, 200, 201])
    output_lengths = get_qwen3_omni_audio_output_lengths(feature_lengths)
    assert [len(Qwen3OmniAudioProcessor.get_token_timestamps(int(length))) for length in feature_lengths] == (
        output_lengths.tolist()
    )


@pytest.mark.L1
@pytest.mark.GPU
def test_real_encoder_checkpoint_extracts_features_from_raw_audio() -> None:
    """Smoke-test the complete standalone frontend against the real artifact."""
    checkpoint_path = os.environ.get("QWEN3_OMNI_AUDIO_ENCODER_TEST_CHECKPOINT")
    if checkpoint_path is None:
        pytest.skip("QWEN3_OMNI_AUDIO_ENCODER_TEST_CHECKPOINT is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Qwen3-Omni audio encoder smoke test")

    credentials_path = os.environ.get("QWEN3_OMNI_AUDIO_ENCODER_TEST_CREDENTIALS", "")
    processor = Qwen3OmniAudioProcessor()
    processor_outputs = processor(torch.linspace(-0.1, 0.1, 16_000), sampling_rate=16_000)
    model = Qwen3OmniAudioModel(
        Qwen3OmniAudioConfig(
            projection_hidden_size=64,
            out_hidden_size=32,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    loaded_names = load_vlm_model(
        model=model.encoder,
        checkpoint_path=checkpoint_path,
        credential_path=credentials_path or None,
        parallel_dims=None,
    )
    model.eval()
    cuda_inputs = {name: tensor.cuda() for name, tensor in processor_outputs.items()}
    cuda_inputs["audio_features"] = cuda_inputs["audio_features"].to(torch.bfloat16)

    with torch.no_grad():
        features, output_lengths = model(**cuda_inputs)

    assert len(loaded_names) == len(model.encoder.state_dict())
    assert features.shape == (1, int(output_lengths.max()), 32)
    assert torch.equal(output_lengths, cuda_inputs["audio_token_lengths"].to(output_lengths.dtype))
    assert torch.isfinite(features).all()
