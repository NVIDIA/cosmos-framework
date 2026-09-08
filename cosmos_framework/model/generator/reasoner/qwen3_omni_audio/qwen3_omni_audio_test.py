# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the Qwen3-Omni Thinking audio frontend."""

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.configuration_qwen3_omni_audio import (
    Qwen3OmniAudioConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_omni_audio.qwen3_omni_audio import (
    Qwen3OmniAudioModel,
    Qwen3OmniThinkingAudioEncoder,
    get_qwen3_omni_audio_output_lengths,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _get_tiny_encoder_config() -> Qwen3OmniMoeAudioEncoderConfig:
    return Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=32,
        n_window=50,
        output_dim=12,
        n_window_infer=100,
        conv_chunksize=2,
        downsample_hidden_size=2,
        attn_implementation="eager",
    )


def _get_tiny_audio_config() -> Qwen3OmniAudioConfig:
    return Qwen3OmniAudioConfig(
        encoder_config=_get_tiny_encoder_config(),
        projection_hidden_size=24,
        out_hidden_size=10,
    )


def test_thinking_defaults_match_released_audio_tower() -> None:
    config = Qwen3OmniAudioConfig().encoder_config

    assert config.num_mel_bins == 128
    assert config.d_model == 1280
    assert config.encoder_layers == 32
    assert config.encoder_attention_heads == 20
    assert config.encoder_ffn_dim == 5120
    assert config.downsample_hidden_size == 480
    assert config.n_window == 50
    assert config.n_window_infer == 800
    assert config.output_dim == 2048
    assert config._attn_implementation == "flash_attention_2"


def test_output_length_formula_matches_native_window_boundaries() -> None:
    input_lengths = torch.tensor([1, 7, 8, 9, 99, 100, 101, 107, 108, 109, 199, 200])

    output_lengths = get_qwen3_omni_audio_output_lengths(input_lengths)

    assert torch.equal(output_lengths, torch.tensor([1, 1, 1, 2, 13, 13, 14, 14, 14, 15, 26, 26]))


def test_encoder_packs_and_repads_variable_length_clips() -> None:
    torch.manual_seed(11)
    encoder = Qwen3OmniThinkingAudioEncoder(_get_tiny_encoder_config()).eval()
    features = torch.randn(2, 17, 128)
    feature_lengths = torch.tensor([17, 9])

    with torch.no_grad():
        batched, output_lengths = encoder(features, feature_lengths)
        first, _ = encoder(features[:1], feature_lengths[:1])
        second, _ = encoder(features[1:2, :9], feature_lengths[1:])

    assert batched.shape == (2, 3, 12)
    assert torch.equal(output_lengths, torch.tensor([3, 2]))
    torch.testing.assert_close(batched[0, :3], first[0, :3])
    torch.testing.assert_close(batched[1, :2], second[0, :2])
    assert torch.equal(batched[1, 2], torch.zeros_like(batched[1, 2]))


def test_composed_audio_model_projects_and_backpropagates() -> None:
    model = Qwen3OmniAudioModel(_get_tiny_audio_config()).train()
    features = torch.randn(2, 17, 128, requires_grad=True)
    feature_lengths = torch.tensor([17, 9])
    token_lengths = torch.tensor([3, 2])

    embeddings, output_lengths = model(features, feature_lengths, token_lengths)
    embeddings.sum().backward()

    assert embeddings.shape == (2, 3, 10)
    assert torch.equal(output_lengths, token_lengths)
    assert features.grad is not None
    assert model.projector.linear_fc2.weight.grad is not None


def test_encoder_state_dict_matches_standalone_artifact_namespace() -> None:
    state_dict = Qwen3OmniThinkingAudioEncoder(_get_tiny_encoder_config()).state_dict()

    assert state_dict
    assert all(name.startswith("encoder.") for name in state_dict)
    assert "encoder.positional_embedding.positional_embedding" not in state_dict


def test_meta_materialization_rebuilds_positions_and_projector() -> None:
    reference = Qwen3OmniAudioModel(_get_tiny_audio_config())
    reference_positions = reference.encoder.encoder.positional_embedding.positional_embedding.clone()
    with torch.device("meta"):
        model = Qwen3OmniAudioModel(_get_tiny_audio_config())
    model.to_empty(device="cpu")

    model.init_weights(buffer_device=torch.device("cpu"))

    positions = model.encoder.encoder.positional_embedding.positional_embedding
    assert torch.equal(positions, reference_positions)
    assert "encoder.encoder.positional_embedding.positional_embedding" not in model.state_dict()
    assert torch.equal(model.projector.norm.weight, torch.ones_like(model.projector.norm.weight))
    assert torch.equal(model.projector.norm.bias, torch.zeros_like(model.projector.norm.bias))
    assert torch.isfinite(model.projector.linear_fc1.weight).all()
    assert torch.isfinite(model.projector.linear_fc1.bias).all()
    assert torch.isfinite(model.projector.linear_fc2.weight).all()
    assert torch.isfinite(model.projector.linear_fc2.bias).all()
