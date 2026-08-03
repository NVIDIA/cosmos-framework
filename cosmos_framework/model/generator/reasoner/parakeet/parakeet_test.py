# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for Parakeet encoder and projector components."""

import pytest
import torch
import transformers
from torch import nn
from transformers import ParakeetEncoderConfig

from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import (
    ParakeetAudioConfig,
    get_nemotron_parakeet_config,
)
from cosmos_framework.model.generator.reasoner.parakeet.parakeet import (
    NemotronParakeetEncoder,
    ParakeetAudioModel,
    ParakeetAudioProjector,
    get_subsampled_lengths,
)


def _get_tiny_encoder_config(
    *,
    convolution_bias: bool = False,
    subsampling_factor: int = 8,
) -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=False,
        conv_kernel_size=3,
        convolution_bias=convolution_bias,
        subsampling_factor=subsampling_factor,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        scale_input=False,
    )


def _get_tiny_audio_config() -> ParakeetAudioConfig:
    return ParakeetAudioConfig(
        encoder_config=_get_tiny_encoder_config(),
        projection_hidden_size=32,
        out_hidden_size=24,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_nemotron_encoder_config_matches_released_checkpoint() -> None:
    config = get_nemotron_parakeet_config()

    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 8
    assert config.intermediate_size == 4096
    assert config.conv_kernel_size == 9
    assert config.convolution_bias is False
    assert config.subsampling_factor == 8
    assert config.subsampling_conv_channels == 256
    assert config.num_mel_bins == 128
    assert config.scale_input is False


@pytest.mark.L0
@pytest.mark.CPU
def test_audio_config_defaults_match_reasoner_projector() -> None:
    config = ParakeetAudioConfig()

    assert config.encoder_config.hidden_size == 1024
    assert config.projection_hidden_size == 4096
    assert config.out_hidden_size == 2688


@pytest.mark.L0
@pytest.mark.CPU
def test_audio_config_accepts_encoder_dictionary_and_reasoner_output_size() -> None:
    encoder_config = get_nemotron_parakeet_config().to_dict()
    encoder_config["hidden_size"] = 32

    config = ParakeetAudioConfig(
        encoder_config=encoder_config,
        projection_hidden_size=48,
        out_hidden_size=64,
    )

    assert config.encoder_config.hidden_size == 32
    assert config.projection_hidden_size == 48
    assert config.out_hidden_size == 64


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"projection_hidden_size": 0}, "projection_hidden_size"),
        ({"out_hidden_size": -1}, "out_hidden_size"),
    ],
)
def test_audio_config_rejects_invalid_projector_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ParakeetAudioConfig(**kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_subsampled_lengths_match_three_stride_two_layers() -> None:
    input_lengths = torch.tensor([1, 2, 7, 8, 9, 2999, 3000], dtype=torch.long)

    output_lengths = get_subsampled_lengths(input_lengths, _get_tiny_encoder_config())

    assert torch.equal(output_lengths, torch.tensor([1, 1, 1, 1, 2, 375, 375], dtype=torch.int))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("subsampling_factor", [0, 1, 6])
def test_subsampled_lengths_reject_non_power_of_two_factor(subsampling_factor: int) -> None:
    config = _get_tiny_encoder_config(subsampling_factor=subsampling_factor)

    with pytest.raises(ValueError, match="integer power of two greater than or equal to 2"):
        get_subsampled_lengths(torch.tensor([8], dtype=torch.long), config)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("subsampling_factor", [0, 1, 6])
def test_encoder_construction_rejects_non_power_of_two_factor(subsampling_factor: int) -> None:
    config = _get_tiny_encoder_config(subsampling_factor=subsampling_factor)

    with pytest.raises(ValueError, match="integer power of two greater than or equal to 2"):
        NemotronParakeetEncoder(config)


@pytest.mark.L0
@pytest.mark.CPU
def test_bias_free_config_removes_transformers_457_compatibility_parameters() -> None:
    model = NemotronParakeetEncoder(_get_tiny_encoder_config())

    conv_bias_keys = {
        key
        for key in model.state_dict()
        if key.endswith(("pointwise_conv1.bias", "depthwise_conv.bias", "pointwise_conv2.bias"))
    }

    assert not conv_bias_keys


@pytest.mark.L0
@pytest.mark.CPU
def test_biased_config_preserves_convolution_biases() -> None:
    config = _get_tiny_encoder_config(convolution_bias=True)
    model = NemotronParakeetEncoder(config)

    conv_bias_keys = {
        key
        for key in model.state_dict()
        if key.endswith(("pointwise_conv1.bias", "depthwise_conv.bias", "pointwise_conv2.bias"))
    }

    assert len(conv_bias_keys) == 3 * config.num_hidden_layers


@pytest.mark.L0
@pytest.mark.CPU
def test_bias_removal_reports_incompatible_transformers_structure() -> None:
    model = NemotronParakeetEncoder(_get_tiny_encoder_config(convolution_bias=True))
    first_bias = model.encoder.layers[0].conv.pointwise_conv1.bias
    model.encoder.layers[1].conv.pointwise_conv2 = nn.Identity()

    with pytest.raises(RuntimeError) as error:
        model._remove_convolution_biases()

    message = str(error.value)
    assert transformers.__version__ in message
    assert "encoder.layers[1].conv.pointwise_conv2.bias" in message
    assert model.encoder.layers[0].conv.pointwise_conv1.bias is first_bias


@pytest.mark.L0
@pytest.mark.CPU
def test_encoder_state_dict_strict_load_preserves_checkpoint_keys() -> None:
    source = NemotronParakeetEncoder(_get_tiny_encoder_config())
    target = NemotronParakeetEncoder(_get_tiny_encoder_config())

    incompatible_keys = target.load_state_dict(source.state_dict(), strict=True)

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []


@pytest.mark.L0
@pytest.mark.CPU
def test_forward_returns_embeddings_and_valid_lengths() -> None:
    torch.manual_seed(7)
    config = _get_tiny_encoder_config()
    model = NemotronParakeetEncoder(config).eval()
    input_features = torch.randn(2, 17, config.num_mel_bins)
    input_lengths = torch.tensor([17, 9], dtype=torch.long)

    with torch.no_grad():
        embeddings, output_lengths = model(input_features, input_lengths)

    assert embeddings.shape == (2, 3, config.hidden_size)
    assert torch.equal(output_lengths, torch.tensor([3, 2], dtype=torch.int))


@pytest.mark.L0
@pytest.mark.CPU
def test_padding_does_not_change_valid_output_prefix() -> None:
    torch.manual_seed(11)
    config = _get_tiny_encoder_config()
    model = NemotronParakeetEncoder(config).eval()
    input_features = torch.randn(2, 17, config.num_mel_bins)
    input_lengths = torch.tensor([17, 9], dtype=torch.long)

    with torch.no_grad():
        batched_embeddings, output_lengths = model(input_features, input_lengths)
        unpadded_embeddings, _ = model(input_features[1:2, :9], input_lengths[1:2])

    # SDPA accumulation changes slightly when the padded attention matrix has a
    # different shape, so compare at a tolerance appropriate for the valid prefix.
    torch.testing.assert_close(
        batched_embeddings[1, : output_lengths[1]],
        unpadded_embeddings[0],
        rtol=1e-2,
        atol=5e-3,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_forward_rejects_invalid_mel_dimension() -> None:
    config = _get_tiny_encoder_config()
    model = NemotronParakeetEncoder(config)
    input_features = torch.randn(1, 8, config.num_mel_bins - 1)

    with pytest.raises(ValueError, match="mel bins"):
        model(input_features, torch.tensor([8], dtype=torch.long))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("input_length", [0, 9])
def test_forward_rejects_length_outside_padded_feature_range(input_length: int) -> None:
    config = _get_tiny_encoder_config()
    model = NemotronParakeetEncoder(config)
    input_features = torch.randn(1, 8, config.num_mel_bins)

    with pytest.raises(ValueError, match="between 1 and the padded feature time dimension"):
        model(input_features, torch.tensor([input_length], dtype=torch.long))


@pytest.mark.L0
@pytest.mark.CPU
def test_projector_matches_qwen_vl_shape_and_bias_contract() -> None:
    config = _get_tiny_audio_config()
    projector = ParakeetAudioProjector(config)

    assert projector.norm.weight.shape == (16,)
    assert projector.linear_fc1.weight.shape == (32, 16)
    assert projector.linear_fc2.weight.shape == (24, 32)
    assert isinstance(projector.norm, nn.LayerNorm)
    assert isinstance(projector.act_fn, nn.GELU)
    assert set(projector.state_dict()) == {
        "norm.weight",
        "norm.bias",
        "linear_fc1.weight",
        "linear_fc1.bias",
        "linear_fc2.weight",
        "linear_fc2.bias",
    }

    batched_output = projector(torch.randn(2, 3, 16))
    packed_output = projector(torch.randn(5, 16))
    assert batched_output.shape == (2, 3, 24)
    assert packed_output.shape == (5, 24)


@pytest.mark.L0
@pytest.mark.CPU
def test_projector_reset_parameters_initializes_norm_and_linears() -> None:
    projector = ParakeetAudioProjector(_get_tiny_audio_config()).to(device="meta")
    projector.to_empty(device="cpu")

    projector.reset_parameters()

    assert torch.equal(projector.norm.weight, torch.ones_like(projector.norm.weight))
    assert torch.equal(projector.norm.bias, torch.zeros_like(projector.norm.bias))
    assert torch.isfinite(projector.linear_fc1.weight).all()
    assert torch.isfinite(projector.linear_fc1.bias).all()
    assert torch.isfinite(projector.linear_fc2.weight).all()
    assert torch.isfinite(projector.linear_fc2.bias).all()


@pytest.mark.L0
@pytest.mark.CPU
def test_composed_audio_model_projects_encoder_outputs() -> None:
    torch.manual_seed(13)
    config = _get_tiny_audio_config()
    model = ParakeetAudioModel(config).eval()
    processor_outputs = {
        "audio_features": torch.randn(2, 17, config.encoder_config.num_mel_bins),
        "audio_feature_lengths": torch.tensor([17, 9], dtype=torch.long),
        "audio_token_lengths": torch.tensor([3, 2], dtype=torch.long),
    }

    with torch.no_grad():
        embeddings, output_lengths = model(**processor_outputs)

    assert embeddings.shape == (2, 3, config.out_hidden_size)
    assert torch.equal(output_lengths, torch.tensor([3, 2], dtype=torch.int))


@pytest.mark.L0
@pytest.mark.CPU
def test_composed_audio_model_backpropagates_through_encoder_and_projector() -> None:
    torch.manual_seed(17)
    config = _get_tiny_audio_config()
    model = ParakeetAudioModel(config).train()

    embeddings, _ = model(
        audio_features=torch.randn(2, 17, config.encoder_config.num_mel_bins),
        audio_feature_lengths=torch.tensor([17, 9], dtype=torch.long),
        audio_token_lengths=torch.tensor([3, 2], dtype=torch.long),
    )
    embeddings.square().mean().backward()

    encoder_gradients = [parameter.grad for parameter in model.encoder.parameters() if parameter.requires_grad]
    projector_gradients = [parameter.grad for parameter in model.projector.parameters() if parameter.requires_grad]
    assert encoder_gradients and all(gradient is not None for gradient in encoder_gradients)
    assert projector_gradients and all(gradient is not None for gradient in projector_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in encoder_gradients if gradient is not None)
    assert all(torch.isfinite(gradient).all() for gradient in projector_gradients if gradient is not None)


@pytest.mark.L0
@pytest.mark.CPU
def test_composed_audio_model_rejects_incorrect_processor_token_lengths() -> None:
    config = _get_tiny_audio_config()
    model = ParakeetAudioModel(config).eval()

    with pytest.raises(ValueError, match="do not match encoder output lengths"):
        model(
            audio_features=torch.randn(1, 17, config.encoder_config.num_mel_bins),
            audio_feature_lengths=torch.tensor([17], dtype=torch.long),
            audio_token_lengths=torch.tensor([4], dtype=torch.long),
        )
