# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot.action_io_projector import (
    ACTION_IO_PROJECTOR_DOMAIN_AWARE,
    ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS,
    ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS,
    SharedWeightBiasLinear,
    SharedWeightNoBiasLinear,
    build_action_io_projector,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    ("projector_cls", "expected"),
    [
        (SharedWeightBiasLinear, torch.tensor([[4.5, 9.5], [4.5, 9.5]])),
        (SharedWeightNoBiasLinear, torch.tensor([[4.0, 10.0], [4.0, 10.0]])),
    ],
)
def test_shared_projector_output_is_independent_of_type_id(
    projector_cls: type[torch.nn.Module], expected: torch.Tensor
) -> None:
    layer = projector_cls(input_size=2, output_size=2, num_types=3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # [O,I]
        if isinstance(layer, SharedWeightBiasLinear):
            layer.bias.copy_(torch.tensor([0.5, -0.5]))  # [O]

    x = torch.tensor([[2.0, 1.0], [2.0, 1.0]])  # [B,I]
    output = layer(x, torch.tensor([-100, 1_000]))  # [B,O]
    scalar_output = layer(x, torch.tensor(99))  # [B,O]
    empty_output = layer(x, torch.empty(0, dtype=torch.long))  # [B,O]

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(scalar_output, expected)
    torch.testing.assert_close(empty_output, expected)


@pytest.mark.parametrize("projector_cls", [SharedWeightBiasLinear, SharedWeightNoBiasLinear])
def test_shared_projector_supports_rank_three_and_rejects_other_ranks(
    projector_cls: type[torch.nn.Module],
) -> None:
    layer = projector_cls(input_size=2, output_size=2, num_types=2)
    output = layer(torch.zeros(2, 3, 2), torch.tensor([0, 1]))  # [B,T,O]
    assert output.shape == (2, 3, 2)

    with pytest.raises(ValueError, match="rank-2 or rank-3"):
        layer(torch.zeros(1, 1, 1, 2), torch.tensor([0]))  # [B,T,H,I], [B]


def test_shared_weight_projectors_have_expected_parameters() -> None:
    bias = SharedWeightBiasLinear(input_size=2, output_size=3, num_types=4)
    no_bias = SharedWeightNoBiasLinear(input_size=2, output_size=3, num_types=4)

    assert bias.weight.shape == (3, 2)
    assert bias.bias.shape == (3,)
    assert list(bias.state_dict()) == ["weight", "bias"]
    assert no_bias.weight.shape == (3, 2)
    assert list(no_bias.state_dict()) == ["weight"]
    assert not hasattr(no_bias, "bias")


def test_builder_selects_supported_projectors() -> None:
    domain_aware = build_action_io_projector(ACTION_IO_PROJECTOR_DOMAIN_AWARE, 2, 3, 4)
    shared_bias = build_action_io_projector(ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS, 2, 3, 4)
    shared_no_bias = build_action_io_projector(ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS, 2, 3, 4)

    assert isinstance(domain_aware, DomainAwareLinear)
    assert isinstance(shared_bias, SharedWeightBiasLinear)
    assert isinstance(shared_no_bias, SharedWeightNoBiasLinear)


def test_projector_initialization_supports_domain_aware_projector() -> None:
    layer = DomainAwareLinear(input_size=8, output_size=4, num_domains=3)
    with torch.no_grad():
        layer.fc.weight.zero_()
        layer.bias.weight.fill_(1.0)

    layer.initialize_action_parameters(std=0.125)

    assert float(layer.fc.weight.detach().std()) > 0.0
    torch.testing.assert_close(layer.bias.weight, torch.zeros_like(layer.bias.weight))


@pytest.mark.parametrize("projector_cls", [SharedWeightBiasLinear, SharedWeightNoBiasLinear])
def test_projector_initialization_supports_shared_projectors(projector_cls: type[torch.nn.Module]) -> None:
    layer = projector_cls(input_size=8, output_size=4, num_types=3)
    with torch.no_grad():
        layer.weight.zero_()
        if isinstance(layer, SharedWeightBiasLinear):
            layer.bias.fill_(1.0)

    layer.initialize_action_parameters(std=0.125)

    assert float(layer.weight.detach().std()) > 0.0
    if isinstance(layer, SharedWeightBiasLinear):
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))


def test_builder_rejects_unknown_projector_type() -> None:
    with pytest.raises(ValueError, match="Unsupported action_io_projector_type"):
        build_action_io_projector("shared_weight_type_bias", 2, 3, 4)


@pytest.mark.parametrize(
    ("projector_type", "projector_cls"),
    [
        (ACTION_IO_PROJECTOR_SHARED_WEIGHT_BIAS, SharedWeightBiasLinear),
        (ACTION_IO_PROJECTOR_SHARED_WEIGHT_NO_BIAS, SharedWeightNoBiasLinear),
    ],
)
def test_network_builds_same_projector_type_for_encoder_and_decoder(
    projector_type: str, projector_cls: type[torch.nn.Module]
) -> None:
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig

    text_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
    )
    language_model = torch.nn.Module()
    language_model.config = text_config
    config = Cosmos3VFMNetworkConfig(
        action_gen=True,
        action_dim=4,
        num_embodiment_domains=3,
        action_io_projector_type=projector_type,
        latent_channel_size=4,
        latent_patch_size=1,
        max_latent_h=1,
        max_latent_w=1,
        max_latent_t=1,
        vlm_config=text_config,
    )

    model = Cosmos3VFMNetwork(language_model, config)

    assert isinstance(model.action2llm, projector_cls)
    assert isinstance(model.llm2action, projector_cls)
    assert hasattr(model, "action_modality_embed")


def test_network_omits_action_modality_embedding_when_disabled() -> None:
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig

    text_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
    )
    language_model = torch.nn.Module()
    language_model.config = text_config
    config = Cosmos3VFMNetworkConfig(
        action_gen=True,
        enable_action_modality_embedding=False,
        action_dim=4,
        latent_channel_size=4,
        latent_patch_size=1,
        max_latent_h=1,
        max_latent_w=1,
        max_latent_t=1,
        vlm_config=text_config,
    )

    model = Cosmos3VFMNetwork(language_model, config)

    assert hasattr(model, "action2llm")
    assert hasattr(model, "llm2action")
    assert not hasattr(model, "action_modality_embed")
