# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Behavioral tests for custom LoRA injection and trainability guarantees."""

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.reasoner.defaults.policy_config import PolicyConfig
from cosmos_framework.configs.toml_config.sft_config import ModelConfig
from cosmos_framework.utils.generator.lora import (
    LoraInjectedLinear,
    init_lora_weights_post_materialization,
    inject_lora_pre_fsdp,
    set_only_lora_trainable,
)


class _TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 3, bias=True)
        self.untouched = nn.Linear(4, 3, bias=False)


def _materialize_meta_adapters_on_cpu(network: nn.Module) -> None:
    network._apply(
        lambda tensor: torch.empty_like(tensor, device="cpu") if tensor.device.type == "meta" else tensor,
        recurse=True,
    )
    init_lora_weights_post_materialization(network)


def test_lora_is_opt_in_by_default() -> None:
    assert PolicyConfig().lora_enabled is False
    assert ModelConfig().lora_enabled is False


def test_injected_adapter_starts_identical_then_learns() -> None:
    torch.manual_seed(7)
    network = _TinyNetwork()
    inputs = torch.randn(5, 4)
    expected_at_init = network.q_proj(inputs).detach().clone()
    base_weight = network.q_proj.weight.detach().clone()
    base_bias = network.q_proj.bias.detach().clone()

    inject_lora_pre_fsdp(
        network,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
    )
    _materialize_meta_adapters_on_cpu(network)

    assert isinstance(network.q_proj, LoraInjectedLinear)
    torch.testing.assert_close(network.q_proj(inputs), expected_at_init, rtol=0, atol=0)
    torch.testing.assert_close(network.q_proj.weight, base_weight, rtol=0, atol=0)
    torch.testing.assert_close(network.q_proj.bias, base_bias, rtol=0, atol=0)
    assert network.q_proj.lora_B.weight.count_nonzero().item() == 0
    assert {name for name, parameter in network.named_parameters() if parameter.requires_grad} == {
        "q_proj.lora_A.weight",
        "q_proj.lora_B.weight",
    }

    optimizer = torch.optim.SGD((parameter for parameter in network.parameters() if parameter.requires_grad), lr=0.1)
    network.q_proj(inputs).square().mean().backward()
    optimizer.step()

    assert network.q_proj.lora_B.weight.count_nonzero().item() > 0
    assert not torch.equal(network.q_proj(inputs), expected_at_init)
    torch.testing.assert_close(network.q_proj.weight, base_weight, rtol=0, atol=0)
    torch.testing.assert_close(network.q_proj.bias, base_bias, rtol=0, atol=0)


def test_lora_only_trainability_is_restored_after_broad_unfreeze() -> None:
    network = _TinyNetwork()
    inject_lora_pre_fsdp(
        network,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
    )

    # Simulate a legacy VLM freeze config such as trainable_params=[".*"].
    network.requires_grad_(True)
    assert any(parameter.requires_grad for name, parameter in network.named_parameters() if "lora_" not in name)

    assert set_only_lora_trainable(network) == 2
    assert all(parameter.requires_grad == ("lora_" in name) for name, parameter in network.named_parameters())


def test_path_exclusion_keeps_adapters_out_of_the_vision_tower() -> None:
    class EdgeShapedNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.ModuleDict({name: nn.Linear(4, 4) for name in ("q_proj", "k_proj", "v_proj")})
            self.language = nn.ModuleDict(
                {name: nn.Linear(4, 4) for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
            )

    network = EdgeShapedNetwork()
    inject_lora_pre_fsdp(
        network,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
        lora_exclude_path_regex=r"^visual\.",
    )

    assert all(not isinstance(module, LoraInjectedLinear) for module in network.visual.values())
    assert all(isinstance(module, LoraInjectedLinear) for module in network.language.values())


def test_injection_fails_when_exclusion_removes_every_target() -> None:
    with pytest.raises(RuntimeError, match="replaced 0 modules"):
        inject_lora_pre_fsdp(
            _TinyNetwork(),
            lora_rank=2,
            lora_alpha=4,
            lora_target_modules="q_proj",
            lora_exclude_path_regex=r".*",
        )
