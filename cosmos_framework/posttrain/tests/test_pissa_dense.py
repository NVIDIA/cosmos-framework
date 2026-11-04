# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned PiSSA dense execution: low-rank ownership, residual math and identity."""

import copy

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapters import DENSE_FORMAT, state_spec
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability, install_adapters
from cosmos_framework.posttrain.tests.test_adapters import FIRST, identity, tiny_net


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rslora", [False, True])
def test_dense_initialization_gradient_update_and_merge(dtype, rslora):
    torch.manual_seed(73)
    base = nn.Linear(16, 24, dtype=dtype)
    original = copy.deepcopy(base)
    layer = PosttrainAdapterLinear(
        base,
        Component(
            targets=[FIRST],
            method="pissa",
            execution="pissa_dense_v1",
            rank=4,
            alpha=7,
            use_rslora=rslora,
            initialization="svd",
        ),
    )
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)
    x = torch.randn(3, 5, 16, dtype=dtype)
    assert torch.equal(layer(x), original(x))
    initial = {name: value.clone() for name, value in layer.state_dict().items()}
    opt = torch.optim.AdamW([layer.lora_A.weight, layer.lora_B.weight], lr=0.01)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = layer(x).float().square().mean()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
            for p in (layer.lora_A.weight, layer.lora_B.weight)
        )
        opt.step()
    for name in ("weight", "bias", "initial_A", "initial_B"):
        assert torch.equal(layer.state_dict()[name], initial[name])
    for name in ("lora_A.weight", "lora_B.weight"):
        assert not torch.equal(layer.state_dict()[name], initial[name])
    merged = layer.merged_state_weight({"layer." + n: t for n, t in layer.state_dict().items()}, "layer")
    assert torch.equal(layer(x), torch.nn.functional.linear(x, merged, layer.bias))
    # Independent FP64 residual expression verifies the algebra separately from
    # the cancellation-preserving implementation and its serialized contract.
    residual = layer.weight.double() - layer._lora_scale * (layer.initial_B.double() @ layer.initial_A.double())
    expected = residual + layer._lora_scale * (layer.lora_B.weight.double() @ layer.lora_A.weight.double())
    torch.testing.assert_close(merged.float(), expected.float().to(dtype).float(), atol=1e-6, rtol=1e-6)


def test_dense_execution_identity_aliases_and_optimizer():
    net = tiny_net()
    alias = "model.language_model.layers.1.self_attn.q_proj"
    net.get_submodule(alias).weight = net.get_submodule(FIRST).weight
    cfg = AdapterConfig(
        components=[Component(targets=[FIRST, alias], method="pissa", execution="pissa_dense_v1", rank=2)]
    )
    manifest = install_adapters(net, cfg)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad])
    assert_trainability(net, opt)
    left, right = net.get_submodule(FIRST), net.get_submodule(alias)
    assert left.weight is right.weight and left.lora_A is right.lora_A and left.initial_A is right.initial_A
    record = identity(manifest)
    with pytest.raises(ValueError, match="format/execution"):
        state_spec(record)
    record["format"] = DENSE_FORMAT
    assert len(state_spec(record)) == 8
    record["resolved"]["targets"][0]["settings"]["execution"] = "unknown_v3"
    with pytest.raises(ValueError, match="execution semantics"):
        state_spec(record)


def test_legacy_identity_stays_unversioned_and_invalid_dense_combinations_fail():
    cfg = AdapterConfig(components=[Component(targets=[FIRST], rank=2)])
    manifest = install_adapters(tiny_net(), cfg)
    assert "execution" not in manifest["targets"][0]["settings"]
    state_spec(identity(manifest))
    with pytest.raises(ValueError, match="requires method=pissa"):
        Component(targets=[FIRST], execution="pissa_dense_v1")
    with pytest.raises(ValueError, match="dropout=0"):
        Component(targets=[FIRST], method="pissa", execution="pissa_dense_v1", dropout=0.1)
