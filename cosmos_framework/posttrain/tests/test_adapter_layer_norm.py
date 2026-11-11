# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""FP32 extra LayerNorm ownership with native BF16 vision activations."""

import copy

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability, install_adapters
from cosmos_framework.posttrain.tests.test_adapters import FIRST, tiny_net


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_extra_layer_norm_bf16_forward_backward_and_ownership(device, checkpointing):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA qualification is a separate installed-package check")
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(17)
    net = tiny_net().to(device=device, dtype=torch.bfloat16)
    net.model.language_model.norm = nn.LayerNorm(1152, device=device, dtype=torch.bfloat16)
    norm = net.model.language_model.norm
    original_ids = id(norm.weight), id(norm.bias)
    frozen = {name: p.detach().clone() for name, p in net.named_parameters() if "norm" not in name}
    config = AdapterConfig(
        components=[AdapterComponentConfig(targets=[FIRST], rank=2)],
        extra_modules=[AdapterExtraModuleConfig(targets=["model.language_model.norm"])],
    )
    install_adapters(net, config)
    assert (id(norm.weight), id(norm.bias)) == original_ids
    assert norm.weight.dtype == norm.bias.dtype == torch.float32
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.001)
    assert_trainability(net, optimizer)
    before = norm.weight.detach().clone(), norm.bias.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(1, 5, 1152, device=device, dtype=torch.bfloat16, requires_grad=True)
        direction = torch.randn_like(inputs)
        reference_input = inputs.detach().double().requires_grad_()
        reference_weight = norm.weight.detach().double().requires_grad_()
        reference_bias = norm.bias.detach().double().requires_grad_()
        # FP64 reference is independent of the installed FP32 execution path.
        expected = torch.nn.functional.layer_norm(reference_input, (1152,), reference_weight, reference_bias, norm.eps)
        actual = checkpoint(norm, inputs, use_reentrant=False) if checkpointing else norm(inputs)
        assert actual.dtype == inputs.dtype
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.015625, rtol=0.008)
        (actual.float() * direction.float()).sum().backward()
        (expected * direction.double()).sum().backward()
        for actual_gradient, expected_gradient in [
            (norm.weight.grad, reference_weight.grad),
            (norm.bias.grad, reference_bias.grad),
        ]:
            assert actual_gradient is not None and bool(torch.isfinite(actual_gradient).all())
            torch.testing.assert_close(actual_gradient.double(), expected_gradient, atol=2e-5, rtol=2e-5)
        assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
        optimizer.step()
    assert not torch.equal(before[0], norm.weight) and not torch.equal(before[1], norm.bias)
    assert_trainability(net, optimizer)
    assert (id(norm.weight), id(norm.bias)) == original_ids
    for name, expected in frozen.items():
        assert torch.equal(net.state_dict()[name], expected), name


def test_extra_layer_norm_fp32_preserves_stock_forward_and_subclass():
    class OffsetNorm(nn.LayerNorm):
        def forward(self, input):
            return super().forward(input) + 0.125

    net = tiny_net()
    net.model.language_model.norm = OffsetNorm(8)
    stock = copy.deepcopy(net.model.language_model.norm)
    config = AdapterConfig(
        components=[AdapterComponentConfig(targets=[FIRST], rank=2)],
        extra_modules=[AdapterExtraModuleConfig(targets=["model.language_model.norm"])],
    )
    install_adapters(net, config)
    values = torch.randn(2, 8)
    torch.testing.assert_close(net.model.language_model.norm(input=values), stock(values), atol=0, rtol=0)
