# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Adapter arithmetic contracts; CPU tests do not qualify CUDA performance."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn import functional as F

from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _adapter(compute_dtype):
    base = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    adapter = LoraInjectedLinear(base, 4, 8, adapter_dtype=torch.float32, adapter_compute_dtype=compute_dtype)
    adapter.reset_adapter_parameters(generator=torch.Generator().manual_seed(84))
    return adapter


@pytest.mark.parametrize("compute_dtype", [None, torch.float32])
def test_default_adapter_arithmetic_remains_exact(compute_dtype):
    adapter = _adapter(compute_dtype)
    with torch.no_grad():
        adapter.lora_B.weight.fill_(0.25)
    x = torch.randn(2, 7, 16, dtype=torch.bfloat16)
    expected = F.linear(x, adapter.weight) + 2 * adapter.lora_B(adapter.lora_A(x.float())).bfloat16()
    torch.testing.assert_close(adapter(x), expected, atol=0, rtol=0)


def test_bf16_arithmetic_matches_explicit_reference_and_retains_fp32_leaf_gradients():
    adapter = _adapter(torch.bfloat16)
    with torch.no_grad():
        adapter.lora_B.weight.fill_(0.25)
    x = torch.randn(2, 7, 16, dtype=torch.bfloat16).requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_a = adapter.lora_A.weight.detach().clone().requires_grad_(True)
    reference_b = adapter.lora_B.weight.detach().clone().requires_grad_(True)
    expected = F.linear(reference_x, adapter.weight) + 2 * F.linear(
        F.linear(reference_x, reference_a.bfloat16()), reference_b.bfloat16()
    )
    actual = adapter(x)
    actual_gradients = torch.autograd.grad(actual.float().square().mean(), (x, *adapter.adapter_parameters().values()))
    expected_gradients = torch.autograd.grad(expected.float().square().mean(), (reference_x, reference_a, reference_b))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    assert all(gradient.dtype == torch.float32 for gradient in actual_gradients[1:])
    assert adapter.weight.grad is None


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_multiple_updates_keep_fp32_parameters_moments_and_frozen_base(compute_dtype, checkpoint):
    adapter = _adapter(compute_dtype)
    base = adapter.weight.detach().clone()
    embedding = nn.Embedding(10, 16, dtype=torch.bfloat16).requires_grad_(False)
    embedded_before = embedding.weight.detach().clone()
    block = checkpoint_wrapper(adapter, preserve_rng_state=True) if checkpoint else adapter
    parameters = adapter.adapter_parameters()
    initial_a, initial_b = (parameter.detach().clone() for parameter in parameters.values())
    optimizer = torch.optim.AdamW(parameters.values(), lr=0.01)
    parameter_ids = {id(parameter) for parameter in parameters.values()}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = block(embedding(torch.tensor([[1, 2, 3]]))).float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in parameters.values())
        assert all(torch.isfinite(p.grad).all() for p in parameters.values())
        assert bool(torch.count_nonzero(adapter.lora_A.weight.grad)) == (step > 0)
        optimizer.step()
        assert all(
            state[key].dtype == torch.float32 for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")
        )
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == parameter_ids
    assert {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad} == parameter_ids
    assert not torch.equal(initial_a, adapter.lora_A.weight)
    assert not torch.equal(initial_b, adapter.lora_B.weight)
    torch.testing.assert_close(adapter.weight, base, atol=0, rtol=0)
    torch.testing.assert_close(embedding.weight, embedded_before, atol=0, rtol=0)
    assert adapter.weight.grad is None and embedding.weight.grad is None


def test_explicit_compute_policy_rejects_unsupported_dtype_or_non_fp32_masters():
    with pytest.raises(ValueError, match="compute dtype"):
        _adapter(torch.float16)
    with pytest.raises(ValueError, match="FP32 adapter parameters"):
        LoraInjectedLinear(nn.Linear(16, 16), 4, 8, adapter_dtype=torch.bfloat16, adapter_compute_dtype=torch.bfloat16)
