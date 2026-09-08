# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(7, 8), (7, 3, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_domain_aware_linear_matches_bmm(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    batch_size, input_size = shape[0], shape[-1]
    output_size = 16
    layer = DomainAwareLinear(input_size=input_size, output_size=output_size, num_domains=3).to(dtype=dtype)
    reference_layer = copy.deepcopy(layer)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    domain_id = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)

    output = layer(x, domain_id)
    weight = reference_layer.fc(domain_id).view(batch_size, input_size, output_size)
    bias = reference_layer.bias(domain_id).view(batch_size, output_size)
    if x.dim() == 2:
        expected = torch.bmm(reference_x.unsqueeze(1), weight).squeeze(1) + bias
    else:
        expected = torch.bmm(reference_x, weight) + bias.unsqueeze(1)
    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    expected.backward(output_gradient)

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    rtol = 1e-5 if dtype == torch.float32 else 2e-1
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(layer.fc.weight.grad, reference_layer.fc.weight.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(layer.bias.weight.grad, reference_layer.bias.weight.grad, atol=atol, rtol=rtol)


@pytest.mark.L0
def test_domain_aware_linear_empty_batch_preserves_autograd() -> None:
    layer = DomainAwareLinear(input_size=8, output_size=16, num_domains=3)
    x = torch.randn(0, 8, requires_grad=True)
    domain_id = torch.empty(0, dtype=torch.long)

    layer(x, domain_id).sum().backward()

    assert x.grad is not None
    assert layer.fc.weight.grad is not None
    assert layer.bias.weight.grad is not None
    torch.testing.assert_close(layer.fc.weight.grad, torch.zeros_like(layer.fc.weight))
    torch.testing.assert_close(layer.bias.weight.grad, torch.zeros_like(layer.bias.weight))


@pytest.mark.L1
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Grouped matrix multiplication requires CUDA")
@pytest.mark.parametrize("shape", [(31, 64), (31, 3, 64)])
def test_domain_aware_linear_cuda_matches_linear(shape: tuple[int, ...]) -> None:
    batch_size, input_size = shape[0], shape[-1]
    output_size = 64
    num_domains = 32
    layer = DomainAwareLinear(input_size=input_size, output_size=output_size, num_domains=num_domains).cuda().bfloat16()
    reference_layer = copy.deepcopy(layer)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    active_domains = torch.tensor([0, 7, 31], device="cuda")
    domain_id = active_domains[torch.arange(batch_size, device="cuda") % active_domains.shape[0]]

    output = layer(x, domain_id)
    expected = torch.empty_like(output)
    for domain in active_domains.unbind():
        mask = domain_id == domain
        weight = reference_layer.fc.weight[domain].view(input_size, output_size)
        expected[mask] = F.linear(reference_x[mask], weight.T, reference_layer.bias.weight[domain])
    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    expected.backward(output_gradient)

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-1)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-2, rtol=2e-1)
    torch.testing.assert_close(layer.fc.weight.grad, reference_layer.fc.weight.grad, atol=4e-2, rtol=2e-1)
    torch.testing.assert_close(layer.bias.weight.grad, reference_layer.bias.weight.grad, atol=1e-1, rtol=2e-1)
    absent_domains = torch.tensor(sorted(set(range(num_domains)) - set(active_domains.tolist())), device="cuda")
    torch.testing.assert_close(
        layer.fc.weight.grad[absent_domains], torch.zeros_like(layer.fc.weight.grad[absent_domains])
    )
    torch.testing.assert_close(
        layer.bias.weight.grad[absent_domains], torch.zeros_like(layer.bias.weight.grad[absent_domains])
    )


@pytest.mark.L1
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Grouped matrix multiplication requires CUDA")
@pytest.mark.parametrize("shape", [(31, 64), (47, 3, 64)])
def test_domain_aware_linear_cuda_compiles_fullgraph(shape: tuple[int, ...]) -> None:
    layer = DomainAwareLinear(input_size=64, output_size=64, num_domains=32).cuda().bfloat16()
    compiled_layer = torch.compile(layer, fullgraph=True, dynamic=True)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    active_domains = torch.tensor([0, 7, 31], device="cuda")
    domain_id = active_domains[torch.arange(shape[0], device="cuda") % active_domains.shape[0]]

    output = compiled_layer(x, domain_id)
    output.backward(torch.randn_like(output))

    assert x.grad is not None
    assert layer.fc.weight.grad is not None
    assert layer.bias.weight.grad is not None


@pytest.mark.L0
def test_domain_aware_linear_loads_existing_state_dict() -> None:
    input_size = 8
    output_size = 16
    num_domains = 3
    existing_state_dict = {
        "fc.weight": torch.randn(num_domains, input_size * output_size),
        "bias.weight": torch.randn(num_domains, output_size),
    }
    layer = DomainAwareLinear(input_size=input_size, output_size=output_size, num_domains=num_domains)

    incompatible_keys = layer.load_state_dict(existing_state_dict, strict=True)

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []
    torch.testing.assert_close(layer.fc.weight, existing_state_dict["fc.weight"])
    torch.testing.assert_close(layer.bias.weight, existing_state_dict["bias.weight"])
