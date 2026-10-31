# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Controlled reload must use the same numerical execution profile as training."""

import pytest
import torch

from cosmos_framework.posttrain.tests.adapter_numerics import configure_qualification_numerics


@pytest.fixture
def numerical_settings(monkeypatch):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        yield
    finally:
        deterministic, warn_only, precision, tf32, benchmark = previous
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = benchmark


def test_controlled_profile_rejects_incompatible_workspace(numerical_settings, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    with pytest.raises(ValueError, match="before launching the process"):
        configure_qualification_numerics()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native grouped MoE requires a CUDA GPU")
def test_controlled_native_moe_forward_and_backward_are_repeatable(numerical_settings):
    from cosmos_framework.model.generator.moe.native import grouped_mm

    torch.manual_seed(42)
    configure_qualification_numerics()
    experts, hidden, intermediate, tokens, top_k = 8, 64, 32, 64, 4
    gate_up = torch.randn(experts, hidden, 2 * intermediate, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    down = torch.randn(experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16) / intermediate**0.5
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores, indices = torch.rand(tokens, experts, device="cuda").topk(top_k, dim=-1)
    scores = (scores / scores.sum(-1, keepdim=True)).to(x.dtype).requires_grad_()
    mask = torch.arange(tokens, device="cuda") < tokens - 7
    counts = torch.bincount(indices[mask].flatten(), minlength=experts).to(torch.int32)
    records = []
    for _ in range(3):
        out = grouped_mm(gate_up, down, torch.nn.functional.silu, x, scores, indices, counts, mask)
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[~mask]) == 0
        gradients = torch.autograd.grad(out.float().square().mean(), (x, scores))
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradients)
        assert all(torch.count_nonzero(g[~mask]) == 0 for g in gradients)
        records.append((out.detach(), *gradients))
    for record in records[1:]:
        for actual, expected in zip(record, records[0], strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
