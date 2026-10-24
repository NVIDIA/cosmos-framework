# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Protect small head-gradient contributions across packed conversations."""

from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.algorithm.loss import cross_entropy as ce

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_head_accumulation_preserves_small_contributions_and_shares_storage(
    monkeypatch: pytest.MonkeyPatch, recompute: bool, weight_dtype: torch.dtype
) -> None:
    storage: set[int] = set()

    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        storage.add(weight.untyped_storage().data_ptr())
        assert weight.dtype == torch.bfloat16
        return F.cross_entropy(F.linear(hidden, weight).float(), labels, ignore_index=ignore_index, reduction="sum")

    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    # All four logits are zero, so their derivatives are exactly 1/4 or -3/4.
    # The large positive/negative terms cancel; the 62 small terms must survive.
    amplitudes = [1024.0] + [1.0] * 31 + [-1024.0] + [1.0] * 31
    hidden = torch.zeros(1, 128, 2, dtype=torch.bfloat16)
    hidden[0, ::2, 0] = torch.tensor(amplitudes, dtype=torch.bfloat16)
    hidden[0, ::2, 1] = 1
    labels = torch.tensor([[-100, 1] * 64])
    boundaries = torch.arange(0, 129, 2, dtype=torch.int32)
    gradients: dict[bool, torch.Tensor] = {}
    for promoted in (False, True):
        storage.clear()
        weight = torch.zeros(4, 2, dtype=weight_dtype, requires_grad=True)
        loss, _ = ce.fused_weighted_cross_entropy_loss(
            hidden,
            labels,
            weight,
            exponent=0.5,
            cu_seq_lens=boundaries,
            recompute=recompute,
            accumulate_head_grad_fp32=promoted,
        )
        loss.backward()
        assert weight.grad is not None
        gradients[promoted] = weight.grad.detach().clone()
        # Recompute may revisit a conversation, but never allocates a head per conversation.
        assert len(storage) == 1
    expected = (
        torch.tensor([0.25, -0.75, 0.25, 0.25], dtype=torch.float64)[:, None]
        * torch.tensor([62.0 / 64, 1.0], dtype=torch.float64)[None, :]
    )
    torch.testing.assert_close(gradients[True], expected.to(weight_dtype), rtol=0, atol=0)
    assert not torch.equal(gradients[False], gradients[True])


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_head_accumulation_preserves_forward_statistics_and_hidden_gradient(
    monkeypatch: pytest.MonkeyPatch, recompute: bool, dtype: torch.dtype, all_ignored: bool
) -> None:
    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return F.cross_entropy(F.linear(hidden, weight).float(), labels, ignore_index=ignore_index, reduction="sum")

    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    torch.manual_seed(519)
    initial_hidden = torch.randn(1, 16, 4, dtype=dtype)
    initial_weight = torch.randn(8, 4, dtype=dtype)
    labels = torch.tensor([[-100, 1, 2, 3, -100, -100, -100, -100, -100, 2, 3, 1, -100, 6, 5, 7]])
    if all_ignored:
        labels.fill_(-100)
    boundaries = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
    results: list[tuple[torch.Tensor, ce.LossStatistics, torch.Tensor]] = []
    for promoted in (False, True):
        hidden = initial_hidden.clone().requires_grad_()
        weight = initial_weight.clone().requires_grad_()
        loss, statistics = ce.fused_weighted_cross_entropy_loss(
            hidden,
            labels,
            weight,
            exponent=0.5,
            cu_seq_lens=boundaries,
            recompute=recompute,
            accumulate_head_grad_fp32=promoted,
        )
        (loss * 0.25).backward()
        assert hidden.grad is not None and weight.grad is not None
        if all_ignored:
            assert torch.count_nonzero(weight.grad) == 0
        results.append((loss.detach(), statistics, hidden.grad))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)
    for field in fields(ce.LossStatistics):
        torch.testing.assert_close(
            getattr(results[0][1], field.name), getattr(results[1][1], field.name), rtol=0, atol=0
        )


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_head_accumulation_does_not_promote_a_frozen_head(monkeypatch: pytest.MonkeyPatch, grad_enabled: bool) -> None:
    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        assert not weight.requires_grad
        return F.cross_entropy(F.linear(hidden, weight).float(), labels, ignore_index=ignore_index, reduction="sum")

    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    hidden = torch.ones(1, 4, 2, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(4, 2, dtype=torch.bfloat16)
    with torch.set_grad_enabled(grad_enabled):
        loss, _ = ce.fused_weighted_cross_entropy_loss(
            hidden,
            torch.tensor([[-100, 1, -100, 2]]),
            weight,
            exponent=0.5,
            cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
            accumulate_head_grad_fp32=True,
        )
        assert loss.requires_grad == grad_enabled
        if grad_enabled:
            loss.backward()
            assert hidden.grad is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp32_anchor_has_constant_forward_storage_and_preserves_full_gradient(dtype: torch.dtype) -> None:
    weight = torch.ones(512, 64, dtype=dtype, requires_grad=True)
    anchor = ce._FP32GradientAnchor.apply(weight)
    assert anchor.shape == weight.shape and anchor.dtype == torch.float32
    assert anchor.untyped_storage().nbytes() == torch.tensor(0, dtype=torch.float32).element_size()
    gradient = torch.linspace(-1, 1, weight.numel(), dtype=torch.float32).reshape_as(weight)
    anchor.backward(gradient)
    assert weight.grad is not None
    torch.testing.assert_close(weight.grad, gradient.to(dtype), rtol=0, atol=0)
