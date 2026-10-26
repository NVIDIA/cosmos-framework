# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed fused loss must preserve independently evaluated conversation objectives."""

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.algorithm.loss import cross_entropy as ce

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_packed_fused_objective_and_gradients_match_separate_conversations(
    monkeypatch: pytest.MonkeyPatch, exponent: float, recompute: bool, all_ignored: bool
) -> None:
    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        # Only replace the CUDA kernel; exercise the real packing/objective/recompute path.
        return F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=ignore_index, reduction="sum")

    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    torch.manual_seed(715)
    lengths = [4, 6, 3, 2]
    hidden = torch.randn(1, sum(lengths), 5, requires_grad=True)
    weight = torch.randn(13, 5, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, -100, 4, -100, 5, 6, 7, -100, -100, -100, -100, -100]])
    if all_ignored:
        labels.fill_(-100)
    boundaries = torch.tensor([0, 4, 10, 13, 15], dtype=torch.int32)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []
    start = 0
    for length in lengths:
        target = labels[0, start + 1 : start + length]
        logits = F.linear(reference_hidden[0, start : start + length - 1], reference_weight)
        count = (target != -100).sum().float()
        numerators.append(
            F.cross_entropy(logits, target, ignore_index=-100, reduction="sum") / count.clamp(min=1).pow(exponent)
        )
        denominators.append(torch.where(count > 0, count.clamp(min=1).pow(1 - exponent), 0.0))
        start += length
    denominator = torch.stack(denominators).sum()
    expected = torch.stack(numerators).sum() / denominator.clamp(min=1)
    actual, stats = ce.fused_weighted_cross_entropy_loss(
        hidden, labels, weight, exponent=exponent, cu_seq_lens=boundaries, recompute=recompute
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(stats.objective_denominator, denominator)
    torch.testing.assert_close(
        ce.weighted_ce_denominator_from_labels(labels, exponent=exponent, cu_seq_lens=boundaries), denominator
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)
