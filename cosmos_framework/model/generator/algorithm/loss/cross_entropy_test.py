# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import (
    cross_entropy_loss,
    weighted_cross_entropy_loss,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _expected(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1].float()  # [B,T-1,V]
    shifted_labels = labels[:, 1:]  # [B,T-1]
    losses = F.cross_entropy(
        shifted_logits.flatten(0, 1),
        shifted_labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)  # [B,T-1]
    valid = shifted_labels != -100  # [B,T-1]
    return (losses * valid).sum(dim=1), valid.sum(dim=1)


def test_plain_ce_exposes_per_sample_statistics() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 11)  # [B,T,V]
    labels = torch.tensor([[0, 1, 2, -100, 4], [5, 6, -100, 8, 9]])  # [B,T]
    _, stats = cross_entropy_loss(logits, labels, return_stats=True)
    expected_sums, expected_counts = _expected(logits, labels)  # [B], [B]

    torch.testing.assert_close(stats.per_sample_token_ce_sum, expected_sums)
    torch.testing.assert_close(stats.per_sample_valid_token_count, expected_counts)
    torch.testing.assert_close(stats.per_sample_objective_numerator, expected_sums)
    torch.testing.assert_close(stats.per_sample_objective_denominator, expected_counts)


def test_weighted_ce_exposes_per_sample_statistics() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 13)  # [B,T,V]
    labels = torch.tensor([[0, 1, 2, 3, 4], [5, 6, -100, -100, 9]])  # [B,T]
    exponent = 0.5
    _, stats = weighted_cross_entropy_loss(logits, labels, exponent=exponent, return_stats=True)
    expected_sums, expected_counts = _expected(logits, labels)  # [B], [B]

    torch.testing.assert_close(stats.per_sample_token_ce_sum, expected_sums)
    torch.testing.assert_close(stats.per_sample_valid_token_count, expected_counts)
    torch.testing.assert_close(
        stats.per_sample_objective_numerator,
        expected_sums / expected_counts.float().pow(exponent),
    )
    torch.testing.assert_close(
        stats.per_sample_objective_denominator,
        expected_counts.float().pow(1 - exponent),
    )
