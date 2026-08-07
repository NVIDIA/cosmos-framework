# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import (
    cross_entropy_loss,
    weighted_cross_entropy_loss,
)


def test_cross_entropy_returns_token_numerator_and_denominator() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, -100], [0, 2, 3, 4]])
    loss, numerator, denominator = cross_entropy_loss(logits, labels, return_stats=True)
    shifted_logits = logits[:, :-1].reshape(-1, 5)
    shifted_labels = labels[:, 1:].reshape(-1)
    expected = F.cross_entropy(shifted_logits.float(), shifted_labels, ignore_index=-100, reduction="none")
    valid = shifted_labels != -100
    assert denominator.item() == int(valid.sum())
    torch.testing.assert_close(numerator, expected[valid].sum())
    torch.testing.assert_close(loss, numerator / denominator)


def test_weighted_objective_keeps_unweighted_token_stats() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[0, 1, -100, -100], [0, 2, 3, 4]])
    objective, numerator, denominator = weighted_cross_entropy_loss(
        logits, labels, exponent=1.0, return_stats=True
    )
    assert objective.ndim == 0
    assert denominator.item() == 4
    assert numerator.item() > 0
