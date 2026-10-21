# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: normalized top-k-plus-residual-other direct OPD arithmetic."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.opd import (
    TopKPlusOtherDivergence,
    topk_plus_other_divergence,
)
from cosmos_framework.posttrain.schema import TeacherSignal


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_matched_partition_is_a_zero_loss_zero_gradient_fixed_point(beta: float) -> None:
    logits = torch.tensor(  # [N,V]
        [[1.2, 0.4, -0.3, -1.0], [0.1, 0.9, -0.2, -0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )
    support_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)  # [N,K]
    full_logprobs = torch.log_softmax(logits.detach(), dim=-1)  # [N,V]
    teacher_logprobs = full_logprobs.gather(-1, support_ids.to(dtype=torch.long))  # [N,K]
    signal = TeacherSignal(topk_ids=support_ids, topk_logprobs=teacher_logprobs)

    per_token = TopKPlusOtherDivergence(beta).per_token(logits, [signal], torch.device("cpu"))  # [N]
    loss = per_token.sum()  # []
    loss.backward()

    assert torch.allclose(per_token.detach(), torch.zeros_like(per_token), atol=1.0e-7)
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1.0e-7)

    if beta in (0.0, 1.0):
        student = torch.tensor([[0.35, 0.25, 0.40]], dtype=torch.float32)  # [N,K+1]
        teacher = torch.tensor([[0.50, 0.20, 0.30]], dtype=torch.float32)  # [N,K+1]
        expected = (teacher * (teacher.log() - student.log())).sum(-1)  # [N]
        if beta == 1.0:
            expected = (student * (student.log() - teacher.log())).sum(-1)  # [N]
        got = topk_plus_other_divergence(student[:, :2].log(), teacher[:, :2].log(), beta=beta)  # [N]
        assert torch.allclose(got, expected, atol=1.0e-7)

    if beta == 1.0:
        logits.grad = None
        mismatched = TeacherSignal(
            topk_ids=support_ids,
            topk_logprobs=torch.tensor([[0.40, 0.30], [0.40, 0.30]], dtype=torch.float32).log(),  # [N,K]
        )
        TopKPlusOtherDivergence(beta).per_token(logits, [mismatched], torch.device("cpu")).sum().backward()
        assert logits.grad is not None
        assert bool((logits.grad[:, 2:].abs() > 0).all())


@pytest.mark.parametrize(
    ("support_ids", "teacher_logprobs", "message"),
    [
        ([1, 1], [-1.0, -2.0], "duplicate top-k support IDs"),
        ([0, 1], [float("nan"), -2.0], "non-finite"),
        ([0, 1], [0.0, 0.0], "support mass exceeds one"),
    ],
)
@pytest.mark.L0
@pytest.mark.CPU
def test_malformed_teacher_support_fails_closed(
    support_ids: list[int],
    teacher_logprobs: list[float],
    message: str,
) -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)  # [N,V]
    signal = TeacherSignal(
        topk_ids=torch.tensor([support_ids], dtype=torch.int64),  # [N,K]
        topk_logprobs=torch.tensor([teacher_logprobs], dtype=torch.float32),  # [N,K]
    )

    with pytest.raises(ValueError, match=message):
        TopKPlusOtherDivergence(beta=1.0).per_token(logits, [signal], torch.device("cpu"))
