# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: local mathematical contracts for the raw top-k-slice divergence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.opd import topk_slice_divergence

# Full-vocabulary logprobs (each row sums to < 1 in probability), 3 tokens x top-4.
STUDENT = torch.tensor(
    [
        [-0.2231435513, -2.3025850930, -2.9957322736, -4.6051701860],
        [-1.2039728043, -1.6094379124, -1.8971199849, -2.3025850930],
        [-0.5108256238, -1.8971199849, -3.5065578973, -4.6051701860],
    ],
    dtype=torch.float32,
)
TEACHER = torch.tensor(
    [
        [-0.1053605157, -3.5065578973, -3.9120230054, -4.6051701860],
        [-2.3025850930, -0.6931471806, -2.3025850930, -3.2188758249],
        [-1.6094379124, -0.9162907319, -2.9957322736, -3.9120230054],
    ],
    dtype=torch.float32,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_beta_endpoints_are_the_two_kl_directions():
    """beta=0 is KL(teacher || student), beta=1 is KL(student || teacher)."""
    student_probs, teacher_probs = STUDENT.exp(), TEACHER.exp()
    forward = (teacher_probs * (TEACHER - STUDENT)).sum(-1)  # KL(teacher || student)
    reverse = (student_probs * (STUDENT - TEACHER)).sum(-1)  # KL(student || teacher)

    assert torch.allclose(topk_slice_divergence(STUDENT, TEACHER, beta=0.0), forward, atol=1e-6)
    assert torch.allclose(topk_slice_divergence(STUDENT, TEACHER, beta=1.0), reverse, atol=1e-6)
    assert not torch.allclose(forward, reverse, atol=1e-3)


@pytest.mark.L0
@pytest.mark.CPU
def test_identical_slices_give_zero_value():
    """Identical raw slices have zero value for every beta."""
    for beta in (0.0, 0.5, 1.0):
        matched_value = topk_slice_divergence(STUDENT, STUDENT.clone(), beta=beta)
        assert torch.allclose(matched_value, torch.zeros(STUDENT.shape[0]), atol=1e-7), (beta, matched_value)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_interior_joins_the_endpoints():
    """Interior beta approaches the same endpoint expressions without a numerical seam."""
    offset = 1e-6
    for beta, endpoint in ((offset, 0.0), (1.0 - offset, 1.0)):
        near = topk_slice_divergence(STUDENT, TEACHER, beta=beta)
        at = topk_slice_divergence(STUDENT, TEACHER, beta=endpoint)
        assert torch.allclose(near, at, atol=1e-4), (beta, near.tolist(), at.tolist())


@pytest.mark.L0
@pytest.mark.CPU
def test_rejects_malformed_input():
    with pytest.raises(ValueError, match=r"jsd_beta must be in \[0, 1\]"):
        topk_slice_divergence(STUDENT, TEACHER, beta=-0.1)
    with pytest.raises(ValueError, match="student and teacher slices must match"):
        topk_slice_divergence(STUDENT, TEACHER[:, :2], beta=1.0)
