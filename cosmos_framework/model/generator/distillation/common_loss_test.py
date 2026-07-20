# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.distillation import common_loss as common_loss_module
from cosmos_framework.model.generator.distillation.common_loss import (
    variational_score_distillation_loss,
    variational_score_distillation_loss_from_gradient,
)

# ---------------------------------------------------------------------------
# Tests for variational_score_distillation_loss
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_vsd_loss_is_zero_when_fake_equals_teacher():
    """When fake_score_x0 == teacher_x0 the VSD gradient is zero, so loss == 0."""
    rng = torch.Generator()
    rng.manual_seed(0)
    gen = [torch.randn(2, 4, 2, 2, generator=rng)]
    teacher = [torch.randn(2, 4, 2, 2, generator=rng)]
    fake = [teacher[0].clone()]  # identical to teacher

    loss = variational_score_distillation_loss(gen, teacher, fake)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.L0
def test_vsd_loss_positive_when_fake_differs_from_teacher():
    """When fake_score_x0 != teacher_x0 the VSD gradient is non-zero, loss > 0."""
    rng = torch.Generator()
    rng.manual_seed(1)
    gen = [torch.randn(2, 4, 2, 2, generator=rng)]
    teacher = [torch.randn(2, 4, 2, 2, generator=rng)]
    fake = [torch.randn(2, 4, 2, 2, generator=rng)]  # different from teacher

    loss = variational_score_distillation_loss(gen, teacher, fake)
    assert loss.item() > 0.0


@pytest.mark.L0
def test_vsd_additional_scale_applied():
    """additional_scale per sample should scale the VSD gradient accordingly."""
    rng = torch.Generator()
    rng.manual_seed(2)
    gen = [torch.randn(2, 4, 2, 2, generator=rng).requires_grad_(True)]
    teacher = [torch.randn(2, 4, 2, 2, generator=rng)]
    fake = [torch.randn(2, 4, 2, 2, generator=rng)]

    # With scale=1.0 (no scaling)
    loss_no_scale = variational_score_distillation_loss(gen, teacher, fake)

    # With scale=2.0 — pseudo_target shifts proportionally → loss changes
    additional_scale = torch.tensor([2.0])
    loss_scaled = variational_score_distillation_loss(gen, teacher, fake, additional_scale=additional_scale)

    # Scaled loss should differ from unscaled loss (scale affects pseudo_target)
    assert loss_scaled.item() != pytest.approx(loss_no_scale.item(), abs=1e-6)


@pytest.mark.L0
def test_vsd_multi_sample_averaging():
    """Multi-sample batch: loss is the mean over per-sample losses."""
    rng = torch.Generator()
    rng.manual_seed(3)

    # Build two samples with different shapes
    gen0 = torch.randn(2, 2, 2, 2, generator=rng)
    teacher0 = torch.randn(2, 2, 2, 2, generator=rng)
    fake0 = torch.randn(2, 2, 2, 2, generator=rng)

    gen1 = torch.randn(2, 4, 2, 2, generator=rng)
    teacher1 = torch.randn(2, 4, 2, 2, generator=rng)
    fake1 = torch.randn(2, 4, 2, 2, generator=rng)

    # Compute joint and individual losses
    loss_joint = variational_score_distillation_loss([gen0, gen1], [teacher0, teacher1], [fake0, fake1])
    loss0 = variational_score_distillation_loss([gen0], [teacher0], [fake0])
    loss1 = variational_score_distillation_loss([gen1], [teacher1], [fake1])

    expected = (loss0 + loss1) / 2.0
    assert loss_joint.item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.L0
def test_vsd_loss_from_gradient_uses_caller_gradient():
    """The pseudo-target loss should expose the caller-provided gradient on gen_data."""
    gen = [torch.ones(1, 1, 1, 1, requires_grad=True)]  # [C,T,H,W]
    raw_grad = [torch.full((1, 1, 1, 1), 2.0)]  # [C,T,H,W]
    weight_reference = [torch.zeros(1, 1, 1, 1)]  # [C,T,H,W]

    loss = variational_score_distillation_loss_from_gradient(gen, raw_grad, weight_reference)
    loss.backward()

    expected_grad = torch.full((1, 1, 1, 1), 2.0 / (1.0 + 1e-6))  # [C,T,H,W]
    assert torch.allclose(gen[0].grad, expected_grad)


@pytest.mark.L0
def test_vsd_sum_reduction_avoids_active_element_averaging() -> None:
    """Sum reduction should avoid active-element averaging of the DMD gradient."""
    raw_grad = [torch.full((1, 1, 1, 2), 2.0)]  # [C,T,H,W]
    weight_reference = [torch.zeros(1, 1, 1, 2)]  # [C,T,H,W]

    gen_mean = [torch.ones(1, 1, 1, 2, requires_grad=True)]  # [C,T,H,W]
    loss_mean = variational_score_distillation_loss_from_gradient(
        gen_mean, raw_grad, weight_reference, reduction="mean"
    )
    loss_mean.backward()

    gen_sum = [torch.ones(1, 1, 1, 2, requires_grad=True)]  # [C,T,H,W]
    loss_sum = variational_score_distillation_loss_from_gradient(gen_sum, raw_grad, weight_reference, reduction="sum")
    loss_sum.backward()

    expected_sum_grad = torch.full((1, 1, 1, 2), 2.0 / (1.0 + 1e-6))  # [C,T,H,W]
    expected_mean_grad = expected_sum_grad / gen_mean[0].numel()  # [C,T,H,W]
    assert torch.allclose(gen_mean[0].grad, expected_mean_grad)
    assert torch.allclose(gen_sum[0].grad, expected_sum_grad)


@pytest.mark.L0
def test_vsd_sum_rcm_reduction_matches_rcm_no_half_factor() -> None:
    """sum_rcm should use RCM's no-half-factor pseudo-target reduction."""
    raw_grad = [torch.full((1, 1, 1, 2), 2.0)]  # [C,T,H,W]
    weight_reference = [torch.zeros(1, 1, 1, 2)]  # [C,T,H,W]

    gen_sum_rcm = [torch.ones(1, 1, 1, 2, requires_grad=True)]  # [C,T,H,W]
    loss_sum_rcm = variational_score_distillation_loss_from_gradient(
        gen_sum_rcm, raw_grad, weight_reference, reduction="sum_rcm"
    )
    loss_sum_rcm.backward()

    expected_rcm_grad = torch.full((1, 1, 1, 2), 4.0)  # [C,T,H,W]
    assert torch.allclose(gen_sum_rcm[0].grad, expected_rcm_grad)


@pytest.mark.L0
@pytest.mark.parametrize("use_loss_mask", [False, True])
def test_vsd_sum_rcm_zeroes_nan_sample_loss_and_grad(use_loss_mask: bool) -> None:
    """sum_rcm should zero loss and grad when a sample's pseudo-target gradient is NaN."""
    gen_sum_rcm = [torch.ones(1, 1, 1, 1, requires_grad=True)]  # [C,T,H,W]
    raw_grad = [torch.full((1, 1, 1, 1), float("nan"))]  # [C,T,H,W]
    weight_reference = [torch.zeros(1, 1, 1, 1)]  # [C,T,H,W]
    loss_mask = [torch.ones(1, 1, 1)] if use_loss_mask else None  # [T,H,W]

    loss_sum_rcm = variational_score_distillation_loss_from_gradient(
        gen_sum_rcm,
        raw_grad,
        weight_reference,
        loss_mask=loss_mask,
        reduction="sum_rcm",
    )
    loss_sum_rcm.backward()

    assert loss_sum_rcm.item() == pytest.approx(0.0)
    assert torch.allclose(gen_sum_rcm[0].grad, torch.zeros_like(gen_sum_rcm[0]))  # [C,T,H,W]


@pytest.mark.L0
@pytest.mark.CPU
def test_public_loss_exports_are_explicit() -> None:
    assert common_loss_module.__all__ == (
        "VSDLossReduction",
        "variational_score_distillation_loss",
        "variational_score_distillation_loss_from_gradient",
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vsd_loss_preserves_public_dtype_contract(dtype: torch.dtype) -> None:
    gen_data = [torch.ones(1, 2, 1, 2, dtype=dtype, requires_grad=True)]  # [C,T,H,W]
    vsd_grad = [torch.full((1, 2, 1, 2), 2.0, dtype=dtype)]  # [C,T,H,W]
    weight_reference = [torch.zeros(1, 2, 1, 2, dtype=dtype)]  # [C,T,H,W]

    loss = variational_score_distillation_loss_from_gradient(gen_data, vsd_grad, weight_reference)
    loss.backward()

    assert loss.dtype is dtype
    assert gen_data[0].grad is not None
    assert gen_data[0].grad.dtype is dtype
    assert torch.isfinite(loss)
    assert torch.isfinite(gen_data[0].grad).all()


@pytest.mark.L0
@pytest.mark.CPU
def test_vsd_list_inputs_honor_mask_and_mean_sum_reductions() -> None:
    vsd_grad = [torch.full((1, 2, 1, 2), 2.0), torch.ones(1, 1, 1, 1)]  # list of [C,T_i,H_i,W_i]
    weight_reference = [torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 1)]  # list of [C,T_i,H_i,W_i]
    loss_mask = [torch.tensor([[[1.0]], [[0.0]]]), torch.ones(1, 1, 1)]  # list of [T_i,H_i,W_i]

    gen_mean = [
        torch.ones(1, 2, 1, 2, requires_grad=True),
        torch.ones(1, 1, 1, 1, requires_grad=True),
    ]  # list of [C,T_i,H_i,W_i]
    mean_loss = variational_score_distillation_loss_from_gradient(
        gen_mean,
        vsd_grad,
        weight_reference,
        loss_mask=loss_mask,
        reduction="mean",
    )
    mean_loss.backward()

    gen_sum = [
        torch.ones(1, 2, 1, 2, requires_grad=True),
        torch.ones(1, 1, 1, 1, requires_grad=True),
    ]  # list of [C,T_i,H_i,W_i]
    sum_loss = variational_score_distillation_loss_from_gradient(
        gen_sum,
        vsd_grad,
        weight_reference,
        loss_mask=loss_mask,
        reduction="sum",
    )
    sum_loss.backward()

    assert gen_mean[0].grad is not None
    assert gen_sum[0].grad is not None
    assert torch.count_nonzero(gen_mean[0].grad[:, 1:]) == 0
    assert torch.count_nonzero(gen_sum[0].grad[:, 1:]) == 0
    assert torch.allclose(gen_sum[0].grad[:, :1], 2.0 * gen_mean[0].grad[:, :1])


@pytest.mark.L0
@pytest.mark.CPU
def test_vsd_gradient_space_matches_fake_minus_teacher_direction() -> None:
    gen_data = [torch.ones(1, 1, 1, 2, requires_grad=True)]  # [C,T,H,W]
    teacher_x0 = [torch.zeros(1, 1, 1, 2)]  # [C,T,H,W]
    fake_score_x0 = [torch.full((1, 1, 1, 2), 2.0)]  # [C,T,H,W]

    loss = variational_score_distillation_loss(
        gen_data,
        teacher_x0,
        fake_score_x0,
        reduction="sum",
    )
    loss.backward()

    expected_grad = torch.full((1, 1, 1, 2), 2.0 / (1.0 + 1e-6))  # [C,T,H,W]
    assert torch.allclose(gen_data[0].grad, expected_grad)
