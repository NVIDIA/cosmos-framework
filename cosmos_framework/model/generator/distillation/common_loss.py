# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Loss functions for distribution matching distillation.

Adapted from fastgen/methods/common_loss.py for the Cosmos3 codebase.
Reference: https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/common_loss.py

All loss functions accept per-sample lists of tensors (variable shapes across
samples) to stay consistent with the MoT pipeline convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__: tuple[str, ...] = (
    "VSDLossReduction",
    "variational_score_distillation_loss",
    "variational_score_distillation_loss_from_gradient",
)

VSDLossReduction = str
_VSD_LOSS_REDUCTIONS: set[str] = {"mean", "sum", "sum_rcm"}


def variational_score_distillation_loss(
    gen_data: list[torch.Tensor],
    teacher_x0: list[torch.Tensor],
    fake_score_x0: list[torch.Tensor],
    loss_mask: list[torch.Tensor] | None = None,
    additional_scale: torch.Tensor | None = None,
    reduction: VSDLossReduction = "mean",
) -> torch.Tensor:
    """Compute the variational score distillation (VSD) loss.

    The VSD gradient is ``(fake_score_x0 - teacher_x0) / weight``, where the
    weight normalises the per-sample gradient magnitude. With ``reduction="mean"``,
    the gradient is divided by the active element count. With ``reduction="sum"``,
    the legacy half-MSE sum keeps the gradient independent of the active element
    count. With ``reduction="sum_rcm"``, the pseudo-target loss follows RCM's
    no-half-factor sum and normalizer clamp. Teacher and fake-score are treated
    as constants in all modes.

    All inputs are per-sample lists to support variable shapes across samples.

    Args:
        gen_data: Student-generated data per sample (carries gradient).
        teacher_x0: x0 predictions from the teacher per sample (detached).
        fake_score_x0: x0 predictions from the fake-score per sample (detached).
        loss_mask: Optional per-sample masks, each ``(T_i, 1, 1)`` where
            ``1`` = generated frame (include in loss/weight), ``0`` = conditioned
            frame (exclude).  When provided, both the weight and MSE loss are
            computed only over generated frames to avoid inflation from
            near-zero conditioned differences.
        additional_scale: Optional per-sample scaling tensor of shape ``(B,)``.
        reduction: ``"mean"`` preserves the existing active-element mean loss.
            ``"sum"`` preserves the existing half-MSE per-sample sum.
            ``"sum_rcm"`` uses the RCM-style no-half-factor per-sample sum.

    Returns:
        Scalar VSD loss averaged over samples.
    """
    with torch.no_grad():
        vsd_grad = [fake_score_x0[i] - teacher_x0[i] for i in range(len(gen_data))]  # list of [C,T_i,H_i,W_i]
    return variational_score_distillation_loss_from_gradient(
        gen_data=gen_data,
        vsd_grad=vsd_grad,
        weight_reference=teacher_x0,
        loss_mask=loss_mask,
        additional_scale=additional_scale,
        reduction=reduction,
    )


def variational_score_distillation_loss_from_gradient(
    gen_data: list[torch.Tensor],
    vsd_grad: list[torch.Tensor],
    weight_reference: list[torch.Tensor],
    loss_mask: list[torch.Tensor] | None = None,
    additional_scale: torch.Tensor | None = None,
    reduction: VSDLossReduction = "mean",
) -> torch.Tensor:
    """Compute VSD pseudo-target loss from a caller-provided raw gradient.

    ``vsd_grad`` is the unnormalised gradient direction.  This helper applies the
    same per-sample adaptive normalisation as :func:`variational_score_distillation_loss`,
    using ``weight_reference`` as the teacher-distance reference, and constructs a
    detached pseudo-target. With ``reduction="sum"``, ``d loss / d gen_data``
    is ``vsd_grad * weight``; with ``reduction="sum_rcm"``, it is
    ``2 * vsd_grad * weight`` to match RCM's no-half-factor pseudo-target loss.
    With ``reduction="mean"``, it is additionally divided by the active element
    count.

    Args:
        gen_data: Student-generated data per sample (carries gradient).
        vsd_grad: Raw VSD gradient per sample, before adaptive normalisation.
        weight_reference: Reference tensors used to compute the adaptive weight.
        loss_mask: Optional per-sample masks, each ``(T_i, 1, 1)`` where
            ``1`` = generated frame (include in loss/weight), ``0`` = conditioned
            frame (exclude).
        additional_scale: Optional per-sample scaling tensor of shape ``(B,)``.
        reduction: ``"mean"`` divides each per-sample pseudo-target loss by
            active element count. ``"sum"`` preserves the existing half-MSE sum.
            ``"sum_rcm"`` matches the RCM DMD pseudo-target reduction.

    Returns:
        Scalar VSD loss averaged over samples.
    """
    per_sample_losses = []
    if reduction not in _VSD_LOSS_REDUCTIONS:
        raise ValueError(f"Unknown VSD loss reduction: {reduction}")
    for i in range(len(gen_data)):
        g_i = gen_data[i]  # [C,T_i,H_i,W_i]
        grad_i = vsd_grad[i]  # [C,T_i,H_i,W_i]
        ref_i = weight_reference[i]  # [C,T_i,H_i,W_i]

        with torch.no_grad():
            # Weight calculation in fp32 for numerical stability
            g_fp32 = g_i.float()  # [C,T_i,H_i,W_i]
            ref_fp32 = ref_i.float()  # [C,T_i,H_i,W_i]

            # Mean absolute difference over generated frames only (avoid inflation
            # from conditioned frames where gen ≈ teacher ≈ clean data)
            if loss_mask is not None:
                mask_i = loss_mask[i].expand_as(g_fp32)  # [C,T_i,H_i,W_i]
                diff_abs_mean = ((g_fp32 - ref_fp32).abs() * mask_i).sum() / mask_i.sum().clamp(min=1)  # scalar
            else:
                diff_abs_mean = (g_fp32 - ref_fp32).abs().mean()  # scalar
            if reduction == "sum_rcm":
                w = 1.0 / diff_abs_mean.clamp(min=1e-5)  # scalar
            else:
                w = 1.0 / (diff_abs_mean + 1e-6)  # scalar

            if additional_scale is not None:
                w = w * additional_scale[i].float()  # scalar

            w = w.to(dtype=g_i.dtype)  # scalar

            weighted_grad = grad_i * w  # [C,T_i,H_i,W_i]
            if reduction == "sum_rcm":
                has_nonfinite_weighted_grad = ~torch.isfinite(weighted_grad).all()  # scalar
                weighted_grad = torch.where(
                    has_nonfinite_weighted_grad,
                    torch.zeros_like(weighted_grad),
                    weighted_grad,
                )  # [C,T_i,H_i,W_i]
            pseudo_target = g_i - weighted_grad  # [C,T_i,H_i,W_i]

        if loss_mask is not None:
            mask_i = loss_mask[i].expand_as(g_i)  # [C,T_i,H_i,W_i]
            loss_sq = (g_i - pseudo_target) ** 2 * mask_i  # [C,T_i,H_i,W_i]
            loss_sum = loss_sq.sum()  # scalar
            if reduction == "mean":
                loss_i = 0.5 * loss_sum / mask_i.sum().clamp(min=1)  # scalar
            elif reduction == "sum_rcm":
                loss_i = loss_sum  # scalar
            else:
                loss_i = 0.5 * loss_sum  # scalar
        else:
            if reduction == "mean":
                loss_i = 0.5 * F.mse_loss(g_i, pseudo_target, reduction="mean")  # scalar
            elif reduction == "sum_rcm":
                loss_sq = (g_i - pseudo_target) ** 2  # [C,T_i,H_i,W_i]
                loss_sum = loss_sq.sum()  # scalar
                loss_i = loss_sum  # scalar
            else:
                loss_i = 0.5 * F.mse_loss(g_i, pseudo_target, reduction="sum")  # scalar
        per_sample_losses.append(loss_i)

    return torch.stack(per_sample_losses).mean()  # scalar
