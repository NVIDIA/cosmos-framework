# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Losses for LiDAR TransformerVAE training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The reference objective sums every enabled term and then adds the gradient
# term a second time. Preserve that factor explicitly so the configured
# gradient weight retains its intended strength.
SOURCE_GRADIENT_TERM_COUNT = 2.0


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Mean KL of diagonal Gaussians to N(0, I), averaged over all elements.

    Evaluated in float32 whatever the surrounding autocast dtype: ``exp``
    overflows float16 once ``log_var`` passes 11.09, and the resulting inf
    survives any weight applied to it, so a single wide posterior would poison
    the total loss rather than merely inflating this term.
    """
    mu = mu.float()  # [B,C,T,H,W]
    log_var = log_var.float()  # [B,C,T,H,W]
    return 0.5 * torch.mean(mu.pow(2) + log_var.exp() - 1.0 - log_var)  # []


def masked_point_l1(
    recon_points: torch.Tensor,  # [B,T,H,W,3]
    target_points: torch.Tensor,  # [B,T,H,W,3]
    valid_mask: torch.Tensor,  # [B,1,T,H,W]
) -> torch.Tensor:  # []
    """Mean L1 distance between reconstructed and target sensor-frame XYZ, in metres.

    Supervising unprojected points rather than the range channel alone charges
    a range error by the metric distance it moves a return, so an error on a
    grazing far-field ray costs what it actually displaces in the scene.
    """
    if recon_points.shape != target_points.shape:
        raise ValueError(f"Point clouds must match, got {tuple(recon_points.shape)} and {tuple(target_points.shape)}")
    mask = valid_mask[:, 0].unsqueeze(-1)  # [B,T,H,W,1]
    error = (recon_points - target_points).abs()  # [B,T,H,W,3]
    return (error * mask).mean()  # []


def _gradient_loss_at_scale(
    prediction: torch.Tensor,  # [B,C,T,H,W]
    target: torch.Tensor,  # [B,C,T,H,W]
    mask: torch.Tensor,  # [B,C,T,H,W]
    *,
    beta: float,
) -> torch.Tensor:  # []
    difference = mask * (prediction - target)  # [B,C,T,H,W]
    grad_w = F.smooth_l1_loss(
        difference[..., :, 1:],
        difference[..., :, :-1],
        reduction="none",
        beta=beta,
    )  # [B,C,T,H,W-1]
    grad_h = F.smooth_l1_loss(
        difference[..., 1:, :],
        difference[..., :-1, :],
        reduction="none",
        beta=beta,
    )  # [B,C,T,H-1,W]
    # Keep a finite difference only where both of its rays returned, so a range
    # discontinuity across a dropped ray does not read as scene structure.
    grad_w = grad_w * mask[..., :, 1:] * mask[..., :, :-1]  # [B,C,T,H,W-1]
    grad_h = grad_h * mask[..., 1:, :] * mask[..., :-1, :]  # [B,C,T,H-1,W]
    image_loss = grad_w.sum(dim=(-2, -1)) + grad_h.sum(dim=(-2, -1))  # [B,C,T]
    return image_loss.sum() / mask.sum().clamp_min(1.0)  # []


def multiscale_gradient_loss(
    prediction: torch.Tensor,  # [B,C,T,H,W]
    target: torch.Tensor,  # [B,C,T,H,W]
    mask: torch.Tensor,  # [B,C,T,H,W]
    *,
    num_scales: int = 4,
    beta: float = 1.0,
) -> torch.Tensor:  # []
    """Smooth-L1 penalty on spatial gradients of the masked error, summed over scales.

    Each scale strides the range view by ``2**scale`` in elevation and azimuth,
    so one over-smoothed edge is charged at several receptive-field sizes. An
    L1 range loss alone is indifferent to where its error sits; this term is
    what keeps decoded surfaces from blurring across depth discontinuities.
    """
    if num_scales < 1:
        raise ValueError(f"num_scales must be positive, got {num_scales}")
    total = prediction.new_zeros(())  # []
    for scale in range(num_scales):
        step = 2**scale
        total = total + _gradient_loss_at_scale(
            prediction[..., ::step, ::step],  # [B,C,T,H',W']
            target[..., ::step, ::step],  # [B,C,T,H',W']
            mask[..., ::step, ::step],  # [B,C,T,H',W']
            beta=beta,
        )  # []
    return total  # []


RECON_REDUCTIONS = ("masked_mean", "masked_mean_all_rays", "mean")


class LidarVAELoss(nn.Module):
    """Masked recon + KL + mask BCE for range/intensity/mask TransformerVAE.

    ``point_weight`` and ``gradient_weight`` add geometric supervision to
    range only: metric displacement and depth discontinuity describe the
    geometry, while intensity retains its separate reconstruction objective.

    ``recon_reduction`` picks the denominator of the reconstruction term:

      * ``masked_mean``: per valid ray, ``sum(mask * err) / mask.sum()``.
      * ``masked_mean_all_rays``: per ray, ``(mask * err).mean()``. This is
        proportional to the fraction of rays that returned. Pair it with
        the geometric terms using their configured reductions; changing only
        one denominator changes the objective's balance as ray drop varies.
      * ``mean``: unmasked, over every ray including dropped ones.
    """

    def __init__(
        self,
        *,
        recon_weight: float = 1.0,
        range_weight: float = 1.0,
        intensity_weight: float = 1.0,
        kl_weight: float = 1e-6,
        validity_weight: float = 0.1,
        point_weight: float = 0.0,
        gradient_weight: float = 0.0,
        gradient_scales: int = 4,
        recon_reduction: str = "masked_mean",
    ) -> None:
        super().__init__()
        if recon_reduction not in RECON_REDUCTIONS:
            raise ValueError(f"Unsupported recon_reduction={recon_reduction!r}, expected one of {RECON_REDUCTIONS}")
        if range_weight < 0.0 or intensity_weight < 0.0:
            raise ValueError("Range and intensity reconstruction weights must be nonnegative")
        if range_weight + intensity_weight <= 0.0:
            raise ValueError("At least one reconstruction channel weight must be positive")
        if point_weight < 0.0 or gradient_weight < 0.0:
            raise ValueError("Point and gradient loss weights must be nonnegative")
        if gradient_scales < 1:
            raise ValueError(f"gradient_scales must be positive, got {gradient_scales}")
        self.recon_weight: float = float(recon_weight)
        self.range_weight: float = float(range_weight)
        self.intensity_weight: float = float(intensity_weight)
        self.kl_weight: float = float(kl_weight)
        self.validity_weight: float = float(validity_weight)
        self.point_weight: float = float(point_weight)
        self.gradient_weight: float = float(gradient_weight)
        self.gradient_scales: int = int(gradient_scales)
        self.recon_reduction: str = recon_reduction

    @property
    def needs_points(self) -> bool:
        """Whether the caller must unproject range to XYZ for this loss."""
        return self.point_weight > 0.0

    def forward(
        self,
        *,
        reconstructions: torch.Tensor,
        target: torch.Tensor,
        posteriors: tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.Tensor,
        validity_logits: torch.Tensor | None = None,
        recon_points: torch.Tensor | None = None,
        target_points: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted total loss.

        Args:
            reconstructions: ``[B, 2|3, T, H, W]``. With 3 channels, channel 2 is
                mask logits. With 2 channels, pass ``validity_logits`` separately.
            target: ``[B, 2, T, H, W]`` range + intensity targets.
            posteriors: ``(mu, log_var)`` each ``[B, z, T', H', W']``.
            valid_mask: ``[B, 1, T, H, W]`` in ``{0,1}`` (or broadcastable).
            validity_logits: optional ``[B, 1, T, H, W]`` when reconstructions
                only contain the data channels.
            recon_points: ``[B, T, H, W, 3]`` metric XYZ unprojected from the
                reconstructed range, required when ``point_weight`` is positive.
            target_points: ``[B, T, H, W, 3]`` metric XYZ unprojected from the
                target range, required when ``point_weight`` is positive.
        """
        if reconstructions.shape[1] not in {2, 3}:
            raise ValueError(f"reconstructions must have 2 or 3 channels, got {tuple(reconstructions.shape)}")
        if target.shape[1] != 2:
            raise ValueError(f"target must have 2 channels (range, intensity), got {tuple(target.shape)}")
        if reconstructions.shape[0] != target.shape[0] or reconstructions.shape[2:] != target.shape[2:]:
            raise ValueError(
                f"reconstructions shape {tuple(reconstructions.shape)} incompatible with target {tuple(target.shape)}"
            )

        data_recon = reconstructions[:, :2]  # [B,2,T,H,W]
        mask = valid_mask.to(dtype=reconstructions.dtype)  # [B,1,T,H,W]
        if mask.shape[1] != 1:
            raise ValueError(f"valid_mask must have 1 channel, got {tuple(mask.shape)}")

        range_abs_err = (data_recon[:, :1] - target[:, :1]).abs()  # [B,1,T,H,W]
        intensity_abs_err = (data_recon[:, 1:2] - target[:, 1:2]).abs()  # [B,1,T,H,W]
        if self.recon_reduction == "masked_mean":
            channel_mask = mask.expand_as(range_abs_err)  # [B,1,T,H,W]
            denom = channel_mask.sum().clamp_min(1.0)  # []
            range_recon_loss = (range_abs_err * channel_mask).sum() / denom  # []
            intensity_recon_loss = (intensity_abs_err * channel_mask).sum() / denom  # []
        elif self.recon_reduction == "masked_mean_all_rays":
            channel_mask = mask.expand_as(range_abs_err)  # [B,1,T,H,W]
            range_recon_loss = (range_abs_err * channel_mask).mean()  # []
            intensity_recon_loss = (intensity_abs_err * channel_mask).mean()  # []
        else:
            range_recon_loss = range_abs_err.mean()  # []
            intensity_recon_loss = intensity_abs_err.mean()  # []
        channel_weight = self.range_weight + self.intensity_weight
        recon_loss = (
            self.range_weight * range_recon_loss + self.intensity_weight * intensity_recon_loss
        ) / channel_weight  # []

        mu, log_var = posteriors  # each [B,C,T,h,w]
        kl_loss = kl_divergence(mu, log_var)  # []

        if validity_logits is None and reconstructions.shape[1] >= 3:
            validity_logits = reconstructions[:, 2:3]  # [B,1,T,H,W]
        validity_loss = reconstructions.new_zeros(())  # []
        if validity_logits is not None:
            validity_loss = F.binary_cross_entropy_with_logits(
                validity_logits,
                mask.expand_as(validity_logits).clamp(0.0, 1.0),
            )  # []

        point_loss = reconstructions.new_zeros(())  # []
        if self.point_weight > 0.0:
            if recon_points is None or target_points is None:
                raise ValueError("point_weight is positive but recon_points/target_points were not provided")
            point_loss = masked_point_l1(recon_points, target_points, mask)  # []

        gradient_loss = reconstructions.new_zeros(())  # []
        if self.gradient_weight > 0.0:
            gradient_loss = multiscale_gradient_loss(
                data_recon[:, :1],  # [B,1,T,H,W]
                target[:, :1],  # [B,1,T,H,W]
                mask,  # [B,1,T,H,W]
                num_scales=self.gradient_scales,
            )  # []

        total = (
            self.recon_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.validity_weight * validity_loss
            + self.point_weight * point_loss
            + SOURCE_GRADIENT_TERM_COUNT * self.gradient_weight * gradient_loss
        )  # []
        metrics: dict[str, torch.Tensor] = {
            "loss": total.detach(),
            "loss_recon": recon_loss.detach(),
            "loss_range_recon": range_recon_loss.detach(),
            "loss_intensity_recon": intensity_recon_loss.detach(),
            "loss_kl": kl_loss.detach(),
            "loss_validity": validity_loss.detach(),
            "loss_point": point_loss.detach(),
            "loss_gradient": gradient_loss.detach(),
        }
        return total, metrics

    def extra_repr(self) -> str:
        return (
            f"recon_weight={self.recon_weight}, range_weight={self.range_weight}, "
            f"intensity_weight={self.intensity_weight}, kl_weight={self.kl_weight}, "
            f"validity_weight={self.validity_weight}, point_weight={self.point_weight}, "
            f"gradient_weight={self.gradient_weight}, gradient_scales={self.gradient_scales}, "
            f"recon_reduction={self.recon_reduction!r}"
        )
