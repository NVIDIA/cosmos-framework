# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rectified-flow matching loss (vision / action / sound modalities).

Extracted from OmniMoTModel._compute_flow_matching_loss. The loss math is
unchanged; the only structural change is that ``tensor_kwargs_fp32`` is now
passed explicitly instead of being read from ``self``.
"""

from __future__ import annotations

import torch

from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow


def compute_flow_matching_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    timesteps: torch.Tensor,
    has_valid_tokens: bool,
    rectified_flow: RectifiedFlow,
    tensor_kwargs_fp32: dict,
    raw_action_dim: list[torch.Tensor] | None = None,
    action_valid_mask: list[torch.Tensor] | None = None,
    normalize_by_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow matching loss for a modality.

    Args:
        pred: Predicted velocity field (list of tensors, one per sample).
        target: Target velocity field (list of tensors, one per sample).
            Under rectified flow the target is ``v = eps - x0``.
        condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
        timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
            base/teacher_forcing (all frames share one timestep) or [B,T_max]
            for diffusion_forcing (per-frame independent timesteps). Time weights
            are applied per-frame before averaging, so non-uniform weight functions
            are handled correctly.
        has_valid_tokens: Whether this modality has valid noisy tokens.
        rectified_flow: The rectified flow object for time weighting.
        tensor_kwargs_fp32: Dict of dtype/device kwargs forwarded to
            ``rectified_flow.train_time_weight``.
        action_valid_mask: Optional per-sample channel-validity masks. Invalid
            action channels are excluded from both numerator and denominator.
        normalize_by_active: When True, normalize per-instance loss by the count of
            active (noisy) elements rather than all elements. Preserves the
            ``sum / active_count`` semantics needed for distillation critics where
            conditioned frames contribute no signal and should not dilute the
            denominator.

    Returns:
        tuple: A tuple containing two elements:
            - Flow matching loss (or dummy loss for gradient consistency).
            - Per-instance loss (or dummy loss for gradient consistency).
    """
    if not has_valid_tokens:
        # Dummy loss to maintain backward graph consistency across ranks
        dummy_loss = 0.0 * sum(p.sum() for p in pred)
        return dummy_loss, dummy_loss.unsqueeze(0)  # make per-instance loss 1-D

    # condition_mask[i] is T-first with trailing singletons: [T,1,1] vision, [T,1] action.
    # tw_i gets the same shape so w(σ_t) broadcasts element-wise over non-T dims.
    per_instance_losses = []
    per_instance_weighted_losses = []

    for i in range(len(pred)):
        T_i = condition_mask[i].shape[0]
        sqerr_i = (pred[i] - target[i]) ** 2  # vision:[C,T,H,W]  action/sound:[T,D]
        noisy_mask_i = 1.0 - condition_mask[i]  # vision:[T,1,1]  action/sound:[T,1]
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            sqerr_i = sqerr_i[:, : raw_action_dim[i]]
        slot_mask_i = None
        if action_valid_mask is not None and action_valid_mask[i] is not None:
            slot_mask_i = action_valid_mask[i].to(dtype=sqerr_i.dtype, device=sqerr_i.device).reshape(1, -1)
            if slot_mask_i.shape[-1] < sqerr_i.shape[-1]:
                raise ValueError(f"action_valid_mask width {slot_mask_i.shape[-1]} < action width {sqerr_i.shape[-1]}")
            slot_mask_i = slot_mask_i[:, : sqerr_i.shape[-1]]
            sqerr_i = sqerr_i * slot_mask_i
        if normalize_by_active:
            active_channels = slot_mask_i.sum() if slot_mask_i is not None else sqerr_i.numel() // noisy_mask_i.numel()
            active_count = (noisy_mask_i.sum() * active_channels).clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)  # []
        elif slot_mask_i is not None:
            # Preserve the default ``.mean()`` contract: conditioned timesteps
            # remain in the denominator, while only invalid channels are removed.
            # ``normalize_by_active=True`` intentionally uses noisy timesteps only.
            active_count = (noisy_mask_i.numel() * slot_mask_i.sum()).clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)  # []
        else:
            per_instance_losses.append((sqerr_i * noisy_mask_i).mean())  # []

        ts_i = timesteps[i, :T_i] if timesteps.dim() > 1 else timesteps[i]  # DF:[T_i]  TF:[1]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32)  # DF:[T_i]  TF:[1]
        tw_i = tw_i.reshape(-1, *([1] * (condition_mask[i].ndim - 1)))  # vision:[T_i,1,1]  action/sound:[T_i,1]
        if normalize_by_active or slot_mask_i is not None:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).sum() / active_count)
        else:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).mean())

    per_instance_loss = torch.stack(per_instance_losses)  # [B]
    per_instance_weighted_loss = torch.stack(per_instance_weighted_losses)  # [B]
    return (
        per_instance_weighted_loss.mean(),  # []
        per_instance_loss,  # [B]
    )
