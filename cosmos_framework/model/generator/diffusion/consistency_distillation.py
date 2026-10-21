# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared helpers for discrete consistency distillation."""

from __future__ import annotations

import torch

from cosmos_framework.model.generator.distillation.common_loss import (
    compute_endpoint_consistency_loss as compute_endpoint_consistency_loss,
)


def shift_rectified_flow_time(
    sigma_unshifted: torch.Tensor,  # [B,N] or [B]
    shift: torch.Tensor | float,  # [B] or []
) -> torch.Tensor:  # [B,N] or [B]
    """Apply the Cosmos RF time-shift formula to unshifted sigma values."""
    shift_tensor = torch.as_tensor(shift, dtype=sigma_unshifted.dtype, device=sigma_unshifted.device)  # [B] or []
    if shift_tensor.ndim == 1 and sigma_unshifted.ndim == 2:
        shift_tensor = shift_tensor.unsqueeze(1)  # [B,1]
    return shift_tensor * sigma_unshifted / (1.0 + (shift_tensor - 1.0) * sigma_unshifted)  # [B,N] or [B]


def sample_dcm_ladder(
    batch_size: int,
    total_steps: int,
    skipping_interval_steps: int,
    shift: torch.Tensor | float,  # [B] or []
    max_timestep: int,
    tensor_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the shifted discrete RF ladder used by dCM."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if skipping_interval_steps <= 0:
        raise ValueError(f"skipping_interval_steps must be positive, got {skipping_interval_steps}")
    if skipping_interval_steps >= total_steps:
        raise ValueError(
            f"skipping_interval_steps must be smaller than total_steps, got {skipping_interval_steps} >= {total_steps}"
        )

    device = tensor_kwargs.get("device", torch.device("cpu"))
    dtype = tensor_kwargs.get("dtype", torch.float32)
    du = 1.0 / float(total_steps)
    max_u = 1.0 - float(skipping_interval_steps) * du
    u = torch.rand((batch_size, 1), device=device, dtype=dtype) * max_u  # [B,1]
    k = torch.arange(skipping_interval_steps + 1, device=device, dtype=dtype).unsqueeze(0)  # [1,N+1]
    sigma_unshifted = 1.0 - (u + k * du)  # [B,N+1]
    sigmas = shift_rectified_flow_time(sigma_unshifted, shift).clamp(0.0, 1.0)  # [B,N+1]
    timesteps = sigmas * float(max_timestep)  # [B,N+1]
    return timesteps, sigmas


def vision_sigma_masks(
    sigmas: torch.Tensor,  # [B_items,1]
    condition_mask: list[torch.Tensor],  # list of [T,1,1]
) -> list[torch.Tensor]:
    """Broadcast sample sigmas over generated vision tokens only."""
    sigma_masks = []
    for i, condition_mask_i in enumerate(condition_mask):
        noisy_mask = 1.0 - condition_mask_i.to(device=sigmas.device, dtype=sigmas.dtype)  # [T,1,1]
        sigma_i = sigmas[i].view(1, 1, 1).expand_as(noisy_mask)  # [T,1,1]
        sigma_masks.append(sigma_i * noisy_mask)  # [T,1,1]
    return sigma_masks  # list of [T,1,1]


def ode_step_with_sigma_masks(
    xt: list[torch.Tensor],  # list of [C,T,H,W]
    v_pred: list[torch.Tensor],  # list of [C,T,H,W]
    sigma_cur: list[torch.Tensor],  # list of [T,1,1]
    sigma_next: list[torch.Tensor],  # list of [T,1,1]
) -> list[torch.Tensor]:
    """Advance noised vision tokens with one Euler step."""
    return [
        xt_i.float()  # [C,T,H,W]
        + (
            sigma_next_i.to(device=xt_i.device, dtype=torch.float32)  # [T,1,1]
            - sigma_cur_i.to(device=xt_i.device, dtype=torch.float32)  # [T,1,1]
        )
        * v_i.float()  # [C,T,H,W]
        for xt_i, v_i, sigma_cur_i, sigma_next_i in zip(xt, v_pred, sigma_cur, sigma_next)
    ]  # list of [C,T,H,W]


def assert_matching_vision_condition_masks(
    reference: list[torch.Tensor],  # list of [T,1,1]
    candidate: list[torch.Tensor],  # list of [T,1,1]
) -> None:
    """Validate that endpoint packs supervise the same vision frames."""
    if len(reference) != len(candidate):
        raise ValueError(
            "dCM endpoint packs must have the same number of vision condition masks; "
            f"got {len(reference)} and {len(candidate)}."
        )
    for idx, (reference_i, candidate_i) in enumerate(zip(reference, candidate)):
        if reference_i.shape != candidate_i.shape or not torch.equal(reference_i, candidate_i):
            raise ValueError(f"dCM endpoint condition masks diverged for vision item {idx}.")
