# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pure generalized advantage estimation over PPO learning positions."""

from __future__ import annotations

import math

import torch

from cosmos_framework.posttrain.algorithm.primitives.numerics import validate_finite


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:  # rewards/mask: [B,T], values: [B,T+1], modes: [B], returns [B,T] x2
    """Compute masked GAE with distinct termination and truncation bootstrap semantics.

    Valid actions must form a non-empty prefix in every row. A true environment
    termination zeroes the critic bootstrap at the final action; a time-limit
    truncation uses ``values[b, final_action + 1]``. Neither boundary propagates
    a nonexistent next-action advantage through the GAE recurrence.
    """
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be finite and in [0, 1], got {gae_lambda}")
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [B,T], got {tuple(rewards.shape)}")
    batch_size, time_steps = rewards.shape
    if batch_size == 0 or time_steps == 0:
        raise ValueError("compute_gae does not accept an empty batch or time dimension")
    if values.shape != (batch_size, time_steps + 1):
        raise ValueError(f"values shape must be [B,T+1]={batch_size, time_steps + 1}, got {tuple(values.shape)}")
    if action_mask.shape != rewards.shape:
        raise ValueError(
            f"action_mask shape must match rewards, got {tuple(action_mask.shape)} and {tuple(rewards.shape)}"
        )
    if terminated.shape != (batch_size,) or truncated.shape != (batch_size,):
        raise ValueError(
            f"terminated and truncated must have shape [B]={(batch_size,)}, got "
            f"{tuple(terminated.shape)} and {tuple(truncated.shape)}"
        )
    if action_mask.dtype is not torch.bool or terminated.dtype is not torch.bool or truncated.dtype is not torch.bool:
        raise TypeError("action_mask, terminated, and truncated must be boolean tensors")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("rewards and values must be floating-point tensors")
    if rewards.device != values.device or rewards.device != action_mask.device:
        raise ValueError("rewards, values, and action_mask must be on the same device")
    if terminated.device != rewards.device or truncated.device != rewards.device:
        raise ValueError("terminated and truncated must be on the same device as rewards")
    validate_finite(rewards, "rewards")
    validate_finite(values, "values")
    exactly_one_mode = terminated ^ truncated  # [B]
    if not bool(exactly_one_mode.all().item()):
        raise ValueError("exactly one of terminated or truncated must be true for every row")
    selected_per_row = action_mask.sum(dim=1)  # [B]
    if bool((selected_per_row == 0).any().item()):
        raise ValueError("action_mask selected zero actions for at least one row")
    follows_padding = action_mask[:, 1:] & ~action_mask[:, :-1]  # [B,T-1]
    if bool(follows_padding.any().item()):
        raise ValueError("valid actions must form a contiguous prefix in every action_mask row")

    advantages = torch.zeros_like(rewards)  # [B,T]
    next_advantage = torch.zeros_like(rewards[:, 0])  # [B]
    false_vector = torch.zeros_like(action_mask[:, 0])  # [B]
    for step in range(time_steps - 1, -1, -1):
        valid = action_mask[:, step]  # [B]
        next_valid = action_mask[:, step + 1] if step + 1 < time_steps else false_vector  # [B]
        boundary = valid & ~next_valid  # [B]
        bootstrap = next_valid | (boundary & truncated)  # [B]
        bootstrap_weight = bootstrap.to(dtype=values.dtype)  # [B]
        continuation_weight = next_valid.to(dtype=values.dtype)  # [B]
        delta = rewards[:, step] + gamma * values[:, step + 1] * bootstrap_weight - values[:, step]  # [B]
        current_advantage = delta + gamma * gae_lambda * continuation_weight * next_advantage  # [B]
        zeros = torch.zeros_like(current_advantage)  # [B]
        current_advantage = torch.where(valid, current_advantage, zeros)  # [B]
        advantages[:, step] = current_advantage  # [B]
        next_advantage = current_advantage  # [B]

    value_prefix = values[:, :-1]  # [B,T]
    unmasked_returns = advantages + value_prefix  # [B,T]
    return_zeros = torch.zeros_like(unmasked_returns)  # [B,T]
    returns = torch.where(action_mask, unmasked_returns, return_zeros)  # [B,T]
    validate_finite(advantages, "GAE advantages")
    validate_finite(returns, "GAE returns")
    return advantages, returns
