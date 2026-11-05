# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Preserve eager reductions and rounding inside compiled inference blocks."""

import torch


@torch.library.custom_op("cosmos_numerics::native_rms_norm", mutates_args=())
def native_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    if torch.is_grad_enabled() and (hidden_states.requires_grad or weight.requires_grad):
        raise RuntimeError("Native RMSNorm is inference-only; disable autograd")
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states.to(input_dtype)


@native_rms_norm.register_fake
def _native_rms_norm_fake(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.empty_like(hidden_states, dtype=torch.promote_types(hidden_states.dtype, weight.dtype))
