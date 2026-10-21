# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-agnostic execution for staged flow-matching objectives."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.posttrain.model.protocol import FlowCondition, FlowConditionMask, FlowMatchingPolicy
from cosmos_framework.posttrain.utils.dist import run_rank_converged


def converged_denoise_error(
    model: FlowMatchingPolicy,
    *,
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: FlowCondition,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    condition_mask: FlowConditionMask,
    context: str,
) -> torch.Tensor:  # x0/epsilon: [B,...], sigma: [B,1], returns [B]
    """Run prepare, distributed forward, and finalization with rank consensus."""

    def prepare() -> Any:
        prepared = model.prepare_denoise_error(
            net=net,
            x0=x0,
            cond=cond,
            sigma=sigma,
            epsilon=epsilon,
            condition_mask=condition_mask,
        )
        if prepared is None:
            raise TypeError(f"{context} prepare_denoise_error returned None")
        return prepared

    prepared = run_rank_converged(prepare, f"{context} forward preparation")

    def predict() -> Any:
        prediction = model.denoise_prediction(net=net, prepared=prepared)
        if prediction is None:
            raise TypeError(f"{context} denoise_prediction returned None")
        return prediction

    prediction = run_rank_converged(predict, f"{context} model forward")

    def finalize() -> torch.Tensor:  # returns [B]
        error = model.finalize_denoise_error(prepared=prepared, prediction=prediction)  # [B]
        if not isinstance(error, torch.Tensor):
            raise TypeError(f"{context} finalize_denoise_error must return a tensor, got {type(error).__name__}")
        if error.ndim != 1:
            raise ValueError(f"{context} finalize_denoise_error must return shape [B], got {tuple(error.shape)}")
        return error

    return run_rank_converged(finalize, f"{context} error finalization")  # [B]
