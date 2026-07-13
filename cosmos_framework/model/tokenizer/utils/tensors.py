# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tensor operation helpers shared by tokenizer training and inference."""

from collections.abc import Sequence

import torch

# PyTorch's CUDA multi-input cat path issued an illegal memory access on
# Blackwell with 130 and 194 views, while 98 passed. Keep a safety margin.
MAX_CAT_INPUTS_PER_CALL = 64


def _normalize_tensor_sequence(
    tensors: Sequence[torch.Tensor],
) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
    """Return a PyTorch-compatible tensor sequence without copying common inputs."""
    if isinstance(tensors, (list, tuple)):
        return tensors
    return list(tensors)


def cat_with_bounded_inputs(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate tensors without exceeding the CUDA multi-input kernel's safe fan-in."""
    tensor_inputs = _normalize_tensor_sequence(tensors)
    if len(tensor_inputs) <= MAX_CAT_INPUTS_PER_CALL:
        return torch.cat(tensor_inputs, dim=dim)  # [*cat_shape]

    current_level = list(tensor_inputs)
    first_device = current_level[0].device
    if any(tensor.device != first_device for tensor in current_level[1:]):
        devices = ", ".join(sorted({str(tensor.device) for tensor in current_level}))
        raise RuntimeError(
            "cat_with_bounded_inputs requires tensors to share one device when the input count "
            f"exceeds {MAX_CAT_INPUTS_PER_CALL}; got devices: {devices}"
        )

    target_dtype = current_level[0].dtype
    for tensor in current_level[1:]:
        target_dtype = torch.promote_types(target_dtype, tensor.dtype)
    if any(tensor.dtype != target_dtype for tensor in current_level):
        current_level = [tensor.to(dtype=target_dtype) for tensor in current_level]  # [*input_shapes]

    while len(current_level) > MAX_CAT_INPUTS_PER_CALL:
        # Each partial preserves all dimensions except for the concatenated axis.
        current_level = [
            torch.cat(current_level[start : start + MAX_CAT_INPUTS_PER_CALL], dim=dim)
            for start in range(0, len(current_level), MAX_CAT_INPUTS_PER_CALL)
        ]  # [*partial_cat_shapes]
    return torch.cat(current_level, dim=dim)  # [*cat_shape]


def stack_with_bounded_inputs(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stack tensors without exposing the underlying cat kernel to unbounded fan-in."""
    tensor_inputs = _normalize_tensor_sequence(tensors)
    if len(tensor_inputs) <= MAX_CAT_INPUTS_PER_CALL:
        return torch.stack(tensor_inputs, dim=dim)  # [*stack_shape]
    expanded_inputs = [tensor.unsqueeze(dim) for tensor in tensor_inputs]  # [*expanded_input_shapes]
    return cat_with_bounded_inputs(expanded_inputs, dim=dim)  # [*stack_shape]


__all__ = ["MAX_CAT_INPUTS_PER_CALL", "cat_with_bounded_inputs", "stack_with_bounded_inputs"]
