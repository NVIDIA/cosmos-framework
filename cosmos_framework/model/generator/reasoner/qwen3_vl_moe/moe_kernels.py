# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Legacy padding utilities; native kernels are owned by the shared MoE package."""

from collections.abc import Callable

import torch

from cosmos_framework.model.generator.moe.native_kernels import (
    TOKEN_GROUP_ALIGN_SIZE_M,
    ValidTokenGroupAlignmentSize,
    _fill_indices_kernel,
    _fill_indices_wrapper,
    _generate_permute_indices,
)

__all__ = [
    "TOKEN_GROUP_ALIGN_SIZE_M",
    "ValidTokenGroupAlignmentSize",
    "_fill_indices_kernel",
    "_fill_indices_wrapper",
    "_generate_permute_indices",
    "_permute",
    "_unpermute",
    "indices_padding_wrapper",
]


def _permute(
    x: torch.Tensor,  # [T,D]
    num_tokens_per_expert: torch.Tensor,  # [E]
    num_experts: int,
    alignment: int = TOKEN_GROUP_ALIGN_SIZE_M,
) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:  # shape, [padded_len,D], [padded_len], [E]
    x_padded_size = x.shape[0] + num_experts * alignment
    padded_max_len = ((x_padded_size + alignment - 1) // alignment) * alignment

    with torch.no_grad():
        (
            permuted_indices,
            padded_num_tokens_per_expert,
        ) = _generate_permute_indices(
            num_tokens_per_expert=num_tokens_per_expert,
            num_experts=num_experts,
            max_len=padded_max_len,
            alignment=alignment,
        )  # [padded_len], [E]

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))  # [T+1,D]
    input_shape = x.shape
    x = x[permuted_indices, :]  # [padded_len,D]

    return input_shape, x, permuted_indices, padded_num_tokens_per_expert


def _unpermute(
    out: torch.Tensor,  # [padded_len,D]
    input_shape: torch.Size,
    permuted_indices: torch.Tensor,  # [padded_len]
) -> torch.Tensor:  # [T,D]
    out_unpermuted = out.new_empty(input_shape)  # [T+1,D]
    out_unpermuted[permuted_indices, :] = out  # [padded_len,D]
    return out_unpermuted[:-1]  # [T,D]


def indices_padding_wrapper(func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        gate_up_proj: torch.Tensor,  # [E,D,2I]
        down_proj: torch.Tensor,  # [E,I,D]
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,  # [T,D]
        num_tokens_per_expert: torch.Tensor,  # [E]
    ) -> torch.Tensor:  # [T,D]
        num_experts = num_tokens_per_expert.shape[0]

        input_shape, x, permuted_indices, padded_num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, num_experts
        )  # shape, [padded_len,D], [padded_len], [E]

        out = func(gate_up_proj, down_proj, act_fn, x, padded_num_tokens_per_expert)  # [padded_len,D]

        out = _unpermute(out, input_shape, permuted_indices)  # [T,D]
        return out

    return wrapper
