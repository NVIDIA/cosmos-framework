# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Literal

import torch
import triton
import triton.language as tl

# Set the token group alignment size for experts in MoE. This is implemented by
# padding each expert size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

# Valid values are: 8, 16, or 32.
# Different values are needed for different cases:

# * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
# * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
# * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
#   so when doing per-token-group quantization on each logically distinct subtensor,
#   we need to ensure the contracting dim is divisible by block_size.
#   In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
#   of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
#   so we need 32 element alignment.
TOKEN_GROUP_ALIGN_SIZE_M = 16
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


@triton.jit
def _fill_indices_kernel(
    num_tokens_per_expert_ptr: tl.tensor,  # [E]
    start_index_values_ptr: tl.tensor,  # [E]
    write_offsets_ptr: tl.tensor,  # [E]
    output_ptr: tl.tensor,  # [max_len]
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
) -> None:
    pid = tl.program_id(axis=0)  # [scalar]
    num_programs = tl.num_programs(axis=0)  # [scalar]

    # map programs (blocks) to the experts and loop (grid stride) if needed
    for expert_id in range(pid, num_experts, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)  # [scalar]

        # load number of tokens for this expert
        start_index = tl.load(start_index_values_ptr + expert_id)  # [scalar]
        length = tl.load(num_tokens_per_expert_ptr + expert_id)  # [scalar]

        # each thread in block processes tokens in parallel
        offsets = tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]

        # tokens are processed in chunks of BLOCK_SIZE
        for chunk_start in range(0, length, BLOCK_SIZE):
            chunk_offsets = chunk_start + offsets  # [BLOCK_SIZE]

            # mask valid indices
            mask = chunk_offsets < length  # [BLOCK_SIZE]

            values = start_index + chunk_offsets  # [BLOCK_SIZE]

            # destination
            dest_indices = write_offset + chunk_offsets  # [BLOCK_SIZE]

            # store
            tl.store(output_ptr + dest_indices, values, mask=mask)


def _fill_indices_wrapper(
    num_tokens_per_expert: torch.Tensor,  # [E]
    start_index_values: torch.Tensor,  # [E]
    write_offsets: torch.Tensor,  # [E]
    num_experts: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
) -> torch.Tensor:  # [max_len]
    # preallocate output
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32, device=num_tokens_per_expert.device)  # [max_len]

    # write offsets is per local expert...
    num_blocks = min(num_experts, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        num_tokens_per_expert,
        start_index_values,
        write_offsets,
        permuted_indices,
        num_experts,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


def _generate_permute_indices(
    num_tokens_per_expert: torch.Tensor,  # [E]
    num_experts: int,
    max_len: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # [max_len], [E]
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        num_tokens_per_expert: number of tokens for each expert.
        num_experts: number of experts.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.

    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """
    start_index_values = torch.cumsum(num_tokens_per_expert, dim=0) - num_tokens_per_expert  # [E]

    # pad out empty experts to alignment requirement
    m_sizes = torch.clamp_min(num_tokens_per_expert, alignment)  # [E]

    # align the chunk sizes (cdiv)
    m_sizes = (m_sizes.to(torch.int32) + alignment - 1) // alignment * alignment  # [E]

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    write_offsets = torch.cumsum(m_sizes, dim=0) - m_sizes  # [E]

    # Select the implementation to use
    permuted_indices = _fill_indices_wrapper(
        num_tokens_per_expert=num_tokens_per_expert,
        start_index_values=start_index_values,
        write_offsets=write_offsets,
        num_experts=num_experts,
        max_len=max_len,
    )  # [max_len]

    return permuted_indices, m_sizes
