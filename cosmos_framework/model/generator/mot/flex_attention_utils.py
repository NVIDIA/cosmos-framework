# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared metadata-run utilities for OOM-safe FlexAttention block masks.

Both bidirectional multiview attention and interactive replay attention describe
visibility with compact per-token metadata.  Evaluating either predicate with
``create_block_mask`` would first materialize a dense token-by-token boolean tensor,
which is prohibitively large for an 11-view 720p sequence.  This module keeps the
shared compression and block-classification algorithm independent of either model's
metadata schema and visibility predicate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.nn.attention.flex_attention import BlockMask

MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def metadata_run_groups(
    field_tensors: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group consecutive tokens whose predicate fields are identical.

    Returns ``(group_id, representatives)``.  ``group_id`` has one entry per
    token, while ``representatives`` stores the first token index for every run.
    The caller chooses the fields, so this works for both the base multiview
    schema and the richer teacher-forcing query/key schemas.
    """
    if not field_tensors:
        raise ValueError("At least one metadata field is required.")
    seq_len = field_tensors[0].numel()
    if any(field.numel() != seq_len for field in field_tensors):
        field_lengths = tuple(field.numel() for field in field_tensors)
        raise ValueError(f"Metadata fields must have equal lengths, got {field_lengths}.")
    stacked_fields = torch.stack(tuple(field.to(torch.long) for field in field_tensors), dim=1)  # [S,F]
    starts_run = torch.ones(seq_len, dtype=torch.bool, device=device)  # [S]
    starts_run[1:] = (stacked_fields[1:] != stacked_fields[:-1]).any(dim=1)  # [S-1]
    group_id = torch.cumsum(starts_run, dim=0) - 1  # [S]
    representatives = torch.nonzero(starts_run, as_tuple=False).squeeze(1)  # [G]
    return group_id, representatives


def _block_presence(
    group_id: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one-hot block-to-metadata-run membership."""  # group_id: [S], returns [B,G]
    presence = torch.zeros(num_blocks, num_groups, dtype=torch.float32, device=device)  # [B,G]
    block_of_token = torch.arange(group_id.numel(), device=device) // block_size  # [S]
    presence[block_of_token, group_id] = 1.0  # [B,G]
    return presence


def _ordered_blocks(dense_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense block matrix to FlexAttention's ordered representation."""
    dense = dense_blocks.to(torch.int32)  # [1,1,BQ,BKV]
    counts = dense.sum(dim=-1).to(torch.int32)  # [1,1,BQ]
    indices = torch.argsort(dense, dim=-1, descending=True, stable=True).to(torch.int32)  # [1,1,BQ,BKV]
    return counts.contiguous(), indices.contiguous()


def build_block_mask_from_metadata_runs(
    *,
    q_group_id: torch.Tensor,
    kv_group_id: torch.Tensor,
    q_representatives: torch.Tensor,
    kv_representatives: torch.Tensor,
    pair_allowed: MaskMod,
    mask_mod: MaskMod,
    q_len: int,
    kv_len: int,
    device: torch.device,
    block_size: tuple[int, int],
) -> BlockMask:
    """Build a FlexAttention block mask without a dense token-pair tensor.

    The visibility predicate is evaluated only for metadata-run representatives.
    Block-to-run presence matrices then classify each block pair as fully visible,
    partially visible, or masked.  Memory therefore scales with the block matrix
    instead of ``q_len * kv_len``.
    """
    q_block_size, kv_block_size = block_size
    if q_len % q_block_size != 0:
        raise ValueError(f"Query length {q_len} is not aligned to block size {q_block_size}.")
    if kv_len % kv_block_size != 0:
        raise ValueError(f"Key/value length {kv_len} is not aligned to block size {kv_block_size}.")
    if q_group_id.numel() != q_len or kv_group_id.numel() != kv_len:
        raise ValueError(
            "Metadata group lengths must match the attention streams; "
            f"got query={q_group_id.numel()}/{q_len}, key/value={kv_group_id.numel()}/{kv_len}."
        )

    num_q_blocks = q_len // q_block_size
    num_kv_blocks = kv_len // kv_block_size
    batch_index = torch.zeros((), dtype=torch.long, device=device)  # []
    allowed = pair_allowed(
        batch_index,
        batch_index,
        q_representatives.unsqueeze(1),  # [GQ,1]
        kv_representatives.unsqueeze(0),  # [1,GKV]
    ).to(torch.float32)  # [GQ,GKV]
    presence_q = _block_presence(
        q_group_id,
        num_blocks=num_q_blocks,
        block_size=q_block_size,
        num_groups=q_representatives.numel(),
        device=device,
    )  # [BQ,GQ]
    presence_kv = (
        presence_q
        if q_group_id is kv_group_id and q_block_size == kv_block_size
        else _block_presence(
            kv_group_id,
            num_blocks=num_kv_blocks,
            block_size=kv_block_size,
            num_groups=kv_representatives.numel(),
            device=device,
        )
    )  # [BKV,GKV]

    allowed_hits = (presence_q @ allowed) @ presence_kv.t()  # [BQ,BKV]
    blocked_hits = (presence_q @ (1.0 - allowed)) @ presence_kv.t()  # [BQ,BKV]
    full_blocks = blocked_hits == 0.0  # [BQ,BKV]
    partial_blocks = (allowed_hits > 0.0) & ~full_blocks  # [BQ,BKV]
    kv_num_blocks, kv_indices = _ordered_blocks(partial_blocks.view(1, 1, num_q_blocks, num_kv_blocks))
    full_kv_num_blocks, full_kv_indices = _ordered_blocks(full_blocks.view(1, 1, num_q_blocks, num_kv_blocks))
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),
    )
