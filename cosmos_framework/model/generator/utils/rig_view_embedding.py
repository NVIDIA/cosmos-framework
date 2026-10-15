# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physical rig identity for camera-major control and target token grids."""

import math
from typing import Any

import torch


def vision_view_ids(
    data_batch: dict[str, Any],
    *,
    batch_size: int,
    item_counts: list[int] | None,
    views_per_item: list[int] | None,
    num_embeddings: int,
) -> list[torch.Tensor]:  # returns one [V] ID tensor per vision item
    """Repeat canonical camera IDs; an explicit singleton -1 denotes a non-rig video."""
    raw_ids = data_batch.get("view_indices_selection")
    if raw_ids is None or len(raw_ids) != batch_size:
        raise ValueError("Rig view embeddings require view_indices_selection for every RGB sample")
    counts = item_counts if item_counts is not None else [1] * batch_size
    result: list[torch.Tensor] = []
    for sample_ids, count in zip(raw_ids, counts, strict=True):
        ids = torch.as_tensor(sample_ids, dtype=torch.long).reshape(-1)  # [V]
        is_non_rig = ids.numel() == 1 and ids.item() == -1
        if not is_non_rig and (ids.numel() == 0 or bool(((ids < 0) | (ids >= num_embeddings - 1)).any())):
            raise ValueError(
                "RGB view IDs must be canonical camera IDs or a singleton -1 for non-rig video; "
                "the final embedding is reserved for LiDAR"
            )
        for _ in range(count):
            if views_per_item is not None and ids.numel() != views_per_item[len(result)]:
                raise ValueError("Physical camera IDs do not match the encoded camera count")
            result.append(ids)
    return result


def add_view_embeddings(
    tokens: torch.Tensor,  # [N,D]
    token_shapes: list[tuple[int, int, int]],
    view_ids: list[torch.Tensor],  # one [V] tensor per item
    embedding: torch.nn.Embedding,
) -> torch.Tensor:  # [N,D]
    """Add camera-major offsets in place to an exclusively owned, freshly projected token buffer.

    Return the same buffer. During training it must be a non-leaf tensor, as produced
    by the caller's projection, and no consumer may need its pre-offset values.
    """
    if sum(math.prod(shape) for shape in token_shapes) != tokens.shape[0]:
        raise ValueError("Camera-major token shapes must match the projected token count")
    token_start = 0
    for shape, ids in zip(token_shapes, view_ids, strict=True):
        count = math.prod(shape)
        if shape[0] % ids.numel() != 0:
            raise ValueError("Camera-major temporal extent must be divisible by the number of view IDs")
        # Non-rig videos contribute no offset or embedding gradient. Clamp only
        # for the lookup; never interpret the sentinel as the LiDAR/last row.
        per_view = embedding(ids.clamp_min(0)) * (ids >= 0).unsqueeze(-1)  # [V,D]
        # The caller has just projected these tokens and has no other consumer
        # of their pre-offset values. Broadcast the small view table rather
        # than allocating repeated offsets, their concatenation, and a result.
        per_view = per_view.to(tokens.dtype)  # [V,D]
        item_tokens = tokens.narrow(0, token_start, count)  # [N_item,D]
        item_tokens.view(ids.numel(), count // ids.numel(), tokens.shape[1]).add_(
            per_view.unsqueeze(1)
        )  # [V,N_per_view,D]
        token_start += count
    return tokens  # [N,D]
