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
    """Repeat each sample's canonical camera IDs for its controls and target."""
    raw_ids = data_batch.get("view_indices_selection")
    if raw_ids is None or len(raw_ids) != batch_size:
        raise ValueError("Rig view embeddings require view_indices_selection for every RGB sample")
    counts = item_counts if item_counts is not None else [1] * batch_size
    result: list[torch.Tensor] = []
    for sample_ids, count in zip(raw_ids, counts, strict=True):
        ids = torch.as_tensor(sample_ids, dtype=torch.long).reshape(-1)  # [V]
        if ids.numel() == 0 or bool(((ids < 0) | (ids >= num_embeddings - 1)).any()):
            raise ValueError("RGB view IDs must be canonical camera IDs; the final embedding is reserved for LiDAR")
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
    """Apply IDs in camera-major order to every spatial and temporal token."""
    offsets: list[torch.Tensor] = []
    for shape, ids in zip(token_shapes, view_ids, strict=True):
        count = math.prod(shape)
        if shape[0] % ids.numel() != 0:
            raise ValueError("Camera-major temporal extent must be divisible by the number of view IDs")
        per_view = embedding(ids)  # [V,D]
        offsets.append(per_view.repeat_interleave(count // ids.numel(), dim=0))  # [N_item,D]
    return tokens + torch.cat(offsets, dim=0).to(tokens.dtype)  # [N,D]
