# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterable
from typing import Any

import torch

_RESOLUTION_768_SHAPES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1184, 880),
    (880, 1184),
    (1360, 768),
    (768, 1360),
)


def get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    """Determine the resolution string from spatial dimensions.

    Maps the spatial shape (height, width) to a resolution string. This is used
    for resolution-dependent shift lookup when using dict-based shift
    configuration.

    Args:
        spatial_shape: Tuple of (height, width) in pixels.

    Returns:
        Resolution string: "256", "480", "720", or "768".

    Raises:
        ValueError: If the spatial shape is unsupported.

    Note:
        See VIDEO_RES_SIZE_INFO for more details on resolution definitions.
        For the current definition of resolution, these conditions are satisfied.
    """
    if spatial_shape in _RESOLUTION_768_SHAPES:
        return "768"

    min_dim = min(spatial_shape[0], spatial_shape[1])
    if min_dim <= 256:
        return "256"
    elif min_dim <= 640:
        return "480"
    elif min_dim <= 960:
        return "720"
    elif min_dim <= 2048:
        # Free-form inputs above the 720 tier (e.g. multi-reference generation
        # producing shapes like (992, 1024)) that are not a canonical 768 shape:
        # route to the closest defined higher tier "768".
        return "768"
    else:
        raise ValueError(f"Unsupported resolution: {spatial_shape}")


def slice_data_batch(
    data_batch: dict[str, Any],
    start: int,
    limit: int,
    multi_item_fields: Iterable[str] = ("image", "images", "video", "videos", "image_size"),
) -> dict[str, Any]:
    """Slice a data batch based on the start and limit indices.

    For most fields, the slice ``[start:limit]`` is applied directly along the
    sample dimension. For fields listed in ``multi_item_fields`` (e.g. ``image``,
    ``images``, ``video``, and ``videos``), each sample may contribute multiple visual items that are
    concatenated in flat order. In that case, when
    ``num_vision_items_per_sample`` is present in ``data_batch``, the slice is
    expanded to cover all visual items belonging to the requested samples.

    Example:
        ``num_vision_items_per_sample = [2, 2]`` and
        ``video = [v1_s1, v2_s1, v1_s2, v2_s2]``. Slicing with
        ``start=0, limit=1`` returns ``video = [v1_s1, v2_s1]``.

    Args:
        data_batch: The data batch to slice.
        start: The start sample index (inclusive).
        limit: The end sample index (exclusive).
        multi_item_fields: Field names whose values store multiple visual
            items per sample concatenated in flat order. Only used when
            ``data_batch`` contains ``num_vision_items_per_sample``.

    Returns:
        The sliced data batch.
    """
    assert start >= 0 and limit > 0, "Start and limit must be positive"
    assert start < limit, "Start must be less than limit"

    num_items = data_batch.get("num_vision_items_per_sample")
    if num_items is not None:
        if isinstance(num_items, torch.Tensor):
            num_items_list = num_items.tolist()
        else:
            num_items_list = list(num_items)
        flat_start = sum(num_items_list[:start])
        flat_limit = sum(num_items_list[:limit])
    else:
        flat_start, flat_limit = start, limit

    multi_item_fields = set(multi_item_fields)

    sliced_batch = {}
    for key, value in data_batch.items():
        if key in multi_item_fields and num_items is not None:
            s, e = flat_start, flat_limit
        else:
            s, e = start, limit
        if isinstance(value, torch.Tensor):
            sliced_batch[key] = value[s:e]
        elif isinstance(value, list):
            sliced_batch[key] = value[s:e]
        else:
            sliced_batch[key] = value
    return sliced_batch
