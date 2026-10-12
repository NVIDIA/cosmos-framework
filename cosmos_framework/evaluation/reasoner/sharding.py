# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic evaluation sharding policies."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def media_balanced_shard(records: list[dict[str, Any]], total_shards: int, shard_id: int) -> list[dict[str, Any]]:
    """Keep each media asset on one shard while balancing record counts."""
    if total_shards <= 0:
        raise ValueError("total_shards must be positive")
    if not 0 <= shard_id < total_shards:
        raise ValueError("shard_id must be in [0, total_shards)")
    if total_shards == 1:
        return list(records)

    groups: OrderedDict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for index, record in enumerate(records):
        media_paths = tuple(str(path) for path in record.get("media_paths", []))
        if media_paths:
            key = (str(record.get("media_mode", "image")), media_paths)
        else:
            key = ("text", index)
        groups.setdefault(key, []).append((index, record))

    # The shared evaluator's completion barrier requires every rank to own a
    # task when there are at least as many records as ranks.  Preserve media
    # affinity whenever it can populate all ranks; otherwise use the original
    # deterministic stride policy for this small/degenerate split.
    if len(records) >= total_shards and len(groups) < total_shards:
        return list(records[shard_id::total_shards])

    ranked_groups = sorted(groups.values(), key=lambda group: (-len(group), group[0][0]))
    shard_loads = [0] * total_shards
    assigned: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(total_shards)]
    for group in ranked_groups:
        target = min(range(total_shards), key=lambda rank: (shard_loads[rank], rank))
        assigned[target].extend(group)
        shard_loads[target] += len(group)

    return [record for _, record in sorted(assigned[shard_id], key=lambda item: item[0])]
