# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pick a subset that spreads over two keys instead of clustering on the best of one.

Both the planner and the exporter need the same thing: take N items out of many, covering as many
hosts as possible, and within a host as many donors as possible. Doing it naively -- sorting by
quality, or cycling one flat order of (host, donor) pairs -- looks fair and is not. With fewer
slots than pairs, a flat cycle walks the sorted order and never reaches the last hosts, which
silently drops whole episodes from a dataset.

So the rotation lives here once, rather than being copied into the two callers and drifting.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T")


def spread(
    items: Iterable[T],
    take: int,
    primary: Callable[[T], Hashable],
    secondary: Callable[[T], Hashable] | None = None,
    rank: Callable[[T], float] | None = None,
    offset: int = 0,
) -> list[T]:
    """``take`` items, cycling over ``primary`` so every group is served before any gets seconds.

    Within a group, items are cycled over ``secondary`` too, and each group starts at a different
    secondary value. Without that offset, groups that rank their secondaries the same way all
    contribute the same one first, and an early cut-off returns many items of a few kinds.

    ``rank`` orders items within a (primary, secondary) cell, highest first; it decides which item
    represents a cell, never which cells are visited.

    ``offset`` shifts where the cycle starts. A caller applying this per host passes the host's
    position, so different hosts lead with different groups and the corpus stays varied even
    though each host is capped at a handful of items.
    """
    order_key = (lambda x: -rank(x)) if rank is not None else (lambda x: 0)
    groups: dict[Hashable, dict[Hashable, list[T]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[primary(item)][secondary(item) if secondary else None].append(item)
    if take <= 0 or not groups:
        return []

    queues: dict[Hashable, list[T]] = {}
    for slot, key in enumerate(sorted(groups, key=_sortable)):
        cells = groups[key]
        for cell in cells.values():
            cell.sort(key=order_key)
        cell_order = sorted(cells, key=lambda c: (order_key(cells[c][0]), _sortable(c)))
        start = (slot + offset) % len(cell_order)
        cell_order = cell_order[start:] + cell_order[:start]
        queue: list[T] = []
        while any(cells[c] for c in cell_order):
            for c in cell_order:
                if cells[c]:
                    queue.append(cells[c].pop(0))
        queues[key] = queue

    out: list[T] = []
    keys = sorted(queues, key=_sortable)
    if keys:
        start = offset % len(keys)
        keys = keys[start:] + keys[:start]
    while len(out) < take and any(queues[k] for k in keys):
        for key in keys:
            if not queues[key]:
                continue
            out.append(queues[key].pop(0))
            if len(out) == take:
                break
    return out


def _sortable(value: Hashable):
    """Sort keys of mixed or missing types without raising, so a None donor cannot break a plan."""
    return (value is None, str(type(value)), str(value))
