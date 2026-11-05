# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Per-rank shard extraction from consolidated (rank-0 gathered) checkpoint components.

A consolidated component (``model.pt``, ``optim.pt``) holds every tensor whole. Resuming a
DTensor-sharded trainer from one by handing the whole dict to ``set_model_state_dict`` /
``set_optimizer_state_dict`` with ``full_state_dict=True`` makes EVERY rank process the whole
component: torch flattens the entire optimizer dict, moves each consolidated tensor whole to the
target device and scatters it from rank 0 (``_distribute_state_dict`` -> ``distribute_tensor``),
so a one-node resume pays world-size times the checkpoint on one host. The DCP reader does the
opposite: each rank reads only the byte range of its own chunk. This module gives the
consolidated layout the DCP reader's memory profile, using the same chunk geometry DCP uses.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch


def fill_sharded_state_from_consolidated(
    target: MutableMapping[str, Any],
    consolidated: Mapping[str, Any],
    *,
    context: str,
    strict: bool = True,
) -> int:
    """Copy this rank's share of every tensor in ``consolidated`` into ``target`` in place.

    ``target`` is the live state a rank extracted WITHOUT ``full_state_dict``
    (``get_model_state_dict(net)`` / ``get_optimizer_state_dict(net, opt)``): DTensor shards for
    sharded state, plain tensors for replicated state, control leaves (``param_groups``) as
    objects. ``consolidated`` is the matching rank-0 gathered state (``model.pt["net"]``,
    ``optim.pt``), normally mmap-backed (``io.load_component``): same nesting, same keys.

    For a DTensor leaf only this rank's chunk of the consolidated tensor is read
    (``compute_local_shape_and_global_offset``, the geometry DCP reads shards by) and copied
    straight into the live local shard; a plain tensor leaf is copied whole; a control leaf is
    replaced by the saved object, as a DCP load would. Nothing goes through the
    ``full_state_dict`` setters, so the bytes a rank materializes are exactly the bytes of its own
    live state whatever the world size -- the returned count. With an mmap-backed source the
    consolidated file only ever occupies reclaimable page cache, and for a ``Shard(0)`` layout
    each rank touches only its own contiguous byte ranges of it.

    ``strict`` rejects consolidated entries the target has no slot for (the model, mirroring
    ``load_state_dict(strict=True)``); optimizer state is matched by the live parameters exactly
    as ``set_optimizer_state_dict`` matches it, so extras are ignored there. A target leaf with
    no saved counterpart is always an error: a silently unfilled moment would resume from zeros.
    """
    return _fill(target, consolidated, prefix="", context=context, strict=strict)


def _fill(
    target: MutableMapping[str, Any],
    consolidated: Mapping[str, Any],
    *,
    prefix: str,
    context: str,
    strict: bool,
) -> int:
    copied = 0
    for key, live in list(target.items()):
        name = f"{prefix}{key}"
        if key not in consolidated:
            raise KeyError(f"{context}: consolidated state has no entry for {name!r}")
        saved = consolidated[key]
        if isinstance(live, MutableMapping):
            if not isinstance(saved, Mapping):
                raise TypeError(
                    f"{context}: {name!r} is a nested state on this rank but {type(saved).__name__} on disk"
                )
            copied += _fill(live, saved, prefix=f"{name}.", context=context, strict=strict)
        elif isinstance(live, torch.Tensor):
            copied += _copy_into_live_tensor(live, saved, name=name, context=context)
        else:
            # Control values (optimizer `param_groups`, counters): the saved object replaces the
            # live one, as the DCP planner does for non-tensor entries.
            target[key] = saved
    if strict:
        extra = sorted(set(consolidated) - set(target))
        if extra:
            raise KeyError(f"{context}: consolidated state has entries this rank has no slot for: {extra[:8]!r}")
    return copied


def _copy_into_live_tensor(live: torch.Tensor, saved: Any, *, name: str, context: str) -> int:
    if not isinstance(saved, torch.Tensor):
        raise TypeError(f"{context}: {name!r} is a tensor on this rank but {type(saved).__name__} on disk")
    if tuple(saved.shape) != tuple(live.shape):
        raise ValueError(
            f"{context}: {name!r} has global shape {tuple(live.shape)} here but {tuple(saved.shape)} on disk"
        )
    dtensor_type = _dtensor_type()
    if dtensor_type is not None and isinstance(live, dtensor_type):
        local = live.to_local()
        offset, local_shape = _local_chunk(live, local, name=name, context=context)
        source = saved[tuple(slice(start, start + size) for start, size in zip(offset, local_shape, strict=True))]
    else:
        local = live
        source = saved
    if local.numel() == 0:
        return 0
    local.copy_(source)  # reads only `source`'s pages; casts if the saved dtype differs
    return local.numel() * local.element_size()


def _dtensor_type() -> type | None:
    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # noqa: BLE001 -- torch.distributed absent: every leaf is a plain tensor
        return None
    return DTensor


def _local_chunk(live: Any, local: torch.Tensor, *, name: str, context: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """This rank's (global offset, local shape) of ``live``; refuses layouts that are not one slice."""
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    for placement in live.placements:
        if placement.is_partial():
            raise ValueError(f"{context}: {name!r} carries a Partial placement; it is not a stored parameter or moment")
        if type(placement).__name__ == "_StridedShard":
            raise ValueError(
                f"{context}: {name!r} is strided-sharded ({placement}); its local shard is not one contiguous "
                "slice of the consolidated tensor, so it cannot be restored from a consolidated checkpoint "
                "(save it as DCP shards instead)"
            )
    local_shape, offset = compute_local_shape_and_global_offset(live.shape, live.device_mesh, live.placements)
    if tuple(local.shape) != tuple(local_shape):
        raise ValueError(
            f"{context}: {name!r} holds a local shard of shape {tuple(local.shape)} but its placements "
            f"{live.placements} describe {tuple(local_shape)}; refusing to guess the layout"
        )
    return tuple(offset), tuple(local_shape)
