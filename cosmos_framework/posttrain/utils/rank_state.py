# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared rank-local value collection and selection primitives."""

from __future__ import annotations

import copy
import pickle
from typing import Any, Literal


class RankStateWorldSizeMismatch(ValueError):
    """A rank-local value cannot be mapped exactly to the current world size."""


_RANK_STATE_VERSION = 1
RankStateMismatchPolicy = Literal["raise", "skip"]


def gather_rank_states(state: Any, *, context: str) -> dict[int, Any]:
    """Gather one small, pickleable value per rank as ``{rank: value}``.

    Preparation errors converge before ``all_gather_object`` so one bad rank cannot
    strand its peers. Returned values are deep copies and therefore safe for an
    asynchronous checkpoint writer.
    """
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    local: Any = None
    preparation_error: Exception | None = None
    try:
        local = copy.deepcopy(state)
        pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:  # noqa: BLE001 — converge before all_gather_object pickles
        preparation_error = error
    raise_if_any_rank_failed(preparation_error, context)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    return {rank: item for rank, item in enumerate(gathered)}


def select_rank_state(
    states: dict[int, Any] | None,
    *,
    on_world_size_mismatch: RankStateMismatchPolicy,
) -> Any:
    """Select this rank's value from a rank-indexed state dictionary.

    ``on_world_size_mismatch="skip"`` returns ``None`` for legacy/non-exact
    continuation; ``"raise"`` protects callers that require exact same-topology state.
    Malformed state always fails.
    """
    import torch.distributed as dist

    if states is None:
        return None
    if not isinstance(states, dict) or any(type(rank) is not int for rank in states):
        raise ValueError("rank-local checkpoint states must be indexed by integer rank")
    if not states:
        raise ValueError("rank-local checkpoint states must contain at least one rank")
    saved_world = len(states)
    if set(states) != set(range(saved_world)):
        raise ValueError("rank-local checkpoint states must contain contiguous ranks")
    current_world = dist.get_world_size() if dist.is_initialized() else 1
    if saved_world != current_world:
        if on_world_size_mismatch == "skip":
            return None
        raise RankStateWorldSizeMismatch(
            f"rank-local checkpoint world_size={saved_world}, current world_size={current_world}"
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    return copy.deepcopy(states[rank])


def collect_rank_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Collect one optional state dict per rank into a versioned component.

    Every rank participates, including when ``state`` is ``None``, so mixed
    present/absent support fails together rather than publishing a partial cursor.
    """
    gathered = gather_rank_states(
        state,
        context="rank-local checkpoint state preparation",
    )
    if all(item is None for item in gathered.values()):
        return None
    if any(item is None for item in gathered.values()):
        raise RuntimeError("rank-local checkpoint state must be present on every rank or none")
    return {
        "version": _RANK_STATE_VERSION,
        "world_size": len(gathered),
        "states": gathered,
    }


def state_for_current_rank(
    component: dict[str, Any] | None,
    *,
    on_world_size_mismatch: RankStateMismatchPolicy = "raise",
) -> dict[str, Any] | None:
    """Select this rank's state from :func:`collect_rank_state` output.

    Returning ``None`` is reserved for an absent component or, when explicitly
    requested, a world-size mismatch. Malformed state always fails.
    """
    if component is None:
        return None
    if not isinstance(component, dict):
        raise ValueError("rank-local checkpoint component must be a dictionary")
    version = component.get("version")
    if type(version) is not int:
        raise ValueError("rank-local checkpoint version must be an integer")
    if version != _RANK_STATE_VERSION:
        raise ValueError(f"unsupported rank-local checkpoint version {version}")
    saved_world = component.get("world_size")
    if type(saved_world) is not int:
        raise ValueError("rank-local checkpoint world size must be an integer")
    if saved_world < 1:
        raise ValueError("rank-local checkpoint world size must be positive")
    states = component.get("states")
    if not isinstance(states, dict) or saved_world != len(states):
        raise ValueError("rank-local checkpoint world size does not match its state inventory")
    if any(not isinstance(state, dict) for state in states.values()):
        raise ValueError("rank-local checkpoint cursor states must be dictionaries")
    selected = select_rank_state(
        states,
        on_world_size_mismatch=on_world_size_mismatch,
    )
    return selected
