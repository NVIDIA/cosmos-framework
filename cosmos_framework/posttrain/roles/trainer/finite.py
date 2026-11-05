# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-consistent numerical checks used before optimizer mutation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch

from cosmos_framework.posttrain.utils.dist import collective_device


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:  # tensor: [*], returns [*local shard]
    """Return the local shard for a DTensor, or the tensor itself otherwise."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor  # [*local shard]


def _iter_tensor_leaves(value: Any, visited: dict[int, Any]) -> Iterator[torch.Tensor]:
    """Yield unseen local tensor leaves while retaining visited identities."""
    is_tensor = isinstance(value, torch.Tensor)
    to_local = getattr(value, "to_local", None)
    is_mapping = isinstance(value, Mapping)
    is_sequence = isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if not (is_tensor or callable(to_local) or is_mapping or is_sequence):
        return

    identity = id(value)
    if identity in visited:
        return
    visited[identity] = value

    if is_tensor or callable(to_local):
        local = _local_tensor(value) if is_tensor else to_local()  # [*local shard]
        if isinstance(local, torch.Tensor):
            if local is not value:
                local_identity = id(local)
                if local_identity in visited:
                    return
                visited[local_identity] = local
            yield local  # [*local shard]
            return
    if is_mapping:
        for item in value.values():
            yield from _iter_tensor_leaves(item, visited)
    elif is_sequence:
        for item in value:
            yield from _iter_tensor_leaves(item, visited)


def iter_tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    """Yield each distinct local tensor leaf from a nested tensor tree.

    State trees may contain DTensors as well as ordinary tensors. A small amount
    of duck typing also keeps this helper usable by tensor wrappers without
    importing distributed-tensor internals into the trainer substrate. Identity
    tracking makes aliases idempotent and cyclic mappings/sequences finite while
    preserving first-encounter order.
    """
    visited: dict[int, Any] = {}
    yield from _iter_tensor_leaves(value, visited)


def _all_ranks_status(local_status: int | torch.Tensor) -> int:  # local_status: []
    """Converge one rank-local finite-check status with one collective."""
    import torch.distributed as dist

    status_device = collective_device() if dist.is_initialized() else torch.device("cpu")
    if isinstance(local_status, torch.Tensor):
        status = local_status.to(device=status_device, dtype=torch.int32)  # []
    else:
        status = torch.tensor(local_status, dtype=torch.int32, device=status_device)  # []
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
    return int(status.item())


def _finite_flags_by_device(tensors: Iterable[torch.Tensor]) -> dict[torch.device, torch.Tensor]:
    """Aggregate tensor finiteness into one scalar flag per local device."""
    flags: dict[torch.device, torch.Tensor] = {}
    for tensor in tensors:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor).all()  # []
        device_flag = flags.get(tensor.device)
        if device_flag is None:
            flags[tensor.device] = finite
        else:
            device_flag.logical_and_(finite)  # []
    return flags


def _local_finite_status(
    flags: Iterable[torch.Tensor],
    *,
    missing_required: bool = False,
    placement_error: bool = False,
) -> torch.Tensor:  # returns: []
    """Encode local finite, missing, and placement state without host scalar reads."""
    import torch.distributed as dist

    status_device = collective_device() if dist.is_initialized() else torch.device("cpu")
    status_code = 3 if placement_error else 2 if missing_required else 0
    status = torch.tensor(status_code, dtype=torch.int32, device=status_device)  # []
    if status_code != 0:
        return status
    for finite in flags:
        nonfinite = torch.logical_not(finite).to(device=status_device, dtype=torch.int32)  # []
        status = torch.maximum(status, nonfinite)  # []
    return status


def all_ranks_tensors_finite(value: Any, *, require_any: bool = False) -> bool:
    """Return whether every floating tensor leaf is finite on every rank.

    Non-floating tensors are finite by definition, and empty tensors are
    vacuously finite. ``require_any`` additionally rejects a tree without any
    tensor leaf on even one participating rank.
    """
    leaves = tuple(iter_tensor_leaves(value))
    flags = _finite_flags_by_device(leaves)
    local_status = _local_finite_status(flags.values(), missing_required=require_any and not leaves)  # []
    return _all_ranks_status(local_status) == 0


def all_ranks_tensor_true(local_flag: torch.Tensor) -> bool:  # local_flag: [], returns scalar bool
    """Logical-AND a scalar tensor flag with one host synchronization total."""
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return bool(local_flag.item())
    flag = local_flag.to(device=collective_device(), dtype=torch.int32)  # []
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def all_ranks_tensor_finite(tensor: torch.Tensor) -> bool:  # tensor: [*], returns scalar bool
    """Return whether ``tensor`` is finite on every participating rank."""
    return all_ranks_tensors_finite(tensor, require_any=True)


def all_ranks_gradients_finite(
    parameters: Iterable[torch.nn.Parameter],
    *,
    require_gradients: bool = False,
) -> bool:
    """Return whether every materialized gradient shard is finite on every rank.

    A local multi-device gradient inventory is a placement error, not a numerical
    rejection. Encode that distinction in the same collective as the finite status
    so every rank raises together rather than one rank stranding its peers.
    ``require_gradients`` makes an empty gradient inventory on any rank a shared
    false result rather than allowing the usual vacuous-finite result.
    """
    from cosmos_framework.posttrain.utils.log import logger

    gradients: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradients.append(_local_tensor(parameter.grad))  # [*local parameter shard]

    devices = {gradient.device for gradient in gradients}
    placement_error = len(devices) > 1
    if placement_error:
        device_names = sorted(str(device) for device in devices)
        logger.error(f"gradient finite check found multiple local devices: {device_names}")
    flags = _finite_flags_by_device(gradients)
    local_status = _local_finite_status(  # []
        flags.values(),
        missing_required=require_gradients and not gradients,
        placement_error=placement_error,
    )

    status = _all_ranks_status(local_status)
    if status == 3:
        raise RuntimeError(
            "gradient finite check found gradients on multiple local devices on at least one rank; "
            "this is a model or optimizer placement error, not a non-finite update"
        )
    return status == 0


def all_ranks_model_optimizer_finite(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> bool:
    """Return whether nested model and optimizer tensor state is finite on every rank."""
    optimizer_groups = tuple(
        {name: value for name, value in group.items() if name != "params"} for group in optimizer.param_groups
    )
    state = {
        "model": model.state_dict(),
        "optimizer_state": optimizer.state,
        "optimizer_param_groups": optimizer_groups,
    }
    return all_ranks_tensors_finite(state, require_any=True)
