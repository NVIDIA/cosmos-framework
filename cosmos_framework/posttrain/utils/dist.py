# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Thin torch.distributed helpers so the ENTRIES never touch dist primitives directly.

The entry loop runs symmetrically on every torchrun rank, but a handful of things are
per-JOB global state (process-group lifecycle, single-decision values like the run
timestamp, world sizing). Those live here; the entry stays pure orchestration. Imports
of torch are function-local so this module stays importable on login nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn, TypeVar, cast, overload

if TYPE_CHECKING:
    import torch

_Result = TypeVar("_Result")
_MISSING = object()


def init_local_process_group() -> None:
    """Join torchrun's NCCL world (the FSDP mesh) and pin this rank's device. Safe to
    call when the group already exists."""
    import os

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def destroy_process_group_if_initialized() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def world_size_or_1() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_initialized() else 1


def rank_or_0() -> int:
    """This rank in the default process group, or 0 outside one (the single-process case)."""
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def collective_device() -> torch.device:
    """Return a device compatible with the default process group's backend.

    Callers must first establish that the default process group is initialized.
    """
    import torch
    import torch.distributed as dist

    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def rank_mean_count(local_count: int, device: torch.device) -> float:
    """Return the mean count across ranks, or the local count outside a process group."""
    import torch
    import torch.distributed as dist

    local = float(local_count)
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return local
    count = torch.tensor([local], device=device)  # [1]
    dist.all_reduce(count, op=dist.ReduceOp.AVG)
    return float(count.item())


def rank_max_count(local_count: int, device: torch.device) -> int:
    """Return the maximum count across ranks, or the local count outside a process group."""
    import torch
    import torch.distributed as dist

    local = int(local_count)
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return local
    count = torch.tensor([local], dtype=torch.int64, device=device)  # [1]
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


def rank_sum_count(local_count: int, device: torch.device) -> int:
    """Return the summed count across ranks, or the local count outside a process group."""
    import torch
    import torch.distributed as dist

    local = int(local_count)
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return local
    count = torch.tensor([local], dtype=torch.int64, device=device)  # [1]
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count.item())


def rank_any(local: bool) -> bool:
    """Return whether ANY rank passed True, or `local` outside a process group.

    The boolean case of `rank_max_count`, given its own name because that is what the callers mean
    and because it can pick the device itself -- a caller reducing a flag has no reason to know which
    device the backend wants.

    Use this when a verdict is VACUOUS on some ranks and must not be allowed to decide for the ranks
    where it is not. The motivating case: `select_chunk` round-robins, so a rank legitimately gets an
    empty chunk when the batch is smaller than the mesh, and a rank that compared nothing must not
    retire a guard on behalf of ranks that hold data. `any` is the correct quantifier there; `all`
    would let one empty rank veto, and a rank-local decision would let it certify.

    Args:
        local: this rank's verdict.

    Returns:
        True iff at least one rank passed True.
    """
    import torch
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return bool(local)
    flag = torch.tensor([1 if local else 0], dtype=torch.int64, device=collective_device())  # [1]
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(int(flag.item()))


def rank_identical(value: _Result, context: str) -> _Result:
    """Return one value after every default-process-group rank proves equality."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    values: list[_Result | None] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    first = values[0]
    if any(candidate != first for candidate in values[1:]):
        raise RuntimeError(f"{context} differs across ranks: {values!r}")
    return cast(_Result, first)


def is_dtensor_sharded(module: torch.nn.Module) -> bool:
    """Return whether any parameter or its storage is a DTensor shard."""
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return False
    return any(
        isinstance(parameter, DTensor) or isinstance(getattr(parameter, "data", None), DTensor)
        for parameter in module.parameters()
    )


@overload
def raise_if_any_rank_failed(local_error: None, context: str) -> None: ...


@overload
def raise_if_any_rank_failed(local_error: BaseException | str, context: str) -> NoReturn: ...


def raise_if_any_rank_failed(local_error: BaseException | str | None, context: str) -> None:
    """Converge recoverable local failures before a later collective phase."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        if local_error is not None:
            if isinstance(local_error, BaseException):
                raise local_error
            raise RuntimeError(f"{context} failed: {local_error}")
        return
    error_text = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    failed = torch.tensor(error_text is not None, dtype=torch.int32, device=collective_device())  # []
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if not bool(failed.item()):
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error_text)
    details = "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)
    raise RuntimeError(f"{context} failed on one or more ranks: {details}")


def run_rank_converged(
    action: Callable[[], _Result],
    context: str,
    *,
    raise_errors: Callable[[BaseException | str | None, str], None] | None = None,
) -> _Result:
    """Run local preparation and return only after every rank reports success."""
    result: _Result | object = _MISSING
    error: BaseException | None = None
    try:
        result = action()
    except BaseException as caught:  # noqa: BLE001 -- peers must not enter a later collective alone
        error = caught
    (raise_errors or raise_if_any_rank_failed)(error, context)
    if result is _MISSING:
        raise RuntimeError(f"{context} completed without a result")
    return cast(_Result, result)


def broadcast_from_rank0(
    value_fn: Callable[[], _Result],
    *,
    context: str = "rank-zero value preparation",
) -> _Result:
    """Evaluate `value_fn()` on rank 0 ONLY and broadcast the result (single-decision
    values, e.g. a freshly minted run timestamp — ranks must not compute it
    independently). Outside a process group: just evaluate locally."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return value_fn()
    value: _Result | None = None

    def prepare() -> None:
        nonlocal value
        if dist.get_rank() == 0:
            value = value_fn()

    run_rank_converged(prepare, context)
    holder: list[_Result | None] = [value]
    dist.broadcast_object_list(holder, src=0)
    return cast(_Result, holder[0])
