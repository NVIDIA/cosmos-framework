# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-zero checkpoint work coordination and its single async slot."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# The next save/finalize joins this slot. Background exceptions must be carried explicitly because
# Python threads otherwise only print them and training could continue after losing a checkpoint.
_write_thread: threading.Thread | None = None
_write_error: BaseException | None = None


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def run_rank0_converged(action: Callable[[], None], context: str) -> None:
    """Run one rank-zero action and propagate every failure to all ranks."""
    error: BaseException | None = None
    if _rank() == 0:
        try:
            action()
        except BaseException as caught:  # noqa: BLE001 -- every rank must exit together
            error = caught
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    raise_if_any_rank_failed(error, context)


def join_pending_rank0() -> None:
    """Join the rank-zero worker and re-raise its recorded failure."""
    global _write_error, _write_thread

    if _write_thread is not None:
        _write_thread.join()
        _write_thread = None
    if _write_error is not None:
        error, _write_error = _write_error, None
        raise RuntimeError(f"async checkpoint write failed: {error}") from error


def _background_write(target: Callable[..., None], args: tuple[Any, ...]) -> None:
    """Run one background job and retain its exception for the next join."""
    global _write_error

    try:
        target(*args)
    except BaseException as error:  # noqa: BLE001 -- join_pending_rank0 re-raises it
        _write_error = error


def start_background_write(
    target: Callable[..., None],
    args: tuple[Any, ...],
    *,
    context: str,
) -> None:
    """Start the rank-zero worker and converge fallible ``Thread.start`` calls."""
    global _write_thread

    launch_error: BaseException | None = None
    if _rank() == 0:
        try:
            thread = threading.Thread(target=_background_write, args=(target, args), daemon=False)
            thread.start()
            # Publish the slot only after a successful start; finalize must never join an unstarted thread.
            _write_thread = thread
        except BaseException as error:  # noqa: BLE001 -- every rank must exit together
            _write_thread = None
            launch_error = error
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    raise_if_any_rank_failed(launch_error, context)


def wait_for_pending() -> None:
    """Collectively join the rank-zero worker and propagate its failure."""
    run_rank0_converged(join_pending_rank0, "checkpoint finalization")
