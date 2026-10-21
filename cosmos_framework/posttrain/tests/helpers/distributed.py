# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared bounded torch.distributed setup for CPU tests."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from datetime import timedelta
from typing import Any


def init_gloo_process_group(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    """Initialize a loopback Gloo group with a short failure timeout."""
    import torch.distributed as dist

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if platform.system() == "Darwin" else "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def run_spawned_ranks(
    worker: Callable[..., None],
    *,
    world_size: int,
    rendezvous_path: Any,
    args: tuple[Any, ...] = (),
    timeout_seconds: float = 30.0,
    context: str = "distributed case",
) -> dict[int, dict[str, Any]]:
    """Run ``worker`` on ``world_size`` spawned ranks and return each rank's reported result.

    Every rank is joined under one deadline and any survivor is terminated, because the failure
    this harness exists to catch is a rank BLOCKED in a collective while its peers exited: a bare
    ``join()`` would hang the suite instead of reporting it. Workers report through a queue rather
    than an exit code so a traceback survives the process boundary; a worker that raises is
    expected to put ``{"worker_error": ...}`` and is surfaced here as a failed assertion, not a
    timeout.
    """
    import queue as _queue
    import time as _time

    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not (dist.is_available() and dist.is_gloo_available()):
        import pytest

        pytest.skip("torch.distributed gloo backend unavailable")

    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    processes = [
        spawn.Process(target=worker, args=(rank, world_size, str(rendezvous_path), output_queue, *args))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    results: dict[int, dict[str, Any]] = {}
    deadline = _time.monotonic() + timeout_seconds
    while len(results) < world_size and _time.monotonic() < deadline:
        try:
            rank, result = output_queue.get(timeout=min(0.5, max(0.0, deadline - _time.monotonic())))
        except _queue.Empty:
            continue
        results[int(rank)] = result
    for process in processes:
        process.join(timeout=max(0.0, deadline - _time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5.0)

    assert not alive, f"{context} left a rank blocked"
    # A rank that reported a result and THEN died on teardown -- a leaked process group, an
    # atexit fault -- still corrupts the run it was part of, and the reported payload alone
    # cannot show that. Two of the call sites this helper replaced asserted this and two had
    # already dropped it; consolidating to the weaker set is how hang-detection quietly rots.
    assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]
    assert set(results) == set(range(world_size)), results
    assert all("worker_error" not in result for result in results.values()), results
    return results
