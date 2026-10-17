# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Filesystem coordination for distributed Cosmos status publishers."""

import time
from pathlib import Path

from cosmos_framework.checkpoint.reasoner import distributed_invocation_id


def wait_for_status_publishers(results_dir: Path, rank: int, world_size: int, timeout_seconds: int) -> None:
    """Ensure rank zero publishes terminal status after all rank callbacks."""
    if world_size <= 1:
        return

    marker_dir = results_dir / ".cosmos_status_barrier" / distributed_invocation_id()
    marker_dir.mkdir(parents=True, exist_ok=True)
    # An old publisher marker must never satisfy a new invocation's barrier.
    with (marker_dir / f"rank_{rank}.ready").open("x") as marker:
        marker.write("ready\n")
    if rank != 0:
        return

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all((marker_dir / f"rank_{i}.ready").is_file() for i in range(world_size)):
            return
        time.sleep(0.1)
    missing = [i for i in range(world_size) if not (marker_dir / f"rank_{i}.ready").is_file()]
    raise TimeoutError(f"Timed out waiting for Cosmos status publishers: {missing}")
