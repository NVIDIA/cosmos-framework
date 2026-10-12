# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Filesystem coordination for distributed Cosmos status publishers."""

import hashlib
import os
import time
from pathlib import Path


def wait_for_status_publishers(results_dir: Path, rank: int, world_size: int, timeout_seconds: int) -> None:
    """Ensure rank zero publishes terminal status after all rank callbacks."""
    if world_size <= 1:
        return

    run_id = os.environ.get("COSMOS_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if not run_id or run_id == "none":
        job_id = os.environ.get("COSMOS_JOB_ID") or os.environ.get("SLURM_JOB_ID")
        if not job_id:
            raise ValueError("Distributed evaluation requires COSMOS_RUN_ID or a launcher-provided job identity")
        run_id = job_id + ":" + os.environ.get("SLURM_STEP_ID", "0")
    marker_dir = results_dir / ".cosmos_status_barrier" / hashlib.sha256(run_id.encode()).hexdigest()[:24]
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
