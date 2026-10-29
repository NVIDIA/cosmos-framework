# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in phase memory records for the native adapter qualification runs.

Set COSMOS_QLORA_METRICS_FILE to a JSONL path. Peaks are since the previous
reset, not additive across phases. Device/process allocations outside the torch
allocator require the launcher's separate NVML sampler. Host HWM is lifetime;
RSS/PSS are instantaneous and cannot substitute for an external host sampler.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch


def _proc_memory(path: str, fields: set[str]) -> dict[str, int]:
    values = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            name, _, rest = line.partition(":")
            if name in fields:
                count, unit = rest.split()
                if unit != "kB":
                    raise RuntimeError(f"unexpected Linux process memory unit {unit!r}")
                values[name] = int(count) * 1024
    return values


def record_qlora_phase(phase: str, *, reset_peak: bool = False) -> dict[str, Any] | None:
    """Record synchronized rank-local phase peaks, optionally reset AFTER recording.

    Disabled calls do not touch CUDA or the filesystem. The supported Linux
    single-rank recipe owns its path; multi-rank launchers must supply distinct
    rank paths if this helper is independently reused.
    """
    target = os.environ.get("COSMOS_QLORA_METRICS_FILE")
    if not target:
        return None
    if not isinstance(phase, str) or not phase:
        raise ValueError("memory phase must be a nonempty string")
    record: dict[str, Any] = {
        "protocol": "cosmos.qlora-memory.v1",
        "phase": phase,
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "0")),
        "monotonic_seconds": time.monotonic(),
        "unix_seconds": time.time(),
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "host_status": _proc_memory("/proc/self/status", {"VmRSS", "VmHWM", "VmSize"}),
        "host_rollup": _proc_memory("/proc/self/smaps_rollup", {"Rss", "Pss", "Private_Clean", "Private_Dirty"}),
        "reset_peak_after_record": reset_peak,
    }
    # No implicit CUDA initialization merely to collect process memory.
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        record["cuda"] = {
            "device": device,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
    print("[qlora-memory] " + encoded, flush=True)
    if reset_peak and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    return record
