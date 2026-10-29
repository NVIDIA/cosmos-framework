# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded, opt-in native SFT CPU/CUDA traces; disabled runs do not create a profiler."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def profile_range(name: str) -> Any:
    """Name a phase only when native tracing was requested."""
    if not os.environ.get("COSMOS_SFT_PROFILE_DIR"):
        return nullcontext()
    return torch.profiler.record_function(name)


class NativeStepProfiler:
    """Profile a bounded update window without changing training or checkpoint state.

    A directory is dedicated to each process/rank. Existing directories are refused
    so a relaunch cannot overwrite traces from another attempt. Timings from traced
    runs are diagnostic: compare throughput in separate, untraced executions.
    """

    def __init__(self) -> None:
        self.profiler: Any = None
        self.directory: Path | None = None
        self.cycles = 0
        self.started = False
        target = os.environ.get("COSMOS_SFT_PROFILE_DIR")
        if not target:
            return
        values = {}
        for name, default, minimum in (("WAIT", 2, 0), ("WARMUP", 1, 1), ("ACTIVE", 2, 1)):
            try:
                value = int(os.environ.get(f"COSMOS_SFT_PROFILE_{name}", str(default)))
            except ValueError as error:
                raise ValueError(f"COSMOS_SFT_PROFILE_{name} must be an integer") from error
            if value < minimum:
                raise ValueError(f"COSMOS_SFT_PROFILE_{name} must be >= {minimum}")
            values[name.lower()] = value
        self.directory = Path(target) / f"rank_{int(os.environ.get('RANK', '0'))}"
        self.directory.mkdir(parents=True, exist_ok=False)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_initialized():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        runtime = {
            "protocol": "cosmos.sft-profile.v1",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "schedule": values,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        (self.directory / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(**values, repeat=1),
            on_trace_ready=self._write_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    def _write_trace(self, profiler: Any) -> None:
        assert self.directory is not None
        stem = self.directory / f"window_{self.cycles}"
        profiler.export_chrome_trace(str(stem) + ".json.gz")
        rows = []
        for event in profiler.key_averages():
            rows.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "cpu_total_us": event.cpu_time_total,
                    "cpu_self_us": event.self_cpu_time_total,
                    "device_total_us": event.device_time_total,
                    "device_self_us": event.self_device_time_total,
                }
            )
        rows.sort(key=lambda row: (row["device_self_us"], row["cpu_self_us"]), reverse=True)
        Path(str(stem) + ".summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        self.cycles += 1

    def start(self) -> None:
        if self.profiler is not None:
            self.profiler.start()
            self.started = True

    def step(self) -> None:
        if self.profiler is not None and self.started:
            self.profiler.step()

    def close(self) -> None:
        if self.profiler is not None and self.started:
            self.profiler.stop()
            self.started = False
