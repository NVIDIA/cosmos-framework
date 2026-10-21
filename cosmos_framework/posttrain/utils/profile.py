# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in phase profiling for driver and worker hot paths."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from cosmos_framework.posttrain.utils.log import logger

PROFILE_ENV = "POSTTRAIN_PROFILE_LOOP"
PROFILE_TENSORS_ENV = "POSTTRAIN_PROFILE_TENSORS"

_nvml_handles: dict[int, Any] = {}


def profiling_enabled() -> bool:
    """Return whether loop profiling is enabled for this process."""
    return os.environ.get(PROFILE_ENV) == "1"


def gpu_memory_used_mib() -> float | None:
    """Used memory in MiB of this process's first visible GPU, as the driver reports it.

    Read through NVML rather than the torch allocator so memory held by another process on the same
    card (the colocated engine or trainer) is counted too, and without touching torch so a CPU-only
    driver process never initializes CUDA. The device is the first entry of ``CUDA_VISIBLE_DEVICES``
    (an index or a UUID) or physical device 0. Returns None when NVML is unavailable.
    """
    try:
        import pynvml

        handle = _nvml_handles.get(0)
        if handle is None:
            pynvml.nvmlInit()
            first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
            if first.startswith(("GPU-", "MIG-")):
                handle = pynvml.nvmlDeviceGetHandleByUUID(first)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(first) if first else 0)
            _nvml_handles[0] = handle
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024.0**2)
    except Exception:  # noqa: BLE001 -- profiling must never fail the run
        return None


def torch_memory_mib() -> tuple[float | None, float | None]:
    """``(allocated, reserved)`` MiB of torch's caching allocator, or Nones when CUDA is not in use here."""
    try:
        import sys

        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return None, None
        return torch.cuda.memory_allocated() / (1024.0**2), torch.cuda.memory_reserved() / (1024.0**2)
    except Exception:  # noqa: BLE001 -- profiling must never fail the run
        return None, None


def live_cuda_tensor_summary(top: int = 8) -> str | None:
    """Largest live CUDA storages in this process, grouped by (type, shape, dtype), as one token.

    Opt-in through ``POSTTRAIN_PROFILE_TENSORS=1`` because it walks every Python object. Each entry
    is ``type:shape:dtype:count:mib`` and entries are joined with ``;``; the total of all live CUDA
    storage is appended as ``total=<mib>``. Returns None when disabled or when CUDA is not in use.
    """
    if os.environ.get(PROFILE_TENSORS_ENV) != "1":
        return None
    try:
        import gc
        import sys
        from collections import defaultdict

        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return None
        seen: set[int] = set()
        groups: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0, 0.0])
        skipped = 0
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
                local = obj.to_local() if hasattr(obj, "to_local") else obj
                if local.device.type != "cuda":
                    continue
                storage = local.untyped_storage()
                key = storage.data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                shape = "x".join(str(d) for d in local.shape) or "0"
                entry = groups[(type(obj).__name__, shape, str(local.dtype).removeprefix("torch."))]
                entry[0] += 1
                entry[1] += storage.nbytes() / (1024.0**2)
            except Exception:  # noqa: BLE001 -- fake, meta or half-built tensors
                skipped += 1
        ranked = sorted(groups.items(), key=lambda item: -item[1][1])[:top]
        total = sum(entry[1] for entry in groups.values())
        rendered = ";".join(f"{t}:{shape}:{dtype}:{int(n)}:{mib:.0f}" for (t, shape, dtype), (n, mib) in ranked)
        return f"{rendered};total={total:.0f};skipped={skipped}"
    except Exception as error:  # noqa: BLE001 -- profiling must never fail the run
        return f"error={type(error).__name__}"


class ProfileRecord:
    """Collect named phase durations and emit one machine-readable log record.

    With ``progress``, each checkpoint is also logged as it ends, so a build that never reaches
    ``emit`` (stalled, cancelled or killed) still shows how far every process got.
    """

    def __init__(
        self,
        component: str,
        *,
        enabled: bool | None = None,
        progress: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not component or any(char.isspace() for char in component):
            raise ValueError(f"profile component must be one token, got {component!r}")
        self.component = component
        self.enabled = profiling_enabled() if enabled is None else enabled
        self._progress = progress
        self._clock = clock
        self._phases: dict[str, float] = {}
        self._memory: dict[str, float] = {}
        self._started = clock() if self.enabled else None
        self._checkpoint = self._started
        if self.enabled:
            self._record_memory("start")

    def _record_memory(self, name: str) -> None:
        used = gpu_memory_used_mib()
        if used is not None:
            self._memory[name] = used
        # Split this process's share: live tensors vs allocator cache vs everything else (NCCL,
        # contexts, workspaces) = device used - reserved. Only when this process already uses CUDA,
        # so a CPU-only driver never initializes it.
        allocated, reserved = torch_memory_mib()
        if allocated is not None and reserved is not None:
            self._memory[f"{name}_alloc"] = allocated
            self._memory[f"{name}_reserved"] = reserved

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one possibly non-sequential phase."""
        self._validate_name(name)
        if not self.enabled:
            yield
            return
        started = self._clock()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + self._clock() - started

    def checkpoint(self, name: str) -> None:
        """Finish the next phase in a sequential timeline."""
        self._validate_name(name)
        if not self.enabled:
            return
        now = self._clock()
        if self._checkpoint is None or self._started is None:
            raise RuntimeError("enabled profile has no checkpoint origin")
        previous = self._checkpoint
        self._phases[name] = self._phases.get(name, 0.0) + now - previous
        self._checkpoint = now
        # Device memory at the end of each phase: consecutive checkpoints bracket an offload/onload.
        self._record_memory(name)
        if self._progress:
            used = self._memory.get(name)
            logger.info(
                f"benchmark phase component={self.component} phase={name} phase_s={now - previous:.3f} "
                f"elapsed_s={now - self._started:.3f}" + ("" if used is None else f" used_mib={used:.1f}")
            )

    def set_phase(self, name: str, seconds: float = 0.0) -> None:
        """Set a phase that was intentionally skipped."""
        self._validate_name(name)
        if self.enabled:
            self._phases[name] = float(seconds)

    def emit(self, **fields: Any) -> str | None:
        """Log and return the completed record, or do nothing when disabled."""
        if not self.enabled:
            return None
        if self._started is None:
            raise RuntimeError("enabled profile has no start time")
        total = self._clock() - self._started
        parts = ["benchmark profile", f"component={self.component}"]
        parts.extend(f"{name}={self._format_value(value)}" for name, value in fields.items())
        parts.extend(f"{name}_s={seconds:.6f}" for name, seconds in self._phases.items())
        parts.extend(f"{name}_mib={mib:.1f}" for name, mib in self._memory.items())
        parts.append(f"total_s={total:.6f}")
        message = " ".join(parts)
        logger.info(message)
        return message

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or name.endswith("_s") or any(char.isspace() for char in name):
            raise ValueError(f"profile phase must be one token without an _s suffix, got {name!r}")

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.6f}"
        rendered = str(value)
        if any(char.isspace() for char in rendered):
            raise ValueError(f"profile field values must be one token, got {rendered!r}")
        return rendered


class GpuSpan:
    """Elapsed GPU time between two CUDA events on the current stream, read back without synchronizing.

    ``__enter__`` records the start event and ``__exit__`` the end event on the current stream. ``seconds()``
    answers only once both events have completed on the device (``query()``), so callers read it after a
    synchronization point the surrounding code already has; it never blocks and never inserts one. ``None``
    in CPU-only processes and while the span is still in flight.
    """

    def __init__(self) -> None:
        self._start: Any = None
        self._end: Any = None

    def __enter__(self) -> GpuSpan:
        import torch

        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._start is not None:
            import torch

            self._end = torch.cuda.Event(enable_timing=True)
            self._end.record()

    def seconds(self) -> float | None:
        """Elapsed seconds, or ``None`` when there is no CUDA or the end event has not completed yet."""
        if self._start is None or self._end is None or not self._end.query():
            return None
        return float(self._start.elapsed_time(self._end)) / 1000.0
