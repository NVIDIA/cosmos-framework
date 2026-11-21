# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded, opt-in native SFT CPU/CUDA traces; disabled runs do not create a profiler."""

from __future__ import annotations

import json
import os
import threading
import time
import weakref
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

import torch


def profile_range(name: str) -> Any:
    """Name a phase only when native tracing was requested."""
    if not os.environ.get("COSMOS_SFT_PROFILE_DIR"):
        return nullcontext()
    return torch.profiler.record_function(name)


_PROFILE_PHASE = ContextVar("native_profile_phase", default="forward")
_MODEL_HOOKS = weakref.WeakKeyDictionary()
_CPU_EVENT_STATE: dict[tuple[str, int], tuple[Path, int]] = {}
_CPU_EVENT_LOCK = threading.Lock()


@contextmanager
def profile_backward():
    """Tag Python recomputation occurring within the native backward call."""
    if not os.environ.get("COSMOS_SFT_PROFILE_DIR"):
        yield
        return
    token = _PROFILE_PHASE.set("backward_recompute")
    try:
        with profile_range("trainer.backward"):
            yield
    finally:
        _PROFILE_PHASE.reset(token)


def ensure_profile_model(net: torch.nn.Module) -> None:
    """Install diagnostic-only module ranges, leaving disabled models untouched.

    These are inclusive forward ranges, not extra GPU kernel durations. A forward
    during the Python backward scope is labeled recomputation; unclassified native
    kernels remain in the raw trace rather than being assigned by assumption.
    """
    if not os.environ.get("COSMOS_SFT_PROFILE_DIR") or net in _MODEL_HOOKS:
        return
    handles = []
    inventory = []
    for name, module in net.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        kind = (
            "adapter_a"
            if leaf == "lora_A"
            else "adapter_b"
            if leaf == "lora_B"
            else "output_projection"
            if leaf == "lm_head"
            else "attention"
            if type(module).__name__.endswith("Attention")
            else None
        )
        if kind is None:
            continue
        active = threading.local()

        def before(_module, _args, *, label=kind, stack=active):
            context = profile_range(f"sft.module.{label}.{_PROFILE_PHASE.get()}")
            context.__enter__()
            if not hasattr(stack, "items"):
                stack.items = []
            stack.items.append(context)

        def after(_module, _args, _output, *, stack=active):
            if getattr(stack, "items", None):
                stack.items.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))
        inventory.append({"name": name, "class": type(module).__name__, "phase": kind})
    _MODEL_HOOKS[net] = handles
    target = Path(os.environ["COSMOS_SFT_PROFILE_DIR"]) / f"modules_{os.getpid()}.json"
    with target.open("x") as stream:
        json.dump({"protocol": "cosmos.sft-module-profile.v1", "modules": inventory}, stream, indent=2)
        stream.write("\n")


def remove_profile_hooks() -> None:
    for handles in list(_MODEL_HOOKS.values()):
        for handle in handles:
            handle.remove()
    _MODEL_HOOKS.clear()


@contextmanager
def profile_cpu_range(name: str):
    """Retain bounded CPU wall spans, including spawned preprocessing workers.

    Spans are inclusive and may overlap parent work. They contain no input text or
    tensors. File-writing overhead is outside each span and profiling is diagnostic.
    """
    target = os.environ.get("COSMOS_SFT_PROFILE_DIR")
    if not target:
        yield
        return
    began = time.monotonic_ns()
    with profile_range(name):
        try:
            yield
        finally:
            ended = time.monotonic_ns()
            pid = os.getpid()
            with _CPU_EVENT_LOCK:
                key = (target, pid)
                path, count = _CPU_EVENT_STATE.get(key, (Path(target) / f"cpu-spans_{pid}.jsonl", 0))
                if count >= 4096:
                    raise RuntimeError("diagnostic CPU profile exceeded 4096 spans per process")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x" if count == 0 else "a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "protocol": "cosmos.sft-cpu-span.v1",
                                "name": name,
                                "pid": pid,
                                "thread": threading.get_ident(),
                                "begin_monotonic_ns": began,
                                "end_monotonic_ns": ended,
                                "inclusive_wall_ns": ended - began,
                            }
                        )
                        + "\n"
                    )
                _CPU_EVENT_STATE[key] = path, count + 1


@contextmanager
def profile_processor(processor: Any):
    """Temporarily annotate the native video decoder without replacing its work."""
    if not os.environ.get("COSMOS_SFT_PROFILE_DIR"):
        yield
        return
    video = getattr(processor, "video_processor", None)
    # The native cache proxy delegates to this exact processor; a cache hit never
    # enters its decoder, and is correctly recorded as no decode work.
    video = getattr(video, "_processor", video)
    original = getattr(video, "_decode_and_sample_videos", None)
    if not callable(original):
        with profile_cpu_range("sft.processor.decode_unavailable"):
            yield
        return
    sentinel = object()
    previous = vars(video).get("_decode_and_sample_videos", sentinel)

    @wraps(original)
    def decode(*args, **kwargs):
        with profile_cpu_range("sft.video_decode_and_sample"):
            return original(*args, **kwargs)

    video._decode_and_sample_videos = decode
    try:
        with profile_cpu_range("sft.processor"):
            yield
    finally:
        if previous is sentinel:
            del video._decode_and_sample_videos
        else:
            video._decode_and_sample_videos = previous


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
            "protocol": "cosmos.sft-profile.v2",
            "diagnostic_only": True,
            "annotation_durations_overlap": True,
            "worker_cpu_spans": "cpu-spans_<pid>.jsonl; inclusive wall spans, not parent-thread time",
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
                    "device_type": str(event.device_type),
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
        try:
            if self.profiler is not None and self.started:
                self.profiler.stop()
        finally:
            self.started = False
            remove_profile_hooks()
