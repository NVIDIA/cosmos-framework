# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Inference-only layer streaming, outside the compiled decoder forwards."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
from torch import nn

from cosmos_framework.inference.offloading_storage import _LayerWeights, _move_resident_tensors, _prepare_weights


@dataclass
class _Slot:
    buffers: dict[torch.dtype, torch.Tensor]
    ready: torch.cuda.Event
    consumed: torch.cuda.Event
    owner: _LayerWeights | None = None
    used: bool = False


class Cosmos3LayerwiseOffload:
    """Stream the active reasoner or generator pathway through two GPU slots.

    Install after CPU checkpoint loading and regional compilation. The wrappers
    call the existing compiled forwards; stream management is never traced.
    CPU masters are immutable: eviction restores CPU views without a D2H copy.
    Slots stay alive across forwards, and are freed only after their copy and
    compute events have drained. One model instance serves one request at a time.
    """

    def __init__(self, net: nn.Module, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Layerwise offload requires a CUDA device")
        layers = list(net.language_model.model.layers)
        if not layers:
            raise ValueError("Layerwise offload requires at least one decoder layer")
        self.active_component: str | None = None
        self._executing = False
        self._closed = False
        self._slots: list[_Slot] = []
        self._wrappers: list[tuple[nn.Module, str, Callable, Callable]] = []
        self._capacities: dict[torch.dtype, int] = {}
        self.copy_stream = torch.cuda.Stream(device=self.device)

        self._groups, grouped = _prepare_weights(net)
        for groups in self._groups.values():
            for weights in groups:
                for dtype, host in weights.host.items():
                    self._capacities[dtype] = max(self._capacities.get(dtype, 0), host.numel())
        _move_resident_tensors(net, grouped, self.device)

        try:
            for index, layer in enumerate(layers):
                self._wrap(layer, "forward", index)
                self._wrap(layer, "reasoner_forward", index)
        except BaseException:
            self.close()
            raise

    def _wrap(self, layer: nn.Module, name: str, index: int) -> None:
        original = getattr(layer, name)

        @torch.compiler.disable(recursive=False)
        # Do not copy Dynamo metadata: older disable() versions unwrap it and
        # would bypass weight staging, calling the CPU-resident layer directly.
        @wraps(original, updated=())
        def forward(*args: Any, **kwargs: Any) -> Any:
            und_only = getattr(kwargs.get("memory_value"), "und_only", False)
            component = "reasoner" if name == "reasoner_forward" or und_only else "generator"
            if name == "forward" and component == "generator" and not kwargs.get("gen_only", False):
                raise ValueError("Layerwise offload requires cached text K/V for generator execution")
            return self._run(original, index, component, args, kwargs)

        setattr(layer, name, forward)
        self._wrappers.append((layer, name, original, forward))

    def activate(self, name: str) -> None:
        if self._closed:
            raise RuntimeError("Layerwise offloader is closed")
        if self._executing:
            raise RuntimeError("Cannot change offload phase during a layer forward")
        if name not in self._groups:
            raise ValueError(f"Unknown Cosmos3 offload component: {name}")
        if name != self.active_component:
            self._drain()
        self.active_component = name
        if not self._slots:
            # Allocate on compute's stream, and establish allocation readiness
            # before the copy stream writes these persistent buffers.
            try:
                slots = []
                with torch.cuda.device(self.device):
                    for _ in range(2):
                        slots.append(
                            _Slot(
                                {
                                    dtype: torch.empty(size, dtype=dtype, device=self.device)
                                    for dtype, size in self._capacities.items()
                                },
                                torch.cuda.Event(),
                                torch.cuda.Event(),
                            )
                        )
                    self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                    for slot in slots:
                        for buffer in slot.buffers.values():
                            buffer.record_stream(self.copy_stream)
                self._slots = slots
            except BaseException:
                self.active_component = None
                raise

    def _prefetch(self, weights: _LayerWeights, slot: _Slot) -> None:
        if slot.owner is weights:
            return
        if slot.used:
            self.copy_stream.wait_event(slot.consumed)
        with torch.cuda.stream(self.copy_stream):
            for dtype, host in weights.host.items():
                slot.buffers[dtype][: host.numel()].copy_(host, non_blocking=True)
            slot.ready.record(self.copy_stream)
        slot.owner = weights

    def _run(self, original: Callable, index: int, component: str, args: tuple, kwargs: dict) -> Any:
        if self._executing:
            raise RuntimeError("Concurrent or reentrant layerwise offload is unsupported")
        if torch.is_grad_enabled():
            raise RuntimeError("Layerwise offload supports inference under torch.no_grad() only")
        if self.active_component != component:
            raise RuntimeError(f"Activate {component} before executing its decoder layers")
        weights = self._groups[component][index]
        slot = self._slots[index % 2]
        compute = torch.cuda.current_stream(self.device)
        self._executing = True
        try:
            self._prefetch(weights, slot)
            compute.wait_event(slot.ready)
            # Explicit draining handles normal teardown; allocator tracking also
            # protects these buffers if the owner is discarded without close().
            for dtype in weights.host:
                slot.buffers[dtype].record_stream(compute)
            weights.bind(slot.buffers)
            # Do not wrap the last layer back to the first: for an odd layer
            # count they share a slot, and the next forward may switch phases.
            if index + 1 < len(self._groups[component]):
                self._prefetch(self._groups[component][index + 1], self._slots[(index + 1) % 2])
            try:
                return original(*args, **kwargs)
            finally:
                slot.consumed.record(compute)
                slot.used = True
                weights.restore()
                slot.owner = None
        except BaseException:
            self.offload()
            raise
        finally:
            self._executing = False

    def _drain(self) -> None:
        # Exceptional cleanup and phase boundaries may wait on the host. The
        # per-layer path uses only stream/event dependencies, never a host wait.
        self.copy_stream.synchronize()
        for slot in self._slots:
            if slot.used:
                slot.consumed.synchronize()
            slot.owner = None
            slot.used = False
        for layers in self._groups.values():
            for weights in layers:
                weights.restore()

    def offload(self) -> None:
        """Drain in-flight work, restore CPU views, and release GPU slots."""
        self._drain()
        self._slots.clear()
        self.active_component = None

    def close(self) -> None:
        """Restore original callables and CPU weights after outstanding work."""
        self.offload()
        for layer, name, original, wrapper in reversed(self._wrappers):
            if getattr(layer, name) is wrapper:
                setattr(layer, name, original)
        self._wrappers.clear()
        self._closed = True
