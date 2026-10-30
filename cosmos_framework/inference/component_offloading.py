# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-local reasoner/generator swapping for single-GPU Cosmos3 inference."""

import torch
from torch import nn

from cosmos_framework.inference.offloading_storage import _move_resident_tensors, _prepare_weights


class Cosmos3CPUOffload:
    """Keep one decoder pathway resident, backed by immutable pinned CPU weights.

    Phase swapping follows vLLM-Omni PR #4695. As in its newer PinnedModuleStager,
    eviction restores CPU references without copying unchanged GPU weights back.
    Transfers complete at phase boundaries without dedicated streams. The
    generator remains resident throughout denoising; shared modules stay on GPU.
    """

    def __init__(self, net: nn.Module, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CPU offload requires a CUDA device")
        self.active_component: str | None = None
        self._groups, grouped = _prepare_weights(net)
        self._device_weights: list[dict[torch.dtype, torch.Tensor]] = []
        _move_resident_tensors(net, grouped, self.device)

    def _load(self, name: str) -> None:
        with torch.cuda.device(self.device):
            self._device_weights = [
                {dtype: torch.empty_like(host, device=self.device) for dtype, host in weights.host.items()}
                for weights in self._groups[name]
            ]
        for weights, backing in zip(self._groups[name], self._device_weights, strict=True):
            for dtype, host in weights.host.items():
                backing[dtype].copy_(host, non_blocking=True)
        torch.cuda.synchronize(self.device)
        for weights, backing in zip(self._groups[name], self._device_weights, strict=True):
            weights.bind(backing)

    def activate(self, name: str) -> None:
        if name not in self._groups:
            raise ValueError(f"Unknown Cosmos3 offload component: {name}")
        try:
            if name != self.active_component:
                self.offload()
                self._load(name)
            self.active_component = name
        except BaseException:
            self.offload()
            raise

    def offload(self) -> None:
        """Drain the active phase and release GPU storage without a D2H copy."""
        had_device_weights = bool(self._device_weights)
        if had_device_weights:
            # Phase boundaries may follow work on a different compute stream.
            torch.cuda.synchronize(self.device)
        for layers in self._groups.values():
            for weights in layers:
                weights.restore()
        self._device_weights.clear()
        self.active_component = None
        if had_device_weights:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
