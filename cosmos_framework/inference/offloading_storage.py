# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared pinned storage for component and layerwise Cosmos3 offloading."""

from dataclasses import dataclass

import torch
from torch import nn


def _weight_parts(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expose physical storage, including scales hidden inside static FP8 weights."""
    if type(tensor) in (torch.Tensor, nn.Parameter):
        return {"": tensor}
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import PrototypeFloat8Tensor

    if not isinstance(tensor, PrototypeFloat8Tensor):
        raise ValueError("CPU offload supports ordinary tensors and ModelOpt static FP8 weights only")
    if tensor.qdata.dtype != torch.float8_e4m3fn or tensor.scale.numel() != 1:
        raise ValueError("CPU offload requires per-tensor ModelOpt FP8 weights")
    names, _ = tensor.__tensor_flatten__()
    return {name: getattr(tensor, name) for name in names}


def _component_layers(net: nn.Module) -> dict[str, list[list[nn.Module]]]:
    groups: dict[str, list[list[nn.Module]]] = {"reasoner": [], "generator": []}
    for layer in net.language_model.model.layers:
        for component, suffix in (("reasoner", ""), ("generator", "_moe_gen")):
            groups[component].append(
                [getattr(layer, name + suffix) for name in ("input_layernorm", "post_attention_layernorm", "mlp")]
                + [
                    getattr(layer.self_attn, name + suffix)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
                ]
            )
    return groups


def _refresh_quantized_wrapper(tensor: torch.Tensor, parts: dict[str, torch.Tensor], metadata) -> None:
    # Updating .data alone does not replace a wrapper's Python tensor attributes.
    # Move/rebind the physical parts first, then refresh its device metadata.
    with torch.inference_mode(tensor.is_inference()):
        tensor.data = type(tensor).__tensor_unflatten__(parts, metadata, tensor.shape, tensor.stride())


@torch.no_grad()
def _move_resident_tensors(net: nn.Module, grouped: set[int], device: torch.device) -> None:
    """Move shared state without materializing either offloaded decoder pathway."""
    for tensor in (*net.parameters(), *net.buffers()):
        if id(tensor) not in grouped:
            parts = _weight_parts(tensor)
            for part in parts.values():
                part.data = part.data.to(device)
            if "" not in parts:
                _, metadata = tensor.__tensor_flatten__()
                _refresh_quantized_wrapper(tensor, parts, metadata)


@dataclass
class _WeightView:
    target: torch.Tensor
    offset: int
    span: int
    shape: torch.Size
    stride: tuple[int, ...]

    def view(self, flat: torch.Tensor) -> torch.Tensor:
        # Preserve parameter semantics even when backing storage was allocated
        # inside inference_mode. A view of an inference tensor inherits its status.
        with torch.inference_mode(self.target.is_inference()):
            return torch.empty(0, dtype=flat.dtype, device=flat.device).set_(
                flat.untyped_storage(), flat.storage_offset() + self.offset, self.shape, self.stride
            )


class _LayerWeights:
    """Immutable pinned masters and layout metadata for one pathway of a layer."""

    def __init__(self, tensors: list[torch.Tensor]):
        self.views: dict[torch.dtype, list[_WeightView]] = {}
        self.host: dict[torch.dtype, torch.Tensor] = {}
        self._quantized = []
        physical = []
        for tensor in tensors:
            parts = _weight_parts(tensor)
            if "" not in parts:
                _, metadata = tensor.__tensor_flatten__()
                self._quantized.append((tensor, parts, metadata))
            physical.extend(parts.values())
        for tensor in physical:
            entries = self.views.setdefault(tensor.dtype, [])
            offset = entries[-1].offset + entries[-1].span if entries else 0
            span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
            entries.append(_WeightView(tensor, offset, span if tensor.numel() else 0, tensor.shape, tensor.stride()))
        # Complete the copies before changing any module storage.
        for dtype, entries in self.views.items():
            flat = torch.empty(entries[-1].offset + entries[-1].span, dtype=dtype, pin_memory=True)
            for entry in entries:
                flat[entry.offset : entry.offset + entry.span].copy_(
                    entry.target.detach().as_strided((entry.span,), (1,))
                )
            self.host[dtype] = flat

    def bind(self, backing: dict[torch.dtype, torch.Tensor]) -> None:
        for dtype, entries in self.views.items():
            for entry in entries:
                entry.target.data = entry.view(backing[dtype])
        for tensor, parts, metadata in self._quantized:
            _refresh_quantized_wrapper(tensor, parts, metadata)

    def restore(self) -> None:
        self.bind(self.host)


def _prepare_weights(net: nn.Module) -> tuple[dict[str, list[_LayerWeights]], set[int]]:
    """Validate CPU placement and snapshot both pathways into pinned masters."""
    grouped: set[int] = set()
    storages: set[int] = set()
    tensor_groups: dict[str, list[list[torch.Tensor]]] = {}
    for component, component_layers in _component_layers(net).items():
        tensor_groups[component] = []
        for modules in component_layers:
            tensors = list({id(t): t for m in modules for t in (*m.parameters(), *m.buffers())}.values())
            for tensor in tensors:
                if tensor.device.type != "cpu":
                    raise ValueError("Load decoder weights on CPU before installing CPU offload")
                if id(tensor) in grouped:
                    raise ValueError("CPU offload does not support aliased decoder weights")
                grouped.add(id(tensor))
                for part in _weight_parts(tensor).values():
                    if part.device.type != "cpu":
                        raise ValueError("Load decoder weights and FP8 scales on CPU before installing CPU offload")
                    storage = part.untyped_storage().data_ptr()
                    if part.numel() and storage in storages:
                        raise ValueError("CPU offload does not support aliased decoder weights")
                    if part.numel():
                        storages.add(storage)
            tensor_groups[component].append(tensors)
    groups = {}
    for component, tensors in tensor_groups.items():
        groups[component] = []
        for layer in tensors:
            weights = _LayerWeights(layer)
            weights.restore()
            groups[component].append(weights)
    return groups, grouped
