# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Ported from NVIDIA/TensorRT-LLM PR #14095
# (tensorrt_llm/_torch/visual_gen/offloading.py).
"""Module parameter offloading utilities for single-GPU inference.

The offload path keeps model loading and quantization unchanged: weights are
loaded into the modules first, then selected module groups are copied into
packed CPU storage. At runtime one group at a time is staged into a reusable GPU
arena and the original module parameters/buffers are rebound to views of that
storage.

This is the model-agnostic core (``ModuleOffloadManager`` + ``OffloadPipeline``).
The Cosmos3-specific wiring (which modules form the offload groups, and where
each group is staged during a generation) lives in ``OmniInference``
(``cosmos_framework/inference/inference.py``).
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _align_offset(offset: int, alignment: int = 256) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


# FlashInfer and other custom kernels can require tensor data pointers to be at
# least 16-byte aligned even for smaller dtypes such as BF16.
_PACKED_TENSOR_ALIGNMENT = 16


OffloadPipelineStage = tuple[str, ...]


@dataclass
class _FlatTensorSpec:
    owner: nn.Module
    name: str
    qualified_name: str
    is_parameter: bool
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    persistent: bool
    offset: int
    nbytes: int


@dataclass
class _GroupLayout:
    """Packed storage layout and rebound views for one offload group."""

    name: str
    nbytes: int
    specs: list[_FlatTensorSpec]
    cpu_storage: torch.Tensor | None = None
    cpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()
    gpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()


class ModuleOffloadManager:
    """Pack module groups into CPU storage and stage one group on GPU.

    The manager owns packed byte buffers:
    - each layout owns ``cpu_storage`` for one offloaded group.
    - ``gpu_arena`` is reused for whichever group is currently active.

    Initializing the manager packs and rebinds one group at a time. This
    requires enough host memory to allocate the current group's packed CPU
    storage before that group's original tensors are released.
    """

    def __init__(
        self,
        groups: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not groups:
            raise ValueError("At least one offload group must be provided")

        self.groups = dict(groups)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.gpu_arena: torch.Tensor | None = None
        self.layouts: dict[str, _GroupLayout] = {}
        self.active_group_name: str | None = None

        for name, module in self.groups.items():
            if not name:
                raise ValueError("Offload group names must be non-empty")
            if not isinstance(module, nn.Module):
                raise TypeError(f"Offload group '{name}' must contain an nn.Module")

    @staticmethod
    def _owner_and_name(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
        if "." not in qualified_name:
            return root, qualified_name
        module_path, name = qualified_name.rsplit(".", 1)
        return root.get_submodule(module_path), name

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[int, int] | None:
        if tensor.numel() == 0:
            return None
        storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
        return tensor.untyped_storage().data_ptr(), storage_offset_bytes

    def _get_alias_spec(
        self,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        tensor: torch.Tensor,
        display_name: str,
    ) -> _FlatTensorSpec | None:
        key = self._storage_key(tensor)
        if key is None:
            return None
        canonical = seen_tensors.get(key)
        if canonical is None:
            return None
        if self._tensor_nbytes(tensor) != canonical.nbytes or tensor.dtype != canonical.dtype:
            raise ValueError(
                "Shared parameters or buffers with different sizes or dtypes are "
                f"not supported by ModuleOffloadManager: '{display_name}' aliases "
                f"'{canonical.qualified_name}'"
            )
        return canonical

    def _build_spec(
        self,
        group_name: str,
        group_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        offset: int,
    ) -> _FlatTensorSpec:
        display_name = f"{group_name}.{qualified_name}"
        if not tensor.is_contiguous():
            raise ValueError(
                f"Cannot offload non-contiguous tensor '{display_name}' with stride {tuple(tensor.stride())}"
            )

        owner, name = self._owner_and_name(group_module, qualified_name)
        return _FlatTensorSpec(
            owner=owner,
            name=name,
            qualified_name=display_name,
            is_parameter=is_parameter,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad if is_parameter else False,
            persistent=True if is_parameter else name not in owner._non_persistent_buffers_set,
            offset=offset,
            nbytes=self._tensor_nbytes(tensor),
        )

    def _iter_group_tensors(self, group_module: nn.Module) -> Iterator[tuple[str, torch.Tensor, bool]]:
        for qualified_name, param in group_module.named_parameters(recurse=True, remove_duplicate=False):
            yield qualified_name, param.detach(), True
        for qualified_name, buffer in group_module.named_buffers(recurse=True, remove_duplicate=False):
            yield qualified_name, buffer.detach(), False

    def _append_layout_spec(
        self,
        group_name: str,
        group_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        offset: int,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        specs: list[_FlatTensorSpec],
    ) -> int:
        """Append a tensor spec and return the next group-local byte offset.

        This handles three layout concerns in one place: alias reuse, packed
        tensor alignment, and spec construction. The offset is local to this
        group and is shared by the CPU storage and reusable GPU arena views.
        """
        display_name = f"{group_name}.{qualified_name}"
        alias = self._get_alias_spec(seen_tensors, tensor, display_name)
        if alias is None:
            offset = _align_offset(offset, _PACKED_TENSOR_ALIGNMENT)

        spec = self._build_spec(
            group_name=group_name,
            group_module=group_module,
            qualified_name=qualified_name,
            tensor=tensor,
            is_parameter=is_parameter,
            offset=alias.offset if alias is not None else offset,
        )
        specs.append(spec)

        if alias is not None:
            return offset

        key = self._storage_key(tensor)
        if key is not None:
            seen_tensors[key] = spec
        return offset + spec.nbytes

    def _collect_group_layout(self, group_name: str, group_module: nn.Module) -> _GroupLayout:
        """Build the packed storage layout for one named module group."""
        offset = 0
        specs: list[_FlatTensorSpec] = []
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec] = {}

        for qualified_name, tensor, is_parameter in self._iter_group_tensors(group_module):
            offset = self._append_layout_spec(
                group_name=group_name,
                group_module=group_module,
                qualified_name=qualified_name,
                tensor=tensor,
                is_parameter=is_parameter,
                offset=offset,
                seen_tensors=seen_tensors,
                specs=specs,
            )

        if not specs:
            raise ValueError(f"Offload group '{group_name}' has no parameters or buffers")

        return _GroupLayout(name=group_name, nbytes=_align_offset(offset), specs=specs)

    def _copy_group_to_cpu_storage(self, layout: _GroupLayout) -> None:
        if layout.cpu_storage is None:
            raise RuntimeError(f"CPU storage for offload group '{layout.name}' has not been allocated")
        for spec in layout.specs:
            if spec.nbytes == 0:
                continue
            try:
                tensor = getattr(spec.owner, spec.name).detach()
                tensor_bytes = tensor.reshape(-1).view(torch.uint8).cpu()
                layout.cpu_storage.narrow(0, spec.offset, spec.nbytes).copy_(tensor_bytes)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to copy offload tensor '{spec.qualified_name}' "
                    f"({_format_bytes(spec.nbytes)}, shape={spec.shape}, dtype={spec.dtype}) "
                    f"to packed CPU storage at offset {spec.offset}."
                ) from e

    def _group_size_summary(self) -> str:
        return ", ".join(f"{name}={_format_bytes(layout.nbytes)}" for name, layout in self.layouts.items())

    def _cuda_allocation_hint(self) -> str:
        if self.device.type != "cuda":
            return ""
        return (
            " If this is due to CUDA memory fragmentation, try setting "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting the process."
        )

    def _allocate_cpu_storage(self, num_bytes: int, group_name: str | None = None) -> torch.Tensor:
        try:
            return torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=self.pin_memory)
        except RuntimeError as e:
            group_context = f", group='{group_name}'" if group_name is not None else ""
            raise RuntimeError(
                "Failed to allocate packed CPU storage for Cosmos3 offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes{group_context}, "
                f"pin_memory={self.pin_memory}, groups=[{self._group_size_summary()}])."
            ) from e

    def _allocate_gpu_arena(self, num_bytes: int) -> torch.Tensor:
        try:
            return torch.empty(num_bytes, dtype=torch.uint8, device=self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to allocate GPU arena for Cosmos3 offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes, "
                f"device={self.device}, groups=[{self._group_size_summary()}])."
                f"{self._cuda_allocation_hint()}"
            ) from e

    def _make_views(
        self,
        layout: _GroupLayout,
        storage: torch.Tensor,
    ) -> tuple[nn.Parameter | torch.Tensor, ...]:
        views: list[nn.Parameter | torch.Tensor] = []
        for spec in layout.specs:
            view = storage.narrow(0, spec.offset, spec.nbytes).view(spec.dtype)
            view = view.as_strided(spec.shape, spec.stride)
            if spec.is_parameter:
                views.append(nn.Parameter(view, requires_grad=spec.requires_grad))
            else:
                views.append(view)
        return tuple(views)

    def _bind_views(
        self,
        layout: _GroupLayout,
        views: tuple[nn.Parameter | torch.Tensor, ...],
    ) -> None:
        for spec, view in zip(layout.specs, views, strict=True):
            if spec.is_parameter:
                if not isinstance(view, nn.Parameter):
                    raise TypeError(
                        f"Expected offload view '{spec.name}' to be an nn.Parameter, got {type(view).__name__}"
                    )
                spec.owner.register_parameter(spec.name, view)
            else:
                if not isinstance(view, torch.Tensor):
                    raise TypeError(
                        f"Expected offload view '{spec.name}' to be a torch.Tensor, got {type(view).__name__}"
                    )
                spec.owner.register_buffer(spec.name, view, persistent=spec.persistent)

    def initialize(self) -> None:
        """Allocate packed storage, copy current tensors, and bind CPU views."""
        if self.layouts:
            raise RuntimeError("ModuleOffloadManager has already been initialized")

        start_time = time.time()
        for name, module in self.groups.items():
            layout = self._collect_group_layout(name, module)
            self.layouts[name] = layout

        total_cpu_bytes = sum(layout.nbytes for layout in self.layouts.values())
        max_gpu_bytes = max(layout.nbytes for layout in self.layouts.values())
        logger.info(
            "Module offload storage layout: "
            f"cpu_total={_format_bytes(total_cpu_bytes)}, "
            f"gpu_arena={_format_bytes(max_gpu_bytes)}, "
            f"groups=[{self._group_size_summary()}], device={self.device}"
        )

        # Pack and rebind one group at a time. This keeps setup simple and fast:
        # offloading requires enough host memory to allocate one group's packed
        # CPU storage before that group's original tensors are released.
        for layout in self.layouts.values():
            logger.info(
                f"Module offload packing group into CPU storage: {layout.name} ({_format_bytes(layout.nbytes)})"
            )
            layout.cpu_storage = self._allocate_cpu_storage(layout.nbytes, group_name=layout.name)
            self._copy_group_to_cpu_storage(layout)
            layout.cpu_views = self._make_views(layout, layout.cpu_storage)
            self._rebind_to_cpu(layout.name)

        # Every group's parameters/buffers were just rebound to CPU storage, so the
        # modules' original on-device weights are now unreferenced. Release that freed
        # device memory back to the driver before allocating the arena, so the arena
        # (and later activations) reclaim it instead of stacking on top of the
        # caching allocator's now-fragmented free blocks. Without this, offloading can
        # use MORE device memory than the joint path (the freed weights are cached, not
        # returned, and large contiguous activations can't reuse the fragmented blocks).
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        self.gpu_arena = self._allocate_gpu_arena(max_gpu_bytes)
        for layout in self.layouts.values():
            layout.gpu_views = self._make_views(layout, self.gpu_arena)

        logger.info(f"Module offload setup completed in {time.time() - start_time:.2f}s")

    def _get_layout(self, name: str) -> _GroupLayout:
        try:
            return self.layouts[name]
        except KeyError as e:
            raise KeyError(f"Unknown offload group '{name}'. Available groups: {sorted(self.layouts)}") from e

    def stage(self, name: str) -> None:
        """Stage one offload group into the GPU arena and rebind its tensors."""
        layout = self._get_layout(name)
        if self.active_group_name == name:
            return
        if layout.cpu_storage is None or self.gpu_arena is None:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")

        if self.active_group_name is not None:
            self._rebind_to_cpu(self.active_group_name)
            self.active_group_name = None

        src = layout.cpu_storage.narrow(0, 0, layout.nbytes)
        dst = self.gpu_arena.narrow(0, 0, layout.nbytes)
        try:
            dst.copy_(src, non_blocking=layout.cpu_storage.is_pinned())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to stage offload group '{name}' ({_format_bytes(layout.nbytes)}) to {self.device}"
            ) from e
        self._rebind_to_gpu(name)
        self.active_group_name = name

    def _rebind_to_cpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.cpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.cpu_views)

    def _rebind_to_gpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.gpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.gpu_views)


class OffloadPipeline:
    """Stage offload groups explicitly from model call-site contexts.

    This class intentionally does not use forward hooks. Pipeline code must wrap
    the relevant call site with ``with offload_pipeline.context("group")`` so
    staging happens before the model invocation and outside any later CUDA graph
    capture.
    """

    def __init__(
        self,
        stages: Sequence[Sequence[str] | str],
        parts: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one offload pipeline stage must be provided")

        self.stages = tuple((stage,) if isinstance(stage, str) else tuple(stage) for stage in stages)
        self.parts = dict(parts)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.manager = ModuleOffloadManager(
            groups=self._build_groups(),
            device=self.device,
            pin_memory=self.pin_memory,
        )
        self._group_name_by_part = {part: self._stage_name(stage) for stage in self.stages for part in stage}

    def _build_groups(self) -> dict[str, nn.Module]:
        groups: dict[str, nn.Module] = {}
        for stage in self.stages:
            group_name = self._stage_name(stage)
            if not stage:
                raise ValueError("Offload pipeline stages must have at least one part")
            if group_name in groups:
                raise ValueError(f"Duplicate offload pipeline stage: {group_name}")

            modules: list[nn.Module] = []
            for part_name in stage:
                try:
                    part = self.parts[part_name]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown offload pipeline part '{part_name}' for stage "
                        f"'{group_name}'. Available parts: {sorted(self.parts)}"
                    ) from e
                modules.append(part)

            group_module = modules[0] if len(modules) == 1 else nn.ModuleList(modules)
            groups[group_name] = group_module

        return groups

    def initialize(self) -> None:
        """Allocate and populate backing storage for all configured stages."""
        self.manager.initialize()

    @staticmethod
    def _stage_name(stage: Sequence[str] | str) -> str:
        return stage if isinstance(stage, str) else "+".join(stage)

    def context(self, part_or_group_name: str):
        """Stage the group containing ``part_or_group_name`` and return a no-op context.

        The active group intentionally stays resident after the call site; the next
        ``stage()`` rebinds it back to CPU before staging another group, and
        ``cleanup()`` handles the final rebind when the pipeline exits.
        """
        group_name = self._group_name_by_part.get(part_or_group_name, part_or_group_name)
        self.manager.stage(group_name)
        return nullcontext()

    def has_part(self, part_name: str) -> bool:
        return part_name in self._group_name_by_part

    def cleanup(self) -> None:
        """Return the active group to CPU-backed views."""
        if self.manager.active_group_name is not None:
            self.manager._rebind_to_cpu(self.manager.active_group_name)
            self.manager.active_group_name = None
