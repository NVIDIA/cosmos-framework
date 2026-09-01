# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Coarse CUDA Graph capture for post-saturation AR inference."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from loguru import logger as log

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState

NUM_CUDA_GRAPH_WARMUP_FORWARDS: int = 2


def _dual_kv_cache_identity(memory_info: dict[str, Any]) -> tuple[int, ...]:
    """Return the identity of the cache objects whose storage a runner uses."""
    dual_kv_cache = memory_info.get("dual_kv_cache")
    if not isinstance(dual_kv_cache, list) or not dual_kv_cache:
        raise ValueError("Coarse CUDA Graph capture requires a non-empty dual_kv_cache list")
    return tuple(id(layer_cache) for layer_cache in dual_kv_cache)


def _collect_static_input_copies(
    destination: Any,
    source: Any,
    destination_tensors: list[torch.Tensor],
    source_tensors: list[torch.Tensor],
    path: str,
) -> None:
    """Validate static input structure and collect corresponding tensor leaves."""
    if isinstance(source, torch.Tensor):
        if not isinstance(destination, torch.Tensor):
            raise TypeError(f"CUDA Graph input changed type at {path}")
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError(
                f"CUDA Graph input metadata changed at {path}: "
                f"captured={(destination.shape, destination.dtype, destination.device)}, "
                f"current={(source.shape, source.dtype, source.device)}"
            )
        destination_tensors.append(destination)
        source_tensors.append(source)
        return

    if is_dataclass(source) and not isinstance(source, type):
        if type(destination) is not type(source):
            raise TypeError(f"CUDA Graph input changed dataclass type at {path}")
        for field in fields(source):
            _collect_static_input_copies(
                getattr(destination, field.name),
                getattr(source, field.name),
                destination_tensors,
                source_tensors,
                f"{path}.{field.name}",
            )
        return

    if isinstance(source, list):
        if not isinstance(destination, list) or len(destination) != len(source):
            raise ValueError(f"CUDA Graph input list structure changed at {path}")
        for index, (destination_item, source_item) in enumerate(zip(destination, source, strict=True)):
            _collect_static_input_copies(
                destination_item,
                source_item,
                destination_tensors,
                source_tensors,
                f"{path}[{index}]",
            )
        return

    if isinstance(source, tuple):
        if not isinstance(destination, tuple) or len(destination) != len(source):
            raise ValueError(f"CUDA Graph input tuple structure changed at {path}")
        for index, (destination_item, source_item) in enumerate(zip(destination, source, strict=True)):
            _collect_static_input_copies(
                destination_item,
                source_item,
                destination_tensors,
                source_tensors,
                f"{path}[{index}]",
            )
        return

    if isinstance(source, dict):
        if not isinstance(destination, dict) or destination.keys() != source.keys():
            raise ValueError(f"CUDA Graph input mapping structure changed at {path}")
        for key in source:
            _collect_static_input_copies(
                destination[key],
                source[key],
                destination_tensors,
                source_tensors,
                f"{path}[{key!r}]",
            )
        return

    if destination != source:
        raise ValueError(f"CUDA Graph non-tensor input changed at {path}: {destination!r} != {source!r}")


def _copy_static_inputs_(destination: Any, source: Any) -> None:
    """Copy tensor leaves into a structurally identical static input object."""
    destination_tensors: list[torch.Tensor] = []
    source_tensors: list[torch.Tensor] = []
    _collect_static_input_copies(destination, source, destination_tensors, source_tensors, "pack")
    copy_groups: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for destination_tensor, source_tensor in zip(destination_tensors, source_tensors, strict=True):
        group = copy_groups.setdefault((destination_tensor.device, destination_tensor.dtype), ([], []))
        group[0].append(destination_tensor)
        group[1].append(source_tensor)
    for grouped_destinations, grouped_sources in copy_groups.values():
        torch._foreach_copy_(grouped_destinations, grouped_sources)  # each destination keeps its captured shape


class _CoarseCUDAGraphRunner:
    """Own one captured full-model graph and its stable input/output storage."""

    def __init__(
        self,
        *,
        model: Any,
        graph_name: str,
        packed_seq: PackedSequence,
        memory_info: dict[str, Any],
        capture_stream: torch.cuda.Stream,
        graph_pool: Any,
    ) -> None:
        self.graph_name: str = graph_name
        self.cache_identity: tuple[int, ...] = _dual_kv_cache_identity(memory_info)
        self.static_pack: PackedSequence = copy.deepcopy(packed_seq)
        self.static_pack.prepare_sequence_pack_metadata()
        static_memory_info = dict(memory_info)
        static_memory_info["coarse_cuda_graph"] = True
        static_memory_info["stage_gen_cache_writes"] = bool(memory_info.get("write_gen_cache", True))
        memory = model.build_memory_state(self.static_pack, static_memory_info)
        if not isinstance(memory, ARMemoryState):
            raise TypeError(f"{graph_name} CUDA Graph capture requires ARMemoryState, got {type(memory).__name__}")
        self.memory: ARMemoryState = memory

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # Warm static torch.compile kernels and their autotuning outside capture.
            for _ in range(NUM_CUDA_GRAPH_WARMUP_FORWARDS):
                model.denoise(data_batch_packed=self.static_pack, memory=self.memory)
        capture_stream.synchronize()

        self.graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=graph_pool, stream=capture_stream):
            self.static_output: dict[str, Any] = model.denoise(
                data_batch_packed=self.static_pack,
                memory=self.memory,
            )
        torch.cuda.current_stream().wait_stream(capture_stream)
        log.info(f"[AR inference] captured coarse post-saturation CUDA Graph: {graph_name}")

    def validate_cache_identity(self, memory_info: dict[str, Any]) -> None:
        """Reject replay when a new generation supplies different cache objects."""
        current_identity = _dual_kv_cache_identity(memory_info)
        if current_identity != self.cache_identity:
            raise RuntimeError(
                f"{self.graph_name} CUDA Graph runner is bound to KV caches from a different AR generation; "
                "reset the post-saturation CUDA Graph manager before replay"
            )

    def replay(self, packed_seq: PackedSequence, frame_idx: int) -> dict[str, Any]:
        """Update static inputs and replay the captured forward."""
        _copy_static_inputs_(self.static_pack, packed_seq)
        self.memory.prepare_for_coarse_cuda_graph_replay(frame_idx)
        self.graph.replay()
        if self.memory.stage_gen_cache_writes:
            self.memory.commit_staged_gen_cache(frame_idx)
        return self.static_output


class ARPostSaturationCUDAGraphManager:
    """Lazily capture separate coarse graphs for denoise and KV refresh."""

    def __init__(self) -> None:
        self._runners: dict[tuple[str, str], _CoarseCUDAGraphRunner] = {}
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph_pool: Any | None = None

    def reset_for_new_generation(self) -> None:
        """Discard graphs bound to the previous generation's cache storage."""
        had_captures = bool(self._runners)
        self._runners.clear()
        self._capture_stream = None
        self._graph_pool = None
        if had_captures:
            log.info("[AR inference] reset post-saturation CUDA Graph runners for a new generation")

    def run(
        self,
        *,
        model: Any,
        kind: str,
        branch: str,
        packed_seq: PackedSequence,
        memory_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture on first use, then replay a branch-specific coarse graph."""
        if kind not in {"denoise", "refresh"}:
            raise ValueError(f"Unsupported post-saturation CUDA Graph kind={kind!r}")
        key = (kind, branch)
        runner = self._runners.get(key)
        if runner is None:
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream()
            if self._graph_pool is None:
                # DP/CP/CFGP=1 guarantees these branch graphs replay
                # sequentially, so their temporary allocations can safely
                # share one private graph pool.
                self._graph_pool = torch.cuda.graph_pool_handle()
            runner = _CoarseCUDAGraphRunner(
                model=model,
                graph_name=f"{kind}/{branch}",
                packed_seq=packed_seq,
                memory_info=memory_info,
                capture_stream=self._capture_stream,
                graph_pool=self._graph_pool,
            )
            self._runners[key] = runner
            return runner.replay(packed_seq, int(memory_info["frame_idx"]))
        runner.validate_cache_identity(memory_info)
        return runner.replay(packed_seq, int(memory_info["frame_idx"]))
