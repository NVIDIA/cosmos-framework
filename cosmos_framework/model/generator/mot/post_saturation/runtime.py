# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Runtime facade for interactive AR post-saturation optimizations."""

from typing import Any

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.mot.post_saturation.cuda_graph import ARPostSaturationCUDAGraphManager
from cosmos_framework.model.generator.mot.post_saturation.static_compile import (
    _is_post_saturation_static_compile_frame,
)

AR_POST_SATURATION_CUDA_GRAPH_MODE = "cuda-graph"
AR_POST_SATURATION_STATIC_COMPILE_MODES = frozenset({"static-compile", AR_POST_SATURATION_CUDA_GRAPH_MODE})


class ARPostSaturationRuntime:
    """Own coarse CUDA Graph state for an installed post-saturation mode."""

    def __init__(self) -> None:
        self._cuda_graph_manager: ARPostSaturationCUDAGraphManager = ARPostSaturationCUDAGraphManager()

    def reset_for_new_generation(self) -> None:
        """Discard graph runners bound to the preceding generation's KV caches."""
        self._cuda_graph_manager.reset_for_new_generation()

    def run_cuda_graph(
        self,
        *,
        model: Any,
        kind: str,
        branch: str,
        packed_seq: PackedSequence,
        memory_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture or replay one coarse post-saturation graph."""
        return self._cuda_graph_manager.run(
            model=model,
            kind=kind,
            branch=branch,
            packed_seq=packed_seq,
            memory_info=memory_info,
        )


def get_ar_post_saturation_runtime(model: Any) -> ARPostSaturationRuntime:
    """Return the runtime installed for coarse post-saturation CUDA Graphs."""
    runtime = getattr(model, "_ar_post_saturation_runtime", None)
    if not isinstance(runtime, ARPostSaturationRuntime):
        raise RuntimeError("AR post-saturation CUDA Graph runtime was not installed")
    return runtime


def uses_ar_post_saturation_static_compile(model: Any) -> bool:
    """Return whether the configured mode uses static post-saturation layers."""
    return model.config.compile.ar_post_saturation_mode in AR_POST_SATURATION_STATIC_COMPILE_MODES


def is_ar_post_saturation_static_compile_frame(model: Any, frame_idx: int) -> bool:
    """Return whether this frame uses the static post-saturation memory path."""
    if not uses_ar_post_saturation_static_compile(model):
        return False
    return _is_post_saturation_static_compile_frame(frame_idx, model.config.kv_cache_inference_size)


def is_ar_post_saturation_cuda_graph_frame(model: Any, frame_idx: int) -> bool:
    """Return whether this frame uses coarse post-saturation CUDA Graph execution."""
    return (
        model.config.compile.ar_post_saturation_mode == AR_POST_SATURATION_CUDA_GRAPH_MODE
        and is_ar_post_saturation_static_compile_frame(model, frame_idx)
    )


def reset_ar_post_saturation_runtime_for_generation(model: Any) -> None:
    """Reset installed coarse graphs; default and static-compile modes are no-ops."""
    if model.config.compile.ar_post_saturation_mode != AR_POST_SATURATION_CUDA_GRAPH_MODE:
        return
    get_ar_post_saturation_runtime(model).reset_for_new_generation()


def run_ar_post_saturation_cuda_graph(
    model: Any,
    *,
    kind: str,
    branch: str,
    packed_seq: PackedSequence,
    memory_info: dict[str, Any],
) -> dict[str, Any]:
    """Capture or replay one installed coarse post-saturation graph."""
    return get_ar_post_saturation_runtime(model).run_cuda_graph(
        model=model,
        kind=kind,
        branch=branch,
        packed_seq=packed_seq,
        memory_info=memory_info,
    )
