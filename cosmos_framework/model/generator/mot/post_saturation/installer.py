# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Installation entry point for interactive AR post-saturation modes."""

from typing import Any

from loguru import logger as log

from cosmos_framework.model.generator.mot.post_saturation.runtime import (
    AR_POST_SATURATION_CUDA_GRAPH_MODE,
    AR_POST_SATURATION_STATIC_COMPILE_MODES,
    ARPostSaturationRuntime,
)
from cosmos_framework.model.generator.mot.post_saturation.static_compile import (
    apply_ar_post_saturation_static_compile,
)


def install_ar_post_saturation_mode(model: Any) -> None:
    """Install optional AR post-saturation routing after checkpoint load."""
    mode = model.config.compile.ar_post_saturation_mode
    if mode == "default":
        return
    if mode not in AR_POST_SATURATION_STATIC_COMPILE_MODES:
        raise ValueError(f"Unsupported ar_post_saturation_mode={mode!r}")
    if getattr(model, "_ar_post_saturation_static_compile_installed", False):
        return

    if mode == AR_POST_SATURATION_CUDA_GRAPH_MODE and model.config.compile.enabled:
        raise ValueError(
            "ar_post_saturation_mode='cuda-graph' installs its own fixed-shape decoder compilation "
            "inside explicitly captured coarse graphs. Hence global torch.compile must be disabled in this mode. "
            "Use --no-use-torch-compile with --ar-post-saturation-mode cuda-graph."
        )
    if mode == AR_POST_SATURATION_CUDA_GRAPH_MODE and model.config.compile.use_cuda_graphs:
        raise ValueError(
            "ar_post_saturation_mode='cuda-graph' explicitly owns its capture stream, graph pool, and "
            "fixed-address input buffers. Hence global use_cuda_graphs must be disabled in this mode. "
            "Use --no-use-cuda-graphs with --ar-post-saturation-mode cuda-graph."
        )

    if model.parallel_dims is not None and model.parallel_dims.dp_enabled:
        raise ValueError(
            f"ar_post_saturation_mode={mode!r} has not been validated with FSDP/HSDP sharded "
            "decoder layers. Use dp_shard_size=1 for this mode."
        )

    if mode == AR_POST_SATURATION_CUDA_GRAPH_MODE:
        parallel_dims = model.parallel_dims
        enabled_parallelisms = [
            name
            for name in ("dp", "cp", "cfgp")
            if parallel_dims is not None and getattr(parallel_dims, f"{name}_enabled", False)
        ]
        if enabled_parallelisms:
            raise ValueError(
                f"ar_post_saturation_mode={mode!r} currently requires dp/CP/CFGP=1; enabled: "
                + ", ".join(enabled_parallelisms)
            )
        if model.config.kv_cache_dtype is not None:
            raise ValueError(
                f"ar_post_saturation_mode={mode!r} currently stages refresh writes and rebuilds fixed-address "
                "history using the uncompressed KV-cache representation. Compressed KV backends have not been "
                "integrated with that capture/replay lifecycle. Use kv_cache_dtype=None."
            )

    apply_ar_post_saturation_static_compile(model.net.language_model, model.config.compile)
    model._ar_post_saturation_static_compile_installed = True
    if mode == AR_POST_SATURATION_CUDA_GRAPH_MODE:
        model._ar_post_saturation_runtime = ARPostSaturationRuntime()
        log.info("[AR inference] installed post-saturation static compile with coarse CUDA Graph capture")
    else:
        log.info("[AR inference] installed post-saturation static torch.compile routing")
