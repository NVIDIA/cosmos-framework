# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in fixed-shape compilation for saturated rolling denoising."""

from typing import Any

from cosmos_framework.utils import log
from cosmos_framework.model.generator.mot.post_saturation.static_compile import (
    apply_ar_post_saturation_static_compile,
)


class RollingPostSaturation:
    """Install shared layer routers once; compile only saturated denoising reads."""

    def __init__(self, host: Any) -> None:
        self.host: Any = host
        if not getattr(host, "_rolling_static_compile_installed", False):
            apply_ar_post_saturation_static_compile(
                host.net.language_model, host.config.compile, emulate_precision_casts=True
            )
            host._rolling_static_compile_installed = True
        log.info("Rolling post-saturation static compile enabled; clean refresh stays eager.")

    def denoise(self, pack: Any, memory: Any, *, saturated: bool) -> Any:
        memory.post_saturation_static_compile = saturated
        return self.host.denoise(data_batch_packed=pack, memory=memory)
