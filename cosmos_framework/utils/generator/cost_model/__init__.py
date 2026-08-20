# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-sample training cost model for Cosmos3 VFM.

``spec`` is deliberately dependency-free (stdlib only) so the token arithmetic
can be exercised without a CUDA/torch environment. ``estimator`` adds the FLOP
and price layers on top and therefore pulls in ``cosmos_framework.tools.flops``.
"""

from cosmos_framework.utils.generator.cost_model.spec import (
    PATCH_SPATIAL,
    SPATIAL_COMPRESSION,
    TEMPORAL_COMPRESSION,
    SampleSpec,
    latent_frames,
    resolve_pixel_shape,
    vision_tokens,
)

__all__ = [
    "PATCH_SPATIAL",
    "SPATIAL_COMPRESSION",
    "TEMPORAL_COMPRESSION",
    "SampleSpec",
    "latent_frames",
    "resolve_pixel_shape",
    "vision_tokens",
]
