# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""On-the-fly HR-to-LR degradation operators for super-resolution training.

The package is device-agnostic (CPU dataloader workers or GPU) and fully seeded:
``degrade_hr_to_lr(hr, profile, scale, seed)`` returns the same LR for the same
inputs on every call, and the sampled parameters are returned as a record.

Modules:
- ``kernels``: blur kernel generators (Real-ESRGAN / BasicSR lineage), numpy.
- ``diffjpeg``: torch JPEG round trip (DiffJPEG lineage), any device.
- ``ops``: per-clip primitives on ``[T,C,H,W]`` float tensors in [0, 1].
- ``profiles``: dataclass configs and the named profile registry (P0, P1, ...).
- ``degrade``: the entry point that runs a profile on an HR clip or image.
"""

from cosmos_framework.data.generator.augmentors.hr_lr_degradation.degrade import (
    DegradationResult,
    degrade_hr_to_lr,
)
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.profiles import (
    PROFILES,
    CleanResizeProfile,
    RealESRGANProfile,
    get_profile,
)

__all__ = [
    "PROFILES",
    "CleanResizeProfile",
    "DegradationResult",
    "RealESRGANProfile",
    "degrade_hr_to_lr",
    "get_profile",
]
