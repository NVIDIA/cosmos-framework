# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Normalize square or rectangular spatial patches at config boundaries."""


def normalize_spatial_patch_hw(patch_hw: int | tuple[int, int]) -> tuple[int, int]:
    """Accept a square side or (height, width), including Hydra's sequence values."""
    height, width = (patch_hw, patch_hw) if isinstance(patch_hw, int) else patch_hw
    if any(not isinstance(side, int) or isinstance(side, bool) or side <= 0 for side in (height, width)):
        raise ValueError("Spatial patch height and width must be positive integers")
    return height, width
