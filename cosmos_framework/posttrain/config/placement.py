# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared config-facing GPU placement vocabulary (DESIGN §4.0)."""

from __future__ import annotations

from typing import Literal, cast

GpuPlacement = Literal["colocated", "disaggregated"]


def normalize_gpu_placement(value: object, *, field_name: str = "gpu_placement") -> GpuPlacement:
    """Validate one config-facing GPU placement value and preserve its literal type."""
    if value not in ("colocated", "disaggregated"):
        raise ValueError(f"{field_name} must be 'colocated' or 'disaggregated', got {value!r}")
    return cast(GpuPlacement, value)
