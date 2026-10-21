# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Generic numerical validation primitives."""

from __future__ import annotations

import torch


def validate_finite(tensor: torch.Tensor, name: str = "tensor") -> None:  # tensor: [...]
    """Fail fast when a numerical input contains NaN or infinity."""
    finite = torch.isfinite(tensor)  # [...]
    all_finite = finite.all()  # []
    if not bool(all_finite.item()):
        raise ValueError(f"{name} must contain only finite values")


__all__ = ["validate_finite"]
