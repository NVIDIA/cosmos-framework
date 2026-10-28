# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Utility functions for ludus_renderer."""

from __future__ import annotations

import torch
from torch import Tensor


def rgb(r: int, g: int, b: int) -> torch.Tensor:
    """Convert RGB values (0-255) to normalized float tensor."""
    return torch.tensor([r, g, b], dtype=torch.float32) / 255


def resample_timestamps(timestamps_us: Tensor, timestep_us: int, duration_us: int = 0) -> Tensor:
    """Resample timestamps with the given timestep.

    Args:
        timestamps_us: Original timestamps in microseconds (assumed sorted)
        timestep_us: New timestep in microseconds
        duration_us: Duration in microseconds (0 = use original duration)

    Returns:
        Resampled timestamps as int64 tensor
    """
    start_us = timestamps_us[0].item() if timestamps_us.dim() > 0 else int(timestamps_us)
    end_us = (
        (timestamps_us[-1].item() if timestamps_us.dim() > 0 else int(timestamps_us))
        if duration_us == 0
        else (start_us + duration_us)
    )
    return torch.tensor(
        list(range(int(start_us), int(end_us) + 1, timestep_us)),
        dtype=torch.int64,
        device=timestamps_us.device if hasattr(timestamps_us, "device") else "cpu",
    ).contiguous()


# Frame transformation matrices

FLU_TO_OPENCV_MATRIX = torch.tensor(
    [
        [0, -1, 0, 0],  # X_opencv = -Y_flu
        [0, 0, -1, 0],  # Y_opencv = -Z_flu
        [1, 0, 0, 0],  # Z_opencv = X_flu
        [0, 0, 0, 1],  # Translation unchanged
    ],
    dtype=torch.float32,
)
"""Frame transformation matrix from FLU to OpenCV convention.

This matrix transforms coordinate frames from FLU (Forward-Left-Up) to OpenCV convention:
- FLU: X=Forward, Y=Left, Z=Up
- OpenCV: X=Right, Y=Down, Z=Forward
"""

OPENCV_TO_OPENGL_MATRIX = torch.tensor(
    [
        [1, 0, 0, 0],  # X_opengl = X_opencv
        [0, -1, 0, 0],  # Y_opengl = -Y_opencv
        [0, 0, -1, 0],  # Z_opengl = -Z_opencv
        [0, 0, 0, 1],  # Translation unchanged
    ],
    dtype=torch.float32,
)
"""Frame transformation matrix from OpenCV to OpenGL convention.

This matrix transforms coordinate frames from OpenCV (Right-Down-Forward) to OpenGL convention:
- OpenCV: X=Right, Y=Down, Z=Forward
- OpenGL: X=Right, Y=Up, Z=Inward
"""

FLU_TO_OPENGL_MATRIX = OPENCV_TO_OPENGL_MATRIX @ FLU_TO_OPENCV_MATRIX
"""Frame transformation matrix from FLU to OpenGL convention.

This matrix transforms coordinate frames from FLU (Forward-Left-Up) to OpenGL convention:
- FLU: X=Forward, Y=Left, Z=Up
- OpenGL: X=Right, Y=Up, Z=Inward
"""


def projection_matrix(x: float = 0.1, n: float = 1.0, f: float = 50.0) -> torch.Tensor:
    """Returns the projection matrix for the given parameters.

    Args:
        x: Horizontal field of view in meters
        n: Near plane distance
        f: Far plane distance

    Returns:
        torch.Tensor: Projection matrix
    """
    return torch.tensor(
        [
            [n / x, 0, 0, 0],
            [0, n / x, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
    )
