# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Scene loading utilities for ludus_renderer.

For clipgt scenes, use load_clipgt_scene() from clipgt.py which provides
a ClipgtGpuScene with:
- timestamped_scene: TimestampedScene ready for GPU upload
- cameras: List of FThetaCamera intrinsics
- ego_track: EgoTrackData for pose computation

Example:
    from ludus_renderer import load_clipgt_scene

    scene = load_clipgt_scene("/path/to/clipgt/scene", device="cuda")
    renderer.upload_scene(scene.timestamped_scene)
"""

# Re-export from clipgt for convenience
from .clipgt import (
    ClipgtGpuScene,
    EgoTrackData,
    load_av2_scene,
    load_clipgt_scene,
)

__all__ = [
    "ClipgtGpuScene",
    "load_clipgt_scene",
    "load_av2_scene",
    "EgoTrackData",
]
