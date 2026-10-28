# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Backward compatibility layer for ludus_renderer.torch.api imports.

New code should import directly from ludus_renderer:
    from ludus_renderer import LudusRenderer, load_av2_scene

This module re-exports the same symbols for backward compatibility:
    from ludus_renderer.torch.api import LudusRenderer  # Still works
"""

from ludus_renderer.clipgt import ClipgtGpuScene, load_av2_scene, load_clipgt_scene
from ludus_renderer.renderer import LudusRenderer

# Historical aliases retained by the compatibility namespace.
Av2GpuScene = ClipgtGpuScene

__all__ = [
    "LudusRenderer",
    "Av2GpuScene",
    "load_av2_scene",
    "load_clipgt_scene",
    "ClipgtGpuScene",
]
