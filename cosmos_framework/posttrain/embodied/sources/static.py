# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Registration for the in-process source, so ``source="static"`` resolves by name.

``StaticUnits`` itself lives in ``embodied/source.py`` next to the protocol it implements --
it is the reference implementation, not a backend. This module exists because
``build_source`` resolves a name by importing ``embodied.sources.<name>``, and a source that
could only be constructed by direct import would force every caller of the pull loop to
branch on which source it wanted. Being able to say ``source="static"`` is what lets the
loop's tests exercise the real loop with no filesystem and no second process.
"""

from __future__ import annotations

from cosmos_framework.posttrain.embodied.registry import register_source
from cosmos_framework.posttrain.embodied.source import StaticUnits

register_source("static", StaticUnits)
