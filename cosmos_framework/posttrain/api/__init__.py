# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental public imports for the Cosmos RL post-training framework.

The namespace is the compatibility boundary; individual exports remain experimental until the
DPO and GRPO acceptance gates in ``docs/reference/compatibility.md`` promote them.
"""

from cosmos_framework.posttrain.control.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Edge,
    Manifest,
    ManifestSchemaError,
    Plane,
    RoleDecl,
    manifest_document,
    manifest_from_document,
)
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, ObsBundle, Status, Trajectory, Transition

__all__ = [
    "Edge",
    "FINAL_REWARD_KEY",
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "ManifestSchemaError",
    "ObsBundle",
    "Plane",
    "RoleDecl",
    "Status",
    "Trajectory",
    "Transition",
    "manifest_document",
    "manifest_from_document",
]
