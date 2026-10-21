# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""control — CONTROL PLANE (DESIGN §2, §4, §5; §9).

Imperative control, NO graph: topology is explicit WG + channel wiring; placement/bind
and the buffer live here. Runs in the controller/driver process.

Present: typed ActorGroup/DeviceGroup placement, buffer (in-memory), and bind_check — the
GPU-less AOT validation over the declared Manifest (wiring + per-edge fields now;
placement, transport, and temperature checks land as their rejection classes are added).
TODO: the runtime placement/transport solver; dispatch; cluster_state; quiesce.
"""

from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.buffer import ExperienceBuffer, MetadataGate
from cosmos_framework.posttrain.control.capability import CapabilityDescriptor
from cosmos_framework.posttrain.control.manifest import Manifest
from cosmos_framework.posttrain.control.placement import (
    ActorGroup,
    DeviceGroup,
    GpuPlacement,
    Placement,
    bind_device_groups,
    gpu_pair_device_groups,
    trainer_experience_device_groups,
)

# Recipe-specific builders (e.g. dpo_manifest) live in control.recipe and are imported from there
# directly — the framework package does not re-export them.
__all__ = [
    "ActorGroup",
    "DeviceGroup",
    "ExperienceBuffer",
    "MetadataGate",
    "GpuPlacement",
    "bind_device_groups",
    "gpu_pair_device_groups",
    "trainer_experience_device_groups",
    "Placement",
    "Manifest",
    "CapabilityDescriptor",
    "bind_check",
]
