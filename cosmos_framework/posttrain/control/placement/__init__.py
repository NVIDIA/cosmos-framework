# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""control.placement — logical actor grouping and physical device placement (DESIGN §4)."""

from cosmos_framework.posttrain.config.placement import GpuPlacement, normalize_gpu_placement
from cosmos_framework.posttrain.control.placement.colocate import (
    ActorGroup,
    DeviceGroup,
    bind_device_groups,
    placement_gpu_count,
)
from cosmos_framework.posttrain.control.placement.gpu_pair import (
    gpu_pair_device_groups,
    resolve_gpu_pair_placement,
    resolve_online_weight_placement,
    trainer_experience_device_groups,
    validate_online_weight_transport,
)
from cosmos_framework.posttrain.control.placement.peer_chain import validate_peer_chain
from cosmos_framework.posttrain.control.placement.placement import Placement

__all__ = [
    "ActorGroup",
    "DeviceGroup",
    "GpuPlacement",
    "Placement",
    "bind_device_groups",
    "gpu_pair_device_groups",
    "normalize_gpu_placement",
    "placement_gpu_count",
    "resolve_gpu_pair_placement",
    "resolve_online_weight_placement",
    "trainer_experience_device_groups",
    "validate_online_weight_transport",
    "validate_peer_chain",
]
