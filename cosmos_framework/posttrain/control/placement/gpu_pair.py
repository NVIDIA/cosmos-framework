# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared GPU-axis lowering for a pair of framework workloads (DESIGN §4.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from cosmos_framework.posttrain.config.placement import GpuPlacement, normalize_gpu_placement
from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup

ExecutionBackend = Literal["local", "ray"]
WeightRelationship = Literal["host", "same_gpu", "cross_device"]


@dataclass(frozen=True)
class ResolvedGpuPairPlacement:
    """One normalized GPU-pair choice shared by runtime grouping and relationship projection."""

    gpu_placement: GpuPlacement
    relationship: Literal["same_gpu", "cross_device"]


def resolve_gpu_pair_placement(value: object, *, field_name: str = "gpu_placement") -> ResolvedGpuPairPlacement:
    """Resolve the single mapping from config vocabulary to shared/split GPU relationship."""
    gpu_placement = normalize_gpu_placement(value, field_name=field_name)
    relationship: Literal["same_gpu", "cross_device"] = "same_gpu" if gpu_placement == "colocated" else "cross_device"
    return ResolvedGpuPairPlacement(gpu_placement, relationship)


def resolve_online_weight_placement(
    backend_value: object,
    gpu_placement_value: object,
) -> tuple[ExecutionBackend, GpuPlacement, WeightRelationship]:
    """Resolve online trainer/experience placement without selecting a transport."""
    if backend_value not in ("local", "ray"):
        raise ValueError(f"grid.backend={backend_value!r}: expected 'local' or 'ray'")
    backend = cast(ExecutionBackend, backend_value)
    resolved = resolve_gpu_pair_placement(gpu_placement_value, field_name="grid.gpu_placement")
    if backend == "local":
        if resolved.gpu_placement == "disaggregated":
            raise ValueError("grid.gpu_placement='disaggregated' requires grid.backend='ray'")
        return backend, resolved.gpu_placement, "host"
    return backend, resolved.gpu_placement, resolved.relationship


def validate_online_weight_transport(relationship: WeightRelationship, transport: str | None) -> None:
    """Require a transport hint compatible with a resolved online weight relationship."""
    if relationship == "cross_device":
        if transport != "nixl":
            raise ValueError(
                "grid.gpu_placement='disaggregated' resolves a cross_device WEIGHT edge, which "
                "currently requires rl.weight_sync_transport='nixl'"
            )
        return
    if transport is not None:
        raise ValueError(
            f"rl.weight_sync_transport={transport!r} cannot serve resolved relationship={relationship!r}; "
            "select grid.gpu_placement='disaggregated' independently when cross-device placement is intended"
        )


def gpu_pair_device_groups(
    left: ActorGroup,
    right: ActorGroup,
    *,
    gpu_placement: GpuPlacement,
    name: str,
    left_name: str,
    right_name: str,
) -> list[DeviceGroup]:
    """Lower one explicit workload pair to a shared or two disjoint GPU DeviceGroups.

    ActorGroup membership remains the independent process axis. If both workloads intentionally
    share one ActorGroup, only colocated GPU placement is meaningful.
    """
    resolved = resolve_gpu_pair_placement(gpu_placement)
    labels = {"name": name, "left_name": left_name, "right_name": right_name}
    for label, value in labels.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    if left is right:
        if resolved.gpu_placement == "disaggregated":
            raise ValueError("gpu_placement='disaggregated' requires two distinct ActorGroups")
        return [DeviceGroup((left,), device="gpu", name=f"{name}_shared_gpus")]
    if resolved.gpu_placement == "colocated":
        return [DeviceGroup((left, right), device="gpu", name=f"{name}_shared_gpus")]
    return [
        DeviceGroup((left,), device="gpu", name=f"{name}_{left_name}_gpus"),
        DeviceGroup((right,), device="gpu", name=f"{name}_{right_name}_gpus"),
    ]


def trainer_experience_device_groups(
    experience: ActorGroup,
    trainer: ActorGroup,
    *,
    gpu_placement: GpuPlacement,
    name: str,
    trainer_name: str = "policy",
) -> list[DeviceGroup]:
    """RL-semantic wrapper for the trainer↔rollout/experience GPU relationship."""
    return gpu_pair_device_groups(
        experience,
        trainer,
        gpu_placement=gpu_placement,
        name=name,
        left_name="experience",
        right_name=trainer_name,
    )
