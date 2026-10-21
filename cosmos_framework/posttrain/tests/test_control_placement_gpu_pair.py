# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for shared GPU-pair topology lowering."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control.placement import (
    ActorGroup,
    gpu_pair_device_groups,
    normalize_gpu_placement,
    trainer_experience_device_groups,
)
from cosmos_framework.posttrain.control.placement.gpu_pair import resolve_gpu_pair_placement
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _setup() -> dict[str, object]:
    return {}


def _actor(name: str) -> ActorGroup:
    return ActorGroup((WorkerGroup(name, "env"),), _setup, name=name)


@pytest.mark.parametrize(
    ("gpu_placement", "expected_names", "expected_members"),
    [
        ("colocated", ("student_shared_gpus",), (("experience", "policy"),)),
        (
            "disaggregated",
            ("student_experience_gpus", "student_policy_gpus"),
            (("experience",), ("policy",)),
        ),
    ],
)
def test_gpu_pair_device_groups_lower_shared_and_disjoint_pools(
    gpu_placement: str,
    expected_names: tuple[str, ...],
    expected_members: tuple[tuple[str, ...], ...],
) -> None:
    experience = _actor("experience")
    policy = _actor("policy")

    groups = gpu_pair_device_groups(
        experience,
        policy,
        gpu_placement=normalize_gpu_placement(gpu_placement),
        name="student",
        left_name="experience",
        right_name="policy",
    )

    assert isinstance(groups, list)
    assert tuple(group.name for group in groups) == expected_names
    assert tuple(tuple(actor.name for actor in group.actor_groups) for group in groups) == expected_members
    assert all(group.device == "gpu" for group in groups)


def test_gpu_pair_device_groups_preserves_same_process_as_colocated_only() -> None:
    shared = _actor("shared")

    groups = gpu_pair_device_groups(
        shared,
        shared,
        gpu_placement="colocated",
        name="training",
        left_name="experience",
        right_name="policy",
    )

    assert groups[0].actor_groups == (shared,)
    with pytest.raises(ValueError, match="requires two distinct ActorGroups"):
        gpu_pair_device_groups(
            shared,
            shared,
            gpu_placement="disaggregated",
            name="training",
            left_name="experience",
            right_name="policy",
        )


@pytest.mark.parametrize(
    ("gpu_placement", "expected_relationship", "expected_group_count"),
    [("colocated", "same_gpu", 1), ("disaggregated", "cross_device", 2)],
)
def test_trainer_experience_lowering_and_relationship_share_one_resolution(
    gpu_placement: str,
    expected_relationship: str,
    expected_group_count: int,
) -> None:
    experience = _actor("experience")
    trainer = _actor("trainer")
    normalized = normalize_gpu_placement(gpu_placement)

    resolved = resolve_gpu_pair_placement(normalized)
    groups = trainer_experience_device_groups(
        experience,
        trainer,
        gpu_placement=normalized,
        name="training",
    )

    assert resolved.relationship == expected_relationship
    assert len(groups) == expected_group_count
    assert (groups[0].actor_groups == (experience, trainer)) == (resolved.relationship == "same_gpu")


def test_normalize_gpu_placement_rejects_unknown_value_with_field_name() -> None:
    with pytest.raises(ValueError, match="grid.gpu_placement.*colocated.*disaggregated.*hybrid"):
        normalize_gpu_placement("hybrid", field_name="grid.gpu_placement")
