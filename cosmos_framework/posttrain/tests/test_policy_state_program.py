# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Contracts for diffusion applied-update state programming."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import torch

from cosmos_framework.posttrain.model.protocol import FlowMatchingPolicy
from cosmos_framework.posttrain.roles.trainer.policy_state import (
    AppliedUpdateBoundary,
    DiffusionPolicyStateProgram,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _StatefulPolicy:
    def __init__(self) -> None:
        self.net = torch.nn.Linear(1, 1)
        self.net_old = torch.nn.Linear(1, 1)
        self.net_ref = torch.nn.Linear(1, 1)
        self.net_ema = torch.nn.Linear(1, 1)
        self.events: list[str] = []

    def update_ema(self, iteration: int) -> None:
        self.events.append(f"ema:{iteration}")

    def refresh_old(self) -> None:
        self.events.append("old")

    def refresh_ref(self) -> None:
        self.events.append("ref")


def _program(policy: _StatefulPolicy) -> DiffusionPolicyStateProgram:
    return DiffusionPolicyStateProgram(
        cast(FlowMatchingPolicy, policy),
        update_ema_after_iter=True,
        update_old_every_iter=2,
        update_ref_every_iter=3,
    )


def test_policy_state_program_uses_precommit_ema_and_postcommit_refresh_clocks() -> None:
    policy = _StatefulPolicy()
    program = _program(policy)
    program.validate_ema()
    program.validate_moving_roles()
    program.initialize()

    observed: list[tuple[bool, bool]] = []
    for pre_commit_index in range(6):
        boundary = AppliedUpdateBoundary(pre_commit_index, pre_commit_index + 1)
        program.before_commit(boundary)
        events = program.after_commit(boundary)
        observed.append((events.old_policy_refreshed, events.reference_refreshed))

    assert policy.events == [
        "old",
        "ref",
        "ema:0",
        "ema:1",
        "old",
        "ema:2",
        "ref",
        "ema:3",
        "old",
        "ema:4",
        "ema:5",
        "old",
        "ref",
    ]
    assert observed == [(False, False), (True, False), (False, True), (True, False), (False, False), (True, True)]


def test_policy_state_program_projects_checkpoint_and_copy_inventories() -> None:
    policy = _StatefulPolicy()
    program = _program(policy)

    descriptors = program.descriptors()
    assert [
        (
            descriptor.name,
            descriptor.semantic_role,
            descriptor.checkpoint_component,
            descriptor.persistence,
        )
        for descriptor in descriptors
    ] == [
        ("net", "policy", "model", "model"),
        ("net_old", "old", "old", "moving_old"),
        ("net_ref", "reference", "ref", "moving_reference"),
        ("net_ema", "ema", "model", "model"),
    ]
    assert tuple(program.networks()) == tuple(descriptor.name for descriptor in descriptors)
    assert tuple(
        descriptor.name
        for descriptor in program.checkpoint_descriptors(
            model_selected=True,
            training_state_selected=True,
            materialize_old=False,
        )
    ) == ("net", "net_ref", "net_ema")
    assert tuple(
        descriptor.name
        for descriptor in program.checkpoint_descriptors(
            model_selected=False,
            training_state_selected=True,
            materialize_old=True,
        )
    ) == ("net_old", "net_ref")
    assert tuple(descriptor.name for descriptor in program.exact_resume_descriptors(old_policy_is_derived=False)) == (
        "net",
        "net_old",
        "net_ref",
        "net_ema",
    )
    assert tuple(descriptor.name for descriptor in program.exact_resume_descriptors(old_policy_is_derived=True)) == (
        "net",
        "net_ref",
        "net_ema",
    )
    assert tuple(descriptor.semantic_role for descriptor in program.copy_descriptors()) == (
        "policy",
        "old",
        "reference",
        "ema",
    )
    with pytest.raises(FrozenInstanceError):
        descriptors[0].semantic_role = "ema"  # type: ignore[misc]


def test_policy_state_program_projects_inactive_snapshot_persistence() -> None:
    policy = _StatefulPolicy()
    program = DiffusionPolicyStateProgram(
        cast(FlowMatchingPolicy, policy),
        update_ema_after_iter=False,
        update_old_every_iter=0,
        update_ref_every_iter=0,
    )

    assert [(descriptor.semantic_role, descriptor.persistence) for descriptor in program.descriptors()] == [
        ("policy", "model"),
        ("old", "copy_only"),
        ("reference", "copy_only"),
        ("ema", "offload_only"),
    ]
    assert tuple(descriptor.semantic_role for descriptor in program.copy_descriptors()) == (
        "policy",
        "old",
        "reference",
    )
    assert program.exact_resume_descriptors(old_policy_is_derived=False) == (program.descriptors()[0],)
    assert program.checkpoint_descriptors(
        model_selected=True,
        training_state_selected=True,
        materialize_old=False,
    ) == (program.descriptors()[0],)
    assert tuple(descriptor.semantic_role for descriptor in program.offload_descriptors()) == (
        "old",
        "reference",
        "ema",
    )


@pytest.mark.parametrize(
    ("pre_commit_index", "committed_step"),
    [(-1, 0), (0, 0), (0, 2), (True, 2)],
)
def test_applied_update_boundary_rejects_ambiguous_clocks(
    pre_commit_index: Any,
    committed_step: Any,
) -> None:
    with pytest.raises(ValueError):
        AppliedUpdateBoundary(pre_commit_index, committed_step)
