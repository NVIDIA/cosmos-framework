# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WG/roles allow declaring a role without a body (built later by setup)."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_type_hints

import pytest

from cosmos_framework.posttrain.roles.core import factory as roles
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


@pytest.mark.L0
@pytest.mark.CPU
def test_roles_declare_without_body():
    wg = roles.trainer("policy")
    assert isinstance(wg, WorkerGroup)
    assert wg.name == "policy" and wg.role == "trainer" and wg.body is None


@pytest.mark.L0
@pytest.mark.CPU
def test_all_role_factories_optional_body() -> None:
    for fn, role in [
        (roles.env, "env"),
        (roles.trainer, "trainer"),
        (roles.rollout, "rollout"),
        (roles.reference, "reference"),
        (roles.critic, "critic"),
        (roles.reward, "reward"),
        (roles.teacher, "teacher"),
        (roles.buffer, "buffer"),
    ]:
        wg = fn("w")
        assert wg.body is None and wg.role == role


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("factory", [roles.trainer, roles.reference, roles.critic])
def test_ppo_model_role_factories_declare_typed_model_parallelism(
    factory: Callable[..., WorkerGroup],
) -> None:
    model_parallelism = (2, 4)

    hints = get_type_hints(factory)
    worker_group = factory("model", model_parallelism=model_parallelism)

    assert hints["model_parallelism"] == tuple[int, int] | None
    assert worker_group.model_parallelism == model_parallelism
    assert "model_parallelism" not in worker_group.placement


@pytest.mark.L0
@pytest.mark.CPU
def test_needs_bulk_defaults_by_role() -> None:
    # rollout/trainer compute on bulk → materialize inputs; others route metadata (§3.4)
    assert roles.rollout("p").needs_bulk is True
    assert roles.trainer("a").needs_bulk is True
    assert roles.reference("ref").needs_bulk is True
    assert roles.critic("critic").needs_bulk is True
    assert roles.env("d").needs_bulk is False
    assert roles.reward("rm").needs_bulk is False
    assert roles.teacher("teacher").needs_bulk is True
    assert roles.buffer("rb").needs_bulk is False
    assert roles.reward("rm", needs_bulk=True).needs_bulk is True  # override (model reward)
