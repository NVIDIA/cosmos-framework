# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing role factory (DESIGN §2.7 Surface 1).

`roles.env(...)`, `roles.rollout(...)`, `roles.trainer(...)`, `roles.reference(...)`,
`roles.critic(...)`, `roles.reward(...)`, `roles.teacher(...)`, and `roles.buffer(...)` declare WorkerGroups.
In v0 you pass a concrete `body` directly; the runner then drives them with plain imperative calls.

TODO(§2.7): allow declaring by spec (body class + args) so YAML / a graph builder
can construct the same WGs; placement kwargs (pin_to / colocate_with) are recorded
on the WG and consumed by Placement (§4).
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

# needs_bulk default per role (DESIGN §3.4): rollout/trainer run model forward passes
# (compute on bulk ⇒ materialize their inputs); env/reward/buffer route/filter on
# metadata (BulkRefs flow through). Override per WG, e.g. a model-based reward:
# `roles.reward("rm", needs_bulk=True)`.


def env(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = False,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    return WorkerGroup(name, "env", body, replicas, ranks, needs_bulk, dispatch_profile=dispatch_profile, **placement)


def rollout(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = True,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    return WorkerGroup(
        name, "rollout", body, replicas, ranks, needs_bulk, dispatch_profile=dispatch_profile, **placement
    )


def trainer(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = True,
    model_parallelism: tuple[int, int] | None = None,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    return WorkerGroup(
        name,
        "trainer",
        body,
        replicas,
        ranks,
        needs_bulk,
        model_parallelism=model_parallelism,
        dispatch_profile=dispatch_profile,
        **placement,
    )


def reference(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = True,
    model_parallelism: tuple[int, int] | None = None,
    **placement: Any,
) -> WorkerGroup:
    """Declare a frozen evaluation-only reference model role."""
    return WorkerGroup(
        name,
        "reference",
        body,
        replicas,
        ranks,
        needs_bulk,
        model_parallelism=model_parallelism,
        **placement,
    )


def critic(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = True,
    model_parallelism: tuple[int, int] | None = None,
    **placement: Any,
) -> WorkerGroup:
    """Declare an independently trainable PPO value-model role."""
    return WorkerGroup(
        name,
        "critic",
        body,
        replicas,
        ranks,
        needs_bulk,
        model_parallelism=model_parallelism,
        **placement,
    )


def reward(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = False,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    return WorkerGroup(
        name, "reward", body, replicas, ranks, needs_bulk, dispatch_profile=dispatch_profile, **placement
    )


def teacher(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = True,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    """OPD teacher (`roles/teacher/`). `needs_bulk` defaults TRUE, unlike the reward role: the teacher
    scores the COMPLETION TOKEN IDS (`transition.action`), which are bulk — left lazy it would
    iterate a `BulkRef` and die on the first batch."""
    return WorkerGroup(
        name, "teacher", body, replicas, ranks, needs_bulk, dispatch_profile=dispatch_profile, **placement
    )


def buffer(
    name: str,
    body: Any = None,
    replicas: int = 1,
    ranks: int = 1,
    needs_bulk: bool = False,
    dispatch_profile: str | None = None,
    **placement: Any,
) -> WorkerGroup:
    return WorkerGroup(
        name, "buffer", body, replicas, ranks, needs_bulk, dispatch_profile=dispatch_profile, **placement
    )
