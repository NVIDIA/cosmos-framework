# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The replica-broadcast probe must not be answered by `WorkerGroup.__getattr__`.

A WorkerGroup fabricates a callable for any unknown attribute, so a `getattr`-based probe reports
"bound" for every unbound rollout. That sent every local/torchrun run down the R2R branch and
failed its first weight sync with "rollout replica broadcast is not bound".
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control.loop.weight_sync import _uses_replica_broadcast
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


class _Body:
    """A plain in-process rollout body."""


@pytest.mark.L0
@pytest.mark.CPU
def test_unbound_worker_group_does_not_claim_replica_broadcast() -> None:
    rollout = WorkerGroup("rollout", "rollout", body=_Body())
    assert rollout._backend is None
    # The regression: __getattr__ hands back a lambda, so a getattr probe sees "not None".
    assert getattr(rollout, "_weight_broadcast_spec", None) is not None
    assert _uses_replica_broadcast(rollout) is False


@pytest.mark.L0
@pytest.mark.CPU
def test_plain_body_reports_its_real_spec() -> None:
    body = _Body()
    assert _uses_replica_broadcast(body) is False
    body._weight_broadcast_spec = object()
    assert _uses_replica_broadcast(body) is True


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("enabled", [True, False])
def test_bound_backend_reports_the_bind_owned_flag(enabled: bool) -> None:
    class _Backend:
        weight_broadcast_enabled = enabled

    rollout = WorkerGroup("rollout", "rollout", body=_Body())
    rollout._bind_backend(_Backend())
    assert _uses_replica_broadcast(rollout) is enabled
