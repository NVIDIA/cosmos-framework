# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from typing import Any

import pytest

from cosmos_framework.posttrain.control.loop.weight_sync import r2r_weight_sync, weight_sync

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Backend:
    weight_broadcast_enabled: bool = True


class _Rollout:
    _backend: _Backend
    _weight_broadcast_spec: object

    def __init__(self, events: list[str], *, bound_backend: bool) -> None:
        self._events = events
        self._weight_broadcast_spec = object()
        if bound_backend:
            self._backend = _Backend()

    def broadcast_live_weights(self, version: int, *, verify_values: bool = False) -> None:
        self._events.append(f"r2r:{version}:verify={verify_values}")

    def prepare_weight_version(self, version: int, producer_identity: Any) -> None:
        self._events.append(f"prepare:{version}:{producer_identity}")

    def commit_weight_version(self, version: int) -> None:
        self._events.append(f"commit:{version}")


class _Policy:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def iteration(self) -> int:
        return 7

    def push_weights(self, rollout: _Rollout) -> None:
        self._events.append("p2r")

    def shared_weight_identity(self) -> None:
        return None


def test_online_relay_orders_existing_p2r_before_r2r_and_publication() -> None:
    events: list[str] = []
    rollout = _Rollout(events, bound_backend=True)

    weight_sync(rollout, _Policy(events), validation_step=7, verify_replica_values=True)

    assert events == ["p2r", "r2r:7:verify=True", "prepare:7:None", "commit:7"]


def test_standalone_r2r_supports_eval_load_then_publish() -> None:
    events: list[str] = []
    rollout = _Rollout(events, bound_backend=False)

    r2r_weight_sync(rollout, model_step=11)

    assert events == ["r2r:11:verify=False", "prepare:11:None", "commit:11"]
