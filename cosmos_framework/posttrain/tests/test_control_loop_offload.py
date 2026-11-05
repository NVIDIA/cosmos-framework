# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Serial GPU offload steps between a colocated rollout and trainer: each helper moves one body and drains it."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control.loop.offload import (
    offload_cycle_enabled,
    offload_trainer,
    onload_trainer,
    prepare_trainer_weight_sync,
    sleep_rollout,
    wake_rollout,
    wake_rollout_weights,
)
from cosmos_framework.posttrain.roles.rollout.base import WeightReceiverBase
from cosmos_framework.posttrain.roles.trainer.base import BaseTrainer


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple] = []


class _Handle:
    """Mimics a mesh handle: settling it records completion, so the drain is observable."""

    def __init__(self, log: _Log, name: str, value: float | None = None) -> None:
        self._log, self._name, self._value = log, name, value

    def as_completed(self):
        self._log.events.append(("settled", self._name))
        yield ((0, 0), self._value)


class _Rollout:
    def __init__(self, log: _Log) -> None:
        self._log = log

    def sleep(self):
        self._log.events.append(("rollout.sleep",))
        return _Handle(self._log, "rollout.sleep")

    def wake_up(self, tags=None):
        self._log.events.append(("rollout.wake_up", tags))
        return _Handle(self._log, "rollout.wake_up")


class _Policy:
    def __init__(self, log: _Log) -> None:
        self._log = log

    def offload(self):
        self._log.events.append(("policy.offload",))
        return _Handle(self._log, "policy.offload", 36000.5)

    def onload(self):
        self._log.events.append(("policy.onload",))
        return _Handle(self._log, "policy.onload", 17800.0)

    def prepare_weight_sync(self):
        self._log.events.append(("policy.prepare_weight_sync",))
        return _Handle(self._log, "policy.prepare_weight_sync", 24000.25)


@pytest.mark.L0
@pytest.mark.CPU
def test_each_helper_issues_one_call_and_drains_it_before_returning():
    log = _Log()
    rollout, policy = _Rollout(log), _Policy(log)
    sleep_rollout(rollout)
    assert onload_trainer(policy) == 17800.0
    assert prepare_trainer_weight_sync(policy) == 24000.25
    wake_rollout_weights(rollout)
    assert offload_trainer(policy) == 36000.5
    wake_rollout(rollout)
    assert log.events == [
        ("rollout.sleep",),
        ("settled", "rollout.sleep"),
        ("policy.onload",),
        ("settled", "policy.onload"),
        ("policy.prepare_weight_sync",),
        ("settled", "policy.prepare_weight_sync"),
        ("rollout.wake_up", ("weights",)),
        ("settled", "rollout.wake_up"),
        ("policy.offload",),
        ("settled", "policy.offload"),
        ("rollout.wake_up", None),
        ("settled", "rollout.wake_up"),
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_bodies_without_residency_methods_are_skipped():
    class _Bare:
        pass

    sleep_rollout(_Bare())
    assert prepare_trainer_weight_sync(_Bare()) == 0.0
    wake_rollout_weights(_Bare())
    wake_rollout(_Bare())
    assert onload_trainer(_Bare()) == 0.0
    assert offload_trainer(_Bare()) == 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_base_bodies_are_inert_so_shared_model_loops_run_unchanged():
    class _Trainer(BaseTrainer):
        def __init__(self) -> None:  # skip BaseTrainer's optimizer/checkpoint wiring
            pass

        def _reduce_metrics(self, metrics):
            return metrics

    rollout = WeightReceiverBase()
    policy = _Trainer()
    sleep_rollout(rollout)
    assert onload_trainer(policy) == 0.0
    assert prepare_trainer_weight_sync(policy) == 0.0
    wake_rollout_weights(rollout)
    assert offload_trainer(policy) == 0.0
    wake_rollout(rollout)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("gpu_placement", "colocated_offload", "expected"),
    [
        ("colocated", True, True),  # the default: shared card, cycle on
        ("colocated", False, False),  # rl.colocated_offload=False: both bodies stay resident
        ("disaggregated", True, False),  # disjoint pools: nothing to hand off
        ("disaggregated", False, False),  # the knob is inert off a shared card, not an error
    ],
)
def test_cycle_runs_only_on_a_shared_card_with_the_knob_on(
    gpu_placement: str, colocated_offload: bool, expected: bool
) -> None:
    assert offload_cycle_enabled(gpu_placement=gpu_placement, colocated_offload=colocated_offload) is expected
