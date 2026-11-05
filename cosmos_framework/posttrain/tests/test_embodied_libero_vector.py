# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run with the frozen simulator interpreter; real vendor scheduling and spawned processes."""

import time
from functools import partial

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.sim import SimProtocolError
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    PI05_SPATIAL_EXECUTION_PROFILE,
    execution_protocol,
    fixed_manifest,
)

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


@pytest.fixture
def session(monkeypatch):
    from libero.libero import benchmark

    from cosmos_framework.posttrain.tests.helpers.libero_vector import ToyBenchmark, ToySession

    monkeypatch.setattr(benchmark, "get_benchmark_dict", lambda: {"libero_spatial": ToyBenchmark})
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    created = ToySession(
        2,
        7,
        0,
        cases=fixed_manifest("train", profile="shared_500").cases,
        execution_horizon=5,
        max_steps=240,
        execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
    )
    try:
        yield created, protocol
    finally:
        created.close()


def test_spawned_session_orders_lanes_preserves_raw_facts_and_resets(session):
    active, protocol = session
    processes = [worker.process for worker in active._vector.workers]
    assert len({process.pid for process in processes}) == active.n_lanes
    states = active._vector.state_snapshots()
    assert [state[0]["seed"] for state in states] == [14, 15]
    expected_rng = np.random.default_rng(7)
    expected_cases = [active._cases[int(expected_rng.integers(len(active._cases)))] for _ in range(4)]
    assert [(row["task_id"], row["init_state_id"]) for row in active.sources(0)] == [
        (case.task_id, case.init_state_id) for case in expected_cases[:2]
    ]
    actions = [np.full((5, 7), lane / 10, dtype=np.float32) for lane in range(2)]
    before = active.sources(0)
    outcome = active.step_chunk(actions)
    for lane, row in enumerate(outcome):
        assert row.control_terminated == (False, False, True, False, False)
        assert row.rewards == (0.0, 0.0, 1.0, -1.0, 0.0)
        assert not row.episode_complete
        np.testing.assert_array_equal(row.action, actions[lane])
        assert row.next_observation["physics_steps"] == protocol.reset_steps + 5
    states = active._vector.state_snapshots()
    assert all(state[0]["render_count"] == 2 * (protocol.reset_steps + 1) for state in states)
    assert all(state[-1] == (True, True) for state in states)
    assert active.sources(0)[0]["episode_id"] == before[0]["episode_id"]
    for _ in range(47):
        outcome = active.step_chunk(actions)
    assert all(row.episode_complete and row.truncated and row.episode["steps"] == 240 for row in outcome)
    assert all(row.episode["success"] and row.episode["return"] == 1 for row in outcome)
    assert all(
        row.next_observation["physics_steps"] == 255 and row.observation["physics_steps"] == 15 for row in outcome
    )
    assert all(source["step_idx"] == 0 for source in active.sources(1))
    assert [(row["task_id"], row["init_state_id"]) for row in active.sources(1)] == [
        (case.task_id, case.init_state_id) for case in expected_cases[2:]
    ]
    active.close()
    active.close()
    assert all(not process.is_alive() for process in processes)


def test_spawned_late_invalid_input_is_atomic_and_child_failure_closes_every_lane(session):
    active, _ = session
    actions = [np.zeros((5, 7), dtype=np.float32) for _ in range(2)]
    invalid = actions[-1].copy()
    invalid[-1, -1] = np.nan
    before = active.state_digest()
    with pytest.raises(ValueError, match=r"actions\[1\]"):
        active.step_chunk([actions[0], invalid])
    assert active.state_digest() == before
    processes = [worker.process for worker in active._vector.workers]
    actions[-1][0, -1] = 99
    started = time.monotonic()
    with pytest.raises((SimProtocolError, BrokenPipeError, ConnectionResetError)) as retained:
        active.step_chunk(actions)
    assert retained.value.__traceback__ is not None
    assert time.monotonic() - started < 15
    assert all(not process.is_alive() for process in processes)
    assert active._vector is None
    with pytest.raises(SimProtocolError, match="closed"):
        active.observe()


def test_camera_toggle_restores_original_flags_without_resetting_observables():
    from cosmos_framework.posttrain.embodied.sims.libero.vector import CAMERAS, LaneReset
    from cosmos_framework.posttrain.tests.helpers.libero_vector import ToyLane

    lane = ToyLane(reset_steps=15, reset_gripper=-1)
    lane.reset_request = LaneReset("task", np.array(0), 3)
    try:
        lane.reset()
        original = lane.env.env._observables[CAMERAS[1]]
        original._enabled = False
        lane.camera_enabled = False
        assert all(not lane.env.env._observables[name]._enabled for name in CAMERAS)
        lane.camera_enabled = True
        assert lane.env.env._observables[CAMERAS[0]]._enabled
        assert not original._enabled
        assert lane.env.render_count == 30
        lane.camera_enabled = False
        with pytest.raises(ValueError, match="native step failure"):
            lane.step(np.array([0, 0, 0, 0, 0, 0, 97]))
        assert original._enabled is False
        assert lane.env is None
    finally:
        lane.close()


def test_partial_native_initialization_and_hung_close_release_real_workers():
    from cosmos_framework.posttrain.embodied.sims.libero.vector import LaneReset, LiberoVectorEnv
    from cosmos_framework.posttrain.tests.helpers.libero_vector import ToyLane

    factory = partial(ToyLane, reset_steps=15, reset_gripper=-1)
    vector = LiberoVectorEnv([factory, factory])
    processes = [worker.process for worker in vector.workers]
    try:
        with pytest.raises(SimProtocolError):
            vector.reset_lanes({0: LaneReset("task", np.array(0), 1), 1: LaneReset("fail-open", np.array(0), 2)})
    finally:
        vector.close()
    assert all(not process.is_alive() for process in processes)
    rebuilt = LiberoVectorEnv([factory])
    process = rebuilt.workers[0].process
    try:
        rebuilt.reset_lanes({0: LaneReset("task", np.array(0), 1)})
        rebuilt.workers[0].send(np.array([0, 0, 0, 0, 0, 0, 98]))
        started = time.monotonic()
        rebuilt.close()
        assert time.monotonic() - started < 10
        assert not process.is_alive()
    finally:
        rebuilt.close()
