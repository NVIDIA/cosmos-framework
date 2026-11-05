# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-episode LIBERO execution retains control facts through transient successes."""

import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_PROTOCOLS,
    PI05_SPATIAL_EXECUTION_PROFILE,
    execution_protocol,
    fixed_manifest,
    libero_seed,
)
from cosmos_framework.posttrain.embodied.sims.libero.reward import LiberoDecisionReward
from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession, LiberoSimulator
from cosmos_framework.posttrain.schema import EnvironmentFeedback

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Physics:
    def __init__(self, success_steps: frozenset[int]):
        self.success_steps = success_steps
        self.commands: list[np.ndarray] = []
        self.resets = 0
        self.steps = 0
        self.closed = False

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> None:
        self.resets += 1
        self.steps = 0

    def set_init_state(self, state: int) -> None:
        self.init_state = state

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self.commands.append(action.copy())
        self.steps += 1
        success = self.steps in self.success_steps
        return {"physics_steps": self.steps}, float(success), success, {}

    def close(self) -> None:
        self.closed = True


class _Benchmark:
    n_tasks = 10

    def get_task_init_states(self, task: int) -> range:
        return range(50)

    def get_task(self, task: int) -> SimpleNamespace:
        return SimpleNamespace(problem_folder=str(task), bddl_file="task.bddl", language=f"task {task}")


@pytest.fixture
def controlled_session(monkeypatch: pytest.MonkeyPatch):
    from cosmos_framework.posttrain.embodied.sims.libero import runtime

    # These existing tests isolate episode arithmetic; actual spawned execution
    # and its native resource lifecycle are covered in test_embodied_libero_vector.
    resolve = runtime.execution_protocol
    monkeypatch.setattr(
        runtime,
        "execution_protocol",
        lambda suite, profile: replace(resolve(suite, profile), parallel_lanes=False, last_control_render=False),
    )
    package, core, utils = ModuleType("libero"), ModuleType("libero.libero"), ModuleType("libero.libero.utils")
    core.benchmark = SimpleNamespace(get_benchmark_dict=lambda: dict.fromkeys(LIBERO_PROTOCOLS, _Benchmark))
    package.libero = core
    utils.get_libero_path = lambda _: "controlled-assets"
    for name, module in (("libero", package), ("libero.libero", core), ("libero.libero.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    opened: list[_Physics] = []
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)

    def open_env(self, lane, bddl, language):
        physics = _Physics(frozenset((protocol.reset_steps + 3,)))
        opened.append(physics)
        return physics

    monkeypatch.setattr(LiberoChunkSession, "_open_env", open_env)
    monkeypatch.setattr(LiberoChunkSession, "_views", lambda self, raw, prompt, lane: dict(raw))
    session = LiberoChunkSession(
        2,
        7,
        0,
        cases=(fixed_manifest("train", profile="shared_500").cases[0],),
        execution_horizon=protocol.execution_horizon,
        max_steps=protocol.max_episode_steps,
        execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
    )
    try:
        yield session, opened, protocol
    finally:
        session.close()


def test_fixed_episode_keeps_raw_success_and_resets_only_at_limit(controlled_session) -> None:
    session, physics, protocol = controlled_session
    assert all(env.resets == 1 for env in physics)
    reset_action = np.zeros(7, dtype=np.float32)
    reset_action[-1] = protocol.reset_gripper
    for env in physics:
        np.testing.assert_array_equal(np.stack(env.commands), np.tile(reset_action, (protocol.reset_steps, 1)))
    actions = [np.linspace(-0.8, 0.8, protocol.execution_horizon * 7).reshape(-1, 7)] * session.n_lanes
    original = actions[0].copy()
    initial_sources = session.sources(0)
    result = session.step_chunk(actions)[0]
    assert result.terminated and not result.truncated
    assert result.control_terminated == (False, False, True, False, False)
    assert result.control_truncated == (False,) * protocol.execution_horizon
    assert result.rewards == (0.0, 0.0, 1.0, -1.0, 0.0)
    assert result.episode_complete is False and result.episode is None
    assert session.sources(0)[0]["episode_id"] == initial_sources[0]["episode_id"]
    np.testing.assert_array_equal(result.action, original)
    np.testing.assert_array_equal(np.stack(physics[0].commands[-protocol.execution_horizon :]), original)
    np.testing.assert_array_equal(actions[0], original)
    for _ in range(protocol.max_episode_steps // protocol.execution_horizon - 1):
        result = session.step_chunk(actions)[0]
    assert result.episode_complete and not result.terminated and result.truncated
    assert result.control_terminated == (False,) * protocol.execution_horizon
    assert result.control_truncated == (False, False, False, False, True)
    assert result.episode["success"] is True
    assert result.episode["steps"] == protocol.max_episode_steps
    assert result.episode["return"] == 1.0
    assert result.next_observation["physics_steps"] == protocol.reset_steps + protocol.max_episode_steps
    assert result.observation["physics_steps"] == protocol.reset_steps
    assert all(env.resets == 2 for env in physics)
    assert session.sources(0)[0]["episode_id"] != initial_sources[0]["episode_id"]
    assert session.sources(0)[0]["step_idx"] == 0


def test_late_invalid_lane_does_not_execute_earlier_lane(controlled_session) -> None:
    session, physics, protocol = controlled_session
    action = np.zeros((protocol.execution_horizon, 7), dtype=np.float32)
    invalid = action.copy()
    invalid[-1, -1] = np.nan
    before = [len(env.commands) for env in physics]
    with pytest.raises(ValueError, match=r"actions\[1\]"):
        session.step_chunk([action, invalid])
    assert [len(env.commands) for env in physics] == before


def test_simultaneous_success_and_time_limit_retain_both_control_facts(controlled_session) -> None:
    session, physics, protocol = controlled_session
    for env in physics:
        env.success_steps = frozenset((protocol.reset_steps + protocol.max_episode_steps,))
    actions = [np.zeros((protocol.execution_horizon, 7), dtype=np.float32)] * session.n_lanes
    for _ in range(protocol.max_episode_steps // protocol.execution_horizon):
        result = session.step_chunk(actions)[0]
    assert result.terminated and result.truncated and result.episode_complete
    assert result.control_terminated[-1] and result.control_truncated[-1]
    assert result.episode["success"] and result.episode["return"] == 1.0


def test_profile_owns_config_identity_and_fixed_evaluation(tmp_path: Path) -> None:
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    for split in ("train", "eval"):
        fixed_manifest(split, profile="shared_500").write(tmp_path / f"{split}.json")
    config = LiberoConfig(
        execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
        execution_horizon=protocol.execution_horizon,
        max_episode_steps=protocol.max_episode_steps,
        eval_profile="shared_500",
        train_manifest=str(tmp_path / "train.json"),
        eval_manifest=str(tmp_path / "eval.json"),
        sim_python=sys.executable,
    )
    environment = LiberoEnvironment(config)
    environment.validate_configuration()
    assert config.identity()["protocol"] == protocol.identity()
    reward = LiberoDecisionReward(config.suite, config.execution_profile)
    feedback = EnvironmentFeedback((0.0, 0.0, 1.0, -1.0, 0.0), protocol.execution_horizon)
    assert reward(feedback, terminated=True, truncated=False) == sum(feedback.control_rewards)
    assert reward.identity()["execution_profile"] == config.execution_profile
    launch = environment.gateway_spec(replica=0, n_lanes=2, seed=42)
    assert launch.options["execution_profile"] == config.execution_profile
    cases = environment.evaluation_cases(0, 1)
    assert [case["task_id"] for case in cases[:10]] == list(range(10))
    rows = [dict(case, success=True, steps=protocol.max_episode_steps, **{"return": 1.0}) for case in cases]
    assert len(environment.validate_evaluation(rows)) == len(cases)
    rows[-1]["steps"] -= protocol.execution_horizon
    with pytest.raises(ValueError, match="fixed episode"):
        environment.validate_evaluation(rows)
    config.execution_horizon = LIBERO_PROTOCOLS["libero_spatial"].execution_horizon
    with pytest.raises(ValueError, match="execution protocol"):
        config.validate_sources()
    with pytest.raises(ValueError, match="execution_profile"):
        execution_protocol("libero_10", PI05_SPATIAL_EXECUTION_PROFILE)
    legacy = LiberoConfig().identity()["protocol"]
    expected = asdict(LIBERO_PROTOCOLS["libero_spatial"])
    expected.pop("action_transform")
    expected.pop("fixed_episode")
    expected.pop("parallel_lanes")
    expected.pop("last_control_render")
    assert legacy == expected
    with pytest.raises(ValueError, match="execution_profile"):
        execution_protocol("libero_spatial", "rlinf_pi05_spatial_v1")


def test_policy_seed_coordinates_do_not_overlap_and_replay_stably() -> None:
    protocol = execution_protocol("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE)
    environment = LiberoEnvironment(
        LiberoConfig(
            execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
            execution_horizon=protocol.execution_horizon,
            max_episode_steps=protocol.max_episode_steps,
            eval_profile="shared_500",
        )
    )
    coordinates = [
        (version, replica, decision) for version in range(2) for replica in range(4) for decision in range(384)
    ]

    def training_seeds() -> list[int]:
        return [
            environment.training_seed(seed=42, version=v, replica=r, n_lanes=2, num_envs=8, decision=d)
            for v, r, d in coordinates
        ]

    training = training_seeds()
    assert len(set(training)) == len(coordinates)
    assert training == training_seeds()
    bootstrap = [environment.bootstrap_seed(seed=42, version=v, replica=r) for v in range(2) for r in range(4)]
    assert len(set(bootstrap)) == len(bootstrap)
    assert not set(bootstrap).intersection(training)
    evaluation = [
        environment.evaluation_seed({"seed": 0, "case": asdict(case), "decision": decision})
        for case in fixed_manifest("eval", profile="shared_500").cases
        for decision in range(protocol.max_episode_steps // protocol.execution_horizon)
    ]
    assert len(set(evaluation)) == len(evaluation)
    context = {"seed": 0, "case": {"task_id": 0, "init_state_id": 1}, "decision": 0}
    assert environment.evaluation_seed(context) == environment.evaluation_seed(dict(context))


@pytest.mark.parametrize("suite", LIBERO_PROTOCOLS)
def test_default_suite_policy_seed_formulas_are_unchanged(suite: str) -> None:
    protocol = execution_protocol(suite)
    environment = LiberoEnvironment(
        LiberoConfig(
            suite=suite,
            execution_horizon=protocol.execution_horizon,
            max_episode_steps=protocol.max_episode_steps,
        )
    )
    seed, version, replica, decision = 42, 2, 3, 11
    context = {"seed": seed, "case": {"task_id": 2, "init_state_id": 7}, "decision": decision}
    if suite == "libero_spatial":
        expected_training = seed + (version * 8 + replica * 2) * 16 + decision
        expected_bootstrap = seed
        expected_evaluation = seed + (2 * 50 + 7) * 16 + decision
    else:
        expected_training = libero_seed(suite, "training-policy", seed, replica, version, decision)
        expected_bootstrap = libero_seed(suite, "bootstrap-value", seed, replica, version)
        expected_evaluation = libero_seed(suite, "evaluation-policy", seed, 2, 7, decision)
    assert (
        environment.training_seed(seed=seed, version=version, replica=replica, n_lanes=2, num_envs=8, decision=decision)
        == expected_training
    )
    assert environment.bootstrap_seed(seed=seed, version=version, replica=replica) == expected_bootstrap
    assert environment.evaluation_seed(context) == expected_evaluation


def test_simulator_profile_uses_rlinf_rank_seed(controlled_session, monkeypatch: pytest.MonkeyPatch) -> None:
    # This isolates seed routing; the dependency/source probe has its own binding tests.
    monkeypatch.setattr(LiberoSimulator, "probe", lambda self: None)
    for suite, profile in (
        ("libero_spatial", PI05_SPATIAL_EXECUTION_PROFILE),
        ("libero_spatial", "default"),
        ("libero_10", "default"),
    ):
        protocol = execution_protocol(suite, profile)
        environment = LiberoEnvironment(
            LiberoConfig(
                suite=suite,
                execution_profile=profile,
                execution_horizon=protocol.execution_horizon,
                max_episode_steps=protocol.max_episode_steps,
                eval_profile="shared_500",
            )
        )
        spec = environment.gateway_spec(replica=3, n_lanes=2, seed=42)
        simulator = LiberoSimulator(embodiment=spec.embodiment, **spec.options)
        evaluation = None
        try:
            session = simulator.open_window(2)
            case = asdict(fixed_manifest("eval", suite, "shared_500").cases[0])
            evaluation = simulator.open_evaluation(case, seed=0)
            if profile == PI05_SPATIAL_EXECUTION_PROFILE:
                assert spec.options["seed"] == 0
                assert session._seed == 3
                assert [physics.seed_value for physics in session._envs] == [6, 7]
                assert evaluation._seed == 3
            else:
                expected = (
                    2 * 8 + 6
                    if suite == "libero_spatial"
                    else libero_seed(suite, "training-environment", 42, 2, 6) % 2**32
                )
                assert session._seed == expected
                assert evaluation._seed == 0
        finally:
            if evaluation is not None:
                evaluation.close()
            simulator.close()
