# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native API boundary tests. No Isaac runtime or model weights are loaded."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.driver import run_unit
from cosmos_framework.posttrain.embodied.rollouts.zero import ZeroRollout
from cosmos_framework.posttrain.embodied.sim import SimProtocolError, SimTransient
from cosmos_framework.posttrain.embodied.sims.robolab import interface as robolab_interface
from cosmos_framework.posttrain.embodied.sims.robolab.interface import get_embodiment_map
from cosmos_framework.posttrain.embodied.sims.robolab.runtime import RoboLabSimulator
from cosmos_framework.posttrain.embodied.sims.robolab.session import RoboLabSession
from cosmos_framework.posttrain.embodied.sims.robolab.tasks import build_work_units, validate_task_metadata
from cosmos_framework.posttrain.embodied.sinks.null import NullEpisodeWriter, NullSink
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class CountingEnvironment:
    """Small native-API state machine: raw reset flags differ from frozen-lane judgments."""

    num_envs = 3
    max_episode_length = 6
    action_manager = SimpleNamespace(total_action_dim=8)
    recorder_manager: Any = None

    def __init__(self) -> None:
        self.steps = 0
        self.resets = 0
        self.commands: list[np.ndarray] = []
        self.closed = False
        self.results: dict[int, bool] = {}

    def raw(self) -> dict[str, Any]:
        image = np.full((3, 4, 4, 3), self.steps, dtype=np.uint8)
        return {
            "image_obs": {
                key: image
                for key in ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam", "head_camera")
            },
            "proprio_obs": {"arm_joint_pos": np.zeros((3, 7)), "gripper_pos": np.zeros((3, 1))},
            "viewport_cam": {"viewport": image},
        }

    def reset_eval_state(self) -> None:
        self.results.clear()

    def reset(self) -> tuple[dict[str, Any], dict]:
        self.resets += 1
        self.steps = 0
        return self.raw(), {}

    def step(self, actions: np.ndarray) -> tuple[Any, ...]:
        self.commands.append(actions.copy())
        self.steps += 1
        # Lane 2 succeeds at control step 3; lane 0 never publishes a native ending.
        if self.steps == 3:
            self.results[2] = True
        return self.raw(), np.ones(3), np.ones(3, dtype=bool), np.zeros(3, dtype=bool), {}

    def get_env_results(self) -> list[dict[str, Any]]:
        return [
            {"env_id": i, "success": self.results.get(i), "step": 3 if i in self.results else None} for i in range(3)
        ]

    def close(self) -> None:
        self.closed = True


class Session(RoboLabSession):
    def _native_step(self, commands: np.ndarray) -> tuple[Any, ...]:
        return self._env.step(commands)

    def _native_metrics(self, physical_lane: int) -> dict[str, Any]:
        return {"physical_lane": physical_lane, "native_step": self._env.steps}


def unit(instances: tuple[int, ...] = (0, 2)) -> WorkUnit:
    return WorkUnit.make(
        suite="robolab",
        scenario={
            "task": "demo",
            "max_steps": 6,
            "num_envs": 3,
            "native_batch_size": 3,
            "seed": 7,
            "action_horizon": 2,
        },
        episodes=[EpisodeKey("robolab", "demo", i, 7) for i in instances],
        est_minutes=1,
    )


def session(work: WorkUnit, env: CountingEnvironment) -> Session:
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=8, instruction="place the object")
    return Session(work, env, cfg, robot="droid", control_freq=15, generation=1)


def test_native_recorder_assigns_distinct_physical_lanes_after_reset() -> None:
    env = CountingEnvironment()
    indices: dict[int, int] = {}
    files: list[str] = []

    def set_file(filename: str) -> None:
        assert env.resets == 2
        files.append(filename)
        indices.clear()

    def set_index(index: int, *, env_ids: list[int]) -> None:
        assert files
        indices.update(dict.fromkeys(env_ids, index))

    env.recorder_manager = SimpleNamespace(set_hdf5_file=set_file, set_episode_index=set_index)
    # Requested logical lanes are a subset; native recorder identities must still
    # cover all physical lanes, including the unrequested lane in the native batch.
    assert session(unit(), env).advance_turn()
    assert files == ["data.hdf5"]
    assert indices == {i: i for i in range(env.num_envs)}


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("during_reset", [False, True])
def test_nonfinite_sim_state_reports_first_step_and_physical_lane(
    monkeypatch: pytest.MonkeyPatch, value: float, during_reset: bool
) -> None:
    env = CountingEnvironment()
    body = session(unit(), env)
    flush_steps: list[int] = []

    def flush(*, verbose: bool) -> None:
        assert not verbose
        flush_steps.append(env.steps)

    env.recorder_manager = SimpleNamespace(flush_buffer=flush)
    original_raw = env.raw

    def raw() -> dict[str, Any]:
        obs = original_raw()
        obs["proprio_obs"]["arm_joint_pos"][2, 3] = value
        return obs

    if not during_reset:
        assert body.advance_turn()
    monkeypatch.setattr(env, "raw", raw)
    command = np.arange(8, dtype=np.float32)
    with pytest.raises(SimTransient) as error:
        if during_reset:
            body.advance_turn()
        else:
            body.step({0: np.zeros(8), 1: command}, render=False)
    message = str(error.value)
    assert "episode=demo/i0002_s7_a0 physical_lane=2" in message
    assert f"step={int(not during_reset)}" in message
    assert flush_steps == [int(not during_reset)]
    if during_reset:
        assert "previous_state=None command=None" in message
    else:
        assert f"previous_state={np.zeros(8).tolist()} command={command.tolist()}" in message
        assert env.steps == 1  # Fails before another policy query or control step.
        assert not body.done(0) and not body.done(1)


@pytest.mark.parametrize("record_frames", [False, True])
@pytest.mark.parametrize("model_horizon", [1, 2, 4])
def test_image_readback_only_when_needed_keeps_policy_and_recorded_frames_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_frames: bool,
    model_horizon: int,
) -> None:
    emb = get_embodiment_map("droid")
    work = unit()
    env = CountingEnvironment()
    body = session(work, env)
    sim = RoboLabSimulator(embodiment=emb.name, n_lanes=3, output_dir=str(tmp_path))
    monkeypatch.setattr(sim, "open", lambda _: body)
    reads: list[int] = []
    original_to_numpy = robolab_interface.to_numpy

    def count_image_reads(value: Any) -> np.ndarray:
        if value.ndim == 3:
            reads.append(env.steps)
        return original_to_numpy(value)

    monkeypatch.setattr(robolab_interface, "to_numpy", count_image_reads)

    class ObservingPolicy(ZeroRollout):
        def act(
            self, obs_batch: Sequence[Mapping[str, Any]], *, context: Mapping[str, Any] | None = None
        ) -> list[np.ndarray]:
            for obs in obs_batch:
                for key in emb.cameras.values():
                    np.testing.assert_array_equal(obs[key], env.steps)
            return super().act(obs_batch, context=context)

    # Exercise the real driver and outcome writer. Observe record-view pixels
    # without starting an encoder; force_render is the same flag video enables.
    original_append = NullEpisodeWriter.append

    def append(writer: NullEpisodeWriter, frame: Any) -> None:
        if record_frames:
            for name in emb.cameras:
                np.testing.assert_array_equal(frame.obs.representations[name], frame.step_idx)
        original_append(writer, frame)

    monkeypatch.setattr(NullEpisodeWriter, "append", append)
    policy = ObservingPolicy(action_dim=emb.action_dim, horizon=model_horizon, embodiment=emb.name)
    sink = NullSink()
    result = run_unit(work, sim, policy, sink, emb, force_render=record_frames, render_every=env.max_episode_length + 1)
    assert result.error is None and len(result.episodes) == len(work.episodes)
    stride = 1 if record_frames else min(model_horizon, work.scenario["action_horizon"])
    assert set(reads) == set(range(0, env.max_episode_length + 1, stride))
    assert len(reads) == len(set(reads)) * len(work.episodes) * len(emb.cameras)


def test_skipped_image_readback_keeps_owned_images_and_current_state() -> None:
    env = CountingEnvironment()
    raw = env.raw()
    old = robolab_interface.extract_observation(raw, 0, "droid", "pick")
    for image in raw["image_obs"].values():
        image.fill(9)
    raw["proprio_obs"]["arm_joint_pos"].fill(2)
    new = robolab_interface.extract_observation(raw, 0, "droid", "pick", cached_images=old)
    assert new["wrist_cam"] is old["wrist_cam"]
    np.testing.assert_array_equal(new["wrist_cam"], 0)
    np.testing.assert_array_equal(new["state"][:7], 2)


def test_native_freeze_results_override_raw_reset_flags_and_preserve_lane_identity() -> None:
    env = CountingEnvironment()
    body = session(unit(), env)
    assert body.advance_turn() and env.resets == 2
    for step in range(1, 7):
        live = [lane for lane in range(body.n_lanes) if not body.done(lane)]
        results = body.step({lane: np.full(8, lane + 1, dtype=np.float32) for lane in live})
        for result in results:
            assert result.obs is body.observe([result.lane])[0]
        np.testing.assert_array_equal(env.commands[-1][1], 0)  # Absent physical lane stays absent.
        if step < 3:
            assert not any(body.done(lane) for lane in range(body.n_lanes))
    assert body.metrics(1)["native"] == {"env_id": 2, "success": True, "step": 3}
    assert body.metrics(0)["budget_exhausted"]
    assert not body.advance_turn()
    assert all(np.all(command[2] == 0) for command in env.commands[3:])
    body.close()
    assert env.closed
    with pytest.raises(SimProtocolError, match="closed"):
        body.observe([0])


def test_task_budget_comes_from_native_inventory_and_missing_metadata_fails(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"task_name": "demo", "episode_s": "90"}]))
    options = dict(tasks=["demo"], metadata=str(path), num_envs=3, seeds=[0, 2], control_freq=15, action_horizon=32)
    units = build_work_units(**options)
    assert len(units) == 2 and all(u.scenario["max_steps"] == 1350 for u in units)
    assert [key.instance for key in units[0].episodes] == [0, 1, 2]
    with pytest.raises(ValueError, match="native metadata"):
        build_work_units(**{**options, "tasks": ["unknown"]})
    with pytest.raises(FileNotFoundError):
        build_work_units(**{**options, "metadata": str(tmp_path / "missing.json")})


def test_native_image_task_metadata_must_match_resolved_plan(tmp_path: Path) -> None:
    metadata = tmp_path / "tasks.json"
    row = {"task_name": "demo", "episode_s": 2, "instruction": "original task"}
    metadata.write_text(json.dumps([row]))
    unit = build_work_units(
        tasks=["demo"], metadata=str(metadata), num_envs=1, seeds=[0], control_freq=15, action_horizon=4
    )[0]
    options = {"task": "demo", "metadata": str(metadata), "expected_sha": unit.scenario["native_task_sha"]}
    validate_task_metadata(**options)
    metadata.write_text(json.dumps([{**row, "instruction": "changed task"}]))
    with pytest.raises(SimProtocolError, match="differs from the metadata used to prepare evaluation"):
        validate_task_metadata(**options)


def test_caption_session_ids_survive_sparse_lanes_and_change_on_reopen() -> None:
    work = unit((2,))
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=8, instruction="pick")
    sessions = [
        Session(work, CountingEnvironment(), cfg, robot="droid", control_freq=15, generation=1, dense_caption=True)
        for _ in range(2)
    ]
    contexts = []
    for body in sessions:
        body.advance_turn()
        context = body.observe([0])[0].representations["_dense_caption"]
        assert context["env_id"] == 2 and context["episode_id"] == 1
        contexts.append(context)
        body.close()
    assert contexts[0]["session_id"] != contexts[1]["session_id"]
