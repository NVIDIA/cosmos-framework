# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.rollouts.droid import DroidEndpoint
from cosmos_framework.posttrain.embodied.rollouts.zero import ZeroRollout
from cosmos_framework.posttrain.embodied.sims.behavior1k.tasks import build_work_units
from cosmos_framework.posttrain.embodied.sims.robolab.interface import get_embodiment_map
from cosmos_framework.posttrain.embodied.sims.robolab.runtime import RoboLabSimulator
from cosmos_framework.posttrain.entrypoints import policy_eval
from cosmos_framework.posttrain.tests.test_embodied_robolab import CountingEnvironment, session, unit


@pytest.mark.parametrize(("flags", "expected"), [([], "default"), (["--instruction-type", "specific"], "specific")])
def test_cli_passes_instruction_variant_to_simulator(tmp_path, monkeypatch, flags, expected):
    metadata = tmp_path / "tasks.json"
    metadata.write_text(json.dumps([{"task_name": "demo", "episode_s": 1}]))
    build_sim = Mock()
    monkeypatch.setattr(policy_eval, "build_sim", build_sim)
    monkeypatch.setattr(policy_eval, "evaluate", lambda *args: {})
    policy_eval.main(
        [
            "robolab",
            "--task",
            "demo",
            "--metadata",
            str(metadata),
            "--output",
            str(tmp_path / "run"),
            "--seed",
            "0",
            "--policy-factory",
            "builtins:object",
            *flags,
        ]
    )
    assert build_sim.call_args.kwargs["instruction_type"] == expected


def test_robolab_reports_real_episode_keys_and_native_outcomes(tmp_path, monkeypatch):
    work = unit()
    emb = get_embodiment_map("droid")
    sim = RoboLabSimulator(embodiment=emb.name, n_lanes=3, output_dir=str(tmp_path))
    monkeypatch.setattr(sim, "open", lambda requested: session(requested, CountingEnvironment()))
    policy = ZeroRollout(emb.action_dim, 4, embodiment=emb.name)
    report = policy_eval.evaluate([work], sim, policy, emb, tmp_path)
    assert report["planned"] == report["completed"] == 2
    assert report["success_rate"] == 0.5
    records = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert {row["instance"]: row["outcome"] for row in records} == {0: "failure", 2: "success"}
    assert json.loads((tmp_path / "summary.json").read_text()) == report


def test_incomplete_work_never_publishes_a_score(tmp_path, monkeypatch):
    work = unit()
    emb = get_embodiment_map("droid")
    monkeypatch.setattr(policy_eval, "run_unit", lambda *a, **kw: SimpleNamespace(episodes=[], error=None))
    with pytest.raises(RuntimeError, match="completed 0"):
        policy_eval.evaluate([work], None, ZeroRollout(8, embodiment=emb.name), emb, tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_behavior_tasks_get_budget_from_native_metadata(tmp_path):
    table = tmp_path / "task.jsonl"
    table.write_text(json.dumps({"task_name": "example", "length": 40}) + "\n")
    units = build_work_units("example", 2, 1, None, None, seed=9, stats_path=str(table))
    assert [u.scenario["max_steps"] for u in units] == [60, 60]
    assert [(e.seed, e.instance) for u in units for e in u.episodes] == [(9, 0), (9, 1)]


def test_nonfinite_metric_does_not_invalidate_other_scores():
    assert policy_eval._metrics({"value": np.array([0.6, np.nan])}) == {"value": [0.6, None]}


def test_droid_request_layout_and_native_gripper(monkeypatch):
    # Use already-sized frames so the adapter exercises real side-view interpolation.
    images = {
        name: np.full((360, 640, 3), level, np.uint8)
        for level, name in enumerate(("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera"), 1)
    }
    monkeypatch.setitem(
        sys.modules, "robolab.core.utils.image_utils", SimpleNamespace(resize_with_pad=lambda x, h, w: x)
    )
    sent = []

    def request(payload):
        sent.append(payload)
        return {"action": np.full((32, 8), 0.6)}

    endpoint = DroidEndpoint.__new__(DroidEndpoint)
    endpoint.client = SimpleNamespace(request=request)
    (chunk,) = endpoint.act([{**images, "state": np.arange(8), "prompt": "pick"}])
    frame = sent[0]["observation/image"]
    assert frame.shape == (540, 640, 3)
    assert np.all(frame[:360] == 1) and np.all(frame[360:, :320] == 2) and np.all(frame[360:, 320:] == 3)
    np.testing.assert_array_equal(sent[0]["observation/joint_position"], np.arange(7))
    np.testing.assert_array_equal(sent[0]["observation/gripper_position"], [7])
    assert sent[0]["prompt"] == "pick"
    assert chunk.shape == (32, 8) and np.all(chunk[:, -1] == 1)
