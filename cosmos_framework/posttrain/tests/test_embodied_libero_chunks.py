# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Controlled simulator outcomes exercise the production chunk boundary logic."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.sims.libero.protocol import execution_protocol
from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class CountingPhysics:
    def __init__(self, start: int, success_at: int | None):
        self.steps, self.success_at = start, success_at
        self.commands = []

    def step(self, action):
        self.commands.append(action.copy())
        self.steps += 1
        success = self.check_success()
        return {"steps": self.steps}, float(success), success, {}

    def check_success(self):
        return self.steps == self.success_at


@pytest.mark.parametrize(
    "start,success_at,executed,success,truncated",
    [
        (0, 3, 3, True, False),
        (719, None, 1, False, True),
        (719, 720, 1, True, False),
    ],
)
def test_chunk_stops_at_success_or_budget_and_keeps_pre_reset_observation(
    monkeypatch, start, success_at, executed, success, truncated
):
    # Unit-only state injection controls the boundary; GPU tests establish full sessions through the registry.
    session = object.__new__(LiberoChunkSession)
    physics = CountingPhysics(start, success_at)
    session.n_lanes, session.lane_offset, session.session_id = 1, 0, "unit"
    session.suite, session.execution_horizon, session.max_steps = "libero_10", 8, 720
    session._protocol = execution_protocol(session.suite)
    session._success_once = [False]
    session._closed, session._auto_reset = False, True
    session._envs, session._steps, session._decisions = [physics], [start], [0]
    session._vector = None
    session._previous_reward, session._returns = [0.0], [0.0]
    session._tasks, session._init_states, session._episodes = [0], [0], [0]
    session._observations = [{}]
    session._suite = SimpleNamespace(get_task=lambda _: SimpleNamespace(language="test task"))
    monkeypatch.setattr(LiberoChunkSession, "_observation", staticmethod(lambda raw, _: dict(raw)))

    def reset(lane):
        session._steps[lane] = 0
        session._observations[lane] = {"steps": 0}

    monkeypatch.setattr(session, "_reset", reset)
    action = np.zeros((8, 7), dtype=np.float32)
    action[:, -1] = [0.0, 0.49, 0.5, 0.51, 1.0, -1.0, 2.0, 0.25]
    result = session.step_chunk([action])[0]
    assert len(physics.commands) == len(result.rewards) == result.action.shape[0] == executed
    assert result.terminated is success and result.truncated is truncated
    assert result.next_observation["steps"] == start + executed
    assert result.observation["steps"] == 0
    assert result.episode["steps"] == start + executed
    np.testing.assert_array_equal(np.stack(physics.commands)[:, -1], np.array([1, 1, 0, -1, -1, 1, -1, 1])[:executed])


def test_runtime_preflight_rejects_dependency_and_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import importlib.metadata
    import importlib.util
    import json
    from functools import partial

    from cosmos_framework.posttrain.embodied.registry import build_sim
    from cosmos_framework.posttrain.embodied.sims.libero import runtime
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import runtime_packages

    source = tmp_path / "libero" / "__init__.py"
    source.parent.mkdir()
    source.write_bytes(b"# packaged simulator source\n")
    (tmp_path / "rlinf-source.json").write_text(
        json.dumps({"libero/__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    versions, locations = runtime_packages(), [str(source.parent)]
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "runtime.py"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: tmp_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: SimpleNamespace(submodule_search_locations=locations))
    construct = partial(build_sim, "libero", embodiment="libero_sim", n_lanes=1)
    construct().close()
    with monkeypatch.context() as patch:
        patch.setitem(versions, "mujoco", "0")
        with pytest.raises(RuntimeError, match="runtime dependencies differ"):
            construct()
    locations[:] = [str(tmp_path / "shadow")]
    with pytest.raises(RuntimeError, match="import source differs"):
        construct()
    locations[:] = [str(source.parent)]
    source.write_bytes(b"# tampered simulator source\n")
    with pytest.raises(RuntimeError, match="source differs"):
        construct()
