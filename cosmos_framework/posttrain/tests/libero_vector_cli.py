# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Isolated real-LIBERO physics/camera comparison; never a throughput benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import pickle
import platform
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _runtime_metadata():
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import runtime_packages

    locked = runtime_packages()
    expected = {name: locked[name] for name in ("mujoco", "robosuite", "numpy", "rlinf-libero")}
    actual = {name: importlib.metadata.version(name) for name in expected}
    assert actual == expected, {"expected": expected, "actual": actual}
    source = Path(__file__).parents[1] / "embodied/sims/libero/rlinf-source.json"
    inventory = json.loads(source.read_text())
    distribution = importlib.metadata.distribution("rlinf-libero")
    for name, expected_sha in inventory.items():
        path = Path(distribution.locate_file(name))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha, str(path)
    binaries = {}
    for name in ("numpy.core._multiarray_umath", "mujoco._functions", "scipy.linalg._fblas"):
        path = Path(importlib.import_module(name).__file__)
        binaries[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "packages": actual,
        "protected_source_files": len(inventory),
        "executable": sys.executable,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "node": platform.node(),
        "binaries": binaries,
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def _record(target, key, observations, rewards, dones, states, rngs):
    from robosuite.utils.transform_utils import quat2axisangle

    for lane, observation in enumerate(observations):
        row = {
            "physics": np.asarray(states[lane]).copy(),
            "reward": float(rewards[lane]),
            "done": bool(dones[lane]),
            "numpy_rng_sha256": hashlib.sha256(pickle.dumps(rngs[lane], protocol=5)).hexdigest(),
            "model_state": np.concatenate(
                (
                    np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
                    np.asarray(quat2axisangle(observation["robot0_eef_quat"]), dtype=np.float32),
                    np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
                )
            ),
            "proprio": {
                name: np.asarray(observation[name]).copy()
                for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            },
            "images": {
                name: {
                    "sha256": hashlib.sha256(observation[name].tobytes()).hexdigest(),
                    "shape": observation[name].shape,
                    "dtype": str(observation[name].dtype),
                }
                for name in ("agentview_image", "robot0_eye_in_hand_image")
                if name in observation
            },
        }
        target[f"{key}.lane{lane}"] = row


def _case(episode, lane):
    return ((episode * 8 + lane) % 10, (episode * 7 + lane) % 50)


def _reference_factory(parameters):
    from rlinf.envs.sim.libero import venv

    from cosmos_framework.posttrain.tests.helpers.libero_reference import ObservedReferenceEnv

    values = dict(parameters)
    seed = values.pop("seed")
    # This factory runs only in an isolated worker. Its reconfigure path retains
    # the read-only property; no native physics, reset or step method is overridden.
    venv.OffScreenRenderEnv = ObservedReferenceEnv
    env = ObservedReferenceEnv(**values)
    env.seed(seed)
    return env


def _reference(args, actions):
    sys.path.insert(0, str(args.reference_source))
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    from rlinf.envs.sim.libero.venv import ReconfigureSubprocEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    def parameters(episode):
        rows = []
        states = []
        for lane in range(args.lanes):
            task_id, state_id = _case(episode, lane)
            task = suite.get_task(task_id)
            rows.append(
                {
                    "bddl_file_name": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                    "camera_heights": 256,
                    "camera_widths": 256,
                    "seed": args.seed * args.lanes + lane,
                }
            )
            states.append(suite.get_task_init_states(task_id)[state_id])
        return rows, states

    initial, _ = parameters(0)
    vector = ReconfigureSubprocEnv([partial(_reference_factory, row) for row in initial])
    data = {}
    init_states = {}
    try:
        for episode in range(args.episodes):
            params, states = parameters(episode)
            init_states[str(episode)] = [_array_identity(np.asarray(state)) for state in states]
            if episode:
                vector.reconfigure_env_fns(params)
            vector.reset()
            vector.set_init_state(states)
            zeros = np.zeros((args.lanes, 7), dtype=np.float32)
            zeros[:, -1] = -1
            for _ in range(15):
                raw, rewards, dones, _ = vector.step(zeros)
            _record(
                data,
                f"reset{episode}",
                raw,
                [0.0] * args.lanes,
                [False] * args.lanes,
                vector.get_sim_state(),
                vector.get_env_attr("native_numpy_rng"),
            )
            for decision in range(48):
                disabled = False
                try:
                    vector.set_camera_rendering(False)
                    disabled = True
                    for control in range(5):
                        if control == 4:
                            vector.set_camera_rendering(True)
                            disabled = False
                        raw, rewards, dones, _ = vector.step(actions[episode, decision, :, control])
                        _record(
                            data,
                            f"episode{episode}.control{decision * 5 + control}",
                            raw,
                            rewards,
                            dones,
                            vector.get_sim_state(),
                            vector.get_env_attr("native_numpy_rng"),
                        )
                finally:
                    if disabled:
                        vector.set_camera_rendering(True)
    finally:
        vector.close()
    return data, {"init_states": init_states}


def _local(args, actions):
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import PI05_SPATIAL_EXECUTION_PROFILE, LiberoCase
    from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoChunkSession, LiberoSimulator
    from cosmos_framework.posttrain.embodied.sims.libero.vector import LiberoVectorEnv
    from cosmos_framework.posttrain.tests.helpers.libero_capture import RecordingNativeLane

    class ScheduledSession(LiberoChunkSession):
        def __init__(self, *values, **kwargs):
            self.draw = 0
            super().__init__(*values, **kwargs)

        def _open_vector(self):
            factory = partial(
                RecordingNativeLane, reset_steps=self._protocol.reset_steps, reset_gripper=self._protocol.reset_gripper
            )
            return LiberoVectorEnv([factory] * self.n_lanes)

        def _choose_case(self):
            if self._cases is not None and len(self._cases) == 1:
                return super()._choose_case()
            episode, lane = divmod(self.draw, self.n_lanes)
            self.draw += 1
            task, state = _case(episode, lane)
            return LiberoCase(task, state)

    simulator = LiberoSimulator(
        embodiment="libero_sim",
        n_lanes=args.lanes,
        seed=args.seed,
        execution_horizon=5,
        max_steps=240,
        execution_profile=PI05_SPATIAL_EXECUTION_PROFILE,
    )
    simulator.session_class = ScheduledSession
    data = {}
    init_states = {}
    try:
        session = simulator.open_window(0)
        vector = session._vector
        original_step = vector.step
        control_index = 0

        def step(batch):
            nonlocal control_index
            result = original_step(batch)
            raw, rewards, dones, _ = result
            episode, control = divmod(control_index, 240)
            snapshots = vector.state_snapshots()
            _record(
                data,
                f"episode{episode}.control{control}",
                raw,
                rewards,
                dones,
                [state[0] for state in snapshots],
                [state[1] for state in snapshots],
            )
            control_index += 1
            return result

        for episode in range(args.episodes):
            # Native reset observations are read in the worker, without forcing another camera update.
            raw = vector.get_env_attr("last_reset_observation")
            init_states[str(episode)] = [
                _array_identity(request.state) for request in vector.get_env_attr("reset_request")
            ]
            snapshots = vector.state_snapshots()
            _record(
                data,
                f"reset{episode}",
                raw,
                [0.0] * args.lanes,
                [False] * args.lanes,
                [state[0] for state in snapshots],
                [state[1] for state in snapshots],
            )
            with patch.object(vector, "step", step):
                for decision in range(48):
                    rows = session.step_chunk(list(actions[episode, decision]))
                    assert all(len(row.rewards) == 5 for row in rows)
            assert all(row.episode_complete and row.episode["steps"] == 240 for row in rows)
        before = simulator.online_state_digest()
        evaluation = simulator.open_evaluation(asdict(LiberoCase(0, 0)), args.seed)
        try:
            evaluation.step_chunk([np.zeros((5, 7), dtype=np.float32)])
        finally:
            evaluation.close()
        assert simulator.online_state_digest() == before
        metadata = {
            "training_state_unchanged": True,
            "completed_episodes": args.episodes * args.lanes,
            "control_records": control_index * args.lanes,
            "init_states": init_states,
        }
    finally:
        simulator.close()
    return data, metadata


def _array_identity(array):
    return {"dtype": str(array.dtype), "shape": array.shape, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _compare(local, reference):
    assert local["input"] == reference["input"], "different experiment inputs"
    assert local["metadata"]["init_states"] == reference["metadata"]["init_states"], "different init-state inputs"
    assert local["records"].keys() == reference["records"].keys(), "different record inventories"
    differences = []
    for key, expected in reference["records"].items():
        actual = local["records"][key]
        for field in ("reward", "done", "images", "numpy_rng_sha256"):
            if actual[field] != expected[field]:
                differences.append(f"{key}.{field}")
        if not np.allclose(actual["physics"], expected["physics"], atol=1e-12, rtol=0):
            differences.append(f"{key}.physics")
        if not np.array_equal(actual["model_state"], expected["model_state"]):
            differences.append(f"{key}.model_state")
        for field, value in expected["proprio"].items():
            if not np.allclose(actual["proprio"][field], value, atol=1e-12, rtol=0):
                differences.append(f"{key}.proprio.{field}")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("local", "reference", "compare"), required=True)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "compare":
        local, reference = (pickle.loads(path.read_bytes()) for path in (args.local, args.reference))
        differences = _compare(local, reference)
        first = next(iter(reference["records"]))
        changed = reference["records"][first]["physics"].copy()
        reference["records"][first]["physics"].flat[0] += 1e-8
        negative_rejected = bool(_compare(local, reference))
        reference["records"][first]["physics"] = changed
        result = {
            "passed": not differences and negative_rejected,
            "records": len(local["records"]),
            "differences": differences,
            "negative_control_rejected": negative_rejected,
            "fp64_atol": 1e-12,
            "fp64_rtol": 0,
            "physics_max_abs": max(
                float(np.abs(row["physics"] - reference["records"][key]["physics"]).max())
                for key, row in local["records"].items()
            ),
            "exact_checks": ["model_state_float32", "images", "reward", "done", "native_numpy_rng", "init_state_bytes"],
            "metadata": local["metadata"],
            "reference_metadata": reference["metadata"],
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("differences", "metadata", "reference_metadata")
                }
                | {"difference_count": len(differences), "first_differences": differences[:10]}
            )
        )
        if not result["passed"]:
            raise SystemExit(1)
        return
    generator = np.random.default_rng(args.seed + 1337)
    actions = generator.uniform(-0.03, 0.03, (args.episodes, 48, args.lanes, 5, 7)).astype(np.float32)
    actions[..., -1] = -1
    runtime = _runtime_metadata()
    records, metadata = (_local if args.phase == "local" else _reference)(args, actions)
    metadata["runtime"] = runtime
    result = {
        "input": {
            "lanes": args.lanes,
            "episodes": args.episodes,
            "seed": args.seed,
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "actions_dtype": str(actions.dtype),
            "actions_shape": actions.shape,
        },
        "records": records,
        "metadata": metadata,
    }
    args.output.write_bytes(pickle.dumps(result, protocol=5))
    print(json.dumps({"phase": args.phase, "records": len(records), "output": str(args.output), **metadata}))


if __name__ == "__main__":
    main()
