# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO-Spatial chunk simulator, loaded only in the isolated simulator process."""

from __future__ import annotations

import hashlib
import os
import pickle
import uuid
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np

from cosmos_framework.posttrain.embodied.embodiment import EmbodimentMap, register_embodiment
from cosmos_framework.posttrain.embodied.sim import PROTOCOL_VERSION, SimCapabilities, SimProtocolError
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    DEFAULT_EXECUTION_PROFILE,
    HASH_POLICY_REPLICA_ENV_SEEDS,
    LIBERO_PROTOCOLS,
    LIBERO_RUNTIME,
    LIBERO_SUITES,
    LiberoCase,
    LiberoManifest,
    execution_protocol,
    libero_seed,
    runtime_packages,
)
from cosmos_framework.posttrain.embodied.window import ChunkResult

register_embodiment(
    EmbodimentMap(
        name="libero_sim",
        cameras={"image": "image", "wrist_image": "wrist_image"},
        proprio_key="state",
        action_dim=7,
        state_dim=8,
    )
)


class LiberoSimulator:
    #: Bound after both classes exist; a test-side subclass may substitute a session type.
    session_class: type[LiberoChunkSession]

    def __init__(
        self,
        *,
        embodiment: str,
        n_lanes: int = 2,
        seed: int = 0,
        lane_offset: int = 0,
        max_steps: int = LIBERO_PROTOCOLS["libero_spatial"].max_episode_steps,
        suite: str = "libero_spatial",
        execution_horizon: int = LIBERO_PROTOCOLS["libero_spatial"].execution_horizon,
        train_manifest: str = "",
        emit_extrinsics: bool = False,
        execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    ) -> None:
        if embodiment != "libero_sim" or emit_extrinsics:
            raise ValueError("LIBERO requires libero_sim and emit_extrinsics=False")
        if suite not in LIBERO_SUITES or n_lanes < 1:
            raise ValueError("LIBERO requires a supported suite and positive n_lanes")
        protocol = execution_protocol(suite, execution_profile)
        if (execution_horizon, max_steps) != (protocol.execution_horizon, protocol.max_episode_steps):
            raise ValueError("LIBERO execution horizon / episode limit differ from the execution protocol")
        self.execution_profile, self._protocol = execution_profile, protocol
        self.suite, self.execution_horizon = suite, execution_horizon
        self.n_lanes, self.seed, self.lane_offset, self.max_steps = n_lanes, seed, lane_offset, max_steps
        self._train = LiberoManifest.read(train_manifest) if train_manifest else None
        if self._train is not None and (self._train.split != "train" or self._train.suite != suite):
            raise ValueError("LIBERO train_manifest must declare the requested suite and train split")
        self._session: LiberoChunkSession | None = None
        # Construction is the actual simulator binding boundary, including direct
        # entrypoint launches that do not use the standalone allocation preflight.
        self.probe()

    def capabilities(self) -> SimCapabilities:
        # The shared gateway sizes one observation buffer from these per-lane declarations.
        cameras = {"image": (256, 256), "wrist_image": (256, 256)}
        return SimCapabilities(
            PROTOCOL_VERSION,
            "libero",
            LIBERO_RUNTIME.libero,
            "libero_sim",
            self.n_lanes,
            cameras,
            8,
            extras=("decision_chunks", "fixed_case_evaluation"),
        )

    def probe(self) -> None:
        """Refuse any interpreter whose installed runtime differs from ``LIBERO_RUNTIME``."""
        import importlib.metadata
        import json

        from packaging.version import Version

        expected = runtime_packages()
        actual = {key: importlib.metadata.version(key) for key in expected}
        # Wheel metadata can retain equivalent legacy spelling (e.g. thop's post release).
        mismatches = {
            key: {"expected": version, "actual": actual[key]}
            for key, version in expected.items()
            if Version(actual[key] if "+" in version else Version(actual[key]).public) != Version(version)
        }
        if mismatches:
            raise RuntimeError(f"LIBERO runtime dependencies differ: {mismatches}")
        # Bind the imported namespace to this distribution and its frozen source bytes.
        import importlib.util
        from pathlib import Path

        distribution = importlib.metadata.distribution("rlinf-libero")
        site = Path(distribution.locate_file("")).resolve()
        spec = importlib.util.find_spec("libero")
        locations = tuple(Path(p).resolve() for p in (spec.submodule_search_locations if spec else ()))
        if locations != (site / "libero",):
            raise RuntimeError(f"RLinf LIBERO import source differs: {locations}")
        expected_source = json.loads(Path(__file__).with_name("rlinf-source.json").read_text())
        for name, digest in expected_source.items():
            path = site / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"RLinf LIBERO source differs: {name}")

    def open_window(self, version: int) -> LiberoChunkSession:
        if self._session is None:
            seed = self.seed + version * 8 + self.lane_offset
            if self._protocol.seed_derivation == HASH_POLICY_REPLICA_ENV_SEEDS:
                seed = self.seed + self.lane_offset // self.n_lanes
            elif self.suite == "libero_10":
                seed = libero_seed(self.suite, "training-environment", self.seed, version, self.lane_offset) % 2**32
            self._session = self.session_class(
                self.n_lanes,
                seed,
                self.lane_offset,
                cases=self._train.cases if self._train else None,
                suite=self.suite,
                execution_horizon=self.execution_horizon,
                max_steps=self.max_steps,
                execution_profile=self.execution_profile,
            )
        return self._session

    def open_evaluation(self, case: dict[str, int | str], seed: int) -> LiberoChunkSession:
        selected = LiberoCase(**case)
        if selected.suite != self.suite:
            raise ValueError("evaluation case suite differs from the simulator")
        if self._protocol.seed_derivation == HASH_POLICY_REPLICA_ENV_SEEDS:
            seed += self.lane_offset // self.n_lanes
        return self.session_class(
            1,
            seed,
            0,
            cases=(selected,),
            auto_reset=False,
            suite=self.suite,
            execution_horizon=self.execution_horizon,
            max_steps=self.max_steps,
            execution_profile=self.execution_profile,
        )

    def online_state_digest(self) -> str:
        """Within-process guard for online physics, observations, progress and the session RNG."""
        return self._session.state_digest() if self._session is not None else "unopened"

    def wants_restart(self) -> bool:
        return False

    def gpu_uuid(self) -> str | None:
        return os.environ.get("CUDA_VISIBLE_DEVICES")

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


class LiberoChunkSession:
    def __init__(
        self,
        n_lanes: int,
        seed: int,
        lane_offset: int,
        *,
        cases: tuple[LiberoCase, ...] | None = None,
        auto_reset: bool = True,
        suite: str = "libero_spatial",
        execution_horizon: int = LIBERO_PROTOCOLS["libero_spatial"].execution_horizon,
        max_steps: int = LIBERO_PROTOCOLS["libero_spatial"].max_episode_steps,
        execution_profile: str = DEFAULT_EXECUTION_PROFILE,
    ) -> None:
        from libero.libero import benchmark

        self.n_lanes, self.lane_offset = n_lanes, lane_offset
        self.session_id = uuid.uuid4().hex
        self.suite, self.execution_horizon, self.max_steps = suite, execution_horizon, max_steps
        self._protocol = execution_protocol(suite, execution_profile)
        if (execution_horizon, max_steps) != (self._protocol.execution_horizon, self._protocol.max_episode_steps):
            raise ValueError("LIBERO session horizon / episode limit differ from the execution protocol")
        self._seed = seed
        self._suite = benchmark.get_benchmark_dict()[suite]()
        self._rng = np.random.default_rng(seed)
        self._cases = cases
        self._auto_reset = auto_reset
        self._init_states = [-1] * n_lanes
        self._returns = [0.0] * n_lanes
        if self._suite.n_tasks != 10 or any(len(self._suite.get_task_init_states(i)) != 50 for i in range(10)):
            raise ValueError(f"{suite} assets must expose ten tasks with fifty initial states each")
        self._envs: list[Any] = [None] * n_lanes
        self._vector = None
        self._tasks = [-1] * n_lanes
        self._episodes = [-1] * n_lanes
        self._steps = [0] * n_lanes
        self._decisions = [0] * n_lanes
        self._previous_reward = [0.0] * n_lanes
        self._success_once = [False] * n_lanes
        self._observations: list[dict] = [{} for _ in range(n_lanes)]
        self._closed = False
        try:
            if self._protocol.parallel_lanes:
                self._vector = self._open_vector()
                self._reset_lanes(list(range(n_lanes)))
            else:
                for lane in range(n_lanes):
                    self._reset(lane)
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise SimProtocolError("LIBERO session is closed")

    def state_digest(self) -> str:
        self._require_open()
        # This is an in-process before/after guard, not a portable checkpoint representation.
        state = (
            self._tasks,
            self._init_states,
            self._episodes,
            self._steps,
            self._decisions,
            self._previous_reward,
            self._success_once,
            self._returns,
            self._observations,
            self._rng.bit_generator.state,
            self._vector.state_snapshots() if self._vector is not None else [env.get_sim_state() for env in self._envs],
        )
        return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()

    def _choose_case(self) -> LiberoCase:
        return (
            self._cases[int(self._rng.integers(len(self._cases)))]
            if self._cases
            else LiberoCase(int(self._rng.integers(10)), int(self._rng.integers(50)), self.suite)
        )

    def _open_vector(self):
        from cosmos_framework.posttrain.embodied.sims.libero.vector import LiberoLane, LiberoVectorEnv

        factory = partial(
            LiberoLane, reset_steps=self._protocol.reset_steps, reset_gripper=self._protocol.reset_gripper
        )
        return LiberoVectorEnv([factory] * self.n_lanes)

    def _reset_lanes(self, lanes: list[int]) -> None:
        from libero.libero.utils import get_libero_path

        from cosmos_framework.posttrain.embodied.sims.libero.vector import LaneReset

        requests = {}
        for lane in lanes:
            case = self._choose_case()
            self._tasks[lane], self._init_states[lane] = case.task_id, case.init_state_id
            task = self._suite.get_task(case.task_id)
            requests[lane] = LaneReset(
                os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                np.asarray(self._suite.get_task_init_states(case.task_id)[case.init_state_id]),
                self._seed * self.n_lanes + lane,
            )
        observations = self._vector.reset_lanes(requests)
        for lane, raw in zip(lanes, observations, strict=True):
            self._reset_statistics(lane, raw)

    def _reset(self, lane: int) -> None:
        case = self._choose_case()
        task_index = case.task_id
        self._init_states[lane] = case.init_state_id
        task = self._suite.get_task(task_index)
        if self._tasks[lane] != task_index:
            if self._envs[lane] is not None:
                self._close_env(lane)
            from libero.libero.utils import get_libero_path

            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self._envs[lane] = self._open_env(lane, bddl, task.language)
            self._tasks[lane] = task_index
        env = self._envs[lane]
        if self._protocol.seed_derivation == HASH_POLICY_REPLICA_ENV_SEEDS:
            env.seed(self._seed * self.n_lanes + lane)
        else:
            env.seed(int(self._rng.integers(2**31)))
        env.reset()
        states = self._suite.get_task_init_states(task_index)
        env.set_init_state(states[case.init_state_id])
        zero = np.zeros(7, dtype=np.float32)
        protocol = self._protocol
        zero[-1] = protocol.reset_gripper
        for _ in range(protocol.reset_steps):
            raw, _, _, _ = env.step(zero)
        self._reset_statistics(lane, raw)

    def _reset_statistics(self, lane: int, raw: dict) -> None:
        self._episodes[lane] += 1
        self._steps[lane] = 0
        self._decisions[lane] = 0
        self._previous_reward[lane] = 0.0
        self._success_once[lane] = False
        self._returns[lane] = 0.0
        self._observations[lane] = self._views(raw, self._suite.get_task(self._tasks[lane]).language, lane)

    def _open_env(self, lane: int, bddl: str, language: str) -> Any:
        """Create the lane's environment; a subclass may borrow another owner's simulator."""
        from libero.libero.envs import OffScreenRenderEnv

        return OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=0,
            ignore_done=True,
            horizon=100000,
        )

    def _close_env(self, lane: int) -> None:
        env, self._envs[lane] = self._envs[lane], None
        env.close()

    def _views(self, raw: dict, prompt: str, lane: int) -> dict:
        """Per-lane observation hook; ``lane`` lets a subclass consult the owning environment."""
        return self._observation(raw, prompt)

    @staticmethod
    def _observation(raw: dict, prompt: str) -> dict:
        from robosuite.utils.transform_utils import quat2axisangle

        position = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        rotation = np.asarray(quat2axisangle(raw["robot0_eef_quat"]), dtype=np.float32)
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        return {
            "image": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
            "state": np.concatenate((position, rotation, gripper)),
            **{
                key: np.asarray([value], dtype=np.float32)
                for key, value in zip(
                    ("x", "y", "z", "roll", "pitch", "yaw"), np.concatenate((position, rotation)), strict=True
                )
            },
            "gripper": gripper,
            "prompt": prompt,
        }

    def observe(self) -> list[dict]:
        self._require_open()
        return self._observations

    def sources(self, window: int) -> list[dict]:
        self._require_open()
        result = []
        for lane in range(self.n_lanes):
            episode = f"{self.session_id}:{lane + self.lane_offset}:{self._episodes[lane]}"
            step = self._decisions[lane]
            result.append(
                {
                    "sample_id": f"{episode}:{step}",
                    "episode_id": episode,
                    "fragment_id": f"{self.session_id}:w{window}:l{lane}",
                    "step_idx": step,
                    "session": self.session_id,
                    "task_id": self._tasks[lane],
                    "init_state_id": self._init_states[lane],
                    "suite": self.suite,
                    "episode_index": self._episodes[lane],
                }
            )
        return result

    def step_chunk(self, actions: list[np.ndarray]) -> list[ChunkResult]:
        self._require_open()
        if len(actions) != self.n_lanes:
            raise ValueError("LIBERO action count differs from the lane count")
        for lane, action in enumerate(actions):
            if action.shape != (self.execution_horizon, 7) or not np.isfinite(action).all():
                raise ValueError(f"LIBERO actions[{lane}] requires finite [{self.execution_horizon},7]")
        if self._vector is not None:
            try:
                return self._parallel_chunk(actions)
            except BaseException:
                self.close()
                raise
        results = []
        for lane, action in enumerate(actions):
            rewards, terminated = [], False
            control_terminated, control_truncated = [], []
            executed = action.copy()
            if self._protocol.action_transform == "binary-inverted-gripper":
                executed[:, -1] = -np.sign(2 * executed[:, -1] - 1)
            for control in executed:
                raw, raw_reward, success, _ = self._envs[lane].step(control)
                if self.suite == "libero_10":
                    success = bool(success) or bool(self._envs[lane].check_success())
                rewards.append(self._record_control(lane, raw_reward, success))
                terminated = terminated or bool(success)
                control_terminated.append(bool(success))
                control_truncated.append(self._steps[lane] >= self.max_steps)
                if self._protocol.stop_on_success and (terminated or self._steps[lane] >= self.max_steps):
                    break
            results.append(
                self._finish_chunk(
                    lane, executed[: len(rewards)], raw, rewards, control_terminated, control_truncated, reset=True
                )
            )
        return results

    def _record_control(self, lane: int, raw_reward: float, success: bool) -> float:
        current = float(success)
        reward = current - self._previous_reward[lane] if self._protocol.reward == "relative" else float(raw_reward)
        self._previous_reward[lane] = current
        if self._protocol.fixed_episode:
            if not self._success_once[lane]:
                self._returns[lane] += reward
            self._success_once[lane] = self._success_once[lane] or bool(success)
        self._steps[lane] += 1
        return reward

    def _finish_chunk(self, lane, executed, raw, rewards, terminations, truncations, *, reset: bool) -> ChunkResult:
        self._decisions[lane] += 1
        terminated = any(terminations)
        final = self._views(raw, self._suite.get_task(self._tasks[lane]).language, lane)
        truncated = self._steps[lane] >= self.max_steps and (self._protocol.fixed_episode or not terminated)
        self._observations[lane] = final
        if not self._protocol.fixed_episode:
            self._returns[lane] += sum(rewards)
        self._success_once[lane] = self._success_once[lane] or terminated
        complete = truncated if self._protocol.fixed_episode else terminated or truncated
        episode = None
        if complete:
            episode = {
                "task_id": self._tasks[lane],
                "suite": self.suite,
                "init_state_id": self._init_states[lane],
                "success": self._success_once[lane] if self._protocol.fixed_episode else terminated,
                "steps": self._steps[lane],
                "return": self._returns[lane],
                "episode_id": f"{self.session_id}:{lane + self.lane_offset}:{self._episodes[lane]}",
            }
            if self._auto_reset and reset:
                self._reset(lane)
        return ChunkResult(
            executed,
            self._observations[lane],
            final,
            tuple(rewards),
            terminated,
            truncated,
            episode,
            control_terminated=tuple(terminations) if self._protocol.fixed_episode else None,
            control_truncated=tuple(truncations) if self._protocol.fixed_episode else None,
            episode_complete=complete if self._protocol.fixed_episode else None,
        )

    def _parallel_chunk(self, actions: list[np.ndarray]) -> list[ChunkResult]:
        executed = np.stack(actions)
        rewards = [[] for _ in actions]
        terminations = [[] for _ in actions]
        truncations = [[] for _ in actions]
        rendering_disabled = False
        try:
            if self._protocol.last_control_render and self.execution_horizon > 1:
                self._vector.cameras(False)
                rendering_disabled = True
            for control in range(self.execution_horizon):
                if self._protocol.last_control_render and control == self.execution_horizon - 1:
                    self._vector.cameras(True)
                    rendering_disabled = False
                raw, native_rewards, native_done, _ = self._vector.step(executed[:, control])
                for lane in range(self.n_lanes):
                    rewards[lane].append(self._record_control(lane, native_rewards[lane], native_done[lane]))
                    terminations[lane].append(bool(native_done[lane]))
                    truncations[lane].append(self._steps[lane] >= self.max_steps)
        finally:
            if rendering_disabled:
                self._vector.cameras(True)
        results = [
            self._finish_chunk(
                lane, executed[lane], raw[lane], rewards[lane], terminations[lane], truncations[lane], reset=False
            )
            for lane in range(self.n_lanes)
        ]
        completed = [lane for lane, result in enumerate(results) if result.completed]
        if self._auto_reset and completed:
            self._reset_lanes(completed)
            for lane in completed:
                results[lane] = replace(results[lane], observation=self._observations[lane])
        return results

    def close(self) -> None:
        self._closed = True
        try:
            if self._vector is not None:
                self._vector.close()
            for lane, env in enumerate(self._envs):
                if env is not None:
                    self._close_env(lane)
        finally:
            self._envs = []
            self._observations = []
            self._vector = None


LiberoSimulator.session_class = LiberoChunkSession
