# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Standard episode sessions over RoboLab's native frozen-lane execution."""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

from cosmos_framework.posttrain.embodied.embodiment import to_sim_obs
from cosmos_framework.posttrain.embodied.sim import SimProtocolError, SimTransient, StepOut
from cosmos_framework.posttrain.embodied.sims.robolab.interface import (
    extract_observation,
    get_embodiment_map,
    to_numpy,
    validate_control_freq,
)
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome, WorkUnit
from cosmos_framework.posttrain.schema.trajectory import ObsBundle


class RoboLabSession:
    """Execute one task batch, caching observations and preserving each lane's terminal result.

    ``unit`` selects episodes; ``env`` and ``cfg`` are the native environment and its
    resolved configuration. ``generation`` identifies this scene/session incarnation.
    """

    def __init__(
        self,
        unit: WorkUnit,
        env: Any,
        cfg: Any,
        *,
        robot: str,
        control_freq: int,
        generation: int,
        dense_caption: bool = False,
        control_mode: str = "joint_pos",
    ) -> None:
        self._unit, self._env, self._cfg = unit, env, cfg
        self._robot = robot
        self._dense_caption = dense_caption
        self._caption_session = uuid.uuid4().hex
        validate_control_freq(robot, control_freq)
        self._embodiment_map = get_embodiment_map(robot, control_mode)
        self.generation = generation
        self.n_lanes = len(unit.episodes)
        self._started = False
        self._closed = False
        self._steps = 0
        self._obs: list[ObsBundle] = []
        self._outcomes: dict[int, Outcome] = {}
        self._metrics: dict[int, dict[str, Any]] = {}
        # A resumed subset retains its original physical lane and batch width. Compacting
        # it would change native scene randomization and silently evaluate another layout.
        if env.num_envs != unit.scenario["native_batch_size"] or env.max_episode_length != unit.scenario["max_steps"]:
            raise SimProtocolError("RoboLab scene width/budget differs from the resolved task")
        if not np.isclose(float(cfg.sim.dt * cfg.decimation), 1 / control_freq):
            raise SimProtocolError("RoboLab control period differs from its interaction contract")
        if env.action_manager.total_action_dim != self._embodiment_map.action_dim:
            raise SimProtocolError("RoboLab action manager differs from its interaction contract")
        if any(key.seed != unit.scenario["seed"] or not 0 <= key.instance < env.num_envs for key in unit.episodes):
            raise SimProtocolError("RoboLab episode identity must retain its batch seed and physical lane")

    def _validate_session_and_lane(self, lane: int | None = None) -> None:
        if self._closed:
            raise SimProtocolError("RoboLab session is closed")
        if lane is not None and (not self._started or not 0 <= lane < self.n_lanes):
            raise SimProtocolError(f"invalid RoboLab lane {lane}")

    def _cache_observations(
        self, raw: dict[str, Any], *, read_images: bool = True, commands: np.ndarray | None = None
    ) -> None:
        previous = self._obs
        self._obs = []
        for lane, key in enumerate(self._unit.episodes):
            observation = extract_observation(
                raw,
                key.instance,
                self._robot,
                self._cfg.instruction,
                cached_images=previous[lane].representations if not read_images else None,
            )
            if lane not in self._outcomes and not np.isfinite(observation["state"]).all():
                # Detect the first bad simulation step, not a later policy query. Keep
                # the native action/state tail that would otherwise die with the worker.
                recorder = getattr(self._env, "recorder_manager", None)
                if recorder is not None:
                    try:
                        recorder.flush_buffer(verbose=False)
                    except Exception:
                        logging.getLogger(__name__).exception("Could not flush RoboLab fault diagnostics")
                previous_state = previous[lane].representations["state"].tolist() if previous else None
                command = commands[key.instance].tolist() if commands is not None else None
                raise SimTransient(
                    f"RoboLab non-finite state: episode={key.slug()} physical_lane={key.instance} "
                    f"step={self._steps} state={observation['state'].tolist()} "
                    f"previous_state={previous_state} command={command}"
                )
            if self._dense_caption:
                observation["_dense_caption"] = {
                    "session_id": self._caption_session,
                    "episode_id": self.generation,
                    "env_id": key.instance,
                }
            self._obs.append(to_sim_obs(observation))

    def advance_turn(self) -> bool:
        self._validate_session_and_lane()
        if self._started:
            if len(self._outcomes) != self.n_lanes:
                raise SimProtocolError("cannot advance RoboLab with unfinished lanes")
            return False
        self._env.reset_eval_state()
        # RoboLab's reference runner resets twice before the first policy query.
        self._env.reset()
        raw, _ = self._env.reset()
        recorder = getattr(self._env, "recorder_manager", None)
        if recorder is not None and hasattr(recorder, "set_hdf5_file"):
            # Match the native runner: each physical lane owns its HDF5 demo. Without
            # explicit indices, streaming flushes append every lane to the same demo.
            recorder.set_hdf5_file("data.hdf5")
            for physical_lane in range(self._env.num_envs):
                recorder.set_episode_index(physical_lane, env_ids=[physical_lane])
        self._cache_observations(raw)
        self._started = True
        return True

    def lane_key(self, lane: int) -> EpisodeKey:
        self._validate_session_and_lane(lane)
        return self._unit.episodes[lane]

    def observe(self, lanes: Sequence[int]) -> list[ObsBundle]:
        for lane in lanes:
            self._validate_session_and_lane(lane)
        return [self._obs[lane] for lane in lanes]

    def _native_step(self, commands: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
        import torch

        return self._env.step(torch.as_tensor(commands, device=self._env.device))

    def _native_metrics(self, physical_lane: int) -> dict[str, Any]:
        from robolab.core.logging.results import get_all_env_events, get_current_subtask_info

        metrics = {"subtask": get_current_subtask_info(self._env, physical_lane)}
        events = get_all_env_events(self._env)
        if events is not None:
            metrics["events"] = events[physical_lane]
        return metrics

    def step(self, actions: Mapping[int, Any], *, render: bool = True) -> list[StepOut]:
        self._validate_session_and_lane()
        if not actions:
            return []
        live = {lane for lane in range(self.n_lanes) if lane not in self._outcomes}
        if set(actions) != live:
            raise SimProtocolError("RoboLab steps all live requested lanes together")
        commands = np.zeros((self._env.num_envs, self._embodiment_map.action_dim), dtype=np.float32)
        for lane, action in actions.items():
            self._validate_session_and_lane(lane)
            value = np.asarray(action, dtype=np.float32)
            if value.shape != (self._embodiment_map.action_dim,) or not np.isfinite(value).all():
                raise SimProtocolError("RoboLab requires one finite action per live lane")
            commands[self.lane_key(lane).instance] = value
        # Native rendering remains part of env.step. The existing render flag gates
        # CPU image readback only; the driver requests fresh images before each query
        # and on every recorded video frame, including for shorter model chunks.
        raw, rewards, _, _, _ = self._native_step(commands)
        self._steps += 1
        self._cache_observations(raw, read_images=render, commands=commands)
        native = {int(row["env_id"]): row for row in self._env.get_env_results()}
        rewards = to_numpy(rewards)
        results = []
        for lane in actions:
            physical = self.lane_key(lane).instance
            row = native[physical]
            # Raw termination flags can describe a <=2-step physics-artifact reset.
            # Only RoboLab's frozen-lane result is a native success/failure judgment.
            success = row["success"]
            ended = success is not None or self._steps >= self._env.max_episode_length
            if ended:
                self._outcomes[lane] = Outcome.SUCCESS if success else Outcome.FAILURE
                self._metrics[lane] = {
                    **self._native_metrics(physical),
                    "native": dict(row),
                    "success": bool(success),
                    "steps": self._steps,
                    "seed": self.lane_key(lane).seed,
                    "budget_exhausted": success is None,
                }
            results.append(
                StepOut(
                    lane,
                    self._obs[lane],
                    float(rewards[physical]),
                    bool(ended and success),
                    bool(ended and not success),
                )
            )
        return results

    def done(self, lane: int) -> bool:
        self._validate_session_and_lane(lane)
        return lane in self._outcomes

    def outcome(self, lane: int) -> Outcome:
        self._validate_session_and_lane(lane)
        if lane not in self._outcomes:
            raise SimProtocolError("RoboLab episode is not terminal")
        return self._outcomes[lane]

    def metrics(self, lane: int) -> Mapping[str, Any]:
        self.outcome(lane)
        return copy.deepcopy(self._metrics[lane])

    def dropped_obs_keys(self) -> tuple[str, ...]:
        return ()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._env.close()
            self._obs = []
