# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-side ownership of PPO sampling records during an isolated environment window."""

from __future__ import annotations

import copy
import os
from dataclasses import replace
from typing import Protocol

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.embodied.window import WindowEnvironment
from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody
from cosmos_framework.posttrain.schema import (
    EnvironmentFeedback,
    ObsBundle,
    Trajectory,
)


def build_window_gateway(
    environment: WindowEnvironment, *, replica: int, n_lanes: int, seed: int, evaluation: bool = False
) -> SimGatewayBody:
    """Construct the framework gateway from plugin-owned data at the actual resource boundary."""
    spec = environment.gateway_spec(replica=replica, n_lanes=n_lanes, seed=seed)
    return SimGatewayBody(
        sim=spec.name,
        sink="null",
        embodiment=spec.embodiment,
        sim_python=spec.python,
        sim_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        sim_options=spec.options,
        sim_ipc_dir=os.environ["B1K_SIM_IPC_DIR"],
        **(
            {"sim_socket_path": os.path.join(os.environ["B1K_SIM_IPC_DIR"], f"evaluation-{os.getpid()}.sock")}
            if evaluation
            else {}
        ),
    )


class DecisionSamplingPolicy(Protocol):
    """Model-owned sampled records and scoring statistics; the collector never interprets a trace."""

    def sample_decisions(
        self, observations: list[dict], *, sources: list[dict], seed: int, weight_version: int
    ) -> list[Trajectory]: ...
    def bootstrap_values(self, observations: list[dict], *, seed: int, weight_version: int) -> tuple[float, ...]: ...
    def evaluation_actions(self, observations: list[dict], *, seed: int, weight_version: int) -> list: ...
    def ppo_identity(self) -> dict: ...


class PPOWindowCollector:
    """Own sampling records and cursors; peers own model and simulator resources."""

    def __init__(
        self,
        recipe: JointPPOConfig,
        replica: int,
        n_lanes: int,
        *,
        environment: WindowEnvironment,
    ) -> None:
        self.recipe = recipe
        self.rollout: DecisionSamplingPolicy | None = None
        self.replica, self.n_lanes = replica, n_lanes
        self.environment = environment
        self._pending: list[Trajectory] = []
        self._records: list[Trajectory] = []
        self._version: int | None = None
        self._next_window = 0
        self._episodes: list[dict] = []
        self._reward = 0.0
        self._evaluation_version: int | None = None
        self._evaluation_before: dict | None = None
        self._closed = False

    def bind_rollout(self, rollout: DecisionSamplingPolicy) -> None:
        if self._version is not None or self._evaluation_version is not None:
            raise RuntimeError("cannot bind rollout during an active window")
        self.rollout = rollout

    def act(self, observations: list[dict], *, context: dict) -> list:
        if self._evaluation_version is not None:
            if context["policy_version"] != self._evaluation_version:
                raise ValueError("evaluation request differs from the published policy version")
            return self.rollout.evaluation_actions(
                observations,
                seed=self.environment.evaluation_seed(context),
                weight_version=self._evaluation_version,
            )
        if self._pending:
            raise RuntimeError("cannot sample before previous environment feedback")
        version = context["policy_version"]
        if version != self._version or len(observations) != self.n_lanes:
            raise ValueError("PPO sampling request differs from the active collection window")
        sources = context["sources"]
        if len(sources) != len(observations):
            raise ValueError("PPO sampling sources differ from the observation count")
        seed = self.environment.training_seed(
            seed=self.recipe.seed,
            version=version,
            replica=self.replica,
            n_lanes=self.n_lanes,
            num_envs=self.recipe.num_envs,
            decision=context["decision"],
        )
        records = self.rollout.sample_decisions(observations, sources=sources, seed=seed, weight_version=version)
        if len(records) != len(observations):
            raise ValueError("model decision inventory differs from the observation count")
        self._pending = records
        return [sample.transitions[0].action for sample in records]

    def feedback(self, observations: list[dict], *, context: dict) -> None:
        if context["policy_version"] != self._version or len(self._pending) != len(observations):
            raise ValueError("PPO feedback differs from the pending decisions")
        rows = list(zip(self._pending, observations, context["sources"], context["feedback"], strict=True))
        feedbacks = []
        for sample, observation, source, result in rows:
            if sample.id != source["sample_id"] or sample.transitions[0].source.episode_id != source["episode_id"]:
                raise ValueError("PPO feedback source does not match its sampled action")
            if "executed_action" not in observation:
                raise ValueError(f"PPO feedback {sample.id!r} is missing the executed action")
            feedback = EnvironmentFeedback(
                tuple(result["rewards"]),
                len(result["rewards"]),
                control_terminated=(
                    tuple(result["control_terminated"]) if result.get("control_terminated") is not None else None
                ),
                control_truncated=(
                    tuple(result["control_truncated"]) if result.get("control_truncated") is not None else None
                ),
                episode_complete=result.get("episode_complete"),
            )
            for name in ("terminated", "truncated"):
                if type(result[name]) is not bool:
                    raise ValueError(f"PPO feedback {sample.id!r} {name} must be a boolean")
            if self.recipe.loss_reduction == "fixed_episode_mean":
                if feedback.control_terminated is None or feedback.episode_complete is None:
                    raise ValueError(f"PPO feedback {sample.id!r} requires control endings and episode_complete")
                if (
                    any(feedback.control_terminated) != result["terminated"]
                    or any(feedback.control_truncated) != result["truncated"]
                ):
                    raise ValueError(f"PPO feedback {sample.id!r} ending differs from its control flags")
            if observation["executed_action"].shape != (feedback.executed_steps, self.environment.action_dim):
                raise ValueError(f"PPO feedback {sample.id!r} action shape differs from executed steps")
            if feedback.executed_steps > self.environment.execution_horizon:
                raise ValueError(f"PPO feedback {sample.id!r} exceeds its execution prefix")
            if feedback.executed_steps != self.environment.execution_horizon and not (
                result["terminated"] or result["truncated"]
            ):
                raise ValueError(f"PPO feedback {sample.id!r} shortened a live decision")
            feedbacks.append(feedback)
        bootstrap = [0.0] * len(rows)
        live = [
            index
            for index, row in enumerate(rows)
            if not row[3]["terminated"]
            and (self.recipe.time_limit_bootstrap == "value" if row[3]["truncated"] else context["window_end"])
        ]
        if live:
            seed = self.environment.bootstrap_seed(seed=self.recipe.seed, version=self._version, replica=self.replica)
            predicted = self.rollout.bootstrap_values(
                [observations[index] for index in live],
                seed=seed,
                weight_version=self._version,
            )
            for index, value in zip(live, predicted, strict=True):
                bootstrap[index] = value
        completed = []
        for index, ((sample, observation, _, result), feedback) in enumerate(zip(rows, feedbacks, strict=True)):
            old = sample.transitions[0]
            raw = {key: copy.deepcopy(value) for key, value in observation.items() if key != "executed_action"}
            step = replace(
                old,
                action=observation["executed_action"].copy(),
                next_obs=ObsBundle(representations={"raw": raw}),
                done=result["terminated"] or result["truncated"],
                truncated=result["truncated"],
                feedback=feedback,
                reward=None,
                collection_end=context["window_end"],
                algo_extra={STATS_KEY: replace(old.algo_extra[STATS_KEY], bootstrap_value=bootstrap[index])},
            )
            completed.append(replace(sample, transitions=[step]))
        self._records.extend(completed)
        self._reward += sum(sum(row[3]["rewards"]) for row in rows)
        self._episodes.extend(row[3]["episode"] for row in rows if row[3].get("episode") is not None)
        self._pending = []

    def begin_window(self, version: int) -> None:
        if self._closed or self.rollout is None:
            raise RuntimeError("collection requires an open collector and a bound rollout peer")
        if self._version is not None or self._evaluation_version is not None:
            raise RuntimeError("cannot begin an overlapping collection or evaluation")
        if version != self._next_window:
            raise ValueError(f"collector expects window {self._next_window}, received {version}")
        self._version = version
        self._episodes, self._reward = [], 0.0

    def validate_window(self, report: dict) -> None:
        expected = self.n_lanes * self.recipe.window_decisions
        if self._version is None or report["policy_version"] != self._version:
            raise RuntimeError("PPO window report differs from the active publication")
        if self._pending or len(self._records) != expected or report["decisions"] != expected:
            raise RuntimeError("PPO window did not produce its exact decision inventory")

    def finish_window(self, report: dict) -> list[Trajectory]:
        try:
            self.validate_window(report)
            result = self._records
            self._next_window += 1
            return result
        finally:
            self.abort_window()

    def abort_window(self) -> None:
        """Discard an uncommitted window without advancing its durable cursor."""
        self._records, self._pending, self._version = [], [], None

    def window_statistics(self) -> dict:
        return {
            "episodes": self._episodes,
            "reward": self._reward,
            "replica": self.replica,
        }

    def begin_evaluation(self, version: int) -> None:
        if self._closed or self.rollout is None:
            raise RuntimeError("evaluation requires an open collector and a bound rollout peer")
        if self._version is not None or self._evaluation_version is not None:
            raise RuntimeError("cannot evaluate during an active collection or evaluation")
        self._evaluation_before = self.state_dict()
        self._evaluation_version = version

    def evaluation_seed(self) -> int:
        return self.recipe.validation.seed

    def finish_evaluation(self, result: dict, seed: int) -> dict:
        try:
            if (
                self._evaluation_version is None
                or result["training_session_unchanged"] is not True
                or result["policy_version"] != self._evaluation_version
            ):
                raise RuntimeError("evaluation changed online state or used another publication")
            if self.state_dict() != self._evaluation_before:
                raise RuntimeError("evaluation changed training collection progress")
            return {**result, "evaluation_seed": seed}
        finally:
            self.abort_evaluation()

    def abort_evaluation(self) -> None:
        self._evaluation_version, self._evaluation_before = None, None

    def state_dict(self) -> dict:
        if self._version is not None:
            raise RuntimeError("cannot checkpoint an active collection window")
        return {
            "next_window": self._next_window,
            "replica": self.replica,
            "n_lanes": self.n_lanes,
            "reset_on_resume": True,
            "recipe": self.recipe.identity(),
            "environment": self.environment.identity(),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state["recipe"] != self.recipe.identity()
            or state["environment"] != self.environment.identity()
            or state["replica"] != self.replica
            or state["n_lanes"] != self.n_lanes
        ):
            raise ValueError("PPO collector checkpoint identity differs")
        self._next_window = state["next_window"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.abort_window()
        self.abort_evaluation()
        self.rollout = None
