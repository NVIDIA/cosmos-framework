# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Caller-driven decision windows over optional chunk-capable simulator sessions.

The simulator owns chunk execution and reset. This loop owns decision provenance and
window boundaries; it does not import a model, PPO, Torch or Ray.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import numpy as np

from cosmos_framework.posttrain.embodied.sim import SimCapabilities


class WindowReport(TypedDict):
    decisions: int
    policy_version: int
    session: str


class EvaluationWindowReport(TypedDict):
    episodes: list[dict[str, Any]]
    policy_version: int
    training_session_unchanged: bool


@dataclass(frozen=True)
class ChunkResult:
    action: np.ndarray
    observation: dict[str, Any]
    next_observation: dict[str, Any]
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    episode: dict[str, Any] | None = None
    control_terminated: tuple[bool, ...] | None = None
    control_truncated: tuple[bool, ...] | None = None
    episode_complete: bool | None = None

    @property
    def completed(self) -> bool:
        """Physical completion; legacy sessions finish at their first ending."""
        return (self.terminated or self.truncated) if self.episode_complete is None else self.episode_complete


class ChunkSession(Protocol):
    n_lanes: int
    session_id: str

    def observe(self) -> list[dict[str, Any]]: ...
    def sources(self, window: int) -> list[dict[str, Any]]: ...
    def step_chunk(self, actions: Sequence[np.ndarray]) -> list[ChunkResult]: ...
    def close(self) -> None: ...


class DecisionRollout(Protocol):
    def act(self, observations: Sequence[Mapping[str, Any]], *, context: dict) -> list[np.ndarray]: ...
    def feedback(self, observations: Sequence[Mapping[str, Any]], *, context: dict) -> None: ...


class ChunkSimulator(Protocol):
    """Peer simulator capability for persistent online windows instead of episode work units."""

    def capabilities(self) -> SimCapabilities: ...
    def probe(self) -> None: ...
    def open_window(self, version: int) -> ChunkSession: ...
    def gpu_uuid(self) -> str | None: ...
    def close(self) -> None: ...


def collect_window(session: ChunkSession, rollout: DecisionRollout, *, version: int, decisions: int) -> WindowReport:
    if type(version) is not int or version < 0 or type(decisions) is not int or decisions < 1:
        raise ValueError("collection version must be non-negative and decisions must be positive")
    count = 0
    for decision in range(decisions):
        sources = session.sources(version)
        context = {"sources": sources, "policy_version": version, "decision": decision}
        actions = rollout.act(session.observe(), context=context)
        outcomes = session.step_chunk(actions)
        if len(outcomes) != session.n_lanes:
            raise ValueError("chunk result count differs from the bound lanes")
        rollout.feedback(
            [{**result.next_observation, "executed_action": result.action} for result in outcomes],
            context={
                **context,
                "window_end": decision + 1 == decisions,
                "feedback": [
                    {
                        "rewards": list(result.rewards),
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "control_terminated": result.control_terminated,
                        "control_truncated": result.control_truncated,
                        "episode_complete": result.completed,
                        "episode": result.episode,
                    }
                    for result in outcomes
                ],
            },
        )
        count += len(outcomes)
    return {"decisions": count, "policy_version": version, "session": session.session_id}


class EvaluationChunkSimulator(Protocol):
    """Independent fixed-case sessions; closing them never resets the online session."""

    def open_evaluation(self, case: dict[str, int | str], seed: int) -> ChunkSession: ...
    def online_state_digest(self) -> str: ...


def evaluate_episodes(
    simulator: EvaluationChunkSimulator,
    rollout: DecisionRollout,
    *,
    cases: list[dict[str, int | str]],
    version: int,
    seed: int,
    decisions: int,
) -> dict:
    episodes = []
    for case in cases:
        session = simulator.open_evaluation(case, seed)
        rewards: list[float] = []
        try:
            for decision in range(decisions):
                context = {"policy_version": version, "decision": decision, "case": case, "seed": seed}
                actions = rollout.act(session.observe(), context=context)
                outcomes = session.step_chunk(actions)
                if len(outcomes) != 1:
                    raise ValueError("fixed-case evaluation requires exactly one lane")
                outcome = outcomes[0]
                rewards.extend(outcome.rewards)
                if outcome.completed:
                    if outcome.episode is None:
                        raise ValueError("evaluation completion is missing episode statistics")
                    episodes.append({**outcome.episode, "control_rewards": rewards})
                    break
            else:
                raise ValueError(f"evaluation case {case} exceeded its configured decision limit")
        finally:
            session.close()
    return {"episodes": episodes, "policy_version": version}


@dataclass(frozen=True)
class EvaluationEpisode:
    """Environment-validated episode facts for shared evaluation aggregation."""

    case_id: str
    group: str
    success: bool
    reward: float
    steps: int
    details: Mapping[str, Any]


@dataclass(frozen=True)
class WindowSimulatorSpec:
    """Plugin-owned launch data; the framework owns construction of its gateway role."""

    name: str
    embodiment: str
    python: str
    options: dict[str, Any]


class WindowEnvironment(Protocol):
    """Task-owned configuration, seeds and fixed-case protocol for a window collector."""

    execution_horizon: int
    action_dim: int
    evaluation_decisions: int

    def identity(self) -> dict: ...
    def validate_configuration(self) -> None: ...
    def gateway_spec(self, *, replica: int, n_lanes: int, seed: int) -> WindowSimulatorSpec: ...
    def training_seed(
        self, *, seed: int, version: int, replica: int, n_lanes: int, num_envs: int, decision: int
    ) -> int: ...
    def bootstrap_seed(self, *, seed: int, version: int, replica: int) -> int: ...
    def evaluation_seed(self, context: Mapping[str, Any]) -> int: ...
    def evaluation_cases(self, replica: int, replicas: int) -> list[dict[str, int | str]]: ...
    def validate_evaluation(self, episodes: Sequence[Mapping[str, Any]]) -> tuple[EvaluationEpisode, ...]: ...
    def evaluation_fingerprint(self) -> str: ...
