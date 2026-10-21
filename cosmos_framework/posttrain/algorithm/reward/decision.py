# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Environment-feedback requests for the shared transactional RewardWorker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cosmos_framework.posttrain.algorithm.reward import (
    BoundRewardProvider,
    RewardCandidate,
    RewardCandidateResult,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderResult,
    RewardProviderSpec,
)
from cosmos_framework.posttrain.schema import EnvironmentFeedback, Status, Trajectory


@dataclass(frozen=True)
class DecisionFeedback:
    feedback: EnvironmentFeedback
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class DecisionRewardRequest(RewardGroupRequest):
    decisions: tuple[DecisionFeedback, ...] = ()


class BoundDecisionRewardProvider(BoundRewardProvider):
    def prepare_group(self, trajectories: Sequence[Trajectory]) -> DecisionRewardRequest:
        decisions = []
        for index, trajectory in enumerate(trajectories):
            if trajectory.status != Status.COMPLETED or len(trajectory.transitions) != 1:
                raise RewardContractError(f"decision reward candidate {index} must contain one completed decision")
            step = trajectory.transitions[0]
            if not isinstance(step.feedback, EnvironmentFeedback):
                raise RewardContractError(f"decision reward candidate {index} is missing environment feedback")
            decisions.append(DecisionFeedback(step.feedback, step.done and not step.truncated, step.truncated))
        return DecisionRewardRequest(
            group_id=trajectories[0].group_id,
            candidates=tuple(RewardCandidate(t.id, completion="") for t in trajectories),
            decisions=tuple(decisions),
        )


class DecisionRewardTransform(Protocol):
    revision: str

    def __call__(self, feedback: EnvironmentFeedback, *, terminated: bool, truncated: bool) -> float: ...
    def identity(self) -> dict[str, object]: ...


class _DecisionGroupProvider:
    def __init__(self, transform: DecisionRewardTransform) -> None:
        self.transform = transform

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        if not isinstance(request, DecisionRewardRequest):
            raise RewardContractError("decision reward requires a typed environment-feedback request")
        return RewardGroupResult(
            request.group_id,
            tuple(
                RewardCandidateResult(
                    candidate.trajectory_id,
                    RewardProviderResult(
                        terms={
                            "task": self.transform(
                                row.feedback,
                                terminated=row.terminated,
                                truncated=row.truncated,
                            )
                        }
                    ),
                )
                for candidate, row in zip(request.candidates, request.decisions, strict=True)
            ),
        )


def build_decision_reward_provider(transform: DecisionRewardTransform) -> BoundRewardProvider:
    """Bind task arithmetic into the existing provider schema and transactional worker."""
    provider = _DecisionGroupProvider(transform)
    spec = RewardProviderSpec(
        factory=lambda: provider,
        revision=transform.revision,
        term_names=("task",),
        semantic_identity=transform.identity(),
    )
    return BoundDecisionRewardProvider("environment", spec, provider)
