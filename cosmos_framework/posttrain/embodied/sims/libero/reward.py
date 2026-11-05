# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO reward transformation bound into the shared RewardWorker transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from cosmos_framework.posttrain.embodied.sims.libero.protocol import DEFAULT_EXECUTION_PROFILE, execution_protocol
from cosmos_framework.posttrain.schema import EnvironmentFeedback


def decision_reward(
    control_rewards: tuple[float, ...],
    *,
    suite: str,
    terminated: bool,
    truncated: bool,
    execution_profile: str = DEFAULT_EXECUTION_PROFILE,
) -> float:
    """Raw feedback is immutable; the environment protocol defines its training reward."""
    if execution_protocol(suite, execution_profile).reward == "relative":
        return sum(control_rewards)
    return -1.0 if truncated and not terminated else 0.0


@dataclass(frozen=True)
class LiberoDecisionReward:
    """Pure environment-owned reward arithmetic; no training/framework dependency."""

    suite: str
    execution_profile: str = DEFAULT_EXECUTION_PROFILE
    revision: ClassVar[str] = "libero-decision-reward-v1"

    def __post_init__(self) -> None:
        execution_protocol(self.suite, self.execution_profile)

    def __call__(self, feedback: EnvironmentFeedback, *, terminated: bool, truncated: bool) -> float:
        return decision_reward(
            feedback.control_rewards,
            suite=self.suite,
            terminated=terminated,
            truncated=truncated,
            execution_profile=self.execution_profile,
        )

    def identity(self) -> dict[str, object]:
        result = {"suite": self.suite, "reward": execution_protocol(self.suite, self.execution_profile).reward}
        if self.execution_profile != DEFAULT_EXECUTION_PROFILE:
            result["execution_profile"] = self.execution_profile
        return result
