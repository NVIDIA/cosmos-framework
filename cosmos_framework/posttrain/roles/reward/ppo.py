# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""PPO reward role composed from the shared reward worker."""

from __future__ import annotations

from cosmos_framework.posttrain.algorithm.ppo.views import (
    compact_ppo_reward_batch,
    project_rollout_for_role,
    seal_ppo_reward_batch,
    validate_ppo_reward_context_batch,
)
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
from cosmos_framework.posttrain.schema import Trajectory


class PPORewardRole:
    """Adapt PPO trajectories to the shared ``RewardWorker`` contract."""

    def __init__(self, worker: RewardWorker) -> None:
        self.worker = worker

    def score(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Score one rollout shard and return complete PPO trajectories."""
        reward_inputs = project_rollout_for_role(trajectories, "reward")
        validate_ppo_reward_context_batch(reward_inputs)
        rewards = compact_ppo_reward_batch(self.worker.score(reward_inputs))
        return seal_ppo_reward_batch(trajectories, rewards)


__all__ = ["PPORewardRole"]
