# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deterministic physical episodes with raw success events inside action chunks."""

from __future__ import annotations

import numpy as np
import torch

from cosmos_framework.posttrain.embodied.window import ChunkResult, collect_window
from cosmos_framework.posttrain.model.protocol import PPODecisionScores
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
from cosmos_framework.posttrain.tests.helpers.joint_ppo import (
    GaussianEnvironment,
    GaussianPolicy,
    GaussianSession,
    reward_provider,
)


class FixedEpisodePolicy(GaussianPolicy):
    horizon = 3

    def __init__(self, *, distributed=False):
        super().__init__()
        self._forward = torch.nn.parallel.DistributedDataParallel(self.net) if distributed else self.net

    def score_ppo_decision(self, step):
        trace = step.policy_info["gaussian_trace"]
        means, values = self._forward(trace.observations)
        logprobs = torch.distributions.Normal(means, 1).log_prob(trace.actions).sum(-1)
        row = trace.row
        return PPODecisionScores(logprobs[row : row + 1], values[row : row + 1], (logprobs.sum() + values.sum()) * 0)

    def sample_decisions(self, *args, **kwargs):
        samples = super().sample_decisions(*args, **kwargs)
        for sample in samples:
            sample.transitions[0].action = np.repeat(sample.transitions[0].action, self.horizon, axis=0)
        return samples


class FixedEpisodeEnvironment(GaussianEnvironment):
    execution_horizon = FixedEpisodePolicy.horizon


class FixedEpisodeSession(GaussianSession):
    def __init__(self, successes=(2, 8), *, replica=0):
        super().__init__(len(successes))
        self.successes = successes
        self.replica = replica
        self.session_id = f"fixed-session-{replica}"

    def sources(self, window):
        return [
            {
                **source,
                "sample_id": f"r{self.replica}:{source['sample_id']}",
                "episode_id": f"r{self.replica}:{source['episode_id']}",
            }
            for source in super().sources(window)
        ]

    def step_chunk(self, actions):
        horizon = FixedEpisodePolicy.horizon
        first = self.turn * horizon
        self.turn += 1
        observations = self.observe()
        results = []
        for lane, action in enumerate(actions):
            flags = tuple(first + index + 1 == self.successes[lane] for index in range(horizon))
            truncated = tuple(first + index + 1 == 3 * horizon for index in range(horizon))
            results.append(
                ChunkResult(
                    action.copy(),
                    observations[lane],
                    observations[lane],
                    tuple(float(flag) for flag in flags),
                    any(flags),
                    any(truncated),
                    control_terminated=flags,
                    control_truncated=truncated,
                    episode_complete=self.turn == 3,
                )
            )
        return results


def collect_fixed(model, recipe, *, successes=(2, 8), replica=0):
    collector = PPOWindowCollector(recipe, replica, len(successes), environment=FixedEpisodeEnvironment())
    collector.bind_rollout(model)
    try:
        collector.begin_window(0)
        session = FixedEpisodeSession(successes, replica=replica)
        report = collect_window(session, collector, version=0, decisions=3)
        return RewardWorker([reward_provider()], training_group_size=1).score(collector.finish_window(report))
    finally:
        collector.close()
