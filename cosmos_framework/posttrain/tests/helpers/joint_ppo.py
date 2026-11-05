# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent Gaussian actor/value and deterministic chunk environment for Joint PPO tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, JointPPOStats
from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidateResult,
    RewardGroupResult,
    RewardProviderResult,
    RewardProviderSpec,
)
from cosmos_framework.posttrain.algorithm.reward.decision import BoundDecisionRewardProvider, DecisionRewardRequest
from cosmos_framework.posttrain.embodied.window import ChunkResult, collect_window
from cosmos_framework.posttrain.model.protocol import PPODecisionScores, PPOParameterPartition
from cosmos_framework.posttrain.schema import ObsBundle, SampleSource, Status, Trajectory, Transition


class GaussianHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 4)
        self.actor = nn.Linear(4, 2)
        self.value = nn.Linear(4, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations).tanh()
        return self.actor(features), self.value(features).squeeze(-1)


@dataclass(frozen=True)
class GaussianTrace:
    observations: torch.Tensor
    actions: torch.Tensor
    row: int


class GaussianPolicy:
    ppo_logprob_tolerance = 1e-6

    def __init__(self) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.net = GaussianHeads()
        self.precision = SimpleNamespace(transfer="float32")
        names = tuple(name for name, _ in self.net.named_parameters())
        self.ppo_parameters = PPOParameterPartition(
            tuple(name for name in names if not name.startswith("value.")),
            tuple(name for name in names if name.startswith("value.")),
        )
        self.ppo_padding = None

    def ppo_identity(self) -> dict:
        return {"model": "independent-gaussian-shared-backbone-v1"}

    def ppo_frozen_state(self) -> dict[str, str]:
        return {}

    def validate_ppo_record(self, step: Transition) -> None:
        trace = step.policy_info["gaussian_trace"]
        if not isinstance(trace, GaussianTrace) or trace.actions.shape != trace.observations.shape:
            raise ValueError("Gaussian trace geometry differs from its sampling invocation")
        if not 0 <= trace.row < len(trace.observations):
            raise ValueError("Gaussian trace row is out of range")

    def score_ppo_decision(self, step: Transition) -> PPODecisionScores:
        trace = step.policy_info["gaussian_trace"]
        means, values = self.net(trace.observations)
        logprobs = torch.distributions.Normal(means, 1).log_prob(trace.actions).sum(-1)
        row = trace.row
        return PPODecisionScores(logprobs[row : row + 1], values[row : row + 1], (logprobs.sum() + values.sum()) * 0)

    def sample_decisions(self, observations, *, sources, seed, weight_version):
        features = torch.tensor(np.stack([row["features"] for row in observations]), dtype=torch.float32)
        with torch.no_grad():
            means, values = self.net(features)
            actions = means + torch.randn(means.shape, generator=torch.Generator().manual_seed(seed))
            logprobs = torch.distributions.Normal(means, 1).log_prob(actions).sum(-1)
        samples = []
        for row, source in enumerate(sources):
            identifier = source["sample_id"]
            step = Transition(
                obs=ObsBundle(id=identifier, representations={"raw": observations[row]}),
                action=actions[row : row + 1].numpy(),
                step_idx=source["step_idx"],
                source=SampleSource(identifier, source["episode_id"], source["fragment_id"]),
                policy_info={"gaussian_trace": GaussianTrace(features, actions, row), "weight_version": weight_version},
                algo_extra={STATS_KEY: JointPPOStats(float(logprobs[row]), float(values[row]))},
                info=dict(source),
            )
            samples.append(
                Trajectory(
                    [step], id=identifier, session=source["session"], group_id=identifier, status=Status.COMPLETED
                )
            )
        return samples

    def bootstrap_values(self, observations, *, seed, weight_version):
        with torch.no_grad():
            _, values = self.net(torch.tensor(np.stack([row["features"] for row in observations]), dtype=torch.float32))
        return tuple(values.tolist())


class GaussianSession:
    session_id = "independent-gaussian-session"

    def __init__(self, n_lanes=2):
        self.n_lanes = n_lanes
        self.turn = 0

    def observe(self):
        return [{"features": np.array([lane + 1, self.turn / 10], dtype=np.float32)} for lane in range(self.n_lanes)]

    def sources(self, window):
        return [
            {
                "sample_id": f"{window}:{lane}:{self.turn}",
                "episode_id": f"episode-{lane}",
                "fragment_id": f"window-{window}-{lane}",
                "session": self.session_id,
                "step_idx": self.turn,
            }
            for lane in range(self.n_lanes)
        ]

    def step_chunk(self, actions):
        before = self.observe()
        self.turn += 1
        after = self.observe()
        return [
            ChunkResult(action, before[lane], after[lane], (float(1 - np.square(action).mean()),), False, False)
            for lane, action in enumerate(actions)
        ]


class GaussianGateway:
    def __init__(self):
        self.session = GaussianSession()
        self.closed = False

    def run_window(self, *, policy, version, decisions):
        if self.closed:
            raise RuntimeError("collection after gateway close")
        return collect_window(self.session, policy, version=version, decisions=decisions)

    def close(self):
        self.closed = True


class GaussianEnvironment:
    execution_horizon = 1
    action_dim = 2

    def identity(self):
        return {"environment": "independent-gaussian-v1"}

    def training_seed(self, *, seed, version, replica, n_lanes, num_envs, decision):
        return seed + 100 * version + 10 * replica + decision

    def bootstrap_seed(self, *, seed, version, replica):
        return seed


class GaussianReward:
    def evaluate(self, request: DecisionRewardRequest):
        return RewardGroupResult(
            request.group_id,
            tuple(
                RewardCandidateResult(
                    candidate.trajectory_id, RewardProviderResult(terms={"task": sum(row.feedback.control_rewards)})
                )
                for candidate, row in zip(request.candidates, request.decisions, strict=True)
            ),
        )


def reward_provider():
    provider = GaussianReward()
    spec = RewardProviderSpec(factory=lambda: provider, revision="gaussian-env-v1")
    return BoundDecisionRewardProvider("environment", spec, provider)
