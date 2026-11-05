# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Gaussian batch replay with recorded forward sizes for public PPO tests."""

import pytest
import torch

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.model.protocol import PPODecisionScores
from cosmos_framework.posttrain.tests.helpers.fixed_episode import FixedEpisodePolicy, collect_fixed
from cosmos_framework.posttrain.utils.dist import run_rank_converged


class BatchedFixedEpisodePolicy(FixedEpisodePolicy):
    def __init__(self, *, distributed=False):
        super().__init__(distributed=distributed)
        self.scored_batches = []

    def score_ppo_decisions(self, steps):
        def prepare():
            for step in steps:
                self.validate_ppo_record(step)
            traces = [step.policy_info["gaussian_trace"] for step in steps]
            return (
                torch.cat([trace.observations[trace.row : trace.row + 1] for trace in traces]),
                torch.cat([trace.actions[trace.row : trace.row + 1] for trace in traces]),
            )

        observations, actions = run_rank_converged(prepare, "Gaussian batch replay inputs")
        self.scored_batches.append((torch.is_grad_enabled(), tuple(step.source.sample_id for step in steps)))
        means, values = self._forward(observations)
        logprobs = torch.distributions.Normal(means, 1).log_prob(actions).sum(-1)
        return PPODecisionScores(logprobs, values, (logprobs.sum() + values.sum()) * 0)


def batch_recipe(**changes):
    return JointPPOConfig(
        **{
            "num_envs": 2,
            "window_decisions": 3,
            "global_minibatch": 6,
            "microbatch": 4,
            "dual_clip": None,
            "loss_reduction": "fixed_episode_mean",
            "time_limit_bootstrap": "zero",
            **changes,
        }
    )


def assert_checkpoint_identity_boundary(original, restored, changed, checkpoint):
    receipt = original.prepare_ppo_targets(collect_fixed(original.model, original.recipe))
    original.train_batch(receipt["batch_id"])
    original.clear_batch(receipt["batch_id"])
    original.set_publication_version(1)
    save_policy_checkpoint(original, checkpoint)
    original.finalize_checkpoints()
    restored.load(checkpoint)
    assert restored.evaluation_state() == original.evaluation_state()
    for actual, expected in zip(restored.model.net.parameters(), original.model.net.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    before = changed.evaluation_state()
    parameters = [value.detach().clone() for value in changed.model.net.parameters()]
    with pytest.raises(RuntimeError, match="identity differs"):
        changed.load(checkpoint)
    assert changed.evaluation_state() == before
    assert not changed.opt.state
    for actual, expected in zip(changed.model.net.parameters(), parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
