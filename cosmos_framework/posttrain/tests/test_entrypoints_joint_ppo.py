# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint PPO reward atomicity and public architecture contracts."""

import pytest

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
from cosmos_framework.posttrain.tests.helpers.joint_ppo import (
    GaussianEnvironment,
    GaussianGateway,
    GaussianPolicy,
    reward_provider,
)
from cosmos_framework.posttrain.tests.helpers.window_control import LocalWindowControl

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_reward_late_invalid_feedback_commits_no_earlier_rewards():
    config = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    model, gateway = GaussianPolicy(), GaussianGateway()
    collector = LocalWindowControl(model, gateway, config, 0, 2, environment=GaussianEnvironment())
    try:
        samples = collector.collect_window(0)
        samples[-1].transitions[0].feedback = None
        reward = RewardWorker([reward_provider()], training_group_size=1)
        with pytest.raises(ValueError, match="feedback"):
            reward.score(samples)
        assert all(sample.transitions[0].reward is None for sample in samples)
    finally:
        collector.close()


def test_joint_manifest_has_reward_replay_and_no_critic_or_reference():
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.entrypoints.joint_ppo import manifest

    graph = manifest()
    assert bind_check(graph).ok
    assert {role.name for role in graph.roles} == {"sim", "eval_sim", "collector", "rollout", "rm", "replay", "policy"}
    assert {(edge.src, edge.dst) for edge in graph.edges} >= {
        ("collector", "rm"),
        ("rm", "replay"),
        ("replay", "policy"),
        ("policy", "rollout"),
    }
