# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint PPO boundary policies and actor-local sample requirements."""

from dataclasses import replace

import pytest

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as make_libero10_config
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_timeout_policy_defaults_and_reference_recipes_have_distinct_identities():
    generic = JointPPOConfig()
    assert generic.time_limit_bootstrap == "value"
    assert generic.identity()["algorithm"] == "joint-ppo-v3"
    reference = replace(generic, time_limit_bootstrap="zero")
    assert generic.identity() != reference.identity()
    for builder in (make_config, make_libero10_config):
        recipe = instantiate(builder().rl)
        assert recipe.time_limit_bootstrap == "zero"
        assert recipe.identity()["time_limit_bootstrap"] == "zero"


def test_timeout_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="rl.time_limit_bootstrap"):
        JointPPOConfig(time_limit_bootstrap="terminal")


@pytest.mark.parametrize("num_envs,ranks", [(1, 1), (2, 2)])
def test_singleton_actor_shards_fail_before_component_construction(num_envs, ranks):
    config = make_config()
    config.rl.num_envs = num_envs
    config.rl.window_decisions = 1
    config.rl.global_minibatch = num_envs
    config.rl.microbatch = 1
    with pytest.raises(ValueError, match="rl.num_envs.*rl.window_decisions.*actor-local normalization"):
        validate_joint_ppo_config(config, ranks)


@pytest.mark.parametrize("num_envs,decisions,ranks", [(2, 2, 2), (4, 1, 2)])
def test_two_decisions_per_actor_pass_the_public_configuration_boundary(
    tmp_path, monkeypatch, num_envs, decisions, ranks
):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    config = make_config()
    config.rl.num_envs = num_envs
    config.rl.window_decisions = decisions
    config.rl.global_minibatch = num_envs * decisions
    config.rl.microbatch = 1
    config.model.checkpoint_path = "test-checkpoint"
    config.environment.config.sim_python = "python"
    for split in ("train", "eval"):
        path = tmp_path / f"{split}.json"
        fixed_manifest(split).write(path)
        setattr(config.environment.config, f"{split}_manifest", str(path))
    validate_joint_ppo_config(config, ranks)
