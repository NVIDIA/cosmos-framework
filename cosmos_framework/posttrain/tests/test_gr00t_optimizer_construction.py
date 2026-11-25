# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native factory/entry routing, with CPU backend doubles only at the CUDA boundary."""

from dataclasses import asdict

import pytest

from cosmos_framework.posttrain.config.rl import JointPPOConfig, OptimizerConfig
from cosmos_framework.posttrain.tests.helpers.optimizer_backend import backend_capture  # noqa: F401


@pytest.mark.parametrize("name", ["adamw", "adamw_bnb_8bit", "paged_adamw_8bit"])
def test_joint_ppo_real_actor_value_factory_preserves_ownership(name, backend_capture):
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.tests.helpers.joint_ppo import GaussianPolicy

    policy = GaussianPolicy()
    policy.net.bfloat16()
    recipe = JointPPOConfig(
        optimizer=OptimizerConfig(name=name, update_policy="fp32_master" if name != "adamw" else "parameter")
    )
    trainer = JointPPOTrainer(policy, recipe)
    names = {id(p): n for n, p in policy.net.named_parameters()}
    assert [g["name"] for g in trainer.opt.param_groups] == ["actor", "value"]
    for group, partition, lr in zip(
        trainer.opt.param_groups,
        (policy.ppo_parameters.actor, policy.ppo_parameters.value),
        (recipe.actor_lr, recipe.value_lr),
    ):
        assert tuple(names[id(p)] for p in group["params"]) == partition
        assert group["lr"] == lr
    assert bool(backend_capture) == (name != "adamw")
    if backend_capture:
        assert backend_capture[0][2].name == name


def test_legacy_joint_identity_does_not_gain_opt_in_optimizer_fields():
    recipe = JointPPOConfig()
    legacy = asdict(recipe.optimizer)
    for field in ("update_policy", "bnb_min_8bit_size", "bnb_fp32_state_names"):
        legacy.pop(field)
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    assert recipe.identity()["optimizer"] == canonical_config_value(legacy)
    recipe.optimizer.update_policy = "fp32_master"
    assert recipe.identity()["optimizer"]["update_policy"] == "fp32_master"


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_joint_quantized_policy_rejected_before_model_construction(name):
    with pytest.raises(ValueError, match="update_policy=fp32_master"):
        JointPPOConfig(optimizer=OptimizerConfig(name=name))
