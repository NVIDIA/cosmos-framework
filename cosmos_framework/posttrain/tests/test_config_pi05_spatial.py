# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pi0.5 preflight rejects unsupported runtime and actor-local episode inventories."""

import sys

import pytest

from cosmos_framework.posttrain.config.config_joint_ppo import resolve_joint_config
from cosmos_framework.posttrain.configs.pi05_libero_spatial_ppo import make_config
from cosmos_framework.posttrain.entrypoints.joint_ppo import validate_config
from cosmos_framework.posttrain.model.pi05.build import validate_joint_recipe

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_pi05_recipe_uses_complete_fixed_episodes_and_native_model_geometry() -> None:
    from cosmos_framework.utils.lazy_config import instantiate

    config = resolve_joint_config(make_config())
    model = instantiate(config.model)
    environment = instantiate(config.environment)
    assert config.rl.loss_reduction == "fixed_episode_mean" and config.rl.dual_clip is None
    assert config.rl.verify_behavior_logprobs is False
    assert config.rl.window_decisions * model.action_chunk == 8 * environment.config.max_episode_steps
    assert config.rl.decisions // config.rl.global_minibatch == 12
    assert config.rl.validation.every_steps == 40 * (config.rl.decisions // config.rl.global_minibatch)
    assert config.recipe_check.recipe.loss_reduction == config.rl.loss_reduction


def test_pi05_recipe_rejects_unmasked_reduction_before_loading_checkpoint() -> None:
    from cosmos_framework.utils.lazy_config import instantiate

    config = resolve_joint_config(make_config())
    config.rl.loss_reduction = "decision_mean"
    with pytest.raises(ValueError, match="rl.loss_reduction=fixed_episode_mean"):
        validate_joint_recipe(instantiate(config.model), instantiate(config.environment).config, config.rl, ranks=4)


def test_pi05_recipe_check_rejects_partial_episode_even_with_divisible_minibatches() -> None:
    from cosmos_framework.utils.lazy_config import instantiate

    config = make_config()
    config.rl.num_envs = 64
    config.rl.window_decisions = 16
    config.rl.global_minibatch = config.rl.decisions
    config.rl.__post_init__()
    config = resolve_joint_config(config)
    with pytest.raises(ValueError, match="rl.window_decisions must contain complete physical episodes"):
        instantiate(config.recipe_check, ranks=4)


@pytest.mark.parametrize("sim_python", [None, ""])
def test_pi05_preflight_rejects_missing_simulator_before_sources(monkeypatch, sim_python):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    if sim_python is None:
        monkeypatch.delenv("POSTTRAIN_SIM_PYTHON", raising=False)
    else:
        monkeypatch.setenv("POSTTRAIN_SIM_PYTHON", sim_python)
    with pytest.raises(ValueError, match="environment.config.sim_python is required"):
        validate_config(make_config(), ranks=8)


def test_pi05_recipe_rejects_singleton_episode_before_loading_checkpoint():
    from cosmos_framework.utils.lazy_config import instantiate

    config = make_config()
    config.rl.num_envs = 8
    environment = instantiate(config.environment).config
    config.rl.window_decisions = environment.max_episode_steps // environment.execution_horizon
    config.rl.global_minibatch = config.rl.decisions
    config = resolve_joint_config(config)
    with pytest.raises(ValueError, match="at least two complete physical episodes per learner rank.*got 1"):
        instantiate(config.recipe_check, ranks=8)


@pytest.mark.parametrize(
    "num_envs,ranks,episodes_per_lane,simulator_setting",
    [(16, 8, 1, "environment"), (8, 8, 2, "config"), (64, 8, 1, "cli"), (64, 8, 8, "config")],
)
def test_pi05_preflight_accepts_complete_episode_inventory_and_simulator_overrides(
    tmp_path, monkeypatch, num_envs, ranks, episodes_per_lane, simulator_setting
):
    from cosmos_framework.utils.config_helper import override
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest
    from cosmos_framework.posttrain.tests.helpers.pi05_loading import write_source

    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("POSTTRAIN_SIM_PYTHON", raising=False)
    if simulator_setting == "environment":
        monkeypatch.setenv("POSTTRAIN_SIM_PYTHON", sys.executable)
    config = make_config()
    if simulator_setting == "config":
        config.environment.config.sim_python = sys.executable
    source = write_source(tmp_path)
    for name in ("checkpoint_path", "tokenizer_path", "norm_stats_path"):
        setattr(config.model, name, getattr(source, name))
    for split in ("train", "eval"):
        path = tmp_path / f"{split}.json"
        fixed_manifest(split, profile="shared_500").write(path)
        setattr(config.environment.config, f"{split}_manifest", str(path))
    environment = config.environment.config
    config.rl.num_envs = num_envs
    config.rl.window_decisions = episodes_per_lane * environment.max_episode_steps // environment.execution_horizon
    config.rl.global_minibatch = config.rl.decisions
    if simulator_setting == "cli":
        config = override(config, ["--", f"environment.config.sim_python={sys.executable}"])
    validate_config(config, ranks)
    assert config.environment.config.sim_python == sys.executable
