# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Joint root configuration and environment boundary validation."""

import json

import pytest

from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_libero10_accepts_online_reporting_and_rejects_geometry_mismatch(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config
    from cosmos_framework.posttrain.model.gr00t.build import CHECKPOINT_REVISION

    monkeypatch.delenv("WANDB_MODE", raising=False)
    config = make_config()
    checkpoint = tmp_path / CHECKPOINT_REVISION / "libero_10"
    checkpoint.mkdir(parents=True)
    for name, value in {
        "config.json": {"action_horizon": 40, "max_action_dim": 132, "num_inference_timesteps": 4},
        "processor_config.json": {
            "processor_kwargs": {"modality_configs": {"libero_sim": {"action": {"delta_indices": list(range(16))}}}}
        },
        "statistics.json": {},
        "embodiment_id.json": {},
        "model.safetensors.index.json": {},
    }.items():
        (checkpoint / name).write_text(json.dumps(value))
    config.model.checkpoint_path = str(checkpoint)
    config.environment.config.sim_python = "python"
    for split in ("train", "eval"):
        path = tmp_path / f"{split}.json"
        fixed_manifest(split, "libero_10", "legacy_100").write(path)
        setattr(config.environment.config, f"{split}_manifest", str(path))
    config.job.wandb_mode = "online"
    validate_joint_ppo_config(config, 2)
    config.model.execution_horizon = 16
    with pytest.raises(ValueError, match="action geometry"):
        validate_joint_ppo_config(config, 2)


def test_gr00t_rejects_unapproved_wandb_destination(monkeypatch):
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config

    monkeypatch.setenv("WANDB_PROJECT", "unapproved_project")
    config = make_config()
    assert config.job.wandb_mode == "disabled"
    config.job.wandb_mode = "online"
    with pytest.raises(ValueError, match="Invalid W&B project"):
        validate_joint_ppo_config(config, 2)


def test_shared_config_overrides_and_fixed_partition(tmp_path):
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    for split in ("train", "eval"):
        fixed_manifest(split).write(tmp_path / f"{split}.json")
    config = load_config(
        "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_ppo.py",
        [
            "--",
            "model.checkpoint_path=test-checkpoint",
            "environment.config.sim_python=python",
            "rl.max_steps=80",
            "rl.optimizer.lr=0.00001",
            f"environment.config.train_manifest={tmp_path / 'train.json'}",
            f"environment.config.eval_manifest={tmp_path / 'eval.json'}",
        ],
    )
    validate_joint_ppo_config(config, 2)
    recipe = instantiate(config.rl)
    assert recipe.max_steps == 80 and recipe.actor_lr == 1e-5
    assert recipe.identity()["optimizer"]["lr"] == recipe.actor_lr
    config.rl.max_steps = 81
    with pytest.raises(ValueError, match="divisible"):
        validate_joint_ppo_config(config, 2)
