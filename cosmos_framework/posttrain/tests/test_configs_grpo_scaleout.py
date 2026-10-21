# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for the two scale-out profiles and their tunable layout helper."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cosmos_framework.posttrain.config.build import resolve_policy_parallelism
from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.configs import grpo_qwen3vl_rlvr, grpo_qwen3vl_scaleout, grpo_qwen3vl_scaleout_local
from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell, validate_batch_divisibility
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch) -> PosttrainGRPOConfig:
    skip_recipe_registration(monkeypatch, grpo_qwen3vl_rlvr._make_root_config, PosttrainGRPOConfig)
    return grpo_qwen3vl_scaleout.make_config()


def test_scaleout_batch_and_policy_are_one_global_update(cfg: PosttrainGRPOConfig) -> None:
    assert resolve_policy_parallelism(cfg, 1024) == (8, 128)
    assert cfg.grid.shapes == {"policy": (1, 1024), "rollout": (256, 4)}
    validate_batch_divisibility(
        batch_size=cfg.rl.batch_per_replica,
        mini_batch_size=cfg.rl.mini_batch_per_replica,
        micro_batch_size=cfg.rl.micro_batch_per_replica,
        dp_world=1024,
    )
    assert (
        prompts_per_data_cell(policy_batch_size=cfg.rl.batch_per_replica, rollouts_per_prompt=4, data_replicas=256) == 1
    )
    assert cfg.rollout.load_format == "dummy" and cfg.rollout.sleep_level == 2 and cfg.rl.colocated_offload
    assert cfg.rl.weight_sync_transport is None and not cfg.rl.weight_sync_r2r_broadcast
    assert cfg.rl.weight_sync_bucket_bytes == grpo_qwen3vl_scaleout.SCALEOUT_WEIGHT_SYNC_BUCKET_BYTES == 4 * 1024**3


def test_layout_change_preserves_learning_and_sampling_contract(cfg: PosttrainGRPOConfig) -> None:
    cfg.rl.min_informative_groups = 32
    before = copy.deepcopy(cfg.rl)
    grpo_qwen3vl_scaleout.apply_scaleout_layout(cfg, shard_degree=64)
    assert cfg.rl == before
    assert resolve_policy_parallelism(cfg, 1024) == (16, 64)
    structured = OmegaConf.structured(cfg.grid)
    assert OmegaConf.to_container(structured)["gpu_node_domains"] == {}


def test_local_profile_requires_explicit_inventory(
    cfg: PosttrainGRPOConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GPU_NODE_DOMAINS_FILE", raising=False)
    with pytest.raises(KeyError, match="GPU_NODE_DOMAINS_FILE"):
        grpo_qwen3vl_scaleout_local.make_config()
    inventory = {f"node-{i}": f"domain-{i // 16}" for i in range(256)}
    path = tmp_path / "domains.json"
    path.write_text(json.dumps(inventory))
    monkeypatch.setenv("GPU_NODE_DOMAINS_FILE", str(path))
    local = grpo_qwen3vl_scaleout_local.make_config()
    assert resolve_policy_parallelism(local, 1024) == (16, 64)
    assert local.grid.gpu_node_domains == inventory
    assert local.rl == cfg.rl


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_ranks": True},
        {"policy_ranks": 0},
        {"shard_degree": -1},
        {"shard_degree": 3},
        {"rollout_tp": 3},
        {"policy_ranks": 8, "shard_degree": 2, "rollout_tp": 4},
    ],
)
def test_incomplete_meshes_fail_before_allocation(cfg: PosttrainGRPOConfig, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        grpo_qwen3vl_scaleout.apply_scaleout_layout(cfg, **kwargs)
