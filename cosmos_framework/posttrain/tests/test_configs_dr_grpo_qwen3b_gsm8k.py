# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU contract for the Dr-GRPO GSM8K recipe."""

from __future__ import annotations

import warnings

import pytest

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.configs import dr_grpo_qwen3b_gsm8k, grpo_qwen3b_gsm8k
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration
from cosmos_framework.utils.config import load_config


@pytest.mark.L0
@pytest.mark.CPU
def test_dr_grpo_recipe_is_a_distinct_configured_grpo_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    skip_recipe_registration(monkeypatch, grpo_qwen3b_gsm8k._make_root_config, PosttrainGRPOConfig)

    cfg = dr_grpo_qwen3b_gsm8k.make_config()

    assert cfg.algorithm == "dr_grpo"
    assert cfg.job.name == "qwen3b_bf16_dr_grpo_seed42"
    assert cfg.rl.grpo_surrogate == "clipped"
    assert cfg.rl.unbiased_advantage is True
    assert cfg.rl.loss_type == "seq-mean-token-sum-fixed"
    assert cfg.rollout.max_tokens == 512


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("algorithm", ["grpo", "dr_grpo", "gspo"])
def test_shipped_gsm8k_recipe_composes_without_generator_defaults(algorithm: str) -> None:
    cfg = load_config(f"cosmos_framework/posttrain/configs/{algorithm}_qwen3b_gsm8k.py", [])
    assert cfg.algorithm == algorithm
    assert cfg.rollout.max_tokens == 512
    if algorithm == "dr_grpo":
        assert cfg.rl.unbiased_advantage is True
        assert cfg.rl.loss_type == "seq-mean-token-sum-fixed"
    if algorithm == "gspo":
        assert cfg.rl.importance_sampling_level == "sequence"
        assert cfg.rl.loss_type == "seq-mean-token-mean"


@pytest.mark.L0
@pytest.mark.CPU
def test_dr_grpo_recipe_inherits_an_eps_the_fixed_normaliser_warns_about(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe flips `loss_type` but inherits the GSM8K TOML's `eps=1e-6`. Knobs are set by attribute
    after construction, so re-running validation on the assembled section is what a launch does when it
    re-instantiates `rl`: it surfaces the fixed-normaliser eps warning, and torch's default silences it."""
    skip_recipe_registration(monkeypatch, grpo_qwen3b_gsm8k._make_root_config, PosttrainGRPOConfig)
    cfg = dr_grpo_qwen3b_gsm8k.make_config()

    assert cfg.rl.optimizer.eps == 1e-6
    with pytest.warns(UserWarning, match="eps-dominated"):
        cfg.rl.__post_init__()
    cfg.rl.optimizer.eps = 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg.rl.__post_init__()
