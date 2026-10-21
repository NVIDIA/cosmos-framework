# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0/CPU contract for the GSPO GSM8K recipe."""

from __future__ import annotations

import warnings

import pytest

from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.configs import grpo_qwen3b_gsm8k, gspo_qwen3b_gsm8k
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration


@pytest.mark.L0
@pytest.mark.CPU
def test_gspo_recipe_selects_the_paper_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe is the GRPO GSM8K run with the GSPO knob set (arXiv 2507.18071 sec. 5.1) and its
    own provenance; re-running validation proves the assembled section is legal and silent."""
    skip_recipe_registration(monkeypatch, grpo_qwen3b_gsm8k._make_root_config, PosttrainGRPOConfig)
    cfg = gspo_qwen3b_gsm8k.make_config()

    assert (cfg.algorithm, cfg.job.name) == ("gspo", "qwen3b_bf16_gspo_seed42")
    assert cfg.rl.importance_sampling_level == "sequence"
    assert cfg.rl.lower_bound_ratio == 0.0
    assert (cfg.rl.epsilon_low, cfg.rl.epsilon_high) == (3e-4, 4e-4)
    assert cfg.rl.loss_type == "seq-mean-token-mean"
    assert cfg.rl.n_mini == 4  # batch 128 / mini 32
    assert cfg.rl.certified_deterministic_forward is True
    # Knobs are set by attribute after construction, so validation has to be re-run explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg.rl.__post_init__()
