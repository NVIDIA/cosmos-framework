# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Production-path smoke tests for posttrain recipe root registration."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig
from cosmos_framework.posttrain.configs import opd_qwen1p5b_countdown as opd_recipe


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.serial
def test_opd_recipe_builds_with_the_registered_production_root() -> None:
    """Keep one non-L0 smoke through the full Hydra catalog registration path."""
    cfg = opd_recipe.build_config()

    assert isinstance(cfg, PosttrainOPDConfig)
    assert cfg.algorithm == "opd"
    assert cfg.job.group == "posttrain_opd_demo"
