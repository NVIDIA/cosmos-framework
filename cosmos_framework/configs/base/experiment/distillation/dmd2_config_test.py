# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

import cosmos_framework.configs.base.experiment.distillation.dmd2_config as dmd2_config_module
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2OptimizerConfig, DMD2RFConfig


@pytest.mark.L0
@pytest.mark.CPU
def test_public_config_exports_are_explicit() -> None:
    assert dmd2_config_module.__all__ == (
        "DMD2OptimizerConfig",
        "DMD2RFConfig",
        "register_dmd2_optimizer",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_dmd2_defaults_preserve_export_and_inference_contract() -> None:
    config = DMD2RFConfig()

    assert config.fixed_step_sampler_config.t_list == [0.999, 0.75, 0.5, 0.25]
    assert config.fixed_step_sampler_config.sample_type == "sde"
    assert config.student_update_freq == 5
    assert config.simulation_mode == "forward"
    assert config.backward_grad_steps == 1
    assert config.vsd_gradient_space == "x0"
    assert config.vsd_loss_reduction == "mean"
    assert config.fake_score_loss_reduction == "active_mean"
    assert config.load_teacher_weights is True
    assert config.teacher_load_from is None
    assert config.student_load_from is None
    assert config.action_gen is False
    assert config.sound_gen is False
    assert config.vlm_config.pretrained_weights.enabled is False
    assert config.vlm_config_teacher.pretrained_weights.enabled is False
    assert config.vlm_config_fake_score.pretrained_weights.enabled is False


@pytest.mark.L0
@pytest.mark.CPU
def test_dmd2_optimizer_defaults_preserve_separate_student_and_critic_routes() -> None:
    config = DMD2OptimizerConfig()

    assert config.net.model is None
    assert config.net.optimizer_type == "FusedAdam"
    assert config.net.lr == 1e-6
    assert config.net.betas == [0.9, 0.99]
    assert config.fake_score.model is None
    assert config.fake_score.optimizer_type == "FusedAdam"
    assert config.fake_score.lr == 2e-7
    assert config.fake_score.betas == [0.0, 0.999]
