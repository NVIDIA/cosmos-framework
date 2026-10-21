# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bimanual stage inputs, model contracts, and fresh optimizer boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf.errors import InterpolationResolutionError, MissingMandatoryValue

from cosmos_framework.configs.simulation.common.bimanual_profiles import EXPERIMENT_NAMES, LOCAL_EXPERIMENT_NAMES
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model.generator.distillation.self_forcing_dmd import SelfForcingDMDModel
from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import LazyConfig

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]

_INPUTS = {
    "COSMOS_SIM_BIMANUAL_GA_CHECKPOINT": "/ckpt/ga",
    "COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT": "/ckpt/stage0",
    "COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT": "/ckpt/stage1",
    "COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT": "/ckpt/stage2",
}
_GA, _S0, _S1, _S2 = _INPUTS
_REQUIRED = ((_GA,), (_S0,), (_S1,), (_S0, _S2))


def _compose(monkeypatch: pytest.MonkeyPatch, stage: int, unset: str | None = None) -> Any:
    for name, value in _INPUTS.items():
        if name in _REQUIRED[stage] and name != unset:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CHECKPOINT_CREDENTIAL_FILE", raising=False)
    return load_config(
        "cosmos_framework/configs/simulation/config.py",
        ["--", f"experiment={EXPERIMENT_NAMES[stage]}"],
        experiment_module="cosmos_framework.configs.simulation.experiment.bimanual.stages",
    )


@pytest.mark.parametrize("stage", range(4))
def test_stage_model_and_warm_start_contract(monkeypatch: pytest.MonkeyPatch, stage: int) -> None:
    config = _compose(monkeypatch, stage)
    model = config.model.config
    assert config.trainer.max_iter == (35_000, 8_000, 4_000, 1_500)[stage]
    assert config.checkpoint.load_training_state is False
    assert config.checkpoint.strict_resume is True
    assert model.action_gen is True
    assert model.sound_gen is False
    assert model.max_action_dim == 64
    assert model.num_embodiment_domains == 32
    assert model.ema.enabled is True
    assert model.video_temporal_causal is (stage != 0)
    assert model.joint_attn_implementation == ("two_way" if stage == 0 else "three_way")
    assert model.vlm_config.tokenizer.config_variant == "hf"
    assert model.tokenizer.bucket_name == ""
    assert model.tokenizer.object_store_credential_path_pretrained == ""
    expected_resolutions = {"480"} if stage == 3 else {"256", "480"}
    assert set(model.rectified_flow_training_config.shift) == expected_resolutions
    assert set(model.tokenizer.encode_chunk_frames) == expected_resolutions
    assert model.action_schema == "unified_v1"
    assert model.action_global_asinh is True
    assert model.action_metric_gripper is True
    assert model.sound_tokenizer is None
    if stage < 3:
        assert config.checkpoint.load_path == ("/ckpt/ga", "/ckpt/stage0", "/ckpt/stage1")[stage]
        assert config.checkpoint.load_ema_to_reg is False
        assert list(config.checkpoint.keys_not_to_resume) == []
        assert list(config.checkpoint.keys_to_skip_loading) == (
            ["net_ema.", "action2llm", "llm2action", "action_modality_embed"] if stage == 0 else []
        )
    if stage == 0:
        assert config.trainer.callbacks.initialize_ema_from_regular.source_checkpoint == "/ckpt/ga"
        assert config.optimizer.lr == 1e-5
        assert config.optimizer.lr_multipliers.action2llm == 5.0
        assert "llm2action" not in config.optimizer.keys_to_select
    else:
        assert model.teacher_forcing_frames_per_chunk == 2
    if stage == 2:
        assert model.teacher_forcing_dcm_teacher_load_from.load_path == "/ckpt/stage1/model"
        assert model.teacher_forcing_dcm_total_steps == 48
        assert model.teacher_forcing_dcm_teacher_guidance == 3.0
    if stage == 3:
        assert model.student_load_from.load_path == "/ckpt/stage2/model"
        assert model.teacher_load_from.load_path == "/ckpt/stage0/model"
        assert model.teacher_guidance == 1.0
        assert model.action_conditioned_teacher_guidance == 1.0
        assert model.student_update_freq == 6
        assert list(model.sf_simulation_steps_per_frame) == [4]
        assert len(model.sf_backward_timestep_schedules) == 1
        assert config.optimizer.net.lr == 1e-6
        assert config.optimizer.fake_score.lr == 2e-7


@pytest.mark.parametrize("stage", (2, 3))
def test_distilled_schedule_matches_four_step_contract(monkeypatch: pytest.MonkeyPatch, stage: int) -> None:
    model_config = _compose(monkeypatch, stage).model.config
    ladder = [1.0, 15 / 16, 5 / 6, 5 / 8]
    assert list(model_config.fixed_step_sampler_config.t_list) == ladder
    if stage == 3:
        # Exercise the actual backend validator without constructing any networks.
        model = object.__new__(SelfForcingDMDModel)
        model.config = model_config
        assert model._expand_self_forcing_midpoint_schedules(3) == [ladder[1:]] * 3
        assert model._self_forcing_timestep_schedule_for_frame(0, 3, step_counts=[4, 4, 4]) == ladder + [0.0]


@pytest.mark.parametrize(("stage", "unset"), [(stage, name) for stage in range(4) for name in _REQUIRED[stage]])
def test_missing_stage_input_fails_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, stage: int, unset: str
) -> None:
    with pytest.raises(InterpolationResolutionError, match=unset):
        _compose(monkeypatch, stage, unset)


@pytest.mark.parametrize("stage", range(4))
def test_recipe_requires_a_real_dataloader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: int) -> None:
    # Recipes are model/optimization definitions; they must not silently use
    # another model's data or train on a synthetic substitute.
    with pytest.raises(MissingMandatoryValue, match="dataloader_train._target_"):
        LazyConfig.save_yaml(_compose(monkeypatch, stage), str(tmp_path / "config.yaml"))


@pytest.mark.parametrize("stage", range(4))
def test_local_toml_uses_native_loader_and_preserves_stage_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: int
) -> None:
    _compose(monkeypatch, stage)
    monkeypatch.setenv("COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE", "/data/bimanual")
    root = Path(__file__).resolve().parents[3]
    config = load_experiment_from_toml(root / "examples/toml/sft_config" / f"{EXPERIMENT_NAMES[stage]}.toml")
    loader = config.dataloader_train
    assert str(loader._target_).endswith("make_bimanual_local_dataloader")
    assert loader.bundle_path == "/data/bimanual"
    assert loader.stage == stage
    assert loader.max_samples_per_batch == (None if stage == 0 else 1)
    assert loader.max_sequence_length == (400_000 if stage == 0 else None)
    assert config.model.config.action_gen is True
    assert config.model.config.vlm_config.tokenizer.pretrained_model_name == "/data/bimanual/resources/tokenizer"
    assert config.checkpoint.load_training_state is False
    assert config.trainer.callbacks.local_simulation_resume is not None
    LazyConfig.save_yaml(config, str(tmp_path / f"{LOCAL_EXPERIMENT_NAMES[stage]}.yaml"))


@pytest.mark.parametrize("stage", range(4))
def test_local_toml_rejects_unset_bundle(monkeypatch: pytest.MonkeyPatch, stage: int) -> None:
    _compose(monkeypatch, stage)
    monkeypatch.delenv("COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE", raising=False)
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(InterpolationResolutionError, match="COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE"):
        load_experiment_from_toml(root / "examples/toml/sft_config" / f"{EXPERIMENT_NAMES[stage]}.toml")
