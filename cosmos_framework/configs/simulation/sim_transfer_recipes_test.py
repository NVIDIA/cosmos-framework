# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compose the Sim-Transfer training recipes the way ``scripts.train_experiment`` does.

Every stage composes from its required inputs, and an unset input stops the job before a model is built.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError, MissingMandatoryValue

from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import LazyConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_INPUTS = {
    "COSMOS_SIM_TRANSFER_STAGE0_CHECKPOINT": "/ckpt/stage0",
    "COSMOS_SIM_TRANSFER_STAGE1_CHECKPOINT": "/ckpt/stage1",
    "COSMOS_SIM_TRANSFER_STAGE2_CHECKPOINT": "/ckpt/stage2",
    "COSMOS_SIM_TRANSFER_LENGTH_INDEX_MANIFEST": "s3://bucket/manifest.json",
    "COSMOS_SIM_TRANSFER_LOCAL_BUNDLE": "/data/bundle",
}
_STAGE0, _STAGE1, _STAGE2, _MANIFEST, _BUNDLE = _INPUTS
_STUDENT = "model.config.student_load_from.load_path"
_TEACHER = "model.config.teacher_load_from.load_path"
# experiment -> (config root, required inputs, composed key -> expected value)
_RECIPES = {
    "sim_transfer_stage1_tf": ("config.py", [_STAGE0], {"checkpoint.load_path": "/ckpt/stage0"}),
    "sim_transfer_stage2_tfdcm": (
        "config.py",
        [_STAGE1],
        {
            "checkpoint.load_path": "/ckpt/stage1",
            "model.config.teacher_forcing_dcm_teacher_load_from.load_path": "/ckpt/stage1/model",
        },
    ),
    "sim_transfer_stage3_sfdmd": (
        "config.py",
        [_STAGE0, _STAGE2, _MANIFEST],
        {
            _STUDENT: "/ckpt/stage2/model",
            _TEACHER: "/ckpt/stage0/model",
            "dataloader_train.dataset.manifest_path": "s3://bucket/manifest.json",
        },
    ),
    "sim_local_transfer": (
        "local.py",
        [_STAGE0, _STAGE2, _BUNDLE],
        {
            _STUDENT: "/ckpt/stage2/model",
            _TEACHER: "/ckpt/stage0/model",
            "dataloader_train.dataloaders.local_data.dataloader.dataset.bundle_path": "/data/bundle",
        },
    ),
}


def _compose(monkeypatch: pytest.MonkeyPatch, experiment: str, unset: str | None = None):
    root, inputs, _ = _RECIPES[experiment]
    # Set only the inputs this recipe declares, so a recipe that needs more fails here.
    for name, value in _INPUTS.items():
        if name in inputs and name != unset:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    return load_config(f"cosmos_framework/configs/simulation/{root}", ["--", f"experiment={experiment}"])


@pytest.mark.parametrize("experiment", _RECIPES)
def test_recipe_composes_from_its_inputs(monkeypatch: pytest.MonkeyPatch, experiment: str) -> None:
    config = _compose(monkeypatch, experiment)
    for key, expected in _RECIPES[experiment][2].items():
        value = config
        for part in key.split("."):
            value = getattr(value, part)
        assert value == expected, key


@pytest.mark.parametrize(
    "experiment",
    [
        "sim_local_transfer_stage0_bidir",
        "sim_local_transfer_stage1_tf",
        "sim_local_transfer_stage2_tfdcm",
        "sim_local_transfer",
    ],
)
def test_local_backbones_use_stage_checkpoints_without_s3_defaults(
    monkeypatch: pytest.MonkeyPatch, experiment: str
) -> None:
    for name, value in _INPUTS.items():
        if name == _MANIFEST:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("COSMOS_SIM_TRANSFER_BASE_CHECKPOINT", "/ckpt/base")
    config = load_config("cosmos_framework/configs/simulation/local.py", ["--", f"experiment={experiment}"])
    model = config.model.config
    expected_backbones = {"vlm_config"}
    if experiment == "sim_local_transfer":
        expected_backbones.update(("vlm_config_teacher", "vlm_config_fake_score"))
    backbones = {key for key in ("vlm_config", "vlm_config_teacher", "vlm_config_fake_score") if key in model}
    assert backbones == expected_backbones
    for key in backbones:
        weights = model[key].pretrained_weights
        assert weights.enabled is False
        assert weights.backbone_path == ""
        assert weights.credentials_path == ""
        assert weights.enable_gcs_patch_in_boto3 is False
    assert "s3://" not in OmegaConf.to_yaml(config.model)
    assert config.checkpoint.load_from_object_store.enabled is False
    assert config.checkpoint.save_to_object_store.enabled is False


@pytest.mark.parametrize(
    ("experiment", "unset"), [(experiment, name) for experiment, (_, inputs, _) in _RECIPES.items() for name in inputs]
)
def test_unset_input_fails_at_compose(monkeypatch: pytest.MonkeyPatch, experiment: str, unset: str) -> None:
    with pytest.raises(InterpolationResolutionError, match=unset):
        _compose(monkeypatch, experiment, unset=unset)


@pytest.mark.parametrize("experiment", ["sim_transfer_stage1_tf", "sim_transfer_stage2_tfdcm"])
def test_trainer_refuses_a_recipe_without_training_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, experiment: str
) -> None:
    # The trainer saves its config on rank 0 before it builds the model.
    with pytest.raises(MissingMandatoryValue, match="dataloader_train._target_"):
        LazyConfig.save_yaml(_compose(monkeypatch, experiment), str(tmp_path / "config.yaml"))
