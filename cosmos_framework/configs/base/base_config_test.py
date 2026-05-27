# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
This file is used to test the config of the cosmos3 vfm project.
It is used to verify the config is loadable.

To run the test, you can use the following command:
pytest -s cosmos_framework/configs/base/base_config_test.py
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from cosmos_framework.utils.config_helper import get_config_module, override


@pytest.mark.L0
@pytest.mark.parametrize(
    "experiment_name",
    [
        "t2i_mot_exp001_009_qwen3_vl_2b_256res_frozen_llm",
    ],
)
def test_config_init_experiment_mot(experiment_name):
    """
    Parameterized test to verify config initialization for multiple experiments.
    PYTHONPATH=. torchrun --nproc_per_node=8 -m pytest -s cosmos_framework/configs/base/config_test_mot.py --L1
    """
    config_file = "configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            f"experiment={experiment_name}",
        ],
    )


def _make_self_mock(*, pretrained_enabled: bool, load_weights_from_pretrained: bool) -> MagicMock:
    """Mock the OmniMoTModel attributes that load_pretrained_model_if_needed reads."""
    self_mock = MagicMock()
    self_mock.vlm_config.pretrained_weights.enabled = pretrained_enabled
    self_mock.config.diffusion_expert_config.load_weights_from_pretrained = load_weights_from_pretrained
    self_mock.config.ema.enabled = False
    return self_mock


@pytest.mark.L0
class TestLoadPretrainedGate:
    """Decision matrix for ``OmniMoTModel.load_pretrained_model_if_needed``.

    Replaces the previous ``OmniMoTModelConfig.validate`` tests now that
    LoadPretrained callback probes ``latest_checkpoint.txt`` / ``load_path`` at
    ``on_train_start`` and forwards the two booleans, instead of mutating the
    config during validation.
    """

    _LOADER_TARGET = "cosmos_framework.model.vfm.omni_mot_model.load_language_model_safetensors"

    def _call(self, self_mock: MagicMock, *, has_resumable_checkpoint: bool, has_load_path: bool) -> MagicMock:
        from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel

        with patch(self._LOADER_TARGET) as loader:
            OmniMoTModel.load_pretrained_model_if_needed(
                self_mock,
                has_resumable_checkpoint=has_resumable_checkpoint,
                has_load_path=has_load_path,
            )
        return loader

    def test_fresh_init_loads_and_copies(self):
        """No checkpoint, no load_path: HF load AND understanding→generation copy."""
        self_mock = _make_self_mock(pretrained_enabled=True, load_weights_from_pretrained=True)
        loader = self._call(self_mock, has_resumable_checkpoint=False, has_load_path=False)
        loader.assert_called_once()
        self_mock.net.language_model.init_moe.assert_called_once()

    def test_resume_skips_everything(self):
        """Resumable checkpoint exists: neither HF load nor copy."""
        self_mock = _make_self_mock(pretrained_enabled=True, load_weights_from_pretrained=True)
        loader = self._call(self_mock, has_resumable_checkpoint=True, has_load_path=False)
        loader.assert_not_called()
        self_mock.net.language_model.init_moe.assert_not_called()

    def test_warm_start_loads_but_skips_copy(self):
        """load_path set, no checkpoint: HF load but skip understanding→generation copy."""
        self_mock = _make_self_mock(pretrained_enabled=True, load_weights_from_pretrained=True)
        loader = self._call(self_mock, has_resumable_checkpoint=False, has_load_path=True)
        loader.assert_called_once()
        self_mock.net.language_model.init_moe.assert_not_called()

    def test_pretrained_disabled_short_circuits(self):
        """pretrained_weights.enabled=False: early return regardless of other flags."""
        self_mock = _make_self_mock(pretrained_enabled=False, load_weights_from_pretrained=True)
        loader = self._call(self_mock, has_resumable_checkpoint=False, has_load_path=False)
        loader.assert_not_called()
        self_mock.net.language_model.init_moe.assert_not_called()
