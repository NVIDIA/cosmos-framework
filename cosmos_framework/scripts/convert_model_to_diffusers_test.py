# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import attrs
import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.scripts.convert_model_to_diffusers import _build_public_export_model_config


def _model_dict(quantization: dict | None) -> dict:
    model = {"config": {}, "_target_": "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel"}
    if quantization is not None:
        model["config"]["quantization"] = quantization
    return model


def _default_quantization_dict(**overrides) -> dict:
    values = attrs.asdict(QuantizationConfig(**overrides))
    values["_target_"] = "cosmos_framework.configs.base.defaults.quantization.QuantizationConfig"
    return values


def test_export_accepts_default_quantization_config() -> None:
    # A checkpoint saved with every QuantizationConfig field at its default —
    # including the mixed-precision fields — must remain exportable.
    _build_public_export_model_config(_model_dict(_default_quantization_dict()))


def test_export_accepts_legacy_quantization_dict_without_new_fields() -> None:
    legacy = {
        "_target_": "cosmos_framework.configs.base.defaults.quantization.QuantizationConfig",
        "exclude_regex": [],
        "include_regex": [],
        "method": None,
        "fp8_granularity": "per_row",
        "modelopt_fp8_checkpoint_path": None,
        "modelopt_fp8_target_fqns": [],
    }
    _build_public_export_model_config(_model_dict(legacy))


def test_export_accepts_missing_quantization_section() -> None:
    _build_public_export_model_config(_model_dict(None))


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "fp8"},
        {"mixed_precision_first_steps": 3},
        {"mixed_precision_last_steps": 3},
    ],
)
def test_export_rejects_enabled_quantization(overrides: dict) -> None:
    with pytest.raises(ValueError, match="Cannot export"):
        _build_public_export_model_config(_model_dict(_default_quantization_dict(**overrides)))


def test_export_rejects_unknown_quantization_field() -> None:
    quantization = _default_quantization_dict()
    quantization["future_field"] = True
    with pytest.raises(ValueError, match="Cannot export"):
        _build_public_export_model_config(_model_dict(quantization))
