# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import attrs
import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.inference.bimanual import build_bimanual_action_metadata
from cosmos_framework.scripts._export_model_helpers import bind_bundled_lidar_vae
from cosmos_framework.scripts.convert_model_to_diffusers import (
    _build_public_export_model_config,
    _carry_export_metadata,
    _copy_bundled_lidar_vae,
    _refuse_existing_weights,
    _write_bimanual_transformer_metadata,
)


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


def test_bimanual_action_metadata_is_written_to_transformer_config(tmp_path) -> None:
    source_config = {"model": {}, **build_bimanual_action_metadata()}
    transformer_config_path = tmp_path / "transformer/config.json"
    transformer_config_path.parent.mkdir()
    transformer_config_path.write_text(json.dumps({"model_type": "cosmos3_omni"}))

    _write_bimanual_transformer_metadata(source_config, tmp_path)

    written = json.loads(transformer_config_path.read_text())
    assert written["cosmos3_nano_sim_bimanual"] == source_config["cosmos3_nano_sim_bimanual"]


@pytest.mark.parametrize(
    "relative", ["model.safetensors", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"]
)
def test_refuses_an_output_dir_that_already_holds_weights(tmp_path, relative) -> None:
    weights = tmp_path / relative
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="use a new output directory"):
        _refuse_existing_weights(tmp_path)


def test_accepts_a_new_or_weightless_output_dir(tmp_path) -> None:
    _refuse_existing_weights(tmp_path / "does-not-exist")
    (tmp_path / "config.json").write_text("{}")
    _refuse_existing_weights(tmp_path)


def test_carries_the_per_view_caption_layout_into_the_diffusers_config() -> None:
    source_config = {
        "model": {},
        "per_view_captions": True,
        "per_view_caption_task": "av_joint_camera_lidar_transfer",
        "unrelated": 1,
    }
    config_dict = {"model_type": "cosmos3_omni"}
    _carry_export_metadata(source_config, config_dict)
    assert config_dict == {
        "model_type": "cosmos3_omni",
        "per_view_captions": True,
        "per_view_caption_task": "av_joint_camera_lidar_transfer",
    }


def test_carries_nothing_for_a_checkpoint_exported_without_the_layout() -> None:
    config_dict = {}
    _carry_export_metadata({"model": {}}, config_dict)
    assert config_dict == {}


def test_copies_and_binds_the_bundled_lidar_vae(tmp_path) -> None:
    source = tmp_path / "hf"
    (source / "lidar_vae").mkdir(parents=True)
    (source / "lidar_vae/diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (source / "lidar_vae/config.json").write_text("{}")
    output = tmp_path / "diffusers"
    output.mkdir()

    assert _copy_bundled_lidar_vae(source, output) is True
    assert (output / "lidar_vae/diffusion_pytorch_model.safetensors").read_bytes() == b"vae"
    assert (output / "lidar_vae/config.json").is_file()

    config_dict = {"model": {"config": {"lidar_tokenizer": {"vae_path": "s3://bucket/x.pt", "bucket_name": "bucket"}}}}
    assert bind_bundled_lidar_vae(config_dict, output) is True
    assert (
        config_dict["model"]["config"]["lidar_tokenizer"]["vae_path"] == "lidar_vae/diffusion_pytorch_model.safetensors"
    )


def test_no_lidar_vae_to_copy(tmp_path) -> None:
    assert _copy_bundled_lidar_vae(tmp_path, tmp_path / "out") is False


@pytest.mark.parametrize("prefix", ["model.visual.", "model.net.language_model.visual.", "net.language_model.visual."])
def test_vision_encoder_checkpoint_prefixes(tmp_path, prefix) -> None:
    converter = pytest.importorskip("cosmos_framework.scripts._convert_model_to_diffusers")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"{prefix}patch_embed.proj.weight": "model.safetensors", "lm_head.weight": "x"}})
    )
    assert converter._checkpoint_vision_weight_prefix(tmp_path) == prefix


def test_vision_encoder_checkpoint_prefix_absent(tmp_path) -> None:
    converter = pytest.importorskip("cosmos_framework.scripts._convert_model_to_diffusers")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "x"}}))
    assert converter._checkpoint_vision_weight_prefix(tmp_path) is None
