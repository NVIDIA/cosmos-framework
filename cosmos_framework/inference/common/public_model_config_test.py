# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from unittest import mock

from cosmos_framework.inference.common import public_model_config

_I4_MODULE_PREFIX = "projects." + "cosmos3." + "cosmos3."
_I4_FILE_PREFIX = "projects/" + "cosmos3/" + "cosmos3/"


def _i4_module(suffix: str) -> str:
    return _I4_MODULE_PREFIX + suffix


def _i4_file(suffix: str) -> str:
    return _I4_FILE_PREFIX + suffix


def _has_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_has_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_has_key(value, key) for value in obj)
    return False


def _qwen_model_config() -> dict:
    return {
        "_recursive_": False,
        "_target_": _i4_module("models.omni_mot_model.OmniMoTModel"),
        "config": {
            "_type": _i4_module("configs.base.defaults.model_config.OmniMoTModelConfig"),
            "activation_checkpointing": {
                "_type": _i4_module("configs.base.defaults.activation_checkpointing.ActivationCheckpointingConfig"),
                "mode": "full",
            },
            "compile": {
                "_target_": _i4_module("configs.base.defaults.compile.CompileConfig"),
                "enabled": False,
            },
            "tokenizer": {
                "_target_": _i4_module("tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface"),
                "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            },
            "vlm_config": {
                "_type": _i4_module("configs.base.defaults.reasoner.VLMConfig"),
                "model_instance": {
                    "_target_": _i4_module("models.mot.unified_mot.Qwen3VLTextForCausalLM"),
                    "config": {
                        "_target_": _i4_module("configs.base.defaults.reasoner.create_vlm_config"),
                        "base_config": {
                            "_target_": _i4_module("models.mot.unified_mot.Qwen3VLMoTConfig.from_json_file"),
                            "json_file": _i4_file("models/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"),
                        },
                    },
                },
            },
        },
    }


def _edge_model_config() -> dict:
    model_config = _qwen_model_config()
    vlm_config = model_config["config"]["vlm_config"]
    vlm_config["model_instance"] = {
        "_target_": _i4_module("models.mot.unified_mot.Nemotron3DenseVLTextForCausalLM"),
        "config": {
            "_target_": _i4_module("configs.base.defaults.reasoner.create_vlm_config"),
            "base_config": {
                "_target_": _i4_module("models.mot.unified_mot.Nemotron3DenseVLMoTConfig.from_json_file"),
                "json_file": _i4_file("models/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"),
            },
        },
    }
    vlm_config["tokenizer"] = {
        "_target_": _i4_module("processors.build_processor_lazy"),
        "tokenizer_type": "nvidia/Cosmos3-Edge-Reasoner",
    }
    vlm_config["pretrained_weights"] = {
        "_type": _i4_module("configs.base.defaults.reasoner.PretrainedWeightsConfig"),
        "checkpoint_format": "nemotron_3_dense_vl",
    }
    return model_config


def _rewrite_strings(obj, replacements: tuple[tuple[str, str], ...]):
    if isinstance(obj, dict):
        return {key: _rewrite_strings(value, replacements) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_strings(value, replacements) for value in obj]
    if not isinstance(obj, str):
        return obj
    for old, new in replacements:
        if obj.startswith(old):
            return new + obj[len(old) :]
    return obj


def _legacy_model_config(model_config: dict) -> dict:
    return _rewrite_strings(
        model_config,
        (
            (
                _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.",
                "projects.cosmos3.vfm.configs.base.defaults.vlm.",
            ),
            (_I4_MODULE_PREFIX + "models.reasoner.", "projects.cosmos3.vfm.models.vlm."),
            (_I4_MODULE_PREFIX, "projects.cosmos3.vfm."),
            (_I4_FILE_PREFIX + "models/reasoner/", "projects/cosmos3/vfm/models/vlm/"),
            (_I4_FILE_PREFIX, "projects/cosmos3/vfm/"),
        ),
    )


def _framework_model_config(model_config: dict) -> dict:
    return _rewrite_strings(
        model_config,
        (
            (
                _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.",
                "cosmos_framework.configs.base.defaults.reasoner.",
            ),
            (_I4_MODULE_PREFIX + "configs.base.", "cosmos_framework.configs.base."),
            (_I4_MODULE_PREFIX + "models.reasoner.", "cosmos_framework.model.generator.reasoner."),
            (_I4_MODULE_PREFIX + "models.", "cosmos_framework.model.generator."),
            (_I4_MODULE_PREFIX + "tokenizers.", "cosmos_framework.model.generator.tokenizers."),
            (_I4_MODULE_PREFIX + "processors.", "cosmos_framework.data.generator.processors."),
            (
                _I4_FILE_PREFIX + "models/reasoner/",
                "cosmos_framework/model/generator/reasoner/",
            ),
            (_I4_FILE_PREFIX + "models/", "cosmos_framework/model/generator/"),
        ),
    )


def _runtime_model_config(model_config: dict) -> dict:
    if public_model_config._module_exists("cosmos_framework.model.generator"):
        return _framework_model_config(model_config)
    return model_config


def _assert_public_config(public_config: dict) -> None:
    text = json.dumps(public_config)
    assert not _has_key(public_config, "_target_")
    assert not _has_key(public_config, "class_name")
    assert not _has_key(public_config, "config_name")
    assert public_config["_target"] == "omni_mot_model"
    assert public_config["config"]["_type"] == "omni_mot_model_config"
    assert public_config["config"]["compile"]["_target"] == "compile_config"
    assert "projects.cosmos3" not in text
    assert "projects/cosmos3" not in text
    assert "cosmos3._src" not in text
    assert "cosmos_framework" not in text
    assert "vfm" not in text
    json_file = public_config["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
    assert json_file.startswith("cosmos3://models/reasoner/")


def test_active_alias_registry_uses_current_i4_paths():
    paths = {
        *public_model_config._TARGET_PATHS_BY_ALIAS.values(),
        *public_model_config._TYPE_PATHS_BY_ALIAS.values(),
    }

    assert paths
    assert all(path.startswith(_I4_MODULE_PREFIX) for path in paths)
    assert all(".vfm." not in path for path in paths)


def test_qwen_public_model_config_round_trip():
    model_config = _runtime_model_config(_qwen_model_config())
    public_config = public_model_config.build_public_model_config(model_config)

    _assert_public_config(public_config)
    assert (
        public_config["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
        == "cosmos3://models/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
    )
    assert public_model_config.restore_model_config_from_public_model_config(public_config) == model_config
    assert public_model_config.load_model_config_from_hf_config({"model": public_config}) == model_config
    assert public_model_config.load_model_config_from_hf_config({"model": model_config}) == model_config


def test_legacy_model_config_produces_same_public_config():
    model_config = _qwen_model_config()
    legacy_config = _legacy_model_config(model_config)

    assert public_model_config.build_public_model_config(
        legacy_config
    ) == public_model_config.build_public_model_config(model_config)


def test_legacy_public_uri_restores_current_layout():
    public_config = public_model_config.build_public_model_config(_runtime_model_config(_qwen_model_config()))
    public_config["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = (
        "cosmos3://vfm/models/vlm/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
    )

    restored = public_model_config.restore_model_config_from_public_model_config(public_config)

    assert restored == _runtime_model_config(_qwen_model_config())


def test_legacy_alias_keys_do_not_consume_regular_metadata():
    regular_config = {
        "metadata": {
            "class_name": "custom_model_class",
            "config_name": "custom_training_config",
        }
    }

    assert not public_model_config.model_config_uses_public_aliases(regular_config)
    assert public_model_config.load_model_config_from_hf_config({"model": regular_config}) == regular_config

    public_config = {
        "_target": "omni_mot_model",
        "metadata": regular_config["metadata"],
    }
    restored = public_model_config.restore_model_config_from_public_model_config(public_config)

    assert restored["metadata"] == regular_config["metadata"]
    assert restored["_target_"] == _runtime_model_config(_qwen_model_config())["_target_"]


def test_legacy_alias_keys_still_restore_registered_aliases():
    legacy_public_config = {
        "class_name": "omni_mot_model",
        "config": {"config_name": "omni_mot_model_config"},
    }

    assert public_model_config.model_config_uses_public_aliases(legacy_public_config)
    restored = public_model_config.restore_model_config_from_public_model_config(legacy_public_config)

    runtime_config = _runtime_model_config(_qwen_model_config())
    assert restored["_target_"] == runtime_config["_target_"]
    assert restored["config"]["_type"] == runtime_config["config"]["_type"]


def test_edge_public_model_config_round_trip():
    model_config = _runtime_model_config(_edge_model_config())
    public_config = public_model_config.build_public_model_config(model_config)

    _assert_public_config(public_config)
    model_instance = public_config["config"]["vlm_config"]["model_instance"]
    assert model_instance["_target"] == "nemotron3_dense_vl_text_for_causal_lm"
    assert model_instance["config"]["base_config"]["_target"] == "nemotron3_dense_vl_mot_config_from_json_file"
    assert public_model_config.restore_model_config_from_public_model_config(public_config) == model_config


def test_edge_public_model_config_restores_framework_layout():
    i4_config = _edge_model_config()
    framework_config = _framework_model_config(i4_config)
    public_config = public_model_config.build_public_model_config(framework_config)

    with mock.patch.object(
        public_model_config,
        "_module_exists",
        side_effect=lambda module: module == "cosmos_framework.model.generator",
    ):
        restored = public_model_config.restore_model_config_from_public_model_config(public_config)

    assert restored == framework_config
