# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
    restore_model_config_from_public_model_config,
)


def _has_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_has_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_has_key(value, key) for value in obj)
    return False


def test_public_model_config_round_trip_removes_internal_metadata():
    model_config = {
        "_recursive_": False,
        "_target_": "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
        "config": {
            "_type": "cosmos_framework.configs.base.defaults.model_config.OmniMoTModelConfig",
            "activation_checkpointing": {
                "_type": "cosmos_framework.configs.base.defaults.activation_checkpointing.ActivationCheckpointingConfig",
                "mode": "full",
            },
            "compile": {
                "_target_": "cosmos_framework.configs.base.defaults.compile.CompileConfig",
                "enabled": False,
            },
            "tokenizer": {
                "_target_": "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface",
                "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            },
            "vlm_config": {
                "_type": "cosmos_framework.configs.base.defaults.reasoner.VLMConfig",
                "model_instance": {
                    "_target_": "cosmos_framework.model.generator.mot.unified_mot.Qwen3VLTextForCausalLM",
                    "config": {
                        "_target_": "cosmos_framework.configs.base.defaults.reasoner.create_vlm_config",
                        "base_config": {
                            "_target_": "cosmos_framework.model.generator.mot.unified_mot.Qwen3VLMoTConfig.from_json_file",
                            "json_file": "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json",
                        },
                    },
                },
            },
        },
    }

    public_model_config = build_public_model_config(model_config)
    text = json.dumps(public_model_config)

    assert not _has_key(public_model_config, "_target_")
    assert not _has_key(public_model_config, "class_name")
    assert not _has_key(public_model_config, "config_name")
    assert public_model_config["_target"] == "omni_mot_model"
    assert public_model_config["config"]["_type"] == "omni_mot_model_config"
    assert public_model_config["config"]["compile"]["_target"] == "compile_config"
    assert "projects.cosmos3" not in text
    assert "projects/cosmos3" not in text
    assert "cosmos3._src" not in text

    restored = restore_model_config_from_public_model_config(public_model_config)

    assert restored == model_config
    assert load_model_config_from_hf_config({"model": public_model_config}) == model_config
    assert load_model_config_from_hf_config({"model": model_config}) == model_config
