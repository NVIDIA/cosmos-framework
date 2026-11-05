# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    load_model_config_from_hf_config,
    restore_model_config_from_public_model_config,
)
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import LidarTokenizerV1Interface
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string, locate


def test_lidar_tokenizer_saved_target_round_trip() -> None:
    # Exercise the shortened target actually written to a training config,
    # rather than only a hand-authored defining-module path.
    target = convert_target_to_string(LidarTokenizerV1Interface)
    public = build_public_model_config({"_target_": target})
    assert public == {"_target": "lidar_tokenizer_v1_interface"}
    restored = restore_model_config_from_public_model_config(public)
    assert locate(restored["_target_"]) is LidarTokenizerV1Interface


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
            "fixed_step_sampler_config": {
                "_type": "cosmos_framework.configs.base.defaults.model_config.FixedStepSamplerConfig",
                "sample_type": "sde",
                "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
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
    assert public_model_config["config"]["fixed_step_sampler_config"]["_type"] == "fixed_step_sampler_config"
    assert "projects.cosmos3" not in text
    assert "projects/cosmos3" not in text
    assert "cosmos3._src" not in text

    restored = restore_model_config_from_public_model_config(public_model_config)

    assert restored == model_config
    assert load_model_config_from_hf_config({"model": public_model_config}) == model_config
    assert load_model_config_from_hf_config({"model": model_config}) == model_config


def test_public_aliases_normalize_serialized_inherited_targets() -> None:
    model_config = {
        "_target_": "cosmos_framework.model.generator.omni_mot_causal_model.OmniMoTCausalModel",
        "config": {
            "_type": "cosmos_framework.model.generator.omni_mot_causal_model.OmniMoTCausalModelConfig",
            "lbl": {
                "_type": "cosmos_framework.model.generator.utils.load_balancing_stats.LBLConfig",
            },
            "vlm_config": {
                "model_instance": {
                    "config": {
                        "base_config": {
                            "_target_": (
                                "cosmos_framework.model.generator.mot.unified_mot._MoTConfigBase.from_json_file"
                            )
                        }
                    }
                }
            },
        },
    }

    public = build_public_model_config(model_config)
    assert public["config"]["lbl"]["_type"] == "lbl_config"
    assert public["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["_target"] == (
        "qwen3_vl_mot_config_from_json_file"
    )

    restored = restore_model_config_from_public_model_config(public)
    assert restored["config"]["lbl"]["_type"] == (
        "cosmos_framework.model.generator.utils.load_balancing_stats.LBLConfig"
    )
    assert restored["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["_target_"] == (
        "cosmos_framework.model.generator.mot.unified_mot.Qwen3VLMoTConfig.from_json_file"
    )


def test_interactive_attention_configs_round_trip():
    """Causal artifacts carry three interactive-only attention configs.

    Without aliases for them, loading an exported causal checkpoint fails with
    "Unknown Cosmos type alias" -- the upstream exporter writes the alias, and
    this table is what has to turn it back into an importable path.

    causal_flex_attention.py is the released name of the upstream codebase's
    interactive/configs/defaults/flex_attention.py, renamed so it does not collide
    with the non-causal multiview attention config, which is asserted here too so an
    over-broad rule for either cannot capture the other.
    """
    model_config = {
        "_recursive_": False,
        "_target_": "cosmos_framework.model.generator.omni_mot_causal_model.OmniMoTCausalModel",
        "config": {
            "_type": "cosmos_framework.model.generator.omni_mot_causal_model.OmniMoTCausalModelConfig",
            "multiview_attention": {
                "_type": "cosmos_framework.configs.base.defaults.multiview_attention.MultiviewAttentionConfig",
                "mask": {
                    "_type": "cosmos_framework.configs.base.defaults.multiview_attention.MultiviewAttentionMaskConfig",
                },
            },
            "causal_flex_attention": {
                "_type": "cosmos_framework.configs.base.defaults.causal_flex_attention.CausalFlexAttentionConfig",
                "mask": {
                    "_type": "cosmos_framework.configs.base.defaults.causal_flex_attention.CausalFlexAttentionMaskConfig",
                },
            },
            "teacher_forcing_replay_policy": {
                "_type": "cosmos_framework.configs.base.defaults.replay_attention.TeacherForcingReplayPolicyConfig",
                "mode": "none",
            },
        },
    }

    public_model_config = build_public_model_config(model_config)
    config = public_model_config["config"]

    assert config["causal_flex_attention"]["_type"] == "causal_flex_attention_config"
    assert config["causal_flex_attention"]["mask"]["_type"] == "causal_flex_attention_mask_config"
    assert config["teacher_forcing_replay_policy"]["_type"] == "teacher_forcing_replay_policy_config"
    # The non-causal siblings keep their own aliases. The alias name is still
    # flex_attention_config for compatibility with artifacts already exported, but it now
    # names MultiviewAttentionConfig -- which is what the upstream codebase emits it for.
    assert config["multiview_attention"]["_type"] == "flex_attention_config"
    assert config["multiview_attention"]["mask"]["_type"] == "flex_attention_mask_config"
    assert "projects.cosmos3" not in json.dumps(public_model_config)

    assert restore_model_config_from_public_model_config(public_model_config) == model_config


@pytest.mark.parametrize(("public_key", "runtime_key"), [("_type", "_type"), ("_target", "_target_")])
@pytest.mark.parametrize(
    ("alias", "class_name", "fields"),
    [
        (
            "cross_view_temporal_window_config",
            "CrossViewTemporalWindowConfig",
            {"bounds": [-0.4, 0.0], "includes_first_frame": False},
        ),
        (
            "same_view_temporal_window_config",
            "SameViewTemporalWindowConfig",
            {
                "sensor_to_sensor_bounds": None,
                "sensor_to_control_bounds": [-0.4, 0.0],
                "control_to_control_bounds": [-0.2, 0.2],
                "control_to_sensor_bounds": None,
            },
        ),
    ],
)
def test_temporal_window_public_config_round_trip(
    public_key: str, runtime_key: str, alias: str, class_name: str, fields: dict[str, object]
) -> None:
    """Published Auto window aliases must load and re-export without changing their bounds."""
    public_config = {"window": {public_key: alias, **fields}}
    runtime_config = {
        "window": {
            runtime_key: f"cosmos_framework.configs.base.defaults.multiview_attention.{class_name}",
            **fields,
        }
    }

    restored = load_model_config_from_hf_config({"model": public_config})

    assert restored == runtime_config
    assert build_public_model_config(restored) == public_config
