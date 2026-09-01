# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy

import attrs
import pytest

from cosmos_framework.inference.common import distillation_export
from cosmos_framework.inference.common.distillation_export import (
    build_student_checkpoint_metadata,
    sanitize_student_model_config,
)



def test_sanitize_student_model_config_removes_distillation_state() -> None:
    fixed_step_sampler_config = {
        "sample_type": "ode",
        "t_list": [1.0, 0.75, 0.5, 0.25],
    }
    model_dict = {
        "_target_": "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
        "_recursive_": False,
        "config": {
            "_type": "cosmos_framework.configs.base.experiment.distillation.dmd2_config.DMD2RFConfig",
            "_metadata": {
                "object_type": ("cosmos_framework.configs.base.experiment.distillation.dmd2_config.DMD2RFConfig"),
            },
            "ema": {"enabled": False},
            "compile": {"enabled": True, "compiled_region": "language"},
            "fixed_step_sampler_config": fixed_step_sampler_config,
            "vlm_config": {"model_name": "student"},
            "vlm_config_teacher": {"model_name": "teacher"},
            "vlm_config_fake_score": {"model_name": "fake_score"},
            "load_teacher_weights": True,
            "teacher_load_from": {"load_path": "internal-teacher"},
            "student_load_from": {"load_path": "internal-student"},
            "optimizer": {"net": {}, "fake_score": {}},
        },
    }

    sanitize_student_model_config(
        model_dict,
        base_model_target="cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
        base_config_type="cosmos_framework.configs.base.defaults.model_config.OmniMoTModelConfig",
        base_config_field_names={"compile", "ema", "fixed_step_sampler_config", "vlm_config"},
    )

    assert model_dict == {
        "_target_": "cosmos_framework.model.generator.omni_mot_model.OmniMoTModel",
        "_recursive_": False,
        "config": {
            "_type": "cosmos_framework.configs.base.defaults.model_config.OmniMoTModelConfig",
            "ema": {"enabled": False},
            "compile": {"enabled": False, "compiled_region": "language"},
            "fixed_step_sampler_config": fixed_step_sampler_config,
            "vlm_config": {"model_name": "student"},
        },
    }


def test_build_student_checkpoint_metadata_omits_source_paths() -> None:
    assert build_student_checkpoint_metadata(use_ema_weights=True) == {
        "checkpoint_type": "hf",
        "source_weights": "ema",
        "student_only": True,
    }
    assert build_student_checkpoint_metadata(use_ema_weights=False) == {
        "checkpoint_type": "hf",
        "source_weights": "regular",
        "student_only": True,
    }


def test_sanitize_student_public_model_config_removes_internal_loaders() -> None:
    model_dict = {
        "config": {
            "tokenizer": {
                "bucket_name": "internal-checkpoint-bucket",
                "object_store_credential_path_pretrained": "/path/to/source.secret",
                "vae_path": "/lustre/training/checkpoints/wan22_vae/Wan2.2_VAE.pth",
            },
            "sound_tokenizer": {
                "bucket_name": "internal-checkpoint-bucket",
                "object_store_credential_path_pretrained": "/path/to/source.secret",
                "avae_path": "pretrained/tokenizers/audio/avae/avae.ckpt",
            },
            "vlm_config": {
                "pretrained_weights": {
                    "enabled": True,
                    "backbone_path": "s3://internal-checkpoint-bucket/reasoner",
                    "credentials_path": "/path/to/source.secret",
                    "enable_gcs_patch_in_boto3": True,
                },
                "tokenizer": {
                    "_target_": (
                        "cosmos_framework.data.generator.sequence_packing.configs.distillation_implementation."
                        "_create_oss_tokenizer_with_internal_download"
                    ),
                    "config_variant": "gcp",
                    "pretrained_model_name": "Qwen/Qwen3-VL-32B-Instruct",
                },
                "model_instance": {
                    "config": {
                        "base_config": {
                            "json_file": (
                                "/checkout/cosmos-framework/cosmos_framework/model/generator/"
                                "reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
                            )
                        }
                    }
                },
            },
        }
    }

    distillation_export.sanitize_student_public_model_config(
        model_dict,
        public_vlm_tokenizer_target=(
            "cosmos_framework.configs.base.defaults.reasoner.create_qwen2_tokenizer_with_download"
        ),
    )

    assert model_dict == {
        "config": {
            "tokenizer": {
                "bucket_name": "bucket",
                "object_store_credential_path_pretrained": "",
                "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            },
            "sound_tokenizer": {
                "bucket_name": "bucket",
                "object_store_credential_path_pretrained": "",
                "avae_path": "pretrained/tokenizers/audio/avae/avae.ckpt",
            },
            "vlm_config": {
                "pretrained_weights": {
                    "enabled": False,
                    "backbone_path": "",
                    "credentials_path": "",
                    "enable_gcs_patch_in_boto3": False,
                },
                "tokenizer": {
                    "_target_": (
                        "cosmos_framework.configs.base.defaults.reasoner.create_qwen2_tokenizer_with_download"
                    ),
                    "config_variant": "hf",
                    "pretrained_model_name": "Qwen/Qwen3-VL-32B-Instruct",
                },
                "model_instance": {
                    "config": {
                        "base_config": {
                            "json_file": (
                                "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
                            )
                        }
                    }
                },
            },
        }
    }


@pytest.mark.parametrize("model_size", ("2B", "4B", "8B", "32B"))
def test_sanitize_student_public_model_config_preserves_qwen_variant(model_size: str) -> None:
    filename = f"Qwen3-VL-{model_size}-Instruct.json"
    model_dict = {
        "config": {
            "vlm_config": {
                "model_instance": {
                    "config": {"base_config": {"json_file": f"/checkout/private/qwen3_vl/configs/{filename}"}}
                }
            }
        }
    }

    distillation_export.sanitize_student_public_model_config(model_dict)

    assert (
        model_dict["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
        == f"cosmos_framework/model/generator/reasoner/qwen3_vl/configs/{filename}"
    )


def test_sanitize_student_public_model_config_is_idempotent() -> None:
    model_dict = {
        "config": {
            "tokenizer": {
                "bucket_name": "bucket",
                "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
            },
            "vlm_config": {
                "model_instance": {
                    "config": {
                        "base_config": {
                            "json_file": (
                                "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
                            )
                        }
                    }
                }
            },
        }
    }

    distillation_export.sanitize_student_public_model_config(model_dict)
    first_result = copy.deepcopy(model_dict)
    distillation_export.sanitize_student_public_model_config(model_dict)

    assert model_dict == first_result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vae_path", "/custom/tokenizers/CustomVAE.pth"),
        ("json_file", "/custom/reasoners/CustomReasoner.json"),
    ],
)
def test_sanitize_student_public_model_config_warns_for_unknown_absolute_paths(
    field: str,
    value: str,
) -> None:
    model_dict = {
        "config": {
            "tokenizer": {"vae_path": "relative/custom-vae.pth"},
            "vlm_config": {
                "model_instance": {"config": {"base_config": {"json_file": "relative/custom-reasoner.json"}}}
            },
        }
    }
    if field == "vae_path":
        model_dict["config"]["tokenizer"]["vae_path"] = value
    else:
        model_dict["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = value

    with pytest.warns(UserWarning, match="may not be portable") as warning_records:
        distillation_export.sanitize_student_public_model_config(model_dict)

    assert warning_records[0].filename == __file__
    if field == "vae_path":
        assert model_dict["config"]["tokenizer"]["vae_path"] == value
    else:
        assert model_dict["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] == value


def test_sanitize_student_public_model_config_preserves_unknown_relative_paths() -> None:
    model_dict = {
        "config": {
            "tokenizer": {"vae_path": "relative/custom-vae.pth"},
            "vlm_config": {
                "model_instance": {"config": {"base_config": {"json_file": "relative/custom-reasoner.json"}}}
            },
        }
    }

    distillation_export.sanitize_student_public_model_config(model_dict)

    assert model_dict["config"]["tokenizer"]["vae_path"] == "relative/custom-vae.pth"
    assert (
        model_dict["config"]["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
        == "relative/custom-reasoner.json"
    )


@pytest.mark.parametrize(
    ("model_dict", "field_name"),
    [
        ({"config": {"tokenizer": {"vae_path": None}}}, "tokenizer.vae_path"),
        (
            {"config": {"vlm_config": {"model_instance": {"config": {"base_config": {"json_file": None}}}}}},
            "vlm_config.model_instance.config.base_config.json_file",
        ),
    ],
)
def test_sanitize_student_public_model_config_rejects_non_string_dependency_path(
    model_dict: dict,
    field_name: str,
) -> None:
    with pytest.raises(TypeError) as error:
        distillation_export.sanitize_student_public_model_config(model_dict)

    assert str(error.value) == f"Expected {field_name} to be a string."


def test_resolve_vision_checkpoint_path_prefers_local_override() -> None:
    fallback_called = False

    def download_checkpoint(_configured_uri: str) -> str:
        nonlocal fallback_called
        fallback_called = True
        return "/downloaded/vision"

    path = distillation_export.resolve_vision_checkpoint_path(
        local_path="/local/vision",
        configured_uri="s3://internal/vision",
        download_checkpoint=download_checkpoint,
    )

    assert path == "/local/vision"
    assert fallback_called is False


def test_resolve_student_base_model_keeps_default_for_bidirectional_config() -> None:
    class _DefaultBaseModel:
        pass

    model_dict = {"config": {"video_temporal_causal": False}}

    model_cls, config_cls = distillation_export.resolve_student_base_model(
        model_dict, default_base_model=_DefaultBaseModel
    )

    assert model_cls is _DefaultBaseModel
    assert config_cls.__name__ == "OmniMoTModelConfig"


def test_resolve_student_base_model_selects_causal_base_for_causal_config() -> None:
    class _DefaultBaseModel:
        pass

    model_dict = {"config": {"video_temporal_causal": True, "teacher_forcing_frames_per_chunk": 4}}

    model_cls, config_cls = distillation_export.resolve_student_base_model(
        model_dict, default_base_model=_DefaultBaseModel
    )

    assert model_cls.__name__ == "OmniMoTCausalModel"
    assert config_cls.__name__ == "OmniMoTCausalModelConfig"
    # The causal-only field must survive the projection field filter.
    assert "teacher_forcing_frames_per_chunk" in {field.name for field in attrs.fields(config_cls)}


def test_resolve_student_base_model_tolerates_missing_config() -> None:
    class _DefaultBaseModel:
        pass

    model_cls, _ = distillation_export.resolve_student_base_model({}, default_base_model=_DefaultBaseModel)

    assert model_cls is _DefaultBaseModel
