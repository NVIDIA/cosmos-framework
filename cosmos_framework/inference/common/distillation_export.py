# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Portable student-only checkpoint export helpers."""

import warnings
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Any

_PUBLIC_WAN_VAE_PATHS = {
    "Wan2.2_VAE.pth": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
}
_PUBLIC_QWEN3_VL_CONFIG_PATHS = {
    filename: f"cosmos_framework/model/generator/reasoner/qwen3_vl/configs/{filename}"
    for filename in (
        "Qwen3-VL-2B-Instruct.json",
        "Qwen3-VL-4B-Instruct.json",
        "Qwen3-VL-8B-Instruct.json",
        "Qwen3-VL-32B-Instruct.json",
    )
}


def _normalize_public_dependency_path(
    value: Any,
    *,
    field_name: str,
    public_paths: dict[str, str],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected {field_name} to be a string.")
    filename = PurePath(value).name
    if filename in public_paths:
        return public_paths[filename]
    if Path(value).is_absolute():
        warnings.warn(
            f"{field_name} contains an unrecognized absolute path and the exported artifact may not be portable: "
            f"{value}",
            UserWarning,
            stacklevel=3,
        )
    return value


def resolve_student_base_model(
    model_dict: dict[str, Any],
    *,
    default_base_model: type,
) -> tuple[type, type]:
    """Return the ``(model, config)`` classes a student config should project onto.

    Temporally-causal students (Self-Forcing / teacher-forcing recipes) must
    project onto the causal base. ``video_temporal_causal`` lives on the base
    config, but ``teacher_forcing_frames_per_chunk`` and the AR cache settings
    are declared by ``OmniMoTCausalModelConfig``, so projecting them onto the
    bidirectional base would silently drop those fields and name a model class
    that cannot run the autoregressive decode loop.

    The lookup lives here rather than in ``cosmos3.scripts.export_model`` because
    that package must not import the internal interactive tree. The causal
    classes are imported lazily so that exporting a non-causal student never
    pulls in the autoregressive stack.
    """
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig

    config = model_dict.get("config")
    is_causal = isinstance(config, dict) and bool(config.get("video_temporal_causal", False))
    if not is_causal:
        return default_base_model, OmniMoTModelConfig

    from cosmos_framework.model.generator.omni_mot_causal_model import (
        OmniMoTCausalModel,
        OmniMoTCausalModelConfig,
    )

    return OmniMoTCausalModel, OmniMoTCausalModelConfig


def build_student_checkpoint_metadata(*, use_ema_weights: bool) -> dict[str, str | bool]:
    """Build portable metadata without source checkpoint or credential paths."""
    return {
        "checkpoint_type": "hf",
        "source_weights": "ema" if use_ema_weights else "regular",
        "student_only": True,
    }


def _migrate_legacy_transfer_replay_config(config: dict[str, Any], *, base_config_field_names: set[str]) -> None:
    """Preserve pre-replay-policy Transfer connectivity when projecting a student."""
    legacy_key = "transfer_control_attention_mode"
    if legacy_key not in config:
        return

    required_fields = {"teacher_forcing_replay_policy", "teacher_forcing_kv_implementation"}
    if not required_fields <= base_config_field_names:
        raise ValueError("Legacy Transfer attention requires a causal base config with teacher-forcing replay support.")

    legacy_modes = {
        "global_control": ("global", False),
        "causal_control": ("causal", False),
        "current_only_control": ("current", False),
        "causal_control_with_rgb_history": ("causal", True),
        "current_only_control_with_rgb_history": ("current", True),
    }
    legacy_mode = config[legacy_key]
    if not isinstance(legacy_mode, str) or legacy_mode not in legacy_modes:
        raise ValueError(f"Unsupported legacy {legacy_key}: {legacy_mode!r}.")
    control_visibility, controls_read_rgb = legacy_modes[legacy_mode]
    expected_policy = {
        "control_visibility": control_visibility,
        "controls_read_strict_past_clean_rgb": controls_read_rgb,
        "clean_pass_causality": "frame",
        "multiview_attention_scope": "all_views",
        "decomposed_temporal_window_seconds": None,
    }
    policy = config.get("teacher_forcing_replay_policy", {})
    if not isinstance(policy, dict):
        raise TypeError("Expected teacher_forcing_replay_policy to be a dictionary during legacy Transfer migration.")
    for key, expected in expected_policy.items():
        if key in policy and policy[key] != expected:
            raise ValueError(
                f"Legacy {legacy_key}={legacy_mode!r} conflicts with teacher_forcing_replay_policy.{key}={policy[key]!r}; "
                f"expected {expected!r}."
            )

    # The legacy Transfer implementation used three-way single-view attention.
    # An explicit different implementation must not silently replace that kernel.
    implementation = config.get("teacher_forcing_kv_implementation", "singleview_threeway_kv")
    if implementation != "singleview_threeway_kv":
        raise ValueError(
            f"Legacy {legacy_key} conflicts with teacher_forcing_kv_implementation={implementation!r}; "
            "expected 'singleview_threeway_kv'."
        )
    config["teacher_forcing_replay_policy"] = {**policy, **expected_policy}
    config["teacher_forcing_kv_implementation"] = implementation
    del config[legacy_key]


def sanitize_student_model_config(
    model_dict: dict[str, Any],
    *,
    base_model_target: str,
    base_config_type: str,
    base_config_field_names: set[str],
) -> None:
    """Convert a distillation model config into a portable student-only base config."""
    config = model_dict.get("config")
    if not isinstance(config, dict):
        raise TypeError("Expected model config to be a dictionary.")

    # Migrate before filtering out training-only fields: dropping the legacy
    # selector first would silently restore global controls without RGB history.
    _migrate_legacy_transfer_replay_config(config, base_config_field_names=base_config_field_names)

    model_dict["_target_"] = base_model_target
    config["_type"] = base_config_type

    allowed_keys = base_config_field_names | {"_type"}
    for key in tuple(config):
        if key not in allowed_keys:
            del config[key]

    compile_config = config.get("compile")
    if isinstance(compile_config, dict):
        compile_config["enabled"] = False


def sanitize_student_public_model_config(
    model_dict: dict[str, Any],
    *,
    public_vlm_tokenizer_target: str | None = None,
) -> None:
    """Replace internal loader settings with portable public checkpoint aliases."""
    config = model_dict.get("config")
    if not isinstance(config, dict):
        raise TypeError("Expected model config to be a dictionary.")

    for tokenizer_key in ("tokenizer", "sound_tokenizer"):
        tokenizer_config = config.get(tokenizer_key)
        if not isinstance(tokenizer_config, dict):
            continue
        if "bucket_name" in tokenizer_config:
            tokenizer_config["bucket_name"] = "bucket"
        if "object_store_credential_path_pretrained" in tokenizer_config:
            tokenizer_config["object_store_credential_path_pretrained"] = ""
        if tokenizer_key == "tokenizer" and "vae_path" in tokenizer_config:
            tokenizer_config["vae_path"] = _normalize_public_dependency_path(
                tokenizer_config["vae_path"],
                field_name="tokenizer.vae_path",
                public_paths=_PUBLIC_WAN_VAE_PATHS,
            )

    vlm_config = config.get("vlm_config")
    if not isinstance(vlm_config, dict):
        return

    model_instance = vlm_config.get("model_instance")
    if isinstance(model_instance, dict):
        model_instance_config = model_instance.get("config")
        if isinstance(model_instance_config, dict):
            base_config = model_instance_config.get("base_config")
            if isinstance(base_config, dict) and "json_file" in base_config:
                base_config["json_file"] = _normalize_public_dependency_path(
                    base_config["json_file"],
                    field_name="vlm_config.model_instance.config.base_config.json_file",
                    public_paths=_PUBLIC_QWEN3_VL_CONFIG_PATHS,
                )

    pretrained_weights = vlm_config.get("pretrained_weights")
    if isinstance(pretrained_weights, dict):
        pretrained_weights["enabled"] = False
        pretrained_weights["backbone_path"] = ""
        pretrained_weights["credentials_path"] = ""
        pretrained_weights["enable_gcs_patch_in_boto3"] = False

    tokenizer_config = vlm_config.get("tokenizer")
    if isinstance(tokenizer_config, dict):
        if public_vlm_tokenizer_target is not None:
            tokenizer_config["_target_"] = public_vlm_tokenizer_target
        if "config_variant" in tokenizer_config:
            tokenizer_config["config_variant"] = "hf"


def resolve_vision_checkpoint_path(
    *,
    local_path: str | None,
    configured_uri: str,
    download_checkpoint: Callable[[str], str],
) -> str:
    """Use a local vision checkpoint when supplied, otherwise download the configured checkpoint."""
    if local_path is not None:
        return local_path
    return download_checkpoint(configured_uri)
