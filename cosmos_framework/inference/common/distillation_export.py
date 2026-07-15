# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Portable student-only checkpoint export helpers."""

from collections.abc import Callable
from typing import Any


def build_student_checkpoint_metadata(*, use_ema_weights: bool) -> dict[str, str | bool]:
    """Build portable metadata without source checkpoint or credential paths."""
    return {
        "checkpoint_type": "hf",
        "source_weights": "ema" if use_ema_weights else "regular",
        "student_only": True,
    }


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

    vlm_config = config.get("vlm_config")
    if not isinstance(vlm_config, dict):
        return

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
