# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import importlib.util
from typing import Any

_PUBLIC_TARGET_KEY = "_target"
_PUBLIC_TYPE_KEY = "_type"
_CLASS_NAME_KEY = "class_name"
_CONFIG_NAME_KEY = "config_name"
_TYPE_KEY = "_type"
_TARGET_KEY = "_target_"

_PUBLIC_VFM_URI_PREFIX = "cosmos3://vfm/"

_TARGET_ALIASES = {
    "projects.cosmos3.vfm.configs.base.defaults.vlm.create_qwen2_tokenizer_with_download": "create_qwen2_tokenizer_with_download",
    "projects.cosmos3.vfm.configs.base.defaults.vlm.create_vlm_config": "create_vlm_config",
    "projects.cosmos3.vfm.models.mot.unified_mot.Qwen3VLMoTConfig.from_json_file": "qwen3_vl_mot_config_from_json_file",
    "projects.cosmos3.vfm.models.mot.unified_mot.Qwen3VLTextForCausalLM": "qwen3_vl_text_for_causal_lm",
    "projects.cosmos3.vfm.models.omni_mot_model.OmniMoTModel": "omni_mot_model",
    "projects.cosmos3.vfm.processors.build_processor_lazy": "build_processor_lazy",
    "projects.cosmos3.vfm.tokenizers.audio.avae.AVAEInterface": "avae_interface",
    "projects.cosmos3.vfm.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface": "wan2pt2_vae_interface",
}

_TYPE_ALIASES = {
    "projects.cosmos3.vfm.configs.base.defaults.activation_checkpointing.ActivationCheckpointingConfig": "activation_checkpointing_config",
    "projects.cosmos3.vfm.configs.base.defaults.compile.CompileConfig": "compile_config",
    "projects.cosmos3.vfm.configs.base.defaults.ema.EMAConfig": "ema_config",
    "projects.cosmos3.vfm.configs.base.defaults.model_config.DiffusionExpertConfig": "diffusion_expert_config",
    "projects.cosmos3.vfm.configs.base.defaults.model_config.LBLConfig": "lbl_config",
    "projects.cosmos3.vfm.configs.base.defaults.model_config.OmniMoTModelConfig": "omni_mot_model_config",
    "projects.cosmos3.vfm.configs.base.defaults.model_config.RectifiedFlowInferenceConfig": "rectified_flow_inference_config",
    "projects.cosmos3.vfm.configs.base.defaults.model_config.RectifiedFlowTrainingConfig": "rectified_flow_training_config",
    "projects.cosmos3.vfm.configs.base.defaults.parallelism.ParallelismConfig": "parallelism_config",
    "projects.cosmos3.vfm.configs.base.defaults.vlm.PretrainedWeightsConfig": "pretrained_weights_config",
    "projects.cosmos3.vfm.configs.base.defaults.vlm.VLMConfig": "vlm_config",
}

# Config objects can be serialized as either `_type` or `_target_` depending on
# whether they came from structured config metadata or LazyCall construction.
_TARGET_ALIASES.update(_TYPE_ALIASES)

_ALIAS_TARGETS = {alias: target for target, alias in _TARGET_ALIASES.items()}
_ALIAS_TYPES = {alias: target for target, alias in _TYPE_ALIASES.items()}


def build_public_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Build public HF model metadata from the internal LazyConfig model dict."""

    return _to_public_model_config(model_config)


def restore_model_config_from_public_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Restore the internal LazyConfig model dict from public HF model metadata."""

    return _from_public_model_config(model_config)


def model_config_uses_public_aliases(model_config: Any) -> bool:
    """Return whether a model config uses public aliases."""

    return _has_public_alias(model_config)


def load_model_config_from_hf_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Load internal model config from either legacy or public HF config."""

    if "model" in config_dict:
        model_config = config_dict["model"]
        if model_config_uses_public_aliases(model_config):
            return restore_model_config_from_public_model_config(model_config)
        return model_config
    raise KeyError("HF config must contain 'model'")


def _has_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_has_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_has_key(value, key) for value in obj)
    return False


def _has_public_alias(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {_PUBLIC_TARGET_KEY, _CLASS_NAME_KEY, _CONFIG_NAME_KEY}:
                return True
            if key == _PUBLIC_TYPE_KEY and _is_type_alias(value):
                return True
            if _has_public_alias(value):
                return True
    elif isinstance(obj, list):
        return any(_has_public_alias(value) for value in obj)
    return False


def _to_public_model_config(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_to_public_model_config(item) for item in obj]
    if isinstance(obj, dict):
        data = {}
        for key, value in obj.items():
            if key == _TARGET_KEY:
                data[_PUBLIC_TARGET_KEY] = _target_to_alias(value)
            elif key == _TYPE_KEY:
                data[_PUBLIC_TYPE_KEY] = _type_to_alias(value)
            else:
                data[key] = _to_public_model_config(value)
        return data
    if isinstance(obj, str):
        return _to_public_string(obj)
    return copy.deepcopy(obj)


def _from_public_model_config(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_from_public_model_config(item) for item in obj]
    if isinstance(obj, dict):
        data = {}
        for key, value in obj.items():
            if key in {_PUBLIC_TARGET_KEY, _CLASS_NAME_KEY}:
                data[_TARGET_KEY] = _alias_to_runtime_target(value)
            elif key == _CONFIG_NAME_KEY or (key == _PUBLIC_TYPE_KEY and _is_type_alias(value)):
                data[_TYPE_KEY] = _alias_to_runtime_type(value)
            else:
                data[key] = _from_public_model_config(value)
        return data
    if isinstance(obj, str):
        return _from_public_string(obj)
    return copy.deepcopy(obj)


def _target_to_alias(target: Any) -> str:
    if not isinstance(target, str):
        raise TypeError(f"Expected target path string, got {type(target)}")
    canonical = _canonicalize_module_path(target)
    try:
        return _TARGET_ALIASES[canonical]
    except KeyError as exc:
        raise ValueError(f"No public alias registered for target path: {target}") from exc


def _type_to_alias(tp: Any) -> str:
    if not isinstance(tp, str):
        raise TypeError(f"Expected type path string, got {type(tp)}")
    canonical = _canonicalize_module_path(tp)
    try:
        return _TYPE_ALIASES[canonical]
    except KeyError as exc:
        raise ValueError(f"No public alias registered for type path: {tp}") from exc


def _alias_to_runtime_target(alias: Any) -> str:
    if not isinstance(alias, str):
        raise TypeError(f"Expected target alias string, got {type(alias)}")
    try:
        return _runtime_module_path(_ALIAS_TARGETS[alias])
    except KeyError as exc:
        raise ValueError(f"Unknown Cosmos target alias: {alias}") from exc


def _alias_to_runtime_type(alias: Any) -> str:
    if not isinstance(alias, str):
        raise TypeError(f"Expected type alias string, got {type(alias)}")
    try:
        return _runtime_module_path(_ALIAS_TYPES[alias])
    except KeyError as exc:
        raise ValueError(f"Unknown Cosmos type alias: {alias}") from exc


def _is_type_alias(value: Any) -> bool:
    return isinstance(value, str) and value in _ALIAS_TYPES


def _canonicalize_module_path(path: str) -> str:
    replacements = (
        # vlm → reasoner rename (upstream i4). Longer/more-specific rules first
        # so the reasoner subtree isn't shadowed by the general vfm rules below.
        ("cosmos_framework.configs.base.defaults.reasoner.", "projects.cosmos3.vfm.configs.base.defaults.vlm."),
        ("cosmos_framework.configs.base.reasoner.", "projects.cosmos3.vfm.configs.base.vlm."),
        ("cosmos_framework.model.generator.reasoner.", "projects.cosmos3.vfm.models.vlm."),
        ("cosmos_framework.data.generator.reasoner.", "projects.cosmos3.vfm.datasets.vlm."),
        ("cosmos_framework.data.generator.augmentors.reasoner.", "projects.cosmos3.vfm.datasets.augmentors.vlm."),
        ("cosmos_framework.utils.generator.reasoner.", "projects.cosmos3.vfm.utils.vlm."),
        ("cosmos3._src.vfm.", "projects.cosmos3.vfm."),
        ("cosmos_framework.configs.base.", "projects.cosmos3.vfm.configs.base."),
        ("cosmos_framework.model.generator.tokenizers.", "projects.cosmos3.vfm.tokenizers."),
        ("cosmos_framework.model.generator.diffusion.", "projects.cosmos3.vfm.diffusion."),
        ("cosmos_framework.model.generator.", "projects.cosmos3.vfm.models."),
        ("cosmos_framework.data.generator.processors.", "projects.cosmos3.vfm.processors."),
        ("cosmos.model.vfm.tokenizers.", "projects.cosmos3.vfm.tokenizers."),
        ("cosmos.model.vfm.diffusion.", "projects.cosmos3.vfm.diffusion."),
        ("cosmos.model.vfm.", "projects.cosmos3.vfm.models."),
        ("cosmos.configs.base.", "projects.cosmos3.vfm.configs.base."),
    )
    for old, new in replacements:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def _runtime_module_path(canonical_path: str) -> str:
    if _module_exists("cosmos_framework.model.generator"):
        return _replace_vfm_module_prefix(canonical_path, package="cosmos_framework")
    if _module_exists("cosmos.model.vfm"):
        return _replace_vfm_module_prefix(canonical_path, package="cosmos")
    if _module_exists("cosmos3._src.vfm"):
        return canonical_path.replace("projects.cosmos3.vfm.", "cosmos3._src.vfm.", 1)
    return canonical_path


def _replace_vfm_module_prefix(canonical_path: str, *, package: str) -> str:
    replacements = (
        # vlm → reasoner (upstream rename). MUST precede the general vfm rules
        # so the specific vlm→reasoner subtree isn't shadowed by them.
        ("projects.cosmos3.vfm.configs.base.defaults.vlm.", f"{package}.configs.base.defaults.reasoner."),
        ("projects.cosmos3.vfm.configs.base.vlm.", f"{package}.configs.base.reasoner."),
        ("projects.cosmos3.vfm.models.vlm.", f"{package}.model.generator.reasoner."),
        ("projects.cosmos3.vfm.datasets.vlm.", f"{package}.data.generator.reasoner."),
        ("projects.cosmos3.vfm.datasets.augmentors.vlm.", f"{package}.data.generator.augmentors.reasoner."),
        ("projects.cosmos3.vfm.utils.vlm.", f"{package}.utils.generator.reasoner."),
        ("projects.cosmos3.vfm.configs.base.", f"{package}.configs.base."),
        ("projects.cosmos3.vfm.models.", f"{package}.model.generator."),
        ("projects.cosmos3.vfm.tokenizers.", f"{package}.model.generator.tokenizers."),
        ("projects.cosmos3.vfm.diffusion.", f"{package}.model.generator.diffusion."),
        ("projects.cosmos3.vfm.processors.", f"{package}.data.generator.processors."),
        ("projects.cosmos3.vfm.datasets.", f"{package}.data.generator."),
        ("projects.cosmos3.vfm.scripts.action.", f"{package}.data.generator.action_scripts."),
        ("projects.cosmos3.vfm.utils.", f"{package}.utils.generator."),
    )
    for old, new in replacements:
        if canonical_path.startswith(old):
            return new + canonical_path[len(old) :]
    return canonical_path


def _to_public_string(value: str) -> str:
    for prefix in ("projects/cosmos3/vfm/", "cosmos3/_src/vfm/"):
        if value.startswith(prefix):
            return _PUBLIC_VFM_URI_PREFIX + value[len(prefix) :]
    for package in ("cosmos_framework", "cosmos"):
        public_value = _public_string_from_runtime_file_prefix(value, package=package)
        if public_value is not None:
            return public_value
    return value


def _public_string_from_runtime_file_prefix(value: str, *, package: str) -> str | None:
    replacements = (
        # reasoner → vlm (inverse of the upstream rename). MUST precede the
        # general vfm rules so the reasoner subtree isn't shadowed by them.
        # Mirrors the module-path rules in ``_canonicalize_module_path``.
        (f"{package}/configs/base/defaults/reasoner/", "configs/base/defaults/vlm/"),
        (f"{package}/configs/base/reasoner/", "configs/base/vlm/"),
        (f"{package}/model/generator/reasoner/", "models/vlm/"),
        (f"{package}/data/generator/augmentors/reasoner/", "datasets/augmentors/vlm/"),
        (f"{package}/data/generator/reasoner/", "datasets/vlm/"),
        (f"{package}/utils/generator/reasoner/", "utils/vlm/"),
        (f"{package}/configs/base/", "configs/base/"),
        (f"{package}/model/generator/tokenizers/", "tokenizers/"),
        (f"{package}/model/generator/diffusion/", "diffusion/"),
        (f"{package}/model/generator/", "models/"),
        (f"{package}/data/generator/processors/", "processors/"),
        (f"{package}/data/generator/action_scripts/", "scripts/action/"),
        (f"{package}/data/generator/", "datasets/"),
        (f"{package}/utils/generator/", "utils/"),
    )
    for old, new in replacements:
        if value.startswith(old):
            return _PUBLIC_VFM_URI_PREFIX + new + value[len(old) :]
    return None


def _from_public_string(value: str) -> str:
    if not value.startswith(_PUBLIC_VFM_URI_PREFIX):
        return value
    suffix = value[len(_PUBLIC_VFM_URI_PREFIX) :]
    if _module_exists("cosmos_framework.model.generator"):
        return _replace_vfm_file_prefix(suffix, package="cosmos_framework")
    if _module_exists("cosmos.model.vfm"):
        return _replace_vfm_file_prefix(suffix, package="cosmos")
    if _module_exists("cosmos3._src.vfm"):
        return f"cosmos3/_src/vfm/{suffix}"
    return f"projects/cosmos3/vfm/{suffix}"


def _replace_vfm_file_prefix(suffix: str, *, package: str) -> str:
    replacements = (
        # vlm → reasoner (upstream rename). MUST precede the general vfm rules
        # so the specific vlm→reasoner subtree isn't shadowed by them. Mirrors
        # the module-path rules in ``_replace_vfm_module_prefix``.
        ("configs/base/defaults/vlm/", f"{package}/configs/base/defaults/reasoner/"),
        ("configs/base/vlm/", f"{package}/configs/base/reasoner/"),
        ("models/vlm/", f"{package}/model/generator/reasoner/"),
        ("datasets/augmentors/vlm/", f"{package}/data/generator/augmentors/reasoner/"),
        ("datasets/vlm/", f"{package}/data/generator/reasoner/"),
        ("utils/vlm/", f"{package}/utils/generator/reasoner/"),
        ("configs/base/", f"{package}/configs/base/"),
        ("models/", f"{package}/model/generator/"),
        ("tokenizers/", f"{package}/model/generator/tokenizers/"),
        ("diffusion/", f"{package}/model/generator/diffusion/"),
        ("processors/", f"{package}/data/generator/processors/"),
        ("datasets/", f"{package}/data/generator/"),
        ("scripts/action/", f"{package}/data/generator/action_scripts/"),
        ("utils/", f"{package}/utils/generator/"),
    )
    for old, new in replacements:
        if suffix.startswith(old):
            return new + suffix[len(old) :]
    return f"{package}/_vfm_unmapped/{suffix}"


def _module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False
