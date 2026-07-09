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

_PUBLIC_URI_PREFIX = "cosmos3://"
_LEGACY_PUBLIC_URI_PREFIX = "cosmos3://vfm/"
_I4_MODULE_PREFIX = "projects." + "cosmos3." + "cosmos3."
_I4_FILE_PREFIX = "projects/" + "cosmos3/" + "cosmos3/"

_TARGET_PATHS_BY_ALIAS = {
    "create_qwen2_tokenizer_with_download": _I4_MODULE_PREFIX
    + "configs.base.defaults.reasoner.create_qwen2_tokenizer_with_download",
    "create_vlm_config": _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.create_vlm_config",
    "nemotron3_dense_vl_mot_config_from_json_file": _I4_MODULE_PREFIX
    + "models.mot.unified_mot.Nemotron3DenseVLMoTConfig.from_json_file",
    "nemotron3_dense_vl_text_for_causal_lm": _I4_MODULE_PREFIX
    + "models.mot.unified_mot.Nemotron3DenseVLTextForCausalLM",
    "qwen3_vl_mot_config_from_json_file": _I4_MODULE_PREFIX + "models.mot.unified_mot.Qwen3VLMoTConfig.from_json_file",
    "qwen3_vl_text_for_causal_lm": _I4_MODULE_PREFIX + "models.mot.unified_mot.Qwen3VLTextForCausalLM",
    "omni_mot_model": _I4_MODULE_PREFIX + "models.omni_mot_model.OmniMoTModel",
    "build_processor_lazy": _I4_MODULE_PREFIX + "processors.build_processor_lazy",
    "avae_interface": _I4_MODULE_PREFIX + "tokenizers.audio.avae.AVAEInterface",
    "wan2pt2_vae_interface": _I4_MODULE_PREFIX + "tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface",
}

_TYPE_PATHS_BY_ALIAS = {
    "activation_checkpointing_config": _I4_MODULE_PREFIX
    + "configs.base.defaults.activation_checkpointing.ActivationCheckpointingConfig",
    "compile_config": _I4_MODULE_PREFIX + "configs.base.defaults.compile.CompileConfig",
    "ema_config": _I4_MODULE_PREFIX + "configs.base.defaults.ema.EMAConfig",
    "diffusion_expert_config": _I4_MODULE_PREFIX + "configs.base.defaults.model_config.DiffusionExpertConfig",
    "lbl_config": _I4_MODULE_PREFIX + "configs.base.defaults.model_config.LBLConfig",
    "omni_mot_model_config": _I4_MODULE_PREFIX + "configs.base.defaults.model_config.OmniMoTModelConfig",
    "rectified_flow_inference_config": _I4_MODULE_PREFIX
    + "configs.base.defaults.model_config.RectifiedFlowInferenceConfig",
    "rectified_flow_training_config": _I4_MODULE_PREFIX
    + "configs.base.defaults.model_config.RectifiedFlowTrainingConfig",
    "parallelism_config": _I4_MODULE_PREFIX + "configs.base.defaults.parallelism.ParallelismConfig",
    "pretrained_weights_config": _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.PretrainedWeightsConfig",
    "vlm_config": _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.VLMConfig",
}

# Config objects can be serialized as either `_type` or `_target_` depending on
# whether they came from structured config metadata or LazyCall construction.
_TARGET_PATHS_BY_ALIAS.update(_TYPE_PATHS_BY_ALIAS)

_CURRENT_TARGET_ALIASES = {path: alias for alias, path in _TARGET_PATHS_BY_ALIAS.items()}
_CURRENT_TYPE_ALIASES = {path: alias for alias, path in _TYPE_PATHS_BY_ALIAS.items()}


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
            if key == _PUBLIC_TARGET_KEY:
                return True
            if key == _CLASS_NAME_KEY and _is_target_alias(value):
                return True
            if key in {_PUBLIC_TYPE_KEY, _CONFIG_NAME_KEY} and _is_type_alias(value):
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
            if key == _PUBLIC_TARGET_KEY or (key == _CLASS_NAME_KEY and _is_target_alias(value)):
                data[_TARGET_KEY] = _alias_to_runtime_target(value)
            elif key in {_PUBLIC_TYPE_KEY, _CONFIG_NAME_KEY} and _is_type_alias(value):
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
    return _module_path_to_alias(target, _CURRENT_TARGET_ALIASES, kind="target")


def _type_to_alias(tp: Any) -> str:
    if not isinstance(tp, str):
        raise TypeError(f"Expected type path string, got {type(tp)}")
    return _module_path_to_alias(tp, _CURRENT_TYPE_ALIASES, kind="type")


def _module_path_to_alias(path: str, aliases: dict[str, str], *, kind: str) -> str:
    current_path = _current_module_path_to_i4(path)
    alias = aliases.get(current_path)
    if alias is None:
        alias = aliases.get(_legacy_module_path_to_i4(path))
    if alias is None:
        raise ValueError(f"No public alias registered for {kind} path: {path}")
    return alias


def _alias_to_runtime_target(alias: Any) -> str:
    if not isinstance(alias, str):
        raise TypeError(f"Expected target alias string, got {type(alias)}")
    try:
        current_path = _TARGET_PATHS_BY_ALIAS[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown Cosmos target alias: {alias}") from exc
    return _runtime_module_path(current_path)


def _alias_to_runtime_type(alias: Any) -> str:
    if not isinstance(alias, str):
        raise TypeError(f"Expected type alias string, got {type(alias)}")
    try:
        current_path = _TYPE_PATHS_BY_ALIAS[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown Cosmos type alias: {alias}") from exc
    return _runtime_module_path(current_path)


def _is_type_alias(value: Any) -> bool:
    return isinstance(value, str) and value in _TYPE_PATHS_BY_ALIAS


def _is_target_alias(value: Any) -> bool:
    return isinstance(value, str) and value in _TARGET_PATHS_BY_ALIAS


def _current_module_path_to_i4(path: str) -> str:
    replacements = (
        (
            "cosmos_framework.configs.base.defaults.reasoner.",
            _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.",
        ),
        ("cosmos_framework.configs.base.reasoner.", _I4_MODULE_PREFIX + "configs.base.reasoner."),
        ("cosmos_framework.model.generator.reasoner.", _I4_MODULE_PREFIX + "models.reasoner."),
        (
            "cosmos_framework.data.generator.augmentors.reasoner.",
            _I4_MODULE_PREFIX + "datasets.augmentors.reasoner.",
        ),
        ("cosmos_framework.data.generator.reasoner.", _I4_MODULE_PREFIX + "datasets.reasoner."),
        ("cosmos_framework.utils.generator.reasoner.", _I4_MODULE_PREFIX + "utils.reasoner."),
        ("cosmos_framework.configs.base.", _I4_MODULE_PREFIX + "configs.base."),
        ("cosmos_framework.model.generator.tokenizers.", _I4_MODULE_PREFIX + "tokenizers."),
        ("cosmos_framework.model.generator.diffusion.", _I4_MODULE_PREFIX + "diffusion."),
        ("cosmos_framework.model.generator.", _I4_MODULE_PREFIX + "models."),
        ("cosmos_framework.data.generator.processors.", _I4_MODULE_PREFIX + "processors."),
        ("cosmos_framework.data.generator.", _I4_MODULE_PREFIX + "datasets."),
        ("cosmos_framework.utils.generator.", _I4_MODULE_PREFIX + "utils."),
    )
    return _replace_prefix(path, replacements)


def _legacy_module_path_to_i4(path: str) -> str:
    """Translate pre-alias module paths for backward-compatible reads only."""

    legacy_i4_path = _replace_prefix(
        path,
        (
            ("cosmos3._src.vfm.", "projects.cosmos3.vfm."),
            ("cosmos_framework.model.vfm.tokenizers.", "projects.cosmos3.vfm.tokenizers."),
            ("cosmos_framework.model.vfm.diffusion.", "projects.cosmos3.vfm.diffusion."),
            ("cosmos_framework.model.vfm.", "projects.cosmos3.vfm.models."),
            ("cosmos_framework.data.vfm.processors.", "projects.cosmos3.vfm.processors."),
            ("cosmos_framework.data.vfm.", "projects.cosmos3.vfm.datasets."),
            ("cosmos_framework.utils.vfm.", "projects.cosmos3.vfm.utils."),
            ("cosmos_framework.configs.base.", "projects.cosmos3.vfm.configs.base."),
            ("cosmos.model.vfm.tokenizers.", "projects.cosmos3.vfm.tokenizers."),
            ("cosmos.model.vfm.diffusion.", "projects.cosmos3.vfm.diffusion."),
            ("cosmos.model.vfm.", "projects.cosmos3.vfm.models."),
            ("cosmos.data.vfm.processors.", "projects.cosmos3.vfm.processors."),
            ("cosmos.data.vfm.", "projects.cosmos3.vfm.datasets."),
            ("cosmos.utils.vfm.", "projects.cosmos3.vfm.utils."),
            ("cosmos.configs.base.", "projects.cosmos3.vfm.configs.base."),
        ),
    )
    return _replace_prefix(
        legacy_i4_path,
        (
            (
                "projects.cosmos3.vfm.configs.base.defaults.vlm.",
                _I4_MODULE_PREFIX + "configs.base.defaults.reasoner.",
            ),
            ("projects.cosmos3.vfm.configs.base.vlm.", _I4_MODULE_PREFIX + "configs.base.reasoner."),
            ("projects.cosmos3.vfm.models.vlm.", _I4_MODULE_PREFIX + "models.reasoner."),
            (
                "projects.cosmos3.vfm.datasets.augmentors.vlm.",
                _I4_MODULE_PREFIX + "datasets.augmentors.reasoner.",
            ),
            ("projects.cosmos3.vfm.datasets.vlm.", _I4_MODULE_PREFIX + "datasets.reasoner."),
            ("projects.cosmos3.vfm.utils.vlm.", _I4_MODULE_PREFIX + "utils.reasoner."),
            ("projects.cosmos3.vfm.", _I4_MODULE_PREFIX),
        ),
    )


def _runtime_module_path(current_i4_path: str) -> str:
    if _module_exists("cosmos_framework.model.generator"):
        return _current_i4_module_path_to_framework(current_i4_path, package="cosmos_framework")
    if _module_exists("cosmos_framework.model.vfm"):
        return _current_i4_module_path_to_legacy_runtime(current_i4_path, package="cosmos_framework")
    if _module_exists("cosmos.model.vfm"):
        return _current_i4_module_path_to_legacy_runtime(current_i4_path, package="cosmos")
    if _module_exists("cosmos3._src.vfm"):
        suffix = current_i4_path.removeprefix(_I4_MODULE_PREFIX)
        return "cosmos3._src.vfm." + _current_suffix_to_legacy_vfm(suffix, separator=".")
    return current_i4_path


def _current_i4_module_path_to_framework(current_i4_path: str, *, package: str) -> str:
    if not current_i4_path.startswith(_I4_MODULE_PREFIX):
        return current_i4_path
    suffix = current_i4_path[len(_I4_MODULE_PREFIX) :]
    replacements = (
        ("configs.base.defaults.reasoner.", f"{package}.configs.base.defaults.reasoner."),
        ("configs.base.reasoner.", f"{package}.configs.base.reasoner."),
        ("models.reasoner.", f"{package}.model.generator.reasoner."),
        ("datasets.augmentors.reasoner.", f"{package}.data.generator.augmentors.reasoner."),
        ("datasets.reasoner.", f"{package}.data.generator.reasoner."),
        ("utils.reasoner.", f"{package}.utils.generator.reasoner."),
        ("configs.base.", f"{package}.configs.base."),
        ("tokenizers.", f"{package}.model.generator.tokenizers."),
        ("diffusion.", f"{package}.model.generator.diffusion."),
        ("models.", f"{package}.model.generator."),
        ("processors.", f"{package}.data.generator.processors."),
        ("datasets.", f"{package}.data.generator."),
        ("scripts.action.", f"{package}.data.generator.action_scripts."),
        ("utils.", f"{package}.utils.generator."),
    )
    return _replace_prefix(suffix, replacements)


def _current_i4_module_path_to_legacy_runtime(current_i4_path: str, *, package: str) -> str:
    """Restore aliases into a legacy installed package layout when detected."""

    if not current_i4_path.startswith(_I4_MODULE_PREFIX):
        return current_i4_path
    suffix = _current_suffix_to_legacy_vfm(current_i4_path[len(_I4_MODULE_PREFIX) :], separator=".")
    replacements = (
        ("configs.base.", f"{package}.configs.base."),
        ("tokenizers.", f"{package}.model.vfm.tokenizers."),
        ("diffusion.", f"{package}.model.vfm.diffusion."),
        ("models.", f"{package}.model.vfm."),
        ("processors.", f"{package}.data.vfm.processors."),
        ("datasets.", f"{package}.data.vfm."),
        ("scripts.action.", f"{package}.data.vfm.action_scripts."),
        ("utils.", f"{package}.utils.vfm."),
    )
    return _replace_prefix(suffix, replacements)


def _to_public_string(value: str) -> str:
    if value.startswith(_LEGACY_PUBLIC_URI_PREFIX):
        suffix = _legacy_vfm_suffix_to_current(value[len(_LEGACY_PUBLIC_URI_PREFIX) :], separator="/")
        return _PUBLIC_URI_PREFIX + suffix
    if value.startswith(_PUBLIC_URI_PREFIX):
        return value
    if value.startswith(_I4_FILE_PREFIX):
        return _PUBLIC_URI_PREFIX + value[len(_I4_FILE_PREFIX) :]
    for package in ("cosmos_framework", "cosmos"):
        suffix = _current_runtime_file_path_to_public_suffix(value, package=package)
        if suffix is not None:
            return _PUBLIC_URI_PREFIX + _legacy_vfm_suffix_to_current(suffix, separator="/")
    legacy_suffix = _legacy_file_path_to_current_suffix(value)
    if legacy_suffix is not None:
        return _PUBLIC_URI_PREFIX + legacy_suffix
    return value


def _current_runtime_file_path_to_public_suffix(value: str, *, package: str) -> str | None:
    replacements = (
        (f"{package}/configs/base/defaults/reasoner/", "configs/base/defaults/reasoner/"),
        (f"{package}/configs/base/reasoner/", "configs/base/reasoner/"),
        (f"{package}/model/generator/reasoner/", "models/reasoner/"),
        (f"{package}/data/generator/augmentors/reasoner/", "datasets/augmentors/reasoner/"),
        (f"{package}/data/generator/reasoner/", "datasets/reasoner/"),
        (f"{package}/utils/generator/reasoner/", "utils/reasoner/"),
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
            return new + value[len(old) :]
    return None


def _legacy_file_path_to_current_suffix(value: str) -> str | None:
    """Translate pre-alias resource paths for backward-compatible reads only."""

    for prefix in ("projects/cosmos3/vfm/", "cosmos3/_src/vfm/"):
        if value.startswith(prefix):
            return _legacy_vfm_suffix_to_current(value[len(prefix) :], separator="/")
    for package in ("cosmos_framework", "cosmos"):
        legacy_suffix = _replace_prefix(
            value,
            (
                (f"{package}/configs/base/", "configs/base/"),
                (f"{package}/model/vfm/tokenizers/", "tokenizers/"),
                (f"{package}/model/vfm/diffusion/", "diffusion/"),
                (f"{package}/model/vfm/", "models/"),
                (f"{package}/data/vfm/processors/", "processors/"),
                (f"{package}/data/vfm/action_scripts/", "scripts/action/"),
                (f"{package}/data/vfm/", "datasets/"),
                (f"{package}/utils/vfm/", "utils/"),
            ),
        )
        if legacy_suffix != value:
            return _legacy_vfm_suffix_to_current(legacy_suffix, separator="/")
    return None


def _from_public_string(value: str) -> str:
    if value.startswith(_LEGACY_PUBLIC_URI_PREFIX):
        suffix = _legacy_vfm_suffix_to_current(value[len(_LEGACY_PUBLIC_URI_PREFIX) :], separator="/")
    elif value.startswith(_PUBLIC_URI_PREFIX):
        suffix = value[len(_PUBLIC_URI_PREFIX) :]
    else:
        return value
    if _module_exists("cosmos_framework.model.generator"):
        return _public_suffix_to_framework_file_path(suffix, package="cosmos_framework")
    if _module_exists("cosmos_framework.model.vfm"):
        return _public_suffix_to_legacy_runtime_file_path(suffix, package="cosmos_framework")
    if _module_exists("cosmos.model.vfm"):
        return _public_suffix_to_legacy_runtime_file_path(suffix, package="cosmos")
    if _module_exists("cosmos3._src.vfm"):
        return "cosmos3/_src/vfm/" + _current_suffix_to_legacy_vfm(suffix, separator="/")
    return _I4_FILE_PREFIX + suffix


def _public_suffix_to_framework_file_path(suffix: str, *, package: str) -> str:
    replacements = (
        ("configs/base/defaults/reasoner/", f"{package}/configs/base/defaults/reasoner/"),
        ("configs/base/reasoner/", f"{package}/configs/base/reasoner/"),
        ("models/reasoner/", f"{package}/model/generator/reasoner/"),
        ("datasets/augmentors/reasoner/", f"{package}/data/generator/augmentors/reasoner/"),
        ("datasets/reasoner/", f"{package}/data/generator/reasoner/"),
        ("utils/reasoner/", f"{package}/utils/generator/reasoner/"),
        ("configs/base/", f"{package}/configs/base/"),
        ("tokenizers/", f"{package}/model/generator/tokenizers/"),
        ("diffusion/", f"{package}/model/generator/diffusion/"),
        ("models/", f"{package}/model/generator/"),
        ("processors/", f"{package}/data/generator/processors/"),
        ("scripts/action/", f"{package}/data/generator/action_scripts/"),
        ("datasets/", f"{package}/data/generator/"),
        ("utils/", f"{package}/utils/generator/"),
    )
    replaced = _replace_prefix(suffix, replacements)
    if replaced != suffix:
        return replaced
    return f"{package}/_cosmos3_unmapped/{suffix}"


def _public_suffix_to_legacy_runtime_file_path(suffix: str, *, package: str) -> str:
    legacy_suffix = _current_suffix_to_legacy_vfm(suffix, separator="/")
    replacements = (
        ("configs/base/", f"{package}/configs/base/"),
        ("tokenizers/", f"{package}/model/vfm/tokenizers/"),
        ("diffusion/", f"{package}/model/vfm/diffusion/"),
        ("models/", f"{package}/model/vfm/"),
        ("processors/", f"{package}/data/vfm/processors/"),
        ("scripts/action/", f"{package}/data/vfm/action_scripts/"),
        ("datasets/", f"{package}/data/vfm/"),
        ("utils/", f"{package}/utils/vfm/"),
    )
    replaced = _replace_prefix(legacy_suffix, replacements)
    if replaced != legacy_suffix:
        return replaced
    return f"{package}/_vfm_unmapped/{legacy_suffix}"


def _legacy_vfm_suffix_to_current(suffix: str, *, separator: str) -> str:
    replacements = (
        (
            separator.join(("configs", "base", "defaults", "vlm", "")),
            separator.join(("configs", "base", "defaults", "reasoner", "")),
        ),
        (
            separator.join(("configs", "base", "vlm", "")),
            separator.join(("configs", "base", "reasoner", "")),
        ),
        (separator.join(("models", "vlm", "")), separator.join(("models", "reasoner", ""))),
        (
            separator.join(("datasets", "augmentors", "vlm", "")),
            separator.join(("datasets", "augmentors", "reasoner", "")),
        ),
        (separator.join(("datasets", "vlm", "")), separator.join(("datasets", "reasoner", ""))),
        (separator.join(("utils", "vlm", "")), separator.join(("utils", "reasoner", ""))),
    )
    return _replace_prefix(suffix, replacements)


def _current_suffix_to_legacy_vfm(suffix: str, *, separator: str) -> str:
    replacements = (
        (
            separator.join(("configs", "base", "defaults", "reasoner", "")),
            separator.join(("configs", "base", "defaults", "vlm", "")),
        ),
        (
            separator.join(("configs", "base", "reasoner", "")),
            separator.join(("configs", "base", "vlm", "")),
        ),
        (separator.join(("models", "reasoner", "")), separator.join(("models", "vlm", ""))),
        (
            separator.join(("datasets", "augmentors", "reasoner", "")),
            separator.join(("datasets", "augmentors", "vlm", "")),
        ),
        (separator.join(("datasets", "reasoner", "")), separator.join(("datasets", "vlm", ""))),
        (separator.join(("utils", "reasoner", "")), separator.join(("utils", "vlm", ""))),
    )
    return _replace_prefix(suffix, replacements)


def _replace_prefix(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for old, new in replacements:
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def _module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False
