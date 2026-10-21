# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig

from cosmos_framework.posttrain.config.model import DataPackerConfig
from cosmos_framework.posttrain.data.packer.base import BaseDataPacker, DataPacker
from cosmos_framework.posttrain.data.packer.decoder_only_llm import (
    DecoderOnlyLLMDataPacker,
    RLPolicyInput,
    shape_policy_input,
)
from cosmos_framework.posttrain.data.packer.qwen3_vl import (
    Qwen3VLCrossTokenTeacherInput,
    Qwen3VLDataPacker,
    Qwen3VLPolicyInput,
)
from cosmos_framework.posttrain.data.packer.qwen3_vl_video import Qwen3VLVideoDataPacker

if TYPE_CHECKING:
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

register = BaseDataPacker.register
get_default_data_packer = BaseDataPacker.get_default_data_packer

# Default registrations for common decoder-only text model_types. A new per-model
# packer (VLM/VLA) drops in as its own module that calls register(...) at import.
register(["gpt2", "qwen2", "qwen3", "llama"], DecoderOnlyLLMDataPacker)
register(["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"], Qwen3VLDataPacker)
# Video rows (frame lists with fps / frames_indices) are shaped by `Qwen3VLVideoDataPacker`, selected per run
# through `data_packer.factory=cosmos_framework.posttrain.data.packer.qwen3_vl_video.make_packer`; the registry
# default above stays the image packer.


def _select_packer_cls(hf_config: Any, *, model_backend: str | None = None) -> type[DataPacker]:
    """Choose a packer from the resolved model family and explicit policy backend."""
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    )

    if model_backend not in (None, "i4", "hf"):
        raise ValueError(f"unknown model_backend {model_backend!r}; expected 'i4' or 'hf'")
    model_type = getattr(hf_config, "model_type", None)
    if model_backend == "hf" and getattr(hf_config, "vision_config", None) is not None:
        from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

        return HFVLMDataPacker
    causal_only = (
        model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        and model_type not in MODEL_FOR_MASKED_LM_MAPPING_NAMES
        and model_type not in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    )
    if model_backend == "hf":
        if causal_only:
            return DecoderOnlyLLMDataPacker
        raise ValueError(
            f"cannot classify HF model_type={model_type!r}: it declares no vision_config and is "
            "not a recognized causal LM"
        )
    registered = BaseDataPacker._REGISTRY.get(model_type)
    if registered is not None:
        return registered
    if getattr(hf_config, "vision_config", None) is not None:
        raise ValueError(
            f"model_type={model_type!r} declares vision_config but has no native packer; "
            "select model_backend='hf' or pass an explicit packer"
        )
    if causal_only:
        return DecoderOnlyLLMDataPacker
    raise ValueError(
        f"cannot classify model_type={model_type!r}: it is neither a registered native architecture "
        "nor a recognized causal LM"
    )


def init_data_packer(
    name_or_path: str,
    packer: DataPacker | None = None,
    trust_remote_code: bool = False,
    *,
    model_backend: str | None = None,
) -> DataPacker:
    """Build and set up the backend-appropriate packer for one model identity.

    An explicit ``packer`` wins. Otherwise the HF config is resolved once: ``model_backend="hf"``
    selects ``HFVLMDataPacker`` for configs with ``vision_config`` and the decoder-only packer for
    recognized causal LMs; ``model_backend="i4"`` or the compatibility default selects the native
    registry. The native path never guesses a text packer for an unregistered VLM.

    Args:
        name_or_path: the HF model id/path; source of both `model_type` and the tokenizer.
        trust_remote_code: execute custom config/tokenizer code shipped by the HF repo. Defaults to
            FALSE, matching `HFModelConfig.trust_remote_code` — selecting a packer must not silently
            widen what a run executes. Threaded to BOTH the `AutoConfig` model_type probe and the
            packer's `setup()` tokenizer load, so there is one switch, not two.
        packer: an explicit `DataPacker` to use instead of registry selection. It must be a
            `DataPacker` (not merely a `BaseDataPacker`): every selected packer is `setup(...)`
            here, and `setup` is declared on `DataPacker`.
        model_backend: ``"hf"`` for the generic HF model/processor path, or ``"i4"``/``None`` for
            registered native packers.

    Returns:
        A `setup`-complete `DataPacker`.

    Raises:
        ValueError: the backend is unknown, the HF model is neither a declared VLM nor a recognized
            causal LM, or the native model has no registered packer.
    """
    if packer is not None:
        chosen = packer
    else:
        hf_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        chosen = _select_packer_cls(hf_config, model_backend=model_backend)()
    chosen.setup(name_or_path, trust_remote_code=trust_remote_code)
    return chosen


def _load_data_packer_factory(factory_path: str) -> Callable[[], object]:
    """Load one configured factory and report its config path at the failure boundary."""
    from cosmos_framework.utils.serialization import load_callable

    try:
        factory = load_callable(factory_path)
    except Exception as error:  # noqa: BLE001 -- preserve arbitrary import failures with config context
        raise ValueError(f"Unable to import DataPacker factory {factory_path!r}: {error}") from error
    if not callable(factory):
        raise TypeError(f"Configured DataPacker factory {factory_path!r} is not callable")
    return factory


def validate_data_packer_factories(config: DataPackerConfig) -> None:
    """Preflight all configured factory paths before distributed actors are placed."""
    factory_paths = [config.factory, config.teacher_model_factory]
    for factory_path in factory_paths:
        if factory_path:
            _load_data_packer_factory(factory_path)


def init_data_packer_from_factory(
    name_or_path: str,
    factory_path: str,
    trust_remote_code: bool = False,
    *,
    model_backend: str | None = None,
) -> DataPacker:
    """Build and set up a DataPacker from one model-owned factory.

    An empty factory path uses the built-in model-type registry. Configured factories are
    zero-argument callables so their import paths remain safe to serialize through Hydra and Ray.
    Calling this function in separate actors produces separate packer objects while preserving one
    model/tokenizer/processor contract.
    """
    if not factory_path:
        return init_data_packer(
            name_or_path,
            trust_remote_code=trust_remote_code,
            model_backend=model_backend,
        )

    factory = _load_data_packer_factory(factory_path)
    try:
        packer = factory()
    except Exception as error:
        raise RuntimeError(f"DataPacker factory {factory_path!r} failed: {error}") from error
    if not isinstance(packer, DataPacker):
        raise TypeError(f"DataPacker factory {factory_path!r} must return a DataPacker, got {type(packer).__name__}")
    return init_data_packer(name_or_path, packer=packer, trust_remote_code=trust_remote_code)


def __getattr__(name: str) -> Any:
    """Expose the generic HF VLM packer without importing it on native paths."""
    if name == "HFVLMDataPacker":
        from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

        return HFVLMDataPacker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDataPacker",
    "DataPacker",
    "DecoderOnlyLLMDataPacker",
    "HFVLMDataPacker",
    "Qwen3VLCrossTokenTeacherInput",
    "Qwen3VLDataPacker",
    "Qwen3VLPolicyInput",
    "Qwen3VLVideoDataPacker",
    "RLPolicyInput",
    "shape_policy_input",
    "register",
    "get_default_data_packer",
    "init_data_packer",
    "init_data_packer_from_factory",
    "validate_data_packer_factories",
]
