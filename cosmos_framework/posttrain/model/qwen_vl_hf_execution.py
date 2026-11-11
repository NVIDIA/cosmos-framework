# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit TF5 ordered-modality execution without effective-norm conversion.

This closes previously inadmissible SFT paths; it never reinterprets the TF4
combined-vision default or the independently qualified stock-control arithmetic.
Both factorized and dense PiSSA components may be combined under this profile.
"""

from importlib.metadata import version

PROFILE = "qwen_vl_hf_ordered_v1"
_TYPES = {
    "qwen2_vl": ("transformers.models.qwen2_vl.modeling_qwen2_vl", "Qwen2VLForConditionalGeneration"),
    "qwen2_5_vl": ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    "qwen3_vl": ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLForConditionalGeneration"),
}


def _validate_owner(net, family):
    if family not in _TYPES or (type(net).__module__, type(net).__name__) != _TYPES[family]:
        raise ValueError(f"{PROFILE} requires an exact stock HF Qwen2/2.5/3-VL model")
    if version("transformers") != "5.8.1" or version("torch").split("+")[0] != "2.10.0":
        raise ValueError(f"{PROFILE} requires Transformers5.8.1 and PyTorch2.10.0")
    configs = [net.config, net.config.text_config, net.config.vision_config]
    configs.extend(getattr(module, "config", None) for module in net.modules())
    if any(getattr(section, "_attn_implementation", "sdpa") != "sdpa" for section in configs if section is not None):
        raise ValueError(f"{PROFILE} requires SDPA")
    return


def validate(net, config, family):
    _validate_owner(net, family)
    if config.model_execution != PROFILE:
        raise ValueError("ordered HF execution must be explicit in adapter identity")


def install(net):
    _validate_owner(net, net.config.model_type)
    if net.config.model_type == "qwen3_vl":
        from cosmos_framework.posttrain.model.qwen3_vl_hf import patch_ordered_forward
    else:
        from cosmos_framework.posttrain.model.qwen2_vl_hf import patch_ordered_forward
    patch_ordered_forward(net.model)
    net._posttrain_model_execution = PROFILE


def canonical_weight_name(family, name):
    """Map the public Qwen2/2.5 checkpoint dialect to the TF5 module owners.

    Current HF saves already use the target namespace. This mapping is exact
    and does not reinterpret tensors or change their layout.
    """
    if family not in ("qwen2_vl", "qwen2_5_vl"):
        return name
    if name.startswith("visual."):
        return "model." + name
    if name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.language_model." + name[len("model.") :]
    return name
