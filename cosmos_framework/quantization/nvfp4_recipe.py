# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Precision selection shared by NVFP4 calibration and checkpoint metadata."""

from __future__ import annotations

import re

_LAYER = re.compile(r"^net\.language_model\.model\.layers\.(\d+)\.(.+)$")
_GEN_ATTN = {"q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen", "qkv_proj_moe_gen"}
_UND_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
_MLP = {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}


def validate_recipe(quant_format, fp8_last_k_layers, mixed_precision_steps, num_layers=None):
    if quant_format not in ("fp8", "nvfp4"):
        raise ValueError("quant_format must be fp8 or nvfp4")
    for name, value in (("fp8_last_k_layers", fp8_last_k_layers), ("mixed_precision_steps", mixed_precision_steps)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if quant_format != "nvfp4" and fp8_last_k_layers:
        raise ValueError("fp8_last_k_layers is only supported for nvfp4")
    if num_layers is not None and (num_layers <= 0 or fp8_last_k_layers > num_layers):
        raise ValueError(f"fp8_last_k_layers must be between 0 and {num_layers}")


def module_precision(name: str, num_layers: int, fp8_last_k_layers: int, quantize_reasoner: bool) -> str:
    """Select decoder linears only; modality towers and projections stay BF16."""
    match = _LAYER.fullmatch(name)
    if match is None:
        return "BF16"
    index, path = int(match[1]), match[2]
    if not 0 <= index < num_layers:
        raise ValueError(f"Decoder index outside model: {name}")
    branch, _, projection = path.partition(".")
    generation = (branch == "self_attn" and projection in _GEN_ATTN) or (branch == "mlp_moe_gen" and projection in _MLP)
    if generation:
        return "FP8" if index >= num_layers - fp8_last_k_layers else "NVFP4"
    reasoner = (branch == "self_attn" and projection in _UND_ATTN) or (branch == "mlp" and projection in _MLP)
    return "FP8" if reasoner and quantize_reasoner else "BF16"


def runtime_module_name(diffusers_name: str) -> str:
    """Translate source Diffusers projection names to Cosmos3 serving FQNs."""
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", diffusers_name)
    if match is None:
        raise ValueError(f"Not a Cosmos3 decoder linear: {diffusers_name}")
    index, path = match.groups()
    for source, target in (
        ("self_attn.add_q_proj", "cross_attention.to_q"),
        ("self_attn.add_k_proj", "cross_attention.to_k"),
        ("self_attn.add_v_proj", "cross_attention.to_v"),
        ("self_attn.to_add_out", "cross_attention.to_out"),
    ):
        if path == source:
            return f"gen_layers.{index}.{target}"
    if path.startswith("mlp_moe_gen."):
        return f"gen_layers.{index}.mlp.{path.removeprefix('mlp_moe_gen.')}"
    return f"language_model.layers.{index}.{path}"
