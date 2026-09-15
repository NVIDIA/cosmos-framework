# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build root ModelOpt metadata from the actual Qwen3-VL reasoner graph.

Follows pipeline_checkpoints/build_reasoner_modelopt_state.py in c3-quantization:
mode 0 comes from a meta-device reasoner; mode 1 comes from the compressed DiT.
Only metadata is built here. Calibrated buffers and FP8 weights stay in safetensors.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.quantization.conversion import quantizer_state
from modelopt.torch.quantization.nn import TensorQuantizer


def dit_to_reasoner_module(name: str) -> str | None:
    """Map shared DiT modules, excluding every generation-only branch."""
    if name == "lm_head":
        return name
    if not name.startswith("layers.") or any(
        token in name for token in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "_moe_gen")
    ):
        return None
    for source, target in (("to_q", "q_proj"), ("to_k", "k_proj"), ("to_v", "v_proj"), ("to_out", "o_proj")):
        name = name.replace(f".self_attn.{source}", f".self_attn.{target}")
    return f"model.language_model.{name}"


def make_empty_reasoner(checkpoint_dir: Path):
    """Build the architecture used by transformers-cosmos3 without loading weights."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config_path = Path(checkpoint_dir) / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") not in {"cosmos3_omni", "qwen3_vl"} or "text_config" not in config:
        raise ValueError(f"Expected a Cosmos3/Qwen3-VL reasoner config at {config_path}.")
    # The local Cosmos3ForConditionalGeneration shim inherits this exact graph.
    # Strip serving metadata: no quantization/load hook is needed to build it.
    config.pop("quantization_config", None)
    with torch.device("meta"):
        return Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(config))


def build_reasoner_modelopt_state(component_state: dict, checkpoint_dir: Path) -> dict:
    """Capture strict quantizer metadata and remap compressed-weight metadata.

    ``checkpoint_dir`` is the directory containing the reasoner checkpoint.
    """
    state = copy.deepcopy(component_state)
    modes = dict(state["modelopt_state_dict"])
    if "quantize" not in modes or "real_quantize" not in modes:
        raise ValueError("Compressed component state must contain quantize and real_quantize modes.")
    real_meta = modes["real_quantize"]["metadata"]
    for key in ("real_quantizer_state", "q_tensor_state"):
        real_meta[key] = {
            mapped: value
            for name, value in real_meta[key].items()
            if (mapped := dit_to_reasoner_module(name)) is not None
        }

    active_modules = set(real_meta["q_tensor_state"])
    model = make_empty_reasoner(checkpoint_dir)
    modules = dict(model.named_modules())
    missing = active_modules - modules.keys()
    if missing:
        raise ValueError(f"Compressed weights have no reasoner module: {sorted(missing)[:3]}.")
    for name in active_modules:
        weight = modules[name].weight
        tensor_state = real_meta["q_tensor_state"][name]
        if weight.shape != tensor_state["metadata"]["shape"]:
            raise ValueError(f"Reasoner weight shape does not match compressed metadata for {name}.")

    config = modes["quantize"]["config"]
    # The DiT's config uses flat names. Explicitly disable reasoner linears that
    # have no FP8 weight, including selectively unquantized language-model layers.
    for name, module in modules.items():
        if isinstance(module, torch.nn.Linear) and name not in active_modules:
            config["quant_cfg"].append({"quantizer_name": f"{name}.*", "enable": False})
    model = apply_mode(model, [("quantize", config)])
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not name.endswith(".input_quantizer"):
            continue
        if module.is_enabled and not module._dynamic and "_amax" not in module._buffers:
            # Restore must allocate the destination for the calibrated static
            # activation amax stored in the checkpoint. No orphan buffers on
            # disabled/dynamic quantizers; weight buffers come from mode 1.
            module.register_buffer("_amax", torch.zeros((), dtype=torch.float32, device="meta"))
    qs = quantizer_state(model)
    if not qs:
        raise ValueError("ModelOpt did not create any reasoner quantizers.")
    modes["quantize"]["metadata"]["quantizer_state"] = qs
    return state
