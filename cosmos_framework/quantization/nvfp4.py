# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ModelOpt NVFP4 calibration configuration and packed Diffusers export."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import modelopt
import modelopt.torch.quantization as mtq
import torch

from .nvfp4_recipe import module_precision, runtime_module_name
from .tensor_formats import FP8_DTYPES, SCALE_SUFFIXES


def build_nvfp4_quant_config(model, num_layers, fp8_last_k_layers, quantize_reasoner, algo):
    if algo != "max":
        raise ValueError("NVFP4 cookbook supports max calibration only")
    # Start disabled; only the decoder linears selected by the recipe are enabled.
    config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    config["quant_cfg"] = [{"quantizer_name": "*", "enable": False}]
    specs = {
        "NVFP4": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
        "FP8": {"num_bits": (4, 3), "axis": None, "block_sizes": None},
    }
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        precision = module_precision(name, num_layers, fp8_last_k_layers, quantize_reasoner)
        if precision == "BF16":
            continue
        for role in ("input_quantizer", "weight_quantizer"):
            config["quant_cfg"].append(
                {"quantizer_name": f"{name}.{role}", "cfg": copy.deepcopy(specs[precision]), "enable": True}
            )
    return config


def overlay_nvfp4_weights(bf16, quantized):
    """Overlay packed W4 and ordinary W8 weights, validating their distinct shapes."""
    merged = dict(bf16)
    layers = {}
    for key, weight in quantized.items():
        if not key.endswith(".weight") or weight.dtype not in (FP8_DTYPES | {torch.uint8}):
            continue
        if key not in bf16:
            raise KeyError(f"Quantized weight absent from source: {key}")
        source = bf16[key]
        packed = weight.dtype == torch.uint8
        expected = (*source.shape[:-1], source.shape[-1] // 2) if packed else source.shape
        if tuple(weight.shape) != tuple(expected):
            raise ValueError(f"Unexpected packed/FP8 shape for {key}: {weight.shape}, expected {expected}")
        module = key.removesuffix(".weight")
        required = (".weight_scale", ".input_scale", ".weight_scale_2") if packed else (".weight_scale", ".input_scale")
        for suffix in required:
            if module + suffix not in quantized:
                raise ValueError(f"Missing {module + suffix}")
        merged[key] = weight
        for suffix in SCALE_SUFFIXES:
            if module + suffix in quantized:
                merged[module + suffix] = quantized[module + suffix]
        layers[runtime_module_name(module)] = {"quant_algo": "NVFP4" if packed else "FP8"}
        if packed:
            layers[runtime_module_name(module)]["group_size"] = 16
    if not layers:
        raise ValueError("No quantized decoder weights were exported")
    return merged, layers


def export_nvfp4_transformer(
    mdl,
    transformer_export_dir: Path,
    src_transformer_dir: Path,
    dtype=torch.bfloat16,
    mixed_precision_steps=3,
    modelopt_config=None,
):
    """Write packed inference tensors and per-linear precision for vLLM-Omni.

    Unlike the FP8 exporter, this does not produce ModelOpt graph restore state.
    """
    from .checkpoint_io import save_sharded_safetensors
    from .export import add_diffusion_step_policy, collect_export_weights

    bf16, quantized = collect_export_weights(mdl.net, dtype, src_transformer_dir)
    merged, layers = overlay_nvfp4_weights(bf16, quantized)
    transformer_export_dir = Path(transformer_export_dir)
    transformer_export_dir.mkdir(parents=True, exist_ok=False)
    save_sharded_safetensors(merged, transformer_export_dir)
    config = json.loads((src_transformer_dir / "config.json").read_text())
    # Always use per-linear dispatch: even a uniform NVFP4 generator can have
    # a BF16 reasoner and dense projections absent from quantized_layers.
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "exclude_modules": [],
        "quantized_layers": layers,
        "producer": {"name": "modelopt", "version": modelopt.__version__},
    }
    add_diffusion_step_policy(config, mixed_precision_steps)
    config["quantization_config"]["runtime"]["diffusion_step_policy"]["reasoner"] = "native"
    (transformer_export_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n")
