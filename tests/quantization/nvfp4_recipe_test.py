# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU recipe and serialization contracts; no checkpoint download or GPU needed."""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    name = "_nvfp4_recipe_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "cosmos_framework/quantization")]
    monkeypatch.setitem(sys.modules, name, package)
    recipe = importlib.import_module(f"{name}.nvfp4_recipe")
    exporter = importlib.import_module(f"{name}.nvfp4")
    yield recipe, exporter
    for key in list(sys.modules):
        if key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key)


def name(index, projection="self_attn.q_proj_moe_gen"):
    return f"net.language_model.model.layers.{index}.{projection}"


@pytest.mark.parametrize("depth,tail", [(36, 6), (64, 0), (36, 0), (36, 36)])
def test_tail_and_reasoner_are_independent(modules, depth, tail):
    recipe, _ = modules
    for index in range(depth):
        expected = "FP8" if index >= depth - tail else "NVFP4"
        for projection in (
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ):
            for reasoner in (True, False):
                assert recipe.module_precision(name(index, projection), depth, tail, reasoner) == expected
        assert recipe.module_precision(name(index, "mlp.up_proj"), depth, tail, True) == "FP8"
        assert recipe.module_precision(name(index, "mlp.up_proj"), depth, tail, False) == "BF16"
    for excluded in ("net.vae2llm", "net.language_model.visual.proj", name(0, "self_attn.q_norm_moe_gen")):
        assert recipe.module_precision(excluded, depth, tail, True) == "BF16"


@pytest.mark.parametrize("tail", [-1, 37, True, 1.5])
def test_invalid_tail(modules, tail):
    recipe, _ = modules
    with pytest.raises(ValueError):
        recipe.validate_recipe("nvfp4", tail, 3, 36)


def test_recipe_rejects_fp8_tail_on_fp8_and_invalid_steps(modules):
    recipe, _ = modules
    with pytest.raises(ValueError):
        recipe.validate_recipe("fp8", 6, 3)
    with pytest.raises(ValueError):
        recipe.validate_recipe("nvfp4", 6, -1)
    recipe.validate_recipe("nvfp4", 6, 0, 36)


@pytest.mark.parametrize(
    "source,target",
    [
        ("layers.35.self_attn.add_q_proj", "gen_layers.35.cross_attention.to_q"),
        ("layers.35.self_attn.to_add_out", "gen_layers.35.cross_attention.to_out"),
        ("layers.35.mlp_moe_gen.down_proj", "gen_layers.35.mlp.down_proj"),
        ("layers.0.self_attn.to_q", "language_model.layers.0.self_attn.to_q"),
        ("layers.0.mlp.up_proj", "language_model.layers.0.mlp.up_proj"),
    ],
)
def test_runtime_keys(modules, source, target):
    assert modules[0].runtime_module_name(source) == target


def test_overlay_keeps_packed_weights_scales_and_dense_components(modules):
    _, exporter = modules
    nv = "layers.0.self_attn.add_q_proj"
    fp = "layers.1.self_attn.add_q_proj"
    dense = torch.ones(16, 32, dtype=torch.bfloat16)
    bf16 = {nv + ".weight": dense, fp + ".weight": dense, "proj_out.weight": dense}
    quantized = {
        nv + ".weight": torch.zeros(16, 16, dtype=torch.uint8),
        nv + ".weight_scale": torch.ones(16, 2, dtype=torch.float8_e4m3fn),
        nv + ".weight_scale_2": torch.tensor(0.5),
        nv + ".input_scale": torch.tensor(0.25),
        fp + ".weight": torch.zeros(16, 32, dtype=torch.float8_e4m3fn),
        fp + ".weight_scale": torch.tensor(0.5),
        fp + ".input_scale": torch.tensor(0.25),
    }
    merged, layers = exporter.overlay_nvfp4_weights(bf16, quantized)
    assert merged["proj_out.weight"] is dense
    assert all(merged[key] is value for key, value in quantized.items())
    assert layers["gen_layers.0.cross_attention.to_q"] == {"quant_algo": "NVFP4", "group_size": 16}
    assert layers["gen_layers.1.cross_attention.to_q"] == {"quant_algo": "FP8"}
    del quantized[nv + ".weight_scale_2"]
    with pytest.raises(ValueError, match="weight_scale_2"):
        exporter.overlay_nvfp4_weights(bf16, quantized)


def test_overlay_rejects_unpacked_uint8(modules):
    key = "layers.0.self_attn.add_q_proj.weight"
    with pytest.raises(ValueError, match="shape"):
        modules[1].overlay_nvfp4_weights(
            {key: torch.ones(16, 32, dtype=torch.bfloat16)},
            {key: torch.ones(16, 32, dtype=torch.uint8)},
        )


def test_modelopt_applies_mixed_config_to_live_quantizers(modules):
    import modelopt.torch.quantization as mtq

    model = torch.nn.Module()
    model.net = torch.nn.Module()
    model.net.language_model = torch.nn.Module()
    model.net.language_model.model = torch.nn.Module()
    model.net.language_model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for projection in ("qkv_proj_moe_gen", "qkv_proj"):
            setattr(layer.self_attn, projection, torch.nn.Linear(32, 32, bias=False))
        model.net.language_model.model.layers.append(layer)
    model.net.vae2llm = torch.nn.Linear(32, 32)
    config = modules[1].build_nvfp4_quant_config(model, 2, 1, True, "max")
    # An empty loop tests ModelOpt conversion/selection without executing an
    # NVFP4 kernel, which needs a Blackwell GPU. No calibrated weights are claimed.
    mtq.quantize(model, config, lambda _: None)
    layers = model.net.language_model.model.layers
    assert layers[0].self_attn.qkv_proj_moe_gen.weight_quantizer.num_bits == (2, 1)
    assert layers[1].self_attn.qkv_proj_moe_gen.weight_quantizer.num_bits == (4, 3)
    assert all(layer.self_attn.qkv_proj.input_quantizer.num_bits == (4, 3) for layer in layers)
    assert not model.net.vae2llm.weight_quantizer.is_enabled
