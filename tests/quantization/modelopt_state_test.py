# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU ModelOpt integration tests, including real FP8 compression and HF loading.

Run without the repository's training/inference conftest in a CPU environment:
pytest --noconftest -c /dev/null tests/quantization/modelopt_state_test.py
"""

import copy
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import pytest
import torch
from modelopt.torch.opt.conversion import ApplyModeError, apply_mode, restore_from_modelopt_state
from modelopt.torch.quantization.qtensor import QTensorWrapper
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration


@pytest.fixture
def export_modules(monkeypatch):
    # Load the actual export modules without quantization/__init__.py, whose
    # cookbook imports GPU calibration/training dependencies eagerly. ModelOpt,
    # Torch, Transformers and the code under test are all real, with no stubs.
    name = "_cosmos_quantization_cpu_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "cosmos_framework/quantization")]
    monkeypatch.setitem(sys.modules, name, package)
    helper = importlib.import_module(f"{name}.modelopt_state")
    exporter = importlib.import_module(f"{name}.export")
    yield helper, exporter
    for key in list(sys.modules):
        if key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key)


def tiny_reasoner(device="cpu"):
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 32,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [],
        },
    )
    with torch.device(device):
        return Qwen3VLForConditionalGeneration(config).bfloat16()


def compressed_dit(quantize_language_model=True):
    torch.manual_seed(0)
    model = torch.nn.Module()
    baseline = tiny_reasoner()
    for name, module in baseline.model.language_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for source, target in (("q_proj", "to_q"), ("k_proj", "to_k"), ("v_proj", "to_v"), ("o_proj", "to_out")):
            name = name.replace(f".self_attn.{source}", f".self_attn.{target}")
        parent = model
        *parents, leaf = name.split(".")
        for part in parents:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.add_module(leaf, copy.deepcopy(module))
    model.layers.get_submodule("0").self_attn.add_q_proj = torch.nn.Linear(16, 16, bias=False).bfloat16()
    model.lm_head = copy.deepcopy(baseline.lm_head)
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"].append({"quantizer_name": "*visual*", "enable": False})
    if not quantize_language_model:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "add_q_proj" not in name:
                config["quant_cfg"].append({"quantizer_name": f"{name}.*", "enable": False})
    model = apply_mode(model, [("quantize", config)])
    for module in model.modules():
        if hasattr(module, "input_quantizer") and module.input_quantizer.is_enabled:
            module.input_quantizer.amax = torch.tensor(3.5)
            module.weight_quantizer.amax = module.weight.detach().float().abs().max()
    mtq.compress(model)
    return model, mto.modelopt_state(model)


def reasoner_payload(helper, dit):
    payload = tiny_reasoner().state_dict()
    for name, value in dit.state_dict().items():
        module, _, leaf = name.rpartition(".")
        if "quantizer" in module:
            parent, _, role = module.rpartition(".")
            mapped = helper.dit_to_reasoner_module(parent)
            mapped = f"{mapped}.{role}" if mapped else None
        else:
            mapped = helper.dit_to_reasoner_module(module)
        if mapped:
            payload[f"{mapped}.{leaf}"] = value
    return payload


@pytest.mark.parametrize("quantize_language_model", [True, False])
def test_compressed_state_round_trip(export_modules, tmp_path, quantize_language_model):
    helper, _ = export_modules
    dit, component = compressed_dit(quantize_language_model)
    original = copy.deepcopy(component)
    tiny_reasoner().config.save_pretrained(tmp_path)
    state = helper.build_reasoner_modelopt_state(component, tmp_path)
    assert component == original  # metadata contains no tensor values
    path = tmp_path / "modelopt_state.pth"
    torch.save(state, path)
    state = torch.load(path, weights_only=False)
    restored = restore_from_modelopt_state(tiny_reasoner(), modelopt_state=state)
    restored.load_state_dict(reasoner_payload(helper, dit), strict=True)
    active = dict(state["modelopt_state_dict"])["real_quantize"]["metadata"]["q_tensor_state"]
    assert bool(active) == quantize_language_model
    for name, module in restored.named_modules():
        if not hasattr(module, "input_quantizer"):
            continue
        if name in active:
            assert isinstance(module.weight, QTensorWrapper)
            assert module.weight.get_state()["quantized_data.dtype"] == torch.float8_e4m3fn
            torch.testing.assert_close(module.input_quantizer.amax, torch.tensor(3.5))
            assert module.input_quantizer.amax.dtype == torch.float32
            assert module.weight_quantizer.amax.dtype == torch.float32
        else:
            assert not module.input_quantizer.is_enabled
            assert module.input_quantizer.amax is None
    assert not any(t.is_meta for t in list(restored.parameters()) + list(restored.buffers()))
    qs = dict(state["modelopt_state_dict"])["quantize"]["metadata"]["quantizer_state"]
    assert any(n.startswith("model.visual.") for n in qs)
    assert "lm_head.input_quantizer" in qs
    assert any(n.endswith("q_bmm_quantizer") for n in qs)
    assert not any("moe_gen" in n or "add_q_proj" in n for n in qs)


def test_huggingface_load_retains_calibrated_buffers(export_modules, tmp_path):
    helper, _ = export_modules
    dit, component = compressed_dit()
    tiny_reasoner().config.save_pretrained(tmp_path)
    state = helper.build_reasoner_modelopt_state(component, tmp_path)
    save_file(reasoner_payload(helper, dit), str(tmp_path / "model.safetensors"))
    torch.save(state, tmp_path / "modelopt_state.pth")
    mto.enable_huggingface_checkpointing()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        tmp_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    module = model.model.language_model.layers[0].self_attn.q_proj
    assert isinstance(module.weight, QTensorWrapper)
    torch.testing.assert_close(module.input_quantizer.amax, torch.tensor(3.5))
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))


def test_phantom_attention_metadata_fails_strict_restore(export_modules, tmp_path):
    helper, _ = export_modules
    _, component = compressed_dit()
    tiny_reasoner().config.save_pretrained(tmp_path)
    state = helper.build_reasoner_modelopt_state(component, tmp_path)
    qs = dict(state["modelopt_state_dict"])["quantize"]["metadata"]["quantizer_state"]
    # Strict restore rejects even a disabled extra entry: names must come from the graph.
    qs["model.language_model.layers.0.self_attn.extra_quantizer"] = copy.deepcopy(qs["lm_head.output_quantizer"])
    with pytest.raises(ApplyModeError, match="Unmatched keys"):
        restore_from_modelopt_state(tiny_reasoner(), modelopt_state=state)


@pytest.mark.parametrize("problem", ["missing_mode", "missing_module", "shape"])
def test_invalid_compressed_metadata_fails(export_modules, tmp_path, problem):
    helper, _ = export_modules
    _, component = compressed_dit()
    tiny_reasoner().config.save_pretrained(tmp_path)
    if problem == "missing_mode":
        component["modelopt_state_dict"] = component["modelopt_state_dict"][:1]
    else:
        metadata = dict(component["modelopt_state_dict"])["real_quantize"]["metadata"]["q_tensor_state"]
        name = next(iter(metadata))
        if problem == "missing_module":
            metadata["layers.999.self_attn.to_q"] = metadata.pop(name)
        else:
            metadata[name]["metadata"]["shape"] = torch.Size([999, 999])
    with pytest.raises(ValueError):
        helper.build_reasoner_modelopt_state(component, tmp_path)


def test_make_empty_reasoner_uses_local_config(export_modules, tmp_path):
    helper, _ = export_modules
    config = tiny_reasoner().config.to_dict()
    config["model_type"] = "cosmos3_omni"
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = helper.make_empty_reasoner(tmp_path)
    assert all(p.is_meta for p in model.parameters())
    assert set(dict(model.named_modules())) == set(dict(tiny_reasoner().named_modules()))
    config["model_type"] = "unknown"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="reasoner config"):
        helper.make_empty_reasoner(tmp_path)


def test_assembly_keeps_distinct_states_and_source_config(export_modules, tmp_path):
    helper, exporter = export_modules
    source, output, staging = (tmp_path / name for name in ("source", "output", "staging"))
    for directory in (source, output, staging):
        directory.mkdir()
    source_config = source / "hf_quant_config.json"
    source_config.write_text('{"source": true}')
    # Also cover a stale output symlink from an earlier assembly.
    (output / "hf_quant_config.json").symlink_to(source_config)
    (source / "modelopt_state.pth").write_bytes(b"source state")
    _, component = compressed_dit()
    torch.save(component, staging / "modelopt_state.pth")
    tiny_reasoner().config.save_pretrained(source)
    (output / "modelopt_state.pth").symlink_to(source / "modelopt_state.pth")
    (staging / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "modelopt", "runtime": {}}})
    )
    save_file({"weight": torch.zeros(1)}, str(staging / "model.safetensors"))
    exporter.assemble_output_dir(source, output, staging)
    assert source_config.read_text() == '{"source": true}'
    assert not (output / "hf_quant_config.json").is_symlink()
    assert "runtime" not in json.loads((output / "hf_quant_config.json").read_text())
    root_state = torch.load(output / "modelopt_state.pth", weights_only=False)
    assert root_state == helper.build_reasoner_modelopt_state(component, output)
    assert torch.load(output / "transformer/modelopt_state.pth", weights_only=False) == component
    assert not (output / "transformer/transformers_modelopt_state.pth").exists()
    assert (source / "modelopt_state.pth").read_bytes() == b"source state"
    assert json.loads((output / "model.safetensors.index.json").read_text())["weight_map"] == {
        "weight": "transformer/model.safetensors",
    }


@pytest.mark.skipif(not os.environ.get("COSMOS3_REASONER_STATE_SOT"), reason="external SOT path not supplied")
def test_matches_external_sot_metadata(export_modules, tmp_path):
    helper, _ = export_modules
    spec = importlib.util.spec_from_file_location("reasoner_state_sot", os.environ["COSMOS3_REASONER_STATE_SOT"])
    sot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sot)
    _, component = compressed_dit()
    tiny_reasoner().config.save_pretrained(tmp_path)
    ours = helper.build_reasoner_modelopt_state(component, tmp_path)
    expected_qs = sot.get_reasoner_quantizer_state(lambda: tiny_reasoner("meta"), component)
    expected = sot.remap(component, expected_qs)
    ours_modes = dict(ours["modelopt_state_dict"])
    expected_modes = dict(expected["modelopt_state_dict"])
    assert ours_modes["quantize"]["metadata"] == expected_modes["quantize"]["metadata"]
    assert ours_modes["real_quantize"] == expected_modes["real_quantize"]


def test_dynamic_activation_has_no_placeholder(export_modules, tmp_path):
    helper, _ = export_modules
    _, component = compressed_dit()
    config = dict(component["modelopt_state_dict"])["quantize"]["config"]
    config["quant_cfg"].append(
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None, "type": "dynamic"}}
    )
    tiny_reasoner().config.save_pretrained(tmp_path)
    state = helper.build_reasoner_modelopt_state(component, tmp_path)
    qs = dict(state["modelopt_state_dict"])["quantize"]["metadata"]["quantizer_state"]
    active = qs["model.language_model.layers.0.self_attn.q_proj.input_quantizer"]
    assert active["_dynamic"]
    assert "_amax" not in active["_pytorch_state_metadata"]["buffers"]


def test_root_writer_uses_export_config_and_local_architecture(export_modules, tmp_path):
    helper, exporter = export_modules
    dit, component = compressed_dit()
    writer = exporter.Fp8DiffusersExporter()
    dict(component["modelopt_state_dict"])["quantize"]["config"] = writer.build_quant_config_json()["modelopt_config"]
    tiny_reasoner().config.save_pretrained(tmp_path)
    writer.write_transformers_modelopt_state(component, tmp_path, tmp_path)
    state = torch.load(tmp_path / "modelopt_state.pth", weights_only=False)
    restored = restore_from_modelopt_state(tiny_reasoner(), modelopt_state=state)
    restored.load_state_dict(reasoner_payload(helper, dit), strict=True)
    torch.testing.assert_close(
        restored.model.language_model.layers[0].self_attn.q_proj.input_quantizer.amax, torch.tensor(3.5)
    )


@pytest.mark.parametrize("quantized", [True, False])
@pytest.mark.parametrize("flat", [True, False])
@pytest.mark.parametrize("cosmos_config", [True, False])
def test_cosmos_shim_loads_shards(export_modules, tmp_path, quantized, flat, cosmos_config):
    helper, exporter = export_modules
    model_path = Path(__file__).resolve().parents[2] / "packages/transformers-cosmos3/transformers_cosmos3/model.py"
    spec = importlib.util.spec_from_file_location("cosmos_transformers_shim", model_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    dit, component = compressed_dit()
    tiny_reasoner().config.save_pretrained(tmp_path)
    if cosmos_config:
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        config["model_type"] = "cosmos3_omni"
        # Real Nano labels its nested vision config qwen3_vl. Loading the root
        # with Qwen3VLConfig incorrectly selects this as the entire config.
        config["vision_config"]["model_type"] = "qwen3_vl"
        config_path.write_text(json.dumps(config))
    state = helper.build_reasoner_modelopt_state(component, tmp_path)
    transformer, vision = {}, {}
    payload = reasoner_payload(helper, dit) if quantized else tiny_reasoner().state_dict()
    for name, tensor in payload.items():
        if not flat:
            transformer[name] = tensor
            continue
        if name.startswith("model.visual."):
            vision[name.removeprefix("model.visual.")] = tensor
            continue
        name = name.removeprefix("model.language_model.")
        for source, target in (("q_proj", "to_q"), ("k_proj", "to_k"), ("v_proj", "to_v"), ("o_proj", "to_out")):
            name = name.replace(f".self_attn.{source}", f".self_attn.{target}")
        transformer[name] = tensor
    for component_name, tensors in (("transformer", transformer), ("vision_encoder", vision)):
        if not tensors:
            continue
        directory = tmp_path / component_name
        directory.mkdir()
        save_file(tensors, str(directory / "model.safetensors"))
    exporter._write_root_weight_index(tmp_path)
    if quantized:
        torch.save(state, tmp_path / "modelopt_state.pth")
    mto.enable_huggingface_checkpointing()
    model = shim.Cosmos3ForConditionalGeneration.from_pretrained(
        tmp_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    module = model.model.language_model.layers[0].self_attn.q_proj
    assert isinstance(module.weight, QTensorWrapper) == quantized
    if quantized:
        torch.testing.assert_close(module.input_quantizer.amax, torch.tensor(3.5))
    actual = model.state_dict()
    assert actual.keys() == payload.keys()
    for name, expected in payload.items():
        torch.testing.assert_close(actual[name].float(), expected.float(), rtol=0, atol=0)
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))


def test_diffusers_load_preserves_buffers_and_qtensor_wrappers(export_modules, tmp_path):
    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config

    _, exporter = export_modules
    helper_path = Path(__file__).resolve().parents[2] / "cosmos_framework/inference/modelopt_diffusers.py"
    spec = importlib.util.spec_from_file_location("cosmos_modelopt_diffusers", helper_path)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    compat.enable_modelopt_diffusers_checkpointing()
    compat.enable_modelopt_diffusers_checkpointing()  # idempotent

    class TinyDiffusersModel(ModelMixin, ConfigMixin):
        @register_to_config
        def __init__(self, width=16):
            super().__init__()
            self.projection = torch.nn.Linear(width, width, bias=False)

    model = TinyDiffusersModel().bfloat16()
    model = apply_mode(model, [("quantize", copy.deepcopy(mtq.FP8_DEFAULT_CFG))])
    model.projection.input_quantizer.amax = torch.tensor(3.5)
    model.projection.weight_quantizer.amax = model.projection.weight.detach().float().abs().max()
    mtq.compress(model)
    torch.save(mto.modelopt_state(model), tmp_path / "modelopt_state.pth")
    saved = model.state_dict()
    save_file(saved, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    config = dict(model.config)
    config["quantization_config"] = exporter.Fp8DiffusersExporter().build_quant_config_json()
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = TinyDiffusersModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16, local_files_only=True)
    assert isinstance(loaded.projection.weight, QTensorWrapper)
    for role in ["input", "weight"]:
        quantizer = getattr(loaded.projection, role + "_quantizer")
        assert "_amax" in quantizer._buffers
        assert "_amax" not in quantizer._parameters
        torch.testing.assert_close(quantizer.amax, saved[f"projection.{role}_quantizer._amax"])
    assert not any(t.is_meta for t in list(loaded.parameters()) + list(loaded.buffers()))
    assert loaded.state_dict().keys() == saved.keys()
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor.float(), saved[name].float(), rtol=0, atol=0)
