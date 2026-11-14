# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Candidate math/lifecycle controls; tiny CPU fixtures do not qualify Gemma2B."""

import copy
import hashlib
import json
from dataclasses import asdict
from types import MethodType

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from cosmos_framework.posttrain.tests.test_gemma_sdpa_layout import identity, tiny  # noqa: F401
from cosmos_framework.utils.generator import gemma_fixed_mma as operators
from cosmos_framework.utils.generator import gemma_sdpa_layout as profile


@pytest.fixture
def cpu_projection(monkeypatch):
    calls = []

    def reference(inputs, weight, bias=None):
        calls.append((inputs.shape, weight.shape, weight.data_ptr()))
        return F.linear(inputs.double(), weight.double(), None if bias is None else bias.double())

    monkeypatch.setattr(operators, "_project", reference)
    return calls


def test_gemma_offset_norm_and_native_rounding_boundary(tiny):  # noqa: F811
    norm = tiny.model.layers[0].input_layernorm
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.4, 0.7, norm.weight.numel()))
        norm.weight[0] = 2**-24  # 1+w rounds to exactly1 in native FP32.
        inputs = torch.randn(1, 33, norm.weight.numel())
        actual = operators.norm(norm, inputs)
        raw = inputs.numpy().astype(np.float64)
        normalized = (raw / np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True) + norm.eps)).astype(np.float32)
        expected = normalized * (np.float32(1) + norm.weight.numpy())
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert float((1.0 + norm.weight)[0]) == 1.0
        # Llama's direct-weight norm would incorrectly zero/nearly zero this
        # component. Preserve Gemma's architecture-specific offset weight.
        assert np.any(np.abs(actual.numpy()[..., 0]) > 0.1)
        shared = inputs[:, -1:]
        assert torch.equal(actual[:, -1:], operators.norm(norm, shared))


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_independent_factorized_reference_bias_and_parameter_ownership(tiny, cpu_projection, method, rslora):  # noqa: F811
    from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear

    torch.manual_seed(281)
    layer = PosttrainAdapterLinear(
        torch.nn.Linear(32, 16),
        AdapterComponentConfig(
            component="language",
            targets=["projection"],
            method=method,
            rank=3,
            alpha=5,
            use_rslora=rslora,
            dropout=0.0 if method == "pissa" else 0.2,
            initialization="svd" if method == "pissa" else "default",
        ),
    ).eval()
    with torch.no_grad():
        layer.lora_B.weight.add_(0.02)
        if method == "dora":
            layer.magnitude.mul_(1.1)
    owners = [
        (name, id(value), value.data_ptr(), value._version, value.clone()) for name, value in layer.named_parameters()
    ]
    initial = [(name, value.clone()) for name, value in layer.named_buffers()]
    x = torch.randn(1, 17, 32)
    with torch.no_grad():
        actual = operators.adapter(layer, x)
    # Independent NumPy expressions retain the separate low-rank/residual
    # algebra. They do not call the implementation's shared adapter helper.
    xx = x.numpy().astype(np.float64)
    w = layer.weight.detach().numpy().astype(np.float64)
    a = layer.lora_A.weight.detach().numpy().astype(np.float64)
    b = layer.lora_B.weight.detach().numpy().astype(np.float64)
    scale = 5 / (np.sqrt(3) if rslora else 3)
    expected = xx @ w.T + layer.bias.detach().numpy().astype(np.float64)
    delta = ((xx @ a.T) @ b.T) * scale
    if method == "lora":
        expected = expected + delta
    elif method == "pissa":
        initial_a, initial_b = [
            value.detach().numpy().astype(np.float64) for value in (layer.initial_A, layer.initial_B)
        ]
        expected = expected + (delta - ((xx @ initial_a.T) @ initial_b.T) * scale)
    else:
        direction = w + (b @ a) * scale
        ratio = layer.magnitude.detach().numpy().astype(np.float64) / np.sqrt(np.sum(direction * direction, axis=1))
        expected = expected + (ratio - 1) * (xx @ w.T) + ratio * delta
    np.testing.assert_allclose(actual.numpy(), expected.astype(np.float32), atol=1e-7, rtol=1e-7)
    assert len(cpu_projection) == {"lora": 3, "dora": 4, "pissa": 5}[method]
    assert cpu_projection[0][2] == layer.weight.data_ptr()
    assert any(shape[-1] == 32 and weight_shape == (3, 32) for shape, weight_shape, _ in cpu_projection)
    assert any(shape[-1] == 3 and weight_shape == (16, 3) for shape, weight_shape, _ in cpu_projection)
    for name, owner, pointer, version, snapshot in owners:
        value = layer.get_parameter(name)
        assert (id(value), value.data_ptr(), value._version) == (owner, pointer, version)
        assert torch.equal(value, snapshot)
    assert all(torch.equal(layer.get_buffer(name), value) for name, value in initial)


def test_real_tiny_gemma_profile_public_cache_ownership_and_restore(tiny, cpu_projection):  # noqa: F811
    model = tiny
    ids = torch.arange(66).remainder(model.config.vocab_size)[None]
    owned = profile._ownership(model)
    original = {name: module.forward for name, module in model.named_modules()}
    with torch.no_grad():
        native = model(ids, use_cache=False).logits
    names = profile.install(model, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    assert "lm_head" in names and "model.norm" in names and "model.layers.0.self_attn" in names
    assert profile.install(model, identity=identity(), profile=profile.FIXED_MMA_PROFILE) == names
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :33], use_cache=True).past_key_values
        snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
        cached = model(ids[:, 33:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        reused = model(ids[:, 33:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
    torch.testing.assert_close(full[:, 33:], cached, atol=2e-5, rtol=2e-5)
    assert torch.equal(cached, reused)
    assert all(
        torch.equal(layer.keys, k) and torch.equal(layer.values, v) for layer, (k, v) in zip(prefix.layers, snapshots)
    )
    assert profile._ownership(model) == owned
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model._cosmos_deployment_profile == profile.FIXED_MMA_PROFILE
    profile.restore(model)
    assert all(model.get_submodule(n).forward == forward for n, forward in original.items())
    with torch.no_grad():
        assert torch.equal(model(ids, use_cache=False).logits, native)
    assert cpu_projection  # Actual tiny model traversed the candidate projections.


@pytest.mark.parametrize("change", ["norm_epsilon", "linear_forward", "weight", "profile"])
def test_candidate_mutation_rejected_before_prefix_reuse(tiny, cpu_projection, change):  # noqa: F811
    profile.install(tiny, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        if change == "norm_epsilon":
            tiny.model.layers[0].input_layernorm.eps *= 2
        if change == "linear_forward":
            tiny.lm_head.forward = lambda *a, **k: None
        if change == "weight":
            tiny.lm_head.weight.add_(0.01)
        if change == "profile":
            tiny._cosmos_deployment_profile = profile.PROFILE
        with pytest.raises(ValueError):
            tiny(torch.tensor([[4]]), past_key_values=prefix)


def test_atomic_rejection_of_foreign_projection_and_norm(tiny, cpu_projection):  # noqa: F811
    attention = tiny.model.layers[0].self_attn
    original_attention = attention.forward
    tiny.lm_head.forward = MethodType(lambda self, x: x, tiny.lm_head)
    with pytest.raises(ValueError, match="unmodified native operators"):
        profile.install(tiny, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    assert attention.forward == original_attention and not hasattr(tiny, "_cosmos_gemma_sdpa_layout")


def test_profile_conflict_and_cache_identity_remain_separate(tiny, cpu_projection):  # noqa: F811
    profile.install(tiny, identity=identity())
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="conflicting"):
        profile.install(tiny, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    profile.restore(tiny)
    profile.install(tiny, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    with torch.no_grad(), pytest.raises(ValueError, match="prefix from this installation"):
        tiny(torch.tensor([[4]]), past_key_values=prefix)


def test_explicit_public_config_metadata_and_tag_admission(tmp_path):
    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata
    from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype
    from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig

    selected = profile.FIXED_MMA_PROFILE
    assert PortableAdapterConfig(str(tmp_path), selected).execution_profile == selected
    with pytest.raises(ValueError, match="separate branches"):
        PortableAdapterConfig(str(tmp_path), selected, cache_effective_weights=True)
    record = {"kind": "portable_adapter", "identity": identity(), "weights_sha256": "a" * 64}
    metadata = merged_export_metadata(
        "unused", identity(), artifact_record=record, output_dtype="float32", execution_profile=selected
    )
    assert metadata["deployment_precision"]["equivalence"] == profile.export_equivalence(selected)
    config = config_with_export_dtype({"model_type": "gemma"}, "float32", execution_profile=selected)
    config["rope_parameters"] = profile.ROPE_PARAMETERS
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(json.dumps(config))
    (merged / ".materialized").write_text(
        json.dumps({"identity": {"output_dtype": "float32", "export_metadata": metadata}})
    )
    assert deployment_profile(str(merged)) == selected
    metadata["deployment_precision"]["equivalence"] = profile.EXPORT_EQUIVALENCE
    (merged / ".materialized").write_text(
        json.dumps({"identity": {"output_dtype": "float32", "export_metadata": metadata}})
    )
    with pytest.raises(ValueError, match="committed FP32"):
        deployment_profile(str(merged))
    with pytest.raises(ValueError, match="float32"):
        merged_export_metadata(
            "unused", identity(), artifact_record=record, output_dtype="bfloat16", execution_profile=selected
        )


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_public_portable_and_tagged_merged_composition(tmp_path, monkeypatch, cpu_projection, method, rslora):
    """Real local artifacts/loaders, with only tiny CPU/operator admission patched.

    The CPU FP64 projection is a composition fixture, not the A100 MMA kernel.
    Whole-model numerical and performance qualification remains separate.
    """
    import transformers

    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
    from cosmos_framework.posttrain.checkpoint.adapters import read_adapter_artifact, save_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
    from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer
    from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig, load_portable_model

    torch.manual_seed(94086)
    base, artifact, merged = [tmp_path / name for name in ("base", "portable", "merged")]
    tokenizer = local_tokenizer()
    config = transformers.GemmaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
        max_position_embeddings=8192,
        hidden_act="gelu",
        attention_dropout=0.0,
    )
    transformers.GemmaForCausalLM(config).bfloat16().eval().save_pretrained(base)
    tokenizer.save_pretrained(base)
    settings = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language",
                targets=["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.v_proj"],
                rank=2,
                alpha=5,
                use_rslora=rslora,
                method=method,
                initialization="svd" if method == "pissa" else "default",
            )
        ]
    )
    source = build_hf_model(
        HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa", grad_ckpt=False),
        adapters=settings,
    ).net
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith("lora_B.weight"):
                parameter.add_(torch.randn_like(parameter) * 0.01)
    source.eval()
    save_adapter_artifact(source, str(artifact))
    declared, state = read_adapter_artifact(str(artifact))

    def snapshots():
        return {
            str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (base, artifact)
            for path in folder.rglob("*")
            if path.is_file()
        }

    before = snapshots()
    monkeypatch.setattr(profile, "ARCHITECTURE", {key: getattr(source.config, key) for key in profile.ARCHITECTURE})
    monkeypatch.setattr(profile, "BASE_SHA256", declared["base"]["sha256"])
    monkeypatch.setattr(profile, "PROCESSOR_SHA256", declared["processor"]["sha256"])
    monkeypatch.setattr(profile, "BUFFER_CONTRACT", copy.deepcopy(declared["model_contract"]))
    monkeypatch.setattr(profile, "_validate_device", lambda model: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    real_load = transformers.AutoModelForCausalLM.from_pretrained

    def cpu_load(*args, **kwargs):
        kwargs["device_map"] = {"": "cpu"}
        return real_load(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", cpu_load)
    selected = profile.FIXED_MMA_PROFILE
    loaded, _, provenance = load_portable_model(
        str(base),
        PortableAdapterConfig(str(artifact), selected),
        dtype="float32",
        device="cuda:0",
    )
    assert provenance["execution_profile"] == selected and provenance["identity"] == declared
    assert all(torch.equal(loaded.state_dict()[name], tensor) for name, tensor in state.items())
    source.float()
    profile.install(
        source, identity=declared, adapter_type=type(source.model.layers[0].self_attn.q_proj), profile=selected
    )
    ids = torch.tensor([[11, 13, 19, 21, 17]])
    owners = profile._ownership(loaded)
    with torch.no_grad():
        expected = source(ids, use_cache=False).logits
        actual = loaded(ids, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert profile._ownership(loaded) == owners
    assert export_merged_adapter(str(artifact), str(base), str(merged), execution_profile=selected) == "built"
    assert export_merged_adapter(str(artifact), str(base), str(merged), execution_profile=selected) == "skipped"
    assert deployment_profile(str(merged)) == selected
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_path, runtime.model_type, runtime.portable_adapter = str(merged), "gemma", None
    runtime.device, runtime.device_map, runtime.dtype = "cuda:0", "auto", "float32"
    runtime.attn_implementation, runtime.rank = "sdpa", 0
    runtime._load()
    assert runtime.adapter_deployment["execution_profile"] == selected
    assert runtime.model._cosmos_deployment_profile == selected
    assert runtime.model.get_input_embeddings().weight is runtime.model.get_output_embeddings().weight
    with torch.no_grad():
        full = runtime.model(ids, use_cache=False).logits
        prefix = runtime.model(ids[:, :3], use_cache=True).past_key_values
        cached = runtime.model(ids[:, 3:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        repeated = runtime.model(ids[:, 3:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
    torch.testing.assert_close(full, expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(cached, full[:, 3:], atol=2e-6, rtol=2e-5)
    assert torch.equal(cached, repeated) and prefix.get_seq_length() == 3
    assert snapshots() == before


@pytest.mark.parametrize("field", ["method", "rank", "alpha", "use_rslora", "dropout"])
def test_adapter_settings_mismatch_rejected_atomically(tiny, cpu_projection, field):  # noqa: F811
    from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear

    settings = AdapterComponentConfig(component="language", targets=["q_proj"], rank=2, alpha=5, use_rslora=True)
    layer = PosttrainAdapterLinear(tiny.model.layers[0].self_attn.q_proj, settings).eval()
    tiny.model.layers[0].self_attn.q_proj = layer
    declared = identity()
    declared["resolved"]["targets"][0]["settings"] = asdict(settings)
    changed = {"method": "dora", "rank": 3, "alpha": 6, "use_rslora": False, "dropout": 0.2}
    declared["resolved"]["targets"][0]["settings"][field] = changed[field]
    original = {name: module.forward for name, module in tiny.named_modules()}
    with pytest.raises(ValueError, match="settings differ"):
        profile.install(tiny, identity=declared, adapter_type=PosttrainAdapterLinear, profile=profile.FIXED_MMA_PROFILE)
    assert all(tiny.get_submodule(name).forward == forward for name, forward in original.items())
    assert not hasattr(tiny, "_cosmos_deployment_profile")


def test_accelerate_projection_wrapper_ownership(tiny, cpu_projection):  # noqa: F811
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    projection = tiny.model.layers[0].self_attn.q_proj
    add_hook_to_module(projection, AlignDevicesHook(execution_device="cpu"))
    native_outer, native_inner = projection.forward, projection._old_forward
    profile.install(tiny, identity=identity(), profile=profile.FIXED_MMA_PROFILE)
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        projection.forward = lambda *args, **kwargs: native_outer(*args, **kwargs) + 1
        with pytest.raises(ValueError, match="wrapper/hook changed"):
            tiny(torch.tensor([[4]]), past_key_values=prefix)
    with pytest.raises(ValueError, match="wrapper/hook changed"):
        profile.restore(tiny)
    projection.forward = native_outer
    profile.restore(tiny)
    assert projection.forward == native_outer and projection._old_forward == native_inner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A100 operator qualification is a separate installed run")
def test_a100_identical_rows_have_fixed_projection_and_norm_reduction():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A100 SM80 required")
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    torch.manual_seed(20261114)
    linear = torch.nn.Linear(128, 64, bias=True).cuda().eval()
    norm = GemmaRMSNorm(128).cuda().eval()
    row = torch.randn(1, 1, 128, device="cuda")
    with torch.no_grad():
        reference = operators.linear(linear, row)
        normalized = operators.norm(norm, row)
        for length in [2, 15, 16, 17, 31, 32, 33, 65]:
            inputs = torch.randn(1, length, 128, device="cuda")
            inputs[:, -1:] = row
            assert torch.equal(operators.linear(linear, inputs)[:, -1:], reference)
            assert torch.equal(operators.norm(norm, inputs)[:, -1:], normalized)
