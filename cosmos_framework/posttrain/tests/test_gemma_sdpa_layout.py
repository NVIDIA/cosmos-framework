# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract/operand tests. CUDA defect repair requires separate real evidence."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.utils.generator import gemma_sdpa_layout as profile


def identity():
    return {
        "format": "cosmos-posttrain-adapter-v1",
        "architecture": "gemma",
        "base_storage_dtype": "bfloat16",
        "base": {"sha256": profile.BASE_SHA256},
        "processor": {"sha256": profile.PROCESSOR_SHA256},
        "model_contract": copy.deepcopy(profile.BUFFER_CONTRACT),
        "dependencies": {"torch": "2.10.0", "transformers": "5.8.1"},
        "resolved": {"targets": [{"name": "model.layers.0.self_attn.q_proj", "settings": {"method": "lora"}}]},
    }


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("query_length", [1, 26, 31, 32, 33, 34, 63, 64, 65])
def test_only_masked_repeated_kv_layout_changes(monkeypatch, masked, query_length):
    from transformers.integrations import sdpa_attention as native

    torch.manual_seed(42)
    query = torch.randn(1, 8, query_length, 16).transpose(1, 2).contiguous().transpose(1, 2)
    key, value = [torch.randn(1, 1, query_length + 17, 16) for _ in range(2)]
    mask = torch.ones(1, 1, query_length, key.shape[2], dtype=torch.bool) if masked else None
    module = SimpleNamespace(num_key_value_groups=8, is_causal=True)
    owned = [x.clone() for x in (query, key, value)]
    calls = []

    def capture(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    native.sdpa_attention_forward(module, query, key, value, mask, dropout=0.0, scaling=0.25)
    profile.attention_interface(module, query, key, value, mask, dropout=0.0, scaling=0.25)
    old, new = calls
    assert old[0] is new[0] is query
    assert all(torch.equal(a, b) for a, b in zip(old[:3], new[:3]))
    assert old[3].keys() == new[3].keys()
    for name in old[3]:
        if name == "attn_mask":
            assert old[3][name] is new[3][name] is mask
        else:
            assert old[3][name] == new[3][name]
    if masked:
        assert old[1].stride(1) == old[2].stride(1) == 0
        assert new[1].is_contiguous() and new[2].is_contiguous()
        assert new[1].shape[1] == new[2].shape[1] == 8
    else:
        assert old[1] is new[1] is key and old[2] is new[2] is value
    assert all(torch.equal(a, b) for a, b in zip(owned, (query, key, value)))


@pytest.fixture
def tiny(monkeypatch):
    from transformers import GemmaConfig, GemmaForCausalLM

    config = GemmaConfig(
        vocab_size=83,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="gelu",
        attention_dropout=0.0,
        max_position_embeddings=8192,
    )
    config._attn_implementation = "sdpa"
    # Test-only fixture admission. Production admission retains the exact2B
    # architecture and oneA100 guard, tested separately below.
    monkeypatch.setattr(profile, "ARCHITECTURE", {k: getattr(config, k) for k in profile.ARCHITECTURE})
    from cosmos_framework.posttrain.model.hf_adapter_buffers import buffer_contract

    monkeypatch.setattr(profile, "BUFFER_CONTRACT", buffer_contract(config))
    monkeypatch.setattr(profile, "_validate_device", lambda model: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    torch.manual_seed(81)
    model = GemmaForCausalLM(config).eval()
    return model


def test_real_native_forward_cache_ownership_and_restore(tiny):
    from transformers.cache_utils import DynamicCache

    model = tiny
    ids = torch.arange(66).remainder(model.config.vocab_size)[None]
    before = profile._ownership(model)
    forwards = {n: m.forward for n, m in model.named_modules() if n.endswith("self_attn")}
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits
        reference_cache = model(ids[:, :33], use_cache=True).past_key_values
        native = model(ids[:, 33:], past_key_values=copy.deepcopy(reference_cache), use_cache=True).logits
    names = profile.install(model, identity=identity())
    assert set(names) == set(forwards)
    assert profile.install(model, identity=identity()) == names
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :33], use_cache=True).past_key_values
        assert type(prefix) is DynamicCache
        assert prefix.layers[0].keys.shape[1] == 1  # Persistent cache is never expanded.
        snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
        fork = copy.deepcopy(prefix)
        suffix = model(ids[:, 33:], past_key_values=fork, use_cache=True).logits
        again = model(ids[:, 33:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        with pytest.raises(ValueError, match="prefix from this installation"):
            model(ids[:, 33:], past_key_values=reference_cache)
    torch.testing.assert_close(full, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(suffix, native, atol=2e-6, rtol=2e-6)
    assert torch.equal(suffix, again)
    assert all(
        torch.equal(layer.keys, k) and torch.equal(layer.values, v) for layer, (k, v) in zip(prefix.layers, snapshots)
    )
    assert profile._ownership(model) == before
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    profile.restore(model)
    assert all(model.get_submodule(n).forward == f for n, f in forwards.items())
    assert not hasattr(model, "_cosmos_gemma_sdpa_layout")
    with torch.no_grad():
        assert torch.equal(model(ids, use_cache=False).logits, reference)


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_factorized_adapter_algebra_and_owned_tensors_unchanged(tiny, method):
    from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear

    base = tiny.model.layers[0].self_attn.q_proj
    layer = PosttrainAdapterLinear(
        base,
        AdapterComponentConfig(
            component="language",
            targets=["projection"],
            method=method,
            rank=2,
            alpha=5,
            use_rslora=True,
            initialization="svd" if method == "pissa" else "default",
            dropout=0.0,
        ),
    ).eval()
    tiny.model.layers[0].self_attn.q_proj = layer
    expected = identity()
    expected["resolved"]["targets"][0]["settings"]["method"] = method
    owned = profile._ownership(tiny)
    original = layer.forward
    profile.install(tiny, identity=expected, adapter_type=PosttrainAdapterLinear)
    with torch.no_grad():
        result = tiny(torch.tensor([[1, 2, 3]]))
    assert result.logits.isfinite().all()
    assert layer.forward == original
    assert profile._ownership(tiny) == owned
    profile.restore(tiny)


@pytest.mark.parametrize("change", ["weights", "scale", "forward", "profile", "config"])
def test_mutation_rejected_before_cached_reuse(tiny, change):
    from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear

    layer = PosttrainAdapterLinear(
        tiny.model.layers[0].self_attn.q_proj,
        AdapterComponentConfig(
            component="language",
            targets=["projection"],
            rank=2,
            dropout=0.0,
        ),
    ).eval()
    tiny.model.layers[0].self_attn.q_proj = layer
    profile.install(tiny, identity=identity(), adapter_type=PosttrainAdapterLinear)
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        if change == "weights":
            layer.lora_A.weight.add_(0.01)
        elif change == "scale":
            layer._lora_alpha *= 2
        elif change == "forward":
            tiny.model.layers[0].self_attn.forward = lambda *a, **k: None
        elif change == "profile":
            tiny._cosmos_deployment_profile = "foreign"
        else:
            tiny.config.head_dim += 1
        with pytest.raises(ValueError):
            tiny(torch.tensor([[4]]), past_key_values=prefix)


def test_guards_before_mutation_and_for_incompatible_execution(tiny):
    from transformers.cache_utils import StaticCache

    original = tiny.model.layers[0].self_attn.forward
    bad = identity()
    bad["base"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded Gemma"):
        profile.install(tiny, identity=bad)
    assert tiny.model.layers[0].self_attn.forward == original
    profile.install(tiny, identity=identity())
    ids = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="eager eval"):
        tiny(ids)
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError, match="autocast"):
            tiny(ids)
        with pytest.raises(ValueError, match="batch-one"):
            tiny(ids.expand(2, -1))
        with pytest.raises(ValueError, match="DynamicCache"):
            tiny(ids, past_key_values=StaticCache(config=tiny.config, max_cache_len=8))
        with pytest.raises(ValueError, match="8192"):
            tiny(torch.zeros(1, 8225, dtype=torch.long))
        tiny.train()
        with pytest.raises(RuntimeError, match="eager eval"):
            tiny(ids)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "gemma2"),
        ("base_storage_dtype", "float32"),
        ("format", "unknown"),
    ],
)
def test_identity_rejection(field, value):
    bad = identity()
    bad[field] = value
    with pytest.raises(ValueError, match="recorded Gemma"):
        profile.validate_identity(bad)


def test_export_profile_and_loader_metadata_are_bound(tmp_path):
    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata
    from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype
    from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig

    PortableAdapterConfig("adapter", profile.PROFILE)
    with pytest.raises(ValueError, match="cache"):
        PortableAdapterConfig("adapter", profile.PROFILE, cache_effective_weights=True)
    record = {"kind": "portable_adapter", "identity": identity(), "weights_sha256": "a" * 64}
    metadata = merged_export_metadata("unused", identity(), artifact_record=record, execution_profile=profile.PROFILE)
    config = config_with_export_dtype({"model_type": "gemma"}, "float32", execution_profile=profile.PROFILE)
    marker = {"identity": {"output_dtype": "float32", "export_metadata": metadata}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".materialized").write_text(json.dumps(marker))
    assert deployment_profile(str(tmp_path)) == profile.PROFILE
    changed = {**config, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "linear", "factor": 2.0}}
    (tmp_path / "config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="RoPE"):
        deployment_profile(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({**config, "sliding_window": 2}))
    with pytest.raises(ValueError, match="cache configuration"):
        deployment_profile(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(config))
    marker["identity"]["export_metadata"]["deployment_precision"]["equivalence"] = "legacy"
    (tmp_path / ".materialized").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="committed FP32"):
        deployment_profile(str(tmp_path))
    with pytest.raises(ValueError, match="float32"):
        merged_export_metadata(
            "unused", identity(), artifact_record=record, output_dtype="bfloat16", execution_profile=profile.PROFILE
        )
    with pytest.raises(ValueError, match="Gemma1"):
        config_with_export_dtype({"model_type": "gemma2"}, "float32", execution_profile=profile.PROFILE)


def test_public_portable_loader_checks_identity_before_injection(monkeypatch):
    from cosmos_framework.posttrain.checkpoint import adapters as artifacts
    from cosmos_framework.posttrain.model import adapters
    from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig, load_portable_model

    expected = identity()
    expected["media"] = {}
    monkeypatch.setattr(artifacts, "read_adapter_snapshot", lambda _: ({"identity": expected}, {}))
    monkeypatch.setattr(artifacts, "snapshot_identity", lambda _, weights: expected["base" if weights else "processor"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *a, **k: SimpleNamespace(
            model_type="gemma", hidden_size=2048, rope_parameters=copy.deepcopy(profile.ROPE_PARAMETERS)
        ),
    )
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: (object(), {}))
    monkeypatch.setattr(artifacts, "adapter_configuration", lambda _: object())
    monkeypatch.setattr(adapters, "resolve_targets", lambda *a, **k: expected["resolved"])
    monkeypatch.setattr(adapters, "install_adapters", lambda *a, **k: pytest.fail("injected before validation"))

    def verify(*args, **kwargs):
        assert kwargs["architecture"] == "gemma"
        raise RuntimeError("exact Gemma identity boundary")

    monkeypatch.setattr(artifacts, "make_adapter_identity", verify)
    with pytest.raises(RuntimeError, match="exact Gemma identity"):
        load_portable_model("base", PortableAdapterConfig("adapter", profile.PROFILE), dtype="float32", device="cuda:0")


def test_cpu_device_is_not_admitted():
    with pytest.raises(ValueError, match="one A100"):
        profile._validate_device(torch.nn.Linear(2, 2))


@pytest.mark.parametrize(
    "parameters",
    [
        {"rope_theta": 10000.0, "rope_type": "linear", "factor": 2.0},
        {"rope_theta": 20000.0, "rope_type": "default"},
        None,
    ],
)
def test_initial_rope_semantics_rejected_before_mutation(tiny, parameters):
    original = tiny.model.layers[0].self_attn.forward
    tiny.config.rope_parameters = parameters
    with pytest.raises(ValueError, match="RoPE"):
        profile.install(tiny, identity=identity())
    assert tiny.model.layers[0].self_attn.forward == original
    assert not hasattr(tiny, "_cosmos_deployment_profile")


def test_serialized_rope_legacy_defaults_resolve_without_input_mutation():
    config = {"model_type": "gemma", "rope_theta": 10000.0}
    before = copy.deepcopy(config)
    profile.validate_configuration(config)
    assert config == before
    with pytest.raises(ValueError, match="RoPE"):
        profile.validate_configuration({**config, "rope_scaling": {"rope_type": "linear", "factor": 2.0}})


@pytest.mark.parametrize("field", list(profile.CACHE_CONFIGURATION))
def test_initial_and_modified_cache_policy_rejected(tiny, field):
    value = ["sliding_attention"] * tiny.config.num_hidden_layers if field == "layer_types" else 2
    setattr(tiny.config, field, value)
    original = tiny.model.layers[0].self_attn.forward
    with pytest.raises(ValueError, match="cache configuration"):
        profile.install(tiny, identity=identity())
    assert tiny.model.layers[0].self_attn.forward == original
    setattr(tiny.config, field, None)
    profile.install(tiny, identity=identity())
    setattr(tiny.config, field, value)
    with torch.no_grad(), pytest.raises(ValueError, match="configuration changed"):
        tiny(torch.tensor([[1, 2]]), use_cache=True)


@pytest.mark.parametrize("change", ["sliding", "lazy_factory", "partial_layers"])
def test_foreign_dynamic_cache_structure_rejected_before_mutation(tiny, change):
    from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

    profile.install(tiny, identity=identity())
    config = copy.deepcopy(tiny.config)
    if change == "sliding":
        config.sliding_window = 2
    cache = DynamicCache(config=config)
    if change == "lazy_factory":
        cache = DynamicCache()
        cache.layer_class_to_replicate = DynamicSlidingWindowLayer
    elif change == "partial_layers":
        cache.layers.pop()
    layers = list(cache.layers)
    with torch.no_grad(), pytest.raises(ValueError, match="DynamicCache"):
        tiny(torch.tensor([[1, 2, 3, 4]]), past_key_values=cache, use_cache=True)
    assert cache.layers == layers and cache.get_seq_length() == 0


def test_ordinary_lazy_dynamic_cache_is_supported(tiny):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    profile.install(tiny, identity=identity())
    cache = DynamicCache()
    with torch.no_grad():
        output = tiny(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
    assert output.past_key_values is cache and cache.get_seq_length() == 3
    assert len(cache.layers) == tiny.config.num_hidden_layers
    assert all(type(layer) is DynamicLayer for layer in cache.layers)


def test_foreign_attention_rejected_atomically(tiny):
    first = tiny.model.layers[0].self_attn.forward
    tiny.model.layers[1].self_attn.forward = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="unmodified stock attention"):
        profile.install(tiny, identity=identity())
    assert tiny.model.layers[0].self_attn.forward == first
    assert not hasattr(tiny, "_cosmos_deployment_profile")


def test_native_input_gradient_hook_preserved(tiny):
    tiny.enable_input_require_grads()
    module = tiny.get_input_embeddings()
    hooks = dict(module._forward_hooks)
    profile.install(tiny, identity=identity())
    with torch.no_grad():
        assert tiny(torch.tensor([[1, 2]])).logits.isfinite().all()
    profile.restore(tiny)
    assert dict(module._forward_hooks) == hooks


@pytest.mark.parametrize("change", ["outer_forward", "hook", "offload"])
def test_accelerate_wrapper_ownership_cannot_bypass_profile(tiny, change):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    attention = tiny.model.layers[0].self_attn
    hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    add_hook_to_module(attention, hook)
    original_inner = attention._old_forward
    outer = attention.forward
    profile.install(tiny, identity=identity())
    installed_inner = attention._old_forward
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2]]), use_cache=True).past_key_values
        if change == "outer_forward":
            attention.forward = lambda *args, **kwargs: pytest.fail("foreign wrapper executed")
        elif change == "hook":
            attention._hf_hook = AlignDevicesHook(execution_device=torch.device("cpu"))
        else:
            hook.offload = True
        with pytest.raises(ValueError, match="wrapper/hook changed"):
            tiny(torch.tensor([[3]]), past_key_values=prefix)
        with pytest.raises(ValueError, match="wrapper/hook changed"):
            profile.install(tiny, identity=identity())
        with pytest.raises(ValueError, match="wrapper/hook changed"):
            profile.restore(tiny)
    assert attention._old_forward == installed_inner  # No partial restoration.
    attention.forward, attention._hf_hook, hook.offload = outer, hook, False
    profile.restore(tiny)
    assert attention._old_forward == original_inner and attention.forward == outer and attention._hf_hook is hook


@pytest.mark.parametrize("change", ["processor", "method", "execution", "model_execution", "dependencies"])
def test_other_incompatible_artifact_contracts_rejected(change):
    bad = identity()
    if change == "processor":
        bad["processor"]["sha256"] = "0" * 64
    elif change in ("method", "execution"):
        bad["resolved"]["targets"][0]["settings"][change] = "unsupported"
    elif change == "model_execution":
        bad["resolved"][change] = "merged_weight_bf16_v1"
    else:
        bad["dependencies"]["transformers"] = "4.57.1"
    with pytest.raises(ValueError, match="recorded Gemma"):
        profile.validate_identity(bad)


def test_public_text_preparation_uses_native_template_without_visual_reader():
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    calls = []

    class Processor:
        def apply_chat_template(self, rows, **kwargs):
            calls.append((rows, kwargs))
            return "native text"

        def __call__(self, **kwargs):
            return kwargs

    runtime = SimpleNamespace(processor=Processor(), model_type="gemma")
    runtime._prepare_smollm_text = lambda conversations: CosmosFrameworkRuntime._prepare_smollm_text(
        runtime, conversations
    )
    value = CosmosFrameworkRuntime._prepare_transformers_inputs(runtime, [{"role": "user", "content": "question"}])
    assert value == {"text": ["native text"], "padding": True, "return_tensors": "pt", "add_special_tokens": False}
    assert calls == [([{"role": "user", "content": "question"}], {"tokenize": False, "add_generation_prompt": True})]
    with pytest.raises(ValueError, match="text-only"):
        CosmosFrameworkRuntime._prepare_transformers_inputs(runtime, [{"role": "user", "content": [{"type": "video"}]}])
