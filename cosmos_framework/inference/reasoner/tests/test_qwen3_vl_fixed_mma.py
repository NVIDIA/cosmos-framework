# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real tiny-model CPU composition; these controls confer no 2B/A100 credit."""

import copy
import json
import os
import subprocess
import sys
from inspect import signature
from types import MethodType, SimpleNamespace

import pytest
import torch
import transformers
from torch.nn import functional as F

from cosmos_framework.inference.reasoner import precision
from cosmos_framework.inference.reasoner import qwen3_vl_fixed_mma as profile
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapter_bf16_execution import _effective_norm as extra_norm_forward
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
from cosmos_framework.posttrain.model.qwen3_vl_hf import _forward as assembly_forward
from cosmos_framework.posttrain.tests.test_portable_qwen_vl import artifact, data, snapshot
from cosmos_framework.utils.generator import portable_adapters
from cosmos_framework.utils.generator import qwen3_vl_projection_contract as contract
from cosmos_framework.utils.hf_snapshot import snapshot_identity

pytestmark = pytest.mark.skipif(transformers.__version__ != "5.8.1", reason="exact TF5.8.1 profile")
target_device_guard = profile._target_device
environment_guard = precision.validate_deployment_environment


@pytest.fixture(autouse=True)
def cpu_hardware(monkeypatch, request):
    previous = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if request.function is not test_installed_fixed_projection_kernel_matches_same_operand_fp64:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
            monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
            monkeypatch.setattr(profile, "_device", lambda model: None)
            monkeypatch.setattr(profile, "_target_device", lambda device: None)
            monkeypatch.setattr(portable_adapters, "_validate_device", lambda device: None)
            monkeypatch.setattr(precision, "validate_deployment_environment", lambda *a, **k: None)
            # CPU composition preserves branch/cast/order with a native callback.
            # The actual kernel test keeps every hardware and projector callable.
            monkeypatch.setattr(profile, "_project", F.linear)
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("exception", [False, True])
def test_real_kernel_fixture_preserves_hardware_projector_and_precision(monkeypatch, exception):
    originals = [
        (torch.cuda, "is_available"),
        (torch.cuda, "is_current_stream_capturing"),
        (profile, "_device"),
        (profile, "_target_device"),
        (portable_adapters, "_validate_device"),
        (precision, "validate_deployment_environment"),
        (profile, "_project"),
    ]
    # Synthetic native-callable sentinels test the actual autouse-fixture boundary
    # without claiming that a CPU invocation executed a CUDA operator.
    for obj, name in originals:
        monkeypatch.setattr(obj, name, lambda *a, **k: "unchanged")
    before = [(obj, name, getattr(obj, name)) for obj, name in originals]
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    previous = torch.get_float32_matmul_precision()
    request = SimpleNamespace(function=test_installed_fixed_projection_kernel_matches_same_operand_fp64)
    with pytest.MonkeyPatch.context() as isolated_patch:
        fixture = cpu_hardware.__wrapped__(isolated_patch, request)
        next(fixture)
        assert all(getattr(obj, name) is value for obj, name, value in before)
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        if exception:
            with pytest.raises(RuntimeError, match="synthetic kernel failure"):
                fixture.throw(RuntimeError("synthetic kernel failure"))
        else:
            with pytest.raises(StopIteration):
                next(fixture)
        assert all(getattr(obj, name) is value for obj, name, value in before)
        assert torch.get_float32_matmul_precision() == previous
        assert torch.backends.cuda.matmul.allow_tf32


def admit_fixture(monkeypatch, base, config):
    # Fixture creation uses the repository's compatible config subclass; public
    # loading uses the pinned stock HF class and its canonical vision dialect.
    config = transformers.AutoConfig.from_pretrained(base, local_files_only=True)
    monkeypatch.setattr(contract, "BASE_SHA256", snapshot_identity(base, weights=True)["sha256"])
    monkeypatch.setattr(contract, "PROCESSOR_SHA256", snapshot_identity(base, weights=False)["sha256"])
    for name, section in [
        ("ROOT_ARCHITECTURE", config),
        ("TEXT_ARCHITECTURE", config.text_config),
        ("VISION_ARCHITECTURE", config.vision_config),
    ]:
        monkeypatch.setattr(contract, name, {k: section.to_dict().get(k) for k in getattr(contract, name)})
    monkeypatch.setattr(contract, "ROPE_PARAMETERS", copy.deepcopy(config.text_config.rope_parameters))


def runtime(base, portable=None):
    kwargs = (
        {"base_execution_profile": profile.PROFILE}
        if portable is None
        else {"portable_adapter": {"directory": str(portable), "execution_profile": profile.PROFILE}}
    )
    return CosmosFrameworkRuntime(str(base), dtype="float32", attn_implementation="sdpa", **kwargs)


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_callback_preserves_native_fp32_boundaries(method, rslora, bias):
    torch.manual_seed(19)
    layer = PosttrainAdapterLinear(
        torch.nn.Linear(16, 12, bias=bias),
        AdapterComponentConfig(
            component="language",
            targets=["projection"],
            method=method,
            rank=3,
            alpha=5,
            use_rslora=rslora,
            dropout=0 if method == "pissa" else 0.2,
        ),
    ).eval()
    with torch.no_grad():
        layer.lora_B.weight.add_(0.03)
        if method == "dora":
            layer.magnitude.mul_(1.1)
        if method == "pissa":
            layer.initial_A.copy_(torch.randn_like(layer.initial_A) * 0.05)
            layer.initial_B.copy_(torch.randn_like(layer.initial_B) * 0.05)
    owners = profile._ownership(layer)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    x = torch.randn(1, 17, 16)
    with torch.no_grad():
        expected = layer(x)
        layer.lora_A.forward = MethodType(profile._linear, layer.lora_A)
        layer.lora_B.forward = MethodType(profile._linear, layer.lora_B)
        actual = profile._adapter(layer, x)
    assert torch.equal(actual, expected)
    assert profile._ownership(layer) == owners
    assert all(torch.equal(layer.state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("method", ["lora", "dora", "pissa", "mixed"])
@pytest.mark.parametrize("rslora", [False, True])
def test_public_portable_real_model_all_modalities_owners_and_reload(tmp_path, monkeypatch, method, rslora):
    base, portable, reference, config = artifact(
        tmp_path, monkeypatch, "qwen3_vl", method, rslora, tied=True, model_execution="qwen3_vl_hf_stock_v1"
    )
    admit_fixture(monkeypatch, base, config)
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("reload cannot decompose"))
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **k: pytest.fail("reload cannot decompose"))
    loaded = runtime(base, portable)
    net = loaded.model
    assert loaded.adapter_deployment["execution_profile"] == profile.PROFILE and loaded.base_deployment is None
    assert net.get_input_embeddings().weight is net.get_output_embeddings().weight
    owners = profile._ownership(net)
    rng = torch.get_rng_state().clone()
    original_attention = reference.model.language_model.layers[0].self_attn.forward.__func__
    assert net.model.language_model.layers[0].self_attn.forward.__func__ is original_attention
    assert (
        net.model.language_model.layers[0].input_layernorm.forward.__func__
        is reference.model.language_model.layers[0].input_layernorm.forward.__func__
    )
    for k, v in reference.state_dict().items():
        assert torch.equal(net.state_dict()[k], v)
    with torch.no_grad():
        for batch in data(config):
            assert torch.equal(net(**batch, use_cache=False).logits, reference(**batch, use_cache=False).logits)
            assert torch.equal(
                net.generate(**batch, max_new_tokens=3, do_sample=False),
                reference.generate(**batch, max_new_tokens=3, do_sample=False),
            )
    assert torch.equal(rng, torch.get_rng_state()) and profile._ownership(net) == owners
    before = list(net._cosmos_qwen3_fixed_mma["names"])
    assert (
        profile.install(net, identity=loaded.adapter_deployment["identity"], adapter_type=PosttrainAdapterLinear)
        == before
    )
    loaded.restore_inference_execution()
    assert net.model.language_model.layers[0].self_attn.q_proj.forward.__func__ is PosttrainAdapterLinear.forward
    assert loaded.adapter_deployment["restored_from_profile"] == profile.PROFILE


@pytest.fixture
def portable_case(tmp_path, monkeypatch):
    base, portable, reference, config = artifact(
        tmp_path, monkeypatch, "qwen3_vl", tied=True, model_execution="qwen3_vl_hf_stock_v1"
    )
    admit_fixture(monkeypatch, base, config)
    return base, portable, reference, config


@pytest.mark.parametrize(
    "mutation",
    [
        "method",
        "alpha",
        "dropout",
        "precision",
        "forward",
        "child",
        "norm",
        "attention",
        "assembly",
        "extra_norm",
        "attention_scale",
        "norm_epsilon",
        "config",
        "tracked_weight",
        "untracked_boundary",
        "algebra",
    ],
)
def test_changed_owner_rejected_before_cache_mutation(portable_case, mutation):
    from transformers.cache_utils import DynamicCache

    base, portable, _, config = portable_case
    rt = runtime(base, portable)
    net = rt.model
    target = net.model.language_model.layers[0].self_attn.q_proj
    if mutation == "method":
        target.method = "dora"
    elif mutation == "alpha":
        target._lora_alpha += 1
    elif mutation == "dropout":
        target.lora_dropout.train()
    elif mutation == "precision":
        net.bfloat16()
    elif mutation in ("forward", "child", "norm", "attention", "algebra", "assembly", "extra_norm"):
        module = {
            "forward": target,
            "child": target.lora_A,
            "norm": net.model.language_model.layers[0].input_layernorm,
            "attention": net.model.language_model.layers[0].self_attn,
            "algebra": target,
            "assembly": net.model,
            "extra_norm": net.model.language_model.norm,
        }[mutation]
        key = "adapter_output" if mutation == "algebra" else "forward"
        original = getattr(module, key)
        setattr(module, key, MethodType(lambda self, *a, **k: original(*a, **k), module))
    elif mutation == "config":
        net.config.text_config.rope_parameters["rope_theta"] *= 2
    elif mutation == "attention_scale":
        net.model.language_model.layers[0].self_attn.scaling *= 2
    elif mutation == "norm_epsilon":
        net.model.language_model.layers[0].input_layernorm.variance_epsilon *= 2
    elif mutation == "tracked_weight":
        with torch.no_grad():
            target.lora_A.weight.add_(1)
    else:
        target.lora_A.weight.data.add_(1)
        rt.clear_adapter_caches()
    cache = DynamicCache(config=config.text_config)
    with torch.no_grad(), pytest.raises((ValueError, RuntimeError)):
        net(input_ids=torch.tensor([[30, 31]]), past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0


@pytest.mark.parametrize("cache_kind", ["foreign", "sliding", "offload", "lazy_sliding"])
def test_cache_admission_precedes_mutation(portable_case, cache_kind):
    from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

    base, portable, _, config = portable_case
    rt = runtime(base, portable)
    net = rt.model
    if cache_kind == "foreign":
        native = transformers.AutoModelForImageTextToText.from_pretrained(
            base, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        with torch.no_grad():
            cache = native(input_ids=torch.tensor([[30, 31]]), use_cache=True).past_key_values
    else:
        cache = DynamicCache(config=config.text_config)
        if cache_kind == "offload":
            cache.offloading = True
        elif cache_kind == "sliding":
            cache.layers[0] = DynamicSlidingWindowLayer(sliding_window=2)
        else:
            cache.layers = []
            cache.layer_class_to_replicate = DynamicSlidingWindowLayer
    length = cache.get_seq_length()
    with torch.no_grad(), pytest.raises(ValueError, match="prefix|DynamicCache"):
        net(input_ids=torch.tensor([[32]]), past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == length


def test_prefix_fork_and_old_installation_rejection(portable_case):
    base, portable, _, _ = portable_case
    rt = runtime(base, portable)
    net = rt.model
    ids = torch.tensor([[30, 31, 32, 33]])
    with torch.no_grad():
        prefix = net(input_ids=ids[:, :2], use_cache=True).past_key_values
        stored = prefix.layers[0].keys.clone()
        first = net(input_ids=ids[:, 2:], past_key_values=rt._fork_dynamic_cache(prefix), use_cache=True).logits
        second = net(input_ids=ids[:, 2:], past_key_values=rt._fork_dynamic_cache(prefix), use_cache=True).logits
    assert torch.equal(first, second) and torch.equal(prefix.layers[0].keys, stored)
    rt.restore_inference_execution()
    profile.install(
        net,
        identity=rt.adapter_deployment["identity"],
        adapter_type=PosttrainAdapterLinear,
        assembly_forward=assembly_forward,
        extra_norm_forward=extra_norm_forward,
    )
    with torch.no_grad(), pytest.raises(ValueError, match="prefix"):
        net(input_ids=ids[:, 2:], past_key_values=prefix, use_cache=True)


@pytest.mark.parametrize("bad", ["base", "processor", "ordered", "dense", "family", "dependency", "dtype"])
def test_wrong_artifact_admission_before_install(portable_case, bad):
    _, _, reference, _ = portable_case
    identity = copy.deepcopy(reference._posttrain_adapter_identity)
    if bad in ("base", "processor"):
        identity[bad]["sha256"] = "0" * 64
    elif bad == "ordered":
        identity["resolved"]["model_execution"] = "qwen_vl_hf_ordered_v1"
    elif bad == "dense":
        identity["resolved"]["targets"][0]["settings"]["execution"] = "pissa_dense_v1"
    elif bad == "family":
        identity["architecture"] = "qwen3_vl_moe"
    elif bad == "dependency":
        identity["dependencies"]["transformers"] = "4.57.6"
    else:
        identity["resolved"]["targets"][0]["settings"]["dtype"] = "bfloat16"
    owners = profile._executions(reference)
    with pytest.raises(ValueError):
        profile.install(reference, identity=identity, adapter_type=PosttrainAdapterLinear)
    assert profile._executions(reference) == owners and not hasattr(reference, "_cosmos_deployment_profile")


@pytest.mark.parametrize("bad", ["triton", "gpu_name", "capability", "cpu"])
def test_public_portable_environment_rejects_before_construction_or_injection(portable_case, monkeypatch, bad):
    from importlib import metadata

    from cosmos_framework.posttrain.model import adapters

    base, portable, _, _ = portable_case
    original_version = metadata.version
    monkeypatch.setattr(
        metadata, "version", lambda name: "3.5.0" if bad == "triton" and name == "triton" else original_version(name)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0) if bad == "capability" else (8, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device: "NVIDIA A30" if bad == "gpu_name" else "NVIDIA A100"
    )
    admission = []

    def checked(profile_name, device):
        admission.append((profile_name, device))
        return environment_guard(profile_name, device)

    monkeypatch.setattr(precision, "validate_deployment_environment", checked)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("construction/injection")
        pytest.fail("Environment rejection must precede all model construction and adapter mutation")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", forbidden)
    monkeypatch.setattr(adapters, "install_adapters", forbidden)
    device = "cpu" if bad == "cpu" else "cuda:0"
    with pytest.raises(ValueError, match="pinned|A100"):
        portable_adapters.load_portable_model(
            str(base),
            {"directory": str(portable), "execution_profile": profile.PROFILE},
            dtype="float32",
            device=device,
            attn_implementation="sdpa",
        )
    assert admission == [(profile.PROFILE, device)] and calls == []


def test_original_stock_portable_retains_environment_admission_boundary(portable_case, monkeypatch):
    base, portable, _, _ = portable_case
    monkeypatch.setattr(
        precision, "validate_deployment_environment", lambda *a, **k: pytest.fail("New guard is profile-specific")
    )
    calls = []

    def stop_before_allocation(*args, **kwargs):
        calls.append("original construction boundary")
        raise RuntimeError("recorded original boundary reached")

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", stop_before_allocation)
    with pytest.raises(RuntimeError, match="original boundary"):
        portable_adapters.load_portable_model(
            str(base),
            {"directory": str(portable), "execution_profile": portable_adapters.STOCK_FACTORIZED_PROFILE},
            dtype="float32",
            device="cpu",
            attn_implementation="sdpa",
        )
    assert calls == ["original construction boundary"]


def test_initial_foreign_norm_refuses_before_any_patch(portable_case):
    _, _, reference, _ = portable_case
    norm = reference.model.language_model.layers[0].input_layernorm
    original = norm.forward
    norm.forward = MethodType(lambda self, x: original(x), norm)
    before = profile._executions(reference)
    with pytest.raises(ValueError, match="native callable"):
        profile.install(
            reference,
            identity=reference._posttrain_adapter_identity,
            adapter_type=PosttrainAdapterLinear,
            assembly_forward=assembly_forward,
            extra_norm_forward=extra_norm_forward,
        )
    assert profile._executions(reference) == before


@pytest.mark.parametrize("name", ["model", "model.language_model.norm"])
def test_public_portable_rejects_replaced_builder_forward(portable_case, monkeypatch, name):
    from cosmos_framework.posttrain.model import adapters

    base, portable, _, _ = portable_case
    original = adapters.install_adapters
    captured = []

    def replaced(net, *args, **kwargs):
        result = original(net, *args, **kwargs)
        module = net.get_submodule(name)
        prior = module.forward
        module.forward = MethodType(lambda self, *a, **k: prior(*a, **k), module)
        captured.append((net, profile._executions(net)))
        return result

    monkeypatch.setattr(adapters, "install_adapters", replaced)
    with pytest.raises(ValueError, match="exact recorded assembly/norm"):
        runtime(base, portable)
    net, before = captured[0]
    assert profile._executions(net) == before and not hasattr(net, "_cosmos_deployment_profile")


def test_tied_output_head_is_frozen_projection_not_new_adapter_target(portable_case):
    from cosmos_framework.posttrain.config.adapters import AdapterConfig
    from cosmos_framework.posttrain.model.adapters import resolve_targets

    _, _, net, _ = portable_case
    cfg = AdapterConfig(
        model_execution="qwen3_vl_hf_stock_v1",
        components=[AdapterComponentConfig(component="language", targets=["lm_head"], rank=2, alpha=4)],
    )
    owners = profile._ownership(net)
    with pytest.raises(ValueError, match="outside language"):
        resolve_targets(net, cfg, family="qwen3_vl")
    bad = copy.deepcopy(net._posttrain_adapter_identity)
    bad["resolved"]["targets"][0]["name"] = "lm_head"
    with pytest.raises(ValueError, match="exact BF16-base"):
        profile.install(
            net,
            identity=bad,
            adapter_type=PosttrainAdapterLinear,
            assembly_forward=assembly_forward,
            extra_norm_forward=extra_norm_forward,
        )
    assert profile._ownership(net) == owners and not hasattr(net, "_cosmos_deployment_profile")


@pytest.mark.parametrize("mutation", ["outer", "hook", "old"])
def test_accelerate_wrapper_ownership_and_restore_rejection(portable_case, mutation):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from transformers.cache_utils import DynamicCache

    _, _, net, config = portable_case
    target = net.model.language_model.layers[0].self_attn.q_proj
    hook = AlignDevicesHook(execution_device="cpu", offload=False)
    add_hook_to_module(target, hook)
    outer = target.forward
    profile.install(
        net,
        identity=net._posttrain_adapter_identity,
        adapter_type=PosttrainAdapterLinear,
        assembly_forward=assembly_forward,
        extra_norm_forward=extra_norm_forward,
    )
    with torch.no_grad():
        assert torch.isfinite(net(input_ids=torch.tensor([[30, 31]])).logits).all()
    assert target.forward is outer and target._hf_hook is hook
    if mutation == "outer":
        target.forward = lambda *a, **k: outer(*a, **k) + 1
    elif mutation == "hook":
        target._hf_hook = AlignDevicesHook(execution_device="cpu", offload=False)
    else:
        target._old_forward = MethodType(PosttrainAdapterLinear.forward, target)
    cache = DynamicCache(config=config.text_config)
    with torch.no_grad(), pytest.raises(ValueError, match="ownership"):
        net(input_ids=torch.tensor([[30, 31]]), past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0
    with pytest.raises(ValueError, match="ownership"):
        profile.restore(net)


@pytest.mark.parametrize("name", ["input_ids", "inputs_embeds"])
@pytest.mark.parametrize("positional", [False, True])
def test_batch_guard_covers_public_positional_inputs(portable_case, name, positional):
    from transformers.cache_utils import DynamicCache

    base, portable, _, config = portable_case
    net = runtime(base, portable).model
    value = (
        torch.ones(2, 4, dtype=torch.long) if name == "input_ids" else torch.zeros(2, 4, config.text_config.hidden_size)
    )
    cache = DynamicCache(config=config.text_config)
    args = []
    kwargs = {"past_key_values": cache}
    if positional:
        names = list(signature(net.forward).parameters)
        args = [None] * (names.index(name) + 1)
        args[-1] = value
        if names.index("past_key_values") < len(args):
            args[names.index("past_key_values")] = kwargs.pop("past_key_values")
    else:
        kwargs[name] = value
    with torch.no_grad(), pytest.raises(ValueError, match="batch-one"):
        net(*args, **kwargs)
    assert cache.get_seq_length() == 0


@pytest.mark.parametrize(
    "kwargs", [{"dtype": "bfloat16"}, {"attn_implementation": "eager"}, {"device_map": "balanced"}]
)
def test_public_incompatible_precision_profile_rejects_before_model(portable_case, monkeypatch, kwargs):
    base, portable, _, _ = portable_case
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *a, **k: pytest.fail("invalid profile reached model construction"),
    )
    options = {"dtype": "float32", "attn_implementation": "sdpa", **kwargs}
    with pytest.raises(ValueError):
        CosmosFrameworkRuntime(
            str(base), portable_adapter={"directory": str(portable), "execution_profile": profile.PROFILE}, **options
        )


@pytest.mark.parametrize(
    "change", ["processor", "tensor", "missing_shard", "extra_shard", "extra_tensor", "wrong_index", "marker"]
)
def test_public_merged_integrity_admission_precedes_model_allocation(portable_case, tmp_path, monkeypatch, change):
    from safetensors.torch import load_file, save_file

    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    base, portable, _, _ = portable_case
    merged = tmp_path / "merged"
    export_merged_adapter(str(portable), str(base), str(merged), execution_profile=profile.PROFILE)
    shard = next(merged.glob("*.safetensors"))
    if change == "processor":
        (merged / "tokenizer_config.json").write_text("{}")
    elif change == "tensor":
        state = load_file(shard)
        state[next(iter(state))].add_(1)
        save_file(state, shard)
    elif change == "missing_shard":
        shard.unlink()
    elif change == "extra_shard":
        (merged / "extra.safetensors").write_bytes(shard.read_bytes())
    elif change == "extra_tensor":
        state = load_file(shard)
        state["foreign.weight"] = torch.zeros(2)
        save_file(state, shard)
    elif change == "wrong_index":
        index = json.loads((merged / "model.safetensors.index.json").read_bytes())
        key = next(iter(index["weight_map"]))
        index["weight_map"]["foreign.weight"] = index["weight_map"].pop(key)
        (merged / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        marker = json.loads((merged / ".materialized").read_bytes())
        marker["options"]["no_verify"] = True
        (merged / ".materialized").write_text(json.dumps(marker))
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *a, **k: pytest.fail("invalid merged source reached model construction"),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        CosmosFrameworkRuntime(str(merged), dtype="float32", attn_implementation="sdpa")


@pytest.mark.parametrize("change", ["processor", "tensor", "marker"])
def test_merged_source_change_during_load_rejected(portable_case, tmp_path, monkeypatch, change):
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    base, portable, _, _ = portable_case
    merged = tmp_path / "merged"
    export_merged_adapter(str(portable), str(base), str(merged), execution_profile=profile.PROFILE)
    original = transformers.AutoModelForImageTextToText.from_pretrained

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "processor":
            (merged / "generation_config.json").write_text("{}")
        elif change == "tensor":
            from safetensors.torch import load_file, save_file

            shard = next(merged.glob("*.safetensors"))
            state = load_file(shard)
            state[next(iter(state))].add_(1)
            save_file(state, shard)
        else:
            marker = merged / ".materialized"
            marker.write_bytes(marker.read_bytes() + b" ")
        return result

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", changing)
    with pytest.raises(ValueError, match="processor|changed during|inventory changed"):
        CosmosFrameworkRuntime(str(merged), dtype="float32", attn_implementation="sdpa")


@pytest.mark.parametrize("name", ["NVIDIA A100-SXM4-80GB", "NVIDIA A30"])
def test_exact_a100_admission_is_not_only_sm80(monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    if "A100" in name:
        target_device_guard(torch.device("cuda:0"))
        environment_guard(profile.PROFILE, "cuda:0")
    else:
        with pytest.raises(ValueError, match="A100"):
            target_device_guard(torch.device("cuda:0"))
        with pytest.raises(ValueError, match="A100"):
            environment_guard(profile.PROFILE, "cuda:0")


def test_fresh_process_public_base_generation_without_training_imports(tmp_path, monkeypatch):
    base = tmp_path / "base"
    config = snapshot(base, "qwen3_vl", tied=True)
    admit_fixture(monkeypatch, base, config)
    script = tmp_path / "fresh.py"
    script.write_text("""import copy,json,sys,torch
from transformers import AutoConfig
from cosmos_framework.inference.reasoner import generate,qwen3_vl_fixed_mma as profile
from cosmos_framework.utils.generator import qwen3_vl_projection_contract as contract
from cosmos_framework.utils.hf_snapshot import snapshot_identity
base,configuration,output=sys.argv[1:]
config=AutoConfig.from_pretrained(base,local_files_only=True)
contract.BASE_SHA256=snapshot_identity(base,weights=True)['sha256']
contract.PROCESSOR_SHA256=snapshot_identity(base,weights=False)['sha256']
for name,section in [('ROOT_ARCHITECTURE',config),('TEXT_ARCHITECTURE',config.text_config),('VISION_ARCHITECTURE',config.vision_config)]:
    setattr(contract,name,{k:section.to_dict().get(k) for k in getattr(contract,name)})
contract.ROPE_PARAMETERS=copy.deepcopy(config.text_config.rope_parameters)
profile._device=lambda model:None
profile._target_device=lambda device:None
profile._project=torch.nn.functional.linear
torch.cuda.is_available=lambda:False
torch.cuda.is_current_stream_capturing=lambda:False
sys.argv=['generate','--config',configuration,'--output',output]
generate.main()
assert not any(n.startswith('cosmos_framework.posttrain') for n in sys.modules)
record=json.load(open(output))
assert record['deployment']['kind']=='unadapted_model_deployment'
assert record['deployment']['execution_profile']==profile.PROFILE
assert 'resolved' not in record['deployment']['identity']
assert len(record['answers'])==1
""")
    cfg = tmp_path / "generation.toml"
    cfg.write_text(
        "[model]\nmodel_name="
        + json.dumps(str(base))
        + '\ndtype="float32"\nattn_implementation="sdpa"\nbase_execution_profile='
        + json.dumps(profile.PROFILE)
        + '\n[generation]\nmax_tokens=3\ntemperature=0.0\nseed=42\n[[tasks]]\nquestion="word30 word31"\n'
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(base), str(cfg), str(tmp_path / "answer.json")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_base_route_and_untracked_write_reload(tmp_path, monkeypatch):
    base = tmp_path / "base"
    config = snapshot(base, "qwen3_vl", tied=True)
    admit_fixture(monkeypatch, base, config)
    rt = runtime(base)
    assert rt.adapter_deployment is None and rt.base_deployment["kind"] == "unadapted_model_deployment"
    assert not any(p.requires_grad for p in rt.model.parameters())
    with torch.no_grad():
        cache = rt.model(input_ids=torch.tensor([[30, 31]]), use_cache=True).past_key_values
    rt.model.model.language_model.layers[0].self_attn.q_proj.weight.data.add_(0.01)
    rt.clear_adapter_caches()
    assert rt.base_deployment["state_modified_since_load"] is True
    with torch.no_grad(), pytest.raises(ValueError, match="invalidated"):
        rt.model(input_ids=torch.tensor([[32]]), past_key_values=cache, use_cache=True)
    rt.restore_inference_execution()
    with pytest.raises(ValueError, match="invalidated"):
        profile.install(rt.model, identity=rt.base_deployment["identity"])
    fresh = runtime(base)
    with torch.no_grad():
        assert torch.isfinite(fresh.model(input_ids=torch.tensor([[30, 31]])).logits).all()


def test_public_tagged_export_reload_and_metadata_rejection(portable_case, tmp_path):
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    base, portable, _, _ = portable_case
    merged = tmp_path / "merged"
    assert export_merged_adapter(str(portable), str(base), str(merged), execution_profile=profile.PROFILE) == "built"
    assert export_merged_adapter(str(portable), str(base), str(merged), execution_profile=profile.PROFILE) == "skipped"
    rt = CosmosFrameworkRuntime(str(merged), dtype="float32", attn_implementation="sdpa")
    assert rt.adapter_deployment["execution_profile"] == profile.PROFILE and rt.base_deployment is None
    with torch.no_grad():
        assert torch.isfinite(rt.model(input_ids=torch.tensor([[30, 31]])).logits).all()
    marker = json.loads((merged / ".materialized").read_bytes())
    marker["identity"]["export_metadata"]["deployment_precision"]["equivalence"] = "legacy"
    (merged / ".materialized").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="committed"):
        precision.deployment_profile(str(merged))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an actual A100; CPU fixture is not operator qualification"
)
def test_installed_fixed_projection_kernel_matches_same_operand_fp64():
    # First Triton initialization must see real CUDA availability, independently
    # of any cached driver from another test. Full-model qualification is separate.
    from cosmos_framework.utils.generator.smollm_fixed_mma import linear

    assert torch.cuda.is_available() and profile._project is not F.linear
    assert not torch.backends.cuda.matmul.allow_tf32
    if torch.cuda.get_device_capability(0) != (8, 0) or "A100" not in torch.cuda.get_device_name(0):
        pytest.skip("A100 only")
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(1, 225, 2048, device="cuda", generator=generator)
    w = torch.randn(64, 2048, device="cuda", generator=generator)
    b = torch.randn(64, device="cuda", generator=generator)
    with torch.no_grad():
        full = linear(x, w, b).float()
        one = linear(x[:, -1:], w, b).float()
        reference = F.linear(x.double(), w.double(), b.double()).float()
    assert torch.equal(full[:, -1:], one)
    torch.testing.assert_close(full, reference, atol=2e-5, rtol=2e-5)
    assert float((full.double() - reference.double()).abs().mean()) <= 2e-6
