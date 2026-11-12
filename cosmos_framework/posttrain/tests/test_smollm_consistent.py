# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small arithmetic/contract tests; real Smol GPU evidence is recorded separately."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
from cosmos_framework.utils.generator import smollm_consistent as profile
from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig


def identity():
    return {
        "format": "cosmos-posttrain-adapter-v1",
        "architecture": "llama",
        "base_storage_dtype": "bfloat16",
        "base": {"sha256": profile.BASE_SHA256},
        "dependencies": {"torch": "2.10.0", "transformers": "5.8.1"},
        "resolved": {"targets": [{"name": "model.layers.0.self_attn.q_proj", "settings": {"method": "lora"}}]},
    }


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_separate_branches_follow_independent_double_reference(method, rslora, monkeypatch):
    monkeypatch.setattr(profile, "_guard", lambda module: None)
    torch.manual_seed(17)
    base = torch.nn.Linear(7, 5, bias=True)
    layer = PosttrainAdapterLinear(
        base,
        AdapterComponentConfig(
            component="language",
            targets=["projection"],
            method=method,
            rank=3,
            alpha=7,
            use_rslora=rslora,
            dropout=0.0 if method == "pissa" else 0.25,
            initialization="svd" if method == "pissa" else "default",
        ),
    ).eval()
    with torch.no_grad():
        layer.lora_A.weight.add_(torch.randn_like(layer.lora_A.weight) * 0.02)
        layer.lora_B.weight.add_(torch.randn_like(layer.lora_B.weight) * 0.02)
        if method == "dora":
            layer.magnitude.mul_(1.03)
    x = torch.randn(1, 9, 7)
    owned = {n: (id(t), t.clone()) for n, t in layer.state_dict(keep_vars=True).items()}
    w = layer.weight.double() + layer._lora_scale * (layer.lora_B.weight.double() @ layer.lora_A.weight.double())
    if method == "pissa":
        w -= layer._lora_scale * (layer.initial_B.double() @ layer.initial_A.double())
    elif method == "dora":
        w *= (layer.magnitude.double() / w.norm(dim=1))[:, None]
    reference = F.linear(x.double(), w, layer.bias.double()).float()
    actual = profile._adapter(layer, x)
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-7)
    assert torch.equal(actual[:, -1:], profile._adapter(layer, x[:, -1:]))
    for name, tensor in layer.state_dict(keep_vars=True).items():
        assert id(tensor) == owned[name][0]
        assert torch.equal(tensor, owned[name][1])


def test_identity_and_configuration_are_bounded():
    good = identity()
    profile.validate_identity(good)
    PortableAdapterConfig("adapter", profile.PROFILE)
    with pytest.raises(ValueError, match="cache"):
        PortableAdapterConfig("adapter", profile.PROFILE, cache_effective_weights=True)
    for field, bad in [
        ("base_storage_dtype", "float32"),
        ("architecture", "qwen3_vl"),
        ("format", "cosmos-posttrain-adapter-v4"),
    ]:
        value = copy.deepcopy(good)
        value[field] = bad
        with pytest.raises(ValueError, match="recorded Smol"):
            profile.validate_identity(value)
    value = copy.deepcopy(good)
    value["base"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded Smol"):
        profile.validate_identity(value)


def test_export_metadata_and_public_profile_are_consistent(tmp_path):
    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata

    record = {"kind": "portable_adapter", "identity": identity(), "weights_sha256": "a" * 64}
    metadata = merged_export_metadata(
        "unused", record["identity"], artifact_record=record, output_dtype="float32", execution_profile=profile.PROFILE
    )
    config = config_with_export_dtype({"model_type": "llama"}, "float32", execution_profile=profile.PROFILE)
    marker = {"identity": {"output_dtype": "float32", "export_metadata": metadata}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".materialized").write_text(json.dumps(marker))
    assert deployment_profile(str(tmp_path)) == profile.PROFILE
    with pytest.raises(ValueError, match="float32"):
        merged_export_metadata(
            "unused",
            record["identity"],
            artifact_record=record,
            output_dtype="bfloat16",
            execution_profile=profile.PROFILE,
        )
    config.pop("cosmos_inference_profile")
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting"):
        deployment_profile(str(tmp_path))


def test_text_processor_does_not_discard_media():
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    runtime = SimpleNamespace(processor=None)
    with pytest.raises(ValueError, match="text-only"):
        CosmosFrameworkRuntime._prepare_smollm_text(
            runtime,
            [
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": "not-a-text-input"}],
                    }
                ]
            ],
        )


def test_norm_and_projection_round_at_fp32_boundary(monkeypatch):
    monkeypatch.setattr(profile, "_guard", lambda module: None)
    x = torch.randn(1, 31, 23)
    linear = torch.nn.Linear(23, 11)
    norm = SimpleNamespace(weight=torch.randn(23), variance_epsilon=1e-5)
    for module, forward in [(linear, profile._linear), (norm, profile._norm)]:
        result = forward(module, x)
        assert result.dtype == x.dtype
        assert torch.equal(result[:, -1:], forward(module, x[:, -1:]))


def test_portable_loader_binds_llama_identity_before_injection(monkeypatch):
    from cosmos_framework.posttrain.checkpoint import adapters as artifacts
    from cosmos_framework.posttrain.model import adapters
    from cosmos_framework.utils.generator.portable_adapters import load_portable_model

    expected = identity()
    expected.update(processor={"sha256": "b" * 64}, media={})
    monkeypatch.setattr(artifacts, "read_adapter_snapshot", lambda _: ({"identity": expected}, {}))
    monkeypatch.setattr(artifacts, "snapshot_identity", lambda _, weights: expected["base" if weights else "processor"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: SimpleNamespace(model_type="llama"))
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: (object(), {}))
    monkeypatch.setattr(artifacts, "adapter_configuration", lambda _: object())
    monkeypatch.setattr(adapters, "resolve_targets", lambda *a, **k: expected["resolved"])
    monkeypatch.setattr(
        adapters, "install_adapters", lambda *a, **k: pytest.fail("injected before identity validation")
    )

    def verify(*args, **kwargs):
        assert kwargs["architecture"] == "llama"
        raise RuntimeError("reached exact llama identity check")

    monkeypatch.setattr(artifacts, "make_adapter_identity", verify)
    with pytest.raises(RuntimeError, match="exact llama identity"):
        load_portable_model("base", PortableAdapterConfig("adapter", profile.PROFILE), dtype="float32", device="cuda:0")


@pytest.mark.parametrize("architecture", ["Gr00tN1d7", "alpamayo_r1", "pi05"])
def test_physical_export_cannot_ignore_smol_profile(monkeypatch, tmp_path, architecture):
    from cosmos_framework.posttrain.checkpoint.tools import adapters

    foreign = identity()
    foreign["architecture"] = architecture
    monkeypatch.setattr(adapters, "read_adapter_snapshot", lambda _: ({"identity": foreign}, {}))
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="recorded Smol"):
        adapters.export_merged_adapter("adapter", "base", str(output), execution_profile=profile.PROFILE)
    assert not output.exists()


def test_foreign_prefix_rejected_before_model_execution(monkeypatch):
    from inspect import signature

    from transformers import LlamaConfig
    from transformers.cache_utils import DynamicCache

    def forward(input_ids=None, past_key_values=None):
        pass

    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    model._cosmos_smollm_consistent = {
        "ownership": (),
        "forwards": [],
        "adapters": [],
        "signature": signature(forward),
        "cache_owner": (profile.PROFILE, "installation-one"),
    }
    monkeypatch.setattr(profile, "_guard", lambda _: None)
    monkeypatch.setattr(profile, "_ownership", lambda _: ())
    cache = DynamicCache(config=LlamaConfig(num_hidden_layers=1))
    value = torch.zeros(1, 1, 2, 4)
    cache.update(value, value, 0)
    kwargs = {"input_ids": torch.ones(1, 1, dtype=torch.long), "past_key_values": cache}
    with pytest.raises(ValueError, match="prefix produced"):
        profile._before_forward(model, (), kwargs)
    cache._cosmos_smollm_owner = model._cosmos_smollm_consistent["cache_owner"]
    profile._before_forward(model, (), kwargs)
    fork = copy.deepcopy(cache)
    profile._before_forward(model, (), {**kwargs, "past_key_values": fork})
    model._cosmos_smollm_consistent["cache_owner"] = (profile.PROFILE, "installation-two")
    with pytest.raises(ValueError, match="prefix produced"):
        profile._before_forward(model, (), kwargs)
