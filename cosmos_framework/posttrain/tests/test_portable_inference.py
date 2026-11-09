# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public portable loading, inference-only branches and cache ownership."""

import copy
from types import MethodType

import pytest
import torch

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
from cosmos_framework.utils.generator import factorized_inference as stable
from cosmos_framework.utils.generator.bf16_cache_contract import QWEN3_VL_PROFILES
from cosmos_framework.utils.generator.portable_adapters import CANONICAL, PortableAdapterConfig, model_cache_signature


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_branch_precision_and_ownership_against_independent_reference(method, rslora, monkeypatch):
    torch.manual_seed(961109)
    layer = PosttrainAdapterLinear(
        torch.nn.Linear(17, 13),
        AdapterComponentConfig(targets=["layer"], rank=3, alpha=7, method=method, use_rslora=rslora),
    ).eval()
    with torch.no_grad():
        layer.lora_B.weight.add_(torch.randn_like(layer.lora_B.weight) * 0.07)
    source = {n: (p, p.clone()) for n, p in layer.named_parameters()}
    state = copy.deepcopy(layer.state_dict())
    monkeypatch.setattr(stable, "_guard", lambda _: None)  # Arithmetic fixture only; real guards tested below.
    original = layer.forward
    layer.forward = MethodType(stable._forward, layer)
    x = torch.randn(11, 17)
    scale = 7 / (3**0.5 if rslora else 3)
    effective = layer.weight.double() + scale * (layer.lora_B.weight.double() @ layer.lora_A.weight.double())
    if method == "pissa":
        effective -= scale * (layer.initial_B.double() @ layer.initial_A.double())
    if method == "dora":
        effective *= (layer.magnitude.double() / effective.norm(dim=1))[:, None]
    with torch.no_grad():
        actual = layer(x)
        reference = torch.nn.functional.linear(x.double(), effective, layer.bias.double()).float()
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=2e-6)
    assert all(p is source[n][0] and torch.equal(p, source[n][1]) for n, p in layer.named_parameters())
    assert all(torch.equal(t, state[n]) for n, t in layer.state_dict().items())
    layer.forward = original


def test_configuration_requires_explicit_compatible_profile():
    with pytest.raises(TypeError):
        PortableAdapterConfig(directory="adapter")
    for profile, cache in [("factorized_v1", False), (stable.PROFILE, True), (CANONICAL, "yes")]:
        with pytest.raises(ValueError):
            PortableAdapterConfig(directory="adapter", execution_profile=profile, cache_effective_weights=cache)


def test_runtime_cache_invalidation_tracks_sources_and_clear():
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model = torch.nn.Linear(4, 3)
    runtime.portable_adapter = object()
    runtime.adapter_deployment = {}
    runtime._adapter_cache_signature = model_cache_signature(runtime.model)
    for name in ("_video_feature_cache", "_video_processor_cache", "_video_input_cache", "_video_prefix_cache"):
        setattr(runtime, name, {"cached": object()})
    with torch.no_grad():
        runtime.model.weight.add_(1)
    runtime._refresh_adapter_caches()
    assert runtime.adapter_deployment["state_modified_since_load"]
    assert all(not getattr(runtime, name) for name in ("_video_feature_cache", "_video_prefix_cache"))
    runtime._video_prefix_cache["cached"] = object()
    runtime._refresh_adapter_caches()
    assert runtime._video_prefix_cache  # Unchanged parameters retain reusable prefixes.
    runtime.model.weight.data.add_(1)
    runtime.adapter_deployment.clear()
    runtime.clear_adapter_caches()  # Explicitly required for unsafe .data/fused writes.
    assert not runtime._video_prefix_cache
    runtime._refresh_adapter_caches()
    assert runtime.adapter_deployment["state_modified_since_load"]


def test_evaluator_records_executed_generation_settings(tmp_path):
    from types import SimpleNamespace

    from cosmos_framework.evaluation.reasoner.base import BaseEvaluator

    class Evaluator(BaseEvaluator):
        def make_tasks(self, *args):
            pass

        def save(self, *args):
            pass

        def compute_metrics(self, *args):
            pass

        def aggregate_metrics_if_all_shards_present(self, *args):
            pass

    evaluator = Evaluator({"generation": {"max_tokens": 32, "temperature": 0.7, "seed": 99}, "evaluation": {"seed": 1}})
    evaluator._send_status_callback = lambda *args: None
    evaluator._adapter_deployment_path = tmp_path / "deployment.json"
    runtime = SimpleNamespace(adapter_deployment={})

    def generate_tasks(tasks, *, generation_config, vision_config):
        runtime.adapter_deployment["effective_generation"] = generation_config
        return ["A"], [0.0]

    runtime.generate_tasks = generate_tasks
    evaluator.model = runtime
    evaluator.run_model_inference([{}], [{}], answer_type="letter")
    import json

    assert json.loads(evaluator._adapter_deployment_path.read_text())["effective_generation"] == {
        "max_tokens": 10,
        "temperature": 0.0,
        "seed": 99,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="public profiles require A100")
@pytest.mark.parametrize("profile", [*QWEN3_VL_PROFILES, stable.PROFILE])
def test_public_loader_strict_identity_ownership_and_restoration(tmp_path, monkeypatch, profile):
    from transformers.cache_utils import StaticCache

    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.adapter_bf16_execution import install_adapter_inference_profile
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.tests import test_adapters_native as fixtures
    from cosmos_framework.posttrain.tests.test_adapter_merged_weight import configuration
    from cosmos_framework.utils.generator.portable_adapters import load_portable_model

    torch.set_float32_matmul_precision("highest")
    cfg = fixtures._dense_config()
    cfg.text_config.hidden_size = cfg.vision_config.out_hidden_size = 128
    cfg.text_config.head_dim = 64
    cfg.text_config.num_attention_heads = 2
    cfg.text_config.num_key_value_heads = 1
    cfg.text_config.rope_parameters["mrope_section"] = [8, 12, 12]
    monkeypatch.setattr(fixtures, "_dense_config", lambda: cfg)
    base, artifact = tmp_path / "base", tmp_path / "adapter"
    fixtures.tiny_snapshot(base, tied=True)
    settings = configuration()
    if profile == stable.PROFILE:
        settings.model_execution = "qwen3_vl_hf_stock_v1"
        for component in settings.components:
            component.execution = "factorized_v1"
    source = (
        build_hf_model(HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa"), adapters=settings)
        .net.eval()
        .cuda()
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith("lora_B.weight"):
                parameter.add_(torch.randn_like(parameter) * 0.01)
    save_adapter_artifact(source, str(artifact))
    dtype = "float32" if profile == stable.PROFILE else "bfloat16"
    runtime = CosmosFrameworkRuntime(
        str(base),
        dtype=dtype,
        portable_adapter={
            "directory": str(artifact),
            "execution_profile": profile,
            "cache_effective_weights": profile in QWEN3_VL_PROFILES,
        },
    )
    loaded = runtime.model
    assert loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight
    batch = {"input_ids": torch.tensor([[1, 2, 3]], device="cuda")}
    if profile in QWEN3_VL_PROFILES:
        install_adapter_inference_profile(source, profile)
    else:
        source.float()
        stable.install(source, adapter_type=PosttrainAdapterLinear)
    before = {n: (id(p), p.clone(), p.requires_grad) for n, p in loaded.named_parameters()}
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).logits, source(**batch).logits, atol=0, rtol=0)
    assert all(
        id(p) == before[n][0] and torch.equal(p, before[n][1]) and p.requires_grad == before[n][2]
        for n, p in loaded.named_parameters()
    )
    if profile == stable.PROFILE:
        with pytest.raises(RuntimeError, match="no_grad"):
            loaded(**batch)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16),
            pytest.raises(RuntimeError, match="autocast"),
        ):
            loaded(**batch)
        with torch.no_grad(), pytest.raises(ValueError, match="DynamicCache"):
            loaded(**batch, past_key_values=StaticCache(config=loaded.config, max_cache_len=32))
        stable.install(loaded, adapter_type=PosttrainAdapterLinear)  # Idempotent, not double wrapped.
        from transformers.cache_utils import DynamicCache

        empty_cache = DynamicCache(config=loaded.config)
        original_signature = model_cache_signature(loaded)
        loaded.set_attn_implementation("eager")
        assert model_cache_signature(loaded) != original_signature
        with torch.no_grad(), pytest.raises(ValueError, match="stock SDPA throughout"):
            loaded(**batch, past_key_values=empty_cache)
        assert empty_cache.get_seq_length() == 0
        loaded.set_attn_implementation("sdpa")
        runtime.restore_factorized_execution()
        stable.restore(source)
        with torch.no_grad():
            torch.testing.assert_close(loaded(**batch).logits, source(**batch).logits, atol=0, rtol=0)
        loaded._cosmos_deployment_profile = "foreign"
        with pytest.raises(ValueError, match="different deployment"):
            stable.install(loaded, adapter_type=PosttrainAdapterLinear)
        del loaded._cosmos_deployment_profile
    # Compatibility failure must happen before the base constructor/injection.
    (base / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(
        "transformers.AutoModelForImageTextToText.from_pretrained",
        lambda *a, **k: pytest.fail("mutated before identity validation"),
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_portable_model(str(base), PortableAdapterConfig(str(artifact), profile), dtype=dtype, device="cuda:0")
