# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical cache profile binding, operator consistency and unchanged ownership."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.utils.generator.bf16_cache_contract import PROFILE, QWEN3_VL_PROFILES, validate_adapter_identity


def identity():
    return {
        "format": "cosmos-posttrain-adapter-v4",
        "architecture": "qwen3_vl",
        "base_storage_dtype": "bfloat16",
        "dependencies": {"torch": "2.10.0", "transformers": "5.8.1"},
        "resolved": {
            "model_execution": "merged_weight_bf16_v1",
            "targets": [{"settings": {"execution": "merged_weight_bf16_v1", "dropout": 0.0}}],
        },
    }


@pytest.mark.parametrize("profile", QWEN3_VL_PROFILES)
def test_contract_and_export_binding(tmp_path, profile):
    from cosmos_framework.inference.reasoner.model_execution import model_execution_profile
    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata
    from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype

    ident = identity()
    manifest = {"kind": "portable_adapter", "identity": ident, "weights_sha256": "digest"}
    metadata = merged_export_metadata(
        "unused", ident, artifact_record=manifest, output_dtype="bfloat16", execution_profile=profile
    )
    config = config_with_export_dtype(
        {"model_type": "qwen3_vl"}, "bfloat16", execution_profile=profile, model_execution="merged_weight_bf16_v1"
    )
    marker = {"identity": {"output_dtype": "bfloat16", "export_metadata": metadata}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".materialized").write_text(json.dumps(marker))
    assert deployment_profile(tmp_path) == profile
    assert model_execution_profile(tmp_path) == "merged_weight_bf16_v1"
    for key, value in [
        ("format", "cosmos-posttrain-adapter-v3"),
        ("architecture", "qwen3_5"),
        ("base_storage_dtype", "float32"),
    ]:
        bad = copy.deepcopy(ident)
        bad[key] = value
        with pytest.raises(ValueError, match="v4 adapter identity"):
            validate_adapter_identity(bad)
    for field in ("resolved", "dependencies"):
        for value in ([], None, 1):
            bad = copy.deepcopy(ident)
            bad[field] = value
            with pytest.raises(ValueError, match="resolved target"):
                validate_adapter_identity(bad)
    with pytest.raises(ValueError, match="bfloat16 output"):
        merged_export_metadata("unused", ident, artifact_record=manifest, execution_profile=profile)
    config.pop("cosmos_inference_profile")
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting"):
        deployment_profile(tmp_path)
    with pytest.raises(ValueError, match="not bound"):
        model_execution_profile(tmp_path)


def test_cpu_install_rejection_preserves_model(tmp_path):
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.tests.test_adapters_native import tiny_snapshot
    from cosmos_framework.utils.generator.bf16_cache import install

    tiny_snapshot(tmp_path / "base", tied=True)
    model = AutoModelForImageTextToText.from_pretrained(
        tmp_path / "base", dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    before = {n: (p, p.clone()) for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match="single A100"):
        install(model)
    assert not hasattr(model, "_cosmos_deployment_profile")
    assert all(p is before[n][0] and torch.equal(p, before[n][1]) for n, p in model.named_parameters())


def test_prefill_launch_is_bounded_to_qualified_geometry():
    from cosmos_framework.utils.generator.bf16_cache import _attention_warp_count

    vl = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl_text"))
    other = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text"))
    assert _attention_warp_count(vl, 8192, 128) == 1
    for model, length, dim in [(vl, 1, 128), (vl, 1023, 128), (vl, 2048, 64), (vl, 2048, 256), (other, 8192, 128)]:
        assert _attention_warp_count(model, length, dim) == 4


@pytest.mark.parametrize("mask_kind", ["none", "bool", "inf", "finite", "broadcast"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical kernels require CUDA")
def test_prefill_launch_matches_original_kernel_bitwise(mask_kind, monkeypatch):
    from cosmos_framework.utils.generator import bf16_cache

    torch.manual_seed(51109)
    m, n, d = 1025, 1049, 128
    # Non-contiguous inputs/cache views and partial query/key tiles.
    q = torch.randn(1, m * 2, 4, d, device="cuda", dtype=torch.bfloat16)[:, ::2].transpose(1, 2)
    k, v = [torch.randn(1, n * 2, 2, d, device="cuda", dtype=torch.bfloat16)[:, ::2].transpose(1, 2) for _ in range(2)]
    mask = None
    if mask_kind != "none":
        valid = torch.arange(n, device="cuda")[None, :] <= torch.arange(m, device="cuda")[:, None] + n - m
        valid[2] = False
        if mask_kind == "bool":
            mask = valid[None, None]
        elif mask_kind == "broadcast":
            mask = torch.tensor(True, device="cuda").expand(1, 1, 1, n)
        else:
            mask = torch.zeros((1, 1, m, n), device="cuda").masked_fill(
                ~valid, float("-inf") if mask_kind == "inf" else torch.finfo(torch.float32).min
            )
    module = SimpleNamespace(training=False, config=SimpleNamespace(model_type="qwen3_vl_text"))
    with torch.no_grad():
        actual, _ = bf16_cache._attention_forward(module, q, k, v, mask)
        monkeypatch.setattr(bf16_cache, "_attention_warp_count", lambda *_: 4)
        reference, _ = bf16_cache._attention_forward(module, q, k, v, mask)
    assert torch.equal(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical kernels require CUDA")
def test_linear_shape_consistency_and_fp64_reference():
    from cosmos_framework.utils.generator.bf16_cache import linear

    with pytest.raises(RuntimeError, match="no_grad"):
        linear(
            torch.ones(1, 64, device="cuda", dtype=torch.bfloat16),
            torch.ones(64, 64, device="cuda", dtype=torch.bfloat16),
        )
    torch.manual_seed(123)
    for k, n in [(129, 67), (512, 512)]:
        x = torch.randn(137, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            full = linear(x, w, b)
            split = torch.cat([linear(x[:3], w, b), linear(x[3:65], w, b), linear(x[65:], w, b)])
            assert torch.equal(full, split)
            reference = torch.nn.functional.linear(x.double(), w.double(), b.double()).bfloat16()
            torch.testing.assert_close(full, reference, atol=0.0625, rtol=0.008)
            assert linear(x[:0], w, b).shape == (0, n)


@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive", "finite_additive"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical kernels require CUDA")
def test_attention_masks_suffixes_and_fully_masked_rows(mask_kind):
    from cosmos_framework.utils.generator.bf16_cache import _attention_forward

    torch.manual_seed(345)
    q = torch.randn(1, 4, 137, 64, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 2, 137, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    mask = torch.ones(137, 137, device="cuda", dtype=torch.bool).tril()[None, None]
    if mask_kind != "none":
        mask[:, :, :, :4] = False
    provided = None if mask_kind == "none" else mask
    if mask_kind in ("additive", "finite_additive"):
        bias = float("-inf") if mask_kind == "additive" else torch.finfo(torch.float32).min
        provided = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, bias)
    module = SimpleNamespace(training=False)
    with torch.no_grad():
        full, _ = _attention_forward(module, q, k, v, provided)
        prefix, _ = _attention_forward(
            module, q[:, :, :65], k[:, :, :65], v[:, :, :65], None if provided is None else provided[:, :, :65, :65]
        )
        suffix, _ = _attention_forward(module, q[:, :, 65:], k, v, None if provided is None else provided[:, :, 65:])
        if mask_kind != "finite_additive":
            assert torch.equal(full, torch.cat([prefix, suffix], 1))
        # With finite biases a fully masked row still has a uniform softmax;
        # adding keys intentionally changes its reference denominator.
        ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q.double(),
                k.repeat_interleave(2, 1).double(),
                v.repeat_interleave(2, 1).double(),
                attn_mask=(mask if provided is None or provided.dtype == torch.bool else provided.double()),
            )
            .transpose(1, 2)
            .bfloat16()
        )
        torch.testing.assert_close(full, ref, atol=0, rtol=0)
        assert torch.isfinite(full).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="profile requires A100")
def test_portable_factor_ownership_and_inference_only_guards(tmp_path, monkeypatch):
    from transformers.cache_utils import DynamicCache, StaticCache

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.adapter_bf16_execution import install_adapter_inference_profile
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability
    from cosmos_framework.posttrain.model.hf import build_hf_model
    from cosmos_framework.posttrain.tests import test_adapters_native as fixtures
    from cosmos_framework.posttrain.tests.test_adapter_merged_weight import configuration
    from cosmos_framework.posttrain.tests.test_adapters_native import tiny_snapshot
    from cosmos_framework.posttrain.tests.test_model_qwen3_5 import _batch
    from cosmos_framework.utils.generator.bf16_cache import _inference_only

    cfg = fixtures._dense_config()
    cfg.text_config.hidden_size = cfg.vision_config.out_hidden_size = 128
    cfg.text_config.head_dim = 64
    cfg.text_config.num_attention_heads = 2
    cfg.text_config.num_key_value_heads = 1
    cfg.text_config.rope_parameters["mrope_section"] = [8, 12, 12]
    monkeypatch.setattr(fixtures, "_dense_config", lambda: cfg)
    tiny_snapshot(tmp_path / "base", tied=True)
    net = (
        build_hf_model(
            HFModelConfig(name_or_path=str(tmp_path / "base"), dtype="bfloat16", attn_impl="sdpa", grad_ckpt=False),
            adapters=configuration(),
        )
        .net.cuda()
        .eval()
    )
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad])
    assert_trainability(net, optimizer)
    before = {n: (p, p.clone(), p.requires_grad) for n, p in net.named_parameters()}
    factors = [
        (module, module.forward.__func__)
        for adapter in net.modules()
        if isinstance(adapter, PosttrainAdapterLinear)
        for module in (adapter.lora_A, adapter.lora_B)
    ]
    training_forward = net.model.forward
    changed = install_adapter_inference_profile(net, PROFILE)
    assert changed == install_adapter_inference_profile(net, PROFILE)
    assert not any("lora_A" in name or "lora_B" in name for name in changed)
    assert all(module.forward.__func__ is forward for module, forward in factors)
    assert all(
        p is before[n][0] and torch.equal(p, before[n][1]) and p.requires_grad == before[n][2]
        for n, p in net.named_parameters()
    )
    assert net.lm_head.weight is net.model.language_model.embed_tokens.weight
    assert_trainability(net, optimizer)
    # The explicit inference override removes only dummy modality executions;
    # real text/image/video/both batches still equal the ordered reference.
    inference_forward = net.model.forward
    calls = []
    hook = net.model.visual.register_forward_pre_hook(lambda _module, _args: calls.append(1))
    with torch.no_grad():
        for modality, count in [("text", 0), ("image", 1), ("video", 1), ("both", 2)]:
            data = {k: v.cuda() for k, v in _batch(net.config, modality, torch.bfloat16).items()}
            calls.clear()
            actual = net(**data, use_cache=False).logits
            assert len(calls) == count
            net.model.forward = training_forward
            calls.clear()
            expected = net(**data, use_cache=False).logits
            assert len(calls) == 2
            assert torch.equal(actual, expected)
            net.model.forward = inference_forward
            settings = dict(
                max_new_tokens=4,
                min_new_tokens=4,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
            calls.clear()
            cached = net.generate(**data, **settings, use_cache=True)
            assert len(calls) == count  # No dummy or repeated vision during decode.
            calls.clear()
            full = net.generate(**data, **settings, use_cache=False)
            assert len(calls) == 4 * count
            assert torch.equal(cached.sequences, full.sequences)
            assert all(torch.equal(a, b) for a, b in zip(cached.logits, full.logits, strict=True))
    hook.remove()
    with pytest.raises(RuntimeError, match="no_grad"):
        _inference_only(net, (), {})
    with torch.no_grad():
        _inference_only(net, (), {"past_key_values": DynamicCache(config=net.config)})
        with pytest.raises(ValueError, match="DynamicCache"):
            _inference_only(net, (), {"past_key_values": StaticCache(config=net.config, max_cache_len=64)})
        static = StaticCache(config=net.config, max_cache_len=64)
        with pytest.raises(ValueError, match="DynamicCache"):
            _inference_only(net, (torch.ones(1, 3, dtype=torch.long), None, None, static), {})
        cache = DynamicCache(config=net.config)
        with pytest.raises(ValueError, match="batch-one"):
            net(input_ids=torch.ones(2, 3, dtype=torch.long, device="cuda"), past_key_values=cache)
        assert cache.get_seq_length() == 0
        net.train()
        with pytest.raises(ValueError, match="eval"):
            install_adapter_inference_profile(net, PROFILE)
        with pytest.raises(RuntimeError, match="eval"):
            _inference_only(net, (), {})

    net.eval().bfloat16()
    with pytest.raises(ValueError, match="expected finite float32"):
        install_adapter_inference_profile(net, PROFILE)
