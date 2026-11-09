# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shape-independent normalization and bounded tensor-core prefill execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.utils.generator.bf16_cache_contract import CORRECTED_PROFILE, FAST_PROFILE


@pytest.mark.parametrize("profile", [CORRECTED_PROFILE, FAST_PROFILE])
def test_merged_environment_preflight_rejects_other_gpu_without_weight_loading(monkeypatch, profile):
    from cosmos_framework.inference.reasoner.precision import validate_deployment_environment

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 5))
    with pytest.raises(ValueError, match="A100.*UUID"):
        validate_deployment_environment(profile, "cuda:0")
    with pytest.raises(ValueError, match="A100"):
        validate_deployment_environment(profile, "cpu")


def test_shared_guard_preserves_qwen3_5_legacy_profile(monkeypatch):
    from inspect import signature

    from cosmos_framework.utils.generator.bf16_cache import _inference_only
    from cosmos_framework.utils.generator.bf16_cache_contract import QWEN3_5_PROFILE

    def forward(input_ids=None, past_key_values=None):
        pass

    model = SimpleNamespace(
        training=False,
        _cosmos_deployment_profile=QWEN3_5_PROFILE,
        _cosmos_forward_signature=signature(forward),
    )  # The legacy Qwen3.5 installer owns no v2 attention-signature attribute.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.no_grad():
        _inference_only(model, (), {"input_ids": torch.ones(1, 3, dtype=torch.long)})


@pytest.mark.parametrize("profile", [CORRECTED_PROFILE, FAST_PROFILE])
def test_public_export_cli_and_api_admit_versioned_profiles(monkeypatch, tmp_path, profile):
    from cosmos_framework.posttrain.checkpoint.tools import adapters

    # Both public admission points must reach artifact validation, not reject
    # a profile accepted only by lower-level metadata serialization.
    class ReachedArtifactValidation(Exception):
        pass

    def read(*args):
        raise ReachedArtifactValidation

    monkeypatch.setattr(adapters, "read_adapter_snapshot", read)
    with pytest.raises(ReachedArtifactValidation):
        adapters.export_merged_adapter(
            "adapter", "base", str(tmp_path / "export"), output_dtype="bfloat16", execution_profile=profile
        )
    with pytest.raises(ReachedArtifactValidation):
        adapters.main(
            [
                "--adapter",
                "adapter",
                "--base",
                "base",
                "--merged",
                "--out",
                str(tmp_path / "export"),
                "--output-dtype",
                "bfloat16",
                "--execution-profile",
                profile,
            ]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="qualified kernels require A100")
def test_real_rounding_boundary_norm_is_independent_of_batch_shape():
    from cosmos_framework.utils.generator.bf16_cache import _rms_norm_forward

    # Layer 14 post-attention input at video-961109's third decision, before
    # generated histories diverge. No model weights other than this norm vector.
    data = torch.load(Path(__file__).parent / "fixtures/canonical_rmsnorm_shape.pt", weights_only=True)
    x, weight = [data[key].cuda() for key in ("input", "weight")]
    module = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    with torch.no_grad():
        single = _rms_norm_forward(module, x)
        for rows in (2, 32, 138, 2048):
            repeated = x.expand(1, rows, -1).contiguous()
            assert torch.equal(_rms_norm_forward(module, repeated), single.expand_as(repeated))
        ref = (x.double() * (x.double().square().mean(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16() * weight
        torch.testing.assert_close(single, ref, atol=0.0078125, rtol=0.008)
        module.weight = weight.float()  # Portable separately-trained norm storage.
        assert torch.equal(_rms_norm_forward(module, x), single)
    with pytest.raises(RuntimeError, match="no_grad"):
        _rms_norm_forward(module, x)


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive", "finite", "broadcast"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="qualified kernels require A100")
def test_tensorcore_prefill_and_cached_suffix_against_reference(mask_kind):
    from cosmos_framework.utils.generator.bf16_cache import _attention_forward, _tensorcore_attention_forward

    torch.manual_seed(961110)
    m, n, d = 1025, 1057, 128
    q = torch.randn(1, m * 2, 4, d, device="cuda", dtype=torch.bfloat16)[:, ::2].transpose(1, 2)
    k, v = [torch.randn(1, n * 2, 2, d, device="cuda", dtype=torch.bfloat16)[:, ::2].transpose(1, 2) for _ in range(2)]
    mask = None
    if mask_kind != "none":
        valid = torch.arange(n, device="cuda")[None, :] <= torch.arange(m, device="cuda")[:, None] + n - m
        valid[2] = False
        if mask_kind == "boolean":
            mask = valid[None, None]
        elif mask_kind == "broadcast":
            mask = torch.ones(1, 1, 1, n, device="cuda", dtype=torch.bool)
        else:
            bias = float("-inf") if mask_kind == "additive" else torch.finfo(torch.float32).min
            mask = torch.zeros(1, 1, m, n, device="cuda").masked_fill(~valid, bias)
    module = SimpleNamespace(training=False, config=SimpleNamespace(model_type="qwen3_vl_text"))
    with torch.no_grad():
        actual, _ = _tensorcore_attention_forward(module, q, k, v, mask)
        reference, _ = _attention_forward(module, q, k, v, mask)
        assert torch.equal(actual, reference)
        suffix_mask = None if mask is None else mask[:, :, -3:]
        suffix, _ = _tensorcore_attention_forward(module, q[:, :, -3:], k, v, suffix_mask)
        assert torch.equal(suffix, reference[:, -3:])
        with pytest.raises(ValueError, match="batch-one causal"):
            _tensorcore_attention_forward(module, q, k, v, mask, dropout=0.1)
        with pytest.raises(ValueError, match="batch-one causal"):
            _tensorcore_attention_forward(module, q, k, v, mask, is_causal=False)


@pytest.mark.parametrize("profile", [CORRECTED_PROFILE, FAST_PROFILE])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="qualified profiles require A100")
def test_profile_conflicts_and_live_attention_changes_precede_cache_mutation(tmp_path, profile, monkeypatch):
    from transformers import AutoModelForImageTextToText
    from transformers.cache_utils import DynamicCache, StaticCache

    from cosmos_framework.posttrain.tests import test_adapters_native as fixtures
    from cosmos_framework.utils.generator.bf16_cache import install

    cfg = fixtures._dense_config()
    cfg.text_config.hidden_size = cfg.vision_config.out_hidden_size = 128
    cfg.text_config.head_dim = 64
    cfg.text_config.num_attention_heads = 2
    cfg.text_config.num_key_value_heads = 1
    cfg.text_config.rope_parameters["mrope_section"] = [8, 12, 12]
    monkeypatch.setattr(fixtures, "_dense_config", lambda: cfg)
    fixtures.tiny_snapshot(tmp_path / "base", tied=True)
    net = (
        AutoModelForImageTextToText.from_pretrained(tmp_path / "base", dtype=torch.bfloat16, attn_implementation="sdpa")
        .eval()
        .cuda()
    )
    install(net, profile=profile)
    assert install(net, profile=profile)
    with pytest.raises(ValueError, match="different deployment"):
        install(net, profile=FAST_PROFILE if profile == CORRECTED_PROFILE else CORRECTED_PROFILE)
    cache = DynamicCache(config=net.config)
    batch = dict(input_ids=torch.ones(1, 3, dtype=torch.long, device="cuda"))
    net.set_attn_implementation({"vision_config": "eager"})
    with torch.no_grad(), pytest.raises(ValueError, match="attention execution changed"):
        net(**batch, past_key_values=cache)
    assert cache.get_seq_length() == 0
    net.set_attn_implementation({"vision_config": "sdpa"})
    net.set_attn_implementation("eager")
    with torch.no_grad(), pytest.raises(ValueError, match="attention execution changed"):
        net(**batch, past_key_values=cache)
    assert cache.get_seq_length() == 0
    with torch.no_grad(), pytest.raises(ValueError, match="DynamicCache"):
        net(**batch, past_key_values=StaticCache(config=net.config, max_cache_len=16))
