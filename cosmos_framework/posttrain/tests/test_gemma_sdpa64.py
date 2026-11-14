# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SDPA64 contract controls; the real Gemma2B campaign remains a separate gate."""

import copy

import pytest
import torch

from cosmos_framework.posttrain.tests.test_gemma_fixed_mma import cpu_projection  # noqa: F401
from cosmos_framework.posttrain.tests.test_gemma_sdpa_layout import identity, tiny  # noqa: F401
from cosmos_framework.utils.generator import gemma_sdpa_layout as profile


@pytest.mark.parametrize("length", [17, 32, 65])
def test_fp64_attention_independent_reference_fp32_cache_and_owned_prefix(tiny, cpu_projection, monkeypatch, length):  # noqa: F811
    native = torch.nn.functional.scaled_dot_product_attention
    observed = []

    def independent(q, k, v, **kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.float64
        mask = kwargs.get("attn_mask")
        if mask is not None:
            assert mask.dtype in (torch.bool, torch.float64)
        assert kwargs["dropout_p"] == 0.0
        result = native(q, k, v, **kwargs)
        # An independent explicit FP64 score/softmax/value expression also
        # exercises asymmetric full-mask, incremental and split-prefix calls.
        repeats = q.shape[1] // k.shape[1]
        kk, vv = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        scores = q @ kk.transpose(-1, -2) * kwargs["scale"]
        if kwargs.get("is_causal"):
            causal = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        expected = scores.softmax(-1) @ vv
        torch.testing.assert_close(result, expected, atol=2e-13, rtol=2e-13)
        observed.append((q.shape, k.shape))
        return result

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", independent)
    owned = profile._ownership(tiny)
    before = {name: module.forward for name, module in tiny.named_modules()}
    profile.install(tiny, identity=identity(), profile=profile.SDPA64_PROFILE)
    ids = torch.arange(length + 4)[None].remainder(tiny.config.vocab_size)
    with torch.no_grad():
        full = tiny(ids, use_cache=False).logits
        prefix = tiny(ids[:, :length], use_cache=True).past_key_values
        tensors = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
        assert all(k.dtype == v.dtype == torch.float32 and k.shape[1] == 1 for k, v in tensors)
        suffix = tiny(ids[:, length:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        repeated = tiny(ids[:, length:], past_key_values=copy.deepcopy(prefix), use_cache=True).logits
        cache = copy.deepcopy(prefix)
        incremental = []
        for offset in range(length, ids.shape[1]):
            value = tiny(ids[:, offset : offset + 1], past_key_values=cache, use_cache=True)
            incremental.append(value.logits)
            cache = value.past_key_values
    assert full.dtype == suffix.dtype == torch.float32
    torch.testing.assert_close(full[:, length:], suffix, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.cat(incremental, dim=1), suffix, atol=2e-6, rtol=2e-5)
    assert torch.equal(suffix, repeated)
    assert all(
        torch.equal(layer.keys, k) and torch.equal(layer.values, v) for layer, (k, v) in zip(prefix.layers, tensors)
    )
    assert profile._ownership(tiny) == owned
    assert observed and cpu_projection
    profile.restore(tiny)
    assert all(tiny.get_submodule(name).forward == function for name, function in before.items())


@pytest.mark.parametrize("previous", [profile.PROFILE, profile.FIXED_MMA_PROFILE])
def test_legacy_arithmetic_and_cache_identity_are_not_reinterpreted(tiny, cpu_projection, previous):  # noqa: F811
    profile.install(tiny, identity=identity(), profile=previous)
    assert tiny.model.layers[0].self_attn.forward.__func__ is profile._attention
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="conflicting"):
        profile.install(tiny, identity=identity(), profile=profile.SDPA64_PROFILE)
    profile.restore(tiny)
    profile.install(tiny, identity=identity(), profile=profile.SDPA64_PROFILE)
    assert tiny.model.layers[0].self_attn.forward.__func__ is profile._attention_fp64
    with torch.no_grad(), pytest.raises(ValueError, match="prefix from this installation"):
        tiny(torch.tensor([[4]]), past_key_values=prefix)


def test_profile_tag_precision_and_mutation_guards(tiny, cpu_projection, tmp_path):  # noqa: F811
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata
    from cosmos_framework.utils.generator.portable_adapters import PortableAdapterConfig

    selected = profile.SDPA64_PROFILE
    assert PortableAdapterConfig(str(tmp_path), selected).execution_profile == selected
    with pytest.raises(ValueError, match="effective-weight cache is unsupported"):
        PortableAdapterConfig(str(tmp_path), selected, cache_effective_weights=True)
    record = {"kind": "portable_adapter", "identity": identity(), "weights_sha256": "a" * 64}
    metadata = merged_export_metadata("unused", identity(), artifact_record=record, execution_profile=selected)
    assert metadata["deployment_precision"]["equivalence"] == profile.export_equivalence(selected)
    assert metadata["deployment_precision"]["equivalence"] != profile.export_equivalence(profile.FIXED_MMA_PROFILE)
    with pytest.raises(ValueError, match="float32"):
        merged_export_metadata(
            "unused", identity(), artifact_record=record, output_dtype="bfloat16", execution_profile=selected
        )
    profile.install(tiny, identity=identity(), profile=selected)
    with pytest.raises(RuntimeError, match="eager eval"):
        tiny(torch.tensor([[1, 2]]))
    with torch.no_grad():
        prefix = tiny(torch.tensor([[1, 2]]), use_cache=True).past_key_values
        with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError, match="autocast"):
            tiny(torch.tensor([[3]]), past_key_values=copy.deepcopy(prefix))
        tiny.model.layers[0].self_attn.scaling *= 2
        with pytest.raises(ValueError, match="attention settings changed"):
            tiny(torch.tensor([[3]]), past_key_values=prefix)
