# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical recurrent arithmetic, request-local cache ownership and profile binding."""

import json
from collections import OrderedDict

import pytest
import torch

from cosmos_framework.utils.generator.bf16_cache_contract import QWEN3_5_PROFILE as PROFILE


def test_q35_profile_identity(tmp_path):
    from cosmos_framework.inference.reasoner.model_execution import model_execution_profile
    from cosmos_framework.inference.reasoner.precision import deployment_profile
    from cosmos_framework.posttrain.checkpoint.adapters import merged_export_metadata
    from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype
    from cosmos_framework.posttrain.tests.test_adapter_canonical_cache import identity

    ident = identity()
    ident["architecture"] = "qwen3_5"
    manifest = {"kind": "portable_adapter", "identity": ident, "weights_sha256": "digest"}
    metadata = merged_export_metadata(
        "unused", ident, artifact_record=manifest, output_dtype="bfloat16", execution_profile=PROFILE
    )
    config = config_with_export_dtype(
        {"model_type": "qwen3_5"}, "bfloat16", execution_profile=PROFILE, model_execution="merged_weight_bf16_v1"
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".materialized").write_text(
        json.dumps({"identity": {"output_dtype": "bfloat16", "export_metadata": metadata}})
    )
    assert deployment_profile(tmp_path) == PROFILE
    assert model_execution_profile(tmp_path) == "merged_weight_bf16_v1"
    with pytest.raises(ValueError, match="compatible"):
        config_with_export_dtype(
            {"model_type": "qwen3_vl"}, "bfloat16", execution_profile=PROFILE, model_execution="merged_weight_bf16_v1"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical recurrent kernel requires A100")
def test_recurrence_split_is_exact_and_matches_independent_reference():
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

    from cosmos_framework.utils.generator.qwen3_5_cache import _recurrent

    torch.manual_seed(951107)
    with torch.no_grad():
        q, k, v = [torch.randn(1, 2, 137, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3)]
        g = -torch.rand(1, 137, 2, device="cuda")
        beta = torch.rand_like(g).bfloat16()
        full, state = _recurrent(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
        first, previous = _recurrent(
            q[:, :65],
            k[:, :65],
            v[:, :65],
            g[:, :65],
            beta[:, :65],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        last, final = _recurrent(
            q[:, 65:],
            k[:, 65:],
            v[:, 65:],
            g[:, 65:],
            beta[:, 65:],
            initial_state=previous,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert torch.equal(torch.cat([first, last], 1), full) and torch.equal(final, state)
        query, key = l2norm(q).double(), l2norm(k).double()
        ref = torch.zeros_like(state, dtype=torch.float64)
        outputs = []
        for t in range(137):
            ref = ref * g[:, t, :, None, None].double().exp()
            delta = (v[:, t].double() - (ref * key[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None].double()
            ref = (ref + key[:, t, :, :, None] * delta[:, :, None, :]).float().double()
            outputs.append((ref * (query[:, t, :, :, None] * (128**-0.5))).sum(-2).bfloat16())
        torch.testing.assert_close(state, ref.float(), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(full, torch.stack(outputs, 1), atol=0, rtol=0)


@pytest.mark.parametrize("modality", ["text", "image", "video", "both"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical cache requires A100")
def test_request_local_cache_full_incremental_and_repeated_prefix(modality):
    from transformers import AutoModelForImageTextToText
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer, StaticCache

    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
    from cosmos_framework.posttrain.tests.test_model_qwen3_5 import _batch, config
    from cosmos_framework.utils.generator.qwen3_5_cache import install

    torch.manual_seed(451)
    cfg = config.__wrapped__()
    cfg.text_config.head_dim = 64
    cfg.text_config.rope_parameters["mrope_section"] = [8, 12, 12]
    cfg.text_config.linear_key_head_dim = cfg.text_config.linear_value_head_dim = 128
    net = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    source_lazy = LinearAttentionLayer.lazy_initialization
    before = {n: (p, p.clone(), p.requires_grad) for n, p in net.named_parameters()}
    changed = install(net)
    assert changed == install(net)
    assert all(
        p is before[n][0] and torch.equal(p, before[n][1]) and p.requires_grad == before[n][2]
        for n, p in net.named_parameters()
    )
    assert LinearAttentionLayer.lazy_initialization is source_lazy
    inputs = {k: v.cuda() for k, v in _batch(cfg, modality, torch.bfloat16).items()}
    settings = dict(
        max_new_tokens=4,
        min_new_tokens=4,
        eos_token_id=2,
        pad_token_id=0,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
        suppress_tokens=[cfg.image_token_id, cfg.video_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id],
    )
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model = net
    runtime._video_prefix_cache = OrderedDict()
    runtime._video_prefix_cache_hits = runtime._video_prefix_cache_misses = 0
    with torch.no_grad():
        defaults = net(**inputs, use_cache=None)
        assert all(
            layer.recurrent_states.dtype == torch.float32
            for layer in defaults.past_key_values.layers
            if type(layer) is LinearAttentionLayer
        )
        full = net.generate(**inputs, **settings, use_cache=False)
        cached = net.generate(**inputs, **settings, use_cache=True)
        assert torch.equal(full.sequences, cached.sequences)
        for x, y in zip(full.logits, cached.logits, strict=True):
            torch.testing.assert_close(x, y, atol=0.005, rtol=0.005)
        recurrent = [
            layer.recurrent_states for layer in cached.past_key_values.layers if type(layer) is LinearAttentionLayer
        ]
        assert recurrent and all(t.dtype == torch.float32 for t in recurrent)
        if "pixel_values_videos" in inputs:
            original = runtime._fork_dynamic_cache
            parents = []

            def observe(parent):
                states = [
                    (layer, layer.recurrent_states.clone())
                    for layer in parent.layers
                    if type(layer) is LinearAttentionLayer
                ]
                parents.extend(states)
                return original(parent)

            runtime._fork_dynamic_cache = observe
            for _ in range(3):
                branch = runtime._generate_with_video_prefix_cache(
                    inputs, active_key=("fixture", modality), capacity=2, generation_kwargs=settings
                )
                assert torch.equal(branch.sequences, cached.sequences)
                for x, y in zip(branch.logits, cached.logits, strict=True):
                    torch.testing.assert_close(x, y, atol=0.005, rtol=0.005)
                assert all(torch.equal(layer.recurrent_states, old) for layer, old in parents)
        foreign = DynamicCache(config=cfg)
        foreign.layers[0].lazy_initialization(conv_states=torch.zeros(1, 512, 4, device="cuda", dtype=torch.bfloat16))
        snapshot = foreign.layers[0].conv_states.clone()
        with pytest.raises(ValueError, match="untagged"):
            net(**inputs, past_key_values=foreign)
        assert torch.equal(snapshot, foreign.layers[0].conv_states)
        with pytest.raises(ValueError, match="DynamicCache"):
            net(inputs["input_ids"], None, None, StaticCache(config=cfg, max_cache_len=64))
        for request in (
            {},
            {
                "input_ids": torch.ones(1, 3, device="cuda", dtype=torch.long),
                "inputs_embeds": torch.zeros(1, 3, cfg.text_config.hidden_size, device="cuda", dtype=torch.bfloat16),
            },
        ):
            untouched = DynamicCache(config=cfg)
            with pytest.raises(ValueError, match="exactly one"):
                net(**request, past_key_values=untouched, use_cache=True)
            assert not hasattr(untouched, "_cosmos_canonical_owner")
            assert not any(
                layer.is_recurrent_states_initialized
                for layer in untouched.layers
                if type(layer) is LinearAttentionLayer
            )
        empty = DynamicCache(config=cfg)
        with pytest.raises(ValueError, match="batch-one"):
            net(input_ids=torch.ones(2, 3, device="cuda", dtype=torch.long), past_key_values=empty)
        assert not any(
            layer.is_recurrent_states_initialized for layer in empty.layers if type(layer) is LinearAttentionLayer
        )
