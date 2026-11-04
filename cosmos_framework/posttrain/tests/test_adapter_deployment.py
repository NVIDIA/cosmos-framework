# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Deployment binding, unchanged tensor ownership, and actual cached multimodal inference."""

import json

import pytest
import torch

from cosmos_framework.inference.reasoner.precision import PROFILE, deployment_profile, install_deployment_profile


def _artifact(path):
    config = {"model_type": "qwen3_vl", "dtype": "bfloat16", "cosmos_inference_profile": PROFILE}
    marker = {
        "identity": {
            "output_dtype": "bfloat16",
            "export_metadata": {
                "format": "cosmos-merged-adapter-v1",
                "deployment_precision": {"dtype": "bfloat16", "execution_profile": PROFILE},
            },
        }
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / ".materialized").write_text(json.dumps(marker))
    return config, marker


def test_profile_requires_config_and_transaction_binding(tmp_path):
    config, marker = _artifact(tmp_path)
    assert deployment_profile(str(tmp_path)) == PROFILE
    config.pop("cosmos_inference_profile")
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting"):
        deployment_profile(str(tmp_path))
    config["cosmos_inference_profile"] = PROFILE
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / ".materialized").unlink()
    with pytest.raises(ValueError, match="missing"):
        deployment_profile(str(tmp_path))
    marker["identity"]["output_dtype"] = "float32"
    (tmp_path / ".materialized").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="committed dense"):
        deployment_profile(str(tmp_path))


@pytest.mark.parametrize("legacy_marker", [None, "old completion marker", "null", "[]", "{}"])
def test_untagged_models_keep_their_existing_load_contract(tmp_path, legacy_marker):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl"}')
    if legacy_marker is not None:
        (tmp_path / ".materialized").write_text(legacy_marker)
    assert deployment_profile(str(tmp_path)) is None


@pytest.mark.parametrize("dtype,attention", [("float32", "sdpa"), ("bfloat16", "eager")])
def test_runtime_rejects_conflicting_precision_before_loading(tmp_path, dtype, attention):
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    _artifact(tmp_path)
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_path, runtime.dtype, runtime.attn_implementation = str(tmp_path), dtype, attention
    with pytest.raises(ValueError, match="requires BF16 residual FP32"):
        runtime._load()


def test_profile_install_preflight_does_not_mutate_cpu_model(tmp_path):
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.tests.test_adapters_native import tiny_snapshot

    source = tmp_path / "source"
    tiny_snapshot(source, tied=True)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    before = {name: (parameter, parameter.detach().clone()) for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="one CUDA device"):
        install_deployment_profile(model, PROFILE)
    assert not hasattr(model, "_cosmos_deployment_profile")
    assert all(p is before[n][0] and torch.equal(p, before[n][1]) for n, p in model.named_parameters())
    model.config.vision_config._attn_implementation = "eager"
    with pytest.raises(ValueError, match="SDPA"):
        install_deployment_profile(model, PROFILE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16-input/FP32-output GEMM requires CUDA")
def test_profile_cached_multimodal_generation_ties_and_reload(tmp_path):
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.tests.adapter_numerics import HFProbePolicy, configure_qualification_numerics
    from cosmos_framework.posttrain.tests.test_adapters_native import batches, tiny_snapshot

    configure_qualification_numerics()
    torch.manual_seed(42)
    source = tmp_path / "source"
    tiny_snapshot(source, tied=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda()
        .eval()
    )
    before = {name: (parameter, parameter.detach().clone()) for name, parameter in model.named_parameters()}
    changed = install_deployment_profile(model, PROFILE)
    assert changed == install_deployment_profile(model, PROFILE)
    assert len(model._forward_pre_hooks) == 1
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    fixtures = batches()
    fixtures[0]["input_ids"] = torch.tensor([[30, 31, 32, 33, 34]])
    for fixture, token in zip(fixtures[1:], (model.config.image_token_id, model.config.video_token_id), strict=True):
        fixture["input_ids"] = torch.tensor(
            [[30, model.config.vision_start_token_id, token, model.config.vision_end_token_id, 31]]
        )
        fixture["mm_token_type_ids"] = torch.tensor([[0, 0, 1 if token == model.config.image_token_id else 2, 0, 0]])
    inputs = [HFProbePolicy(model).prepare_policy_batch(batch) for batch in fixtures]
    with pytest.raises(RuntimeError, match="no_grad"):
        model(**inputs[0])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="no autocast"):
            model(**inputs[0])
    outputs = []
    with torch.no_grad():
        for data in inputs:
            output = model(**data, use_cache=False).logits
            assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
            outputs.append(output)
            # A random tiny model has no learned text vocabulary: prevent it
            # from inventing media placeholders with no corresponding pixels.
            generation = dict(
                max_new_tokens=3,
                do_sample=False,
                suppress_tokens=[
                    model.config.image_token_id,
                    model.config.video_token_id,
                    model.config.vision_start_token_id,
                    model.config.vision_end_token_id,
                ],
            )
            cached = model.generate(**data, **generation, use_cache=True)
            full = model.generate(**data, **generation, use_cache=False)
            assert torch.equal(cached, full)
        from collections import OrderedDict

        from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

        runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
        runtime.model, runtime.rank = model, 0
        runtime._video_prefix_cache = OrderedDict()
        runtime._video_prefix_cache_hits = runtime._video_prefix_cache_misses = 0
        for _ in range(2):
            branch = runtime._generate_with_video_prefix_cache(
                inputs[-1], active_key=("fixed-video",), capacity=2, generation_kwargs=generation
            )
            assert torch.equal(branch, full)
        assert runtime._video_prefix_cache_hits == runtime._video_prefix_cache_misses == 1
    assert all(p is before[n][0] and torch.equal(p, before[n][1]) for n, p in model.named_parameters())
    target = tmp_path / "roundtrip"
    model.save_pretrained(target)
    reloaded = (
        AutoModelForImageTextToText.from_pretrained(target, dtype="auto", attn_implementation="sdpa").cuda().eval()
    )
    install_deployment_profile(reloaded, PROFILE)
    assert reloaded.lm_head.weight is reloaded.model.language_model.embed_tokens.weight
    with torch.no_grad():
        for data, expected in zip(inputs, outputs, strict=True):
            torch.testing.assert_close(reloaded(**data, use_cache=False).logits, expected, rtol=0, atol=0)
