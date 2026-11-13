# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native generation accepts redundant processor metadata without changing math.

These are real small CPU models and processor/artifact controls, not checkpoint
or GPU qualification. Run under both supported Transformers environments.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections import OrderedDict

import numpy as np
import pytest
import torch
from transformers import AutoProcessor

from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
    consume_generation_mm_token_type_ids,
    validate_generation_mm_token_type_ids,
)
from cosmos_framework.posttrain.checkpoint.adapters import (
    WEIGHTS,
    adapter_configuration,
    load_adapter_artifact,
    read_adapter_artifact,
    save_adapter_artifact,
    write_adapter_artifact,
)
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model
from cosmos_framework.posttrain.tests.test_adapters_native import tiny_snapshot
from cosmos_framework.posttrain.tests.test_native_qwen_generation import _inputs, _net

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _metadata(ids, config, video_value=2):
    return (ids == config.image_token_id).long() + (ids == config.video_token_id).long() * video_value


def _generate(net, batch, *, cache=True):
    net.model.rope_deltas = None
    with torch.inference_mode():
        return net.generate(
            **batch,
            use_cache=cache,
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            suppress_tokens=[
                net.config.vision_start_token_id,
                net.config.vision_end_token_id,
                net.config.image_token_id,
                net.config.video_token_id,
            ],
            return_dict_in_generate=True,
            output_logits=True,
        )


def _assert_same(actual, expected):
    torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(actual.logits), torch.stack(expected.logits), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("modality", ["text", "image", "video"])
@pytest.mark.parametrize("cache", [False, True])
def test_processor_metadata_preserves_native_generation_and_ownership(family, override, modality, cache):
    net = _net(family, override)
    batch = _inputs(modality)
    before = {name: (id(value), value.detach().clone()) for name, value in net.named_parameters()}
    reference = _generate(net, batch, cache=cache)
    reference_rope = net.model.rope_deltas.clone()
    for video_value in (0, 2):
        metadata = _metadata(batch["input_ids"], net.config, video_value)
        inputs = {**batch, "mm_token_type_ids": metadata}
        saved = {key: value.clone() for key, value in inputs.items()}
        _assert_same(_generate(net, inputs, cache=cache), reference)
        torch.testing.assert_close(net.model.rope_deltas, reference_rope, atol=0, rtol=0)
        for key, value in inputs.items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    for name, value in net.named_parameters():
        owner, saved = before[name]
        assert id(value) == owner
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("invalid", ["dtype", "shape", "text_label", "media_label", "unknown_kwarg", "no_ids"])
def test_invalid_processor_metadata_and_unknown_kwargs_fail_before_forward(family, invalid):
    net = _net(family, True)
    batch = _inputs("image")
    batch["mm_token_type_ids"] = _metadata(batch["input_ids"], net.config)
    match = "mm_token_type_ids"
    if invalid == "dtype":
        batch["mm_token_type_ids"] = batch["mm_token_type_ids"].float()
    elif invalid == "shape":
        batch["mm_token_type_ids"] = batch["mm_token_type_ids"][:, :-1]
    elif invalid == "text_label":
        batch["mm_token_type_ids"][0, 0] = 3
    elif invalid == "media_label":
        batch["mm_token_type_ids"][0, 2] = 2
    elif invalid == "unknown_kwarg":
        batch["silently_discarded_processor_typo"] = torch.tensor([1])
        match = "silently_discarded_processor_typo"
    elif invalid == "no_ids":
        batch["inputs_embeds"] = net.get_input_embeddings()(batch.pop("input_ids"))
    before = {name: (id(value), value.clone()) for name, value in net.named_parameters()}
    calls = []
    handle = net.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError, match=match):
            _generate(net, batch)
    finally:
        handle.remove()
    assert not calls
    for name, value in net.named_parameters():
        owner, saved = before[name]
        assert id(value) == owner
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


def test_metadata_consumption_preserves_other_kwargs_and_requires_one_complete_encoding():
    net = _net("qwen3_vl", True)
    ids = torch.tensor([[1, 120, 121, 121]])
    metadata = _metadata(ids, net.config)
    kwargs = {"input_ids": ids, "mm_token_type_ids": metadata, "unknown": object()}
    result = consume_generation_mm_token_type_ids(net.config, None, kwargs)
    assert result.keys() == {"input_ids", "unknown"}
    assert result["input_ids"] is ids and result["unknown"] is kwargs["unknown"]
    assert kwargs["mm_token_type_ids"] is metadata
    assert consume_generation_mm_token_type_ids(net.config, ids, result) is result
    metadata[0, -1] = 0
    with pytest.raises(ValueError, match="disagrees"):
        validate_generation_mm_token_type_ids(net.config, ids, metadata)
    with pytest.raises(ValueError, match="integer tensor"):
        validate_generation_mm_token_type_ids(net.config, ids, metadata.tolist())
    with pytest.raises(ValueError, match="integer tensor"):
        validate_generation_mm_token_type_ids(net.config, ids, metadata.to("meta"))


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
def test_direct_generation_preparation_consumes_only_metadata(family):
    net = _net(family, True)
    batch = _inputs("image")
    metadata = _metadata(batch["input_ids"], net.config)
    prepared = net.prepare_inputs_for_generation(
        **batch, mm_token_type_ids=metadata, cache_position=torch.arange(batch["input_ids"].shape[1])
    )
    assert "mm_token_type_ids" not in prepared
    assert prepared["pixel_values"] is batch["pixel_values"]
    assert prepared["image_grid_thw"] is batch["image_grid_thw"]
    assert prepared["position_ids"] is None
    # Positional IDs at the once-only boundary are the same supported request.
    kwargs = {key: value for key, value in batch.items() if key != "input_ids"}
    kwargs["mm_token_type_ids"] = metadata
    actual, name, remaining = net._prepare_model_inputs(batch["input_ids"], None, kwargs)
    assert actual is batch["input_ids"] and name == "input_ids"
    assert "mm_token_type_ids" not in remaining
    assert kwargs["mm_token_type_ids"] is metadata


def _processor_batch(processor, modality):
    media = {}
    text = "Describe this input."
    if modality == "image":
        text = "<|vision_start|><|image_pad|><|vision_end|>" + text
        media["images"] = [np.arange(192, dtype=np.uint8).reshape(8, 8, 3)]
    elif modality == "video":
        text = "<|vision_start|><|video_pad|><|vision_end|>" + text
        media["videos"] = [np.arange(384, dtype=np.uint8).reshape(2, 8, 8, 3)]
        media["video_metadata"] = [{"fps": 2.0, "total_num_frames": 2, "frames_indices": [0, 1]}]
    return processor(text=[text], return_tensors="pt", return_mm_token_type_ids=True, **media)


def _check_public_prefix(net, batch):
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    # Use the actual public generation/prefix implementation with the CPU model;
    # constructor device admission belongs to installed GPU qualification.
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model = net
    runtime._video_prefix_cache = OrderedDict()
    runtime._video_prefix_cache_hits = 0
    runtime._video_prefix_cache_misses = 0
    options = dict(
        max_new_tokens=3,
        min_new_tokens=3,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        suppress_tokens=[
            net.config.vision_start_token_id,
            net.config.vision_end_token_id,
            net.config.image_token_id,
            net.config.video_token_id,
        ],
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    net.model.rope_deltas = None
    with torch.inference_mode():
        reference = runtime._generate_with_video_prefix_cache(
            batch, active_key=("synthetic-video",), capacity=0, generation_kwargs=options
        )
        first = runtime._generate_with_video_prefix_cache(
            batch, active_key=("synthetic-video",), capacity=2, generation_kwargs=options
        )
        saved = copy.deepcopy(next(iter(runtime._video_prefix_cache.values()))["past_key_values"])
        repeated = runtime._generate_with_video_prefix_cache(
            batch, active_key=("synthetic-video",), capacity=2, generation_kwargs=options
        )
    _assert_same(first, repeated)
    torch.testing.assert_close(first.sequences, reference.sequences, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(first.logits), torch.stack(reference.logits), rtol=1e-5, atol=5e-7)
    live = next(iter(runtime._video_prefix_cache.values()))["past_key_values"]
    assert live.get_seq_length() == saved.get_seq_length()
    for actual, expected in zip(live.layers, saved.layers, strict=True):
        torch.testing.assert_close(actual.keys, expected.keys, rtol=0, atol=0)
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    assert runtime._video_prefix_cache_misses == runtime._video_prefix_cache_hits == 1


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_vl_moe"])
def test_public_native_prefix_reuse_admits_processor_metadata(family):
    net = _net(family, True)
    batch = _inputs("video")
    batch["mm_token_type_ids"] = _metadata(batch["input_ids"], net.config)
    _check_public_prefix(net, batch)


@pytest.mark.parametrize("tied", [False, True])
def test_real_processor_native_portable_reload_and_strict_artifacts(tmp_path, tied):
    base = tmp_path / "base"
    tiny_snapshot(base, tied=tied)
    # The tiny checkpoint must declare real multimodal RoPE, just like both
    # admitted dense native model bindings. This is part of its base identity.
    config_path = base / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["rope_scaling"] = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    config_path.write_text(json.dumps(config))
    model_config = Qwen3VLModelConfig(
        name_or_path=str(base), dtype="float32", activation_checkpointing="none", attn_impl="sdpa"
    )
    settings = AdapterConfig(
        components=[
            AdapterComponentConfig(
                component="language", targets=["model.language_model.layers.0.self_attn.q_proj"], rank=2
            )
        ]
    )
    original = build_qwen3_vl_model(model_config, adapters=settings).net.eval()
    with torch.no_grad():
        original.model.language_model.layers[0].self_attn.q_proj.lora_B.weight.add_(0.025)
    portable = tmp_path / "portable"
    save_adapter_artifact(original, str(portable))
    identity, state = read_adapter_artifact(str(portable))
    warm_start = adapter_configuration(identity)
    warm_start.warm_start = str(portable)
    reloaded = build_qwen3_vl_model(model_config, adapters=warm_start).net.eval()
    assert reloaded._posttrain_adapter_identity == original._posttrain_adapter_identity
    for name, value in original.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)
    if tied:
        assert reloaded.lm_head.weight is reloaded.get_input_embeddings().weight
    processor = AutoProcessor.from_pretrained(base)
    for modality in ("text", "image", "video"):
        batch = _processor_batch(processor, modality)
        assert "mm_token_type_ids" in batch
        reference = _generate(original, {key: value for key, value in batch.items() if key != "mm_token_type_ids"})
        _assert_same(_generate(original, batch), reference)
        _assert_same(_generate(reloaded, batch), reference)
        if modality == "video":
            _check_public_prefix(reloaded, batch)

    # Metadata admission must not loosen artifact identity or completeness.
    wrong = copy.deepcopy(identity)
    wrong["media"]["test_incompatible_preprocessing"] = True
    write_adapter_artifact(str(tmp_path / "wrong"), wrong, state)
    before = {name: (id(value), value.detach().clone()) for name, value in reloaded.named_parameters()}
    with pytest.raises(ValueError, match="compatibility mismatch"):
        load_adapter_artifact(reloaded, str(tmp_path / "wrong"))
    shutil.copytree(portable, tmp_path / "incomplete")
    (tmp_path / "incomplete" / WEIGHTS).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_adapter_artifact(reloaded, str(tmp_path / "incomplete"))
    for name, value in reloaded.named_parameters():
        owner, saved = before[name]
        assert id(value) == owner
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
