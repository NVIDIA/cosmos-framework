# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-device HF Qwen2/2.5-VL adapters through actual multimodal blocks."""

import copy

import pytest
import torch
import transformers
from safetensors.torch import save_file

from cosmos_framework.posttrain.checkpoint.adapters import prepare_merged_state, save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.test_adapters_native import batches


def small_config(family):
    vision = (
        dict(embed_dim=32, hidden_size=32, mlp_ratio=2)
        if family == "qwen2_vl"
        else dict(hidden_size=32, out_hidden_size=32, intermediate_size=64, fullatt_block_indexes=[0], window_size=8)
    )
    return transformers.AutoConfig.for_model(
        family,
        text_config=dict(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(depth=1, num_heads=4, patch_size=2, temporal_patch_size=2, spatial_merge_size=2, **vision),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl"])
@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
@pytest.mark.parametrize("ac", [False, True])
def test_all_components_image_video_text_updates_warm_reload_and_merge(tmp_path, family, method, rslora, ac):
    torch.manual_seed(51)
    source = transformers.AutoModelForImageTextToText.from_config(small_config(family), attn_implementation="sdpa")
    base = tmp_path / "base"
    source.save_pretrained(base)
    # These fixtures use the current materialized HF namespace. The original
    # public Qwen2-VL checkpoint namespace and its processor remain unqualified
    # for this SFT workflow; family binding alone does not claim that support.
    save_file({n: t.contiguous() for n, t in source.state_dict().items()}, str(base / "model.safetensors"))
    source.eval()
    config = AdapterConfig(
        components=[
            Component(component=component, targets=targets, method=method, use_rslora=rslora, rank=2)
            for component, targets in [
                (
                    "language",
                    [
                        "model.language_model.layers.*.self_attn.q_proj",
                        "model.language_model.layers.*.self_attn.v_proj",
                    ],
                ),
                ("projector", ["model.visual.merger.mlp.0", "model.visual.merger.mlp.2"]),
                ("vision", ["model.visual.blocks.0.attn.qkv"]),
            ]
        ]
    )
    model_config = HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa", grad_ckpt=ac)
    model = build_hf_model(model_config, adapters=config)
    net = model.net
    net.eval()
    data = batches()
    if family == "qwen2_vl":
        for batch in data:
            tokens = torch.zeros_like(batch["input_ids"])
            tokens[batch["input_ids"] == source.config.image_token_id] = 1
            tokens[batch["input_ids"] == source.config.video_token_id] = 2
            batch["mm_token_type_ids"] = tokens
    with torch.no_grad():
        for batch in data:
            torch.testing.assert_close(net(**batch).logits, source(**batch).logits, atol=2e-6, rtol=2e-5)
    manifest = net._posttrain_adapter_manifest
    assert {n for n, m in net.named_modules() if isinstance(m, PosttrainAdapterLinear)} == {
        r["name"] for r in manifest["targets"]
    }
    initial = copy.deepcopy(net.state_dict())
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.01)
    assert_trainability(net, opt)
    net.train()
    for batch in data * 2:
        opt.zero_grad(set_to_none=True)
        loss = net(**batch).logits.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        # Text-only single-device batches do not call the vision tower; all
        # advertised multimodal target gradients must exist on image/video rows.
        if "pixel_values" in batch or "pixel_values_videos" in batch:
            assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
        opt.step()
    for name, parameter in net.named_parameters():
        assert torch.equal(parameter, initial[name]) is not parameter.requires_grad, name
    net.eval()
    with torch.no_grad():
        expected = [net(**batch).logits for batch in data]
    directory = tmp_path / "portable"
    save_adapter_artifact(net, str(directory))
    config.warm_start = str(directory)
    restored = build_hf_model(model_config, adapters=config).net.eval()
    source.load_state_dict(prepare_merged_state(net.state_dict(), net._posttrain_adapter_identity), strict=True)
    with torch.no_grad():
        for batch, reference in zip(data, expected, strict=True):
            torch.testing.assert_close(restored(**batch).logits, reference, atol=0, rtol=0)
            torch.testing.assert_close(source(**batch).logits, reference, atol=5e-6, rtol=5e-5)
