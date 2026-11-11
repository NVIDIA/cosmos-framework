# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Family bindings through actual HF models and the existing model lifecycle."""

import copy
import json

import pytest
import torch
import transformers

from cosmos_framework.posttrain.checkpoint.adapters import prepare_merged_state, save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability, resolve_targets
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer


def text_snapshot(path, family):
    tokenizer = local_tokenizer()
    config = transformers.AutoConfig.for_model(
        family,
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    net = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    net.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return net


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2"])
@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_text_family_loaded_base_updates_reload_merge(tmp_path, family, method, rslora):
    torch.manual_seed(3)
    original = text_snapshot(tmp_path / "base", family).eval()
    base = copy.deepcopy(original.state_dict())
    cfg = AdapterConfig(
        components=[
            Component(
                targets=["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"],
                layers=[0, 1],
                rank=2,
                method=method,
                use_rslora=rslora,
            )
        ]
    )
    hf = HFModelConfig(name_or_path=str(tmp_path / "base"), dtype="float32", attn_impl="eager", grad_ckpt=True)
    model = build_hf_model(hf, adapters=cfg)
    assert model.net.get_input_embeddings().weight is model.net.get_output_embeddings().weight
    ids = torch.tensor([[11, 13, 19, 21, 17]])
    model.net.eval()
    torch.testing.assert_close(model(input_ids=ids), original(input_ids=ids).logits, atol=2e-6, rtol=2e-5)
    before = {n: p.clone() for n, p in model.net.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.net.parameters() if p.requires_grad], lr=1e-2)
    assert_trainability(model.net, optimizer)
    model.net.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].flatten())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.net.parameters() if p.requires_grad
        )
        optimizer.step()
    assert all(not torch.equal(before[n], p) for n, p in model.net.named_parameters() if p.requires_grad)
    for name, parameter in model.net.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(base[name], parameter), name
    model.net.eval()
    expected = model(input_ids=ids).detach()
    portable = tmp_path / "adapter"
    save_adapter_artifact(model.net, str(portable))
    cfg.warm_start = str(portable)
    loaded = build_hf_model(hf, adapters=cfg)
    loaded.net.eval()
    torch.testing.assert_close(loaded(input_ids=ids), expected, atol=0, rtol=0)
    merged = prepare_merged_state(model.net.state_dict(), model.net._posttrain_adapter_identity)
    original.load_state_dict(merged, strict=True)
    torch.testing.assert_close(original(input_ids=ids).logits, expected, atol=5e-6, rtol=5e-5)


def test_each_selector_and_component_are_checked_in_text_family(tmp_path):
    net = text_snapshot(tmp_path, "qwen3")
    for groups, match in [
        ([Component(targets=["model.layers.0.self_attn.q_proj", "model.missing.*"])], "matched no modules"),
        ([Component(component="vision", targets=["model.layers.0.self_attn.q_proj"])], "outside vision"),
        ([Component(targets=["model.layers.*.self_attn.q_proj"], layers=[5])], "matched no modules"),
        (
            [
                Component(targets=["model.layers.0.self_attn.q_proj"], rank=2),
                Component(targets=["model.layers.0.self_attn.q_proj"], rank=3),
            ],
            "conflicting",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            resolve_targets(net, AdapterConfig(components=groups))


@pytest.mark.parametrize("family", ["qwen3", "gemma", "gemma2"])
def test_text_family_stock_merged_export(tmp_path, family):
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    text_snapshot(tmp_path / "base", family)
    if family in ("gemma", "gemma2"):
        # The qualified public Gemma checkpoints omit this default field and
        # their safetensors omit the aliased lm_head.weight. Bind the adapter
        # to those exact source bytes before exercising the public exporter.
        path = tmp_path / "base" / "config.json"
        source_config = json.loads(path.read_text())
        source_config.pop("tie_word_embeddings", None)
        path.write_text(json.dumps(source_config) + "\n")
        assert transformers.AutoConfig.from_pretrained(path.parent).tie_word_embeddings
    cfg = AdapterConfig(components=[Component(targets=["model.layers.0.self_attn.q_proj"], rank=2, method="dora")])
    model = build_hf_model(
        HFModelConfig(name_or_path=str(tmp_path / "base"), dtype="float32", attn_impl="sdpa"), adapters=cfg
    )
    save_adapter_artifact(model.net, str(tmp_path / "adapter"))
    assert export_merged_adapter(str(tmp_path / "adapter"), str(tmp_path / "base"), str(tmp_path / "merged")) == "built"
    assert export_merged_adapter(str(tmp_path / "adapter"), str(tmp_path / "base"), str(tmp_path / "merged")) == "skipped"
    loaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    model.net.eval()
    ids = torch.tensor([[11, 13, 19]])
    torch.testing.assert_close(loaded(input_ids=ids).logits, model(input_ids=ids), atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("method,rslora", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_hf_qwen3_5_all_components_ac_and_media(tmp_path, method, rslora):
    from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear
    from cosmos_framework.posttrain.tests.test_model_qwen3_5 import _batch, config

    path = tmp_path / "base"
    model_config = config.__wrapped__()
    source = transformers.AutoModelForImageTextToText.from_config(model_config)
    source.save_pretrained(path)
    common = dict(rank=2, method=method, use_rslora=rslora)
    cfg = AdapterConfig(
        components=[
            Component(component="language", targets=["model.language_model.layers.*.mlp.up_proj"], **common),
            Component(component="vision", targets=["model.visual.blocks.*.attn.qkv"], layers=[0], **common),
            Component(component="projector", targets=["model.visual.merger.linear_fc1"], **common),
        ]
    )
    model = build_hf_model(
        HFModelConfig(name_or_path=str(path), dtype="float32", attn_impl="sdpa", grad_ckpt=True), adapters=cfg
    )
    expected = {row["name"] for row in model.net._posttrain_adapter_manifest["targets"]}
    assert {name for name, m in model.net.named_modules() if isinstance(m, PosttrainAdapterLinear)} == expected
    before = {n: p.clone() for n, p in model.net.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.net.parameters() if p.requires_grad], lr=0.01)
    assert_trainability(model.net, optimizer)
    model.net.train()
    for modality in ("image", "video", "text", "both"):
        batch = _batch(model_config, modality)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_policy_batch(model.prepare_policy_batch(batch)).logits
        loss = output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.net.parameters() if p.requires_grad
        )
        optimizer.step()
    assert all(not torch.equal(before[n], p) for n, p in model.net.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[n], p) for n, p in model.net.named_parameters() if not p.requires_grad)
