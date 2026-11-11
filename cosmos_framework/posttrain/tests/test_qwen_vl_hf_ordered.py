# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ordered missing-modality execution and original checkpoint namespace loading."""

import copy

import pytest
import torch
import transformers
from safetensors.torch import save_file

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.tests.test_adapter_qwen2_vl_families import small_config
from cosmos_framework.posttrain.tests.test_adapters_native import batches


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_ordered_original_weights_updates_warm_start_and_stock_merge(tmp_path, monkeypatch, family, checkpointing):
    from cosmos_framework.posttrain.checkpoint.adapters import prepare_merged_state

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(43)
    source = transformers.AutoModelForImageTextToText.from_config(small_config(family), attn_implementation="sdpa")
    base = tmp_path / "base"
    source.save_pretrained(base)
    from cosmos_framework.posttrain.tests.helpers.sft_packer import CHAT_TEMPLATE, local_processor, local_tokenizer

    tokenizer = local_tokenizer()
    media = local_processor(tokenizer)
    cls = transformers.Qwen2VLProcessor if family == "qwen2_vl" else transformers.Qwen2_5_VLProcessor
    processor = cls(
        image_processor=media.image_processor,
        tokenizer=tokenizer,
        video_processor=transformers.Qwen2VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        chat_template=CHAT_TEMPLATE,
    )
    processor.save_pretrained(base)
    # Match the immutable public Qwen2/2.5-VL snapshots, not just recent HF saves.
    raw = {
        n.replace("model.language_model.", "model.").replace("model.visual.", "visual."): v.contiguous()
        for n, v in source.state_dict().items()
    }
    save_file(raw, str(base / "model.safetensors"))
    cfg = AdapterConfig(
        model_execution="qwen_vl_hf_ordered_v1",
        extra_modules=[AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"])],
        components=[
            Component(component="language", targets=["model.language_model.layers.*.self_attn.q_proj"], rank=2),
            Component(
                component="projector",
                targets=["model.visual.merger.mlp.2"],
                method="dora",
                rank=3,
                alpha=5,
                use_rslora=True,
            ),
            Component(
                component="vision",
                targets=["model.visual.blocks.0.attn.qkv"],
                method="pissa",
                execution="pissa_dense_v1",
                rank=2,
            ),
        ],
    )
    model_cfg = HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa", grad_ckpt=checkpointing)
    net = build_hf_model(model_cfg, adapters=cfg).net
    data = batches()
    for b in data:
        # Match this architecture's real vision delimiters and temporal RoPE.
        b["input_ids"][b["input_ids"] == 118] = 122
        b["input_ids"][b["input_ids"] == 119] = 123
        b["mm_token_type_ids"] = (b["input_ids"] == 120).long() + 2 * (b["input_ids"] == 121).long()
        if family == "qwen2_5_vl" and "video_grid_thw" in b:
            b["second_per_grid_ts"] = torch.tensor([0.5])
    source.eval()
    net.eval()
    with torch.no_grad():
        for b in data:
            torch.testing.assert_close(net(**b).logits, source(**b).logits, atol=2e-6, rtol=2e-5)
    frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
    owned = {n: id(p) for n, p in net.named_parameters()}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.001)
    assert_trainability(net, opt)
    net.train()
    for b in data * 2:
        opt.zero_grad(set_to_none=True)
        loss = net(**b).logits.square().mean()
        loss.backward()
        # Dummy calls connect vision on text-only ranks; a zero gradient is valid.
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
        opt.step()
    assert {n: id(p) for n, p in net.named_parameters()} == owned
    assert all(torch.equal(p, frozen[n]) for n, p in net.named_parameters() if n in frozen)
    net.eval()
    portable = tmp_path / "portable"
    save_adapter_artifact(net, str(portable))
    cfg.warm_start = str(portable)
    restored = build_hf_model(model_cfg, adapters=cfg).net.eval()
    source.load_state_dict(prepare_merged_state(net.state_dict(), net._posttrain_adapter_identity), strict=True)
    from cosmos_framework.inference.reasoner.model_execution import model_execution_profile, validate_loaded_model
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter

    merged = tmp_path / "merged"
    assert export_merged_adapter(str(portable), str(base), str(merged)) == "built"
    assert export_merged_adapter(str(portable), str(base), str(merged)) == "skipped"
    disk = transformers.AutoModelForImageTextToText.from_pretrained(
        merged, dtype="auto", attn_implementation="sdpa"
    ).eval()
    assert model_execution_profile(merged) == "qwen_vl_hf_ordered_v1"
    validate_loaded_model(disk)
    for name, value in source.state_dict().items():
        assert torch.equal(disk.state_dict()[name], value), name
    with torch.no_grad():
        for b in data:
            expected = net(**b).logits
            torch.testing.assert_close(disk(**b).logits, expected, atol=5e-6, rtol=5e-5)
            torch.testing.assert_close(restored(**b).logits, expected, atol=0, rtol=0)
            torch.testing.assert_close(source(**b).logits, expected, atol=5e-6, rtol=5e-5)
        # Use common token histories, keeping multimodal positions and cache state.
        prefix = data[-1]
        cached = net(**prefix, use_cache=True)
        for token in [7, 8, 9]:
            prefix = copy.deepcopy(prefix)
            prefix["input_ids"] = torch.cat([prefix["input_ids"], torch.tensor([[token]])], -1)
            prefix["attention_mask"] = torch.ones_like(prefix["input_ids"])
            prefix["mm_token_type_ids"] = torch.cat(
                [prefix["mm_token_type_ids"], torch.zeros((1, 1), dtype=torch.long)], -1
            )
            positions, _ = net.model.get_rope_index(
                prefix["input_ids"],
                video_grid_thw=prefix["video_grid_thw"],
                attention_mask=prefix["attention_mask"],
                mm_token_type_ids=prefix["mm_token_type_ids"],
                **({"second_per_grid_ts": prefix["second_per_grid_ts"]} if family == "qwen2_5_vl" else {}),
            )
            step = net.prepare_inputs_for_generation(
                prefix["input_ids"],
                attention_mask=prefix["attention_mask"],
                past_key_values=cached.past_key_values,
                position_ids=positions,
                use_cache=True,
                next_sequence_length=1,
            )
            cached = net(**step)
            torch.testing.assert_close(
                cached.logits[:, -1], net(**prefix, use_cache=False).logits[:, -1], atol=5e-6, rtol=5e-5
            )


def test_weight_alias_collision_rejected_before_second_copy(tmp_path):
    from cosmos_framework.posttrain.model.qwen_vl_hf_execution import canonical_weight_name
    from cosmos_framework.posttrain.model.specs import LoadPlan
    from cosmos_framework.posttrain.model.weight_loader import load_weights

    net = torch.nn.Module()
    net.add_module("model", torch.nn.Module())
    net.model.add_module("language_model", torch.nn.Module())
    net.model.language_model.add_module("norm", torch.nn.Linear(2, 2, bias=False))
    save_file(
        {"model.norm.weight": torch.ones(2, 2), "model.language_model.norm.weight": torch.zeros(2, 2)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="same model owner"):
        load_weights(net, str(tmp_path), LoadPlan(rename=lambda n: canonical_weight_name("qwen2_vl", n)))


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen2_5_vl"])
def test_ordered_rejects_live_child_attention_before_mutation(family):
    from cosmos_framework.posttrain.model.qwen_vl_hf_execution import install

    net = transformers.AutoModelForImageTextToText.from_config(small_config(family), attn_implementation="sdpa")
    net.model.language_model.layers[0].self_attn.config = copy.deepcopy(net.config.text_config)
    net.model.language_model.layers[0].self_attn.config._attn_implementation = "eager"
    before = net.model.forward
    with pytest.raises(ValueError, match="requires SDPA"):
        install(net)
    assert net.model.forward == before
    assert not hasattr(net, "_posttrain_model_execution")


def test_ordered_qwen3_preserves_norm_semantics_and_stock_export(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
    from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, tiny_snapshot

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = tmp_path / "base"
    tiny_snapshot(base, tied=True)
    config = adapter_config("lora")
    config.model_execution = "qwen_vl_hf_ordered_v1"
    config.components[-1].method = "pissa"
    config.components[-1].execution = "pissa_dense_v1"
    config.extra_modules = [AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"])]
    cfg = HFModelConfig(name_or_path=str(base), dtype="float32", attn_impl="sdpa", grad_ckpt=False)
    model = build_hf_model(cfg, adapters=config).net.eval()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.model.language_model.norm.forward.__func__.__module__.startswith("transformers.")
    portable, merged = tmp_path / "portable", tmp_path / "merged"
    save_adapter_artifact(model, str(portable))
    export_merged_adapter(str(portable), str(base), str(merged))
    stock = transformers.AutoModelForImageTextToText.from_pretrained(
        merged, dtype="auto", attn_implementation="sdpa"
    ).eval()
    with torch.no_grad():
        for batch in batches():
            batch = dict(batch)
            original = batch["input_ids"]
            batch["input_ids"] = original.clone()
            for old, name in (
                (118, "vision_start_token_id"),
                (119, "vision_end_token_id"),
                (120, "image_token_id"),
                (121, "video_token_id"),
            ):
                batch["input_ids"][original == old] = getattr(model.config, name)
            batch["mm_token_type_ids"] = (original == 120).long() + 2 * (original == 121).long()
            torch.testing.assert_close(model(**batch).logits, stock(**batch).logits, atol=5e-6, rtol=5e-5)
