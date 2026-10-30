# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual SigLIP/Gemma prefix-to-action gradients, adapters and portable state."""

import copy

import pytest
import torch
import transformers
from torch import nn
from transformers import GemmaConfig, PaliGemmaConfig, SiglipVisionConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionModel

from cosmos_framework.posttrain.checkpoint.adapters import (
    load_adapter_artifact,
    make_adapter_identity,
    prepare_merged_state,
    save_adapter_artifact,
)
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as C
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability, install_adapters, optimizer_groups
from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaModel
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
from cosmos_framework.posttrain.model.pi05.processing import Pi05Inputs
from cosmos_framework.posttrain.model.pi05.sft import validate_prefix_targets

pytestmark = pytest.mark.skipif(
    transformers.__version__ != "4.57.6", reason="pi05 full-model SFT binds Transformers 4.57.6 SigLIP names"
)


def tiny_pi05():
    torch.manual_seed(19)
    model = Pi05Model.__new__(Pi05Model)
    nn.Module.__init__(model)
    model.config = Pi05ModelConfig("unused", "unused", "unused", action_chunk=10)
    text = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
    )
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=16, patch_size=4
    )
    vlm = PaliGemmaConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=16)
    model.paligemma_with_expert = nn.Module()
    pg = model.paligemma_with_expert.paligemma = nn.Module()
    pg.model = nn.Module()
    pg.model.language_model = Pi05GemmaModel(text, adaptive=False)
    pg.model.vision_tower = SiglipVisionModel(vision)
    pg.model.multi_modal_projector = PaliGemmaMultiModalProjector(vlm)
    pg.lm_head = nn.Linear(16, 64, bias=False)
    pg.lm_head.weight = pg.model.language_model.embed_tokens.weight
    model.paligemma_with_expert.gemma_expert = nn.Module()
    model.paligemma_with_expert.gemma_expert.model = Pi05GemmaModel(text, adaptive=True)
    model.action_in_proj, model.action_out_proj = nn.Linear(32, 16), nn.Linear(16, 32)
    model.time_mlp_in, model.time_mlp_out = nn.Linear(16, 16), nn.Linear(16, 16)
    model.value_head = nn.Linear(16, 1)
    return model


def batch():
    inputs = Pi05Inputs(
        tuple(torch.randn(1, 3, 16, 16) for _ in range(3)),
        torch.tensor([[True, True, False]]),
        torch.tensor([[2, 3, 5, 7]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 32),
    )
    mask = torch.zeros(1, 10, 32)
    mask[:, :, :7] = 1
    return dict(
        inputs=inputs,
        actions=torch.randn(1, 10, 32),
        action_mask=mask,
        time=torch.tensor([0.3]),
        noise=torch.randn(1, 10, 32),
    )


def adapter_config(method, rs):
    common = dict(method=method, use_rslora=rs, rank=2, alpha=4)
    prefix = "paligemma_with_expert.paligemma.model"
    return AdapterConfig(
        components=[
            C(
                component="language",
                targets=[
                    f"{prefix}.language_model.layers.*.self_attn.k_proj",
                    f"{prefix}.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            C(component="projector", targets=[f"{prefix}.multi_modal_projector.linear"], **common),
            C(
                component="vision",
                targets=[f"{prefix}.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj"],
                layers=[1],
                **common,
            ),
            C(
                component="action",
                targets=[
                    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.v_proj",
                    "action_in_proj",
                    "action_out_proj",
                ],
                **common,
            ),
        ]
    )


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_multimodal_prefix_adapters_update_reload_and_merge(tmp_path, method, rs, checkpointing):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    from cosmos_framework.posttrain.model.adapters import canonical_name
    from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaDecoderLayer

    net = tiny_pi05()
    reference = copy.deepcopy(net)
    inputs = batch()
    net.eval()
    expected = net(**inputs).loss.detach()
    cfg = adapter_config(method, rs)
    validate_prefix_targets(net, cfg)
    frozen = {name: p.clone() for name, p in net.named_parameters()}
    install_adapters(net, cfg, family="pi05")
    net._posttrain_adapter_identity = make_adapter_identity(
        "",
        "",
        net._posttrain_adapter_manifest,
        {},
        base_dtype="float32",
        architecture="pi05",
        source_identity={
            "base": {"sha256": "fixture"},
            "processor": {"sha256": "fixture"},
            "model_contract": {"fixture": True},
        },
    )
    torch.testing.assert_close(net(**inputs).loss, expected, atol=2e-6, rtol=2e-5)
    initial = {name: p.clone() for name, p in net.named_parameters() if p.requires_grad}
    if checkpointing:
        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, (Pi05GemmaDecoderLayer, SiglipEncoderLayer)),
        )
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    optimizer = build_optimizer(
        net, OptimizerConfig(name="adamw", lr=0.01), parameter_groups=optimizer_groups(net, 0.01)
    )
    assert_trainability(net, optimizer)
    net.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = net(**inputs)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
        optimizer.step()
    for name, p in net.named_parameters():
        name = canonical_name(name)
        if p.requires_grad:
            assert not torch.equal(p, initial[name]), name
        else:
            assert torch.equal(p, frozen[name]), name
    net.eval()
    expected = net(**inputs).loss.detach()
    save_adapter_artifact(net, str(tmp_path / "portable"))
    loaded = tiny_pi05()
    install_adapters(loaded, cfg, family="pi05", initialize=False)
    loaded._posttrain_adapter_identity = net._posttrain_adapter_identity
    load_adapter_artifact(loaded, str(tmp_path / "portable"))
    loaded.eval()
    torch.testing.assert_close(loaded(**inputs).loss, expected, rtol=0, atol=0)
    reference.load_state_dict(prepare_merged_state(net.state_dict(), net._posttrain_adapter_identity), strict=True)
    reference.eval()
    torch.testing.assert_close(reference(**inputs).loss, expected, atol=5e-6, rtol=5e-5)
    assert (
        loaded.paligemma_with_expert.paligemma.lm_head.weight
        is loaded.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
    )


def test_final_prefix_query_is_rejected_before_injection():
    cfg = AdapterConfig(
        components=[
            C(targets=["paligemma_with_expert.paligemma.model.language_model.layers.1.self_attn.q_proj"], rank=2)
        ]
    )
    with pytest.raises(ValueError, match="does not consume final-prefix output"):
        validate_prefix_targets(tiny_pi05(), cfg)


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_native_merged_artifact_preserves_source_precision_and_tying(tmp_path, monkeypatch, method, rs):
    import json
    from dataclasses import replace

    from safetensors.torch import save_file

    from cosmos_framework.posttrain.model.pi05 import adapter_export
    from cosmos_framework.posttrain.model.pi05.sft import source_identity
    from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights
    from cosmos_framework.posttrain.tests.helpers.pi05_loading import write_processor_source

    base = tmp_path / "base"
    base.mkdir()
    config = replace(write_processor_source(base), action_chunk=10)
    original = tiny_pi05()
    save_file({n: v.clone() for n, v in original.state_dict().items()}, str(base / "model.safetensors"))
    source = Pi05Source.resolve(config)
    net = tiny_pi05()
    net.config = config
    load_model_weights(net, source)
    net.apply_reference_precision()
    source_contract = source_identity(net, source)
    cfg = adapter_config(method, rs)
    install_adapters(net, cfg, family="pi05")
    net._posttrain_adapter_identity = make_adapter_identity(
        str(base),
        str(base),
        net._posttrain_adapter_manifest,
        {},
        base_dtype="model_defined",
        architecture="pi05",
        source_identity=source_contract,
    )
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.001)
    net.eval()
    save_adapter_artifact(net, str(tmp_path / "portable"))
    frozen = {n: t.clone() for n, t in net.state_dict().items()}

    def constructor(config, **kwargs):
        model = tiny_pi05()
        model.config = config
        return model

    monkeypatch.setattr(adapter_export, "Pi05Model", constructor)
    merged = tmp_path / "merged"
    adapter_export.export_merged(str(tmp_path / "portable"), str(base), str(merged), processor=str(base))
    assert all(torch.equal(v, net.state_dict()[n]) for n, v in frozen.items())
    loaded = adapter_export.load_merged(str(merged))
    net.float()
    inputs = batch()
    torch.testing.assert_close(loaded.net(**inputs).prediction, net(**inputs).prediction, atol=5e-5, rtol=5e-5)
    pg = loaded.net.paligemma_with_expert.paligemma
    assert pg.lm_head.weight is pg.model.language_model.embed_tokens.weight
    with pytest.raises(ValueError, match="would re-round"):
        Pi05Source.resolve(replace(config, checkpoint_path=str(merged)))
    marker = json.loads((merged / "merged_model.json").read_text())
    assert marker["adapter_identity"]["base_storage_dtype"] == "model_defined"
    (merged / "norm_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="changed file"):
        adapter_export.load_merged(str(merged))
