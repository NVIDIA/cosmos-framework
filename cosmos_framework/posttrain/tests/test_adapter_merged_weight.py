# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""New rounded-weight execution has its own graph and immutable artifact identity."""

import copy

import pytest
import torch
import transformers

from cosmos_framework.posttrain.checkpoint.adapters import (
    adapter_configuration,
    load_adapter_artifact,
    prepare_merged_state,
    save_adapter_artifact,
    state_spec,
)
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig, AdapterExtraModuleConfig
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.model.adapter_bf16_execution import PROFILE, install_effective_norm
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.model.qwen3_vl_hf import patch_ordered_forward
from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, tiny_snapshot
from cosmos_framework.posttrain.tests.test_model_qwen3_5 import _batch
from cosmos_framework.posttrain.tests.test_model_qwen3_5 import config as q35_config


def configuration():
    cfg = adapter_config("lora")
    cfg.components[1].method = "dora"
    cfg.components[1].use_rslora = True
    cfg.components[2].method = "pissa"
    cfg.components[2].initialization = "svd_lowrank"
    for group in cfg.components:
        group.execution = PROFILE
    cfg.model_execution = PROFILE
    cfg.extra_modules = [AdapterExtraModuleConfig(targets=["model.language_model.norm"], lr_multiplier=0.1)]
    cfg.__post_init__()
    return cfg


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("rslora", [False, True])
def test_rounded_linear_gradients_initialization_and_nonmutating_merge(method, rslora):
    torch.manual_seed(42)
    base = torch.nn.Linear(12, 16, dtype=torch.bfloat16)
    original = copy.deepcopy(base)
    cfg = AdapterComponentConfig(
        targets=["module"], method=method, rank=3, alpha=7, use_rslora=rslora, execution=PROFILE
    )
    layer = PosttrainAdapterLinear(base, cfg)
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)
    values = torch.randn(8, 12, dtype=torch.bfloat16)
    torch.testing.assert_close(layer(values), original(values), atol=0, rtol=0)
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = layer(values).float().square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)
        optimizer.step()
    assert torch.equal(layer.weight, before["weight"])
    assert torch.equal(layer.bias, before["bias"])
    for name, parameter in layer.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
    state = {"module." + key: value for key, value in layer.state_dict().items()}
    state["module.weight"] = state["module.weight"].float()
    merged = layer.merged_state_weight(state, "module")
    assert torch.equal(merged, layer.effective_weight())
    torch.testing.assert_close(
        layer(values), torch.nn.functional.linear(values, merged.bfloat16(), layer.bias), atol=0, rtol=0
    )
    if method == "pissa":
        assert torch.equal(layer.initial_A, before["initial_A"])
        assert torch.equal(layer.initial_B, before["initial_B"])


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5"])
def test_extra_norm_effective_precision_and_parameter_ownership(family):
    if family == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm as Norm
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm as Norm
    layer = Norm(16).float()
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * 0.01)
    reference = copy.deepcopy(layer).bfloat16()
    parameter = layer.weight
    install_effective_norm(layer)
    values = torch.randn(4, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(layer(values), reference(values), atol=0, rtol=0)
    layer(values).float().square().sum().backward()
    assert layer.weight is parameter and parameter.dtype == torch.float32
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    reference.float().load_state_dict(layer.state_dict())
    torch.testing.assert_close(layer(values.float()), reference(values.float()), atol=0, rtol=0)


@pytest.mark.parametrize("modality", ["text", "image", "video", "both"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_ordered_hf5_forward_matches_stock_values_and_gradients(modality, checkpointing):
    source_config = q35_config.__wrapped__()
    cfg = transformers.Qwen3VLConfig.from_dict(source_config.to_dict())
    cfg.vision_config.deepstack_visual_indexes = [0, 1]
    source = transformers.AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    adapted = copy.deepcopy(source)
    patch_ordered_forward(adapted.model)
    if checkpointing:
        for net in (source, adapted):
            net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    data = _batch(cfg, modality, torch.bfloat16)
    expected = source(**data, use_cache=False).logits
    actual = adapted(**data, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    for (name, left), (_, right) in zip(source.named_parameters(), adapted.named_parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
        elif name.startswith("model.visual"):
            assert right.grad is not None and torch.count_nonzero(right.grad) == 0, name


def test_new_artifact_lifecycle_and_both_stock_export_precisions(tmp_path):
    base = tmp_path / "base"
    tiny_snapshot(base, tied=True)
    cfg = configuration()
    model_cfg = HFModelConfig(name_or_path=str(base), dtype="bfloat16", attn_impl="sdpa", grad_ckpt=True)
    policy = build_hf_model(model_cfg, adapters=cfg)
    assert policy.net.lm_head.weight is policy.net.model.language_model.embed_tokens.weight
    optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.001)
    assert_trainability(policy.net, optimizer)
    with torch.no_grad():
        for parameter in policy.net.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.003)
    portable = tmp_path / "portable"
    save_adapter_artifact(policy.net, str(portable))
    identity = policy.net._posttrain_adapter_identity
    assert identity["format"] == "cosmos-posttrain-adapter-v4"
    bad = copy.deepcopy(identity)
    bad["format"] = "cosmos-posttrain-adapter-v3"
    with pytest.raises(ValueError, match="format/execution mismatch"):
        state_spec(bad)
    warm = adapter_configuration(identity)
    warm.warm_start = str(portable)
    reloaded = build_hf_model(model_cfg, adapters=warm)
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4)}
    policy.net.eval()
    reloaded.net.eval()
    with torch.no_grad():
        torch.testing.assert_close(policy.net(**inputs).logits, reloaded.net(**inputs).logits, atol=0, rtol=0)
    from cosmos_framework.inference.reasoner.model_execution import model_execution_profile, validate_loaded_model

    for dtype in ("bfloat16", "float32"):
        if dtype == "float32":
            policy.net.float()
        target = tmp_path / dtype
        assert export_merged_adapter(str(portable), str(base), str(target), output_dtype=dtype) == "built"
        assert model_execution_profile(target) == PROFILE
        loaded = transformers.AutoModelForImageTextToText.from_pretrained(
            target, dtype="auto", attn_implementation="sdpa"
        ).eval()
        validate_loaded_model(loaded)
        with torch.no_grad():
            torch.testing.assert_close(policy.net(**inputs).logits, loaded(**inputs).logits, atol=0, rtol=0)
        expected = prepare_merged_state(policy.net.state_dict(), identity, merge_dtype=torch.float32)
        for name, value in loaded.state_dict().items():
            assert torch.equal(value, expected[name].to(value.dtype)), name
    legacy = copy.deepcopy(identity)
    legacy["resolved"].pop("model_execution")
    for row in legacy["resolved"]["targets"]:
        row["settings"].pop("execution")
    legacy["format"] = "cosmos-posttrain-adapter-v1"
    reloaded.net._posttrain_adapter_identity = legacy
    with pytest.raises(ValueError, match="compatibility mismatch"):
        load_adapter_artifact(reloaded.net, str(portable))


def test_execution_combinations_fail_before_injection():
    with pytest.raises(ValueError, match="dropout=0"):
        AdapterComponentConfig(targets=["module"], execution=PROFILE, dropout=0.1)
    with pytest.raises(ValueError, match="matching model"):
        AdapterConfig(components=[AdapterComponentConfig(targets=["module"], execution=PROFILE)])
    with pytest.raises(ValueError, match="matching model"):
        AdapterConfig(model_execution=PROFILE, components=[AdapterComponentConfig(targets=["module"])])
