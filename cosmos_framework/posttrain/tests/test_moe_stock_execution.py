# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Whole-model execution identity, missing-media graphs, and stock compatibility."""

import copy

import pytest
import torch

from cosmos_framework.posttrain.checkpoint.adapters import (
    MODEL_EXECUTION_FORMAT,
    adapter_configuration,
    save_adapter_artifact,
    state_spec,
)
from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter, validate_export_targets
from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
from cosmos_framework.posttrain.model.adapter_moe_execution import PROFILE
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability
from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model
from cosmos_framework.posttrain.tests.adapter_numerics import HFProbePolicy
from cosmos_framework.posttrain.tests.qualify_adapters import logits
from cosmos_framework.posttrain.tests.reload_moe_adapters import hf_portable
from cosmos_framework.posttrain.tests.test_adapter_moe_export import native_moe_snapshot
from cosmos_framework.posttrain.tests.test_adapters_native import adapter_config, batches


def settings():
    cfg = adapter_config("pissa")
    cfg.model_execution = PROFILE
    for group in cfg.components:
        group.execution = "pissa_dense_v1"
    return cfg


@pytest.mark.parametrize("ac", ["none", "full"])
def test_native_stock_multimodal_gradients_and_portable_contract(tmp_path, ac):
    torch.manual_seed(43)
    native_moe_snapshot(tmp_path / "base")
    cfg = settings()
    policy = build_qwen3_vl_model(
        Qwen3VLModelConfig(
            name_or_path=str(tmp_path / "base"), dtype="bfloat16", attn_impl="sdpa", activation_checkpointing=ac
        ),
        adapters=cfg,
    )
    identity = policy.net._posttrain_adapter_identity
    assert identity["format"] == MODEL_EXECUTION_FORMAT
    assert identity["resolved"]["model_execution"] == PROFILE
    assert adapter_configuration(identity).model_execution == PROFILE
    state_spec(identity)
    validate_export_targets(str(tmp_path / "base"), identity)
    tampered = copy.deepcopy(identity)
    tampered["format"] = "cosmos-posttrain-adapter-v2"
    with pytest.raises(ValueError, match="format/execution"):
        state_spec(tampered)
    optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003)
    assert_trainability(policy.net, optimizer)
    frozen = {n: p.clone() for n, p in policy.net.state_dict().items() if "lora_" not in n}
    initial = {n: p.clone() for n, p in policy.net.named_parameters() if p.requires_grad}
    shapes = []
    hook = policy.net.model.visual.register_forward_pre_hook(lambda m, inputs: shapes.append(inputs[0].shape[0]))
    data = batches()
    data.append(
        {
            "input_ids": torch.cat([data[1]["input_ids"], data[2]["input_ids"]], dim=1),
            "attention_mask": torch.cat([data[1]["attention_mask"], data[2]["attention_mask"]], dim=1),
            "pixel_values": data[1]["pixel_values"],
            "image_grid_thw": data[1]["image_grid_thw"],
            "pixel_values_videos": data[2]["pixel_values_videos"],
            "video_grid_thw": data[2]["video_grid_thw"],
        }
    )
    for batch in data:
        kinds = torch.zeros_like(batch["input_ids"])
        kinds[batch["input_ids"] == policy.net.config.image_token_id] = 1
        kinds[batch["input_ids"] == policy.net.config.video_token_id] = 2
        batch["mm_token_type_ids"] = kinds
    for batch in data * 2:
        before = len(shapes)
        policy.net.train()
        optimizer.zero_grad(set_to_none=True)
        values = policy.forward_policy_batch(policy.prepare_policy_batch(batch)).logits
        assert len(shapes) - before == 2
        loss = values.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in policy.net.parameters() if p.requires_grad
        )
        optimizer.step()
    hook.remove()
    for name, value in frozen.items():
        assert torch.equal(value, policy.net.state_dict()[name]), name
    for name, value in initial.items():
        assert not torch.equal(value, policy.net.get_parameter(name)), name
    portable = tmp_path / "portable"
    save_adapter_artifact(policy.net, str(portable))
    native_outputs = logits(policy, data)
    warm = adapter_configuration(identity)
    warm.warm_start = str(portable)
    restored = build_qwen3_vl_model(
        Qwen3VLModelConfig(
            name_or_path=str(tmp_path / "base"), dtype="bfloat16", attn_impl="sdpa", activation_checkpointing=ac
        ),
        adapters=warm,
    )
    for actual, expected in zip(logits(restored, data), native_outputs, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    policy.net.float()
    stock = hf_portable(str(tmp_path / "base"), str(portable), identity).float()
    for actual, expected in zip(logits(HFProbePolicy(stock), data), logits(policy, data), strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
    # The new v3 profile is qualified independently of the guarded legacy v1/v2
    # modes. Exercise the public exporter, its transaction, and actual loader.
    from cosmos_framework.inference.reasoner.model_execution import model_execution_profile
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    live_state = {name: value.clone() for name, value in policy.net.state_dict().items()}
    request = dict(
        adapter=str(portable),
        base=str(tmp_path / "base"),
        out=str(tmp_path / "merged"),
        output_dtype="float32",
    )
    assert export_merged_adapter(**request) == "built"
    assert export_merged_adapter(**request) == "skipped"
    for name, value in live_state.items():
        assert torch.equal(value, policy.net.state_dict()[name]), name
    assert model_execution_profile(request["out"]) == PROFILE
    runtime = CosmosFrameworkRuntime.__new__(CosmosFrameworkRuntime)
    runtime.model_path, runtime.dtype, runtime.attn_implementation = request["out"], "auto", None
    runtime.model_type, runtime.device, runtime.device_map, runtime.rank = "qwen3_vl_moe", "cpu", "auto", 0
    runtime._load()
    assert not any(isinstance(module, PosttrainAdapterLinear) for module in runtime.model.modules())
    for actual, expected in zip(logits(HFProbePolicy(runtime.model), data), logits(policy, data), strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
        assert (actual - expected).abs().mean() <= 5e-5
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_stock_profile_invalid_method_and_dependency_fail_before_injection(tmp_path, monkeypatch):
    cfg = settings()
    cfg.components[0].execution = "factorized_v1"
    with pytest.raises(ValueError, match="dense PiSSA on every target"):
        cfg.__post_init__()
    torch.manual_seed(43)
    native_moe_snapshot(tmp_path / "base")
    monkeypatch.setattr("cosmos_framework.posttrain.model.adapter_moe_execution.version", lambda name: "4.57.6")
    with pytest.raises(ValueError, match="qualified Transformers"):
        build_qwen3_vl_model(
            Qwen3VLModelConfig(name_or_path=str(tmp_path / "base"), dtype="bfloat16"), adapters=settings()
        )


@pytest.mark.parametrize("target", ["text_config", "vision_config", "live_text", "live_vision"])
def test_stock_profile_checks_effective_attention_configs(target):
    import transformers

    from cosmos_framework.posttrain.model.adapter_moe_execution import validate_model_execution
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _moe_config

    config = transformers.Qwen3VLMoeConfig.from_dict(_moe_config().to_dict())
    model = transformers.AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa")
    validate_model_execution(model, settings(), "qwen3_vl_moe")
    if target.startswith("live_"):
        module = (
            model.model.language_model.layers[0].self_attn
            if target == "live_text"
            else model.model.visual.blocks[0].attn
        )
        module.config = copy.deepcopy(module.config)
        selected = module.config
    else:
        selected = getattr(model.config, target)
    selected._attn_implementation = "eager"
    with pytest.raises(ValueError, match="requires SDPA"):
        validate_model_execution(model, settings(), "qwen3_vl_moe")
