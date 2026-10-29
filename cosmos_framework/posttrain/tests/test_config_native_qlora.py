# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Configuration and optimizer ownership, without claiming CUDA kernel qualification."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.build import build_optimizer, refuse_rounding_or_unavailable_optimizer
from cosmos_framework.posttrain.config.model import (
    NativeLoRAConfig,
    NativePolicyBuildSpec,
    QuantizedBaseTrainingConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"


def _model_config(**kwargs):
    return Qwen3VLModelConfig(attn_impl="sdpa", adapter=NativeLoRAConfig(), **kwargs)


def _root(model_cfg):
    return SimpleNamespace(
        model_backend="i4",
        model=NativePolicyBuildSpec(builder=BUILDER, model_id="/verified/snapshot", model_config=model_cfg),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"method": "dora"},
        {"initialization": "pissa"},
        {"parameter_dtype": "bfloat16"},
        {"target_profile": "all-linear"},
        {"rank": 0},
        {"rank": True},
        {"alpha": float("nan")},
        {"dropout": 0.1},
        {"bias": "all"},
        {"modules_to_save": ["lm_head"]},
        {"initialization_seed": -1},
    ],
)
def test_adapter_rejects_unsupported_training_contract(override):
    with pytest.raises(ValueError, match="NativeLoRAConfig"):
        NativeLoRAConfig(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"backend": "other"},
        {"backend_version": "0.49.1"},
        {"format": "fp4"},
        {"block_size": 128},
        {"nested": False},
        {"nested": 1},
        {"source_dtype": "float32"},
        {"storage_dtype": "bfloat16"},
        {"compute_dtype": "float16"},
        {"target_profile": "all-linear"},
        {"offload": "cpu"},
    ],
)
def test_quantization_rejects_unsupported_format_without_fallback(override):
    with pytest.raises(ValueError, match="QuantizedBaseTrainingConfig"):
        QuantizedBaseTrainingConfig(**override)


def test_quantization_requires_explicit_adapter_and_component_precision():
    with pytest.raises(ValueError, match="requires an adapter"):
        Qwen3VLModelConfig(quantized_base=QuantizedBaseTrainingConfig())
    with pytest.raises(ValueError, match="BF16 base"):
        _model_config(dtype="float32")
    with pytest.raises(ValueError, match="attn_impl"):
        Qwen3VLModelConfig(adapter=NativeLoRAConfig())
    assert _model_config().quantized_base is None
    assert _model_config(quantized_base=QuantizedBaseTrainingConfig()).adapter.parameter_dtype == "float32"


def test_optimizer_precision_exception_is_limited_to_explicit_native_adapters():
    optimizer = OptimizerConfig(name="adamw", impl="fused")
    refuse_rounding_or_unavailable_optimizer(_root(_model_config()), optimizer)
    refuse_rounding_or_unavailable_optimizer(
        _root(_model_config(quantized_base=QuantizedBaseTrainingConfig())), optimizer
    )
    with pytest.raises(ValueError, match="most updates"):
        refuse_rounding_or_unavailable_optimizer(_root(Qwen3VLModelConfig()), optimizer)
    foreign = _root(_model_config())
    foreign.model.builder = "other.build"
    with pytest.raises(ValueError, match="most updates"):
        refuse_rounding_or_unavailable_optimizer(foreign, optimizer)


@pytest.mark.parametrize(
    "optimizer",
    [
        OptimizerConfig(name="adamw"),
        OptimizerConfig(name="fusedadam"),
        OptimizerConfig(name="adamw", impl="fused", train_keys=["lora"]),
        OptimizerConfig(name="adamw", impl="fused", decay_groups="hf"),
    ],
)
def test_optimizer_cannot_override_model_owned_trainability(optimizer):
    with pytest.raises(ValueError, match="native LoRA"):
        refuse_rounding_or_unavailable_optimizer(_root(_model_config()), optimizer)


def test_mutated_config_is_revalidated_at_optimizer_preflight():
    cfg = _model_config()
    cfg.adapter.parameter_dtype = "bfloat16"
    with pytest.raises(ValueError, match="parameter_dtype"):
        refuse_rounding_or_unavailable_optimizer(_root(cfg), OptimizerConfig(name="adamw", impl="fused"))


def test_optimizer_membership_follows_live_parameters_after_wrapping():
    torch = pytest.importorskip("torch")
    net = torch.nn.Module()
    net.base = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
    net.wrapped_adapter = torch.nn.Linear(3, 2, bias=False)
    # Canonical names intentionally differ from named_parameters, as activation checkpointing permits.
    net._cosmos_adapter_parameters = {"canonical.lora_A.weight": net.wrapped_adapter.weight}
    optimizer = build_optimizer(net, OptimizerConfig(name="adamw", impl="fused"))
    assert [p for group in optimizer.param_groups for p in group["params"]] == [net.wrapped_adapter.weight]
    net.base.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="exactly the final adapter"):
        build_optimizer(net, OptimizerConfig(name="adamw", impl="fused"))
    net.base.weight.requires_grad_(False)
    net.wrapped_adapter.to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(net, OptimizerConfig(name="adamw", impl="fused"))


def test_paired_recipes_preserve_objective_data_processor_and_schedule(monkeypatch):
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_lora import make_config as lora_config
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config as qlora_config

    monkeypatch.setenv("SFT_MODEL_PATH", "/verified/snapshot")
    monkeypatch.setenv("SFT_DATA_ROOT", "/data/wts")
    control, quantized = lora_config(), qlora_config()
    assert asdict(control.rl) == asdict(quantized.rl)
    left, right = asdict(control.model), asdict(quantized.model)
    assert left["model_config"].pop("quantized_base") is None
    assert right["model_config"].pop("quantized_base") == asdict(QuantizedBaseTrainingConfig())
    assert left == right
    assert control.model.parallelism.dp_shard == control.model.parallelism.dp_replicate == 1
    assert control.rl.optimizer.train_keys == []
    assert control.rl.data.path == "/data/wts/train.jsonl"
    assert control.rl.validation.data.path == "/data/wts/validation.jsonl"
    assert not control.rl.validation.data.shuffle
    assert control.rl.mask_think_opener is False
    assert control.rl.ce_exponent == 0.0
    assert control.rl.processor_kwargs["videos_kwargs"]["num_frames"] == 8
    assert control.rl.processor_kwargs["images_kwargs"]["size"]["longest_edge"] == 262_144


def test_native_recipe_dotlist_roundtrip_keeps_typed_adapter_config():
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.utils.config_helper import override

    config = override(
        make_config(),
        ["--", "rl.max_steps=3", "rl.lr_schedule.cycle_steps=3", "rl.optimizer.warmup_steps=1"],
    )
    cfg = config.model.model_config
    assert isinstance(cfg, Qwen3VLModelConfig)
    assert isinstance(cfg.adapter, NativeLoRAConfig)
    assert isinstance(cfg.quantized_base, QuantizedBaseTrainingConfig)
    assert isinstance(cfg.source_identity, dict)
    assert cfg.adapter.modules_to_save == ()
    refuse_rounding_or_unavailable_optimizer(config, config.rl.optimizer)
