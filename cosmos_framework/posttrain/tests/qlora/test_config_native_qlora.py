# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native adapter configuration and optimizer ownership (CPU only; no CUDA kernels)."""

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
    native_adapter_config_identity,
    quantization_config_identity,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"


def _model_config(**kwargs):
    return Qwen3VLModelConfig(attn_impl="sdpa", adapter=NativeLoRAConfig(), **kwargs)


def _root(model_cfg):
    return SimpleNamespace(
        model_backend="native",
        model=NativePolicyBuildSpec(builder=BUILDER, model_id="/verified/snapshot", model_config=model_cfg),
    )


@pytest.mark.parametrize(
    "override",
    [
        {"initialization": "pissa"},
        {"rank": 0},
        {"rank": True},
        {"alpha": float("nan")},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
        {"dropout": -float("inf")},
        {"dropout": True},
        {"dropout": "0.05"},
        {"initialization_seed": -1},
        {"initialization_artifact": "/unexpected/bundle"},
        {"initialization_artifact_sha256": "a" * 64},
        {"initialization": "nf4_residual_svd"},
        {
            "initialization": "nf4_residual_svd",
            "initialization_artifact": "relative/bundle",
            "initialization_artifact_sha256": "a" * 64,
        },
        {
            "initialization": "nf4_residual_svd",
            "initialization_artifact": "/absolute/bundle",
            "initialization_artifact_sha256": "not-a-sha",
        },
    ],
)
def test_adapter_rejects_unsupported_training_contract(override):
    with pytest.raises(ValueError, match="NativeLoRAConfig"):
        NativeLoRAConfig(**override)


@pytest.mark.parametrize("dropout", [0, 0.0, 0.05, 0.5, 0.999])
def test_adapter_dropout_is_explicit_and_part_of_model_identity(dropout):
    baseline = NativeLoRAConfig()
    candidate = NativeLoRAConfig(dropout=dropout)
    expected = native_adapter_config_identity(baseline)
    expected["dropout"] = dropout
    assert native_adapter_config_identity(candidate) == expected
    assert baseline.dropout == 0.0
    for quantized_base in (None, QuantizedBaseTrainingConfig()):
        cfg = Qwen3VLModelConfig(attn_impl="sdpa", adapter=candidate, quantized_base=quantized_base)
        assert cfg.adapter.dropout == dropout


@pytest.mark.parametrize("name", ["method", "parameter_dtype", "target_profile", "bias", "modules_to_save"])
def test_fixed_lora_representation_is_recorded_but_not_configurable(name):
    with pytest.raises(TypeError):
        NativeLoRAConfig(**{name: "other"})
    assert name in native_adapter_config_identity(NativeLoRAConfig())


def test_residual_initialization_identity_is_content_bound():
    baseline = NativeLoRAConfig()
    expected = {
        "method": "lora",
        "parameter_dtype": "float32",
        "target_profile": "qwen3_vl_8b_language_v1",
        "bias": "none",
        "modules_to_save": [],
        **asdict(baseline),
    }
    del expected["initialization_artifact"]
    del expected["initialization_artifact_sha256"]
    assert native_adapter_config_identity(baseline) == expected
    options = {
        "initialization": "nf4_residual_svd",
        "initialization_artifact_sha256": "a" * 64,
    }
    first = NativeLoRAConfig(initialization_artifact="/first/bundle", **options)
    moved = NativeLoRAConfig(initialization_artifact="/second/bundle", **options)
    assert native_adapter_config_identity(first) == native_adapter_config_identity(moved)
    assert native_adapter_config_identity(first)["initialization_artifact_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="frozen NF4 base"):
        Qwen3VLModelConfig(attn_impl="sdpa", adapter=first)
    with pytest.raises(ValueError, match="FP32 adapter computation"):
        Qwen3VLModelConfig(
            attn_impl="sdpa",
            adapter=first,
            quantized_base=QuantizedBaseTrainingConfig(),
            adapter_compute_dtype="bfloat16",
        )
    assert (
        Qwen3VLModelConfig(attn_impl="sdpa", adapter=first, quantized_base=QuantizedBaseTrainingConfig()).adapter
        is first
    )


@pytest.mark.parametrize(
    "name", ["backend", "format", "block_size", "nested", "storage_dtype", "compute_dtype", "offload"]
)
def test_fixed_nf4_representation_is_not_configurable(name):
    with pytest.raises(TypeError):
        QuantizedBaseTrainingConfig(**{name: "other"})


def test_quantization_identity_records_the_fixed_nf4_representation():
    assert quantization_config_identity(QuantizedBaseTrainingConfig()) == {
        "backend": "bitsandbytes",
        "format": "nf4",
        "block_size": 64,
        "nested": True,
        "source_dtype": "bfloat16",
        "storage_dtype": "uint8",
        "compute_dtype": "bfloat16",
        "target_profile": "qwen3_vl_8b_language_v1",
        "offload": "none",
    }
    excluded = quantization_config_identity(QuantizedBaseTrainingConfig(excluded_language_layers=(7, 3)))
    assert excluded["excluded_language_layers"] == (3, 7)


def test_quantization_requires_explicit_adapter_and_component_precision():
    with pytest.raises(ValueError, match="requires an adapter"):
        Qwen3VLModelConfig(quantized_base=QuantizedBaseTrainingConfig())
    with pytest.raises(ValueError, match="BF16 base"):
        _model_config(dtype="float32")
    with pytest.raises(ValueError, match="attn_impl"):
        Qwen3VLModelConfig(adapter=NativeLoRAConfig())
    assert _model_config().quantized_base is None


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
    monkeypatch.setenv("SFT_DATA_ROOT", "/data/sft")
    control, quantized = lora_config(), qlora_config()
    assert asdict(control.rl) == asdict(quantized.rl)
    left, right = asdict(control.model), asdict(quantized.model)
    assert left["model_config"].pop("quantized_base") is None
    assert right["model_config"].pop("quantized_base") == asdict(QuantizedBaseTrainingConfig())
    assert left == right
    assert control.model.parallelism.dp_shard == control.model.parallelism.dp_replicate == 1
    assert control.rl.optimizer.train_keys == []
    assert control.rl.data.path == "/data/sft/train.jsonl"
    assert control.rl.validation.data.path == "/data/sft/validation.jsonl"
    assert not control.rl.validation.data.shuffle
    assert control.rl.mask_think_opener is False
    assert control.rl.ce_exponent == 0.0
    video = control.rl.processor_kwargs["videos_kwargs"]
    assert video["num_frames"] == 8
    assert video["fps"] is None
    # Qwen3-VL's pinned video resize bounds total T*H*W, not per-frame H*W.
    assert video["size"] == {"shortest_edge": 32_768, "longest_edge": 524_288}
    assert video["size"]["shortest_edge"] // video["num_frames"] == 4096
    assert video["size"]["longest_edge"] // video["num_frames"] == 65_536
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
    refuse_rounding_or_unavailable_optimizer(config, config.rl.optimizer)
