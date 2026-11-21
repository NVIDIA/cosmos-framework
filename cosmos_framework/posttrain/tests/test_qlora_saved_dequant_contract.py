# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU config/restore contracts; no CUDA or full-model performance qualification."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.posttrain.config.model import (
    NativeLoRAConfig,
    QuantizedBaseTrainingConfig,
    Qwen3VLModelConfig,
    native_adapter_config_identity,
    native_adapter_execution_policy,
    quantization_config_identity,
    validate_quantized_linear_execution,
)
from cosmos_framework.posttrain.model.quantization import loader

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def config(**overrides):
    return Qwen3VLModelConfig(
        **{
            "attn_impl": "sdpa",
            "adapter": NativeLoRAConfig(),
            "quantized_base": QuantizedBaseTrainingConfig(),
            "quantized_linear_execution": "checkpointed_dequant",
            **overrides,
        }
    )


def identity(*, logits_mode="full"):
    cfg = config(sft_logits_mode=logits_mode)
    return {
        "format": "cosmos.adapter-model.v2",
        "model_format": "cosmos_native_qwen3_vl",
        "adapter": native_adapter_config_identity(cfg.adapter),
        "quantization": quantization_config_identity(cfg.quantized_base),
        "execution_policy": loader.resolved_execution_policy(cfg),
    }


def test_default_and_existing_opt_in_identity_bytes_remain_unchanged():
    assert Qwen3VLModelConfig().quantized_linear_execution == "bnb"
    assert native_adapter_execution_policy(config(quantized_linear_execution="bnb")) is None
    # The saved v1 policy shape intentionally has no new default field.
    expected = {
        "protocol": "cosmos.native-adapter-execution.v1",
        "activation_checkpointing": {
            "mode": "none",
            "implementation": "none",
            "preserve_rng_state": True,
            "determinism_check": "default",
            "save_ops_regex": [],
        },
        "adapter_compute_dtype": "float32",
        "sft_logits_mode": "full",
        "language_compile": {
            "enabled": False,
            "dynamic": True,
            "fullgraph": False,
            "backend": "inductor",
            "fail_on_recompile_limit_hit": True,
        },
    }
    cfg = config(quantized_linear_execution="bnb", activation_checkpointing="none")
    assert json.dumps(native_adapter_execution_policy(cfg), sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert "quantized_linear_execution" not in loader.execution_config_from_identity({})
    assert "quantized_linear_execution" not in loader.execution_config_from_identity(
        {"execution_policy": loader.resolved_execution_policy(cfg)}
    )


@pytest.mark.parametrize("logits_mode", ["full", "supervised"])
def test_opt_in_structured_roundtrip_and_complete_execution_identity(logits_mode):
    cfg = OmegaConf.to_object(OmegaConf.structured(config(sft_logits_mode=logits_mode)))
    assert cfg.quantized_linear_execution == "checkpointed_dequant"
    saved = identity(logits_mode=logits_mode)
    assert saved["execution_policy"]["protocol"] == "cosmos.native-adapter-execution.v2"
    assert saved["execution_policy"]["quantized_linear"] == {
        "protocol": "cosmos.checkpointed-dequant.v1",
        "mode": "checkpointed_dequant",
        "scope": "full_nonreentrant_language_block",
        "inference": "bnb",
    }
    assert loader.execution_config_from_identity(saved) == {
        "activation_checkpointing": "full",
        "adapter_compute_dtype": "float32",
        "sft_logits_mode": logits_mode,
        "compile_language_model": False,
        "quantized_linear_execution": "checkpointed_dequant",
    }
    loader.validate_execution_policy_identity(cfg, saved)


@pytest.mark.parametrize("logits_mode", [None, True, 1, "", "future", ["supervised"]])
def test_malformed_logits_mode_rejected_by_dependency_light_guard(logits_mode):
    cfg = config()
    cfg.sft_logits_mode = logits_mode
    with pytest.raises(ValueError, match="checkpointed_dequant"):
        validate_quantized_linear_execution(cfg)


@pytest.mark.parametrize("logits_mode", ["full", "supervised"])
def test_full_and_supervised_execution_identities_are_not_interchangeable(logits_mode):
    other = "supervised" if logits_mode == "full" else "full"
    with pytest.raises(ValueError, match="execution policy"):
        loader.validate_execution_policy_identity(config(sft_logits_mode=other), identity(logits_mode=logits_mode))


@pytest.mark.parametrize("execution", [None, True, 1, "", "future", ["bnb"]])
def test_unknown_execution_never_silently_uses_bnb(execution):
    with pytest.raises(ValueError, match="quantized_linear_execution"):
        config(quantized_linear_execution=execution)


@pytest.mark.parametrize(
    "overrides",
    [
        {"adapter": None, "quantized_base": None},
        {"quantized_base": None},
        {"activation_checkpointing": "none"},
        {"activation_checkpointing": "selective"},
        {"adapter_compute_dtype": "bfloat16"},
        {"compile_language_model": True},
        {"quantized_base": QuantizedBaseTrainingConfig(excluded_language_layers=(6, 15))},
        {
            "adapter": NativeLoRAConfig(
                initialization="nf4_residual_svd",
                initialization_artifact="/factors",
                initialization_artifact_sha256="a" * 64,
            )
        },
    ],
)
def test_unsupported_combinations_fail_in_dependency_light_config(overrides):
    with pytest.raises(ValueError, match="checkpointed_dequant"):
        config(**overrides)


@pytest.mark.parametrize(
    "name,value",
    [
        ("activation_checkpointing", "none"),
        ("compile_language_model", True),
        ("quantized_base", None),
        ("quantized_linear_execution", "future"),
        ("sft_logits_mode", "future"),
        ("sft_logits_mode", None),
    ],
)
def test_mutated_config_fails_before_source_io_or_runtime_allocation(monkeypatch, name, value):
    from cosmos_framework.posttrain.model.quantization import source

    cfg = config()
    setattr(cfg, name, value)
    monkeypatch.setattr(source, "validate_adapter_source", lambda *_: pytest.fail("source IO before policy guard"))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: pytest.fail("runtime reached"))
    with pytest.raises(ValueError):
        loader.NativeAdapterLifecycle(cfg, object(), None)
    with pytest.raises(ValueError):
        loader.validate_runtime(cfg, object(), None)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_policy",
        "missing_record",
        "missing_mode",
        "unknown_mode",
        "default_mode",
        "unknown_protocol",
        "old_protocol",
        "future_subprotocol",
        "extra_field",
        "unknown_scope",
        "changed_inference",
        "no_ac",
        "selective_ac",
        "compile",
        "bf16_adapter",
        "unknown_logits",
        "bf16_base",
        "wrong_nf4",
        "missing_adapter",
        "residual",
        "exclusions",
        "legacy_format",
        "missing_format",
        "missing_math",
    ],
)
def test_saved_policy_rejects_missing_unknown_and_conflicting_records(defect):
    saved = identity()
    policy = saved["execution_policy"]
    record = policy["quantized_linear"]
    if defect == "missing_policy":
        saved.pop("execution_policy")
    elif defect == "missing_record":
        policy.pop("quantized_linear")
    elif defect == "missing_mode":
        record.pop("mode")
    elif defect in ("unknown_mode", "default_mode"):
        record["mode"] = "bnb" if defect == "default_mode" else "future"
    elif defect in ("unknown_protocol", "old_protocol"):
        policy["protocol"] = "cosmos.native-adapter-execution.v1" if defect == "old_protocol" else "future"
    elif defect == "future_subprotocol":
        record["protocol"] = "cosmos.checkpointed-dequant.future"
    elif defect == "extra_field":
        record["cache_whole_model"] = True
    elif defect == "unknown_scope":
        record["scope"] = "model"
    elif defect == "changed_inference":
        record["inference"] = "checkpointed_dequant"
    elif defect in ("no_ac", "selective_ac"):
        policy["activation_checkpointing"]["mode"] = "none" if defect == "no_ac" else "selective"
    elif defect == "compile":
        policy["language_compile"]["enabled"] = True
    elif defect == "bf16_adapter":
        policy["adapter_compute_dtype"] = "bfloat16"
    elif defect == "unknown_logits":
        policy["sft_logits_mode"] = "future"
    elif defect == "bf16_base":
        saved["quantization"] = None
    elif defect == "wrong_nf4":
        saved["quantization"]["backend_version"] = "future"
    elif defect == "missing_adapter":
        saved.pop("adapter")
    elif defect == "residual":
        saved["adapter"]["initialization"] = "nf4_residual_svd"
    elif defect == "exclusions":
        saved["quantization"]["excluded_language_layers"] = [6]
    elif defect == "legacy_format":
        saved["format"] = "cosmos.adapter-model.v1"
    elif defect == "missing_format":
        saved.pop("format")
    else:
        policy.pop("math_precision")
    with pytest.raises(ValueError):
        loader.execution_config_from_identity(saved)


@pytest.mark.parametrize("logits_mode", ["full", "supervised"])
def test_real_probe_restore_reconstructs_mode_and_rejects_explicit_conflicts(logits_mode):
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.tests.qlora_integration import _restore_probe_execution

    cfg = make_config()
    saved = identity(logits_mode=logits_mode)
    _restore_probe_execution(cfg, saved, [])
    assert cfg.model.model_config.quantized_linear_execution == "checkpointed_dequant"
    assert cfg.model.model_config.sft_logits_mode == logits_mode
    with pytest.raises(ValueError, match="explicit adapter reload override quantized_linear_execution"):
        _restore_probe_execution(make_config(), saved, ["model.model_config.quantized_linear_execution=bnb"])
    other = "supervised" if logits_mode == "full" else "full"
    conflicting = make_config()
    # The native config loader applies the dotlist before the restore helper sees it.
    conflicting.model.model_config.sft_logits_mode = other
    with pytest.raises(ValueError, match="explicit adapter reload override sft_logits_mode"):
        _restore_probe_execution(conflicting, saved, [f"model.model_config.sft_logits_mode={other}"])
    with pytest.raises(ValueError, match="execution policy"):
        loader.validate_execution_policy_identity(config(quantized_linear_execution="bnb"), saved)
    with pytest.raises(ValueError, match="execution policy"):
        loader.validate_execution_policy_identity(config(), {"format": "cosmos.adapter-model.v1"})


def make_opt_in_fixture(tmp_path, monkeypatch, *, logits_mode="full"):
    """Reuse the all-252-target CPU loader fixture, intercepting only its config/BNB doubles."""
    from cosmos_framework.posttrain.tests import test_quantized_exclusions as fixture

    calls = {"preflight": [], "constructor": []}

    def namespace(**kwargs):
        if "quantized_base" in kwargs:
            kwargs["quantized_linear_execution"] = "checkpointed_dequant"
            kwargs["sft_logits_mode"] = logits_mode
        if "NativeLoraLinear4bit" in kwargs:
            packed = kwargs["NativeLoraLinear4bit"]

            def constructor(*args, **options):
                calls["constructor"].append(options.copy())
                return packed(*args, **options)

            kwargs["NativeLoraLinear4bit"] = constructor
            kwargs["kernel_preflight"] = lambda *args, **options: calls["preflight"].append(options.copy())
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(fixture, "SimpleNamespace", namespace)
    net, life, source, loaded = fixture.make_lifecycle_fixture(tmp_path, monkeypatch)
    return net, life, calls


@pytest.mark.parametrize("logits_mode", ["full", "supervised"])
def test_actual_loader_routes_policy_to_canary_and_all_targets_without_changing_base(
    tmp_path, monkeypatch, logits_mode
):
    from cosmos_framework.posttrain.tests.test_quantized_exclusions import make_lifecycle_fixture

    default, default_life, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch)
    net, life, calls = make_opt_in_fixture(tmp_path, monkeypatch, logits_mode=logits_mode)
    assert len(calls["preflight"]) == 1 and len(calls["constructor"]) == 252
    for options in calls["preflight"] + calls["constructor"]:
        assert options["quantized_linear_execution"] == "checkpointed_dequant"
        assert options["adapter_compute_dtype"] is torch.float32
    actual = copy.deepcopy(net._cosmos_adapter_metadata)
    policy = actual.pop("execution_policy")
    assert actual.pop("format") == "cosmos.adapter-model.v2"
    previous = copy.deepcopy(default._cosmos_adapter_metadata)
    assert previous.pop("format") == "cosmos.adapter-model.v2"
    assert previous.pop("execution_policy") == loader.resolved_execution_policy(default_life.cfg, include_defaults=True)
    assert actual == previous  # Packed identities, source, initialization, targets, strategy and all dtypes.
    assert policy == loader.resolved_execution_policy(life.cfg)
    assert len(net._cosmos_adapter_parameters) == 504
    net._cosmos_validate_frozen_base()
    with pytest.raises(ValueError, match="identity differs"):
        default._cosmos_adapter_checkpoint_codec.validate_metadata(net._cosmos_adapter_checkpoint_codec.metadata())


@pytest.mark.parametrize("logits_mode", ["full", "supervised"])
def test_real_dcp_resume_rejects_changed_mode_before_loading_tensors(tmp_path, monkeypatch, logits_mode):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.entrypoints.sft import _preflight_adapter_resume
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save

    net, life, _ = make_opt_in_fixture(tmp_path, monkeypatch, logits_mode=logits_mode)
    processor, training = {"fixture": "processor"}, {"fixture": "training"}
    codec = net._cosmos_adapter_checkpoint_codec
    codec.bind_processor_identity(processor)
    codec.bind_training_identity(training)
    trainer = LLMTrainer(
        net,
        Objective(lambda *a: pytest.fail("no training in preflight test"), reduction=Reduction.MEAN),
        torch.optim.AdamW(net._cosmos_adapter_parameters.values(), lr=0.01),
        checkpoint_format="dcp",
    )
    path = _save(trainer, tmp_path)
    saved = adapter.read_adapter_metadata(str(path), require_resume=True)
    assert saved["identity"]["execution_policy"]["quantized_linear"]["mode"] == "checkpointed_dequant"
    assert len(saved["parameters"]) == 504
    cfg = make_config()
    cfg.model.model_config.adapter = life.cfg.adapter
    cfg.model.model_config.quantized_base = life.cfg.quantized_base
    cfg.model.model_config.source_identity = life.source_identity
    cfg.model.model_config.quantized_linear_execution = "checkpointed_dequant"
    cfg.model.model_config.sft_logits_mode = logits_mode
    checkpoint = SimpleNamespace(resume_checkpoint=str(path), manager=trainer.checkpoint_manager)
    _preflight_adapter_resume(cfg, checkpoint, processor, training)
    cfg.model.model_config.sft_logits_mode = "supervised" if logits_mode == "full" else "full"
    with pytest.raises(ValueError, match="execution policy"):
        _preflight_adapter_resume(cfg, checkpoint, processor, training)
    cfg.model.model_config.sft_logits_mode = logits_mode
    cfg.model.model_config.quantized_linear_execution = "bnb"
    with pytest.raises(ValueError, match="execution policy"):
        _preflight_adapter_resume(cfg, checkpoint, processor, training)


@pytest.mark.parametrize("defect", ["missing_policy", "missing_record", "wrong_scope"])
def test_committed_artifact_schema_rejects_before_tensor_inventory(tmp_path, monkeypatch, defect):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer

    trainer = _trainer()
    saved = identity()
    if defect == "missing_policy":
        saved.pop("execution_policy")
    elif defect == "missing_record":
        saved["execution_policy"].pop("quantized_linear")
    else:
        saved["execution_policy"]["quantized_linear"]["scope"] = "whole_model"
    trainer._adapter_checkpoint_codec().identity.update(saved)
    checkpoint = _save(trainer, tmp_path)
    monkeypatch.setattr(adapter, "validate_adapter_tensor_inventory", lambda *_a: pytest.fail("tensor load attempted"))
    with pytest.raises(ValueError, match="execution_policy"):
        adapter.read_adapter_metadata(str(checkpoint))
