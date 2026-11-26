# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Execution identities must distinguish every setting that changes adapter arithmetic."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from torch import nn

from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.model.quantization.loader import (
    MODEL_IDENTITY_FORMAT,
    execution_config_from_identity,
    resolved_execution_policy,
    runtime_precision_policy,
    validate_execution_policy_identity,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _cfg(**overrides):
    return SimpleNamespace(
        **{
            "activation_checkpointing": "full",
            "adapter_compute_dtype": "float32",
            "sft_logits_mode": "full",
            "compile_language_model": False,
            "quantized_linear_execution": "bnb",
            **overrides,
        }
    )


def _identity(cfg):
    return {"format": MODEL_IDENTITY_FORMAT, "execution_policy": resolved_execution_policy(cfg)}


def test_default_policy_records_precision_and_roundtrips():
    cfg = _cfg()
    identity = _identity(cfg)
    assert identity["execution_policy"]["math_precision"] == runtime_precision_policy()
    assert execution_config_from_identity(identity) == vars(cfg)
    validate_execution_policy_identity(cfg, identity)
    with pytest.raises(ValueError, match="execution policy"):
        validate_execution_policy_identity(_cfg(adapter_compute_dtype="bfloat16"), identity)


def test_default_native_dcp_and_portable_roundtrip(tmp_path):
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, publish_adapter_artifact
    from cosmos_framework.posttrain.tests.qlora.helpers import adapter_values, make_trainer, save_checkpoint

    def construct():
        trainer = make_trainer()
        trainer._adapter_checkpoint_codec().identity.update(_identity(_cfg()))
        return trainer

    source = construct()
    source.train([None])
    checkpoint = save_checkpoint(source, tmp_path)
    artifact = publish_adapter_artifact(str(checkpoint), str(tmp_path / "portable/checkpoints/iter_000001"))
    saved = adapter_values(source)
    source.train([None])
    expected = adapter_values(source)
    resumed = construct()
    reloaded = construct()
    assert resumed.load(str(checkpoint)) == 1
    load_adapter_weights(artifact, reloaded._adapter_checkpoint_codec(), manager=reloaded.checkpoint_manager)
    for name, parameter in adapter_values(reloaded).items():
        torch.testing.assert_close(parameter, saved[name], atol=0, rtol=0)
    resumed.train([None])
    for name, parameter in adapter_values(resumed).items():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
    assert set(resumed.opt.state) == set(resumed._adapter_checkpoint_codec().parameters.values())


@pytest.mark.parametrize("purpose", ["reload", "resume"])
@pytest.mark.parametrize("defect", ["nondefault", "base", "missing_policy", "missing_math", "null_policy"])
def test_codec_rejects_changed_or_missing_execution_identity(purpose, defect):
    from cosmos_framework.posttrain.tests.qlora.helpers import make_trainer

    trainer = make_trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update(_identity(_cfg()))
    saved = codec.metadata(trainer.opt if purpose == "resume" else None)
    if defect == "nondefault":
        codec.identity["execution_policy"] = resolved_execution_policy(_cfg(adapter_compute_dtype="bfloat16"))
    elif defect == "base":
        saved["identity"]["base"]["revision"] = "b" * 40
    elif defect == "missing_policy":
        saved["identity"].pop("execution_policy")
    elif defect == "missing_math":
        del saved["identity"]["execution_policy"]["math_precision"]
    else:
        saved["identity"]["execution_policy"] = None
    if defect == "missing_math" and purpose == "reload":
        # Runtime math flags are resume-only; read_adapter_metadata still rejects an incomplete record.
        codec.validate_metadata(saved, purpose=purpose)
        return
    with pytest.raises(ValueError, match="identity differs"):
        codec.validate_metadata(saved, trainer.opt if purpose == "resume" else None, purpose=purpose)


@pytest.mark.parametrize(
    "field",
    [
        "hardware",
        "torch_version",
        "torch_git_version",
        "dependencies",
        "initial_adapters",
        "initialization",
        "strategy",
        "math_precision",
        "activation_checkpointing",
        "sft_logits_mode",
        "language_compile",
        "quantized_linear_execution",
        "initialization_seed",
        "dropout",
    ],
)
def test_reload_ignores_recorded_environment_but_resume_compares_it(field):
    from cosmos_framework.posttrain.tests.qlora.helpers import make_trainer

    trainer = make_trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update(
        {
            **_identity(_cfg()),
            "hardware": {"name": "GPU A", "capability": [8, 0]},
            "torch_version": "2.0",
            "torch_git_version": "a" * 40,
            "dependencies": {"python": "3.12.0", "transformers": "4.57.0"},
            "initial_adapters": {"q.lora_A.weight": {"sha256": "a" * 64}},
            "initialization": {"method": "nf4_residual_svd"},
            "strategy": {"activation_checkpointing": "full_nonreentrant"},
            "adapter": {"rank": 2, "alpha": 4.0, "dropout": 0.0, "initialization_seed": 42},
        }
    )
    saved = codec.metadata(trainer.opt)
    identity = saved["identity"]
    policy = identity["execution_policy"]
    if field == "math_precision":
        policy["math_precision"]["cuda_matmul_allow_tf32"] ^= True
    elif field in ("activation_checkpointing", "sft_logits_mode", "language_compile", "quantized_linear_execution"):
        changed = _cfg(
            activation_checkpointing="none",
            sft_logits_mode="supervised",
            compile_language_model=True,
            quantized_linear_execution="checkpointed_dequant",
        )
        policy[field] = resolved_execution_policy(changed)[field]
    elif field in ("initialization_seed", "dropout"):
        identity["adapter"][field] = {"initialization_seed": 7, "dropout": 0.1}[field]
    elif field in ("torch_version", "torch_git_version"):
        identity[field] = "other"
    else:
        identity[field] = {"changed": True}
    for purpose in ("reload", "inference"):
        codec.validate_metadata(saved, purpose=purpose)
    with pytest.raises(ValueError, match="identity differs"):
        codec.validate_metadata(saved, trainer.opt, purpose="resume")


@pytest.mark.parametrize("change", ["rank", "alpha", "adapter_compute_dtype"])
def test_reload_compares_model_defining_adapter_fields(change):
    from cosmos_framework.posttrain.tests.qlora.helpers import make_trainer

    trainer = make_trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update({**_identity(_cfg()), "adapter": {"rank": 2, "alpha": 4.0, "dropout": 0.0}})
    saved = codec.metadata()
    if change == "adapter_compute_dtype":
        saved["identity"]["execution_policy"]["adapter_compute_dtype"] = "bfloat16"
    else:
        saved["identity"]["adapter"][change] = 8
    for purpose in ("reload", "inference"):
        with pytest.raises(ValueError, match="identity differs"):
            codec.validate_metadata(saved, purpose=purpose)


def test_dcp_precision_mismatch_fails_before_tensor_restore(tmp_path, monkeypatch):
    import torch

    from cosmos_framework.posttrain.model.quantization import loader
    from cosmos_framework.posttrain.tests.qlora.helpers import adapter_values, make_trainer, save_checkpoint

    source = make_trainer()
    source._adapter_checkpoint_codec().identity.update(_identity(_cfg()))
    source.train([None])
    checkpoint = save_checkpoint(source, tmp_path)
    changed = deepcopy(runtime_precision_policy())
    changed["allow_bf16_reduced_precision_reduction"] = not changed["allow_bf16_reduced_precision_reduction"]
    monkeypatch.setattr(loader, "runtime_precision_policy", lambda: changed)
    restored = make_trainer()
    restored._adapter_checkpoint_codec().identity.update(_identity(_cfg()))
    before = adapter_values(restored)
    monkeypatch.setattr(
        restored.checkpoint_manager,
        "load_dcp_component",
        lambda *_args, **_kwargs: pytest.fail("precision mismatch must fail before restoring tensors"),
    )
    with pytest.raises(ValueError, match="identity differs"):
        restored.load(str(checkpoint))
    for name, parameter in adapter_values(restored).items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"format": MODEL_IDENTITY_FORMAT},
        {"format": "cosmos.adapter-model.v2"},
        {"format": MODEL_IDENTITY_FORMAT, "execution_policy": {}},
    ],
)
def test_model_schema_cannot_hide_missing_or_unknown_execution_metadata(identity):
    with pytest.raises(ValueError, match="identity|execution_policy"):
        execution_config_from_identity(identity)


@pytest.mark.parametrize(
    "defect", ["missing_flag", "wrong_bool", "missing_backend", "unknown_precision", "bad_environment"]
)
def test_precision_schema_rejects_malformed_records_even_before_a_model_is_built(defect):
    identity = _identity(_cfg(adapter_compute_dtype="bfloat16"))
    precision = identity["execution_policy"]["math_precision"]
    if defect == "missing_flag":
        del precision["cuda_matmul_allow_tf32"]
    elif defect == "wrong_bool":
        precision["cuda_matmul_allow_tf32"] = 1
    elif defect == "missing_backend":
        del precision["fp32_precision"]["cuda_matmul"]
    elif defect == "unknown_precision":
        precision["fp32_precision"]["cuda_matmul"] = "automatic"
    else:
        precision["environment"]["NVIDIA_TF32_OVERRIDE"] = 0
    with pytest.raises(ValueError, match="math_precision"):
        execution_config_from_identity(identity)


@pytest.mark.parametrize(
    "overrides",
    [
        {"activation_checkpointing": "none"},
        {"activation_checkpointing": "selective"},
        {"adapter_compute_dtype": "bfloat16"},
        {"sft_logits_mode": "supervised"},
        {"compile_language_model": True},
        {
            "activation_checkpointing": "selective",
            "adapter_compute_dtype": "bfloat16",
            "compile_language_model": True,
            "sft_logits_mode": "supervised",
        },
    ],
)
def test_nondefault_policy_roundtrips_complete_options_and_math_flags(overrides):
    cfg = _cfg(**overrides)
    identity = _identity(cfg)
    assert identity["execution_policy"]["math_precision"] == runtime_precision_policy()
    assert execution_config_from_identity(identity) == vars(cfg)
    validate_execution_policy_identity(cfg, identity)
    with pytest.raises(ValueError, match="execution policy"):
        validate_execution_policy_identity(_cfg(), identity)


@pytest.mark.parametrize(
    "defect",
    [
        "protocol",
        "missing_math",
        "missing_compile",
        "save_regex",
        "unknown",
        "invalid_compute",
        "invalid_mode",
        "integer_bool",
        "null",
    ],
)
def test_reconstruction_rejects_missing_unknown_or_unsupported_execution_metadata(defect):
    identity = _identity(_cfg(activation_checkpointing="selective"))
    policy = identity["execution_policy"]
    if defect == "protocol":
        policy["protocol"] = "future-version"
    elif defect == "missing_math":
        del policy["math_precision"]
    elif defect == "missing_compile":
        del policy["language_compile"]
    elif defect == "save_regex":
        policy["activation_checkpointing"]["save_ops_regex"] = []
    elif defect == "unknown":
        policy["unknown"] = True
    elif defect == "invalid_compute":
        policy["adapter_compute_dtype"] = "float16"
    elif defect == "invalid_mode":
        policy["activation_checkpointing"]["mode"] = "unknown"
    elif defect == "integer_bool":
        policy["language_compile"]["enabled"] = 1
    else:
        identity["execution_policy"] = None
    with pytest.raises(ValueError, match="execution_policy"):
        execution_config_from_identity(identity)


def test_current_math_policy_mismatch_is_rejected_without_mutation(monkeypatch):
    from cosmos_framework.posttrain.model.quantization import loader

    cfg = _cfg(adapter_compute_dtype="bfloat16")
    identity = _identity(cfg)
    actual_before = runtime_precision_policy()
    changed = deepcopy(actual_before)
    changed["cuda_matmul_allow_tf32"] = not changed["cuda_matmul_allow_tf32"]
    monkeypatch.setattr(loader, "runtime_precision_policy", lambda: changed)
    with pytest.raises(ValueError, match="runtime math precision"):
        validate_execution_policy_identity(cfg, identity)
    assert runtime_precision_policy() == actual_before


def test_codec_rejects_same_parameters_with_changed_execution_identity():
    net = nn.Linear(4, 2, bias=False)
    net._cosmos_adapter_parameters = {"lora_A.weight": net.weight}
    original = build_adapter_checkpoint_codec(
        net, identity={"execution_policy": resolved_execution_policy(_cfg(adapter_compute_dtype="bfloat16"))}
    )
    changed = build_adapter_checkpoint_codec(
        net, identity={"execution_policy": resolved_execution_policy(_cfg(activation_checkpointing="none"))}
    )
    with pytest.raises(ValueError, match="identity differs"):
        changed.validate_metadata(original.metadata())


@pytest.mark.parametrize("defect", ["missing_policy", "unknown_format", "incomplete_math"])
def test_committed_native_checkpoint_schema_fails_before_tensor_loading(tmp_path, monkeypatch, defect):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.tests.qlora.helpers import make_trainer, save_checkpoint

    trainer = make_trainer()
    identity = trainer._adapter_checkpoint_codec().identity
    identity.update(
        {
            "model_format": "cosmos_native_qwen3_vl",
            "format": MODEL_IDENTITY_FORMAT,
            "execution_policy": resolved_execution_policy(_cfg(adapter_compute_dtype="bfloat16")),
        }
    )
    if defect == "missing_policy":
        del identity["execution_policy"]
    elif defect == "unknown_format":
        identity["format"] = "cosmos.adapter-model.v2"
    else:
        del identity["execution_policy"]["math_precision"]["fp32_precision"]
    checkpoint = save_checkpoint(trainer, tmp_path)
    monkeypatch.setattr(
        adapter,
        "validate_adapter_tensor_inventory",
        lambda *_args: pytest.fail("schema must fail before tensor inventory/load"),
    )
    with pytest.raises(ValueError, match="execution_policy|identity format"):
        adapter.read_adapter_metadata(str(checkpoint))
