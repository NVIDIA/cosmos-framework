# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Execution identities must distinguish arithmetic changes without guessing legacy settings."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from torch import nn

from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.model.quantization.loader import (
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
            **overrides,
        }
    )


def test_default_identity_retains_legacy_reload_without_fabricating_precision_history():
    assert resolved_execution_policy(_cfg()) is None
    assert execution_config_from_identity({}) == vars(_cfg())
    validate_execution_policy_identity(_cfg(), {})
    with pytest.raises(ValueError, match="execution policy"):
        validate_execution_policy_identity(_cfg(adapter_compute_dtype="bfloat16"), {})


def test_new_default_policy_records_precision_and_existing_optimized_policy_is_unchanged():
    cfg = _cfg()
    recorded = resolved_execution_policy(cfg, include_defaults=True)
    assert recorded["math_precision"] == runtime_precision_policy()
    identity = {"format": "cosmos.adapter-model.v2", "execution_policy": recorded}
    assert execution_config_from_identity(identity) == vars(cfg)
    validate_execution_policy_identity(cfg, identity)
    for overrides in ({"sft_logits_mode": "supervised"}, {"activation_checkpointing": "none"}):
        cfg = _cfg(**overrides)
        assert resolved_execution_policy(cfg, include_defaults=True) == resolved_execution_policy(cfg)


@pytest.mark.parametrize("field", ["cuda_matmul_allow_tf32", "allow_bf16_reduced_precision_reduction"])
def test_new_default_runtime_mismatch_rejected_by_reload_preflight(field, monkeypatch):
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.model.quantization import loader
    from cosmos_framework.posttrain.tests.qlora_integration import _restore_probe_execution

    identity = {
        "format": "cosmos.adapter-model.v2",
        "execution_policy": resolved_execution_policy(_cfg(), include_defaults=True),
    }
    changed = deepcopy(runtime_precision_policy())
    changed[field] = not changed[field]
    monkeypatch.setattr(loader, "runtime_precision_policy", lambda: changed)
    # The real probe preflight runs before build_backbone or any adapter tensor load.
    with pytest.raises(ValueError, match="runtime math precision"):
        _restore_probe_execution(make_config(), identity, [])


@pytest.mark.parametrize("legacy", [False, True])
def test_default_native_dcp_and_portable_roundtrip_with_explicit_legacy_scope(tmp_path, legacy):
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, publish_adapter_artifact
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _parameters, _save, _trainer

    policy = resolved_execution_policy(_cfg(), include_defaults=True)

    def construct(*, old=False):
        trainer = _trainer()
        identity = trainer._adapter_checkpoint_codec().identity
        identity["format"] = "cosmos.adapter-model.v1" if old else "cosmos.adapter-model.v2"
        if not old:
            identity["execution_policy"] = deepcopy(policy)
        return trainer

    source = construct(old=legacy)
    source.train([None])
    checkpoint = _save(source, tmp_path)
    artifact = publish_adapter_artifact(str(checkpoint), str(tmp_path / "portable/checkpoints/iter_000001"))
    saved = _parameters(source)
    source.train([None])
    expected = _parameters(source)
    resumed = construct()
    reloaded = construct()
    if legacy:
        with pytest.warns(UserWarning, match="no recorded runtime math precision"):
            assert resumed.load(str(checkpoint)) == 1
        with pytest.warns(UserWarning, match="no recorded runtime math precision"):
            load_adapter_weights(artifact, reloaded._adapter_checkpoint_codec(), manager=reloaded.checkpoint_manager)
    else:
        assert resumed.load(str(checkpoint)) == 1
        load_adapter_weights(artifact, reloaded._adapter_checkpoint_codec(), manager=reloaded.checkpoint_manager)
    for name, parameter in _parameters(reloaded).items():
        torch.testing.assert_close(parameter, saved[name], atol=0, rtol=0)
    resumed.train([None])
    for name, parameter in _parameters(resumed).items():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
    # A subsequent save records the current process's precision, never a fabricated old observation.
    assert resumed._adapter_checkpoint_codec().identity["execution_policy"] == policy
    assert set(resumed.opt.state) == set(resumed._adapter_checkpoint_codec().parameters.values())


@pytest.mark.parametrize("purpose", ["reload", "resume"])
@pytest.mark.parametrize(
    "defect", ["nondefault", "base", "missing_math", "present_null", "missing_v2_policy", "null_v2_policy"]
)
def test_legacy_compatibility_does_not_hide_other_identity_or_policy_changes(purpose, defect):
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer

    trainer = _trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update(
        format="cosmos.adapter-model.v2", execution_policy=resolved_execution_policy(_cfg(), include_defaults=True)
    )
    saved = codec.metadata(trainer.opt if purpose == "resume" else None)
    saved["identity"]["format"] = "cosmos.adapter-model.v1"
    saved["identity"].pop("execution_policy")
    if defect == "nondefault":
        codec.identity["execution_policy"] = resolved_execution_policy(_cfg(sft_logits_mode="supervised"))
    elif defect == "base":
        saved["identity"]["base"]["revision"] = "b" * 40
    elif defect == "present_null":
        saved["identity"]["execution_policy"] = None
    else:
        saved["identity"]["format"] = "cosmos.adapter-model.v2"
        if defect == "missing_math":
            saved["identity"]["execution_policy"] = deepcopy(codec.identity["execution_policy"])
            del saved["identity"]["execution_policy"]["math_precision"]
        elif defect == "null_v2_policy":
            saved["identity"]["execution_policy"] = None
    with pytest.raises(ValueError, match="identity differs"):
        codec.validate_metadata(saved, trainer.opt if purpose == "resume" else None, purpose=purpose)


def test_new_default_dcp_precision_mismatch_fails_before_tensor_restore(tmp_path, monkeypatch):
    import torch

    from cosmos_framework.posttrain.model.quantization import loader
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _parameters, _save, _trainer

    source = _trainer()
    source._adapter_checkpoint_codec().identity.update(
        format="cosmos.adapter-model.v2", execution_policy=resolved_execution_policy(_cfg(), include_defaults=True)
    )
    source.train([None])
    checkpoint = _save(source, tmp_path)
    changed = deepcopy(runtime_precision_policy())
    changed["allow_bf16_reduced_precision_reduction"] = not changed["allow_bf16_reduced_precision_reduction"]
    monkeypatch.setattr(loader, "runtime_precision_policy", lambda: changed)
    restored = _trainer()
    restored._adapter_checkpoint_codec().identity.update(
        format="cosmos.adapter-model.v2", execution_policy=resolved_execution_policy(_cfg(), include_defaults=True)
    )
    before = _parameters(restored)
    monkeypatch.setattr(
        restored.checkpoint_manager,
        "load_dcp_component",
        lambda *_args, **_kwargs: pytest.fail("precision mismatch must fail before restoring tensors"),
    )
    with pytest.raises(ValueError, match="identity differs"):
        restored.load(str(checkpoint))
    for name, parameter in _parameters(restored).items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_legacy_precision_compatibility_does_not_authorize_cross_build_resume():
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer

    trainer = _trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update(
        format="cosmos.adapter-model.v2", execution_policy=resolved_execution_policy(_cfg(), include_defaults=True)
    )
    codec.identity["training_contract"].update(code_revision="a" * 40, wheel_sha256="b" * 64)
    saved = codec.metadata(trainer.opt)
    saved["identity"]["format"] = "cosmos.adapter-model.v1"
    saved["identity"].pop("execution_policy")
    saved["identity"]["training_contract"]["code_revision"] = "c" * 40
    with pytest.warns(UserWarning, match="no recorded runtime math precision"):
        with pytest.raises(ValueError, match="identity differs"):
            codec.validate_metadata(saved, trainer.opt, purpose="resume")


@pytest.mark.parametrize(
    "identity",
    [
        {"format": "cosmos.adapter-model.v2"},
        {"format": "cosmos.adapter-model.future"},
        {"format": "cosmos.adapter-model.v1", "execution_policy": {}},
    ],
)
def test_model_schema_cannot_hide_missing_or_unknown_execution_metadata(identity):
    with pytest.raises(ValueError, match="identity|execution_policy"):
        execution_config_from_identity(identity)


@pytest.mark.parametrize(
    "defect", ["missing_flag", "wrong_bool", "missing_backend", "unknown_precision", "bad_environment"]
)
def test_precision_schema_rejects_malformed_records_even_before_a_model_is_built(defect):
    identity = {
        "format": "cosmos.adapter-model.v2",
        "execution_policy": resolved_execution_policy(_cfg(adapter_compute_dtype="bfloat16")),
    }
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
    identity = {"execution_policy": resolved_execution_policy(cfg)}
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
    identity = {"execution_policy": resolved_execution_policy(_cfg(activation_checkpointing="selective"))}
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
    identity = {"execution_policy": resolved_execution_policy(cfg)}
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
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer

    trainer = _trainer()
    identity = trainer._adapter_checkpoint_codec().identity
    identity.update(
        {
            "model_format": "cosmos_native_qwen3_vl",
            "format": "cosmos.adapter-model.v2",
            "execution_policy": resolved_execution_policy(_cfg(adapter_compute_dtype="bfloat16")),
        }
    )
    if defect == "missing_policy":
        del identity["execution_policy"]
    elif defect == "unknown_format":
        identity["format"] = "cosmos.adapter-model.future"
    else:
        del identity["execution_policy"]["math_precision"]["fp32_precision"]
    checkpoint = _save(trainer, tmp_path)
    monkeypatch.setattr(
        adapter,
        "validate_adapter_tensor_inventory",
        lambda *_args: pytest.fail("schema must fail before tensor inventory/load"),
    )
    with pytest.raises(ValueError, match="execution_policy|identity format"):
        adapter.read_adapter_metadata(str(checkpoint))
