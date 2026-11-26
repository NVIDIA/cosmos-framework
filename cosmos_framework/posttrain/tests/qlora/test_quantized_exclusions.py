# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Mixed NF4/BF16 base coverage on CPU stand-ins (the NF4 kernels are covered by GPU tests)."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.config.model import (
    NativeLoRAConfig,
    QuantizedBaseTrainingConfig,
    Qwen3VLModelConfig,
    quantization_config_identity,
)
from cosmos_framework.posttrain.model.quantization import initialization, loader
from cosmos_framework.posttrain.tests.qlora.helpers import make_lifecycle_fixture
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("layers", [(True,), (-1,), (36,), (1, 1), (0, 1, 2), ("6",), "6", None, {6}])
def test_invalid_or_unbounded_exclusions_refused(layers):
    with pytest.raises(ValueError, match="excluded_language_layers"):
        QuantizedBaseTrainingConfig(excluded_language_layers=layers)


def test_default_identity_is_the_fixed_nf4_representation_and_nonempty_is_canonical():
    expected = {
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
    assert quantization_config_identity(QuantizedBaseTrainingConfig()) == expected
    assert quantization_config_identity(None) is None
    cfg = QuantizedBaseTrainingConfig(excluded_language_layers=[15, 6])
    assert cfg.excluded_language_layers == (6, 15)
    assert quantization_config_identity(cfg) == dict(expected, excluded_language_layers=(6, 15))


def test_residual_combination_rejected_before_artifact_or_base_reads(tmp_path, monkeypatch):
    quant = QuantizedBaseTrainingConfig(excluded_language_layers=(6,))
    adapter = NativeLoRAConfig(
        initialization="nf4_residual_svd",
        initialization_artifact=str(tmp_path / "manifest.json"),
        initialization_artifact_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="excluded language"):
        Qwen3VLModelConfig(attn_impl="sdpa", adapter=adapter, quantized_base=quant)
    with pytest.raises(ValueError, match="excluded language"):
        initialization.InitializationArtifact(
            adapter.initialization_artifact, "a" * 64, adapter=adapter, source_identity={}, quantization=quant
        )
    # Preparation also accepts a zero-B source config; it must reject exclusions itself.
    cfg = SimpleNamespace(adapter=NativeLoRAConfig(), quantized_base=quant)
    with pytest.raises(ValueError, match="excluded language"):
        initialization.prepare_initialization_artifact(cfg, tmp_path / "out", model_config=object())
    assert not (tmp_path / "out").exists()


def test_unmatched_or_partial_excluded_layer_refused():
    targets = loader.canonical_targets()
    targets.pop("model.language_model.layers.6.self_attn.q_proj")
    with pytest.raises(ValueError, match="exact canonical targets"):
        loader.quantization_coverage(QuantizedBaseTrainingConfig(excluded_language_layers=(6,)), targets)


@pytest.mark.parametrize("layers", [(), (6,), (6, 15)])
def test_actual_lifecycle_covers_all_adapters_and_only_materializes_exclusions(tmp_path, monkeypatch, layers):
    net, life, source, loaded = make_lifecycle_fixture(tmp_path, monkeypatch, layers)
    assert len(loaded) == 252 - 7 * len(layers)
    parameters = net._cosmos_adapter_parameters
    assert len(parameters) == len({id(p) for p in parameters.values()}) == 504
    assert all(p.dtype == torch.float32 and p.requires_grad for p in parameters.values())
    for name, module in life.targets.items():
        if name in life.quantized_targets:
            assert module.weight.dtype == torch.uint8
        else:
            assert type(module) is LoraInjectedLinear and torch.equal(module.weight, source[name + ".weight"])
        assert not module.weight.requires_grad
    optimizer = torch.optim.AdamW(parameters.values(), lr=0.01)
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    net._cosmos_adapter_checkpoint_codec.assert_live(optimizer=optimizer)
    assert set(optimizer.state) == set(parameters.values())
    net._cosmos_validate_frozen_base()
    identity = json.loads(json.dumps(net._cosmos_adapter_metadata))
    assert identity["format"] == loader.MODEL_IDENTITY_FORMAT
    assert ("quantization_coverage" in identity) == bool(layers)
    assert identity["execution_policy"] == loader.resolved_execution_policy(life.cfg)
    assert len(identity["quantized_tensors"]) == 252 - 7 * len(layers)
    assert len(identity["excluded_frozen_tensors"]) == 7 * len(layers)
    if layers:
        loader.execution_config_from_identity(identity)
        bad = copy.deepcopy(identity)
        bad["quantization"]["excluded_language_layers"] = [7]
        with pytest.raises(ValueError, match="coverage"):
            loader.execution_config_from_identity(bad)
        bad = copy.deepcopy(identity)
        bad.pop("quantization_coverage")
        with pytest.raises(ValueError, match="coverage"):
            loader.execution_config_from_identity(bad)
        wrong = copy.copy(life.cfg)
        wrong.quantized_base = QuantizedBaseTrainingConfig()
        with pytest.raises(ValueError, match="quantization differs"):
            loader.validate_execution_policy_identity(wrong, identity)


def test_adapter_rng_initialization_exact_and_frozen_both_paths_guarded(tmp_path, monkeypatch):
    full, _, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch)
    mixed, life, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15))
    for name, value in full._cosmos_adapter_parameters.items():
        assert torch.equal(value, mixed._cosmos_adapter_parameters[name])
    dense = life.targets["model.language_model.layers.6.self_attn.q_proj"].weight
    with torch.no_grad():
        dense.add_(1)
    with pytest.raises(RuntimeError, match="frozen native base tensor"):
        mixed._cosmos_validate_frozen_base()
    with torch.no_grad():
        dense.sub_(1)
    # BF16 addition/subtraction is not guaranteed reversible, so test packed integrity on a fresh model.
    packed, life, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6,))
    with torch.no_grad():
        next(iter(life.quantized_targets.values())).weight.add_(1)
    with pytest.raises(RuntimeError, match="packed base weight"):
        packed._cosmos_validate_frozen_base()


def test_checkpoint_rebuild_rejects_changed_exclusion_and_preserves_all_parameters(tmp_path, monkeypatch):
    left, _, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15))
    right, _, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15))
    right._cosmos_adapter_checkpoint_codec.validate_metadata(left._cosmos_adapter_checkpoint_codec.metadata())
    with torch.no_grad():
        for p in left._cosmos_adapter_parameters.values():
            p.add_(0.25)
    right._cosmos_adapter_checkpoint_codec.view.load_state_dict(left._cosmos_adapter_checkpoint_codec.view.state_dict())
    assert all(torch.equal(p, right._cosmos_adapter_parameters[n]) for n, p in left._cosmos_adapter_parameters.items())
    other, _, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6,))
    with pytest.raises(ValueError, match="identity differs"):
        other._cosmos_adapter_checkpoint_codec.validate_metadata(left._cosmos_adapter_checkpoint_codec.metadata())


def test_real_all_adapter_dcp_rejects_wrong_exclusions_during_preflight(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.entrypoints.sft import preflight_adapter_resume
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.qlora.helpers import save_checkpoint

    net, life, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6, 15))
    processor, training = {"fixture": "processor"}, {"fixture": "training"}
    codec = net._cosmos_adapter_checkpoint_codec
    codec.bind_processor_identity(processor)
    codec.bind_training_identity(training)
    optimizer = torch.optim.AdamW(net._cosmos_adapter_parameters.values(), lr=0.01)
    trainer = LLMTrainer(
        net,
        Objective(lambda *a: pytest.fail("no training in preflight test"), reduction=Reduction.MEAN),
        optimizer,
        checkpoint_format="dcp",
    )
    path = save_checkpoint(trainer, tmp_path)
    saved = read_adapter_metadata(str(path), require_resume=True)
    assert len(saved["parameters"]) == 504 and saved["identity"]["format"] == loader.MODEL_IDENTITY_FORMAT
    config = make_config()
    config.model.model_config.adapter = life.cfg.adapter
    config.model.model_config.quantized_base = life.cfg.quantized_base
    config.model.model_config.source_identity = life.source_identity
    checkpoint = SimpleNamespace(resume_checkpoint=str(path), manager=trainer.checkpoint_manager)
    preflight_adapter_resume(config, checkpoint, life.source_identity, processor, training)
    config.model.model_config.quantized_base = QuantizedBaseTrainingConfig(excluded_language_layers=(6,))
    with pytest.raises(ValueError, match="excluded_language_layers"):
        preflight_adapter_resume(config, checkpoint, life.source_identity, processor, training)


@pytest.mark.parametrize(
    "change",
    [
        "missing_packed",
        "dense_as_packed",
        "missing_dense",
        "extra_coverage",
        "bool_layer",
        "missing_adapter",
        "residual",
    ],
)
def test_new_identity_requires_complete_exact_coverage(tmp_path, monkeypatch, change):
    net, _, _, _ = make_lifecycle_fixture(tmp_path, monkeypatch, (6,))
    identity = json.loads(json.dumps(net._cosmos_adapter_metadata))
    if change == "missing_packed":
        identity["quantized_tensors"].pop(next(iter(identity["quantized_tensors"])))
    elif change == "dense_as_packed":
        identity["quantized_tensors"]["model.language_model.layers.6.self_attn.q_proj"] = {}
    elif change == "missing_dense":
        identity["excluded_frozen_tensors"].pop(next(iter(identity["excluded_frozen_tensors"])))
    elif change == "extra_coverage":
        identity["quantization_coverage"]["unknown"] = True
    elif change == "bool_layer":
        identity["quantization"]["excluded_language_layers"] = [True]
    elif change == "missing_adapter":
        identity["adapter"] = None
    else:
        identity["adapter"]["initialization"] = "nf4_residual_svd"
    with pytest.raises(ValueError, match="mixed-base"):
        loader.execution_config_from_identity(identity)
