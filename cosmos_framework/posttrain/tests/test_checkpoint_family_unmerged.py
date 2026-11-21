# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Standalone native LoRA artifacts; CPU tests do not qualify NF4 accelerator parity."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.dense_adapter import _digest_json
from cosmos_framework.posttrain.checkpoint.family_dense import (
    FAMILY_DENSE_MANIFEST,
    FAMILY_UNMERGED_MANIFEST,
    export_family_unmerged,
    load_family_unmerged,
    read_family_dense,
    read_family_unmerged,
)
from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
from cosmos_framework.posttrain.tests.test_checkpoint_family_dense import (
    BUILD,
    AliasedTaskNet,
    TaskNet,
    _export,
    _prepared,
)
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def exported(tmp_path, *, aliased=False):
    net, state, source, data = _prepared(tmp_path, net=AliasedTaskNet() if aliased else None)
    output = tmp_path / "unmerged"
    manifest = export_family_unmerged(
        net, checkpoint=source, output=str(output), exporter_build=BUILD, max_shard_size_bytes=48
    )
    return net, state, source, data, output, manifest


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_unmerged_retains_exact_native_factors_heads_base_buffers_and_separate_arithmetic(tmp_path, autocast, aliased):
    net, state, _, data, output, manifest = exported(tmp_path, aliased=aliased)
    before = {name: tensor_digest(value) for name, value in net.named_parameters()}
    fresh = AliasedTaskNet(device="meta") if aliased else TaskNet(device="meta")
    rng = torch.get_rng_state().clone()
    load_family_unmerged(fresh, output, device="cpu", expected_identity=state.identity, manifest=manifest)
    assert torch.equal(rng, torch.get_rng_state())
    assert type(fresh.backbone) is LoraInjectedLinear
    assert fresh.backbone.weight.dtype == torch.bfloat16
    assert fresh.backbone.lora_A.weight.dtype == fresh.backbone.lora_B.weight.dtype == torch.float32
    assert fresh.backbone._lora_scale == net.backbone._lora_scale
    assert all(not value.is_meta and not value.requires_grad for value in fresh.parameters())
    assert not fresh.training and not hasattr(fresh, "_cosmos_adapter_checkpoint_codec")
    assert set(manifest["trained_tensors"]) == set(state.codec.parameters)
    assert {name: tensor_digest(value) for name, value in fresh.named_parameters()} == before
    assert manifest["source_metadata"]["identity"] == state.codec.identity
    assert manifest["numerical_contract"]["merged"] is False
    assert manifest["numerical_contract"]["requantized"] is False
    assert manifest["resumable_training_checkpoint"] is False
    if aliased:
        assert fresh.embedding.weight is fresh.output.weight
    net.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        torch.testing.assert_close(fresh(data), net(data), rtol=0, atol=0)
    for name, value in fresh.named_buffers():
        torch.testing.assert_close(value, net.get_buffer(name), rtol=0, atol=0)
    assert before == {name: tensor_digest(value) for name, value in net.named_parameters()}
    state.validate_frozen_base()


def test_fresh_process_load_needs_no_original_weights_quantizer_or_adapter_initialization(tmp_path):
    net, _, source, data, output, _ = exported(tmp_path, aliased=True)
    net.eval()
    torch.save({"input": data, "output": net(data)}, tmp_path / "fixture.pt")
    shutil.rmtree(source)
    assert not os.path.exists(source)
    script = r"""
import sys, torch
from cosmos_framework.posttrain.tests.test_checkpoint_family_dense import AliasedTaskNet
from cosmos_framework.posttrain.checkpoint import family_dense, adapter
from cosmos_framework.posttrain.model.quantization import family
from cosmos_framework.utils.generator.lora import NativeLoRAMixin
def forbidden(*a, **k): raise AssertionError('original source, quantizer or adapter initialization called')
adapter.load_adapter_weights = forbidden
family_dense.load_adapter_weights = forbidden
family_dense._effective_weight = forbidden
family.install_family_backbone = forbidden
NativeLoRAMixin.reset_adapter_parameters = forbidden
# Deployed loading must work with the quantizer module entirely unavailable.
sys.modules['cosmos_framework.posttrain.model.quantization.bnb'] = None
sys.modules['bitsandbytes'] = None
net = AliasedTaskNet(device='meta')
rng = torch.get_rng_state().clone()
manifest = family_dense.read_family_unmerged(sys.argv[1])
family_dense.load_family_unmerged(net, sys.argv[1], device='cpu', expected_identity={'base':manifest['source_metadata']['identity']['base']})
assert torch.equal(rng, torch.get_rng_state())
assert net.embedding.weight is net.output.weight
fixture = torch.load(sys.argv[2], weights_only=True)
torch.testing.assert_close(net(fixture['input']), fixture['output'], rtol=0, atol=0)
assert all(not p.requires_grad for p in net.parameters())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), str(tmp_path / "fixture.pt")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_formats_are_explicit_and_never_autodetected(tmp_path):
    _, _, _, _, output, _ = exported(tmp_path)
    assert not (output / FAMILY_DENSE_MANIFEST).exists()
    with pytest.raises(ValueError, match="incomplete"):
        read_family_dense(output)
    other = tmp_path / "other"
    other.mkdir()
    *_, dense, _ = _export(other)
    with pytest.raises(ValueError, match="incomplete"):
        read_family_unmerged(dense)


@pytest.mark.parametrize(
    "change",
    [
        "missing_factor",
        "factor_digest",
        "factor_dtype",
        "rank",
        "alpha",
        "mode",
        "quantization",
        "precision",
        "merge",
        "missing_head",
        "arithmetic_source",
        "base_digest",
    ],
)
def test_unmerged_rejects_missing_state_wrong_arithmetic_or_changed_source(tmp_path, change):
    _, state, _, _, output, manifest = exported(tmp_path)
    identity = manifest["source_metadata"]["identity"]
    if change == "missing_factor":
        del manifest["tensors"]["backbone.lora_A.weight"]
    elif change == "factor_digest":
        manifest["tensors"]["backbone.lora_B.weight"]["sha256"] = "0" * 64
    elif change == "factor_dtype":
        manifest["tensors"]["backbone.lora_A.weight"]["dtype"] = "bfloat16"
    elif change in {"rank", "alpha"}:
        identity["adapter"][change] += 1
    elif change == "mode":
        identity["adapter"]["mode"] = "bf16_frozen"
    elif change == "quantization":
        identity["quantization"] = {"backend": "another"}
    elif change == "precision":
        identity["precision"]["adapter"] = "bfloat16"
    elif change == "merge":
        manifest["numerical_contract"]["merged"] = True
    elif change == "missing_head":
        del manifest["trained_tensors"]["head.bias"]
    elif change == "arithmetic_source":
        identity["implementation"]["adapter"] = "0" * 64
    else:
        manifest["tensors"]["backbone.weight"]["sha256"] = "0" * 64
    manifest["source_metadata_sha256"] = _digest_json(manifest["source_metadata"])
    (output / FAMILY_UNMERGED_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_family_unmerged(TaskNet(device="meta"), output, device="cpu", expected_identity=state.identity)


@pytest.mark.parametrize("key", ["quantization", "adapter"])
@pytest.mark.parametrize("unmerged", [False, True])
def test_missing_identity_keys_report_value_error_for_both_formats(tmp_path, key, unmerged):
    *_, output, manifest = exported(tmp_path) if unmerged else _export(tmp_path)
    del manifest["source_metadata"]["identity"][key]
    manifest["source_metadata_sha256"] = _digest_json(manifest["source_metadata"])
    marker = FAMILY_UNMERGED_MANIFEST if unmerged else FAMILY_DENSE_MANIFEST
    (output / marker).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source or numerical contract"):
        (read_family_unmerged if unmerged else read_family_dense)(output)


def test_current_native_adapter_implementation_is_required_even_without_full_expected_identity(tmp_path):
    _, _, _, _, output, manifest = exported(tmp_path)
    manifest["source_metadata"]["identity"]["implementation"]["adapter"] = "0" * 64
    manifest["source_metadata_sha256"] = _digest_json(manifest["source_metadata"])
    (output / FAMILY_UNMERGED_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="arithmetic source"):
        load_family_unmerged(
            TaskNet(device="meta"), output, device="cpu", expected_identity={"format": "cosmos.family-backbone.v1"}
        )


def test_interruption_cannot_publish_unmerged_artifact(tmp_path, monkeypatch):
    import safetensors.torch

    net, _, source, _ = _prepared(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("interrupted publication")

    monkeypatch.setattr(safetensors.torch, "save_file", fail)
    output = tmp_path / "unmerged"
    with pytest.raises(OSError, match="interrupted"):
        export_family_unmerged(net, checkpoint=source, output=str(output), exporter_build=BUILD)
    assert not (output / FAMILY_UNMERGED_MANIFEST).exists()
    with pytest.raises(ValueError, match="incomplete"):
        read_family_unmerged(output)


def test_unmerged_does_not_silently_cast_or_change_adapter_compute(tmp_path):
    net, _, source, _ = _prepared(tmp_path)
    net.backbone._adapter_compute_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="arithmetic"):
        export_family_unmerged(net, checkpoint=source, output=str(tmp_path / "unmerged"), exporter_build=BUILD)
    assert not (tmp_path / "unmerged").exists()


def test_nonzero_dropout_metadata_and_native_scaling_survive_inference_only_load(tmp_path):
    net, state, source, data = _prepared(tmp_path, dropout=0.05)
    output = tmp_path / "unmerged"
    export_family_unmerged(net, checkpoint=source, output=str(output), exporter_build=BUILD)
    fresh = TaskNet(device="meta")
    load_family_unmerged(fresh, output, device="cpu", expected_identity=state.codec.identity)
    assert type(fresh.backbone.lora_dropout) is nn.Dropout
    assert fresh.backbone.lora_dropout.p == 0.05
    assert fresh.backbone._lora_rank == net.backbone._lora_rank == 2
    assert fresh.backbone._lora_alpha == net.backbone._lora_alpha == 4
    assert fresh.backbone._lora_use_rslora is False
    assert all(not module.training for module in fresh.modules())
    net.eval()
    torch.testing.assert_close(fresh(data), net(data), atol=0, rtol=0)


def test_export_cli_requires_one_explicit_representation():
    from cosmos_framework.posttrain.checkpoint.tools.export_family_dense import parser

    required = ["--config", "native", "--checkpoint", "portable", "--output", "fresh"]
    assert parser().parse_args(required + ["--unmerged"]).unmerged
    assert parser().parse_args(required + ["--merged-backbone-dtype", "bfloat16"]).unmerged is False
    for extra in ([], ["--unmerged", "--merged-backbone-dtype", "bfloat16"]):
        with pytest.raises(SystemExit):
            parser().parse_args(required + extra)
