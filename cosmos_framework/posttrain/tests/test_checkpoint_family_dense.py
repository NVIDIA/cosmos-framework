# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native dense task artifacts on CPU; NF4 kernels/task quality need accelerator evidence."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata, save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.family_dense import (
    FAMILY_DENSE_MANIFEST,
    _mixed_shard_plan,
    export_family_dense,
    load_family_dense,
    read_family_dense,
)
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone
from cosmos_framework.posttrain.model.quantization.loader import tensor_digest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]
BUILD = {"code_revision": "1" * 40, "wheel_sha256": "2" * 64}


class TaskNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(39)
            self.backbone = nn.Linear(3, 4, bias=False, device=device, dtype=torch.bfloat16)
            self.head = nn.Linear(4, 2, device=device, dtype=torch.float32)
        self.register_buffer("action_scale", torch.tensor([0.2, 0.3], dtype=torch.float32), persistent=False)
        self.register_buffer("frame_id", torch.tensor([17], dtype=torch.int64))

    def forward(self, x):
        return self.head(self.backbone(x.to(self.backbone.weight.dtype)).float()) * self.action_scale


class AliasedTaskNet(TaskNet):
    def __init__(self, device="cpu", *, tied=True):
        super().__init__(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(53)
            self.embedding = nn.Embedding(5, 3, device=device, dtype=torch.bfloat16)
            self.output = nn.Linear(3, 5, bias=False, device=device, dtype=torch.bfloat16)
        if tied:
            self.output.weight = self.embedding.weight

    def forward(self, x):
        return super().forward(x + self.embedding(torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)))


def _prepared(tmp_path, *, mode="bf16_lora", net=None, dropout=0.0):
    net = TaskNet() if net is None else net
    state = install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode=mode, rank=2, alpha=4, seed=7, dropout=dropout),
        targets={"backbone": (4, 3)},
        auxiliary_parameters=("head.weight", "head.bias"),
        source_identity={"repository": "fixture/native-action", "revision": "a" * 40},
        task_identity={"family": "fixture", "action_schema": "two-scaled-axes", "horizon": 1},
        device=torch.device("cpu"),
    )
    state.codec.bind_processor_identity({"format": "fixture", "pixels": [4, 4]})
    state.codec.bind_training_identity({"steps": 3, "seed": 9})
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=0.01)
    data = torch.tensor([[0.1, 0.2, -0.3], [-0.3, 0.1, 0.2]])
    target = torch.tensor([[0.01, -0.02], [-0.02, 0.01]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (net(data) - target).square().mean().backward()
        optimizer.step()
    source = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000003"),
        state.codec.parameters,
        state.codec.metadata(optimizer),
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    return net, state, source, data


def _export(tmp_path, *, mode="bf16_lora", dtype="float32", max_bytes=60, sidecars=None):
    net, state, source, data = _prepared(tmp_path, mode=mode)
    output = tmp_path / "dense"
    manifest = export_family_dense(
        net,
        checkpoint=source,
        output=str(output),
        merged_backbone_dtype=dtype,
        max_shard_size_bytes=max_bytes,
        exporter_build=BUILD,
        sidecars=sidecars,
    )
    return net, state, source, data, output, manifest


@pytest.mark.parametrize("mode", ["bf16_lora", "bf16_frozen"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_dense_export_preserves_native_task_heads_buffers_and_exact_declared_merge(tmp_path, mode, dtype):
    net, state, source, data, output, manifest = _export(tmp_path, mode=mode, dtype=dtype)
    loaded = TaskNet(device="meta")
    rng = torch.get_rng_state().clone()
    load_family_dense(loaded, output, device="cpu", expected_identity=state.identity, manifest=manifest)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(not p.requires_grad and not p.is_meta for p in loaded.parameters())
    assert not loaded.training and not hasattr(loaded, "_cosmos_adapter_checkpoint_codec")
    assert type(loaded.backbone) is nn.Linear
    assert loaded.backbone.weight.dtype == getattr(torch, dtype)
    assert loaded.head.weight.dtype == loaded.head.bias.dtype == torch.float32
    assert loaded.frame_id.dtype == torch.int64
    assert "action_scale" in loaded._non_persistent_buffers_set
    torch.testing.assert_close(loaded.head.weight, net.head.weight, atol=0, rtol=0)
    torch.testing.assert_close(loaded.head.bias, net.head.bias, atol=0, rtol=0)
    torch.testing.assert_close(loaded.action_scale, net.action_scale, atol=0, rtol=0)
    expected = net.backbone.weight.float()
    if mode.endswith("_lora"):
        expected = expected + net.backbone._lora_scale * (net.backbone.lora_B.weight @ net.backbone.lora_A.weight)
    torch.testing.assert_close(loaded.backbone.weight, expected.to(getattr(torch, dtype)), atol=0, rtol=0)
    # This is a controlled fixture bound, not an Alpamayo/GR00T quality tolerance.
    torch.testing.assert_close(loaded(data), net(data), atol=3e-4, rtol=0.02)
    assert manifest["resumable_training_checkpoint"] is False
    assert manifest["native_loader_required"] is True
    assert manifest["source_metadata"] == read_adapter_metadata(source)
    assert manifest["numerical_contract"] == {
        "base": "W_bf16",
        "merge": "effective_base + (alpha/rank) * B @ A" if mode.endswith("_lora") else "effective_base",
        "accumulation_dtype": "float32",
        "merge_device": "cpu",
        "merged_backbone_dtype": dtype,
        "other_parameters_and_buffers": "preserve_native_dtype",
        "requantized": False,
    }
    assert not any("lora_" in name for name in manifest["tensors"])
    state.validate_frozen_base()


def test_native_loader_roundtrips_in_a_fresh_interpreter(tmp_path):
    _, state, _, data, output, manifest = _export(tmp_path)
    expected = TaskNet(device="meta")
    load_family_dense(expected, output, device="cpu", expected_identity=state.identity)
    input_path, result_path = tmp_path / "input.pt", tmp_path / "result.pt"
    torch.save(data, input_path)
    script = """
import sys, torch
from cosmos_framework.posttrain.tests.test_checkpoint_family_dense import TaskNet
from cosmos_framework.posttrain.checkpoint.family_dense import read_family_dense, load_family_dense
manifest=read_family_dense(sys.argv[1])
net=TaskNet(device='meta')
load_family_dense(net,sys.argv[1],device='cpu',expected_identity={'base':manifest['source_metadata']['identity']['base']})
torch.save({'outputs':net(torch.load(sys.argv[2],weights_only=True)), 'state':net.state_dict()},sys.argv[3])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(output), str(input_path), str(result_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert result.returncode == 0, result.stderr
    actual = torch.load(result_path, weights_only=True)
    torch.testing.assert_close(actual["outputs"], expected(data), atol=0, rtol=0)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual["state"][name], value, atol=0, rtol=0)
    assert manifest["numerical_contract"]["merge_device"] == "cpu"


def test_mixed_dtype_packing_is_deterministic_and_oversized_tensor_is_alone():
    inventory = {
        "z": {"shape": [2], "dtype": "int64"},
        "a": {"shape": [3], "dtype": "bfloat16"},
        "b": {"shape": [20], "dtype": "float32"},
        "c": {"shape": [2], "dtype": "float32"},
    }
    plan = _mixed_shard_plan(inventory, 24)
    assert plan["a"].startswith("model-00001-")
    assert plan["b"].startswith("model-00002-")
    assert plan["c"] == plan["z"] and plan["c"].startswith("model-00003-")
    assert plan == _mixed_shard_plan(dict(reversed(list(inventory.items()))), 24)


def test_smaller_shards_change_packaging_without_changing_any_tensor(tmp_path):
    net, state, source, _, output, initial = _export(tmp_path, max_bytes=40)
    other = export_family_dense(
        net,
        checkpoint=source,
        output=str(tmp_path / "other"),
        merged_backbone_dtype="float32",
        exporter_build=BUILD,
        max_shard_size_bytes=128,
    )
    assert len(initial["files"]) > len(other["files"])
    for name, record in initial["tensors"].items():
        assert {k: v for k, v in record.items() if k != "file"} == {
            k: v for k, v in other["tensors"][name].items() if k != "file"
        }
    marker = (output / FAMILY_DENSE_MANIFEST).read_bytes()
    read_family_dense(output, expected_manifest_sha256=sha256(marker).hexdigest(), expected_identity=state.identity)
    with pytest.raises(ValueError, match="SHA256"):
        read_family_dense(output, expected_manifest_sha256="0" * 64)


def test_nested_native_sidecars_are_copied_bound_and_verified(tmp_path):
    source_dir = tmp_path / "metadata"
    source_dir.mkdir()
    config = source_dir / "config.json"
    config.write_text('{"family":"fixture"}\n')
    processor = source_dir / "processor.json"
    processor.write_text('{"action_scale":0.2}\n')
    _, _, _, _, output, manifest = _export(
        tmp_path, sidecars={"native/config.json": str(config), "processor/processor.json": str(processor)}
    )
    assert manifest["sidecars"] == ["native/config.json", "processor/processor.json"]
    assert (output / "processor/processor.json").read_bytes() == processor.read_bytes()
    (output / "processor/processor.json").write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        read_family_dense(output)


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "model.safetensors.index.json", "model-00001.safetensors", "a/../b"]
)
def test_sidecar_unsafe_or_reserved_paths_fail_before_output_creation(tmp_path, name):
    source = tmp_path / "config.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="sidecar"):
        _export(tmp_path, sidecars={name: str(source)})
    assert not (tmp_path / "dense").exists()


def test_interrupted_shard_write_never_publishes_or_reuses_output(tmp_path, monkeypatch):
    import safetensors.torch

    net, _, source, _ = _prepared(tmp_path)
    original = safetensors.torch.save_file
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected interrupted shard write")
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", interrupted)
    options = dict(
        checkpoint=source,
        output=str(tmp_path / "dense"),
        merged_backbone_dtype="float32",
        exporter_build=BUILD,
        max_shard_size_bytes=8,
    )
    with pytest.raises(OSError, match="interrupted"):
        export_family_dense(net, **options)
    assert not (tmp_path / "dense" / FAMILY_DENSE_MANIFEST).exists()
    with pytest.raises(ValueError, match="incomplete"):
        read_family_dense(tmp_path / "dense")
    with pytest.raises(FileExistsError):
        export_family_dense(net, **options)
    read_adapter_metadata(source)


@pytest.mark.parametrize("mismatch", ["identity", "shape", "missing_head", "buffer_persistence", "already_adapted"])
def test_native_dense_loader_refuses_incompatible_architecture_before_mutation(tmp_path, mismatch):
    net, state, _, _, output, _ = _export(tmp_path)
    fresh = TaskNet(device="meta")
    identity = copy.deepcopy(state.identity)
    if mismatch == "identity":
        identity["base"]["revision"] = "b" * 40
    elif mismatch == "shape":
        fresh.head = nn.Linear(4, 3, device="meta")
    elif mismatch == "missing_head":
        del fresh.head
    elif mismatch == "buffer_persistence":
        fresh._non_persistent_buffers_set.remove("action_scale")
    else:
        fresh = net
    with pytest.raises(ValueError):
        load_family_dense(fresh, output, device="cpu", expected_identity=identity)
    if fresh is not net:
        assert all(value.is_meta for value in fresh.parameters())


@pytest.mark.parametrize(
    "change",
    ["numerics", "missing_aux", "changed_head_digest", "buffer_dtype", "index_size", "extra_file", "shard_bytes"],
)
def test_reader_rejects_incomplete_inconsistent_or_modified_artifacts(tmp_path, change):
    _, _, _, _, output, manifest = _export(tmp_path)
    if change == "numerics":
        manifest["numerical_contract"]["base"] = "original_unquantized_W"
    elif change == "missing_aux":
        del manifest["trained_tensors"]["head.bias"]
    elif change == "changed_head_digest":
        manifest["tensors"]["head.weight"]["sha256"] = "0" * 64
    elif change == "buffer_dtype":
        manifest["tensors"]["frame_id"]["dtype"] = "float32"
    elif change == "index_size":
        index = output / "model.safetensors.index.json"
        value = json.loads(index.read_text())
        value["metadata"]["total_size"] += 1
        index.write_text(json.dumps(value))
        data = index.read_bytes()
        manifest["files"][index.name] = {"size": len(data), "sha256": sha256(data).hexdigest()}
    elif change == "extra_file":
        (output / "unexpected").write_text("extra")
    else:
        shard = output / next(iter(manifest["tensors"].values()))["file"]
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
    (output / FAMILY_DENSE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_family_dense(output)


def test_native_loader_requires_explicit_expected_identity(tmp_path):
    _, _, _, _, output, _ = _export(tmp_path)
    with pytest.raises(ValueError, match="explicit expected identity"):
        load_family_dense(TaskNet(device="meta"), output, device="cpu", expected_identity={})


def test_export_proves_unchanged_base_and_preserved_task_parameters(tmp_path):
    net, state, source, _ = _prepared(tmp_path)
    before = {name: tensor_digest(value) for name, value in net.named_parameters()}
    export_family_dense(
        net, checkpoint=source, output=str(tmp_path / "dense"), merged_backbone_dtype="float32", exporter_build=BUILD
    )
    assert before == {name: tensor_digest(value) for name, value in net.named_parameters()}
    state.validate_frozen_base()


@pytest.mark.parametrize("mode", ["bf16_lora", "bf16_frozen"])
def test_frozen_embedding_output_alias_exports_both_keys_and_restores_one_parameter(tmp_path, mode):
    net, state, source, data = _prepared(tmp_path, mode=mode, net=AliasedTaskNet())
    output = tmp_path / "dense"
    manifest = export_family_dense(
        net,
        checkpoint=source,
        output=str(output),
        merged_backbone_dtype="bfloat16",
        exporter_build=BUILD,
        max_shard_size_bytes=64,
    )
    assert manifest["parameter_aliases"] == [["embedding.weight", "output.weight"]]
    assert manifest["tensors"]["embedding.weight"]["file"] != manifest["tensors"]["output.weight"]["file"]
    for key in ("shape", "dtype", "sha256"):
        assert manifest["tensors"]["embedding.weight"][key] == manifest["tensors"]["output.weight"][key]
    restored = AliasedTaskNet(device="meta")
    load_family_dense(restored, output, device="cpu", expected_identity=state.identity)
    assert restored.embedding.weight is restored.output.weight
    assert not restored.embedding.weight.requires_grad
    torch.testing.assert_close(restored.embedding.weight, net.embedding.weight, atol=0, rtol=0)
    torch.testing.assert_close(restored(data), net(data), atol=3e-4, rtol=0.02)
    assert net.embedding.weight is net.output.weight
    state.validate_frozen_base()


def test_alias_ownership_is_required_on_both_exported_and_native_architecture_sides(tmp_path):
    net, state, source, _ = _prepared(tmp_path, net=AliasedTaskNet())
    output = tmp_path / "dense"
    export_family_dense(
        net, checkpoint=source, output=str(output), merged_backbone_dtype="float32", exporter_build=BUILD
    )
    untied = AliasedTaskNet(device="meta", tied=False)
    with pytest.raises(ValueError, match="alias ownership"):
        load_family_dense(untied, output, device="cpu", expected_identity=state.identity)
    assert all(parameter.is_meta for parameter in untied.parameters())
    marker = output / FAMILY_DENSE_MANIFEST
    manifest = json.loads(marker.read_text())
    manifest["parameter_aliases"] = []
    marker.write_text(json.dumps(manifest))
    tied = AliasedTaskNet(device="meta")
    with pytest.raises(ValueError, match="alias ownership"):
        load_family_dense(tied, output, device="cpu", expected_identity=state.identity)
    assert all(parameter.is_meta for parameter in tied.parameters())


@pytest.mark.parametrize(
    "groups",
    [
        [["output.weight", "embedding.weight"]],
        [["embedding.weight"]],
        [["embedding.weight", "output.weight"], ["embedding.weight", "output.weight"]],
        [["head.bias", "head.weight"]],
        [["backbone.weight", "embedding.weight"]],
        [["action_scale", "embedding.weight"]],
        [["embedding.weight", "missing.weight"]],
    ],
)
def test_malformed_target_or_auxiliary_alias_metadata_is_rejected(tmp_path, groups):
    _, _, _, _, output, manifest = _export(tmp_path)
    manifest["parameter_aliases"] = groups
    (output / FAMILY_DENSE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="alias"):
        read_family_dense(output)


def test_aliased_tensor_cannot_have_different_values_even_with_self_consistent_shard_hashes(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    net, _, source, _ = _prepared(tmp_path, net=AliasedTaskNet())
    output = tmp_path / "dense"
    manifest = export_family_dense(
        net,
        checkpoint=source,
        output=str(output),
        merged_backbone_dtype="bfloat16",
        exporter_build=BUILD,
        max_shard_size_bytes=8,
    )
    name = "output.weight"
    record = manifest["tensors"][name]
    path = output / record["file"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        value = handle.get_tensor(name).clone()
    value.add_(1)
    save_file({name: value}, str(path), metadata={"format": "pt"})
    record.update(tensor_digest(value))
    payload = path.read_bytes()
    manifest["files"][path.name] = {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
    (output / FAMILY_DENSE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="aliased parameters differ"):
        read_family_dense(output)


def test_aliased_native_dense_policy_reloads_in_fresh_process(tmp_path):
    net, _, source, data = _prepared(tmp_path, net=AliasedTaskNet())
    output = tmp_path / "dense"
    export_family_dense(
        net, checkpoint=source, output=str(output), merged_backbone_dtype="float32", exporter_build=BUILD
    )
    inputs, result = tmp_path / "input.pt", tmp_path / "result.pt"
    torch.save(data, inputs)
    script = """
import sys,torch
from cosmos_framework.posttrain.tests.test_checkpoint_family_dense import AliasedTaskNet
from cosmos_framework.posttrain.checkpoint.family_dense import read_family_dense,load_family_dense
manifest=read_family_dense(sys.argv[1]); net=AliasedTaskNet(device='meta')
load_family_dense(net,sys.argv[1],device='cpu',expected_identity={'base':manifest['source_metadata']['identity']['base']})
assert net.embedding.weight is net.output.weight
torch.save(net(torch.load(sys.argv[2],weights_only=True)),sys.argv[3])
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(output), str(inputs), str(result)], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
    torch.testing.assert_close(torch.load(result, weights_only=True), net(data), atol=3e-4, rtol=0.02)
