# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dense scoring control flow; tiny CPU tensors do not qualify GPU reload."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
from tools.native_qlora_qualification import score
from tools.native_qlora_qualification.test_score_preprocessing import execution as execution


@pytest.mark.parametrize(
    "path,pin,checkpoint",
    [("dense", None, None), (None, "a" * 64, None), ("dense", "a" * 64, "adapter"), ("dense", "A" * 64, None)],
)
def test_dense_mode_requires_external_manifest_pin_and_excludes_adapter(path, pin, checkpoint):
    with pytest.raises(ValueError, match="dense"):
        score.run(SimpleNamespace(dense_export=path, dense_export_sha256=pin, checkpoint=checkpoint))


def manifest_for(net):
    return {
        "tensors": {
            name: {**tensor_digest(value), "file": f"{name}.safetensors"} for name, value in net.named_parameters()
        }
    }


def test_dense_verification_checks_real_loaded_bytes_membership_and_trainability():
    net = torch.nn.Linear(3, 2).requires_grad_(False)
    manifest = manifest_for(net)
    score.verify_dense_parameters(net, manifest)
    net.weight[0, 0] += 1
    with pytest.raises(ValueError, match="exported bytes"):
        score.verify_dense_parameters(net, manifest)
    manifest = manifest_for(net)
    net.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="distinct frozen"):
        score.verify_dense_parameters(net, manifest)
    net.requires_grad_(False)
    net.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    with pytest.raises(ValueError, match="membership"):
        score.verify_dense_parameters(net, manifest)


def test_dense_verification_rejects_aliases_and_dtype_changes():
    net = torch.nn.Linear(3, 2).requires_grad_(False)
    manifest = manifest_for(net)
    net.double()
    with pytest.raises(ValueError, match="exported bytes"):
        score.verify_dense_parameters(net, manifest)
    net.register_parameter("alias", net.weight)
    with pytest.raises(ValueError, match="distinct frozen"):
        score.verify_dense_parameters(net, manifest)


@pytest.mark.parametrize("failure", [None, "base", "loaded_weight", "artifact_changed", "forward"])
def test_dense_uses_existing_hard_gold_scorer_and_cleans_up(execution, monkeypatch, failure):
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.checkpoint import dense_adapter
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.entrypoints import sft as entrypoint
    from cosmos_framework.posttrain.model import qwen3_vl
    from cosmos_framework.posttrain.utils import adapter_identity
    from cosmos_framework.utils import config as config_loader

    baseline, expected = execution.run("adapter-reference", 100_000)
    config = config_loader.load_config("fixture")
    policy = build.build_backbone(config)
    net = policy.net
    manifest = {
        **manifest_for(net),
        "base_identity": dict(config.model.model_config.source_identity),
        "output_dtype": "float32",
        "adapter_tensors": {},
        "source_metadata": {"identity": {"fixture": "CPU source"}},
    }
    if failure == "base":
        manifest["base_identity"]["revision"] = "wrong"
    export = execution.root / "dense"
    export.mkdir()
    loads, reads = [], []

    def read(path, *, expected_manifest_sha256):
        assert path == str(export) and expected_manifest_sha256 == "a" * 64
        reads.append(path)
        value = deepcopy(manifest)
        if failure == "artifact_changed" and len(reads) > 1:
            value["output_dtype"] = "bfloat16"
        return value

    def load(path, **kwargs):
        assert path == str(export)
        assert kwargs == {
            "dtype": torch.float32,
            "attn_implementation": "sdpa",
            "device_map": {"": 0},
            "local_files_only": True,
            "trust_remote_code": False,
        }
        loads.append(path)
        if failure == "loaded_weight":
            net.logits[0, 0, 0] += 1
        return net

    def processor(packer, *, export_path, expected_manifest_sha256):
        assert export_path == str(export) and expected_manifest_sha256 == "a" * 64
        return packer.validation_semantics_identity()

    def forbid(*_args, **_kwargs):
        raise AssertionError("dense scoring must not construct the original adapter model or optimizer")

    def wrap(value, family):
        assert value is net and family == "qwen3_vl"
        return policy

    monkeypatch.setattr(dense_adapter, "read_dense_export", read)
    monkeypatch.setattr(adapter_identity, "dense_adapter_processor_identity", processor)
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", load)
    monkeypatch.setattr(qwen3_vl, "Qwen3VLPolicy", wrap)
    monkeypatch.setattr(entrypoint, "sft_policy_source", forbid)
    monkeypatch.setattr(entrypoint, "validate_config", forbid)
    monkeypatch.setattr(build, "build_backbone", forbid)
    monkeypatch.setattr(build, "build_optimizer", forbid)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    execution.failure = "forward" if failure == "forward" else None
    output = execution.root / "dense-scoring"
    args = SimpleNamespace(
        data=str(execution.root / "rows.jsonl"),
        output=str(output),
        max_rows=3,
        seed=2026,
        config="fixture",
        override=[],
        arm="dense",
        checkpoint=None,
        dense_export=str(export),
        dense_export_sha256="a" * 64,
        preprocessing_cache_bytes=100_000,
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            score.run(args)
        assert not (output / "summary.json").exists()
        assert execution.events[-1] == "destroy"
        if failure == "base":
            assert execution.events.count("init") == 1
        assert not net._forward_hooks
        assert all(pipeline._closed for pipeline in execution.pipelines)
        return
    result = score.run(args)
    actual = json.loads((output / "summary.json").read_text())
    assert result["metrics"] == actual["metrics"]
    assert loads == [str(export)] and len(reads) == 2
    assert actual["adapter_state"] == "merged_effective_base"
    assert actual["quantized_matrices"] == 0
    assert actual["metrics"] == expected["metrics"]
    assert actual["processor_contract"] == expected["processor_contract"]
    assert actual["objective_identity"] == expected["objective_identity"]
    for name in ("scores.jsonl", "selected_rows.jsonl", "selection.json"):
        assert (output / name).read_bytes() == (baseline / name).read_bytes()
    assert json.loads((output / "summary.json").read_text())["complete"] is True
    assert not net._forward_hooks
    assert execution.events[-1] == "destroy"


@pytest.mark.parametrize("field", ["builder", "dp_shard", "dp_replicate", "grid_backend", "grid_grouping", "base"])
def test_dense_recipe_contract_rejects_unsupported_metadata_without_reading_source(field):
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config

    config = make_config()
    config.model.model_id = "/missing/original/base"
    manifest = {"base_identity": dict(config.model.model_config.source_identity)}
    assert score.dense_scoring_config(config, manifest).deterministic
    if field == "builder":
        config.model.builder = "other.unsupported.builder"
    elif field in {"dp_shard", "dp_replicate"}:
        setattr(config.model.parallelism, field, 2)
    elif field == "grid_backend":
        config.grid.backend = "ray"
    elif field == "grid_grouping":
        config.grid.grouping = "per_role"
    else:
        manifest["base_identity"]["revision"] = "wrong"
    with pytest.raises(ValueError, match="dense"):
        score.dense_scoring_config(config, manifest)
