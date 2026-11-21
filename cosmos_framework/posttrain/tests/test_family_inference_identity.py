# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real codec/portable IO contracts; tiny CPU tensors do not qualify policy numerics."""

import builtins
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import _require_subset, load_adapter_weights, save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.family_dense import export_family_dense, export_family_unmerged
from cosmos_framework.posttrain.checkpoint.family_inference import (
    _CURRENT_GR00T,
    _CURRENT_SHARED,
    _OLD_GR00T,
    _OLD_SHARED,
    INFERENCE_COMPATIBILITY,
    family_inference_identity_for_read,
    gr00t_unmerged_inference_identity_for_read,
)
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec


def identities(family, mode="nf4_lora", seed=17):
    """Selected native settings; non-source tensor/provenance fields are small test stand-ins."""
    adapter = {"mode": mode, "rank": 32, "alpha": 64.0, "dropout": 0.0, "seed": seed}
    task = {
        "family": "alpamayo-r1-trajectory-only-v1",
        "activation_checkpointing": "full",
        "inference_steps": 10,
        "integrator": "euler",
        "loss": "native-flow-matching",
        "reasoning_supervision": False,
        "train_action_head": True,
    }
    if family == "gr00t":
        task = {
            "family": "gr00t_n1d7",
            "implementation": copy.deepcopy(_OLD_GR00T),
            "config": {
                "action_dim": 7,
                "backbone_adaptation": {
                    **adapter,
                    "profile": "gr00t_n1d7_reason2_2b_layer16_v1",
                    "train_action_head": True,
                },
                "camera_map": {"image": "image", "wrist_image": "wrist_image"},
                "denoising_steps": 4,
                "embodiment_tag": "libero_sim",
                "execution_horizon": 8,
                "language_key": "annotation.language.language_instruction",
                "model_action_horizon": 40,
                "prompt_key": "prompt",
                "state_map": {key: key for key in ("gripper", "pitch", "roll", "x", "y", "yaw", "z")},
                "training_prefix": 10,
            },
            "processor_config": {"fixture": "native-inputs"},
            "statistics": {"fixture": "native-actions"},
            "precision": {"compute": "bfloat16-autocast-including-adapter-gemm"},
        }
    old = {
        "format": "cosmos.family-backbone.v1",
        "implementation": copy.deepcopy(_OLD_SHARED),
        "task_contract": task,
        "adapter": adapter,
        "base": {"revision": "immutable-fixture", "sha256": "a" * 64},
        "processor_contract": {"fixture": "immutable-processor"},
        "torch_version": "2.10.0+cu128",
        "dependencies": {"transformers": "4.57.6", "safetensors": "0.8.0"},
        "precision": {"target_source_cast": "bfloat16", "adapter": "float32", "auxiliary": "float32"},
        "quantization": {"fixture": "nf4"} if mode == "nf4_lora" else None,
        "targets": {"backbone.projection": [2, 2]},
        "auxiliary_parameters": ["action_head.weight"],
        "frozen_tensors": {"fixture": "frozen-state"},
        "quantized_tensors": {"fixture": "quantization-state"},
        "training_contract": {"steps": 512 if family == "alpamayo" else 200},
    }
    current = copy.deepcopy(old)
    current["implementation"] = copy.deepcopy(_CURRENT_SHARED)
    if family == "gr00t":
        current["task_contract"]["implementation"] = copy.deepcopy(_CURRENT_GR00T)
    return old, current


def codec(identity):
    net = nn.Linear(2, 2, bias=False)
    net._cosmos_adapter_parameters = {"action_head.weight": net.weight}
    result = AdapterCheckpointCodec(net, identity=identity)
    net._cosmos_adapter_checkpoint_codec = result
    return net, result


@pytest.mark.parametrize("family", ["alpamayo", "gr00t"])
@pytest.mark.parametrize("mode", ["nf4_lora", "bf16_lora"])
@pytest.mark.parametrize("seed", [17, 42])
def test_reviewed_inference_changes_only_source_comparison_and_never_resume_or_export(family, mode, seed):
    old, current = identities(family, mode, seed)
    old_net, saved = codec(old)
    net, live = codec(current)
    metadata = saved.metadata()
    before_metadata, before_identity = copy.deepcopy(metadata), copy.deepcopy(live.identity)
    assert live.validate_metadata(metadata, purpose="inference") == INFERENCE_COMPATIBILITY
    assert metadata == before_metadata and live.identity == before_identity
    with pytest.raises(ValueError, match="identity differs"):
        live.validate_metadata(metadata)
    optimizer = torch.optim.AdamW(net.parameters())
    with pytest.raises(ValueError, match="identity differs"):
        live.validate_metadata(metadata, optimizer, purpose="resume")
    with pytest.raises(ValueError, match="cannot validate an optimizer"):
        live.validate_metadata(metadata, optimizer, purpose="inference")
    assert old_net.weight.shape == net.weight.shape


@pytest.mark.parametrize("family", ["alpamayo", "gr00t"])
@pytest.mark.parametrize(
    "path,value",
    [
        (("implementation", "family"), "unknown-old-source"),
        (("implementation", "adapter"), "unknown-adapter-arithmetic"),
        (("base", "revision"), "wrong-base"),
        (("processor_contract", "fixture"), "wrong-processor"),
        (("torch_version",), "2.11.0"),
        (("dependencies", "transformers"), "5.0.0"),
        (("precision", "adapter"), "bfloat16"),
        (("quantization",), {"fixture": "wrong-storage"}),
        (("adapter", "seed"), 42),
        (("adapter", "rank"), 64),
        (("adapter", "dropout"), 0.05),
        (("head_adapters",), {"profile": "new-head-policy"}),
        (("auxiliary_parameters",), []),
        (("targets", "backbone.projection"), [4, 2]),
        (("frozen_tensors", "fixture"), "changed-base"),
        (("quantized_tensors", "fixture"), "changed-quantization"),
    ],
)
def test_inference_keeps_every_non_source_identity_check(family, path, value):
    old, current = identities(family)
    target = old
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    old_net, saved = codec(old)
    net, live = codec(current)
    with pytest.raises(ValueError, match="identity differs"):
        live.validate_metadata(saved.metadata(), purpose="inference")
    assert old_net.weight.shape == net.weight.shape


@pytest.mark.parametrize("side", ["old", "current"])
@pytest.mark.parametrize("change", ["source", "profile", "train_head", "prefix", "steps", "seed", "head"])
def test_gr00t_unknown_task_closures_and_unreviewed_settings_fail(side, change):
    old, current = identities("gr00t")
    item = old if side == "old" else current
    task = item["task_contract"]
    if change == "source":
        task["implementation"]["policy.py"] = "unknown"
    elif change == "head":
        task["config"]["backbone_adaptation"]["action_head_adapter_profile"] = "new"
    elif change in ("profile", "train_head", "seed"):
        key = "train_action_head" if change == "train_head" else change
        task["config"]["backbone_adaptation"][key] = "changed"
    else:
        task["config"]["training_prefix" if change == "prefix" else "denoising_steps"] += 1
    old_net, saved = codec(old)
    net, live = codec(current)
    with pytest.raises(ValueError, match="identity differs"):
        live.validate_metadata(saved.metadata(), purpose="inference")
    assert old_net.weight.shape == net.weight.shape


def test_matching_current_identities_keep_existing_strict_resume_and_reload():
    _, current = identities("gr00t")
    net, live = codec(current)
    optimizer = torch.optim.AdamW(net.parameters())
    metadata = live.metadata(optimizer)
    assert live.validate_metadata(metadata) is None
    assert live.validate_metadata(metadata, purpose="inference") is None
    assert live.validate_metadata(metadata, optimizer, purpose="resume") is None
    metadata["parameters"]["action_head.weight"]["shape"] = [4, 4]
    with pytest.raises(ValueError, match="parameters differs"):
        live.validate_metadata(metadata, purpose="inference")


@pytest.mark.parametrize("exporter", [export_family_dense, export_family_unmerged])
def test_real_portable_inference_load_does_not_enable_legacy_reexport(tmp_path, exporter):
    old, current = identities("gr00t", "bf16_lora")
    source_net, saved = codec(old)
    with torch.no_grad():
        source_net.weight.fill_(3.25)
    manager = CheckpointManager(checkpoint_format="dcp")
    artifact = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000200"), saved.parameters, saved.metadata(), manager=manager
    )
    net, live = codec(current)
    assert load_adapter_weights(artifact, live, manager=manager, purpose="inference") == INFERENCE_COMPATIBILITY
    torch.testing.assert_close(net.weight, source_net.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity differs"):
        load_adapter_weights(artifact, live, manager=manager)
    with pytest.raises(ValueError, match="identity differs"):
        exporter(
            net,
            checkpoint=artifact,
            output=str(tmp_path / "must-not-publish"),
            exporter_build={"code_revision": "a" * 40, "wheel_sha256": "b" * 64},
            **({"merged_backbone_dtype": "bfloat16"} if exporter is export_family_dense else {}),
        )
    assert not (tmp_path / "must-not-publish").exists()


def test_unmerged_subset_uses_same_source_pair_without_weakening_task_or_runtime():
    old, current = identities("gr00t")
    keys = ("base", "task_contract", "adapter", "torch_version", "dependencies")
    expected = {key: copy.deepcopy(current[key]) for key in keys}
    adjusted = gr00t_unmerged_inference_identity_for_read(old, expected, implementation=_CURRENT_SHARED)
    assert adjusted is not expected and expected["task_contract"]["implementation"] == _CURRENT_GR00T
    _require_subset(old, adjusted)
    for path in (("base", "revision"), ("task_contract", "processor_config"), ("torch_version",)):
        changed = copy.deepcopy(old)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "wrong"
        comparison = gr00t_unmerged_inference_identity_for_read(changed, expected, implementation=_CURRENT_SHARED)
        with pytest.raises(ValueError):
            _require_subset(changed, comparison)
    assert gr00t_unmerged_inference_identity_for_read(old, expected, implementation=_OLD_SHARED) is expected
    alpa, _ = identities("alpamayo")
    assert gr00t_unmerged_inference_identity_for_read(alpa, expected, implementation=_CURRENT_SHARED) is expected


def test_reviewed_source_constants_and_unmerged_comparison_need_no_bitsandbytes(monkeypatch):
    original_import = builtins.__import__

    def without_bnb(name, *args, **kwargs):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            pytest.fail("unmerged identity checking imported bitsandbytes")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_bnb)
    from cosmos_framework.posttrain.model.quantization.family import _implementation_identity

    assert _implementation_identity() == _CURRENT_SHARED
    root = Path(__file__).parents[1] / "model/gr00t"
    assert {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _CURRENT_GR00T} == _CURRENT_GR00T
    old, current = identities("gr00t")
    expected = {key: current[key] for key in ("base", "task_contract", "adapter", "torch_version", "dependencies")}
    _require_subset(
        old, gr00t_unmerged_inference_identity_for_read(old, expected, implementation=_implementation_identity())
    )


def test_incomplete_identity_is_not_normalized():
    old, current = identities("gr00t")
    for field in ("implementation", "adapter", "task_contract"):
        changed = copy.deepcopy(old)
        del changed[field]
        assert family_inference_identity_for_read(changed, current) is current


def test_rollout_loader_explicitly_allows_reviewed_inference_without_rewriting_identity(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.entrypoints import action_sft
    from cosmos_framework.posttrain.entrypoints.gr00t_rollout import load_rollout_policy
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    old, current = identities("gr00t", "bf16_lora")
    old["base"] = current["base"] = recorded_source()
    old_net, saved = codec(old)
    net, live = codec(current)
    with torch.no_grad():
        old_net.weight.fill_(3.25)
    artifact = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000200"),
        saved.parameters,
        saved.metadata(),
        manager=CheckpointManager(checkpoint_format="dcp"),
    )
    monkeypatch.setattr(action_sft, "_factory", lambda path: lambda *args, **kwargs: SimpleNamespace(net=net))
    config = SimpleNamespace(policy_factory="cosmos_framework.posttrain.model.gr00t.action_sft.build_policy")
    loaded = load_rollout_policy(config, artifact, device="cpu", output=tmp_path / "loaded")
    assert loaded.inference_compatibility == INFERENCE_COMPATIBILITY
    assert live.identity == current
    torch.testing.assert_close(loaded.net.weight, old_net.weight, rtol=0, atol=0)


@pytest.mark.parametrize("unmerged", [False, True])
def test_native_evaluation_rejects_unqualified_legacy_backbone_reuse(tmp_path, monkeypatch, unmerged):
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.entrypoints import action_eval, action_sft
    from cosmos_framework.posttrain.utils import determinism, qlora_metrics
    from cosmos_framework.utils import misc

    old, current = identities("gr00t")
    old_net, saved = codec(old)
    net, live = codec(current)
    metadata = saved.metadata()
    config = SimpleNamespace(policy_factory="fixture")
    monkeypatch.setattr(
        action_sft, "validate_config", lambda _: SimpleNamespace(seed=17, reuse_gr00t_evaluation_backbone=True)
    )
    monkeypatch.setattr(action_sft, "_factory", lambda _: lambda *a, **k: SimpleNamespace(net=net))
    monkeypatch.setattr(adapter, "read_adapter_metadata", lambda *a, **k: metadata)
    monkeypatch.setattr(
        adapter, "load_adapter_weights", lambda *a, **k: pytest.fail("unqualified reuse loaded weights")
    )
    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", lambda *a, **k: None)
    monkeypatch.setattr(determinism, "enable_determinism", lambda **k: None)
    monkeypatch.setattr(misc, "set_random_seed", lambda **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="backbone reuse"):
        action_eval.run(config, checkpoint="fixture", output=str(tmp_path / "evaluation"), unmerged=unmerged)
    assert live.identity == current and old_net.weight.shape == net.weight.shape
