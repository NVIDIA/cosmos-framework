# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU comparison contracts; only actual native runs qualify execution parity."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.posttrain.config.model import native_adapter_execution_policy

spec = importlib.util.spec_from_file_location("ac_observation_lifecycle", Path(__file__).with_name("lifecycle.py"))
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


def metadata_pair():
    result = []
    for mode in ("full", "none"):
        result.append(
            {
                "identity": {
                    "strategy": {
                        "activation_checkpointing": "full_nonreentrant" if mode == "full" else "none",
                        "kind": "single_device",
                    },
                    "execution_policy": native_adapter_execution_policy(
                        SimpleNamespace(activation_checkpointing=mode), include_defaults=True
                    ),
                    "base": {"revision": "frozen-source"},
                    "quantized_tensors": {"language.weight": "frozen-quant-state"},
                    "training_contract": {"code_revision": "same-build", "grad_accum": 8},
                },
                "parameters": ["language.lora_A.weight", "language.lora_B.weight"],
                "optimizer": {"type": "AdamW", "groups": ["language.lora_A.weight", "language.lora_B.weight"]},
            }
        )
    return result


def test_only_explicit_execution_delta_is_acknowledged_without_mutation():
    values = metadata_pair()
    original = copy.deepcopy(values)
    result = lifecycle.activation_checkpointing_metadata_delta(*values)
    assert values == original
    assert len(result["acknowledged_execution_differences"]) == 3
    assert result["reference_metadata_sha256"] != result["candidate_metadata_sha256"]


@pytest.mark.parametrize(
    "damage", ["base", "frozen", "build", "accumulation", "optimizer", "membership", "dtype", "mode"]
)
def test_other_model_or_training_differences_are_not_normalized(damage):
    left, right = metadata_pair()
    if damage == "base":
        right["identity"]["base"]["revision"] = "other-source"
    elif damage == "frozen":
        right["identity"]["quantized_tensors"]["language.weight"] = "changed-state"
    elif damage == "build":
        right["identity"]["training_contract"]["code_revision"] = "other-build"
    elif damage == "accumulation":
        right["identity"]["training_contract"]["grad_accum"] = 4
    elif damage == "optimizer":
        right["optimizer"]["type"] = "AdamW8bit"
    elif damage == "membership":
        right["optimizer"]["groups"].append("language.weight")
    elif damage == "dtype":
        right["identity"]["execution_policy"]["adapter_compute_dtype"] = "bfloat16"
    else:
        right["identity"]["execution_policy"]["activation_checkpointing"]["mode"] = "selective"
    with pytest.raises(ValueError):
        lifecycle.activation_checkpointing_metadata_delta(left, right)


def test_missing_or_identical_execution_identity_requires_another_comparison():
    left, right = metadata_pair()
    with pytest.raises(ValueError, match="full-to-none"):
        lifecycle.activation_checkpointing_metadata_delta(left, left)
    del right["identity"]["execution_policy"]
    with pytest.raises(ValueError, match="complete explicit"):
        lifecycle.activation_checkpointing_metadata_delta(left, right)


@pytest.mark.parametrize("damage", [None, "model", "optim", "rng", "cursor", "metadata"])
def test_real_cpu_payloads_use_exact_native_values_and_controls(tmp_path, monkeypatch, damage):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.tests import qlora_integration as integration

    paths = [tmp_path / name for name in ("full", "none")]
    metadata = metadata_pair()
    inspected = []
    monkeypatch.setattr(
        integration,
        "inspect_checkpoint",
        lambda path, *, expected_step: inspected.append((path, expected_step)),
    )
    monkeypatch.setattr(
        adapter,
        "read_adapter_metadata",
        lambda path, *, require_resume: copy.deepcopy(metadata[paths.index(Path(path))]),
    )
    for index, path in enumerate(paths):
        path.mkdir()
        for component in ("model", "optim"):
            value = torch.tensor([1.0, 2.0])
            if index and damage == component:
                value[0] += 0.0001
            dcp.save({"adapter" if component == "model" else "exp_avg": value}, checkpoint_id=path / component)
        torch.save(
            {"optimizer_step": 3, "rng": torch.tensor([1, 2 if not index or damage != "rng" else 3])},
            path / "trainer.pt",
        )
        torch.save({"cursor": 24 if not index or damage != "cursor" else 25}, path / "data.pt")
    if damage == "metadata":
        metadata[1]["identity"]["base"]["revision"] = "other-source"
    if damage is None:
        result = lifecycle.compare_activation_checkpointing_checkpoints(*paths, step=3)
        assert result["passed"] and result["observation_only"] and result["controls_rng_and_data_exact"]
        assert result["resume_compatible"] is False
        assert result["literal_tensor_bytes_or_serialized_dcp_equality_claimed"] is False
        with pytest.raises(AssertionError, match="metadata"):
            integration.compare_checkpoints(SimpleNamespace(reference=paths[0], candidate=paths[1], atol=0, rtol=0))
    else:
        with pytest.raises((AssertionError, ValueError)):
            lifecycle.compare_activation_checkpointing_checkpoints(*paths, step=3)
    assert inspected == [(str(path), 3) for path in paths]
