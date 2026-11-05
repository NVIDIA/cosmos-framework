# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Calibrated PPO identity, baseline rejection, frozen-state and target diagnostic contracts."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import JointPPOTargetDiagnostic
from cosmos_framework.posttrain.algorithm.joint_ppo.targets import target_diagnostic_metrics
from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.libero_analysis import compare_baseline_episodes
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import GaussianPolicy
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_legacy_identity_and_explicit_buffer_mode_are_distinct():
    legacy = make_model_config("test-checkpoint", 8)
    payload = asdict(legacy)
    payload.pop("backbone_buffer_dtype")
    assert legacy.identity_hash() == canonical_json_sha256(payload, context="legacy config")
    restored = Gr00tN1d7ModelConfig(**payload)
    assert restored.identity() == legacy.identity()
    changed = make_model_config("test-checkpoint", 8, backbone_buffer_dtype="bfloat16")
    assert changed.identity_hash() != restored.identity_hash()
    assert changed.identity()["backbone_buffer_dtype"] == "bfloat16"


def test_spatial_baseline_keeps_legacy_aggregate_checks(tmp_path):
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest
    from cosmos_framework.posttrain.tests.helpers.libero_analysis import load_baseline_reference

    manifest = fixed_manifest("eval")
    path = tmp_path / "eval.json"
    manifest.write(path)
    rows = [
        {
            "task_id": case.task_id,
            "init_state_id": case.init_state_id,
            "success": case.init_state_id % 2 == 0,
            "steps": 256,
            "return": float(case.init_state_id % 2 == 0),
        }
        for case in manifest.cases
    ]
    reference = tmp_path / "baseline.json"
    reference.write_text(
        json.dumps(
            {
                "optimizer_step": 0,
                "publication_version": 0,
                "manifest": manifest.digest,
                "sampling_mode": "ode",
                "episodes": rows,
            }
        )
    )
    recipe = LiberoConfig(eval_manifest=str(path))
    loaded = load_baseline_reference(recipe, str(reference))
    assert compare_baseline_episodes(loaded, list(reversed(rows)), include_control_rewards=False)["exact"]
    changed = [dict(row) for row in rows]
    changed[-1]["steps"] -= 16
    mismatch = compare_baseline_episodes(loaded, changed, include_control_rewards=False)
    assert not mismatch["exact"] and mismatch["mismatches"][0]["fields"] == ["steps"]
    changed = [dict(row) for row in rows]
    for row in changed[:2]:
        row["success"], row["return"] = not row["success"], 1.0 - row["return"]
    assert sum(row["success"] for row in rows) == sum(row["success"] for row in changed)
    assert len(compare_baseline_episodes(loaded, changed, include_control_rewards=False)["mismatches"]) == 2


def test_baseline_reference_rejects_corrupt_feedback_before_inference(tmp_path):
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_RUNTIME, fixed_manifest
    from cosmos_framework.posttrain.model.gr00t.ppo import NATIVE_ODE_REVISION
    from cosmos_framework.posttrain.tests.helpers.libero_analysis import load_baseline_reference

    manifest = fixed_manifest("eval", "libero_10", "aligned_200")
    manifest.write(tmp_path / "evaluation.json")
    reference = tmp_path / "baseline.json"
    recipe = LiberoConfig(
        suite="libero_10",
        execution_horizon=8,
        max_episode_steps=720,
        eval_profile="aligned_200",
        eval_manifest=str(tmp_path / "evaluation.json"),
    )
    rows = [
        {**asdict(case), "success": True, "steps": 1, "return": 1.0, "control_rewards": [1.0]}
        for case in manifest.cases
    ]
    payload = {
        "optimizer_step": 0,
        "publication_version": 0,
        "suite": recipe.suite,
        "manifest": manifest.digest,
        "sampling_mode": "native_ode",
        "ode_revision": NATIVE_ODE_REVISION,
        "environment": LIBERO_RUNTIME.identity(),
        "model_source": {"checkpoint": "test"},
        "episodes": rows,
    }
    reference.write_text(json.dumps(payload))
    assert (
        load_baseline_reference(
            recipe, str(reference), sampling_mode="native_ode", model_source=payload["model_source"]
        )
        == rows
    )
    with pytest.raises(ValueError, match="model_source"):
        load_baseline_reference(
            recipe, str(reference), sampling_mode="native_ode", model_source={"checkpoint": "other"}
        )
    rows[-1]["control_rewards"] = []
    reference.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent raw feedback"):
        load_baseline_reference(recipe, str(reference), sampling_mode="native_ode")


def test_constant_targets_report_explained_variance_as_undefined():
    records = [
        JointPPOTargetDiagnostic(str(i), float(i), 0.0, 0.0, 0.0, -float(i), 0.0, False, False, True, 16)
        for i in range(3)
    ]
    metrics = target_diagnostic_metrics(records)
    assert metrics["train/value_explained_variance_defined"] == 0
    assert "train/value_explained_variance" not in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_frozen_backbone_mutation_is_detected():
    model = GaussianPolicy()
    model.config = replace(model.config, backbone_buffer_dtype="bfloat16")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model.backbone = torch.nn.Linear(2, 2, dtype=torch.bfloat16).requires_grad_(False)
    learner = JointPPOTrainer(model, JointPPOConfig())
    learner.evaluation_state()
    with torch.no_grad():
        model.backbone.weight.add_(1)
    with pytest.raises(RuntimeError, match="frozen model"):
        learner.evaluation_state()
