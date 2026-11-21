# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import Experiment, assert_budget


def test_paired_arms_change_only_base_recipe_and_label():
    left = Experiment("nf4_control", "nf4", seed=17, microbatch=2)
    right = dataclasses.replace(left, name="bf16_control", arm="bf16")
    assert [x for x in left.overrides() if not x.startswith("job.name=")] == [
        x for x in right.overrides() if not x.startswith("job.name=")
    ]
    assert "rl.grad_accum=4" in left.overrides()
    assert "model.model_config.adapter.initialization_seed=17" in left.overrides()
    assert "rl.validation.seed=17" in left.overrides()


@pytest.mark.parametrize(
    "changes",
    [
        {"arm": "fp4"},
        {"rank": True},
        {"microbatch": 3},
        {"learning_rate": float("nan")},
        {"warmup_steps": 300},
        {"frames": 7},
        {"prefetch": 1},
        {"video_min_pixels": 999999},
        {"initialization": "unknown"},
        {"dropout": True},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
        {"initialization_artifact": "/unexpected/manifest.json"},
        {"initialization": "nf4_residual_svd"},
        {
            "initialization": "nf4_residual_svd",
            "initialization_artifact": "relative/manifest.json",
            "initialization_artifact_sha256": "a" * 64,
        },
    ],
)
def test_invalid_experiments_fail_before_launch(changes):
    with pytest.raises(ValueError):
        Experiment("test", "nf4", **changes) if "arm" not in changes else Experiment("test", **changes)


def test_budget_counts_failed_jobs_and_outstanding_reservations():
    records = [
        {"job_id": "failed", "terminal": True, "elapsed_gpu_hours": 2},
        {"job_id": "pending", "terminal": False, "reserved_gpu_hours": 4},
    ]
    assert assert_budget(records, next_gpu_hours=1) == 6
    with pytest.raises(ValueError, match="exceeded"):
        assert_budget(records, next_gpu_hours=43)
    with pytest.raises(ValueError, match="duplicate"):
        assert_budget(records + records, next_gpu_hours=0)


def test_residual_experiment_is_explicit_and_baseline_serialization_unchanged():
    baseline = Experiment("nf4_control", "nf4")
    assert not any(key.startswith("initialization") for key in baseline.as_dict())
    assert not any("initialization_artifact" in value for value in baseline.overrides())
    residual = dataclasses.replace(
        baseline,
        name="nf4_residual",
        initialization="nf4_residual_svd",
        initialization_artifact="/prepared/manifest.json",
        initialization_artifact_sha256="a" * 64,
    )
    assert residual.as_dict()["initialization"] == "nf4_residual_svd"
    assert 'model.model_config.adapter.initialization="nf4_residual_svd"' in residual.overrides()
    assert 'model.model_config.adapter.initialization_artifact="/prepared/manifest.json"' in residual.overrides()
    with pytest.raises(ValueError, match="requires NF4"):
        dataclasses.replace(residual, arm="bf16")


def test_dropout_recipe_is_explicit_and_equal_between_precision_arms(tmp_path):
    baseline = Experiment("nf4_control", "nf4")
    assert "dropout" not in baseline.as_dict()
    assert not any("dropout" in value for value in baseline.overrides())
    candidate = dataclasses.replace(baseline, dropout=0.05)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(candidate.as_dict()))
    assert Experiment.read(path) == candidate
    assert "model.model_config.adapter.dropout=0.05" in candidate.overrides()
    control = dataclasses.replace(candidate, arm="bf16")
    assert control.overrides() == candidate.overrides()
    assert control.as_dict()["dropout"] == 0.05


@pytest.mark.parametrize(
    "arm, historical_sha256",
    [
        ("nf4", "0ee56403d283eb9f4937ef66f9d3cd58a89e5ff06b753b36ff27670b231a0618"),
        ("bf16", "8cc5ec45849822735f2bda0b45c8b64e06178ac4faab0e147f38b77bd90fae79"),
    ],
)
def test_zero_smoothing_preserves_historical_spec_and_override_bytes(arm, historical_sha256):
    # Captured before adding the field: both serialized spec and ordered argv.
    for experiment in (
        Experiment("control", arm),
        Experiment("control", arm, wts_answer_smoothing=0),
        Experiment("control", arm, wts_answer_smoothing=-0.0),
    ):
        assert "wts_answer_smoothing" not in experiment.as_dict()
        assert not any("wts_answer_smoothing" in value for value in experiment.overrides())
        raw = json.dumps(
            {"spec": experiment.as_dict(), "overrides": experiment.overrides()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert hashlib.sha256(raw).hexdigest() == historical_sha256


@pytest.mark.parametrize("arm", ["nf4", "bf16"])
@pytest.mark.parametrize("epsilon", [0.02, 0.999])
def test_smoothing_roundtrip_changes_only_the_training_override(tmp_path, arm, epsilon):
    baseline = Experiment("control", arm)
    candidate = dataclasses.replace(baseline, wts_answer_smoothing=epsilon)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(candidate.as_dict()))
    assert Experiment.read(path) == candidate
    assert candidate.as_dict() == {**baseline.as_dict(), "wts_answer_smoothing": epsilon}
    assert candidate.overrides() == [*baseline.overrides(), f"rl.wts_answer_smoothing={epsilon}"]
    other = dataclasses.replace(candidate, arm="bf16" if arm == "nf4" else "nf4")
    assert other.overrides() == candidate.overrides()


@pytest.mark.parametrize(
    "epsilon", [True, False, -0.02, 1, 1.01, float("nan"), float("inf"), -float("inf"), "0.02", None]
)
def test_invalid_smoothing_refuses_before_launch(epsilon):
    with pytest.raises(ValueError, match="wts_answer_smoothing"):
        Experiment("control", "nf4", wts_answer_smoothing=epsilon)
