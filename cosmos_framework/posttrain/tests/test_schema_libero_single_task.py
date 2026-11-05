# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-task source inventory, public binding, and completed-episode coverage."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cosmos_framework.posttrain.control.validation.rollout import episode_metrics
from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LiberoManifest,
    fixed_manifest,
    load_partition,
)
from cosmos_framework.posttrain.tests.helpers.libero_analysis import completed_case_counts

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _environment(directory: Path, task_id: int | None, profile: str = "single_task_50") -> LiberoEnvironment:
    directory.mkdir(exist_ok=True)
    for split in ("train", "eval"):
        fixed_manifest(split, "libero_10", profile, task_id=task_id).write(directory / f"{split}.json")
    environment = LiberoEnvironment(
        LiberoConfig(
            suite="libero_10",
            execution_horizon=8,
            max_episode_steps=720,
            train_manifest=str(directory / "train.json"),
            eval_manifest=str(directory / "eval.json"),
            eval_profile=profile,
            sim_python=sys.executable,
        )
    )
    environment.validate_configuration()
    return environment


@pytest.mark.parametrize("task_id", (0, 9))
def test_single_task_roundtrip_and_public_uneven_evaluation(tmp_path: Path, task_id: int) -> None:
    environment = _environment(tmp_path, task_id)
    train, evaluation = load_partition(environment.config.train_manifest, environment.config.eval_manifest)
    other_task = _environment(tmp_path / "other", (task_id + 1) % 10)
    shared = _environment(tmp_path / "shared", None, "shared_500")
    assert environment.identity()["partition"] != other_task.identity()["partition"]
    assert environment.identity()["partition"] != shared.identity()["partition"]
    expected = {(task_id, state) for state in range(50)}
    for manifest in (train, evaluation):
        assert {(case.task_id, case.init_state_id) for case in manifest.cases} == expected
        assert manifest.task_id == task_id and manifest.payload()["schema_version"] == 5
        assert len(manifest.cases) == len(expected)
    shards = [environment.evaluation_cases(rank, 8) for rank in range(8)]
    assert [len(shard) for shard in shards] == [7, 7, 6, 6, 6, 6, 6, 6]
    rows = [
        dict(case, success=rank < 2, steps=environment.execution_horizon, **{"return": float(rank)})
        for rank, shard in enumerate(shards)
        for case in shard
    ]
    metrics = episode_metrics(environment.validate_evaluation(rows))
    assert metrics["val/success_rate"] == sum(map(len, shards[:2])) / len(expected)
    assert metrics["val/success_rate"] != sum(rank < 2 for rank in range(8)) / 8
    assert metrics[f"val/task_{task_id}/success_rate"] == metrics["val/success_rate"]
    for invalid in (rows[:-1], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError, match="inventory"):
            environment.validate_evaluation(invalid)
    environment.config.eval_profile = "shared_500"
    with pytest.raises(ValueError, match="configured suite/evaluation profile"):
        environment.validate_configuration()
    environment.config.eval_manifest = shared.config.eval_manifest
    with pytest.raises(ValueError, match="matching training and evaluation"):
        environment.validate_configuration()
    environment.config.eval_profile = other_task.config.eval_profile
    environment.config.eval_manifest = other_task.config.eval_manifest
    with pytest.raises(ValueError, match="matching task_id"):
        environment.validate_configuration()


@pytest.mark.parametrize("corruption", ("other_task", "task_id_wrong"))
def test_single_task_rejects_raw_manifest_corruption(tmp_path: Path, corruption: str) -> None:
    payload = fixed_manifest("eval", "libero_10", "single_task_50", task_id=9).payload()
    if corruption == "other_task":
        payload["cases"][-1]["task_id"] = 0
    else:
        payload["task_id"] = 0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest"):
        LiberoManifest.read(path)


def test_completed_coverage_discards_rollback_and_uncommitted_suffix() -> None:
    manifest = fixed_manifest("train", "libero_10", "single_task_50", task_id=9)
    first, abandoned, replacement, pending = manifest.cases[:4]
    rows = [
        (4, (first,)),
        (6, (first,)),
        (8, (abandoned, abandoned)),
        (12, (abandoned,)),
        (8, (replacement, replacement)),
        (12, (pending, pending)),
    ]
    counts = completed_case_counts(rows, manifest, committed_step=8)
    assert set(counts) == set(manifest.cases)
    assert counts[first] == counts[replacement] == 2
    assert counts[abandoned] == counts[pending] == 0
    assert not all(count >= 2 for count in counts.values())
    with pytest.raises(ValueError, match="outside its source"):
        completed_case_counts([(4, (replace(first, task_id=0),))], manifest, committed_step=4)
    with pytest.raises(ValueError, match="training manifest"):
        completed_case_counts([], replace(manifest, split="eval"), committed_step=0)
