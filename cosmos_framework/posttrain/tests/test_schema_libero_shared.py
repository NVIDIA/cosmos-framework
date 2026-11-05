# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared initial-state inventories at the source, evaluation and resume boundaries."""

import json
import sys
from pathlib import Path

import pytest

from cosmos_framework.posttrain.control.validation.rollout import episode_metrics
from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
    LIBERO_PROTOCOLS,
    LIBERO_SUITES,
    LiberoManifest,
    fixed_manifest,
    load_partition,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _environment(directory: Path, suite: str, profile: str) -> LiberoEnvironment:
    directory.mkdir(exist_ok=True)
    for split in ("train", "eval"):
        fixed_manifest(split, suite, profile).write(directory / f"{split}.json")
    protocol = LIBERO_PROTOCOLS[suite]
    environment = LiberoEnvironment(
        LiberoConfig(
            suite=suite,
            execution_horizon=protocol.execution_horizon,
            max_episode_steps=protocol.max_episode_steps,
            train_manifest=str(directory / "train.json"),
            eval_manifest=str(directory / "eval.json"),
            eval_profile=profile,
            sim_python=sys.executable,
        )
    )
    environment.validate_configuration()
    return environment


@pytest.mark.parametrize("suite", LIBERO_SUITES)
def test_shared_inventory_and_episode_weighted_uneven_shards(tmp_path: Path, suite: str) -> None:
    environment = _environment(tmp_path, suite, "shared_500")
    train, evaluation = load_partition(str(tmp_path / "train.json"), str(tmp_path / "eval.json"))
    legacy = _environment(tmp_path / "legacy", suite, "legacy_100")
    assert environment.identity()["partition"] != legacy.identity()["partition"]
    expected = {(task, state) for task in range(10) for state in range(50)}
    for manifest in (train, evaluation):
        assert {(case.task_id, case.init_state_id) for case in manifest.cases} == expected
        assert len(manifest.cases) == len(expected)
        assert manifest.payload()["schema_version"] == 4
        assert manifest.payload()["profile"] == "shared_500"
    shards = [environment.evaluation_cases(rank, 8) for rank in range(8)]
    assert [len(shard) for shard in shards] == [63] * 4 + [62] * 4
    rows = [
        dict(case, success=rank < 4, steps=environment.execution_horizon, **{"return": float(rank)})
        for rank, shard in enumerate(shards)
        for case in shard
    ]
    metrics = episode_metrics(environment.validate_evaluation(rows))
    assert metrics["val/success_rate"] == sum(len(shard) for shard in shards[:4]) / len(expected)
    assert metrics["val/success_rate"] != sum(rank < 4 for rank in range(8)) / 8
    for task in range(10):
        task_rows = [row for row in rows if row["task_id"] == task]
        assert len(task_rows) == 50
        assert metrics[f"val/task_{task}/success_rate"] == sum(row["success"] for row in task_rows) / 50
    for invalid in (rows[:-1], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError, match="inventory"):
            environment.validate_evaluation(invalid)
    environment.config.eval_profile = "legacy_100"
    with pytest.raises(ValueError, match="configured suite/evaluation profile"):
        environment.validate_configuration()
    environment.config.eval_manifest = legacy.config.eval_manifest
    with pytest.raises(ValueError, match="shared_500"):
        environment.validate_configuration()


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "late_case"))
def test_shared_raw_manifest_rejects_invalid_input(tmp_path: Path, corruption: str) -> None:
    value = fixed_manifest("eval", "libero_10", "shared_500").payload()
    if corruption == "missing":
        value["cases"].pop()
    elif corruption == "duplicate":
        value["cases"][-1] = value["cases"][0]
    else:
        value["cases"][-1]["init_state_id"] = 50
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest|init_state_id"):
        LiberoManifest.read(path)
