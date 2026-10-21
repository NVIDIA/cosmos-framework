# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The small qualification run preserves production manifests and checkpoint identities."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.posttrain.embodied.sims.libero.protocol import load_partition
from cosmos_framework.posttrain.entrypoints.joint_ppo import validation_identity
from cosmos_framework.posttrain.tests.pi05_integration_cli import ensure_manifests, make_integration_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_three_window_qualification_and_one_window_resume_keep_identities(tmp_path: Path) -> None:
    manifests = ensure_manifests(tmp_path)
    train, evaluation = load_partition(*(str(path) for path in manifests))
    assert train.profile == evaluation.profile == "shared_500"
    assert len(train.cases) == len(evaluation.cases) == 500
    args = SimpleNamespace(
        checkpoint=tmp_path / "checkpoint",
        tokenizer=tmp_path / "tokenizer.model",
        norm_stats=tmp_path / "norm_stats.json",
        sim_python=Path("/qualified/libero/python"),
        resume=None,
    )
    original = make_integration_config(args, manifests)
    args.resume = tmp_path / "iter_000009"
    restored = make_integration_config(args, manifests)
    first, second = instantiate(original.rl), instantiate(restored.rl)
    assert first.decisions == 384 and first.decisions // first.global_minibatch == 3
    assert first.max_steps == 9 and second.max_steps == 12
    assert first.save_every == second.save_every == 3
    assert first.validation.before_training and first.validation.every_steps == 9
    assert first.identity() == second.identity()
    assert validation_identity(original) == validation_identity(restored)
    assert second.resume == str(args.resume)


def test_manifest_pair_is_validated_before_any_write(tmp_path: Path) -> None:
    directory = tmp_path / "manifests"
    directory.mkdir()
    evaluation = directory / "libero_spatial-shared_500-eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "suite": "libero_spatial",
                "split": "eval",
                "profile": "shared_500",
                "cases": [],
            }
        )
    )
    with pytest.raises(ValueError, match="partition"):
        ensure_manifests(tmp_path)
    assert not (directory / "libero_spatial-shared_500-train.json").exists()
