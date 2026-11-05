# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Architecture guard: algorithm entrypoints see checkpoint semantics, not storage."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ENTRYPOINTS = ("dpo.py", "grpo.py", "nft.py", "opd.py", "joint_ppo.py")
_ENGINE_TRAINERS = ("base.py", "diffusion.py", "llm.py")
_CHECKPOINT_PACKAGE = "cosmos_framework.posttrain.checkpoint"
_LIFECYCLE_MODULE = "cosmos_framework.posttrain.control.checkpointing"
_FORBIDDEN_NAMES = {
    "checkpoint_store_spec",
    "checkpoint_upload_spec",
    "load_trainer_checkpoint",
    "primary_checkpoint_store",
    "resolve_rank_checkpoint_resume",
}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_entrypoint_checkpoint_dependencies_stop_at_the_lifecycle_facade(entrypoint: str) -> None:
    path = Path(__file__).parents[1] / "entrypoints" / entrypoint
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(_CHECKPOINT_PACKAGE), (
                    f"{entrypoint} imports checkpoint implementation {alias.name!r}; use CheckpointLifecycle"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(_CHECKPOINT_PACKAGE), (
                f"{entrypoint} imports checkpoint implementation {module!r}; use CheckpointLifecycle"
            )
            if module == _LIFECYCLE_MODULE:
                assert {alias.name for alias in node.names} == {"CheckpointLifecycle"}
        elif isinstance(node, ast.Name):
            assert node.id not in _FORBIDDEN_NAMES
        elif isinstance(node, ast.Attribute):
            assert node.attr != "is_object_store"
            assert node.attr != "store"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "s3://" not in node.value


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("trainer", _ENGINE_TRAINERS)
def test_engine_trainers_do_not_branch_on_checkpoint_store(trainer: str) -> None:
    path = Path(__file__).parents[1] / "roles" / "trainer" / trainer
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "store", f"{trainer} branches on CheckpointStore; expose the policy through the manager"
        elif trainer != "base.py" and isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == "CheckpointManager"), (
                f"{trainer} constructs a checkpoint manager instead of using the injected manager boundary"
            )
