# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Optimizer inspection cannot turn an empty checkpoint into missing Adam state."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.test_roles_trainer_sft_validation import _assert_equal

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _objective(model, batch):
    return (model(batch["x"]) - 1).square().mean(), {}


def _trainer(checkpoint_format="dcp"):
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    return LLMTrainer(
        model,
        Objective(_objective, Reduction.MEAN),
        torch.optim.AdamW(model.parameters(), lr=0.01),
        checkpoint_format=checkpoint_format,
        save_async=False,
    )


def _inspect(trainer):
    # The campaign audit and ordinary optimizer introspection both use this
    # supported defaultdict access. It creates empty owner entries, no moments.
    return {id(p): copy.deepcopy(trainer.opt.state[p]) for p in trainer._net().parameters()}


def test_empty_owned_state_entries_are_still_pristine():
    trainer = _trainer()
    assert all(value == {} for value in _inspect(trainer).values())
    assert trainer.opt.state
    assert trainer.optimizer_is_pristine()
    assert trainer._update_state_dict()["optimizer_pristine"] == 1


@pytest.mark.parametrize("nonempty", ["step_zero", "moment", "group_step", "master", "foreign_owner", "invalid_state"])
def test_real_or_unowned_optimizer_state_is_never_discarded(nonempty):
    trainer = _trainer()
    _inspect(trainer)
    parameter = next(trainer._net().parameters())
    if nonempty == "step_zero":
        trainer.opt.state[parameter]["step"] = torch.tensor(0.0)
    elif nonempty == "moment":
        trainer.opt.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
    elif nonempty == "group_step":
        trainer.opt.param_groups[0]["step"] = 0
    elif nonempty == "master":
        trainer.opt.param_groups_master = []
    elif nonempty == "foreign_owner":
        trainer.opt.state[torch.nn.Parameter(torch.zeros(1))] = {}
    else:
        trainer.opt.state[parameter] = None
    assert not trainer.optimizer_is_pristine()
    assert "optimizer_pristine" not in trainer._update_state_dict()


@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
@pytest.mark.parametrize("reuse_stepped_target", [False, True])
def test_inspected_preupdate_checkpoint_resumes_first_real_adam_update(
    tmp_path: Path, checkpoint_format, reuse_stepped_target
):
    original = _trainer(checkpoint_format)
    _inspect(original)
    weights = copy.deepcopy(original._net().state_dict())
    groups = copy.deepcopy(original.opt.state_dict()["param_groups"])
    path = str(tmp_path / "iter_000000")
    original.save(path)
    original.finalize_checkpoints()
    state = original.checkpoint_manager.load_component(path, "trainer")
    assert state["iter"] == state["optimizer_step"] == state["attempt_id"] == 0
    assert state["optimizer_pristine"] == 1
    assert not any(original.opt.state.values())
    _assert_equal(weights, original._net().state_dict())
    _assert_equal(groups, original.opt.state_dict()["param_groups"])
    restored = _trainer(checkpoint_format)
    if reuse_stepped_target:
        restored.train({"x": torch.ones(2, 2)})
        assert restored.iteration() == 1 and not restored.optimizer_is_pristine()
    assert restored.load(path) == 0
    assert restored.optimizer_is_pristine() and not restored.opt.state
    for batch in (torch.tensor([[0.0, 1.0], [1.0, 2.0]]), torch.tensor([[3.0, -1.0]])):
        original.train({"x": batch})
        restored.train({"x": batch})
        _assert_equal(original._net().state_dict(), restored._net().state_dict())
        _assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    assert restored.iteration() == 2 and not restored.optimizer_is_pristine()


def test_legacy_missing_pristine_marker_remains_conservative():
    assert not LLMTrainer._checkpoint_recorded_pristine_optimizer({"iter": 0, "optimizer_step": 0})


def test_frozen_actor_clock_is_not_mistaken_for_an_optimizer_step():
    trainer = _trainer()
    # PPO's frozen-actor stretch can advance its trainer clock without an
    # optimizer step. Preserve that existing distinction after inspection.
    trainer._iter = trainer._attempt_id = 3
    _inspect(trainer)
    assert trainer.optimizer_is_pristine()
    assert trainer._update_state_dict()["optimizer_pristine"] == 1


def test_postupdate_checkpoint_retains_initialized_state_and_nonzero_clocks(tmp_path: Path):
    original = _trainer()
    original.train({"x": torch.ones(2, 2)})
    _inspect(original)
    path = str(tmp_path / "iter_000001")
    original.save(path)
    original.finalize_checkpoints()
    state = original.checkpoint_manager.load_component(path, "trainer")
    assert state["iter"] == state["optimizer_step"] == 1 and "optimizer_pristine" not in state
    restored = _trainer()
    assert restored.load(path) == 1
    _assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    original.train({"x": torch.zeros(1, 2)})
    restored.train({"x": torch.zeros(1, 2)})
    _assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    _assert_equal(original._net().state_dict(), restored._net().state_dict())


def test_missing_initialized_moment_still_fails_strict_resume(tmp_path: Path):
    from torch.distributed.checkpoint.api import CheckpointException

    original = _trainer()
    original.train({"x": torch.ones(2, 2)})
    path = tmp_path / "iter_000001"
    original.save(str(path))
    original.finalize_checkpoints()
    metadata_path = path / "optim/.metadata"
    metadata = pickle.loads(metadata_path.read_bytes())
    key = "state.weight.exp_avg"
    assert key in metadata.state_dict_metadata
    del metadata.state_dict_metadata[key]
    metadata_path.write_bytes(pickle.dumps(metadata))
    target = _trainer()
    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: state.weight.exp_avg"):
        target.load(str(path))
