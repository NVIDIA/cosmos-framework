# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact coverage and optimizer invariants at finite, uneven SFT update tails."""

import copy

import pytest
import torch

from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.config.build import build_warmup_cosine_lr_scheduler
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv
from cosmos_framework.posttrain.roles.env.sft_cursor import FiniteSFTCursor
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Rows:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"messages": [{"role": "assistant", "content": str(i)}]}

    def cursor_fingerprint(self):
        return f"finite-test-{self.n}"


@pytest.mark.parametrize("n,world,update", [(1, 4, 4), (5, 4, 4), (12, 4, 4), (16, 4, 4), (17, 4, 4)])
def test_finite_epochs_cover_every_row_once_with_explicit_padding(n, world, update):
    by_epoch = [[], []]
    padding = 0
    for rank in range(world):
        env = SFTDatasetEnv(Rows(n), rank=rank, world_size=world, seed=42, training_epochs=2, samples_per_update=update)
        slots = env.cursor.slots_per_epoch
        for epoch in range(2):
            rows = env.get_samples(slots)
            for row in rows:
                extra = row.transitions[0].algo_extra
                if extra.get("training_padding"):
                    assert extra["validation_weight"] == 0
                    padding += 1
                else:
                    by_epoch[epoch].append(int(row.id.rsplit(":", 1)[1]))
        with pytest.raises(RuntimeError, match="exhausted"):
            env.get_samples(1)
    assert [sorted(rows) for rows in by_epoch] == [list(range(n)), list(range(n))]
    assert padding == 2 * (slots * world - n)


def test_cursor_resume_at_every_position_including_padding_and_epoch_boundary():
    def cursor(**kwargs):
        return FiniteSFTCursor(
            Rows(5), rank=3, world_size=4, shuffle=True, seed=7, epochs=kwargs.get("epochs", 2), samples_per_update=4
        )

    original = cursor()
    total = original.epochs * original.slots_per_epoch
    for pos in range(total + 1):
        current = cursor()
        for _ in range(pos):
            current.next_index()
        state = current.state_dict()
        restored = cursor()
        restored.load_state_dict(state)
        assert [restored.next_index() for _ in range(total - pos)] == [current.next_index() for _ in range(total - pos)]
        bad = copy.deepcopy(state)
        bad["position"] = total + 1
        with pytest.raises(ValueError, match="position"):
            cursor().load_state_dict(bad)
        with pytest.raises(ValueError, match="identity"):
            cursor(epochs=3).load_state_dict(state)


def test_padding_only_microbatches_preserve_global_mean_and_adam_state():
    torch.manual_seed(19)
    packer = MemoryPacker()
    real = [packer.add([1, 2, 3], [-100, 2, 3]), packer.add([4, 5, 6, 7, 8], [-100, 5, 6, 7, 8])]
    pad = packer.add([1, 2, 3], [-100, 2, 3], weight=0.0)
    pad.transitions[0].algo_extra["training_padding"] = True
    policy = Policy()
    reference = copy.deepcopy(policy)
    trainer = LLMTrainer(
        policy,
        make_objective(packer, allow_empty_training_padding=True),
        torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.1),
        grad_accum=4,
        max_grad_norm=0,
    )
    baseline = LLMTrainer(
        reference,
        make_objective(packer),
        torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1),
        max_grad_norm=0,
    )
    baseline.train(real)
    for row in [*real, pad, pad]:
        trainer.train([row])
    assert trainer.iteration() == 1
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    for a, b in zip(trainer.opt.state.values(), baseline.opt.state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-7, rtol=2e-6)
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    for _ in range(3):
        trainer.train([pad])
    with pytest.raises(RuntimeError, match="only padding"):
        trainer.train([pad])
    assert trainer.iteration() == 1
    for name, p in policy.named_parameters():
        assert torch.equal(p, before[name])
    invalid = packer.add([1, 2], [-100, -100])
    with pytest.raises(ValueError, match="no globally supervised"):
        make_objective(packer, allow_empty_training_padding=True)(policy, [invalid])


def test_scheduler_resume_binds_construction_before_loading_weights(tmp_path):
    def trainer(cycle):
        torch.manual_seed(7)
        policy = Policy()
        packer = MemoryPacker()
        row = packer.add([1, 2, 3], [-100, 2, 3])
        opt = torch.optim.AdamW(policy.parameters(), lr=0.1)
        sched = build_warmup_cosine_lr_scheduler(opt, warmup_steps=0, cycle_steps=cycle, f_start=1, f_min=0.01)
        return LLMTrainer(policy, make_objective(packer), opt, lr_scheduler=sched), row

    source, row = trainer(8)
    source.train([row])
    source.train([row])
    path = str(tmp_path / "checkpoint")
    source.save(path, publish=False, data_state={"fixture": True})
    source.finalize_checkpoints()
    source.train([row])
    resumed, row = trainer(8)
    resumed.load(path)
    resumed.train([row])
    assert resumed.opt.param_groups[0]["lr"] == source.opt.param_groups[0]["lr"]
    for actual, expected in zip(resumed.model.parameters(), source.model.parameters()):
        assert torch.equal(actual, expected)
    changed, _ = trainer(80)
    before = [p.detach().clone() for p in changed.model.parameters()]
    with pytest.raises(RuntimeError, match="scheduler exact-resume identity differs"):
        changed.load(path)
    for actual, expected in zip(changed.model.parameters(), before):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "fault",
    [None, "missing_rng", "wrong_rng_world", "missing_data", "stale_cursor", "invalid_attempt", "noninteger_iter"],
)
def test_finite_resume_preflight_refuses_incomplete_state_before_parameter_mutation(tmp_path, fault):
    from cosmos_framework.posttrain.checkpoint.rng import validate_exact_rng_state
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    def make():
        torch.manual_seed(71)
        policy, packer = Policy(), MemoryPacker()
        for _ in range(5):
            packer.add([1, 2, 3], [-100, 2, 3])
        data = SFTDatasetEnv(Rows(5), shuffle=False, training_epochs=2)
        objective = make_objective(packer, allow_empty_training_padding=True)
        objective.identity = {**objective.identity, "training_coverage": data.cursor.identity}
        trainer = LLMTrainer(policy, objective, torch.optim.AdamW(policy.parameters(), lr=0.01), max_grad_norm=0)
        return trainer, data

    source, data = make()
    source.train(data.get_samples(1))
    state, cursor = source._trainer_state(), data.state_dict()
    if fault == "missing_rng":
        state.pop("rng")
    elif fault == "wrong_rng_world":
        state["rng"][1] = copy.deepcopy(state["rng"][0])
    elif fault == "missing_data":
        cursor = None
    elif fault == "stale_cursor":
        _, fresh_data = make()
        cursor = fresh_data.state_dict()
    elif fault == "invalid_attempt":
        state["attempt_id"] = -1
    elif fault == "noninteger_iter":
        state["iter"] = 1.0
    source._trainer_state = lambda: state
    path = str(tmp_path / "checkpoint")
    source.save(path, publish=False, data_state=cursor)
    source.finalize_checkpoints()
    resumed, resumed_data = make()
    before = [p.detach().clone() for p in resumed.model.parameters()]
    rng = torch.get_rng_state().clone()
    lifecycle = CheckpointLifecycle(resumed.checkpoint_manager, str(tmp_path), path)

    def preflight(trainer_state, data_state):
        validate_exact_rng_state(trainer_state.get("rng"), device=torch.device("cpu"))
        resumed_data.restore_finite_checkpoint_cursor(data_state, trainer_state)

    if fault is not None:
        with pytest.raises((ValueError, RuntimeError)):
            lifecycle.load_trainer_and_component(resumed, "data", before_load=preflight)
        assert not resumed.opt.state
        assert torch.equal(torch.get_rng_state(), rng)
        for actual, expected in zip(resumed.model.parameters(), before):
            assert torch.equal(actual, expected)
    else:
        lifecycle.load_trainer_and_component(resumed, "data", before_load=preflight)
        source.train(data.get_samples(1))
        resumed.train(resumed_data.get_samples(1))
        assert resumed_data.cursor.position == data.cursor.position == 2
        for actual, expected in zip(resumed.model.parameters(), source.model.parameters()):
            assert torch.equal(actual, expected)
