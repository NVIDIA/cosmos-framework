# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for the model-agnostic flow-BC objective on a toy collate and a toy model: reduction, metrics,
identity, DP-global normalization, and that the objective reaches the model only through `flow_matching_loss`."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, Reduction
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

HORIZON, WIDTH, ACTION_DIM, PREFIX = 8, 12, 5, 4


class _ToyModel:
    """Exposes only what the objective may touch: `flow_matching_loss(inputs) -> FlowMatchingLossOutput`."""

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen: list[dict] = []
        self.last_loss: torch.Tensor | None = None

    def flow_matching_loss(self, inputs: dict) -> FlowMatchingLossOutput:
        self.seen.append(inputs)
        mask = inputs["action_mask"]
        # Isaac's shape: a masked mean with a 1e-6 guard in the divisor.
        loss = (inputs["action"] * mask).sum() / (mask.sum() + 1e-6) * self.weight
        self.last_loss = loss
        return FlowMatchingLossOutput(loss=loss, action_mask=mask)


def _collate(trajectories: list[Trajectory]) -> dict:
    actions = torch.stack([t.action for traj in trajectories for t in traj.transitions])
    mask = torch.zeros_like(actions)
    mask[:, :PREFIX, :ACTION_DIM] = 1.0
    return {"action": actions, "action_mask": mask}


def _batch(n: int = 2) -> list[Trajectory]:
    return [
        Trajectory(
            transitions=[Transition(obs=ObsBundle(), action=torch.full((HORIZON, WIDTH), float(i + 1)))],
            id=f"t{i}",
            status=Status.COMPLETED,
        )
        for i in range(n)
    ]


def _fake_two_ranks(monkeypatch, dp_mean_count: float) -> list:
    """Fake the one collective the objective issues (AVG of the masked count); gloo has no AVG, so a
    real two-process group cannot stand in. Returns the list of ops issued."""
    calls: list = []

    def fake_all_reduce(tensor, op=None):
        calls.append(op)
        tensor.fill_(dp_mean_count)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    return calls


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_drives_the_model_loss_and_publishes_its_metrics() -> None:
    objective = make_objective(_collate, training_prefix=PREFIX, action_dim=ACTION_DIM)
    assert objective.reduction is Reduction.TOKEN_SUM  # normalized by the DP-mean masked count

    model = _ToyModel()
    loss, metrics = objective(model, _batch())
    assert loss.requires_grad
    assert model.seen[0]["action_mask"].sum() == 2 * PREFIX * ACTION_DIM
    assert metrics["flow_bc_loss"] == pytest.approx(float(loss.detach()))
    assert metrics["flow_bc_masked_elements"] == 2 * PREFIX * ACTION_DIM
    assert set(objective.adapt_metrics(metrics, "train")) == {"train/loss_avg", "train/flow_bc_masked_elements"}


@pytest.mark.L0
@pytest.mark.CPU
def test_single_rank_loss_is_bitwise_the_model_loss(monkeypatch) -> None:
    """The parity guard: outside a process group, and on equal shards inside one, the DP scale is exactly 1.0
    and the returned loss is the model's own loss byte for byte."""
    objective = make_objective(_collate, training_prefix=PREFIX, action_dim=ACTION_DIM)
    model = _ToyModel()
    batch = _batch()
    masked = 2 * PREFIX * ACTION_DIM

    loss, metrics = objective(model, batch)
    assert torch.equal(loss, model.last_loss)
    assert metrics[OBJECTIVE_WEIGHT_SUM] == float(masked)

    calls = _fake_two_ranks(monkeypatch, dp_mean_count=float(masked))
    loss, metrics = objective(model, batch)
    assert dist.ReduceOp.AVG in calls
    assert torch.equal(loss, model.last_loss)
    assert metrics[OBJECTIVE_WEIGHT_SUM] == float(masked)


@pytest.mark.L0
@pytest.mark.CPU
def test_unequal_shards_normalize_to_the_global_masked_mean(monkeypatch) -> None:
    """FSDP averages rank gradients with equal weight, so each rank's loss must be its masked SUM over the
    DP-mean masked count: then the mean of the rank losses is the whole batch's masked mean."""
    objective = make_objective(_collate, training_prefix=PREFIX, action_dim=ACTION_DIM)
    model = _ToyModel()
    full = _batch(3)
    whole = float(objective(model, full)[0].detach())

    per_trajectory = PREFIX * ACTION_DIM
    dp_mean = (1 * per_trajectory + 2 * per_trajectory) / 2
    calls = _fake_two_ranks(monkeypatch, dp_mean_count=dp_mean)
    short, short_metrics = objective(model, full[:1])
    long, long_metrics = objective(model, full[1:])

    assert dist.ReduceOp.AVG in calls, "the DP-global denominator collective was not issued"
    assert short_metrics[OBJECTIVE_WEIGHT_SUM] == long_metrics[OBJECTIVE_WEIGHT_SUM] == dp_mean
    assert (float(short.detach()) + float(long.detach())) / 2 == pytest.approx(whole), (
        "the mean of the rank losses must equal the whole batch's masked mean"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_carries_the_supervised_region_so_a_resume_cannot_change_the_loss() -> None:
    identity = make_objective(_collate, training_prefix=PREFIX, action_dim=ACTION_DIM).identity
    assert identity == {"algorithm": "flow_bc", "training_prefix": PREFIX, "action_dim": ACTION_DIM}
    assert make_objective(_collate, training_prefix=PREFIX + 1, action_dim=ACTION_DIM).identity != identity
