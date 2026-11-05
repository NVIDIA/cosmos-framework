# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ray/Gloo save-resume gate for the LLM policy lifecycle shared by OPD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
from cosmos_framework.posttrain.roles.rollout.local import LocalRollout
from cosmos_framework.posttrain.roles.trainer import LLMTrainer
from cosmos_framework.posttrain.workers.grid import GridBackend
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

_WORLD_SIZE = 2


def _opd_policy_objective(model: Any, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tiny differentiable stand-in for OPD's objective-injected LLM policy step."""
    net = getattr(model, "net", model)
    inputs = torch.ones(1, 2)  # [B,D]
    prediction = net(inputs)  # [B,D]
    loss = prediction.square().mean()  # []
    return loss, {"train/loss": loss.detach()}  # []


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()  # [*state]
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _assert_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
        return
    assert actual == expected


def _adam_moments(state: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    parameter_states = state["state"]
    assert isinstance(parameter_states, dict) and parameter_states
    moments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for parameter_state in parameter_states.values():
        assert {"step", "exp_avg", "exp_avg_sq"} <= set(parameter_state)
        exp_avg = parameter_state["exp_avg"]  # [*parameter]
        exp_avg_sq = parameter_state["exp_avg_sq"]  # [*parameter]
        assert isinstance(exp_avg, torch.Tensor)
        assert isinstance(exp_avg_sq, torch.Tensor)
        moments.append((exp_avg, exp_avg_sq))
    return moments


class _CheckpointProbeTrainer(LLMTrainer):
    """Real LLMTrainer with read-only state probes used only by this Ray gate."""

    def optimizer_state_cpu(self) -> dict[str, Any]:
        state = _cpu_snapshot(self.opt.state_dict())
        assert isinstance(state, dict)
        return state

    def model_state_cpu(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, value in self.model.state_dict().items():
            state[name] = value.detach().cpu().clone()  # [*parameter]
        return state


class _CheckpointProbeRollout(LocalRollout):
    """Independent receiver model with observable weights and publication version."""

    def model_state_cpu(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, value in self.model.state_dict().items():
            state[name] = value.detach().cpu().clone()  # [*parameter]
        return state

    def weight_version(self) -> int:
        return self._weight_version


def _setup_opd_policy() -> dict[str, Any]:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    return {
        "policy": _CheckpointProbeTrainer(
            model,
            Objective(_opd_policy_objective, reduction=Reduction.MEAN),
            optimizer,
            checkpoint_format="dcp",
        )
    }


def _setup_opd_rollout() -> dict[str, Any]:
    torch.manual_seed(17)
    return {"rollout": _CheckpointProbeRollout(torch.nn.Linear(2, 2))}


def _make_policy_and_rollout_grids(group_id: str) -> tuple[WorkerGroup, WorkerGroup]:
    from cosmos_framework.posttrain.workers.ray import _Host

    policy_grid: dict[tuple[int, int], tuple[str, Any]] = {}
    rollout_grid: dict[tuple[int, int], tuple[str, Any]] = {}
    for rank in range(_WORLD_SIZE):
        policy_store_id = f"{group_id}_policy_r0_k{rank}"
        rollout_store_id = f"{group_id}_rollout_r0_k{rank}"
        policy_grid[(0, rank)] = (policy_store_id, _Host.remote())
        rollout_grid[(0, rank)] = (rollout_store_id, _Host.remote())
    master_addr, master_port = ray.get(policy_grid[(0, 0)][1].rendezvous_endpoint.remote())
    ray.get(
        [
            policy_grid[(0, rank)][1].init_distributed.remote(master_addr, master_port, rank, _WORLD_SIZE)
            for rank in range(_WORLD_SIZE)
        ]
    )
    ray.get(
        [
            host.run_setup.remote(setup, True, 1 << 20, (), store_id, 1, 1)
            for setup, grid in ((_setup_opd_policy, policy_grid), (_setup_opd_rollout, rollout_grid))
            for store_id, host in grid.values()
        ]
    )
    peers = {store_id: host for grid in (policy_grid, rollout_grid) for store_id, host in grid.values()}
    ray.get([host.set_peers.remote(peers) for host in peers.values()])
    ray.get(
        [
            host.set_weight_backend.remote({"relationship": "host"}, (body_name,))
            for body_name, grid in (("policy", policy_grid), ("rollout", rollout_grid))
            for _store_id, host in grid.values()
        ]
    )
    policy = WorkerGroup("policy", role="trainer", replicas=1, ranks=_WORLD_SIZE)
    rollout = WorkerGroup("rollout", role="rollout", replicas=1, ranks=_WORLD_SIZE)
    # LLMTrainer already reduces metrics across its K-rank Torch world; the one execution replica
    # therefore publishes one already-global dictionary rather than independent replica contributions.
    policy._bind_backend(GridBackend(policy_grid, shape=(1, _WORLD_SIZE), dp_reduce_op="identical"))
    rollout._bind_backend(GridBackend(rollout_grid, shape=(1, _WORLD_SIZE), dp_reduce_op="identical"))
    return policy, rollout


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_llm_policy_checkpoint_round_trip_on_ray(tmp_path: Path) -> None:
    """Restore policy weights/moments, then publish them to an independent rollout mesh."""
    ray.init(
        address="local",
        num_cpus=5,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/c3-opd-checkpoint-ray",
    )
    try:
        policy, rollout = _make_policy_and_rollout_grids("opd_checkpoint")
        first = policy.train([]).result()
        assert first["optim/update_outcome"] == "applied"
        assert policy.iteration().result() == 1
        saved_model = policy.model_state_cpu().all()
        saved_optimizer = policy.optimizer_state_cpu().all()
        for state in saved_optimizer.values():
            _adam_moments(state)

        checkpoint = str(tmp_path / "checkpoints" / "iter_000001")
        assert set(policy.save(checkpoint).all().values()) == {None}
        assert set(policy.finalize_checkpoints().all().values()) == {None}
        assert (Path(checkpoint) / "model" / ".metadata").is_file()
        assert (Path(checkpoint) / "trainer.pt").is_file()

        policy.train([]).result()
        assert policy.iteration().result() == 2
        mutated_optimizer = policy.optimizer_state_cpu().all()
        assert any(
            not torch.equal(before, after)
            for cell in saved_optimizer
            for before_pair, after_pair in zip(
                _adam_moments(saved_optimizer[cell]),
                _adam_moments(mutated_optimizer[cell]),
                strict=True,
            )
            for before, after in zip(before_pair, after_pair, strict=True)
        )

        assert set(policy.load(checkpoint).all().values()) == {1}
        assert set(policy.iteration().all().values()) == {1}
        restored_model = policy.model_state_cpu().all()
        restored_optimizer = policy.optimizer_state_cpu().all()
        _assert_state_equal(restored_model, saved_model)
        _assert_state_equal(restored_optimizer, saved_optimizer)

        rollout_before = rollout.model_state_cpu().all()
        assert set(rollout.weight_version().all().values()) == {0}
        assert any(
            not torch.equal(restored_model[cell][name], rollout_before[cell][name])
            for cell in restored_model
            for name in restored_model[cell]
        )

        weight_sync(rollout, policy, validation_step=1)
        rollout_after = rollout.model_state_cpu().all()
        _assert_state_equal(rollout_after, restored_model)
        assert set(rollout.weight_version().all().values()) == {1}

        continued = policy.train([]).result()
        assert continued["optim/update_outcome"] == "applied"
        assert policy.iteration().result() == 2
    finally:
        ray.shutdown()
