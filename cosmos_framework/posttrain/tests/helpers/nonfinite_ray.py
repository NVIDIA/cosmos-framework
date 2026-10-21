# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared two-rank Ray check for non-finite consensus and recovery."""

from __future__ import annotations

import os
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
from cosmos_framework.posttrain.workers.grid import GridBackend
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

_WORLD_SIZE = 2


class _Policy:
    """Minimal policy body for an engine-level distributed update check."""

    def __init__(self, device: torch.device) -> None:
        self.net = torch.nn.Linear(2, 1, bias=False).to(device)


class NonFiniteTrainerProbe:
    """Inject one-rank loss and gradient faults into a real trainer."""

    def __init__(self) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("non-finite trainer probe requires an initialized process group")
        self.rank = dist.get_rank()
        self.backend = str(dist.get_backend()).lower()
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if "nccl" in self.backend else torch.device("cpu")
        )
        self.mode = "finite"
        torch.manual_seed(11)
        self.policy = _Policy(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.net.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

        def objective(
            model: _Policy,
            _batch: list[Any],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss = model.net.weight.square().sum()  # []
            if self.mode == "loss" and self.rank == 1:
                loss = loss * torch.tensor(float("nan"), device=self.device)  # []
            return loss, {"dpo_loss": loss.detach()}

        self.trainer = DiffusionTrainer(
            self.policy,
            Objective(objective, Reduction.MEAN),
            self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

    @staticmethod
    def _inject_nonfinite_gradient(gradient: torch.Tensor) -> torch.Tensor:  # [1,2] -> [1,2]
        return gradient * float("nan")  # [1,2]

    def _state_is_pristine(
        self,
        initial_weight: torch.Tensor,  # [1,2]
        initial_scheduler: dict[str, Any],
    ) -> bool:
        return (
            torch.equal(self.policy.net.weight, initial_weight)
            and not self.optimizer.state_dict()["state"]
            and self.scheduler.state_dict() == initial_scheduler
            and all(parameter.grad is None for parameter in self.policy.net.parameters())
        )

    def run_nonfinite_sequence(self) -> dict[str, Any]:
        """Reject one-rank loss and gradient faults, then apply a finite update."""
        initial_weight = self.policy.net.weight.detach().clone()  # [1,2]
        initial_scheduler = self.scheduler.state_dict()

        self.mode = "loss"
        loss_skipped = self.trainer.train([])
        loss_state_unchanged = self._state_is_pristine(initial_weight, initial_scheduler)

        self.mode = "finite"
        hook = self.policy.net.weight.register_hook(self._inject_nonfinite_gradient) if self.rank == 1 else None
        try:
            gradient_skipped = self.trainer.train([])
        finally:
            if hook is not None:
                hook.remove()
        gradient_state_unchanged = self._state_is_pristine(initial_weight, initial_scheduler)

        applied = self.trainer.train([])
        final_weight = self.policy.net.weight.detach().cpu().reshape(-1).tolist()  # [P]
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "backend": self.backend,
            "outcomes": [
                loss_skipped["optim/update_outcome"],
                gradient_skipped["optim/update_outcome"],
                applied["optim/update_outcome"],
            ],
            "skip_reasons": [
                loss_skipped["optim/skip_reason"],
                gradient_skipped["optim/skip_reason"],
            ],
            "loss_state_unchanged": loss_state_unchanged,
            "gradient_state_unchanged": gradient_state_unchanged,
            "recovered_weight_changed": not torch.equal(self.policy.net.weight, initial_weight),
            "optimizer_state_created": bool(self.optimizer.state_dict()["state"]),
            "attempt_id": applied["optim/attempt_id"],
            "optimizer_step": applied["optim/optimizer_step"],
            "skips_total": applied["optim/nonfinite_skips_total"],
            "skips_consecutive": applied["optim/nonfinite_skips_consecutive"],
            "final_weight": final_weight,
        }


def setup_nonfinite_body() -> dict[str, Any]:
    """Build the trainer probe after the actor has joined its process group."""
    return {"policy": NonFiniteTrainerProbe()}


def make_nonfinite_grid(group_id: str, hosts: list[Any], *, backend: str) -> WorkerGroup:
    """Bind existing Ray actors to a two-rank Gloo or NCCL policy grid."""
    import ray

    if len(hosts) != _WORLD_SIZE:
        raise ValueError(f"non-finite test grid requires {_WORLD_SIZE} hosts, got {len(hosts)}")
    master_addr, master_port = ray.get(hosts[0].rendezvous_endpoint.remote())
    ray.get(
        [
            host.init_distributed.remote(master_addr, master_port, rank, _WORLD_SIZE, backend)
            for rank, host in enumerate(hosts)
        ]
    )
    grid = {(0, rank): (f"{group_id}_r0_k{rank}", host) for rank, host in enumerate(hosts)}
    # This probe uses the live two-rank process group directly but no sharded model.
    # Report ranks=1 to run_setup so it does not inject a production DeviceMesh.
    ray.get(
        [host.run_setup.remote(setup_nonfinite_body, False, 0, (), store_id, 1, 1) for store_id, host in grid.values()]
    )
    policy = WorkerGroup("policy", role="trainer", replicas=1, ranks=_WORLD_SIZE)
    policy._bind_backend(GridBackend(grid, shape=(1, _WORLD_SIZE)))
    return policy


def assert_nonfinite_consensus(policy: WorkerGroup, *, expected_backend: str) -> None:
    """Verify rank-consistent rejection, no mutation, and recovery."""
    results = policy.run_nonfinite_sequence().all()
    assert set(results) == {(0, 0), (0, 1)}
    assert {result["rank"] for result in results.values()} == {0, 1}
    assert len({result["pid"] for result in results.values()}) == _WORLD_SIZE
    for result in results.values():
        assert expected_backend in result["backend"]
        assert result["outcomes"] == ["skipped", "skipped", "applied"]
        assert result["skip_reasons"] == ["nonfinite_loss", "nonfinite_gradient"]
        assert result["loss_state_unchanged"]
        assert result["gradient_state_unchanged"]
        assert result["recovered_weight_changed"]
        assert result["optimizer_state_created"]
        assert result["attempt_id"] == 3
        assert result["optimizer_step"] == 1
        assert result["skips_total"] == 2
        assert result["skips_consecutive"] == 0
    rank0_weight = torch.tensor(results[(0, 0)]["final_weight"])  # [P]
    rank1_weight = torch.tensor(results[(0, 1)]["final_weight"])  # [P]
    torch.testing.assert_close(rank0_weight, rank1_weight, rtol=0, atol=0)
