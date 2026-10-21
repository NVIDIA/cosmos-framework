# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU test: the DPO objective composes with FSDP2 across ranks (DESIGN §1.1 / §6).

Locks the P2 core mechanism — the decoupled `algorithm.dpo` objective run under a real
FSDP2 shard group, the way the grid's intra-replica rank axis works: `fully_shard` the
policy net, run a real DPO step on rank-sharded synthetic pairs, and reduce the loss over
ALL ranks (not just rank-0). Model-free (toy `FlowMatchingPolicy`) + fast, so it guards
the objective↔FSDP seam independently of the heavy 8B run. Ranks are `mp.spawn`ed workers
(the test process is the driver); an assert failing in any worker propagates as a
`ProcessRaisedException`, failing the test.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

DIM, PAIRS, STEPS, BETA = 16, 4, 5, 500.0


def _to_cuda(traj):
    pref = traj.transitions[0].targets  # typed Preference(winner, loser)
    for side in ("winner", "loser"):
        d = getattr(pref, side)
        setattr(pref, side, {k: v.cuda() for k, v in d.items()})
    return traj


def _worker(rank: int, world: int, port: int) -> None:
    """One FSDP rank: shard the policy net, run DPO steps, all-reduce the loss."""
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from cosmos_framework.posttrain.algorithm.dpo import make_objective
    from cosmos_framework.posttrain.tests.helpers.preference_env import SyntheticPreferenceEnv
    from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    try:
        mesh = dist.device_mesh.init_device_mesh("cuda", (world,))  # 1-D shard mesh (pure FSDP)

        # Toy policy; shard the trainable net across the mesh. The frozen ref stays
        # replicated per rank (no grad) — the real model would shard it too.
        model = ToyFlowModel(dim=DIM, seed=0)
        model.net = model.net.cuda()
        model.net_ref = model.net_ref.cuda()
        fully_shard(model.net, mesh=mesh)

        opt = torch.optim.AdamW(model.net.parameters(), lr=1e-2)
        objective = make_objective(BETA)
        env = SyntheticPreferenceEnv(dim=DIM, seed=rank)  # each rank draws its own DP shard

        losses = []
        for _ in range(STEPS):
            batch = [_to_cuda(t) for t in env.get_samples(PAIRS)]
            opt.zero_grad(set_to_none=True)
            loss, _ = objective(model, batch)
            loss.backward()  # FSDP2 reduce-scatters grads across the mesh
            opt.step()
            red = loss.detach().clone()
            dist.all_reduce(red, op=dist.ReduceOp.AVG)  # the grid's `train` reduction
            losses.append(float(red))

        # (1) all-reduced loss stays finite; (2) it actually moves (a real optimizer step);
        assert all(v == v and abs(v) != float("inf") for v in losses), f"non-finite loss: {losses}"
        assert abs(losses[-1] - losses[0]) > 0.0, f"loss did not change: {losses}"
        # (3) the reduction is identical on every rank (it is an all-reduce).
        check = torch.tensor([losses[-1]], device="cuda")
        gathered = [torch.zeros_like(check) for _ in range(world)]
        dist.all_gather(gathered, check)
        spread = max(float(g) for g in gathered) - min(float(g) for g in gathered)
        assert spread < 1e-4, f"all-reduced loss differs across ranks: spread={spread}"
    finally:
        dist.destroy_process_group()


@pytest.mark.L1
@pytest.mark.GPU
def test_dpo_objective_under_fsdp2_across_ranks():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    import socket

    import torch.multiprocessing as mp

    world = 2
    with socket.socket() as s:  # grab a free port for the rendezvous
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(world, port), nprocs=world, join=True)
