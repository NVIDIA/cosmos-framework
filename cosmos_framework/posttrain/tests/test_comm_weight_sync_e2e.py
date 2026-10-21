# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""End-to-end cross-device weight sync over a REAL NCCL weight plane (spec §7.1), 2 GPUs, 2 procs.

Rank 0 is the policy; rank 1 is a random-initialized HFRollout on a different GPU. The two procs
form the side group and run a genuinely concurrent broadcast (rank-0 export ∥ rank-1 apply). We
assert PARAMETER equality — not generations (different kernels can diverge a token even with
identical weights) — after the bootstrap sync (kills a no-op) and after a training-step update
(proves propagation). This is the hardware proof of the loopback correctness in
test_roles_rollout_apply_weights.py and the transport in test_comm_transfer_weight_comm.py.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _e2e_worker(rank, world, host, port, q):
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover
    from cosmos_framework.posttrain.comm.transfer.weight_comm import build_weight_comm
    from cosmos_framework.posttrain.comm.weight_transfer import WeightMapper, WeightSync
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout.hf import HFRollout

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    comm = build_weight_comm(host, port, rank, world, backend="nccl")

    if rank == 0:  # policy: known weights -> export v1 (bootstrap), train one step, export v2
        torch.manual_seed(0)
        policy = nn.Linear(8, 8).to(dev)
        be = WeightSync(NcclMover(comm, src=0))
        v1 = {k: v.detach().cpu() for k, v in WeightMapper().export_state(policy).items()}
        be.export(policy, version=1)  # broadcast-send (rendezvous with rank-1 apply)
        with torch.no_grad():  # one "training step": weights genuinely change
            policy.weight += 1.0
            policy.bias += 0.5
        v2 = {k: v.detach().cpu() for k, v in WeightMapper().export_state(policy).items()}
        be.export(policy, version=2)
        q.put((0, v1, v2))
    else:  # rollout: random init, populated entirely by the two syncs
        rollout_net = nn.Linear(8, 8).to(dev)
        nn.init.normal_(rollout_net.weight, std=5.0)
        nn.init.normal_(rollout_net.bias, std=5.0)
        # Packer is REQUIRED at construction now; this rig is a bare nn.Linear weight receiver that
        # never generates, so gpt2's (small, cached) tokenizer stands in as the unused shaping seam.
        rollout = HFRollout(rollout_net, packer=init_data_packer("gpt2"))
        rollout._weight_backend = WeightSync(NcclMover(comm, src=0))
        # Body surface is peer-only; the manually co-scheduled whole-model collective recv lives on
        # the BACKEND layer (recv layout self-derived from the rollout's own model).
        rollout._weight_backend.apply(rollout.model)  # recv v1 (rendezvous with rank-0 export)
        after1 = {k: v.detach().cpu() for k, v in rollout_net.state_dict().items()}
        rollout._weight_backend.apply(rollout.model)  # recv v2
        after2 = {k: v.detach().cpu() for k, v in rollout_net.state_dict().items()}
        q.put((1, after1, after2))
    comm.close()


@pytest.mark.L1
@pytest.mark.GPU
def test_weight_sync_e2e_two_gpu_nccl():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs (policy and rollout on different devices)")
    host, port, world = "127.0.0.1", _free_port(), 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_e2e_worker, args=(r, world, host, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(world):
        r, *rest = q.get()
        out[r] = rest
    for p in procs:
        p.join(timeout=120)

    (policy_v1, policy_v2) = out[0]
    (rollout_after1, rollout_after2) = out[1]
    # Bootstrap sync: random rollout now equals policy v1 (transport + layout correct; kills no-op).
    for k in policy_v1:
        assert torch.equal(rollout_after1[k], policy_v1[k]), f"{k} mismatch after bootstrap sync"
    # Second sync propagates the training-step update (not just the initial load).
    for k in policy_v2:
        assert torch.equal(rollout_after2[k], policy_v2[k]), f"{k} mismatch after update sync"
    # Sanity: the update actually changed the weights (so the second assert is non-trivial).
    assert not torch.equal(policy_v1["weight"], policy_v2["weight"])
