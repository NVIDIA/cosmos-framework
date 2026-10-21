# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Side-PG transport: a WeightComm forms a group independent of any default group and
broadcasts a tensor rank-0 -> all. CPU/gloo (L0); a GPU/nccl sibling asserts the real path."""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp


def _bcast_worker(rank, world, host, port, backend, out_q):
    from cosmos_framework.posttrain.comm.transfer.weight_comm import build_weight_comm

    # In production each rank is a GPU-isolated Ray actor (its own cuda:0); a bare multiprocessing
    # test is not, so pin each nccl rank to its own device or both would collide on cuda:0 and hang.
    dev = None
    if backend == "nccl":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    comm = build_weight_comm(host, port, rank, world, backend=backend)
    t = torch.arange(4, dtype=torch.float32) if rank == 0 else torch.zeros(4, dtype=torch.float32)
    if dev is not None:
        t = t.to(dev)
    comm.broadcast(t, src=0)
    out_q.put((rank, t.tolist()))
    comm.close()


def _run(backend):
    host, port, world = "127.0.0.1", _free_port(), 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_bcast_worker, args=(r, world, host, port, backend, q)) for r in range(world)]
    for p in procs:
        p.start()
    got = dict(q.get() for _ in range(world))
    for p in procs:
        p.join(timeout=60)
    assert got[0] == [0.0, 1.0, 2.0, 3.0]
    assert got[1] == [0.0, 1.0, 2.0, 3.0]  # rank-1 received rank-0's values


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
@pytest.mark.no_xdist
def test_weight_comm_broadcast_cpu_gloo():
    _run("gloo")


@pytest.mark.L1
@pytest.mark.GPU
def test_weight_comm_broadcast_gpu_nccl():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    _run("nccl")
