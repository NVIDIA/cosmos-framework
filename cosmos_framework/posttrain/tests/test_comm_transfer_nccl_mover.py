# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NcclMover stage(broadcast-send in sorted order)/fetch(broadcast-recv per the CONSUMER's own
recv_plan) compose to identity over a loopback comm — no manifest crosses, so no push->apply data
dependency. resolve_mover wires the comm. WeightSync.apply drives the COLLECTIVE path end to end."""

from __future__ import annotations

import pytest
import torch


class _LoopComm:
    """Records broadcasts on the producer pass and replays them on the consumer pass (same process,
    src rank == this rank), so stage()+fetch() compose without a real NCCL group."""

    rank = 0
    world_size = 2

    def __init__(self):
        self._buf, self._i = [], 0

    def broadcast(self, tensor, src):
        if self.rank == src:
            self._buf.append(tensor.clone())
        else:
            tensor.copy_(self._buf[self._i])
            self._i += 1

    def send(self, *a): ...
    def recv(self, *a): ...
    def close(self): ...


@pytest.mark.L0
@pytest.mark.CPU
def test_nccl_mover_stage_fetch_identity():
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover

    comm = _LoopComm()
    state = {"b": torch.ones(2, 2), "a": torch.arange(3, dtype=torch.float32)}  # unsorted on purpose

    comm.rank = 0  # producer sends in SORTED order: a, then b
    handle = NcclMover(comm, src=0).stage(state)
    assert [m[0] for m in handle.meta["plan"]] == ["a", "b"]  # sorted

    comm.rank = 1  # consumer recvs per its own recv_plan (sorted), replaying recorded broadcasts
    mv = NcclMover(comm, src=0)
    mv.recv_plan = [("a", (3,), "torch.float32"), ("b", (2, 2), "torch.float32")]
    out = mv.fetch(None)
    assert torch.equal(out["a"], state["a"]) and torch.equal(out["b"], state["b"])


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_mover_wires_comm():
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover, resolve_mover

    mv = resolve_mover(
        relationship="cross_device", plane="weight", capability="tensor", hint="collective", comm=_LoopComm()
    )
    assert isinstance(mv, NcclMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_weightsync_apply_collective_end_to_end():
    """WeightSync.apply builds recv_plan from the CONSUMER model and loads the broadcast weights."""
    import torch.nn as nn

    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync

    comm = _LoopComm()
    producer = nn.Linear(4, 3)
    consumer = nn.Linear(4, 3)
    nn.init.zeros_(consumer.weight)
    nn.init.zeros_(consumer.bias)
    assert not torch.equal(consumer.weight, producer.weight)

    comm.rank = 0  # export (producer stage broadcasts producer.state_dict() sorted)
    payload = WeightSync(NcclMover(comm, src=0)).export(producer, version=1)
    comm.rank = 1  # apply (consumer builds recv_plan from its own model, recvs, loads)
    WeightSync(NcclMover(comm, src=0)).apply(consumer, payload)
    assert torch.equal(consumer.weight, producer.weight) and torch.equal(consumer.bias, producer.bias)
