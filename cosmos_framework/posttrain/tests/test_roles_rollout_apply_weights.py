# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Weight-sync correctness (spec §7.1), HF-free: rollout starts from RANDOM and is populated by the
sync alone; after a policy update, a second sync propagates it. A single shared loopback WeightComm
composes producer export and rollout apply in one process (the real NCCL path is the GPU e2e)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.comm.transfer.movers import NcclMover
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync
from cosmos_framework.posttrain.roles.rollout.hf import HFRollout


class _StubPacker:
    """Minimal stand-in for the DataPacker on the WEIGHT-ONLY path.

    `HFRollout` requires a packer at construction (it is the prompt-shaping seam), but this module
    exercises the rollout purely as a weight RECEIVER — it never calls `generate`, so the packer's
    tokenizer is never touched. Importing the real `init_data_packer` here would pull
    `transformers` at module scope (`data/packer/__init__.py` does `from transformers import
    AutoConfig`), turning a formerly HF-free L0/CPU module into one that fails COLLECTION wherever
    that optional dependency is absent. The stub keeps this module dependency-free."""

    tokenizer = None


class _LoopComm:
    """Records broadcasts on the producer pass (rank==src) and replays them on the consumer pass."""

    rank = 0
    world_size = 2

    def __init__(self):
        self._buf, self._i = [], 0

    def broadcast(self, t, src):
        if self.rank == src:
            self._buf.append(t.clone())
        else:
            t.copy_(self._buf[self._i])
            self._i += 1

    def send(self, *a): ...
    def recv(self, *a): ...
    def close(self): ...


def _sync(policy, rollout, comm):
    """Drive export ∥ apply over ONE shared loopback comm: producer stage records (rank 0), rollout
    apply replays (rank 1). apply is called with NO payload — exactly the concurrent COLLECTIVE path."""
    comm._buf, comm._i = [], 0
    comm.rank = 0
    WeightSync(NcclMover(comm, src=0)).export(policy, version=1)
    comm.rank = 1
    # Body surface is peer-only; the whole-model collective recv lives on the BACKEND layer
    # (recv layout self-derived from the rollout's own model).
    rollout._weight_backend.apply(rollout.model)


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_weights_from_random_then_after_update():
    torch.manual_seed(0)
    policy = nn.Linear(4, 3)
    rollout_net = nn.Linear(4, 3)
    nn.init.normal_(rollout_net.weight, std=5.0)  # RANDOM, deliberately far from policy
    # Packer is REQUIRED at construction now; weight-only rig (never generates) → gpt2 stands in.
    rollout = HFRollout(rollout_net, packer=_StubPacker())

    comm = _LoopComm()
    rollout._weight_backend = WeightSync(NcclMover(comm, src=0))  # bound sink shares the loopback comm

    assert not torch.allclose(rollout_net.weight, policy.weight)
    _sync(policy, rollout, comm)
    assert torch.equal(rollout_net.weight, policy.weight)  # populated by sync alone (kills no-op)

    with torch.no_grad():  # simulate one training step: policy weights genuinely change
        policy.weight += 1.0
    _sync(policy, rollout, comm)
    assert torch.equal(rollout_net.weight, policy.weight)  # sync propagates the UPDATE


@pytest.mark.L0
@pytest.mark.CPU
def test_prepare_recv_reports_whole_tensor_layout():
    from cosmos_framework.posttrain.roles.rollout.base import ShardSpec

    # Packer is REQUIRED at construction now; weight-only rig (never generates) → gpt2 stands in.
    rollout = HFRollout(nn.Linear(4, 3), packer=_StubPacker())
    layout = rollout.prepare_recv()
    assert layout["weight"] == ShardSpec(offset=0, length=3, dim=0)  # out_features rows, whole, dim 0
    assert layout["bias"] == ShardSpec(0, 3, 0)
