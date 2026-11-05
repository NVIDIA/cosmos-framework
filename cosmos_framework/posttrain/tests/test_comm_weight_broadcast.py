# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.comm.tensor_broadcast import TensorBroadcastSession, build_tensor_broadcast_plan
from cosmos_framework.posttrain.comm.weight_broadcast import (
    CollectiveBroadcast,
    broadcast_live_tensors,
    live_model_tensors,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Model(nn.Module):
    weight: nn.Parameter
    alias: nn.Parameter
    codes: torch.Tensor

    def __init__(self, value: float) -> None:
        super().__init__()
        parameter = torch.full((2, 3), value)  # [2,3]
        self.weight = nn.Parameter(parameter)
        self.alias = self.weight
        buffer_base = torch.full((2, 3), value + 10.0)  # [2,3]
        self.register_buffer("table", buffer_base.t())  # [3,2]
        self.register_buffer("scale", torch.tensor(value + 20.0))  # []
        codes = torch.arange(6, dtype=torch.int32).reshape(2, 3)  # [2,3]
        self.register_buffer("codes", codes.t())  # [3,2]
        self.register_buffer("cache", torch.full((1,), value), persistent=False)  # [1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:  # value: [B,3]
        return value @ self.weight.t()  # [B,2]


class _ReplayState:
    payloads: list[torch.Tensor]

    def __init__(self) -> None:
        self.payloads = []


class _ReplayWork:
    def wait(self) -> bool:
        return True


class _ReplayComm:
    rank: int
    world_size: int
    _state: _ReplayState
    _read_index: int

    def __init__(self, state: _ReplayState, rank: int) -> None:
        self.rank = rank
        self.world_size = 2
        self._state = state
        self._read_index = 0

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        assert src == 0
        if self.rank == src:
            self._state.payloads.append(tensor.clone())  # [*wire]
            return
        tensor.copy_(self._state.payloads[self._read_index])  # [*wire]
        self._read_index += 1

    def submit_broadcast(self, tensor: torch.Tensor, src: int) -> _ReplayWork:  # tensor: [*wire]
        self.broadcast(tensor, src)
        return _ReplayWork()

    def all_reduce_max(self, tensor: torch.Tensor) -> None:  # tensor: [1]
        return None

    def send(self, tensor: torch.Tensor, dst: int) -> None:  # tensor: [*wire]
        raise NotImplementedError

    def recv(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        raise NotImplementedError

    def close(self) -> None:
        return None


def test_live_tensor_broadcast_updates_parameters_buffers_and_noncontiguous_views() -> None:
    source = _Model(3.0)
    follower = _Model(-2.0)
    state = _ReplayState()

    source_digest = broadcast_live_tensors(
        live_model_tensors(source),
        CollectiveBroadcast(_ReplayComm(state, 0)),
        verify_values=True,
    )
    follower_digest = broadcast_live_tensors(
        live_model_tensors(follower),
        CollectiveBroadcast(_ReplayComm(state, 1)),
        verify_values=True,
    )

    assert source_digest == follower_digest
    assert torch.equal(follower.weight, source.weight)
    assert torch.equal(follower.table, source.table)
    assert torch.equal(follower.scale, source.scale)
    assert torch.equal(follower.codes, source.codes)
    assert not torch.equal(follower.cache, source.cache)
    assert not follower.table.is_contiguous()
    assert [item.name for item in live_model_tensors(source)].count("parameter:alias") == 1
    assert len([item for item in live_model_tensors(source) if item.name.startswith("parameter:")]) == 1


def test_tensor_broadcast_session_reuses_plan_and_packs_mixed_dtype_views() -> None:
    source = _Model(3.0)
    follower = _Model(-2.0)
    state = _ReplayState()
    source_session = TensorBroadcastSession(
        live_model_tensors(source),
        CollectiveBroadcast(_ReplayComm(state, 0)),
    )

    source_session.broadcast()
    first_weight = source.weight.detach().clone()  # [2,3]
    first_table = source.table.detach().clone()  # [3,2]
    first_scale = source.scale.detach().clone()  # []
    first_codes = source.codes.detach().clone()  # [3,2]
    with torch.no_grad():
        source.weight.add_(5.0)  # [2,3]
        source.table.add_(7.0)  # [3,2]
        source.scale.add_(9.0)  # []
        source.codes.add_(11)  # [3,2]
    source_session.broadcast()

    follower_session = TensorBroadcastSession(
        live_model_tensors(follower),
        CollectiveBroadcast(_ReplayComm(state, 1)),
    )
    follower_session.broadcast()
    assert torch.equal(follower.weight, first_weight)
    assert torch.equal(follower.table, first_table)
    assert torch.equal(follower.scale, first_scale)
    assert torch.equal(follower.codes, first_codes)
    follower_session.broadcast()
    assert torch.equal(follower.weight, source.weight)
    assert torch.equal(follower.table, source.table)
    assert torch.equal(follower.scale, source.scale)
    assert torch.equal(follower.codes, source.codes)
    assert len(source_session.plan.packed) == 1
    assert {entry.binding.tensor.dtype for entry in source_session.plan.packed[0].entries} == {
        torch.float32,
        torch.int32,
    }
    split_plan = build_tensor_broadcast_plan(live_model_tensors(source), bucket_bytes=32)
    assert [bucket.num_bytes for bucket in split_plan.packed] == [24, 24]
    assert split_plan.max_staging_bytes == 24
    # One manifest broadcast plus three data broadcasts per repeated session call.
    assert len(state.payloads) == 7


def test_live_tensor_broadcast_rejects_corresponding_rank_manifest_drift() -> None:
    source = _Model(3.0)
    follower = nn.Linear(4, 2)
    state = _ReplayState()
    broadcast_live_tensors(live_model_tensors(source), CollectiveBroadcast(_ReplayComm(state, 0)))

    with pytest.raises(RuntimeError, match="live-tensor manifests differ"):
        broadcast_live_tensors(live_model_tensors(follower), CollectiveBroadcast(_ReplayComm(state, 1)))
