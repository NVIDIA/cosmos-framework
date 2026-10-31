# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Distributed update evidence distinguishes logical tensors from padded shards."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.posttrain.tests.qualify_adapters import verify_trainable_updates


def _partial_shard_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        before = {"padded_action.lora_B.weight": torch.zeros(2, 3)}
        after = {name: value.clone() for name, value in before.items()}
        if rank == 0:
            after["padded_action.lora_B.weight"][0, 0] = 0.1
        local, whole = verify_trainable_updates(before, after)
        assert local == {"padded_action.lora_B.weight": rank == 0}
        assert whole == {"padded_action.lora_B.weight": True}
        with pytest.raises(AssertionError, match="did not update on any rank"):
            verify_trainable_updates(before, before)
    finally:
        dist.destroy_process_group()


def test_partial_output_shards_require_a_global_update(tmp_path):
    mp.spawn(_partial_shard_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def test_single_rank_unchanged_tensor_still_fails():
    before = {"adapter": torch.zeros(2, 3)}
    with pytest.raises(AssertionError, match="adapter"):
        verify_trainable_updates(before, before)
    local, whole = verify_trainable_updates(before, {"adapter": torch.ones(2, 3)})
    assert local == whole == {"adapter": True}
