# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observe final optimizer gradients under both FSDP2 data-parallel layouts."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _gradient_observer_worker(rank, world, replicate, path):
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
    from cosmos_framework.posttrain.tests.qualify_adapters import local_value, observe_optimizer_gradients

    init_gloo_process_group(rank, world, path)
    try:
        torch.manual_seed(42)
        net = torch.nn.Linear(4, 4)
        dims = ParallelDims(dp_replicate=replicate, dp_shard=world // replicate, world_size=world)
        dims.build_device_mesh("cpu")
        parallelize(net, ShardPlan((), None, torch.float32), dims)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0)
        before = {name: local_value(p).clone() for name, p in net.named_parameters()}
        records, handle = observe_optimizer_gradients(net, optimizer)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            net(torch.full((2, 4), float(rank + 1))).square().mean().backward()
            optimizer.step()
        handle.remove()
        assert all(row["observed"] and row["finite"] and row["nonzero_local"] for row in records.values()), records
        assert all(not torch.equal(before[name], local_value(p)) for name, p in net.named_parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("replicate", "shard"), [(1, 2), (2, 1), (2, 2)])
def test_gradient_observer_uses_final_reduced_parameters(tmp_path, replicate, shard):
    world = replicate * shard
    mp.spawn(_gradient_observer_worker, args=(world, replicate, str(tmp_path / "rendezvous")), nprocs=world, join=True)
