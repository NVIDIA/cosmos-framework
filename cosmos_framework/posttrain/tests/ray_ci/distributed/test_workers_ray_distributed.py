# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-node distributed smoke tests for the post-training Ray substrate."""

from __future__ import annotations

from typing import Any

import pytest

ray = pytest.importorskip("ray")


@pytest.mark.L0
@pytest.mark.GPU
def test_ray_actors_join_nccl_group_and_all_reduce(ray_two_gpu_hosts: list[Any]) -> None:
    """Two GPU actors on one Ray node must form a real NCCL process group."""
    world_size = len(ray_two_gpu_hosts)

    node_ids = ray.get([host.node_id.remote() for host in ray_two_gpu_hosts])
    assert len(set(node_ids)) == 1

    visible_gpus = ray.get([host.gpu.remote() for host in ray_two_gpu_hosts])
    assert all(gpu not in {"", "?"} for gpu in visible_gpus)
    assert len(set(visible_gpus)) == world_size

    master_addr, master_port = ray.get(ray_two_gpu_hosts[0].rendezvous_endpoint.remote())
    ranks = ray.get(
        [
            host.init_distributed.remote(master_addr, master_port, rank, world_size)
            for rank, host in enumerate(ray_two_gpu_hosts)
        ]
    )
    assert ranks == list(range(world_size))

    reduced = ray.get([host.dist_all_reduce_probe.remote() for host in ray_two_gpu_hosts])
    expected_sum = float(sum(range(world_size)))
    assert reduced == pytest.approx([expected_sum] * world_size)
