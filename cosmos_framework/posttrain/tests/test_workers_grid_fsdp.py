# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU test: NCCL rendezvous across Ray `_Host` actors (DESIGN §1.1 FSDP).

Locks the framework piece that lets the grid run real FSDP (vs the rank-0 mirror): a
torch.distributed group formed ACROSS Ray actors via `_Host.init_distributed`, verified
by an all-reduce. Model-free + fast, so it guards the rendezvous independently of the
heavy 8B grid run (`train.py` with `grid.backend=ray`). Driven by Ray
(not torchrun): the test process is the driver, the actors are the ranks.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.workers.ray import _Host  # noqa: E402


@pytest.mark.L1
@pytest.mark.GPU
def test_nccl_rendezvous_across_ray_actors():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    world = 2
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        actors = [_Host.options(num_gpus=1).remote() for _ in range(world)]
        addr, port = ray.get(actors[0].rendezvous_endpoint.remote())
        # init must be launched concurrently — init_process_group blocks until all join.
        ranks = ray.get([a.init_distributed.remote(addr, port, r, world) for r, a in enumerate(actors)])
        assert sorted(ranks) == list(range(world))
        # all_reduce(sum of ranks) must equal 0+1+...+(world-1) on EVERY rank.
        vals = ray.get([a.dist_all_reduce_probe.remote() for a in actors])
        assert vals == [float(world * (world - 1) // 2)] * world
    finally:
        ray.shutdown()
