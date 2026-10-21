# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actor setup publishes the current K-rank world through the ``set_mesh`` seam.

`run_setup` must, AFTER the Pass-2 rendezvous and BEFORE setup_fn, hand each (r, k) actor a
ParallelDims for its execution replica's K-rank world. Without an explicit trainer
``model_parallelism`` this compatibility view is pure FSDP ``(dp_replicate=1, dp_shard=K)``; policy
HSDP is covered separately by ``test_control_placement_hsdp_mesh``. We read the injection back per
actor via `_Host.pdims`; the readback is mesh-free so the live DeviceMesh never serializes to the
driver.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
ray = pytest.importorskip("ray")


def _noop_setup():
    # A trivial body: the mesh is held on the ACTOR (injected before bodies exist), so the body
    # itself is irrelevant to what we assert — it just satisfies run_setup's {name: body} contract.
    class W:
        def ping(self):
            return "ok"

    return {"w": W()}


@pytest.mark.L1
@pytest.mark.CPU
def test_grid_injects_parallel_dims():
    from cosmos_framework.posttrain.workers.grid import spawn_grid

    ray.init(ignore_reinit_error=True, num_cpus=6, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        # 2x2 CPU mesh with an explicit rendezvous so Pass 2.5 fires (do_init and world>1).
        grid = spawn_grid(
            _noop_setup,
            replicas=2,
            ranks=2,
            group_id="mesh",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            init_dist=True,
        )
        assert set(grid) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        # Each execution replica owns its own K-rank process group. With no trainer mesh spec the
        # injected compatibility view is pure FSDP over K; placement R is never dp_replicate.
        for rk, (_sid, host) in grid.items():
            pd = ray.get(host.pdims.remote())
            assert pd is not None, f"cell {rk} missing injected pdims"
            assert pd.dp_replicate == 1, f"cell {rk} dp_replicate={pd.dp_replicate}"
            assert pd.dp_shard == 2, f"cell {rk} dp_shard={pd.dp_shard}"
            assert pd.world_size == 2, f"cell {rk} world_size={pd.world_size}"
    finally:
        ray.shutdown()
