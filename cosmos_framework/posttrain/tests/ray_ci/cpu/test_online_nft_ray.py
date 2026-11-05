# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Two-rank Ray routing smoke for the colocated online-NFT vertical."""

from __future__ import annotations

from typing import Any

import pytest

ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.config.rl import NFTConfig
from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
from cosmos_framework.posttrain.entrypoints.nft import online_nft_step
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.tests.helpers.online_nft_ray import setup_online_nft_bodies
from cosmos_framework.posttrain.workers import grid as grid_module


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_routes_complete_transaction_across_ray_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    spawn_grid = grid_module.spawn_grid

    def spawn_grid_with_cpu_gloo(*args: Any, **kwargs: Any) -> Any:
        """Force the model-free CPU fixture into the replica-scoped Gloo world used by the test."""
        kwargs["init_dist"] = True
        return spawn_grid(*args, **kwargs)

    monkeypatch.setattr(grid_module, "spawn_grid", spawn_grid_with_cpu_gloo)
    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir="/tmp/cosmos3-online-nft-ray-cpu",
    )
    try:
        units = {
            "data": roles.env("data", replicas=1, ranks=2, dispatch_profile="nft"),
            "rollout": roles.rollout("rollout", replicas=1, ranks=2, dispatch_profile="nft"),
            "rm": roles.reward("rm", replicas=1, ranks=2, needs_bulk=True, dispatch_profile="nft"),
            "replay": roles.buffer("replay", replicas=1, ranks=2, dispatch_profile="nft"),
            "policy": roles.trainer(
                "policy",
                replicas=1,
                ranks=2,
                model_parallelism=(1, 2),
                dispatch_profile="nft",
            ),
        }
        actor_group = ActorGroup(tuple(units.values()), setup_online_nft_bodies, name="online_nft_test")
        bind_device_groups(
            [DeviceGroup((actor_group,), device="cpu", name="online_nft_test")],
            backend="ray",
            threshold=0,
            weight_sync=True,
        )
        # Ray initializes a two-rank Gloo world, but the toy model is not FSDP-sharded. This proves
        # routed two-rank execution, scalar reward gathering, and transactional clock convergence;
        # it does not prove production FSDP gradient synchronization or model capacity.
        metrics = online_nft_step(units, NFTConfig(max_steps=1, group_size=2), step=0)
        assert metrics["optim/update_outcome"] == "applied"
        assert metrics["train/reward_provider/dance_grpo/sample_count"] == 2
        assert metrics["train/reward_provider/dance_grpo/raw_score"] == pytest.approx(0.5)
        assert metrics["rollout/reward_group_mean"] == pytest.approx(0.5)
        assert metrics["rollout/reward_group_spread"] == pytest.approx(1.0)
        assert metrics["rollout/reward_group_size"] == 2
        assert metrics["rollout/reward_group_informative"] == 1
        iterations = units["policy"].iteration().all()
        cursors = units["data"].state_dict().all()
        assert len(iterations) == 2
        assert len(cursors) == 2
        assert set(iterations.values()) == {1}
        assert {state["committed_rows"] for state in cursors.values()} == {1}
        assert {state["optimizer_step"] for state in cursors.values()} == {1}
    finally:
        ray.shutdown()
