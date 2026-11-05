# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Declared topology must determine real node reservations before model construction."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from cosmos_framework.posttrain.config.rl import GridConfig
from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups, colocate, topology
from cosmos_framework.posttrain.control.placement.topology import (
    GpuTopologyNode,
    plan_gpu_topology,
    wait_for_gpu_topology,
)
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _node(index: int, *, gpus: float = 4.0) -> dict[str, Any]:
    address = f"10.0.0.{index}"
    return {
        "Alive": True,
        "NodeID": f"ray-{index}",
        "NodeManagerAddress": address,
        "NodeManagerHostname": f"worker-{index}",
        "Resources": {"GPU": gpus, "CPU": 8.0, f"node:{address}": 1.0},
    }


def _plan(
    nodes: list[dict[str, Any]], domains: dict[str, str], *, shard: int = 4, expected: int = 8
) -> tuple[GpuTopologyNode, ...]:
    return plan_gpu_topology(nodes, domains, shard_degree=shard, expected_gpus=expected, cpus_per_gpu=2)


def test_domain_order_keeps_each_shard_group_within_declared_fabric() -> None:
    nodes = [_node(index) for index in (4, 1, 3, 2)]
    domains = {"worker-4": "rack-b", "worker-1": "rack-a", "worker-3": "rack-a", "worker-2": "rack-b"}
    plan = _plan(nodes, domains, shard=8, expected=16)
    assert [node.address for node in plan] == ["10.0.0.1", "10.0.0.3", "10.0.0.2", "10.0.0.4"]
    rank_domains = [node.domain for node in plan for _ in range(node.gpus)]
    assert all(len(set(rank_domains[start : start + 8])) == 1 for start in range(0, 16, 8))
    assert _plan(list(reversed(nodes)), dict(reversed(list(domains.items()))), shard=8, expected=16) == plan


@pytest.mark.parametrize(
    ("domains", "message"),
    [
        ({"worker-1": "a"}, "omits live GPU nodes"),
        ({"worker-1": "a", "10.0.0.1": "a", "worker-2": "b"}, "more than once"),
        ({"worker-1": "a", "typo": "b"}, "omits live GPU nodes"),
        ({"worker-1": "a", "worker-2": ""}, "nonempty strings"),
        ({"worker-1": "a", " worker-2": "b"}, "edge whitespace"),
    ],
)
def test_topology_rejects_partial_duplicate_or_malformed_mapping(domains: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _plan([_node(1), _node(2)], domains)


def test_topology_rejects_ambiguous_hostname() -> None:
    nodes = [_node(1), _node(2)]
    nodes[1]["NodeManagerHostname"] = "worker-1"
    with pytest.raises(ValueError, match="multiple live Ray GPU nodes"):
        _plan(nodes, {"worker-1": "a", "10.0.0.2": "b"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("GPU", 0.5, "invalid physical GPU capacity"),
        ("GPU", 2.0, "equal physical GPU counts"),
        ("CPU", 1.0, "needs 8 CPUs"),
        ("node:10.0.0.2", 0.0, "lacks its node-affinity resource"),
    ],
)
def test_topology_rejects_incomplete_node_resources(field: str, value: float, message: str) -> None:
    nodes = [_node(1), _node(2)]
    nodes[1]["Resources"][field] = value
    with pytest.raises(ValueError, match=message):
        _plan(nodes, {"worker-1": "a", "worker-2": "b"})


def test_topology_rejects_shard_groups_crossing_domain_boundaries() -> None:
    with pytest.raises(ValueError, match="not divisible by policy shard degree 8"):
        _plan([_node(1), _node(2)], {"worker-1": "a", "worker-2": "b"}, shard=8)
    with pytest.raises(ValueError, match="has 8 GPUs.*needs 4"):
        _plan([_node(1), _node(2)], {"worker-1": "a", "worker-2": "b"}, expected=4)


def test_registration_wait_accepts_workers_joining_after_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = iter([[], [_node(1)], [_node(2), _node(1)]])
    sleeps: list[float] = []
    monkeypatch.setattr(topology.time, "sleep", sleeps.append)
    plan = wait_for_gpu_topology(
        lambda: next(snapshots), {"worker-1": "a", "worker-2": "b"}, shard_degree=4, expected_gpus=8, cpus_per_gpu=2
    )
    assert len(plan) == 2
    assert len(sleeps) == 2


def test_registration_wait_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 300.0])
    monkeypatch.setattr(topology.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="not ready within 300s.*worker-1"):
        wait_for_gpu_topology(lambda: [], {"worker-1": "a"}, shard_degree=4, expected_gpus=4, cpus_per_gpu=2)


def test_grid_topology_requires_ray_colocated_placement() -> None:
    assert GridConfig().gpu_node_domains == {}
    for options in ({}, {"backend": "ray", "gpu_placement": "disaggregated"}):
        with pytest.raises(ValueError, match="requires Ray colocated"):
            GridConfig(gpu_node_domains={"worker-1": "a"}, **options)
    assert GridConfig(backend="ray", gpu_node_domains={"worker-1": "a"}).gpu_node_domains == {"worker-1": "a"}


def test_binder_pins_domain_order_before_actor_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    reservations: list[list[dict[str, float]]] = []
    policy = WorkerGroup("policy", "trainer", replicas=1, ranks=8, model_parallelism=(2, 4))
    rollout = WorkerGroup("rollout", "rollout", replicas=2, ranks=4)
    device = DeviceGroup(
        (ActorGroup((policy,), lambda: {}, name="policy"), ActorGroup((rollout,), lambda: {}, name="rollout")),
        device="gpu",
        name="shared",
    )

    def reserve(bundles: list[dict[str, float]], *, strategy: str) -> object:
        assert strategy == "STRICT_PACK"
        reservations.append(bundles)
        return object()

    class RendezvousReached(Exception):
        pass

    def spawn(*args: Any, **kwargs: Any) -> dict:
        assert kwargs["bundles_per_pg"] == 4
        assert len(kwargs["pgs"]) == 2
        raise RendezvousReached

    ray = types.ModuleType("ray")
    ray.nodes = lambda: [_node(1), _node(2)]
    pg = types.ModuleType("ray.util.placement_group")
    pg.placement_group = reserve
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", pg)
    monkeypatch.setattr(colocate, "_ensure_ray", lambda *args: None)
    monkeypatch.setattr(colocate, "_declared_allocation_gpus", lambda: None)
    monkeypatch.setattr(colocate, "_ray_capacity_is_complete", lambda: False)
    monkeypatch.setattr(colocate, "_wait_placement_groups", lambda *args, **kwargs: None)
    monkeypatch.setattr(colocate, "_cleanup_ray_bind", lambda *args: None)
    from cosmos_framework.posttrain.workers import grid

    monkeypatch.setattr(grid, "spawn_grid", spawn)
    with pytest.raises(RendezvousReached):
        bind_device_groups([device], gpu_node_domains={"worker-1": "z", "worker-2": "a"})
    assert len(reservations) == 2
    for bundles, address in zip(reservations, ("10.0.0.2", "10.0.0.1"), strict=True):
        assert len(bundles) == 4
        assert all(bundle == {"GPU": 1.0, "CPU": 2, f"node:{address}": 0.0001} for bundle in bundles)


def test_explicit_topology_refuses_rollout_tp_crossing_nodes() -> None:
    with pytest.raises(ValueError, match="not divisible by rollout TP degree 8"):
        plan_gpu_topology(
            [_node(1), _node(2)],
            {"worker-1": "a", "worker-2": "a"},
            shard_degree=4,
            expected_gpus=8,
            cpus_per_gpu=2,
            rollout_degree=8,
        )


def test_sft_refuses_ignored_topology_before_source_or_process_setup() -> None:
    from cosmos_framework.posttrain.entrypoints.sft import validate_config

    config = types.SimpleNamespace(grid=types.SimpleNamespace(gpu_node_domains={"worker-1": "rack-a"}))
    with pytest.raises(ValueError, match="does not support grid.gpu_node_domains"):
        validate_config(config)
