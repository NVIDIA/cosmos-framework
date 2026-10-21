# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for explicit actor and device grouping."""

from __future__ import annotations

import inspect
import sys
import types
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Literal

import pytest

from cosmos_framework.posttrain.control import placement
from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
from cosmos_framework.posttrain.control.placement.colocate import (
    _align_host_affinity_grids_by_uuid,
    _bundles_per_placement_group,
    _declared_allocation_gpus,
    _ray_capacity_is_complete,
    _resolve_device_groups,
    _resolve_weight_edge,
    _SpawnedActorGroup,
    _validate_declared_gpu_capacity,
    _validate_gpu_identities,
    _validate_host_affinity_resources,
    _validate_peer_gpu_exposure,
    _validate_registered_gpu_capacity,
)
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _empty_setup() -> dict[str, object]:
    return {}


def _worker(name: str, role: str, shape: tuple[int, int]) -> WorkerGroup:
    return WorkerGroup(name, role, replicas=shape[0], ranks=shape[1])


def _actor(name: str, *workers: WorkerGroup) -> ActorGroup:
    return ActorGroup(members=tuple(workers), setup=_empty_setup, name=name)


def _device(name: str, device: Literal["cpu", "gpu"], *actor_groups: ActorGroup) -> DeviceGroup:
    return DeviceGroup(actor_groups=tuple(actor_groups), device=device, name=name)


class _FakePlacement:
    def __init__(self, shapes: dict[str, tuple[int, int]] | None = None) -> None:
        self.shapes = shapes or {}
        self.calls: list[str] = []

    def resolve(self, wg: WorkerGroup) -> dict[str, int]:
        self.calls.append(wg.name)
        replicas, ranks = self.shapes.get(wg.name, (wg.replicas, wg.ranks))
        return {"replicas": replicas, "ranks": ranks}


def test_actor_and_device_groups_are_frozen_and_tuple_backed() -> None:
    worker = _worker("worker", "env", (1, 1))
    actor_group = _actor("actor", worker)
    device_group = _device("device", "cpu", actor_group)

    assert actor_group.members == (worker,)
    assert isinstance(actor_group.members, tuple)
    assert device_group.actor_groups == (actor_group,)
    assert isinstance(device_group.actor_groups, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(actor_group, "members", ())
    with pytest.raises(FrozenInstanceError):
        setattr(device_group, "actor_groups", ())


def test_actor_group_keeps_peer_gpus_hidden_unless_asked_with_a_bool() -> None:
    worker = _worker("rollout", "rollout", (1, 4))

    assert _actor("isolated", worker).expose_peer_gpus is False
    assert ActorGroup(members=(worker,), setup=_empty_setup, expose_peer_gpus=True).expose_peer_gpus is True
    with pytest.raises(TypeError, match="expose_peer_gpus must be a bool"):
        ActorGroup(members=(worker,), setup=_empty_setup, expose_peer_gpus=1)  # type: ignore[arg-type]


def test_peer_gpu_exposure_requires_one_full_node_tp_replica() -> None:
    worker = _worker("rollout", "rollout", (2, 4))
    actor = ActorGroup(members=(worker,), setup=_empty_setup, name="rollout", expose_peer_gpus=True)
    (resolved_device,) = _resolve_device_groups([_device("gpus", "gpu", actor)], _FakePlacement())
    resolved = resolved_device.actor_groups

    _validate_peer_gpu_exposure(resolved, colocate="pg", bundles_per_pg=4, node_gpu_counts=(4, 4))
    with pytest.raises(ValueError, match="ranks must equal.*capacity"):
        _validate_peer_gpu_exposure(resolved, colocate="pg", bundles_per_pg=8, node_gpu_counts=(8,))
    with pytest.raises(ValueError, match="candidate GPU counts include"):
        _validate_peer_gpu_exposure(resolved, colocate="pg", bundles_per_pg=4, node_gpu_counts=(4, 8))
    with pytest.raises(ValueError, match="cannot prove"):
        _validate_peer_gpu_exposure(
            resolved,
            colocate="host_affinity",
            bundles_per_pg=None,
            node_gpu_counts=None,
        )


def test_actor_group_accepts_a_real_member_with_the_dominant_shape() -> None:
    dominant = _worker("dominant", "rollout", (2, 4))
    helper = _worker("helper", "env", (2, 1))
    placement = _FakePlacement()

    (resolved_device,) = _resolve_device_groups(
        [_device("cpu_helpers", "cpu", _actor("rollout_process", dominant, helper))],
        placement,
    )
    (resolved_actor,) = resolved_device.actor_groups

    assert resolved_actor.shape == (2, 4)
    assert resolved_actor.cells == 8
    assert resolved_actor.worker_shapes == (("dominant", 2, 4), ("helper", 2, 1))
    assert placement.calls == ["dominant", "helper"]


def test_model_parallelism_is_available_to_new_model_roles() -> None:
    reference = _worker("reference", "reference", (1, 2))
    reference.model_parallelism = (1, 2)

    (resolved_device,) = _resolve_device_groups(
        [_device("reference_gpus", "gpu", _actor("reference_process", reference))],
        _FakePlacement(),
    )

    assert resolved_device.actor_groups[0].model_parallelism == (1, 2)


def test_actor_group_rejects_a_sparse_cartesian_product_without_a_dominant_member() -> None:
    replicas_only = _worker("replicas_only", "env", (64, 1))
    ranks_only = _worker("ranks_only", "reward", (1, 4))

    with pytest.raises(ValueError, match="dominant"):
        _resolve_device_groups(
            [_device("cpu_helpers", "cpu", _actor("sparse_process", replicas_only, ranks_only))],
            _FakePlacement(),
        )


@pytest.mark.parametrize("axis,value", [("replicas", 0), ("ranks", -1), ("replicas", True), ("ranks", 1.5)])
def test_actor_group_rejects_invalid_resolved_axes(axis: str, value: object) -> None:
    worker = _worker("worker", "env", (1, 1))

    class _InvalidPlacement:
        @staticmethod
        def resolve(_worker: WorkerGroup) -> dict[str, object]:
            shape: dict[str, object] = {"replicas": 1, "ranks": 1}
            shape[axis] = value
            return shape

    with pytest.raises(ValueError, match=f"invalid {axis}"):
        _resolve_device_groups(
            [_device("cpu", "cpu", _actor("actor", worker))],
            _InvalidPlacement(),
        )


def test_gpu_device_group_aligns_equal_products_by_flat_actor_index() -> None:
    two_by_four = _actor("two_by_four", _worker("a", "env", (2, 4)))
    four_by_two = _actor("four_by_two", _worker("b", "reward", (4, 2)))

    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpu_pool", "gpu", two_by_four, four_by_two)],
        _FakePlacement(),
    )

    assert tuple(actor.shape for actor in resolved_device.actor_groups) == ((2, 4), (4, 2))
    assert tuple(actor.cells for actor in resolved_device.actor_groups) == (8, 8)
    assert {actor.device_group_name for actor in resolved_device.actor_groups} == {"shared_gpu_pool"}


def test_gpu_device_group_rejects_unequal_actor_products() -> None:
    eight_cells = _actor("eight_cells", _worker("a", "env", (2, 4)))
    four_cells = _actor("four_cells", _worker("b", "reward", (1, 4)))

    with pytest.raises(ValueError, match="actor product"):
        _resolve_device_groups(
            [_device("invalid_shared_gpu_pool", "gpu", eight_cells, four_cells)],
            _FakePlacement(),
        )


def test_cpu_device_group_permits_unequal_actor_products() -> None:
    eight_cells = _actor("eight_cells", _worker("a", "env", (2, 4)))
    three_cells = _actor("three_cells", _worker("b", "reward", (3, 1)))

    (resolved_device,) = _resolve_device_groups(
        [_device("cpu_pool", "cpu", eight_cells, three_cells)],
        _FakePlacement(),
    )

    assert tuple(actor.cells for actor in resolved_device.actor_groups) == (8, 3)


def test_resolver_rejects_duplicate_device_and_actor_group_names() -> None:
    first = _actor("actor", _worker("first", "env", (1, 1)))
    second = _actor("actor", _worker("second", "reward", (1, 1)))
    with pytest.raises(ValueError, match="ActorGroup names must be unique"):
        _resolve_device_groups([_device("device", "cpu", first, second)], _FakePlacement())

    left = _device("device", "cpu", _actor("left", _worker("left", "env", (1, 1))))
    right = _device("device", "cpu", _actor("right", _worker("right", "reward", (1, 1))))
    with pytest.raises(ValueError, match="DeviceGroup names must be unique"):
        _resolve_device_groups([left, right], _FakePlacement())


def test_resolver_rejects_duplicate_worker_identity_and_name() -> None:
    shared = _worker("shared", "env", (1, 1))
    with pytest.raises(ValueError, match="cannot belong to more than one ActorGroup"):
        _resolve_device_groups(
            [_device("cpu", "cpu", _actor("first", shared), _actor("second", shared))],
            _FakePlacement(),
        )

    first = _worker("duplicate", "env", (1, 1))
    second = _worker("duplicate", "reward", (1, 1))
    with pytest.raises(ValueError, match="WorkerGroup names must be unique"):
        _resolve_device_groups(
            [_device("cpu", "cpu", _actor("first", first), _actor("second", second))],
            _FakePlacement(),
        )


def test_separate_gpu_device_groups_keep_disjoint_64_plus_4_plus_8_domains() -> None:
    policy = _actor("policy_actor", _worker("policy", "trainer", (1, 64)))
    rollout = _actor("rollout_actor", _worker("rollout", "rollout", (1, 4)))
    teacher = _actor("teacher_actor", _worker("teacher", "teacher", (1, 8)))

    resolved = _resolve_device_groups(
        [
            _device("policy_gpus", "gpu", policy),
            _device("rollout_gpus", "gpu", rollout),
            _device("teacher_gpus", "gpu", teacher),
        ],
        _FakePlacement(),
    )

    domain_sizes = tuple(max(actor.cells for actor in group.actor_groups) for group in resolved)
    assert tuple(group.name for group in resolved) == ("policy_gpus", "rollout_gpus", "teacher_gpus")
    assert tuple(group.actor_groups[0].device_group_name for group in resolved) == (
        "policy_gpus",
        "rollout_gpus",
        "teacher_gpus",
    )
    assert domain_sizes == (64, 4, 8)
    assert sum(domain_sizes) == 76


def test_placement_group_size_respects_gpu_and_cpu_density() -> None:
    first = _actor("first", _worker("first_wg", "env", (1, 1)))
    second = _actor("second", _worker("second_wg", "reward", (1, 1)))
    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpus", "gpu", first, second)],
        _FakePlacement(),
    )
    nodes = [
        {"NodeID": "cpu_limited", "Resources": {"GPU": 8.0, "CPU": 4.0}},
        {"NodeID": "balanced", "Resources": {"GPU": 8.0, "CPU": 32.0}},
    ]

    assert _bundles_per_placement_group(nodes, resolved_device) == 2


def test_placement_group_preflight_rejects_nodes_without_cpu_for_one_bundle() -> None:
    actor_groups = tuple(_actor(f"actor_{index}", _worker(f"worker_{index}", "env", (1, 1))) for index in range(3))
    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpus", "gpu", *actor_groups)],
        _FakePlacement(),
    )
    nodes = [
        {"NodeID": "left", "Resources": {"GPU": 8.0, "CPU": 2.0}},
        {"NodeID": "right", "Resources": {"GPU": 4.0, "CPU": 1.0}},
    ]

    with pytest.raises(RuntimeError, match="requires 3 CPUs per one-GPU placement bundle"):
        _bundles_per_placement_group(nodes, resolved_device)


def test_registered_gpu_capacity_rejects_disjoint_pools_before_pg_creation() -> None:
    class _RayResources:
        @staticmethod
        def cluster_resources() -> dict[str, float]:
            return {"GPU": 8.0}

    _validate_registered_gpu_capacity(_RayResources(), 8.0)
    with pytest.raises(RuntimeError, match="require 9 disjoint physical GPUs.*only 8 registered GPUs"):
        _validate_registered_gpu_capacity(_RayResources(), 9.0)


def test_declared_allocation_is_read_only_from_a_complete_pair() -> None:
    assert _declared_allocation_gpus({"NNODES": "19", "GPUS_PER_NODE": "8"}) == (
        152.0,
        "NNODES=19 x GPUS_PER_NODE=8",
    )
    assert _declared_allocation_gpus({}) is None
    # launch_multinode_torchrun.sh exports GPUS_PER_NODE and never NNODES. Half a declaration must
    # read as "not declared" rather than failing every torchrun job.
    assert _declared_allocation_gpus({"GPUS_PER_NODE": "8"}) is None
    assert _declared_allocation_gpus({"NNODES": "19"}) is None
    assert _declared_allocation_gpus({"NNODES": "two", "GPUS_PER_NODE": "8"}) is None
    assert _declared_allocation_gpus({"NNODES": "0", "GPUS_PER_NODE": "8"}) is None


def test_declared_gpu_capacity_rejects_oversized_topology_before_ray_starts() -> None:
    # The K + T rule: colocated OPD puts the student pair on 8 GPUs and the frozen teacher on its
    # own disjoint GPU, so a single 8-GPU node cannot host it.
    _validate_declared_gpu_capacity(8.0, 8.0, "NNODES=1 x GPUS_PER_NODE=8")
    with pytest.raises(
        RuntimeError,
        match="require 9 disjoint physical GPUs.*provides 8 .NNODES=1 x GPUS_PER_NODE=8.",
    ):
        _validate_declared_gpu_capacity(9.0, 8.0, "NNODES=1 x GPUS_PER_NODE=8")


def test_joined_ray_cluster_is_not_treated_as_a_settled_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    # A joined cluster is still gaining workers, so its registered GPU count is a lower bound. Using
    # it as a verdict kills valid multi-node launches whose workers have not registered yet.
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")
    assert _ray_capacity_is_complete() is False
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    assert _ray_capacity_is_complete() is True


class _RayStarted(RuntimeError):
    """Raised in place of ``_ensure_ray`` to prove a guard ran before Ray was touched."""


class _LiveCapacityChecked(RuntimeError):
    """Raised in place of the registered-GPU guard to prove it was consulted."""


class _ReachedNodeScan(RuntimeError):
    """Raised from ``ray.nodes`` to prove both capacity guards were passed."""


def _capacity_probe(monkeypatch: pytest.MonkeyPatch, *, ray_started: bool) -> list[DeviceGroup]:
    """Wire bind_device_groups so each capacity decision surfaces as a distinct exception.

    A stub stands in for ``ray`` so these stay CPU-only contracts: the capacity decisions under test
    all happen before any Ray call, and the required CPU gate runs on images without Ray installed.
    """
    from cosmos_framework.posttrain.control.placement import colocate

    def _ensure(_num_cpus: int, _temp_dir: str | None) -> None:
        if not ray_started:
            raise _RayStarted("bind_device_groups started Ray before checking declared capacity")

    def _live(_ray_module: object, _required: float) -> None:
        raise _LiveCapacityChecked("the registered-GPU guard ran")

    def _nodes() -> list[dict[str, object]]:
        raise _ReachedNodeScan("both capacity guards passed")

    stub_ray = types.ModuleType("ray")
    stub_ray.__path__ = []  # type: ignore[attr-defined]  -- a package, so ray.util resolves
    stub_ray.nodes = _nodes  # type: ignore[attr-defined]
    stub_ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    stub_util = types.ModuleType("ray.util")
    stub_util.__path__ = []  # type: ignore[attr-defined]
    stub_pg = types.ModuleType("ray.util.placement_group")
    stub_pg.placement_group = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", stub_ray)
    monkeypatch.setitem(sys.modules, "ray.util", stub_util)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", stub_pg)
    monkeypatch.setattr(colocate, "_ensure_ray", _ensure)
    monkeypatch.setattr(colocate, "_validate_registered_gpu_capacity", _live)
    # 8 student GPUs plus a disjoint 1-GPU teacher pool: the K + T rule, so 9 in total.
    return [
        _device("student_gpus", "gpu", _actor("student", _worker("policy", "trainer", (1, 8)))),
        _device("teacher_gpus", "gpu", _actor("teacher", _worker("teacher", "teacher", (1, 1)))),
    ]


def test_declared_shortage_is_rejected_before_ray_is_started(monkeypatch: pytest.MonkeyPatch) -> None:
    device_groups = _capacity_probe(monkeypatch, ray_started=False)
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")

    # _RayStarted would mean the guard ran too late to save the cluster-join cost.
    with pytest.raises(
        RuntimeError,
        match="require 9 disjoint physical GPUs.*provides 8 .NNODES=1 x GPUS_PER_NODE=8.",
    ):
        bind_device_groups(device_groups, backend="ray")


def test_joined_cluster_capacity_is_not_asserted_against_a_partial_view(monkeypatch: pytest.MonkeyPatch) -> None:
    device_groups = _capacity_probe(monkeypatch, ray_started=True)
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("GPUS_PER_NODE", "8")

    # 9 of a declared 16 fits, and the joined cluster's registered count must not be consulted:
    # workers are still registering, so a shortfall there proves nothing and would kill this launch.
    with pytest.raises(_ReachedNodeScan):
        bind_device_groups(device_groups, backend="ray")


def test_local_cluster_still_fails_fast_without_a_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    device_groups = _capacity_probe(monkeypatch, ray_started=True)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("GPUS_PER_NODE", raising=False)

    # A locally started cluster owns every GPU it will ever have, so it keeps the immediate verdict.
    with pytest.raises(_LiveCapacityChecked):
        bind_device_groups(device_groups, backend="ray")


def test_host_affinity_resource_preflight_requires_exact_full_gpu_pool() -> None:
    actor = _actor("policy_actors", _worker("policy", "rollout", (2, 1)))
    actor.members[0].placement["slot"] = "policy_slot"
    (resolved_device,) = _resolve_device_groups(
        [_device("policy_gpus", "gpu", actor)],
        _FakePlacement(),
    )
    (resolved_actor,) = resolved_device.actor_groups
    oversized_pool = [
        {
            "Alive": True,
            "NodeID": "policy-raylet",
            "Resources": {"GPU": 4.0, "policy_slot": 2.0, "node_0": 2.0},
        }
    ]

    with pytest.raises(RuntimeError, match="complete physical GPU pool.*gpu_slots_per_host=2.*GPU.*4"):
        _validate_host_affinity_resources(oversized_pool, (resolved_actor,), 2)

    exact_pool = [
        {
            "Alive": True,
            "NodeID": "policy-raylet",
            "Resources": {"GPU": 2.0, "policy_slot": 2.0, "node_0": 2.0},
        }
    ]
    _validate_host_affinity_resources(exact_pool, (resolved_actor,), 2)


@pytest.mark.parametrize(
    ("device_groups", "relationship", "policy_actor_group", "rollout_actor_group"),
    [
        (
            [
                _device(
                    "shared_process_gpus",
                    "gpu",
                    _actor(
                        "shared_process",
                        _worker("same_process_policy", "trainer", (1, 2)),
                        _worker("same_process_rollout", "rollout", (1, 2)),
                    ),
                )
            ],
            "same_process",
            "shared_process",
            "shared_process",
        ),
        (
            [
                _device(
                    "shared_gpus",
                    "gpu",
                    _actor("policy_process", _worker("same_gpu_policy", "trainer", (1, 8))),
                    _actor("rollout_process", _worker("same_gpu_rollout", "rollout", (4, 2))),
                )
            ],
            "same_gpu",
            "policy_process",
            "rollout_process",
        ),
        (
            [
                _device(
                    "policy_gpus",
                    "gpu",
                    _actor("policy_process", _worker("cross_device_policy", "trainer", (1, 4))),
                ),
                _device(
                    "rollout_gpus",
                    "gpu",
                    _actor("rollout_process", _worker("cross_device_rollout", "rollout", (1, 2))),
                ),
            ],
            "cross_device",
            "policy_process",
            "rollout_process",
        ),
        (
            [
                _device(
                    "cpu_models",
                    "cpu",
                    _actor("policy_process", _worker("host_policy", "trainer", (1, 2))),
                    _actor("rollout_process", _worker("host_rollout", "rollout", (1, 2))),
                )
            ],
            "host",
            "policy_process",
            "rollout_process",
        ),
    ],
    ids=["same-process", "same-gpu", "cross-device", "host"],
)
def test_weight_relationship_is_derived_from_group_structure(
    device_groups: list[DeviceGroup],
    relationship: str,
    policy_actor_group: str,
    rollout_actor_group: str,
) -> None:
    resolved = _resolve_device_groups(device_groups, _FakePlacement())

    edge = _resolve_weight_edge(resolved)

    assert edge is not None
    assert edge.relationship == relationship
    assert edge.policy_actor_group == policy_actor_group
    assert edge.rollout_actor_group == rollout_actor_group


def test_bind_device_groups_is_the_only_public_binder_without_gpu_fraction_arguments() -> None:
    assert "num_gpus_each" not in inspect.signature(bind_device_groups).parameters
    assert "num_gpus" not in inspect.signature(bind_device_groups).parameters
    for retired in ("Colocation", "bind_actor_group", "bind_colocated", "bind_colocated_groups"):
        assert not hasattr(placement, retired)


def test_local_backend_rejects_unprovable_gpu_device_group() -> None:
    worker = _worker("worker", "env", (1, 1))
    with pytest.raises(ValueError, match="no Ray GPU placement or CUDA UUID proof"):
        bind_device_groups([_device("gpu", "gpu", _actor("actor", worker))], backend="local")


def test_local_backend_rejects_host_affinity() -> None:
    worker = _worker("worker", "env", (1, 1))
    with pytest.raises(ValueError, match="does not support host_affinity"):
        bind_device_groups(
            [_device("cpu", "cpu", _actor("actor", worker))],
            backend="local",
            colocate="host_affinity",
        )


def _gpu_identity(cuda_uuid: str, *, node: str, visible: str) -> dict[str, object]:
    return {
        "cuda_uuid": cuda_uuid,
        "ray_node_id": node,
        "cuda_visible_devices": visible,
        "ray_accelerator_ids": ("0",),
    }


def test_gpu_identity_uses_uuid_not_cross_raylet_logical_ids() -> None:
    policy = _actor("policy_process", _worker("policy", "trainer", (1, 2)))
    rollout = _actor("rollout_process", _worker("rollout", "rollout", (1, 2)))
    resolved = _resolve_device_groups([_device("shared_gpus", "gpu", policy, rollout)], _FakePlacement())

    _validate_gpu_identities(
        resolved,
        {
            "policy_process": {
                (0, 0): _gpu_identity("uuid-0", node="policy-raylet", visible="0"),
                (0, 1): _gpu_identity("uuid-1", node="policy-raylet", visible="1"),
            },
            "rollout_process": {
                (0, 0): _gpu_identity("uuid-0", node="rollout-raylet", visible="0"),
                (0, 1): _gpu_identity("uuid-1", node="rollout-raylet", visible="1"),
            },
        },
    )


def test_host_affinity_aligns_permuted_actor_grids_by_physical_uuid() -> None:
    policy = _actor("policy_process", _worker("policy", "env", (2, 2)))
    rollout = _actor("rollout_process", _worker("rollout", "reward", (1, 4)))
    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpus", "gpu", policy, rollout)],
        _FakePlacement(),
    )
    policy_resolved, rollout_resolved = resolved_device.actor_groups
    policy_grid = {
        (0, 0): ("policy_0", "policy_uuid_0_host"),
        (0, 1): ("policy_1", "policy_uuid_1_host"),
        (1, 0): ("policy_2", "policy_uuid_2_host"),
        (1, 1): ("policy_3", "policy_uuid_3_host"),
    }
    rollout_grid = {
        (0, 0): ("rollout_0", "rollout_uuid_2_host"),
        (0, 1): ("rollout_1", "rollout_uuid_0_host"),
        (0, 2): ("rollout_2", "rollout_uuid_3_host"),
        (0, 3): ("rollout_3", "rollout_uuid_1_host"),
    }
    spawned = {
        "policy_process": _SpawnedActorGroup(policy_resolved, policy_grid),
        "rollout_process": _SpawnedActorGroup(rollout_resolved, rollout_grid),
    }
    identities = {
        "policy_process": {
            (0, 0): _gpu_identity("uuid-0", node="policy-raylet", visible="0"),
            (0, 1): _gpu_identity("uuid-1", node="policy-raylet", visible="1"),
            (1, 0): _gpu_identity("uuid-2", node="policy-raylet", visible="2"),
            (1, 1): _gpu_identity("uuid-3", node="policy-raylet", visible="3"),
        },
        "rollout_process": {
            (0, 0): _gpu_identity("uuid-2", node="rollout-raylet", visible="2"),
            (0, 1): _gpu_identity("uuid-0", node="rollout-raylet", visible="0"),
            (0, 2): _gpu_identity("uuid-3", node="rollout-raylet", visible="3"),
            (0, 3): _gpu_identity("uuid-1", node="rollout-raylet", visible="1"),
        },
    }
    peers = {store_id: host for grid in (policy_grid, rollout_grid) for store_id, host in grid.values()}

    _align_host_affinity_grids_by_uuid((resolved_device,), spawned, identities)

    assert rollout_grid == {
        (0, 0): ("rollout_1", "rollout_uuid_0_host"),
        (0, 1): ("rollout_3", "rollout_uuid_1_host"),
        (0, 2): ("rollout_0", "rollout_uuid_2_host"),
        (0, 3): ("rollout_2", "rollout_uuid_3_host"),
    }
    assert [identities["rollout_process"][cell]["cuda_uuid"] for cell in sorted(rollout_grid)] == [
        "uuid-0",
        "uuid-1",
        "uuid-2",
        "uuid-3",
    ]
    assert all(peers[store_id] == host for grid in (policy_grid, rollout_grid) for store_id, host in grid.values())
    _validate_gpu_identities((resolved_device,), identities)


def test_host_affinity_uuid_pool_mismatch_fails_without_mutating_grids() -> None:
    first_left = _actor("first_left", _worker("first_left_wg", "env", (1, 2)))
    first_right = _actor("first_right", _worker("first_right_wg", "reward", (1, 2)))
    policy = _actor("policy_process", _worker("policy", "trainer", (1, 2)))
    rollout = _actor("rollout_process", _worker("rollout", "rollout", (1, 2)))
    first_device, mismatch_device = _resolve_device_groups(
        [
            _device("first_gpus", "gpu", first_left, first_right),
            _device("mismatch_gpus", "gpu", policy, rollout),
        ],
        _FakePlacement(),
    )
    first_left_resolved, first_right_resolved = first_device.actor_groups
    policy_resolved, rollout_resolved = mismatch_device.actor_groups
    first_left_grid = {(0, 0): ("first_left_0", object()), (0, 1): ("first_left_1", object())}
    first_right_grid = {(0, 0): ("first_right_0", object()), (0, 1): ("first_right_1", object())}
    policy_grid = {(0, 0): ("policy_0", object()), (0, 1): ("policy_1", object())}
    rollout_grid = {(0, 0): ("rollout_0", object()), (0, 1): ("rollout_1", object())}
    original_first_right_grid = dict(first_right_grid)
    original_rollout_grid = dict(rollout_grid)
    spawned = {
        "first_left": _SpawnedActorGroup(first_left_resolved, first_left_grid),
        "first_right": _SpawnedActorGroup(first_right_resolved, first_right_grid),
        "policy_process": _SpawnedActorGroup(policy_resolved, policy_grid),
        "rollout_process": _SpawnedActorGroup(rollout_resolved, rollout_grid),
    }
    identities = {
        "first_left": {
            (0, 0): _gpu_identity("first-uuid-0", node="first-left-raylet", visible="0"),
            (0, 1): _gpu_identity("first-uuid-1", node="first-left-raylet", visible="1"),
        },
        "first_right": {
            (0, 0): _gpu_identity("first-uuid-1", node="first-right-raylet", visible="1"),
            (0, 1): _gpu_identity("first-uuid-0", node="first-right-raylet", visible="0"),
        },
        "policy_process": {
            (0, 0): _gpu_identity("uuid-0", node="policy-raylet", visible="0"),
            (0, 1): _gpu_identity("uuid-1", node="policy-raylet", visible="1"),
        },
        "rollout_process": {
            (0, 0): _gpu_identity("uuid-0", node="rollout-raylet", visible="0"),
            (0, 1): _gpu_identity("uuid-other", node="rollout-raylet", visible="1"),
        },
    }

    with pytest.raises(RuntimeError, match="missing UUIDs=.*uuid-1.*unexpected UUIDs=.*uuid-other"):
        _align_host_affinity_grids_by_uuid((first_device, mismatch_device), spawned, identities)

    assert first_right_grid == original_first_right_grid
    assert rollout_grid == original_rollout_grid


def test_gpu_identity_rejects_same_host_and_logical_id_on_different_physical_gpus() -> None:
    policy = _actor("policy_process", _worker("policy", "trainer", (1, 1)))
    rollout = _actor("rollout_process", _worker("rollout", "rollout", (1, 1)))
    resolved = _resolve_device_groups([_device("shared_gpus", "gpu", policy, rollout)], _FakePlacement())

    with pytest.raises(RuntimeError, match="not the same physical GPU"):
        _validate_gpu_identities(
            resolved,
            {
                "policy_process": {(0, 0): _gpu_identity("uuid-0", node="same-host", visible="0")},
                "rollout_process": {(0, 0): _gpu_identity("uuid-1", node="same-host", visible="0")},
            },
        )


def test_gpu_identity_reports_flat_slot_count_invariant_instead_of_stop_iteration() -> None:
    first = _actor("first", _worker("first_wg", "env", (1, 1)))
    second = _actor("second", _worker("second_wg", "reward", (1, 1)))
    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpus", "gpu", first, second)],
        _FakePlacement(),
    )
    first_resolved, second_resolved = resolved_device.actor_groups
    invalid_second = replace(second_resolved, ranks=2)
    invalid_device = replace(resolved_device, actor_groups=(first_resolved, invalid_second))

    with pytest.raises(RuntimeError, match="reported 1 flat slots.*reported 2"):
        _validate_gpu_identities(
            (invalid_device,),
            {
                "first": {(0, 0): _gpu_identity("uuid-0", node="first", visible="0")},
                "second": {
                    (0, 0): _gpu_identity("uuid-0", node="second", visible="0"),
                    (0, 1): _gpu_identity("uuid-1", node="second", visible="1"),
                },
            },
        )


def test_gpu_identity_reports_empty_resolved_device_group_as_internal_invariant() -> None:
    actor = _actor("actor", _worker("worker", "env", (1, 1)))
    (resolved_device,) = _resolve_device_groups(
        [_device("shared_gpus", "gpu", actor)],
        _FakePlacement(),
    )
    invalid_device = replace(resolved_device, actor_groups=())

    with pytest.raises(RuntimeError, match="internal GPU identity invariant.*contains no ActorGroups"):
        _validate_gpu_identities((invalid_device,), {})


def test_cuda_uuid_v2_error_names_minimum_driver_api() -> None:
    pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers.ray import _require_cuda_uuid_v2

    with pytest.raises(RuntimeError, match=r"CUDA 11\.4\+ Driver API.*R470"):
        _require_cuda_uuid_v2(object())


def test_gpu_identity_aligns_transposed_shapes_by_flat_index() -> None:
    two_by_four = _actor("two_by_four", _worker("a", "env", (2, 4)))
    four_by_two = _actor("four_by_two", _worker("b", "reward", (4, 2)))
    resolved = _resolve_device_groups(
        [_device("shared_gpus", "gpu", two_by_four, four_by_two)],
        _FakePlacement(),
    )
    uuids = tuple(f"uuid-{index}" for index in range(8))

    _validate_gpu_identities(
        resolved,
        {
            "two_by_four": {
                (index // 4, index % 4): _gpu_identity(uuid, node="a", visible=str(index))
                for index, uuid in enumerate(uuids)
            },
            "four_by_two": {
                (index // 2, index % 2): _gpu_identity(uuid, node="b", visible=str(index))
                for index, uuid in enumerate(uuids)
            },
        },
    )


def test_gpu_identity_rejects_overlapping_device_group_pools() -> None:
    left = _device("left_gpus", "gpu", _actor("left", _worker("left_wg", "env", (1, 1))))
    right = _device("right_gpus", "gpu", _actor("right", _worker("right_wg", "reward", (1, 1))))
    resolved = _resolve_device_groups([left, right], _FakePlacement())

    with pytest.raises(RuntimeError, match="disjoint physical GPU pools"):
        _validate_gpu_identities(
            resolved,
            {
                "left": {(0, 0): _gpu_identity("same-uuid", node="left", visible="0")},
                "right": {(0, 0): _gpu_identity("same-uuid", node="right", visible="0")},
            },
        )


def test_live_gpu_identity_proof_precedes_setup_and_uses_bind_rollback() -> None:
    source = (Path(__file__).resolve().parents[1] / "control" / "placement" / "colocate.py").read_text()
    bind_at = source.index("def bind_device_groups")
    guard_at = source.index("def _guard_bind", bind_at)
    identity_at = source.index("gpu_identities = _guard_bind", guard_at)
    align_at = source.index("_align_host_affinity_grids_by_uuid(", identity_at)
    validate_at = source.index("_validate_gpu_identities(", align_at)
    setup_at = source.index("setup_grid(", validate_at)

    assert guard_at < identity_at < align_at < validate_at < setup_at
    assert "_cleanup_ray_bind(ray, spawned, all_placement_groups)" in source[guard_at:identity_at]


class _IdentityBody:
    def __init__(self, model: object) -> None:
        self.model = model


def test_local_same_process_bind_requires_and_accepts_shared_storage_identity() -> None:
    policy = _worker("policy", "trainer", (1, 1))
    rollout = _worker("rollout", "rollout", (1, 1))
    shared_model = object()

    def _setup() -> dict[str, object]:
        return {
            "policy": _IdentityBody(shared_model),
            "rollout": _IdentityBody(shared_model),
        }

    bind_device_groups(
        [_device("local", "cpu", ActorGroup((policy, rollout), _setup, "shared_process"))],
        backend="local",
        weight_sync=True,
    )

    from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

    assert isinstance(policy.body._weight_backend.mover, SharedMover)
    assert isinstance(rollout.body._weight_backend.mover, SharedMover)


def test_local_same_process_bind_rejects_distinct_storage_identity() -> None:
    policy = _worker("policy", "trainer", (1, 1))
    rollout = _worker("rollout", "rollout", (1, 1))

    def _setup() -> dict[str, object]:
        return {
            "policy": _IdentityBody(object()),
            "rollout": _IdentityBody(object()),
        }

    with pytest.raises(RuntimeError, match="exact model storage"):
        bind_device_groups(
            [_device("local", "cpu", ActorGroup((policy, rollout), _setup, "shared_process"))],
            backend="local",
            weight_sync=True,
        )
    assert policy.body is None and rollout.body is None
    assert policy._backend is None and rollout._backend is None


def test_local_weight_endpoint_without_model_fails_before_publish() -> None:
    policy = _worker("policy", "trainer", (1, 1))
    rollout = _worker("rollout", "rollout", (1, 1))
    device_group = _device(
        "local",
        "cpu",
        ActorGroup((policy,), lambda: {"policy": object()}, "policy_process"),
        ActorGroup((rollout,), lambda: {"rollout": object()}, "rollout_process"),
    )

    with pytest.raises(RuntimeError, match="exposes no model"):
        bind_device_groups([device_group], backend="local", weight_sync=True)
    assert policy.body is None and rollout.body is None


def test_local_setup_must_return_exact_actor_group_bodies() -> None:
    worker = _worker("worker", "env", (1, 1))
    actor_group = ActorGroup(
        (worker,),
        lambda: {"worker": object(), "undeclared": object()},
        "actor",
    )

    with pytest.raises(RuntimeError, match="expected exactly"):
        bind_device_groups([_device("local", "cpu", actor_group)], backend="local")
    assert worker.body is None
