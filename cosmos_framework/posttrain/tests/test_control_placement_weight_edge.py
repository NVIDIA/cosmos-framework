# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for edge-scoped asymmetric placement validation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.control.placement.colocate import (
    ActorGroup,
    DeviceGroup,
    _required_placement_gpus,
    _resolve_device_groups,
    _resolve_weight_edge,
    _run_nixl_probe,
    _validate_weight_edge_topology,
    _wait_placement_groups,
    _WeightEdge,
    bind_device_groups,
)
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _actor_group(name: str, role: str, shape: tuple[int, int]) -> ActorGroup:
    wg = WorkerGroup(name, role, replicas=shape[0], ranks=shape[1])
    return ActorGroup((wg,), lambda: {}, name=f"{name}_actors")


class _FakePlacement:
    def __init__(self, shapes: dict[str, tuple[int, int]] | None = None) -> None:
        self.shapes = shapes or {}
        self.calls: list[str] = []

    def resolve(self, wg: WorkerGroup) -> dict[str, int]:
        self.calls.append(wg.name)
        replicas, ranks = self.shapes.get(wg.name, (wg.replicas, wg.ranks))
        return {"replicas": replicas, "ranks": ranks}


def _edge(
    device_groups: list[DeviceGroup],
    *,
    backend: str,
    transport: str | None,
    colocate: str,
) -> _WeightEdge | None:
    resolved = _resolve_device_groups(device_groups, _FakePlacement())
    edge = _resolve_weight_edge(resolved)
    if edge is not None:
        _validate_weight_edge_topology(edge, backend=backend, transport=transport, colocate=colocate)
    return edge


def test_weight_edge_and_span_guard_share_one_resolved_shape_snapshot() -> None:
    policy = WorkerGroup("policy", "trainer", replicas=1, ranks=1)
    helper = WorkerGroup("replay", "buffer", replicas=1, ranks=8)
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=1)
    actor_groups = [
        ActorGroup((policy, helper), lambda: {}, name="policy_actors"),
        ActorGroup((rollout,), lambda: {}, name="rollout_actors"),
    ]
    placement = _FakePlacement({"policy": (1, 4), "replay": (1, 4), "rollout": (2, 2)})
    resolved = _resolve_device_groups(
        [
            DeviceGroup((actor_groups[0],), device="gpu", name="policy_devices"),
            DeviceGroup((actor_groups[1],), device="gpu", name="rollout_devices"),
        ],
        placement,
    )
    edge = _resolve_weight_edge(resolved)

    assert edge is not None
    assert edge.policy_shape == (1, 4)
    assert edge.rollout_shape == (2, 2)
    assert placement.calls == ["policy", "replay", "rollout"]


def test_bind_resolves_each_wg_shape_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.control.placement import placement as placement_module

    policy = WorkerGroup("policy", "trainer", replicas=1, ranks=1)
    helper = WorkerGroup("replay", "buffer", replicas=1, ranks=8)
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=1)
    resolved = _FakePlacement({"policy": (1, 1), "replay": (1, 1), "rollout": (1, 1)})

    class _OneShotPlacement:
        def resolve(self, wg: WorkerGroup) -> dict[str, int]:
            if wg.name in resolved.calls:
                raise AssertionError(f"Placement.resolve called twice for {wg.name}")
            return resolved.resolve(wg)

    class _ModelBody:
        model = object()

    monkeypatch.setattr(placement_module, "Placement", _OneShotPlacement)
    bind_device_groups(
        [
            DeviceGroup(
                (
                    ActorGroup((policy, helper), lambda: {"policy": _ModelBody(), "replay": object()}),
                    ActorGroup((rollout,), lambda: {"rollout": _ModelBody()}),
                ),
                device="cpu",
            )
        ],
        backend="local",
        weight_sync=True,
    )
    assert resolved.calls == ["policy", "replay", "rollout"]


def test_weight_endpoint_span_guard_rejects_resolved_mismatch_hidden_by_declarations() -> None:
    policy = WorkerGroup("policy", "trainer", replicas=1, ranks=1)
    helper = WorkerGroup("replay", "buffer", replicas=1, ranks=1)
    rollout = WorkerGroup("rollout", "rollout", replicas=1, ranks=2)
    device_groups = [
        DeviceGroup((ActorGroup((policy, helper), lambda: {}),), device="gpu"),
        DeviceGroup((ActorGroup((rollout,), lambda: {}),), device="gpu"),
    ]
    placement = _FakePlacement({"policy": (1, 1), "replay": (1, 2), "rollout": (1, 2)})

    with pytest.raises(ValueError, match=r"trainer WG 'policy'.*ActorGroup physical shape"):
        _resolve_device_groups(device_groups, placement)


def test_unequal_nixl_edge_ignores_frozen_model_group_product() -> None:
    actor_groups = [
        _actor_group("policy", "trainer", (1, 64)),
        _actor_group("rollout", "rollout", (1, 4)),
        _actor_group("teacher", "teacher", (1, 8)),
    ]
    device_groups = [
        DeviceGroup((actor_group,), device="gpu", name=f"devices_{index}")
        for index, actor_group in enumerate(actor_groups)
    ]
    edge = _edge(
        device_groups,
        backend="ray",
        transport="nixl",
        colocate="pg",
    )
    assert edge is not None
    assert edge.policy_shape == (1, 64)
    assert edge.rollout_shape == (1, 4)
    assert edge.relationship == "cross_device"
    # Independent domains reserve the actual 64+4+8 cells rather than max(64,4,8)*3.
    resolved = _resolve_device_groups(device_groups, _FakePlacement())
    assert _required_placement_gpus(resolved) == 76


@pytest.mark.parametrize(
    ("layout", "backend", "transport", "colocate", "match"),
    [
        ("separate_gpu", "ray", None, "pg", "unequal-product.*cross_device.*nixl"),
        ("same_gpu", "ray", "nixl", "pg", "same actor product"),
        ("same_cpu", "local", "nixl", "pg", "unequal-product.*cross_device.*nixl"),
        ("separate_gpu", "ray", "nixl", "host_affinity", "unequal-product.*cross_device.*nixl"),
    ],
)
def test_unequal_edge_rejects_non_disaggregated_or_non_nixl_bind(
    layout: str,
    backend: str,
    transport: str | None,
    colocate: str,
    match: str,
) -> None:
    actor_groups = [
        _actor_group("policy", "trainer", (1, 4)),
        _actor_group("rollout", "rollout", (1, 2)),
    ]
    if layout == "separate_gpu":
        device_groups = [DeviceGroup((actor_group,), device="gpu") for actor_group in actor_groups]
    else:
        device_groups = [DeviceGroup(tuple(actor_groups), device="gpu" if layout == "same_gpu" else "cpu")]
    with pytest.raises(ValueError, match=match):
        _edge(
            device_groups,
            backend=backend,
            transport=transport,
            colocate=colocate,
        )


def test_equal_product_ipc_pair_packs_without_unrelated_teacher() -> None:
    rollout = _actor_group("rollout", "rollout", (2, 2))
    teacher = _actor_group("teacher", "teacher", (1, 8))
    policy = _actor_group("policy", "trainer", (1, 4))
    device_groups = [
        DeviceGroup((rollout, policy), device="gpu", name="student_devices"),
        DeviceGroup((teacher,), device="gpu", name="teacher_devices"),
    ]
    edge = _edge(
        device_groups,
        backend="ray",
        transport=None,
        colocate="pg",
    )
    assert edge is not None and edge.relationship == "same_gpu"
    resolved = _resolve_device_groups(device_groups, _FakePlacement())
    assert [[actor.name for actor in group.actor_groups] for group in resolved] == [
        ["rollout_actors", "policy_actors"],
        ["teacher_actors"],
    ]


def test_relay_mode_validates_p2r_against_only_rollout_replica_zero() -> None:
    edge = _WeightEdge(
        policy_actor_group="policy_actors",
        rollout_actor_group="rollout_actors",
        policy_names=("policy",),
        rollout_names=("rollout",),
        policy_shape=(1, 8),
        rollout_shape=(4, 2),
        relationship="same_gpu",
    )

    _validate_weight_edge_topology(
        edge,
        backend="ray",
        transport=None,
        colocate="pg",
        rollout_target_shape=(1, 2),
    )


def test_host_affinity_is_same_gpu_even_with_full_actor_requests() -> None:
    device_groups = [
        DeviceGroup(
            (
                _actor_group("policy", "trainer", (1, 2)),
                _actor_group("rollout", "rollout", (1, 2)),
            ),
            device="gpu",
        )
    ]
    edge = _edge(
        device_groups,
        backend="ray",
        transport=None,
        colocate="host_affinity",
    )
    assert edge is not None and edge.relationship == "same_gpu"


def test_transport_hint_does_not_change_same_gpu_placement() -> None:
    policy = WorkerGroup("policy", "trainer")
    rollout = WorkerGroup("rollout", "rollout")
    with pytest.raises(ValueError, match="relationship='same_gpu'.*cross_device"):
        bind_device_groups(
            [
                DeviceGroup(
                    (ActorGroup((policy,), lambda: {}), ActorGroup((rollout,), lambda: {})),
                    device="gpu",
                )
            ],
            backend="ray",
            weight_sync=True,
            weight_sync_transport="nixl",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_slots_per_host": 0},
        {"gpu_slots_per_host": True},
        {"weight_sync_bucket_bytes": 0},
        {"weight_sync_probe_bytes": 0},
        {"weight_sync_min_gbps": float("inf")},
    ],
)
def test_weight_runtime_arguments_fail_before_setup(kwargs: dict) -> None:
    device_group = DeviceGroup((_actor_group("data", "env", (1, 1)),), device="cpu")
    with pytest.raises(ValueError):
        bind_device_groups([device_group], backend="local", **kwargs)


def test_weight_transport_requires_weight_sync_enabled() -> None:
    device_group = DeviceGroup((_actor_group("data", "env", (1, 1)),), device="cpu")

    with pytest.raises(ValueError, match="weight_sync_transport=.*requires weight_sync=True"):
        bind_device_groups(
            [device_group],
            backend="local",
            weight_sync=False,
            weight_sync_transport="nixl",
        )


def test_replica_broadcast_mode_requires_ray_nccl_placement() -> None:
    rollout = _actor_group("rollout", "rollout", (1, 1))

    with pytest.raises(ValueError, match="requires backend='ray'"):
        bind_device_groups(
            [DeviceGroup((rollout,), device="cpu")],
            backend="local",
            weight_sync_r2r_broadcast=True,
        )


def test_full_local_bind_never_attaches_weight_backend_to_teacher() -> None:
    class _Body:
        model = object()

    policy = WorkerGroup("policy", "trainer")
    rollout = WorkerGroup("rollout", "rollout")
    teacher = WorkerGroup("teacher", "teacher", ranks=8)
    bind_device_groups(
        [
            DeviceGroup(
                (
                    ActorGroup((policy,), lambda: {"policy": _Body()}),
                    ActorGroup((rollout,), lambda: {"rollout": _Body()}),
                    ActorGroup((teacher,), lambda: {"teacher": _Body()}),
                ),
                device="cpu",
            )
        ],
        backend="local",
        weight_sync=True,
    )
    assert hasattr(policy.body, "_weight_backend")
    assert hasattr(rollout.body, "_weight_backend")
    assert not hasattr(teacher.body, "_weight_backend")


def test_placement_group_timeout_reports_required_and_registered_gpus() -> None:
    class _Timeout(Exception):
        pass

    class _PlacementGroup:
        @staticmethod
        def ready() -> object:
            return object()

    ray = SimpleNamespace(
        get=lambda _refs, timeout: (_ for _ in ()).throw(_Timeout()),
        exceptions=SimpleNamespace(GetTimeoutError=_Timeout),
        cluster_resources=lambda: {"GPU": 72.0},
    )
    with pytest.raises(RuntimeError, match=r"requires 76 GPUs.*72 registered GPUs"):
        _wait_placement_groups(ray, [_PlacementGroup()], required_gpus=76.0, timeout_s=0.01)


def test_probe_cleanup_failure_does_not_mask_primary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Remote:
        def __init__(self, action: Callable[[], None]) -> None:
            self._action = action

        def remote(self, *_args: object) -> Callable[[], None]:
            return self._action

    def _raise_primary() -> None:
        raise ValueError("primary probe failure")

    def _raise_cleanup() -> None:
        raise RuntimeError("cleanup failure")

    source = SimpleNamespace(
        nixl_probe_publish=_Remote(_raise_primary),
        nixl_probe_release=_Remote(_raise_cleanup),
    )
    target = SimpleNamespace(nixl_probe_release=_Remote(lambda: None))

    def _get(value: object) -> object:
        if isinstance(value, list):
            return [_get(item) for item in value]
        return value() if callable(value) else value

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=_get))
    with pytest.raises(ValueError, match="primary probe failure"):
        _run_nixl_probe(source, target, 1, 0.0)


def test_probe_changes_the_timed_value_witness_and_rejects_a_stale_receive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Remote:
        def __init__(self, action: Callable[..., object]) -> None:
            self._action = action

        def remote(self, *args: object) -> Callable[[], object]:
            return lambda: self._action(*args)

    published: list[int] = []
    fetched: list[int] = []
    received: int | None = None

    def _publish(_nbytes: object, _buffer_bytes: object, fill: object) -> int:
        published.append(int(fill))
        return int(fill)

    def _fetch(handle: object, fill: object) -> dict[str, int | float]:
        nonlocal received
        assert int(handle) == int(fill)
        fetched.append(int(fill))
        if len(fetched) == 1:
            received = int(handle)
        # Simulate a silent no-op on the timed pass: the receive allocation retains warm-up bytes.
        if received != int(fill):
            raise RuntimeError("NIXL probe value witness failed")
        return {"bytes": 4, "gbps": 1.0}

    source = SimpleNamespace(
        nixl_probe_publish=_Remote(_publish),
        nixl_probe_release=_Remote(lambda: None),
        node_id=_Remote(lambda: "source-node"),
    )
    target = SimpleNamespace(
        nixl_probe_fetch=_Remote(_fetch),
        nixl_probe_release=_Remote(lambda: None),
    )

    def _get(value: object) -> object:
        if isinstance(value, list):
            return [_get(item) for item in value]
        return value() if callable(value) else value

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=_get))
    with pytest.raises(RuntimeError, match="NIXL probe value witness failed"):
        _run_nixl_probe(source, target, 4, 0.0)
    assert published == fetched == [0xA5, 0x5A]


def _probe_hosts_with_timings(seconds_by_bytes: dict[int, float]) -> tuple[SimpleNamespace, SimpleNamespace, list[int]]:
    """Fake probe hosts whose fetch reports wall-clock seconds by transfer size and moves every byte."""

    class _Remote:
        def __init__(self, action: Callable[..., object]) -> None:
            self._action = action

        def remote(self, *args: object) -> Callable[[], object]:
            return lambda: self._action(*args)

    sizes: list[int] = []

    def _publish(nbytes: object, _buffer_bytes: object, fill: object) -> tuple[int, int]:
        sizes.append(int(nbytes))
        return int(nbytes), int(fill)

    def _fetch(handle: object, fill: object) -> dict[str, int | float | str]:
        nbytes, sent_fill = handle  # type: ignore[misc]
        assert int(sent_fill) == int(fill)
        seconds = seconds_by_bytes[int(nbytes)]
        return {"bytes": int(nbytes), "seconds": seconds, "gbps": int(nbytes) / seconds / 1e9, "node_id": "target"}

    source = SimpleNamespace(
        nixl_probe_publish=_Remote(_publish),
        nixl_probe_release=_Remote(lambda: None),
        node_id=_Remote(lambda: "source-node"),
    )
    target = SimpleNamespace(nixl_probe_fetch=_Remote(_fetch), nixl_probe_release=_Remote(lambda: None))
    return source, target, sizes


def _patch_ray_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def _get(value: object) -> object:
        if isinstance(value, list):
            return [_get(item) for item in value]
        return value() if callable(value) else value

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=_get))


@pytest.mark.L0
@pytest.mark.CPU
def test_probe_gates_on_the_fabric_slope_not_the_single_transfer_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """InfiniBand-like timings: 256 MiB in 6.4 ms reads 42 GB/s raw, but 32 MiB takes 3.1 ms, so the
    fixed cost is ~2.7 ms and the fabric moves the extra 224 MiB at ~71 GB/s."""
    big, small = 256 * 2**20, 32 * 2**20
    source, target, sizes = _probe_hosts_with_timings({big: 6.4e-3, small: 3.1e-3})
    _patch_ray_get(monkeypatch)
    report = _run_nixl_probe(source, target, big, 60.0)  # a raw-wall-clock floor of 60 would have failed
    assert sizes == [big, big, small]  # warm-up, timed, and the smaller slope pass
    assert report["gbps"] == pytest.approx(big / 6.4e-3 / 1e9)
    assert report["fabric_gbps"] == pytest.approx((big - small) / 3.3e-3 / 1e9)
    assert report["overhead_seconds"] == pytest.approx(3.1e-3 - small / report["fabric_gbps"] / 1e9)


@pytest.mark.L0
@pytest.mark.CPU
def test_probe_falls_back_to_the_single_transfer_when_the_fixed_cost_dominates(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVLink-like timings: both passes are almost all fixed cost, so the slope is noise; the
    single-transfer figure (a lower bound) is what the floor sees."""
    big, small = 256 * 2**20, 32 * 2**20
    source, target, _ = _probe_hosts_with_timings({big: 3.10e-3, small: 3.05e-3})
    _patch_ray_get(monkeypatch)
    report = _run_nixl_probe(source, target, big, 20.0)
    assert report["fabric_gbps"] == pytest.approx(report["gbps"])
    with pytest.raises(RuntimeError, match="below the 100.00 GB/s fabric floor"):
        _run_nixl_probe(source, target, big, 100.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_probe_skips_the_slope_pass_for_tiny_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target, sizes = _probe_hosts_with_timings({4 * 2**20: 1e-3})
    _patch_ray_get(monkeypatch)
    _run_nixl_probe(source, target, 4 * 2**20, 0.0)  # 1/8 of 4 MiB is below the 1 MiB slope minimum
    assert sizes == [4 * 2**20, 4 * 2**20]
