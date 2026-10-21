# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Test ActorGroup binding drives the SAME B.1 calls over local and ray (DESIGN §2.7/§4.2).
Also: ActorGroup shape = max over WGs, and WG-shape-aware dispatch
(a ranks=1 role runs only on its (r, 0) cell; dp_aligned broadcasts/gathers across ranks)."""

from __future__ import annotations

from typing import Any

import pytest

from cosmos_framework.posttrain.control.placement.colocate import (
    ActorGroup,
    DeviceGroup,
    bind_device_groups,
)
from cosmos_framework.posttrain.roles import roles


class _Counter:
    def __init__(self, start=0):
        self.n = start

    def bump(self, k=1):
        self.n += k
        return self.n


def _setup():
    return {"x": _Counter(10)}


@pytest.mark.L0
@pytest.mark.CPU
def test_gpu_device_group_rejects_mismatched_actor_products():
    # A GPU DeviceGroup pairs actors by flat cell index, so its ActorGroups need equal products.
    # Guard fires before any Ray scheduling.
    pytest.importorskip("ray")
    a = roles.rollout("a", replicas=2, ranks=1, slot="a_slot")
    b = roles.env("b", replicas=3, ranks=1, slot="b_slot")  # mismatched replicas
    with pytest.raises(ValueError, match="same actor product"):
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup((a,), _setup), ActorGroup((b,), _setup)),
                    device="gpu",
                )
            ],
            backend="ray",
            colocate="host_affinity",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_host_affinity_actor_group_requires_one_consistent_slot() -> None:
    pytest.importorskip("ray")
    first = roles.env("first", slot="first_slot")
    second = roles.reward("second", slot="second_slot")
    with pytest.raises(ValueError, match="consistent non-empty placement slot"):
        bind_device_groups(
            [DeviceGroup((ActorGroup((first, second), lambda: {}),), device="gpu")],
            backend="ray",
            colocate="host_affinity",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_host_affinity_requires_distinct_raylet_slots() -> None:
    pytest.importorskip("ray")
    first = roles.rollout("first", slot="shared_slot")
    second = roles.env("second", slot="shared_slot")

    with pytest.raises(ValueError, match="distinct raylet slot per ActorGroup"):
        bind_device_groups(
            [DeviceGroup((ActorGroup((first,), _setup), ActorGroup((second,), _setup)), device="gpu")],
            backend="ray",
            colocate="host_affinity",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_host_affinity_rejects_rank_mesh_without_rendezvous() -> None:
    pytest.importorskip("ray")
    worker = roles.rollout("policy", replicas=1, ranks=2, slot="policy_slot")

    with pytest.raises(ValueError, match="requires ranks=1.*no distributed rendezvous"):
        bind_device_groups(
            [DeviceGroup((ActorGroup((worker,), _setup),), device="gpu")],
            backend="ray",
            colocate="host_affinity",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_host_affinity_requires_complete_per_host_gpu_pools() -> None:
    pytest.importorskip("ray")
    policy = roles.rollout("policy", replicas=2, ranks=1, slot="policy_slot")
    sim = roles.env("sim", replicas=2, ranks=1, slot="sim_slot")

    with pytest.raises(ValueError, match="complete per-host GPU pools.*not divisible.*gpu_slots_per_host=4"):
        bind_device_groups(
            [DeviceGroup((ActorGroup((policy,), _setup), ActorGroup((sim,), _setup)), device="gpu")],
            backend="ray",
            colocate="host_affinity",
            gpu_slots_per_host=4,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_local_runs_setup_in_process_and_calls_body():
    x = roles.env("x")
    bind_device_groups([DeviceGroup((ActorGroup((x,), _setup),), device="cpu")], backend="local")
    assert x.bump(5).result() == 15  # body built by setup, called via Handle


@pytest.mark.L0
@pytest.mark.CPU
def test_ray_colocates_in_one_actor():
    ray = pytest.importorskip("ray")
    x = roles.env("x")
    bind_device_groups(
        [DeviceGroup((ActorGroup((x,), _setup),), device="cpu")],
        backend="ray",
        temp_dir="/tmp/rcray",
    )
    try:
        assert x.bump(5).result() == 15
        assert x.bump(2).result() == 17  # state persists in the actor
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_ray_setup_must_return_exact_actor_group_bodies() -> None:
    ray = pytest.importorskip("ray")
    first = roles.env("first")
    second = roles.reward("second")

    def _incomplete_setup() -> dict[str, object]:
        return {"first": object()}

    try:
        with pytest.raises(RuntimeError, match="expected exactly.*first.*second"):
            bind_device_groups(
                [DeviceGroup((ActorGroup((first, second), _incomplete_setup),), device="cpu")],
                backend="ray",
            )
        assert first._backend is None and second._backend is None
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.GPU
def test_bind_grid_pg_pins_same_rk_across_groups_to_same_gpu():
    """Two equal-product ActorGroups in one GPU DeviceGroup share four GPU bundles. The binder's
    placement-group wiring (cells pinned by row-major flat index) must:
    (1) put same-(r,k) cells from different groups on the SAME physical GPU, and
    (2) put different-(r,k) cells of the SAME group on DIFFERENT physical GPUs.
    Without the PG (pre-fix), Ray packs sequentially so within-group cells share a GPU."""
    torch = pytest.importorskip("torch")

    if torch.cuda.device_count() < 4:
        pytest.skip("needs >=4 GPUs")
    _ray = pytest.importorskip("ray")

    from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    class Probe:
        def gpu(self):
            import os

            import ray

            return (os.environ.get("CUDA_VISIBLE_DEVICES"), tuple(ray.get_gpu_ids()))

    def setup_a():
        return {"a": Probe()}

    def setup_b():
        return {"b": Probe()}

    try:
        a = roles.env("a", replicas=2, ranks=2)
        b = roles.trainer("b", replicas=1, ranks=4)
        bind_device_groups(
            [DeviceGroup((ActorGroup((a,), setup_a), ActorGroup((b,), setup_b)), device="gpu")],
            backend="ray",
            threshold=0,
        )
        gpus_a = a.gpu().all()  # {(r,k): (CUDA_VISIBLE_DEVICES, ray_gpu_ids)}
        gpus_b = b.gpu().all()

        # (1) Equal-product groups share GPUs by row-major flat actor index.
        a_cells = sorted(gpus_a)
        b_cells = sorted(gpus_b)
        for a_cell, b_cell in zip(a_cells, b_cells, strict=True):
            assert gpus_a[a_cell] == gpus_b[b_cell], (a_cell, b_cell, gpus_a[a_cell], gpus_b[b_cell])

        # (2) Within group `a`, all four cells are on distinct GPUs.
        assert len({gpus_a[rk] for rk in gpus_a}) == 4
    finally:
        _ray.shutdown()


@pytest.mark.L0
@pytest.mark.GPU
def test_bind_grid_pg_spans_nodes_when_mesh_exceeds_one() -> None:
    """Multi-node tiling: when total bundles exceed one node's GPU/CPU capacity,
    bind_device_groups creates multiple STRICT_PACK PGs (one per node-worth of bundles).
    Capacity saturation forces them onto distinct nodes. Row-major bundle layout means
    the first node-worth of (r, k) cells lands on one node and the next on another — so
    a sufficiently large replica can straddle nodes.

    Skipped on single-node or heterogeneous-capacity clusters."""
    _ray = pytest.importorskip("ray")
    torch = pytest.importorskip("torch")

    if not _ray.is_initialized():
        _ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    live_gpu_nodes = [n for n in _ray.nodes() if n.get("Alive") and n["Resources"].get("GPU", 0)]
    if len(live_gpu_nodes) < 2:
        pytest.skip("needs >=2 Ray nodes")
    node_capacities = [
        min(int(node["Resources"]["GPU"]), int(node["Resources"].get("CPU", 0))) for node in live_gpu_nodes
    ]
    usable_capacities = [capacity for capacity in node_capacities if capacity > 0]
    if len(usable_capacities) < 2 or len(set(usable_capacities)) != 1:
        pytest.skip(f"needs >=2 homogeneous GPU/CPU-capable Ray nodes, got capacities={node_capacities}")
    bundles_per_node = usable_capacities[0]
    if torch.cuda.device_count() < bundles_per_node:
        pytest.skip("driver-side device_count too small")

    from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    class Probe:
        def node(self) -> str:
            import ray

            return ray._private.services.get_node_ip_address()

    def setup_a() -> dict[str, Probe]:
        return {"a": Probe()}

    # K spans 2 nodes' worth of GPU/CPU bundle capacity deliberately, single replica.
    K = bundles_per_node * 2
    try:
        a = roles.env("a", replicas=1, ranks=K)
        bind_device_groups([DeviceGroup((ActorGroup((a,), setup_a),), device="gpu")], backend="ray", threshold=0)
        nodes = a.node().all()  # {(0, k): node_ip}
        # First node-worth of ranks on one node; next node-worth on another.
        first_half = {nodes[(0, k)] for k in range(bundles_per_node)}
        second_half = {nodes[(0, k)] for k in range(bundles_per_node, K)}
        assert len(first_half) == 1 and len(second_half) == 1
        assert first_half != second_half  # distinct nodes
    finally:
        _ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_host_affinity_pairs_two_raylets_by_physical_gpu_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live two-raylet proof for a multi-GPU pod; requires the eval-style custom resources."""
    import os

    if os.environ.get("POSTTRAIN_MULTI_RAYLET_GPU") != "1":
        pytest.skip("set POSTTRAIN_MULTI_RAYLET_GPU=1 on an eval-style two-raylet GPU cluster")
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)

    live_nodes = [node for node in ray.nodes() if node.get("Alive")]

    def _slot_capacity(slot: str) -> int:
        return max(
            (
                min(
                    int(node["Resources"].get("GPU", 0)),
                    int(node["Resources"].get(slot, 0)),
                    int(node["Resources"].get("node_0", 0)),
                )
                for node in live_nodes
            ),
            default=0,
        )

    slots = min(_slot_capacity("policy_slot"), _slot_capacity("sim_slot"))
    if slots < 2:
        pytest.skip(f"needs policy_slot and sim_slot raylets sharing >=2 physical GPUs, got slots={slots}")

    class Probe:
        def ready(self) -> bool:
            return True

    def setup_policy() -> dict[str, Probe]:
        return {"policy": Probe()}

    def setup_sim() -> dict[str, Probe]:
        return {"sim": Probe()}

    policy = roles.rollout("policy", replicas=slots, ranks=1, slot="policy_slot")
    sim = roles.env("sim", replicas=slots, ranks=1, slot="sim_slot")
    import cosmos_framework.posttrain.control.placement.colocate as placement_module

    original_alignment = placement_module._align_host_affinity_grids_by_uuid
    raw_uuid_orders: dict[str, tuple[str, ...]] = {}

    def _record_raw_then_align(
        resolved_device_groups: Any,
        spawned_by_name: Any,
        identities_by_actor_group: dict[str, dict[tuple[int, int], dict[str, Any]]],
    ) -> None:
        for actor_name, identities in identities_by_actor_group.items():
            raw_uuid_orders[actor_name] = tuple(identities[cell]["cuda_uuid"] for cell in sorted(identities))
        original_alignment(resolved_device_groups, spawned_by_name, identities_by_actor_group)

    monkeypatch.setattr(placement_module, "_align_host_affinity_grids_by_uuid", _record_raw_then_align)
    live_hosts: list[Any] = []
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    (
                        ActorGroup((policy,), setup_policy, name="policy_actors"),
                        ActorGroup((sim,), setup_sim, name="sim_actors"),
                    ),
                    device="gpu",
                    name="shared_eval_gpus",
                )
            ],
            backend="ray",
            colocate="host_affinity",
            gpu_slots_per_host=slots,
            host_single_thread=True,
        )
        policy_backend = policy._backend
        sim_backend = sim._backend
        live_hosts = [*policy_backend.actors.values(), *sim_backend.actors.values()]
        cells = tuple(sorted(policy_backend.actors))
        assert cells == tuple(sorted(sim_backend.actors))
        policy_identities = ray.get([policy_backend.actors[cell].gpu_identity.remote() for cell in cells])
        sim_identities = ray.get([sim_backend.actors[cell].gpu_identity.remote() for cell in cells])
        policy_uuids = tuple(identity["cuda_uuid"] for identity in policy_identities)
        sim_uuids = tuple(identity["cuda_uuid"] for identity in sim_identities)

        assert policy_uuids == sim_uuids
        assert len(set(policy_uuids)) == slots
        if os.environ.get("POSTTRAIN_EXPECT_PERMUTED_GPU_IDS") == "1":
            assert set(raw_uuid_orders["policy_actors"]) == set(raw_uuid_orders["sim_actors"])
            assert raw_uuid_orders["policy_actors"] != raw_uuid_orders["sim_actors"]
        assert {identity["ray_node_id"] for identity in policy_identities}.isdisjoint(
            {identity["ray_node_id"] for identity in sim_identities}
        )
    finally:
        for host in live_hosts:
            ray.kill(host, no_restart=True)
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_grid_2x2_fans_out_to_four_actors():
    _ray = pytest.importorskip("ray")

    from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    def _setup():
        import os

        class W:
            def whoami(self):
                return os.getpid()

        return {"w": W()}

    try:
        w = roles.env("w", replicas=2, ranks=2)
        bind_device_groups([DeviceGroup((ActorGroup((w,), _setup),), device="cpu")], backend="ray", threshold=0)
        pids = w.whoami().all()
        # all four (replica, rank) slots are present
        assert set(pids) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        # each actor runs in its own process (four distinct PIDs)
        assert len(set(pids.values())) == 4
    finally:
        _ray.shutdown()


def _mixed_ranks_setup():
    # One colocated group holding a ranks=TP rollout + ranks=1 data/reward. `make` tags the
    # producing actor by pid; `consume` returns (own pid, received value) so the test can prove
    # WHICH cell ran and WHICH producer value it received.
    import os

    class Data:
        def make(self):
            return os.getpid()

    class Policy:
        def consume(self, x):
            return (os.getpid(), x)

    class Reward:
        def consume(self, y):
            return y

    return {"policy": Policy(), "data": Data(), "reward": Reward()}


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_ray_joins_cluster_when_ray_address_set(monkeypatch):
    import sys
    import types

    from cosmos_framework.posttrain.control.placement.colocate import _ensure_ray

    calls = {}
    fake = types.SimpleNamespace(is_initialized=lambda: False, init=lambda **kw: calls.update(kw))
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")

    _ensure_ray(num_cpus=9, temp_dir="/tmp/x")

    # Full kwargs locked down: joins the cluster, and passes NO num_cpus/_temp_dir/dashboard.
    assert calls == {"address": "10.0.0.1:6379", "ignore_reinit_error": True, "log_to_driver": False}


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_wg_shape_aware_dispatch_broadcast_and_gather():
    # policy declares ranks=2, data/reward stay ranks=1 → group mesh is max (1x2). WG-shape-aware
    # dispatch must run each WG only over its OWN cells, and dp_aligned must broadcast (low→high)
    # / gather (high→low) across the rank mismatch.
    ray = pytest.importorskip("ray")
    import ray as _ray

    if _ray.is_initialized():
        _ray.shutdown()
    policy = roles.rollout("policy", ranks=2)
    data = roles.env("data")  # ranks=1
    reward = roles.reward("reward")  # ranks=1
    bind_device_groups(
        [DeviceGroup((ActorGroup((policy, data, reward), _mixed_ranks_setup),), device="cpu")],
        backend="ray",
        temp_dir="/tmp/rcray",
    )
    try:
        # (1) execution gating: the ranks=1 data WG runs ONLY on its (0, 0) cell, not mirrored to (0, 1).
        dcells = data.make().all()
        assert set(dcells) == {(0, 0)}
        data_pid = dcells[(0, 0)]

        # (2) broadcast low→high: BOTH policy TP cells receive data's single (0, 0) output.
        pcells = policy.consume(data.make()).all()  # {(0,0): (pid,recv), (0,1): (pid,recv)}
        assert set(pcells) == {(0, 0), (0, 1)}  # policy runs on both TP cells
        assert pcells[(0, 0)][1] == data_pid == pcells[(0, 1)][1]  # both got data(0,0)'s value
        assert pcells[(0, 0)][0] != pcells[(0, 1)][0]  # distinct actor processes

        # (3) gather high→low: the ranks=1 reward WG takes policy's (0, 0) output only.
        rcells = reward.consume(policy.consume(data.make())).all()
        assert set(rcells) == {(0, 0)}
        assert rcells[(0, 0)][1] == data_pid  # reward saw policy rank-0's (own_pid, data_pid) tuple
    finally:
        ray.shutdown()


def _rollout_only_setup():
    import os

    class Policy:
        def consume(self, x):
            return (os.getpid(), x)

    return {"policy": Policy()}


def _data_reward_setup():
    import os

    class Data:
        def make(self):
            return os.getpid()

    class Reward:
        def consume(self, y):
            return y

    return {"data": Data(), "reward": Reward()}


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_ray_local_init_when_no_ray_address(monkeypatch):
    import sys
    import types

    from cosmos_framework.posttrain.control.placement.colocate import _ensure_ray

    calls = {}
    fake = types.SimpleNamespace(is_initialized=lambda: False, init=lambda **kw: calls.update(kw))
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    _ensure_ray(num_cpus=9, temp_dir="/tmp/x")

    # Full kwargs locked down: local single-node init (sizing hack + temp dir), NO address.
    assert calls == {
        "num_cpus": 9,
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "log_to_driver": False,
        "_temp_dir": "/tmp/x",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_accepts_typed_groups_and_rejects_raw_tuple():
    from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    def _setup():
        return {"a": object(), "b": object()}

    # Canonical typed form.
    a1, b1 = WorkerGroup("a", "env"), WorkerGroup("b", "reward")
    bind_device_groups([DeviceGroup((ActorGroup((a1, b1), _setup),), device="cpu")], backend="local")
    assert a1.body is not None and b1.body is not None
    # Raw tuple topology is retired: process and device grouping must both be explicit.
    a2, b2 = WorkerGroup("a", "env"), WorkerGroup("b", "reward")
    with pytest.raises(TypeError, match="DeviceGroup"):
        bind_device_groups([((a2, b2), _setup)], backend="local")


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_cross_group_mixed_ranks_broadcast_and_gather():
    # SAME broadcast/gather as above, but across SEPARATE groups (tier-2): the ranks=2 rollout
    # lives in its own actor mesh, the ranks=1 data/reward in another. Proves the mixed-rank data
    # plane is group-agnostic — dp_aligned is shape-based and the forwarded refs are global Ray
    # ObjectRefs pulled one-sided via the cross-group peer registry.
    ray = pytest.importorskip("ray")
    import ray as _ray

    if _ray.is_initialized():
        _ray.shutdown()
    policy = roles.rollout("policy", ranks=2)  # group A: 1x2 mesh
    data = roles.env("data")  # group B: 1x1 mesh
    reward = roles.reward("reward")
    bind_device_groups(
        [
            DeviceGroup(
                (
                    ActorGroup((policy,), _rollout_only_setup),
                    ActorGroup((data, reward), _data_reward_setup),
                ),
                device="cpu",
            )
        ],
        backend="ray",
        temp_dir="/tmp/rcray",
    )
    try:
        dcells = data.make().all()
        assert set(dcells) == {(0, 0)}  # data group is 1x1
        data_pid = dcells[(0, 0)]

        # cross-group broadcast low→high: both rollout ranks (group A) pull data's (0,0) (group B).
        pcells = policy.consume(data.make()).all()
        assert set(pcells) == {(0, 0), (0, 1)}
        assert pcells[(0, 0)][1] == data_pid == pcells[(0, 1)][1]
        assert pcells[(0, 0)][0] != pcells[(0, 1)][0]

        # cross-group gather high→low: reward (group B) takes rollout rank-0's (group A) output.
        rcells = reward.consume(policy.consume(data.make())).all()
        assert set(rcells) == {(0, 0)}
        assert rcells[(0, 0)][1] == data_pid
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_device_group_rejects_unknown_device():
    from cosmos_framework.posttrain.control.placement.colocate import ActorGroup, DeviceGroup
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    def _setup():
        return {"a": object()}

    a = WorkerGroup("a", "env")
    with pytest.raises(ValueError, match="must be 'cpu' or 'gpu'"):
        DeviceGroup((ActorGroup((a,), _setup),), device="tpu")


@pytest.mark.L0
@pytest.mark.CPU
def test_dist_scope_is_per_execution_replica_for_every_live_actor_group() -> None:
    """Policy HSDP factors its sole K-rank world; rollout replicas each own one K-rank TP world."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.control.placement.colocate import _dist_scope

    def _wg(role):
        # replicas/name are read by _dist_scope's trainer invariant.
        return SimpleNamespace(role=role, replicas=1, name=f"{role}0")

    assert _dist_scope((_wg("trainer"),)) == "replica"
    assert _dist_scope((_wg("trainer"), _wg("buffer"))) == "replica"
    assert _dist_scope((_wg("rollout"),)) == "replica"
    assert _dist_scope((_wg("env"), _wg("rollout"), _wg("reward"))) == "replica"
    assert _dist_scope((_wg("trainer"), _wg("rollout"))) == "replica"
    assert _dist_scope((_wg("env"),)) == "replica"


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_execution_replication_is_rejected_without_escape_hatch() -> None:
    """A trainer has one execution replica; model HSDP belongs inside its K-rank world."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.control.placement.colocate import _dist_scope

    def _wg(role, replicas=1):
        return SimpleNamespace(role=role, replicas=replicas, name=f"{role}0")

    with pytest.raises(ValueError, match="POLICY_REPLICAS"):
        _dist_scope((_wg("trainer", 2), _wg("rollout", 2)))

    # single replica: nothing to synchronize, must not raise
    assert _dist_scope((_wg("trainer", 1), _wg("rollout", 1))) == "replica"
    assert _dist_scope((_wg("env"), _wg("rollout"), _wg("reward"), _wg("trainer"))) == "replica"
    with pytest.raises(ValueError, match="POLICY_REPLICAS"):
        _dist_scope((_wg("trainer", 2),))
    assert _dist_scope((_wg("env", 2), _wg("rollout", 2))) == "replica"


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_model_parallelism_is_separate_from_placement_shape() -> None:
    from cosmos_framework.posttrain.control.placement import ActorGroup
    from cosmos_framework.posttrain.control.placement.colocate import _resolve_group_model_parallelism
    from cosmos_framework.posttrain.roles import roles

    policy = roles.trainer("policy", replicas=1, ranks=8, model_parallelism=(2, 4))
    actor_group = ActorGroup((policy,), lambda: {"policy": object()})
    assert _resolve_group_model_parallelism(
        actor_group,
        {"replicas": 1, "ranks": 8},
        {"policy": (1, 8)},
    ) == (2, 4)

    policy.model_parallelism = (2, 3)
    with pytest.raises(ValueError, match="must equal POLICY_RANKS=8"):
        _resolve_group_model_parallelism(
            actor_group,
            {"replicas": 1, "ranks": 8},
            {"policy": (1, 8)},
        )

    policy.replicas = 2
    with pytest.raises(ValueError, match="POLICY_REPLICAS=2"):
        _resolve_group_model_parallelism(
            actor_group,
            {"replicas": 2, "ranks": 8},
            {"policy": (2, 8)},
        )
