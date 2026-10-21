# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""ActorGroup / DeviceGroup placement and binding primitives (DESIGN §4).
This is placement/bind, NOT graph construction (§2.2: graph machinery is an async-only
optimization, skipped under sync). The B.1 train() loop then drives the bound WGs
directly.

ActorGroup declares process sharing. DeviceGroup declares CPU/GPU placement and same-device sharing.
The binder derives private Ray resource fractions, transport relationships, and disjoint GPU pools
from those typed declarations; resource arithmetic is never a public topology control.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from cosmos_framework.posttrain.comm.codec import DEFAULT_THRESHOLD
from cosmos_framework.posttrain.comm.weight_pairing import full_tensor_actor_targets
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

NIXL_PROBE_BYTES = 256 * 1024 * 1024
NIXL_PROBE_MIN_GBPS = 20.0
PLACEMENT_GROUP_READY_TIMEOUT_S = 300.0
BIND_IDENTITY_TIMEOUT_S = PLACEMENT_GROUP_READY_TIMEOUT_S
# TorchDistComm's process-group timeout is 300 seconds. The driver must wait longer so c10d can
# report the missing participant before bind cleanup terminates actors that are still rendezvousing.
R2R_PROCESS_GROUP_READY_TIMEOUT_S = 360.0


def _resolve_group_shapes(
    place: Any,
    wgs: tuple[WorkerGroup, ...],
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    """Resolve every WG once and derive the group's physical replicas×ranks mesh.

    Taking one snapshot keeps weight-edge validation, actor spawning, and per-WG dispatch on the
    same placement result when a future scheduler resolves shapes differently from WG declarations.

    Args:
        place: the placement resolver — ``place.resolve(wg)`` returns a WG's
            ``{"replicas", "ranks"}``.
        wgs: the WorkerGroups sharing this group's one mesh.

    Returns:
        The physical group shape (the per-axis max over members) and each WG's resolved shape.
    """
    wg_shapes: dict[str, tuple[int, int]] = {}
    for wg in wgs:
        resolved = place.resolve(wg)
        if not isinstance(resolved, Mapping):
            raise TypeError(f"Placement.resolve({wg.name!r}) must return a mapping, got {type(resolved).__name__}")
        axes: list[int] = []
        for axis in ("replicas", "ranks"):
            value = resolved.get(axis)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Placement.resolve({wg.name!r}) returned invalid {axis}={value!r}; "
                    "each ActorGroup axis must be a positive integer"
                )
            axes.append(value)
        wg_shapes[wg.name] = (axes[0], axes[1])
    group_shape = {
        "replicas": max(shape[0] for shape in wg_shapes.values()),
        "ranks": max(shape[1] for shape in wg_shapes.values()),
    }
    dominant_shape = (group_shape["replicas"], group_shape["ranks"])
    if dominant_shape not in wg_shapes.values():
        raise ValueError(
            "ActorGroup has no dominant member occupying its resolved physical shape "
            f"{dominant_shape}; member shapes are {list(wg_shapes.items())}. The per-axis maximum "
            "would create a sparse Cartesian product with actor cells owned by no WorkerGroup."
        )
    return group_shape, wg_shapes


def _ensure_ray(num_cpus: int, temp_dir: str | None) -> None:
    """Idempotently bring up (or join) Ray for a colocated ray-backend bind. Already initialized ⇒
    no-op. With ``RAY_ADDRESS`` set (the slurm multinode launcher exports the head ``<host>:<port>``),
    JOIN that cluster so its real resources apply (``num_cpus`` is NOT passed — invalid when
    connecting to an existing cluster). Otherwise start a local single-node cluster sized to
    ``num_cpus`` (the actor-count sizing hack, single-node only), rooted at ``temp_dir`` if given.

    Args:
        num_cpus: CPUs to size a locally-started cluster to; ignored when joining via
            ``RAY_ADDRESS``.
        temp_dir: session/spill root for a locally-started cluster; ignored when joining.
    """
    import ray

    if ray.is_initialized():
        return
    # Multinode: the slurm launcher starts a Ray cluster and exports RAY_ADDRESS=<head>:<port>.
    # When present, JOIN that cluster and let its real resources apply — do NOT pass num_cpus
    # (num_cpus is invalid/ignored when connecting to an existing cluster; the sizing hack below is
    # a single-node-only workaround). Absent ⇒ the unchanged local single-node init.
    addr = os.environ.get("RAY_ADDRESS")
    if addr:
        ray.init(address=addr, ignore_reinit_error=True, log_to_driver=False)
        return
    kwargs = dict(ignore_reinit_error=True, num_cpus=num_cpus, include_dashboard=False, log_to_driver=False)
    if temp_dir:
        kwargs["_temp_dir"] = temp_dir
    ray.init(**kwargs)


@dataclass(frozen=True)
class ActorGroup:
    """WorkerGroups sharing one Ray actor / OS process at every resolved physical cell."""

    members: tuple[WorkerGroup, ...]
    setup: Callable[[], dict[str, Any]]
    name: str | None = None
    # Extra environment for the actor processes of this group (Ray runtime_env env_vars).
    env_vars: Mapping[str, str] | None = None
    # Show each actor its node's other GPUs, its own first, before CUDA starts (vLLM custom
    # all-reduce maps tensor-parallel peers' buffers over CUDA IPC). See `_HostImpl.expose_peer_gpus`.
    expose_peer_gpus: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or any(not isinstance(member, WorkerGroup) for member in self.members):
            raise TypeError("ActorGroup.members must be a tuple of WorkerGroup values")
        if not isinstance(self.expose_peer_gpus, bool):
            raise TypeError(f"ActorGroup.expose_peer_gpus must be a bool, got {self.expose_peer_gpus!r}")
        if not self.members:
            raise ValueError("ActorGroup.members must not be empty")
        if not callable(self.setup):
            raise TypeError(f"ActorGroup.setup must be callable, got {type(self.setup).__name__}")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"ActorGroup.name must be a non-empty string when set, got {self.name!r}")


@dataclass(frozen=True)
class DeviceGroup:
    """ActorGroups mapped slot-by-slot to one CPU or GPU placement domain."""

    actor_groups: tuple[ActorGroup, ...]
    device: Literal["cpu", "gpu"]
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor_groups, tuple) or any(
            not isinstance(actor_group, ActorGroup) for actor_group in self.actor_groups
        ):
            raise TypeError("DeviceGroup.actor_groups must be a tuple of ActorGroup values")
        if not self.actor_groups:
            raise ValueError("DeviceGroup.actor_groups must not be empty")
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"DeviceGroup.device must be 'cpu' or 'gpu', got {self.device!r}")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"DeviceGroup.name must be a non-empty string when set, got {self.name!r}")


@dataclass(frozen=True)
class _ResolvedActorGroup:
    """One named ActorGroup after every member's placement shape is resolved exactly once."""

    name: str
    actor_group: ActorGroup
    device_group_name: str
    replicas: int
    ranks: int
    worker_shapes: tuple[tuple[str, int, int], ...]
    model_parallelism: tuple[int, int] | None
    dist_scope: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.replicas, self.ranks

    @property
    def cells(self) -> int:
        return self.replicas * self.ranks

    def shape_for(self, worker_name: str) -> tuple[int, int]:
        for name, replicas, ranks in self.worker_shapes:
            if name == worker_name:
                return replicas, ranks
        raise KeyError(f"ActorGroup {self.name!r} has no WorkerGroup {worker_name!r}")


@dataclass(frozen=True)
class _ResolvedDeviceGroup:
    """One explicit device placement domain containing resolved ActorGroups."""

    name: str
    device: Literal["cpu", "gpu"]
    actor_groups: tuple[_ResolvedActorGroup, ...]

    @property
    def gpu_cells(self) -> int:
        return self.actor_groups[0].cells if self.device == "gpu" else 0


@dataclass(frozen=True)
class _SpawnedActorGroup:
    """A resolved ActorGroup paired with its live Ray actor grid."""

    resolved: _ResolvedActorGroup
    grid: dict[tuple[int, int], tuple[str, Any]]


@dataclass(frozen=True)
class _WeightEdge:
    """The sole implicit trainer→rollout WEIGHT edge understood by the live binder."""

    policy_actor_group: str
    rollout_actor_group: str
    policy_names: tuple[str, ...]
    rollout_names: tuple[str, ...]
    policy_shape: tuple[int, int]
    rollout_shape: tuple[int, int]
    relationship: str


@dataclass(frozen=True)
class _RolloutBroadcast:
    """Bind-time identity of the one rollout mesh participating in R2R broadcast."""

    actor_group: str
    body_names: tuple[str, ...]
    shape: tuple[int, int]


def _require_model_wg_spans_group(
    actor_group: ActorGroup,
    role: str,
    group_shape: dict[str, int],
    wg_shapes: dict[str, tuple[int, int]],
) -> None:
    """Require a WEIGHT endpoint WG to occupy every cell in its physical ActorGroup mesh."""
    physical_shape = (group_shape["replicas"], group_shape["ranks"])
    for wg in actor_group.members:
        wg_shape = wg_shapes[wg.name]
        if wg.role == role and wg_shape != physical_shape:
            raise ValueError(
                f"weight_sync: the model-bearing {role} WG {wg.name!r} resolves to shape "
                f"(replicas={wg_shape[0]}, ranks={wg_shape[1]}), but its ActorGroup's resolved "
                f"shape is (replicas={physical_shape[0]}, ranks={physical_shape[1]}) — the max over "
                f"its resolved members {[(m.name, *wg_shapes[m.name]) for m in actor_group.members]}. "
                "A WEIGHT endpoint must span its ActorGroup mesh; shrink the oversized helper or grow "
                "the endpoint WG."
            )


def _resolve_group_model_parallelism(
    actor_group: ActorGroup,
    group_shape: dict[str, int],
    wg_shapes: dict[str, tuple[int, int]],
) -> tuple[int, int] | None:
    """Resolve one model role's DP factorization for an ActorGroup.

    Placement owns the role shape ``(R, K)``; the role declaration separately owns
    ``(dp_replicate, dp_shard)``. Online policy execution has exactly one replica, and the model
    factorization must cover that replica's complete K-rank Torch world.
    """
    model_roles = [
        wg for wg in actor_group.members if wg.role == "trainer" or getattr(wg, "model_parallelism", None) is not None
    ]
    if not model_roles:
        return None
    if len(model_roles) != 1:
        raise NotImplementedError(
            f"one ActorGroup supports at most one model world, got {[wg.name for wg in model_roles]}"
        )
    model_role = model_roles[0]
    is_trainer = model_role.role == "trainer"
    model_shape = wg_shapes[model_role.name]
    physical_shape = (group_shape["replicas"], group_shape["ranks"])
    if model_shape != physical_shape:
        subject = "trainer" if is_trainer else "model"
        raise ValueError(
            f"{subject} WG {model_role.name!r} resolves to {model_shape}, but its ActorGroup physical "
            f"shape is {physical_shape}; a {subject} must occupy the complete model world"
        )
    replicas, ranks = model_shape
    if replicas != 1:
        if is_trainer:
            raise ValueError(
                f"trainer WG {model_role.name!r} has POLICY_REPLICAS={replicas}; policy execution replicas "
                "must be 1, while HSDP replication belongs to model dp_replicate inside POLICY_RANKS"
            )
        raise ValueError(
            f"model WG {model_role.name!r} has execution replicas={replicas}; model execution replicas "
            "must be 1, while HSDP replication belongs to model dp_replicate inside its rank world"
        )
    parallelism = getattr(model_role, "model_parallelism", None) or (1, ranks)
    if (
        not isinstance(parallelism, tuple)
        or len(parallelism) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in parallelism)
    ):
        subject = "trainer" if is_trainer else "model"
        raise ValueError(
            f"{subject} WG {model_role.name!r} model_parallelism must be positive "
            f"(dp_replicate, dp_shard), got {parallelism!r}"
        )
    dp_replicate, dp_shard = parallelism
    if dp_replicate * dp_shard != ranks:
        rank_field = "POLICY_RANKS" if is_trainer else "ranks"
        subject = "trainer" if is_trainer else "model"
        raise ValueError(
            f"{subject} WG {model_role.name!r} model_parallelism dp_replicate={dp_replicate} * "
            f"dp_shard={dp_shard} must equal {rank_field}={ranks}"
        )
    return dp_replicate, dp_shard


def _resolve_device_groups(
    device_groups: list[DeviceGroup],
    place: Any,
) -> tuple[_ResolvedDeviceGroup, ...]:
    """Resolve and validate the complete actor/device topology exactly once.

    ActorGroup owns process sharing; DeviceGroup owns physical-device sharing. A GPU DeviceGroup
    aligns its child ActorGroups by row-major flat actor index, so their actor products must match.
    CPU DeviceGroups carry no device-affinity promise and therefore permit unequal products.
    """
    if not isinstance(device_groups, list) or any(not isinstance(group, DeviceGroup) for group in device_groups):
        raise TypeError("bind_device_groups expects a list of DeviceGroup values")
    if not device_groups:
        raise ValueError("bind_device_groups requires at least one DeviceGroup")

    resolved_devices: list[_ResolvedDeviceGroup] = []
    seen_device_names: set[str] = set()
    seen_actor_names: set[str] = set()
    seen_actor_ids: set[int] = set()
    seen_worker_ids: set[int] = set()
    seen_worker_names: set[str] = set()
    actor_index = 0
    for device_index, device_group in enumerate(device_groups):
        device_name = device_group.name or f"device_group_{device_index}"
        if device_name in seen_device_names:
            raise ValueError(f"DeviceGroup names must be unique, got duplicate {device_name!r}")
        seen_device_names.add(device_name)

        resolved_actors: list[_ResolvedActorGroup] = []
        for actor_group in device_group.actor_groups:
            if id(actor_group) in seen_actor_ids:
                raise ValueError("one ActorGroup cannot belong to more than one DeviceGroup")
            seen_actor_ids.add(id(actor_group))
            actor_name = actor_group.name or f"actor_group_{actor_index}"
            actor_index += 1
            if actor_name in seen_actor_names:
                raise ValueError(f"ActorGroup names must be unique, got duplicate {actor_name!r}")
            seen_actor_names.add(actor_name)
            for worker in actor_group.members:
                if id(worker) in seen_worker_ids:
                    raise ValueError(f"WorkerGroup {worker.name!r} cannot belong to more than one ActorGroup")
                seen_worker_ids.add(id(worker))
                if worker.name in seen_worker_names:
                    raise ValueError(f"WorkerGroup names must be unique across ActorGroups, got {worker.name!r}")
                seen_worker_names.add(worker.name)

            group_shape, worker_shape_map = _resolve_group_shapes(place, actor_group.members)
            model_parallelism = _resolve_group_model_parallelism(actor_group, group_shape, worker_shape_map)
            replicas = group_shape["replicas"]
            ranks = group_shape["ranks"]
            resolved_actors.append(
                _ResolvedActorGroup(
                    name=actor_name,
                    actor_group=actor_group,
                    device_group_name=device_name,
                    replicas=replicas,
                    ranks=ranks,
                    worker_shapes=tuple(
                        (worker.name, *worker_shape_map[worker.name]) for worker in actor_group.members
                    ),
                    model_parallelism=model_parallelism,
                    dist_scope=_dist_scope(actor_group.members, replicas=replicas),
                )
            )

        if device_group.device == "gpu":
            products = {actor_group.cells for actor_group in resolved_actors}
            if len(products) != 1:
                raise ValueError(
                    f"GPU DeviceGroup {device_name!r} requires every ActorGroup to have the same actor product "
                    f"R*K for flat-index device alignment, got "
                    f"{[(actor_group.name, actor_group.shape, actor_group.cells) for actor_group in resolved_actors]}"
                )
        resolved_devices.append(
            _ResolvedDeviceGroup(
                name=device_name,
                device=device_group.device,
                actor_groups=tuple(resolved_actors),
            )
        )
    return tuple(resolved_devices)


def _resolve_weight_edge(
    resolved_device_groups: tuple[_ResolvedDeviceGroup, ...],
) -> _WeightEdge | None:
    """Resolve the one trainer→rollout edge and derive its relationship from grouping alone."""
    resolved_actor_groups = tuple(
        actor_group for device_group in resolved_device_groups for actor_group in device_group.actor_groups
    )
    policy_members = [
        (actor_group, worker)
        for actor_group in resolved_actor_groups
        for worker in actor_group.actor_group.members
        if worker.role == "trainer"
    ]
    rollout_members = [
        (actor_group, worker)
        for actor_group in resolved_actor_groups
        for worker in actor_group.actor_group.members
        if worker.role == "rollout"
    ]
    if len(policy_members) > 1 or len(rollout_members) > 1:
        raise NotImplementedError(
            "weight_sync supports exactly one trainer→rollout WG pair, but got "
            f"{len(policy_members)} trainer WG(s) and {len(rollout_members)} rollout WG(s)"
        )
    if not policy_members or not rollout_members:
        return None

    policy_actor_group, policy_wg = policy_members[0]
    rollout_actor_group, rollout_wg = rollout_members[0]
    _require_model_wg_spans_group(
        policy_actor_group.actor_group,
        "trainer",
        {"replicas": policy_actor_group.replicas, "ranks": policy_actor_group.ranks},
        {name: (replicas, ranks) for name, replicas, ranks in policy_actor_group.worker_shapes},
    )
    _require_model_wg_spans_group(
        rollout_actor_group.actor_group,
        "rollout",
        {"replicas": rollout_actor_group.replicas, "ranks": rollout_actor_group.ranks},
        {name: (replicas, ranks) for name, replicas, ranks in rollout_actor_group.worker_shapes},
    )
    policy_shape = policy_actor_group.shape
    rollout_shape = rollout_actor_group.shape
    if policy_shape[0] != 1:
        raise ValueError(
            f"weight_sync policy shape {policy_shape} has POLICY_REPLICAS={policy_shape[0]}; "
            "policy execution replicas must be 1, while HSDP replication belongs to model "
            "dp_replicate inside POLICY_RANKS"
        )
    if policy_actor_group.name == rollout_actor_group.name:
        relationship = "same_process"
    elif policy_actor_group.device_group_name == rollout_actor_group.device_group_name:
        device_group = next(
            group for group in resolved_device_groups if group.name == policy_actor_group.device_group_name
        )
        relationship = "same_gpu" if device_group.device == "gpu" else "host"
    else:
        device_by_actor = {
            actor_group.name: device_group.device
            for device_group in resolved_device_groups
            for actor_group in device_group.actor_groups
        }
        if device_by_actor[policy_actor_group.name] == "cpu" or device_by_actor[rollout_actor_group.name] == "cpu":
            relationship = "host"
        else:
            relationship = "cross_device"

    return _WeightEdge(
        policy_actor_group=policy_actor_group.name,
        rollout_actor_group=rollout_actor_group.name,
        policy_names=(policy_wg.name,),
        rollout_names=(rollout_wg.name,),
        policy_shape=policy_shape,
        rollout_shape=rollout_shape,
        relationship=relationship,
    )


def _resolve_rollout_broadcast(
    resolved_device_groups: tuple[_ResolvedDeviceGroup, ...],
) -> _RolloutBroadcast:
    """Resolve and validate one complete GPU rollout mesh independently of a policy edge."""
    matches = [
        (device_group, actor_group, worker)
        for device_group in resolved_device_groups
        for actor_group in device_group.actor_groups
        for worker in actor_group.actor_group.members
        if worker.role == "rollout"
    ]
    if len(matches) != 1:
        raise NotImplementedError(
            f"replica broadcast supports exactly one rollout WorkerGroup, but placement contains {len(matches)}"
        )
    device_group, actor_group, worker = matches[0]
    if device_group.device != "gpu":
        raise ValueError("replica broadcast currently requires a GPU DeviceGroup and NCCL backend")
    _require_model_wg_spans_group(
        actor_group.actor_group,
        "rollout",
        {"replicas": actor_group.replicas, "ranks": actor_group.ranks},
        {name: (replicas, ranks) for name, replicas, ranks in actor_group.worker_shapes},
    )
    return _RolloutBroadcast(
        actor_group=actor_group.name,
        body_names=(worker.name,),
        shape=actor_group.shape,
    )


def _validate_weight_edge_topology(
    weight_edge: _WeightEdge,
    *,
    backend: str,
    transport: str | None,
    colocate: str,
    rollout_target_shape: tuple[int, int] | None = None,
) -> None:
    """Validate transport-specific product constraints after placement relationship is resolved."""
    policy_product = weight_edge.policy_shape[0] * weight_edge.policy_shape[1]
    full_rollout_product = weight_edge.rollout_shape[0] * weight_edge.rollout_shape[1]
    target_shape = weight_edge.rollout_shape if rollout_target_shape is None else rollout_target_shape
    rollout_product = target_shape[0] * target_shape[1]
    if policy_product == rollout_product:
        return
    if (
        rollout_target_shape is not None
        and weight_edge.relationship == "same_gpu"
        and policy_product == full_rollout_product
    ):
        # A same-GPU DeviceGroup has equal full physical products. Relay routing selects replica
        # zero's flat cells 0..K-1 and maps each to the same policy flat index, preserving device
        # alignment; the remaining policy cells are explicit idle collective participants.
        return
    if backend != "ray" or weight_edge.relationship != "cross_device" or transport != "nixl" or colocate != "pg":
        raise ValueError(
            "unequal-product weight_sync requires a Ray disaggregated cross_device WEIGHT edge "
            "with weight_sync_transport='nixl' and colocate='pg'; got "
            f"policy={weight_edge.policy_shape} ({policy_product} cells), "
            f"P2R target={target_shape} ({rollout_product} cells), backend={backend!r}, "
            f"relationship={weight_edge.relationship!r}, transport={transport!r}, colocate={colocate!r}"
        )


def _required_placement_gpus(resolved_device_groups: tuple[_ResolvedDeviceGroup, ...]) -> float:
    """Return the physical GPU count owned by explicit, mutually disjoint GPU DeviceGroups."""
    return float(sum(device_group.gpu_cells for device_group in resolved_device_groups))


def placement_gpu_count(device_groups: list[DeviceGroup]) -> int:
    """The GPUs the resolved placement of ``device_groups`` occupies; 0 for a CPU-only (local) placement.

    The same resolution ``bind_device_groups`` performs, so every GPU DeviceGroup counts once (a
    colocated pair shares its cards, disaggregated roles and an OPD teacher each add their own pool).
    Run metadata such as a telemetry world size reads it from here rather than inferring it from
    role shapes, which cannot see groups the entry placed separately.
    """
    from cosmos_framework.posttrain.control.placement.placement import Placement

    resolved = _resolve_device_groups(_as_device_groups(device_groups), Placement())
    return int(_required_placement_gpus(resolved))


def _declared_allocation_gpus(env: Mapping[str, str] | None = None) -> tuple[float, str] | None:
    """Return the GPU count the launcher declared for this job, or ``None`` when it did not.

    ``launch_multinode.sh`` exports ``NNODES`` and ``GPUS_PER_NODE`` alongside ``RAY_ADDRESS``.
    Unlike ``ray.cluster_resources()`` these are fixed for the job's whole lifetime, so they answer
    "is this allocation big enough" without racing worker registration.

    Returns the GPU count paired with a human-readable description of where it came from, so the
    rejection can name the two variables rather than only their product.

    A partial or unparseable declaration yields ``None`` rather than an error: the torchrun launcher
    exports ``GPUS_PER_NODE`` without ``NNODES``, and a capacity guard must not turn that into a
    failed job. Callers treat ``None`` as "not declared" and leave capacity to the runtime.
    """
    values = os.environ if env is None else env
    try:
        nodes = int(values["NNODES"])
        gpus_per_node = int(values["GPUS_PER_NODE"])
    except (KeyError, TypeError, ValueError):
        return None
    if nodes < 1 or gpus_per_node < 1:
        return None
    return float(nodes * gpus_per_node), f"NNODES={nodes} x GPUS_PER_NODE={gpus_per_node}"


def _ray_capacity_is_complete() -> bool:
    """Whether ``ray.cluster_resources()`` is a settled fact here rather than a converging one.

    A locally started cluster registers its own GPUs before ``ray.init`` returns, so its count is
    the whole truth and can still fail fast. A JOINED cluster (``RAY_ADDRESS``) keeps gaining
    workers while the driver runs, so its count is a lower bound and proves no shortage.
    """
    return not os.environ.get("RAY_ADDRESS")


def _validate_registered_gpu_capacity(ray_module: Any, required_gpus: float) -> None:
    """Reject a PG topology exceeding Ray's registered GPUs. Complete resource views only."""
    registered_gpus = float(ray_module.cluster_resources().get("GPU", 0.0))
    if required_gpus > registered_gpus + 1.0e-9:
        raise RuntimeError(
            f"GPU DeviceGroups require {required_gpus:g} disjoint physical GPUs, but Ray reports "
            f"only {registered_gpus:g} registered GPUs; increase the allocation or reduce the "
            "declared DeviceGroup pools"
        )


def _validate_declared_gpu_capacity(required_gpus: float, declared_gpus: float, source: str) -> None:
    """Reject a PG topology whose disjoint pools exceed the allocation declared for this job."""
    if required_gpus > declared_gpus + 1.0e-9:
        raise RuntimeError(
            f"GPU DeviceGroups require {required_gpus:g} disjoint physical GPUs, but the declared "
            f"allocation provides {declared_gpus:g} ({source}); increase the allocation or reduce "
            "the declared DeviceGroup pools"
        )


def _bundles_per_placement_group(nodes: list[dict[str, Any]], device_group: _ResolvedDeviceGroup) -> int:
    """Return a STRICT_PACK size that fits both GPU and per-actor CPU density on a live node."""
    cpus_per_bundle = len(device_group.actor_groups)
    capacities: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for node in nodes:
        resources = node.get("Resources", {})
        gpu_count = float(resources.get("GPU", 0.0))
        cpu_count = float(resources.get("CPU", 0.0))
        capacity = min(math.floor(gpu_count), math.floor(cpu_count / cpus_per_bundle))
        capacities.append(capacity)
        diagnostics.append(
            {
                "node": node.get("NodeID", node.get("NodeManagerAddress", "?")),
                "GPU": gpu_count,
                "CPU": cpu_count,
                "bundle_capacity": capacity,
            }
        )
    feasible_capacities = [capacity for capacity in capacities if capacity > 0]
    if not feasible_capacities:
        raise RuntimeError(
            f"GPU DeviceGroup {device_group.name!r} requires {cpus_per_bundle} CPUs per one-GPU "
            f"placement bundle, but no live Ray GPU node can host one; node capacities={diagnostics!r}"
        )
    # Keeping each PG within the smallest usable node lets Ray place it on any node that can host
    # one bundle, rather than creating a GPU-sized PG that only the highest-CPU node can satisfy.
    return min(feasible_capacities)


def _validate_peer_gpu_exposure(
    actor_groups: tuple[_ResolvedActorGroup, ...],
    *,
    colocate: str,
    bundles_per_pg: int | None,
    node_gpu_counts: tuple[int, ...] | None,
) -> None:
    """Allow peer visibility only when one TP replica consumes the node's complete GPU pool."""
    for actor_group in actor_groups:
        if not actor_group.actor_group.expose_peer_gpus:
            continue
        if colocate != "pg" or bundles_per_pg is None:
            raise ValueError(
                f"ActorGroup {actor_group.name!r} exposes peer GPUs but placement {colocate!r} "
                "cannot prove one complete node-local tensor-parallel replica"
            )
        if actor_group.ranks != bundles_per_pg:
            raise ValueError(
                f"ActorGroup {actor_group.name!r} exposes every GPU on its node, so its ranks must "
                f"equal the node placement capacity; got ranks={actor_group.ranks}, capacity={bundles_per_pg}"
            )
        mismatched = sorted({count for count in node_gpu_counts or () if count != actor_group.ranks})
        if mismatched:
            raise ValueError(
                f"ActorGroup {actor_group.name!r} exposes every physical GPU on its selected node, "
                f"so every placement candidate must have exactly ranks={actor_group.ranks} GPUs; "
                f"candidate GPU counts include {mismatched}"
            )


def _validate_host_affinity_resources(
    nodes: list[dict[str, Any]],
    actor_groups: tuple[_ResolvedActorGroup, ...],
    gpu_slots_per_host: int,
) -> None:
    """Require one exact full-pool raylet for every ActorGroup slot and pod block."""
    live_nodes = [node for node in nodes if node.get("Alive")]
    for actor_group in actor_groups:
        slot = actor_group.actor_group.members[0].placement.get("slot")
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"host_affinity ActorGroup {actor_group.name!r} lost its validated slot resource")
        pod_count = actor_group.cells // gpu_slots_per_host
        for pod_index in range(pod_count):
            pod_resource = f"node_{pod_index}"
            candidates = [
                node
                for node in live_nodes
                if float(node.get("Resources", {}).get(slot, 0.0)) > 0
                and float(node.get("Resources", {}).get(pod_resource, 0.0)) > 0
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"host_affinity ActorGroup {actor_group.name!r} requires exactly one Ray node "
                    f"with slot={slot!r} and {pod_resource!r}; got {len(candidates)} candidates"
                )
            resources = candidates[0].get("Resources", {})
            advertised = {
                "GPU": float(resources.get("GPU", 0.0)),
                slot: float(resources.get(slot, 0.0)),
                pod_resource: float(resources.get(pod_resource, 0.0)),
            }
            if any(capacity != float(gpu_slots_per_host) for capacity in advertised.values()):
                raise RuntimeError(
                    f"host_affinity ActorGroup {actor_group.name!r} requires a complete physical GPU "
                    f"pool with exact resource capacity gpu_slots_per_host={gpu_slots_per_host}, "
                    f"but the selected raylet advertises {advertised!r}"
                )


def _ordered_gpu_uuids(
    device_group_name: str,
    actor_group: _ResolvedActorGroup,
    identities: dict[tuple[int, int], dict[str, Any]] | None,
) -> tuple[str, ...]:
    """Validate one ActorGroup identity snapshot and return UUIDs in its row-major flat order."""
    expected_cells = {(replica, rank) for replica in range(actor_group.replicas) for rank in range(actor_group.ranks)}
    if identities is None or set(identities) != expected_cells:
        raise RuntimeError(
            f"GPU identity proof for ActorGroup {actor_group.name!r} is incomplete: "
            f"expected cells={sorted(expected_cells)}, got={sorted(identities or {})}"
        )
    ordered: list[str] = []
    for flat_index in range(actor_group.cells):
        cell = (flat_index // actor_group.ranks, flat_index % actor_group.ranks)
        identity = identities[cell]
        cuda_uuid = identity.get("cuda_uuid")
        if not isinstance(cuda_uuid, str) or not cuda_uuid:
            raise RuntimeError(
                f"GPU identity proof for DeviceGroup {device_group_name!r}, "
                f"ActorGroup {actor_group.name!r}, flat slot {flat_index} cell {cell} returned "
                f"no CUDA UUID: {identity!r}"
            )
        ordered.append(cuda_uuid)
    if len(set(ordered)) != len(ordered):
        raise RuntimeError(
            f"GPU DeviceGroup {device_group_name!r} maps multiple flat slots of ActorGroup "
            f"{actor_group.name!r} to the same CUDA UUID: {ordered}"
        )
    return tuple(ordered)


def _align_host_affinity_grids_by_uuid(
    resolved_device_groups: tuple[_ResolvedDeviceGroup, ...],
    spawned_by_name: dict[str, _SpawnedActorGroup],
    identities_by_actor_group: dict[str, dict[tuple[int, int], dict[str, Any]]],
) -> None:
    """Assign shared-pool host-affinity actors to logical flat slots by physical CUDA UUID.

    The pod resource selects a GPU pool but independent raylets may enumerate that pool in different
    orders. Actors have not run setup or rendezvous yet, so each non-canonical ActorGroup grid can be
    safely re-keyed to the first ActorGroup's UUID order. The ``(store_id, actor)`` tuple stays intact
    so the already-built peer registry remains a valid store-id-to-actor mapping. Consequently the
    store-id suffix records the actor's original spawn cell, not its post-alignment logical cell.
    """
    plans: list[
        tuple[
            dict[tuple[int, int], tuple[str, Any]],
            dict[tuple[int, int], tuple[str, Any]],
            str,
            dict[tuple[int, int], dict[str, Any]],
        ]
    ] = []
    for device_group in resolved_device_groups:
        if device_group.device != "gpu" or len(device_group.actor_groups) < 2:
            continue
        canonical_actor = device_group.actor_groups[0]
        canonical_uuids = _ordered_gpu_uuids(
            device_group.name,
            canonical_actor,
            identities_by_actor_group.get(canonical_actor.name),
        )
        canonical_pool = set(canonical_uuids)
        for actor_group in device_group.actor_groups[1:]:
            identities = identities_by_actor_group.get(actor_group.name)
            ordered = _ordered_gpu_uuids(device_group.name, actor_group, identities)
            actor_pool = set(ordered)
            if actor_pool != canonical_pool:
                raise RuntimeError(
                    f"host_affinity cannot pair ActorGroup {actor_group.name!r} with canonical "
                    f"ActorGroup {canonical_actor.name!r} in GPU DeviceGroup {device_group.name!r}: "
                    f"missing UUIDs={sorted(canonical_pool - actor_pool)}, "
                    f"unexpected UUIDs={sorted(actor_pool - canonical_pool)}"
                )
            if ordered == canonical_uuids:
                continue
            grid = spawned_by_name[actor_group.name].grid
            expected_cells = {
                (replica, rank) for replica in range(actor_group.replicas) for rank in range(actor_group.ranks)
            }
            if set(grid) != expected_cells:
                raise RuntimeError(
                    f"host_affinity grid for ActorGroup {actor_group.name!r} is incomplete: "
                    f"expected cells={sorted(expected_cells)}, got={sorted(grid)}"
                )
            if identities is None:
                raise RuntimeError(
                    f"internal host_affinity invariant failed: ActorGroup {actor_group.name!r} "
                    "lost its validated GPU identity snapshot"
                )
            source_cells = tuple(
                (flat_index // actor_group.ranks, flat_index % actor_group.ranks)
                for flat_index in range(actor_group.cells)
            )
            source_cell_by_uuid = dict(zip(ordered, source_cells, strict=True))
            aligned_grid: dict[tuple[int, int], tuple[str, Any]] = {}
            aligned_identities: dict[tuple[int, int], dict[str, Any]] = {}
            for flat_index, cuda_uuid in enumerate(canonical_uuids):
                target_cell = (flat_index // actor_group.ranks, flat_index % actor_group.ranks)
                source_cell = source_cell_by_uuid[cuda_uuid]
                aligned_grid[target_cell] = grid[source_cell]
                aligned_identities[target_cell] = identities[source_cell]
            plans.append((grid, aligned_grid, actor_group.name, aligned_identities))

    # Apply only after every DeviceGroup has a complete bijection, so a later mismatch cannot leave
    # earlier grids partially re-keyed before the outer bind guard rolls back their actors.
    for grid, aligned_grid, actor_name, aligned_identities in plans:
        grid.clear()
        grid.update(aligned_grid)
        identities_by_actor_group[actor_name] = aligned_identities


def _validate_gpu_identities(
    resolved_device_groups: tuple[_ResolvedDeviceGroup, ...],
    identities_by_actor_group: dict[str, dict[tuple[int, int], dict[str, Any]]],
) -> None:
    """Prove same-DeviceGroup sharing and cross-DeviceGroup disjointness with CUDA UUIDs."""
    device_uuid_pools: dict[str, set[str]] = {}
    for device_group in resolved_device_groups:
        if device_group.device == "cpu":
            continue
        canonical_uuids: tuple[str, ...] | None = None
        for actor_group in device_group.actor_groups:
            ordered = _ordered_gpu_uuids(
                device_group.name,
                actor_group,
                identities_by_actor_group.get(actor_group.name),
            )
            if canonical_uuids is None:
                canonical_uuids = ordered
            elif len(ordered) != len(canonical_uuids):
                canonical_actor = device_group.actor_groups[0]
                raise RuntimeError(
                    f"internal GPU identity invariant failed for DeviceGroup {device_group.name!r}: "
                    f"ActorGroup {canonical_actor.name!r} reported {len(canonical_uuids)} flat slots, "
                    f"while ActorGroup {actor_group.name!r} reported {len(ordered)}"
                )
            elif ordered != canonical_uuids:
                mismatch = next(index for index, pair in enumerate(zip(canonical_uuids, ordered)) if pair[0] != pair[1])
                canonical_actor = device_group.actor_groups[0]
                canonical_cell = (
                    mismatch // canonical_actor.ranks,
                    mismatch % canonical_actor.ranks,
                )
                actor_cell = (mismatch // actor_group.ranks, mismatch % actor_group.ranks)
                raise RuntimeError(
                    f"GPU DeviceGroup {device_group.name!r} flat slot {mismatch} is not the same physical GPU: "
                    f"ActorGroup {canonical_actor.name!r} cell {canonical_cell} has UUID "
                    f"{canonical_uuids[mismatch]!r}, while ActorGroup {actor_group.name!r} cell "
                    f"{actor_cell} has UUID {ordered[mismatch]!r}; diagnostics="
                    f"{identities_by_actor_group!r}"
                )
        if canonical_uuids is None:
            raise RuntimeError(
                f"internal GPU identity invariant failed: DeviceGroup {device_group.name!r} contains no ActorGroups"
            )
        device_uuid_pools[device_group.name] = set(canonical_uuids)

    device_names = tuple(device_uuid_pools)
    for left_index, left_name in enumerate(device_names):
        for right_name in device_names[left_index + 1 :]:
            overlap = device_uuid_pools[left_name] & device_uuid_pools[right_name]
            if overlap:
                raise RuntimeError(
                    f"GPU DeviceGroups {left_name!r} and {right_name!r} must own disjoint physical GPU pools, "
                    f"but both resolved CUDA UUIDs {sorted(overlap)}"
                )


def _cleanup_ray_bind(
    ray_module: Any,
    spawned: list[_SpawnedActorGroup],
    placement_groups: list[Any],
) -> None:
    """Best-effort rollback for a bind that failed after reserving Ray resources."""
    for spawned_group in reversed(spawned):
        for _store_id, host in spawned_group.grid.values():
            try:
                ray_module.kill(host, no_restart=True)
            except Exception:  # noqa: BLE001 -- preserve the original bind failure
                pass
    try:
        from ray.util.placement_group import remove_placement_group

        for group in reversed(placement_groups):
            try:
                remove_placement_group(group)
            except Exception:  # noqa: BLE001 -- preserve the original bind failure
                pass
    except Exception:  # noqa: BLE001 -- Ray may be partially initialized
        pass


def _as_device_groups(items: list[DeviceGroup]) -> list[DeviceGroup]:
    """Validate the canonical typed device-group list before any placement work."""
    bad = [group for group in items if not isinstance(group, DeviceGroup)]
    if bad:
        raise TypeError(f"bind_device_groups expects a list of DeviceGroup entries; got {type(bad[0]).__name__}")
    return items


def _dist_scope(members: tuple[WorkerGroup, ...], replicas: int | None = None) -> str:
    """Return the per-execution-replica rendezvous used by live ActorGroups.

    Rollout/teacher replicas each need an independent K-rank vLLM TP world. A policy has exactly one
    execution replica, so its one K-rank world contains the complete model-owned
    ``dp_replicate × dp_shard`` DeviceMesh. No live binder path uses placement R as a Torch mesh axis.

    Args:
        members: the colocation's WorkerGroups.
        replicas: the RESOLVED replica count for this colocation. Passed in rather
            than recomputed so the pre-spawn guard and the spawn-time call cannot disagree; None
            falls back to the max over `members`.

    Returns:
        Always ``"replica"`` for the current live topology.

    Raises:
        ValueError: a trainer ActorGroup has more than one execution replica.
    """
    model_roles = [wg for wg in members if wg.role == "trainer" or getattr(wg, "model_parallelism", None) is not None]
    r_max = replicas if replicas is not None else max((wg.replicas for wg in members), default=1)
    if model_roles and r_max != 1:
        if any(wg.role == "trainer" for wg in model_roles):
            raise ValueError(
                f"ActorGroup {sorted(wg.name for wg in members)} contains a trainer at "
                f"POLICY_REPLICAS={r_max}; policy execution replicas must be 1. Configure HSDP through "
                "the trainer's model_parallelism=(dp_replicate, dp_shard) inside POLICY_RANKS."
            )
        raise ValueError(
            f"ActorGroup {sorted(wg.name for wg in members)} contains a model role at "
            f"execution replicas={r_max}; model execution replicas must be 1. Configure HSDP through "
            "the role's model_parallelism=(dp_replicate, dp_shard) inside its rank world."
        )
    return "replica"


def _cross_device_transport(transport: str | None) -> tuple[str | None, str]:
    """Pick the transport hint for a `cross_device` weight edge.

    Returns:
        `(hint, reason)`. `hint is None` means the edge is unservable and `reason` is the message
        to fail with — the NCCL path is still not auto-wired (`_Host.init_weight_plane` is never
        called at bind), so a comm-less `NcclMover` would HANG at the first push rather than
        error, which is why this refuses at bind instead of deferring.
    """
    if transport is None:
        return None, (
            "cross-device (disaggregated / different-GPU) weight_sync is not auto-wired by "
            "bind_device_groups yet: it does not call _Host.init_weight_plane to build the "
            "weight plane, so the NcclMover would have no communicator. Put aligned trainer/rollout "
            "ActorGroups in one GPU DeviceGroup (→ IPC), use a CPU DeviceGroup (→ Copy), pass "
            "weight_sync_transport='nixl', or drive init_weight_plane manually (see "
            "tests/test_comm_weight_sync_e2e.py)."
        )
    _check_transport_hint(transport)
    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, why = nixl_available()
    if not ok:
        return None, f"weight_sync_transport={transport!r} requested but unavailable: {why}"
    return transport, ""


def _check_transport_hint(transport: str) -> None:
    """Reject a `weight_sync_transport` that no mover is registered under.

    Takes the STRING rather than a bool. Comparing `weight_sync_transport == "nixl"` and passing
    the result folded every typo (`"NIXL"`, `"ucx"`, a stray space) into "no transport requested",
    which then produced the NCCL guidance telling the caller to *pass* `weight_sync_transport=
    'nixl'` — advice they had just followed.

    `_HINT_MOVERS` is populated by import SIDE EFFECT (`workers/nixl.py` registers itself), so
    "is this hint real?" can only be answered once the owning module is imported; import the one
    hinted mover we ship first, then let `movers.py` own the message — it lists what is
    registered and names the module to import, and bind previously made that unreachable.
    """
    import cosmos_framework.posttrain.workers.nixl  # noqa: F401 — registers hint 'nixl'
    from cosmos_framework.posttrain.comm.transfer.movers import _HINT_MOVERS

    if transport not in _HINT_MOVERS:
        known = ", ".join(sorted(_HINT_MOVERS)) or "<none registered>"
        raise ValueError(
            f"unknown weight_sync_transport {transport!r} (registered: {known}). Pass None to use "
            f"the relationship default, or a registered hint."
        )


def _run_nixl_probe(
    source_host: Any,
    target_host: Any,
    nbytes: int,
    min_gbps: float,
    *,
    buffer_bytes: int | None = None,
) -> dict[str, Any]:
    """Warm the peer route, then time a real actor-to-actor transfer before model setup."""
    import gc

    import ray

    handle: Any = None
    primary_error: BaseException | None = None
    try:
        # Three passes over the same actor-owned buffers, each with its own fill so a stale or no-op
        # transfer cannot pass the value witness. The first establishes NIXL remote-agent metadata and
        # is ~8x slower than the steady route, so it is never timed. A single timed transfer's wall
        # clock still carries a fixed cost of a few ms (Ray's RDT control hop plus NIXL descriptor
        # exchange) that has nothing to do with the fabric; at 256 MiB it is a third of the total on
        # InfiniBand. The third pass, at 1/8 the size, gives the slope between the two sizes, which is
        # the fabric rate the floor gates.
        def _timed_pass(size: int, fill: int) -> dict[str, Any]:
            nonlocal handle
            handle = ray.get(source_host.nixl_probe_publish.remote(int(size), buffer_bytes, fill))
            result = dict(ray.get(target_host.nixl_probe_fetch.remote(handle, fill)))
            handle = None
            gc.collect()
            if int(result.get("bytes", -1)) != int(size):
                raise RuntimeError(f"NIXL capability probe moved {result.get('bytes')} bytes, expected {size}")
            return result

        _timed_pass(int(nbytes), 0xA5)  # warm-up: establishes remote-agent metadata, ~8x slower
        report = _timed_pass(int(nbytes), 0x5A)
        report["source_node_id"] = str(ray.get(source_host.node_id.remote()))
        raw_gbps = float(report.get("gbps", 0.0))
        small_bytes = int(nbytes) // 8
        measured = raw_gbps
        if small_bytes >= 2**20 and float(report.get("seconds", 0.0)) > 0.0:
            small = _timed_pass(small_bytes, 0x3C)
            delta_seconds = float(report["seconds"]) - float(small.get("seconds", 0.0))
            # The slope is only meaningful when the extra bytes account for a real share of the wall
            # clock; when the fixed cost dominates both passes, the single-transfer figure (a lower
            # bound on the fabric rate) is kept instead of a noisy quotient of two tiny numbers.
            if delta_seconds >= 0.25 * float(report["seconds"]):
                measured = (int(nbytes) - small_bytes) / delta_seconds / 1e9
            report["overhead_seconds"] = max(
                float(small.get("seconds", 0.0)) - small_bytes / max(measured, 1e-9) / 1e9, 0.0
            )
        report["fabric_gbps"] = measured
        logger.info(
            f"NIXL capability probe: fabric {measured:.2f} GB/s (single {int(nbytes) / 2**20:.0f} MiB transfer "
            f"{raw_gbps:.2f} GB/s in {float(report.get('seconds', 0.0)) * 1e3:.1f} ms, fixed cost "
            f"{float(report.get('overhead_seconds', 0.0)) * 1e3:.1f} ms; floor {float(min_gbps):.2f} GB/s), "
            f"source_node={report['source_node_id'][:8]} target_node={str(report.get('node_id', ''))[:8]}"
        )
        if measured < float(min_gbps):
            raise RuntimeError(
                f"NIXL capability probe measured {measured:.2f} GB/s below the {min_gbps:.2f} GB/s "
                "fabric floor; check UCX_TLS/UCX_NET_DEVICES, CUDA-enabled UCX, and fabric routing"
            )
        return report
    except BaseException as exc:  # noqa: BLE001 -- preserve this failure across best-effort cleanup
        primary_error = exc
        raise
    finally:
        # The producer's wait_tensor_freed guard keys on the published tensor object's lifetime.
        # Drop the driver copy of the nested ObjectRef before asking the actor to release its buffer.
        handle = None
        gc.collect()
        try:
            ray.get([source_host.nixl_probe_release.remote(), target_host.nixl_probe_release.remote()])
        except BaseException as cleanup_error:  # noqa: BLE001 -- do not mask the capability failure
            if primary_error is None:
                raise
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(f"NIXL probe cleanup also failed: {cleanup_error!r}")


def _wait_placement_groups(
    ray_module: Any,
    placement_groups: list[Any],
    *,
    required_gpus: float,
    timeout_s: float = PLACEMENT_GROUP_READY_TIMEOUT_S,
) -> None:
    """Bound placement readiness so an oversized topology rolls back instead of hanging forever."""
    try:
        ray_module.get([group.ready() for group in placement_groups], timeout=float(timeout_s))
    except ray_module.exceptions.GetTimeoutError as exc:
        registered_gpus = float(ray_module.cluster_resources().get("GPU", 0.0))
        raise RuntimeError(
            f"placement groups were not ready within {timeout_s:.0f}s: topology requires "
            f"{required_gpus:g} GPUs, Ray reports {registered_gpus:g} registered GPUs; "
            "check allocation size, worker registration, and placement constraints"
        ) from exc


def bind_device_groups(
    device_groups: list[DeviceGroup],
    backend: str = "ray",
    num_cpus: int = 2,
    temp_dir: str | None = "/tmp/rcray",
    threshold: int = DEFAULT_THRESHOLD,
    weight_sync: bool = False,
    colocate: str = "pg",
    gpu_slots_per_host: int = 1,
    host_single_thread: bool = False,
    weight_sync_transport: str | None = None,
    weight_sync_bucket_bytes: int | None = None,
    weight_sync_probe_bytes: int = NIXL_PROBE_BYTES,
    weight_sync_min_gbps: float = NIXL_PROBE_MIN_GBPS,
    weight_sync_r2r_broadcast: bool = False,
    data_pool_config: dict[str, int] | None = None,
    gpu_node_domains: Mapping[str, str] | None = None,
) -> None:
    """Bind explicit process groups onto explicit CPU/GPU placement domains.

    ActorGroup membership is the only process-sharing truth. DeviceGroup membership is the only
    device-sharing truth: one GPU ActorGroup owns an exclusive pool; multiple GPU ActorGroups share
    each card by flat actor index; different GPU DeviceGroups receive disjoint pools. Ray fractional
    requests are derived privately and never select topology.
    """
    groups = _as_device_groups(device_groups)
    if backend not in ("ray", "local"):
        raise ValueError(f"unknown backend {backend!r}")
    if colocate not in ("pg", "host_affinity"):
        raise ValueError(f"colocate must be 'pg' or 'host_affinity', got {colocate!r}")
    if isinstance(gpu_slots_per_host, bool) or not isinstance(gpu_slots_per_host, int) or gpu_slots_per_host < 1:
        raise ValueError(f"gpu_slots_per_host must be a positive integer, got {gpu_slots_per_host!r}")
    if weight_sync_bucket_bytes is not None and int(weight_sync_bucket_bytes) < 1:
        raise ValueError(f"weight_sync_bucket_bytes must be positive when set, got {weight_sync_bucket_bytes}")
    if int(weight_sync_probe_bytes) < 1:
        raise ValueError(f"weight_sync_probe_bytes must be positive, got {weight_sync_probe_bytes}")
    if not math.isfinite(float(weight_sync_min_gbps)) or float(weight_sync_min_gbps) < 0:
        raise ValueError(f"weight_sync_min_gbps must be finite and >= 0, got {weight_sync_min_gbps}")
    if not weight_sync and weight_sync_transport is not None:
        raise ValueError(f"weight_sync_transport={weight_sync_transport!r} requires weight_sync=True")
    if gpu_node_domains is not None:
        from cosmos_framework.posttrain.control.placement.topology import validate_gpu_node_domains

        validate_gpu_node_domains(gpu_node_domains)
    if not isinstance(weight_sync_r2r_broadcast, bool):
        raise TypeError(f"weight_sync_r2r_broadcast must be bool, got {weight_sync_r2r_broadcast!r}")
    if data_pool_config is not None:
        required_pool_keys = {
            "page_bytes",
            "cuda_capacity_bytes",
            "cpu_capacity_bytes",
            "cuda_cache_bytes",
            "cpu_cache_bytes",
            "cpu_min_bytes",
        }
        missing_pool_keys = required_pool_keys.difference(data_pool_config)
        if missing_pool_keys:
            raise ValueError(f"data_pool_config is missing keys {sorted(missing_pool_keys)}")

    from cosmos_framework.posttrain.control.placement.placement import Placement

    resolved_devices = _resolve_device_groups(groups, Placement())
    resolved_actors = tuple(
        actor_group for device_group in resolved_devices for actor_group in device_group.actor_groups
    )
    topology_shard_degree: int | None = None
    if gpu_node_domains:
        gpu_devices = [group for group in resolved_devices if group.device == "gpu"]
        if backend != "ray" or colocate != "pg" or len(gpu_devices) != 1:
            raise ValueError("gpu_node_domains requires Ray PG placement with exactly one shared GPU DeviceGroup")
        model_meshes = {actor.model_parallelism for actor in gpu_devices[0].actor_groups if actor.model_parallelism}
        if len(model_meshes) != 1:
            raise ValueError("gpu_node_domains requires one unambiguous policy model mesh")
        topology_shard_degree = next(iter(model_meshes))[1]
    if backend == "local" and any(device_group.device == "gpu" for device_group in resolved_devices):
        raise ValueError(
            "backend='local' has no Ray GPU placement or CUDA UUID proof; use CPU DeviceGroups for "
            "the local/torchrun execution path"
        )
    if backend == "local" and colocate != "pg":
        raise ValueError("backend='local' does not support host_affinity; it is a Ray GPU placement mechanism")
    if weight_sync_r2r_broadcast and backend != "ray":
        raise ValueError("weight_sync_r2r_broadcast currently requires backend='ray' for NCCL rollout actors")
    rollout_broadcast = _resolve_rollout_broadcast(resolved_devices) if weight_sync_r2r_broadcast else None
    weight_edge = _resolve_weight_edge(resolved_devices) if weight_sync else None
    weight_hint: str | None = None
    if weight_sync and weight_edge is None and weight_sync_transport is not None:
        raise ValueError(
            f"weight_sync_transport={weight_sync_transport!r} was configured, but no trainer→rollout WEIGHT edge exists"
        )
    if weight_edge is not None:
        _validate_weight_edge_topology(
            weight_edge,
            backend=backend,
            transport=weight_sync_transport,
            colocate=colocate,
            rollout_target_shape=(1, weight_edge.rollout_shape[1]) if rollout_broadcast is not None else None,
        )
        if weight_edge.relationship == "cross_device":
            if backend != "ray":
                raise ValueError("cross-device DeviceGroups require backend='ray'")
            weight_hint, reason = _cross_device_transport(weight_sync_transport)
            if weight_hint is None:
                raise NotImplementedError(reason)
        elif weight_sync_transport is not None:
            _check_transport_hint(weight_sync_transport)
            raise ValueError(
                f"weight_sync_transport={weight_sync_transport!r} was passed, but grouping resolved "
                f"relationship={weight_edge.relationship!r}. Transport hints apply to cross_device "
                "WEIGHT edges only."
            )

    if backend == "local":
        bodies_by_actor: dict[str, dict[str, Any]] = {}
        for actor_group in resolved_actors:
            bodies = actor_group.actor_group.setup()
            bodies_by_actor[actor_group.name] = bodies
            expected_names = {worker.name for worker in actor_group.actor_group.members}
            if set(bodies) != expected_names:
                raise RuntimeError(
                    f"ActorGroup {actor_group.name!r} setup returned bodies {sorted(bodies)}, "
                    f"expected exactly {sorted(expected_names)}"
                )

        if weight_edge is not None and weight_edge.relationship == "same_process":
            bodies = bodies_by_actor[weight_edge.policy_actor_group]
            policy_body = bodies[weight_edge.policy_names[0]]
            rollout_body = bodies[weight_edge.rollout_names[0]]
            policy_model = getattr(policy_body, "model", None)
            rollout_model = getattr(rollout_body, "model", None)
            if policy_model is None or rollout_model is None:
                raise RuntimeError("same-process weight sync requires both bodies to expose model storage")
            from cosmos_framework.posttrain.comm.weight_transfer import shared_model_identity

            policy_identity = shared_model_identity(policy_model)
            rollout_identity = shared_model_identity(rollout_model)
            if policy_identity != rollout_identity:
                raise RuntimeError(
                    "same-process SharedMover requires policy and rollout to share exact model storage; "
                    f"policy identity={policy_identity!r}, rollout identity={rollout_identity!r}"
                )

        if weight_edge is not None:
            from cosmos_framework.posttrain.comm.transfer.movers import resolve_mover
            from cosmos_framework.posttrain.comm.weight_transfer import WeightSync

            participant_names: dict[str, list[str]] = {}
            participant_names.setdefault(weight_edge.policy_actor_group, []).extend(weight_edge.policy_names)
            participant_names.setdefault(weight_edge.rollout_actor_group, []).extend(weight_edge.rollout_names)
            for actor_name, body_names in participant_names.items():
                for body_name in dict.fromkeys(body_names):
                    body = bodies_by_actor[actor_name][body_name]
                    if not hasattr(body, "model"):
                        raise RuntimeError(
                            f"declared weight endpoint body {body_name!r} in ActorGroup {actor_name!r} exposes no model"
                        )
                    body._weight_backend = WeightSync(
                        resolve_mover(
                            relationship=weight_edge.relationship,
                            plane="weight",
                            capability="tensor",
                            hint=weight_hint,
                        ),
                        transfer_dtype=getattr(body, "transfer_dtype", None),
                    )
        for actor_group in resolved_actors:
            bodies = bodies_by_actor[actor_group.name]
            for worker in actor_group.actor_group.members:
                worker.body = bodies[worker.name]
                worker._bind_backend(None)
        return

    import ray

    from cosmos_framework.posttrain.workers.grid import GridBackend, setup_grid, spawn_grid

    if colocate == "host_affinity" and any(device_group.device != "gpu" for device_group in resolved_devices):
        raise ValueError("host_affinity is a GPU placement mechanism; CPU DeviceGroups carry no host-affinity promise")
    if colocate == "host_affinity":
        for actor_group in resolved_actors:
            slots = {worker.placement.get("slot") for worker in actor_group.actor_group.members}
            if len(slots) != 1 or not all(isinstance(slot, str) and slot for slot in slots):
                raise ValueError(
                    f"host_affinity ActorGroup {actor_group.name!r} requires one consistent non-empty "
                    f"placement slot across all members, got {slots!r}"
                )
            if actor_group.ranks != 1:
                raise ValueError(
                    f"host_affinity ActorGroup {actor_group.name!r} requires ranks=1 because UUID "
                    "alignment occurs before setup and this path has no distributed rendezvous; "
                    f"got shape={actor_group.shape}"
                )
        actor_slots = [actor_group.actor_group.members[0].placement["slot"] for actor_group in resolved_actors]
        if len(actor_slots) != len(set(actor_slots)):
            raise ValueError(
                "host_affinity requires a distinct raylet slot per ActorGroup so each can cover "
                f"its full physical GPU pool, got {actor_slots!r}"
            )
        for actor_group in resolved_actors:
            if actor_group.cells % gpu_slots_per_host:
                raise ValueError(
                    f"host_affinity requires complete per-host GPU pools for deterministic UUID "
                    f"placement: ActorGroup {actor_group.name!r} has {actor_group.cells} cells, "
                    f"which is not divisible by gpu_slots_per_host={gpu_slots_per_host}"
                )
        _validate_peer_gpu_exposure(
            resolved_actors,
            colocate=colocate,
            bundles_per_pg=None,
            node_gpu_counts=None,
        )

    total_actors = sum(actor_group.cells for actor_group in resolved_actors)
    required_placement_gpus = _required_placement_gpus(resolved_devices)
    # Reject an oversized topology against the DECLARED allocation, before Ray starts at all.
    # ray.cluster_resources() cannot answer this on a joined cluster: launch_multinode.sh starts the
    # workers in the background and runs the driver immediately ("No barrier ... Ray does the wait
    # for all nodes for us"), so the registered count is still climbing here and a shortfall proves
    # nothing. The declared allocation is fixed for the job, so it fails fast without that race, and
    # the bounded placement-group wait below stays authoritative for everything it cannot prove.
    declared_allocation = _declared_allocation_gpus()
    if colocate == "pg" and required_placement_gpus and declared_allocation is not None:
        _validate_declared_gpu_capacity(required_placement_gpus, *declared_allocation)
    _ensure_ray(max(num_cpus, total_actors + 1), temp_dir)
    if colocate == "host_affinity":
        _validate_host_affinity_resources(ray.nodes(), resolved_actors, gpu_slots_per_host)
    # A locally started cluster owns every GPU it will ever have, so it can still reject an
    # oversized topology here even when no allocation was declared.
    if colocate == "pg" and required_placement_gpus and _ray_capacity_is_complete():
        _validate_registered_gpu_capacity(ray, required_placement_gpus)
    placement_plans: dict[str, tuple[list[Any], int, float]] = {
        actor_group.name: ([], actor_group.cells, 0.0) for actor_group in resolved_actors
    }
    all_placement_groups: list[Any] = []
    if colocate == "pg" and required_placement_gpus:
        from ray.util.placement_group import placement_group

        nodes = [node for node in ray.nodes() if node.get("Alive") and node["Resources"].get("GPU", 0)]
        if not nodes and not gpu_node_domains:
            raise RuntimeError("no Ray nodes with GPUs available")
        for device_group in resolved_devices:
            if device_group.device == "cpu":
                continue
            total_bundles = device_group.gpu_cells
            topology_nodes = ()
            if gpu_node_domains and topology_shard_degree is not None:
                from cosmos_framework.posttrain.control.placement.topology import wait_for_gpu_topology

                topology_nodes = wait_for_gpu_topology(
                    ray.nodes,
                    gpu_node_domains,
                    shard_degree=topology_shard_degree,
                    expected_gpus=total_bundles,
                    cpus_per_gpu=len(device_group.actor_groups),
                    rollout_degree=math.lcm(
                        *(
                            actor.ranks
                            for actor in device_group.actor_groups
                            if any(member.role == "rollout" for member in actor.actor_group.members)
                        )
                    ),
                    timeout_s=PLACEMENT_GROUP_READY_TIMEOUT_S,
                )
                bundles_per_pg = topology_nodes[0].gpus
                logger.info(
                    "Pinned GPU topology (domain, address, GPUs): %s",
                    [(node.domain, node.address, node.gpus) for node in topology_nodes],
                )
            else:
                bundles_per_pg = _bundles_per_placement_group(nodes, device_group)
            _validate_peer_gpu_exposure(
                device_group.actor_groups,
                colocate=colocate,
                bundles_per_pg=bundles_per_pg,
                node_gpu_counts=tuple(int(node.gpus) for node in topology_nodes)
                if topology_nodes
                else tuple(int(math.floor(float(node["Resources"].get("GPU", 0.0)))) for node in nodes),
            )
            pgs: list[Any] = []
            try:
                for pool_index in range((total_bundles + bundles_per_pg - 1) // bundles_per_pg):
                    size = min(bundles_per_pg, total_bundles - pool_index * bundles_per_pg)
                    node_resources = (
                        {topology_nodes[pool_index].node_resource: topology_nodes[pool_index].resource_per_gpu}
                        if topology_nodes
                        else {}
                    )
                    placement = placement_group(
                        # Ray actors that specify any resource (num_gpus here) default to one
                        # lifetime CPU, so every ActorGroup consumes one CPU from its slot bundle.
                        [{"GPU": 1.0, "CPU": len(device_group.actor_groups), **node_resources} for _ in range(size)],
                        strategy="STRICT_PACK",
                    )
                    pgs.append(placement)
                    all_placement_groups.append(placement)
                _wait_placement_groups(ray, pgs, required_gpus=required_placement_gpus)
            except BaseException:  # noqa: BLE001 -- roll back every reserved domain before re-raising
                _cleanup_ray_bind(ray, [], all_placement_groups)
                raise
            gpu_request = 1.0 / len(device_group.actor_groups)
            for actor_group in device_group.actor_groups:
                placement_plans[actor_group.name] = (pgs, bundles_per_pg, gpu_request)
    elif colocate == "host_affinity":
        # Distinct raylets each advertise their own logical copy of the physical GPU pool. Request one
        # full logical GPU per actor, then prove actual physical UUID identity below.
        for device_group in resolved_devices:
            for actor_group in device_group.actor_groups:
                placement_plans[actor_group.name] = ([], actor_group.cells, 1.0)

    spawned: list[_SpawnedActorGroup] = []
    peers: dict[str, Any] = {}
    endpoint_actor_names = (
        tuple(dict.fromkeys((weight_edge.policy_actor_group, weight_edge.rollout_actor_group)))
        if weight_edge is not None
        else ()
    )
    for actor_group in resolved_actors:
        pgs, bundles_per_pg, gpu_request = placement_plans[actor_group.name]
        members = actor_group.actor_group.members
        materialize = tuple(worker.name for worker in members if worker.needs_bulk)
        host_slot = members[0].placement.get("slot") if colocate == "host_affinity" else None
        try:
            grid = spawn_grid(
                actor_group.actor_group.setup,
                actor_group.replicas,
                actor_group.ranks,
                actor_group.name,
                gpu_request,
                threshold,
                materialize,
                init_dist=False if colocate == "host_affinity" else None,
                pgs=pgs,
                k_stride=actor_group.ranks,
                bundles_per_pg=bundles_per_pg,
                host_slot=host_slot,
                cells_per_host=gpu_slots_per_host,
                dist_scope=actor_group.dist_scope,
                host_single_thread=host_single_thread,
                env_vars=actor_group.actor_group.env_vars,
                expose_peer_gpus=actor_group.actor_group.expose_peer_gpus,
                run_setup=False,
                model_parallelism=actor_group.model_parallelism,
                enable_tensor_transport=(
                    data_pool_config is not None
                    or (
                        weight_edge is not None
                        and weight_edge.relationship == "cross_device"
                        and weight_hint == "nixl"
                        and actor_group.name in endpoint_actor_names
                    )
                ),
                data_pool_config=data_pool_config,
            )
        except BaseException:  # noqa: BLE001 -- release earlier grids and every placement group
            _cleanup_ray_bind(ray, spawned, all_placement_groups)
            raise
        spawned_group = _SpawnedActorGroup(resolved=actor_group, grid=grid)
        spawned.append(spawned_group)
        for store_id, host in grid.values():
            peers[store_id] = host
    spawned_by_name = {spawned_group.resolved.name: spawned_group for spawned_group in spawned}
    all_hosts = [host for spawned_group in spawned for _store_id, host in spawned_group.grid.values()]

    def _guard_bind(action: Callable[[], Any]) -> Any:
        try:
            return action()
        except BaseException:  # noqa: BLE001 -- preserve the original bind failure after rollback
            _cleanup_ray_bind(ray, spawned, all_placement_groups)
            raise

    def _collect_gpu_identities() -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
        identities: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        gpu_actor_names = tuple(
            actor_group.name
            for device_group in resolved_devices
            if device_group.device == "gpu"
            for actor_group in device_group.actor_groups
        )
        for actor_name in gpu_actor_names:
            grid = spawned_by_name[actor_name].grid
            cells = tuple(sorted(grid))
            values = ray.get(
                [grid[cell][1].gpu_identity.remote() for cell in cells],
                timeout=BIND_IDENTITY_TIMEOUT_S,
            )
            identities[actor_name] = dict(zip(cells, values, strict=True))
        return identities

    gpu_identities = _guard_bind(_collect_gpu_identities)
    if colocate == "host_affinity":
        _guard_bind(
            lambda: _align_host_affinity_grids_by_uuid(
                resolved_devices,
                spawned_by_name,
                gpu_identities,
            )
        )
    _guard_bind(lambda: _validate_gpu_identities(resolved_devices, gpu_identities))
    _guard_bind(lambda: ray.get([host.set_peers.remote(peers) for host in all_hosts]))

    if weight_edge is not None and weight_edge.relationship == "cross_device" and weight_hint == "nixl":
        policy_spawned = spawned_by_name[weight_edge.policy_actor_group]
        rollout_spawned = spawned_by_name[weight_edge.rollout_actor_group]
        from cosmos_framework.posttrain.comm.weight_transfer import BUCKET_BYTES

        _guard_bind(
            lambda: _run_nixl_probe(
                policy_spawned.grid[(0, 0)][1],
                rollout_spawned.grid[(0, 0)][1],
                int(weight_sync_probe_bytes),
                float(weight_sync_min_gbps),
                buffer_bytes=max(int(weight_sync_probe_bytes), int(weight_sync_bucket_bytes or BUCKET_BYTES)),
            )
        )

    for spawned_group in spawned:
        actor_group = spawned_group.resolved
        _guard_bind(
            lambda actor_group=actor_group, grid=spawned_group.grid: setup_grid(
                grid,
                actor_group.actor_group.setup,
                replicas=actor_group.replicas,
                ranks=actor_group.ranks,
                threshold=threshold,
                materialize_names=tuple(worker.name for worker in actor_group.actor_group.members if worker.needs_bulk),
                dist_scope=actor_group.dist_scope,
                model_parallelism=actor_group.model_parallelism,
                expected_names=tuple(worker.name for worker in actor_group.actor_group.members),
                data_pool_config=data_pool_config,
            )
        )

    if rollout_broadcast is not None:
        rollout_spawned = spawned_by_name[rollout_broadcast.actor_group]
        replicas, ranks = rollout_broadcast.shape

        def _configure_rollout_broadcast() -> None:
            source_ranks_by_node: dict[str, list[int]] = {}
            rollout_identities = gpu_identities[rollout_broadcast.actor_group]
            for rank_index in range(ranks):
                node_id = str(rollout_identities[(0, rank_index)]["ray_node_id"])
                source_ranks_by_node.setdefault(node_id, []).append(rank_index)
            allocations = tuple(source_ranks_by_node.values())
            allocated_endpoints = ray.get(
                [
                    rollout_spawned.grid[(0, rank_indices[0])][1].rendezvous_endpoints.remote(len(rank_indices))
                    for rank_indices in allocations
                ],
                timeout=BIND_IDENTITY_TIMEOUT_S,
            )
            endpoints: dict[int, tuple[str, int]] = {}
            for rank_indices, (store_host, store_ports) in zip(
                allocations,
                allocated_endpoints,
                strict=True,
            ):
                if len(store_ports) != len(rank_indices):
                    raise RuntimeError(
                        f"rendezvous allocator returned {len(store_ports)} ports for {len(rank_indices)} groups"
                    )
                endpoints.update(
                    {
                        rank_index: (store_host, int(store_port))
                        for rank_index, store_port in zip(rank_indices, store_ports, strict=True)
                    }
                )
            refs: list[Any] = []
            for rank_index in range(ranks):
                store_host, store_port = endpoints[rank_index]
                for replica_index in range(replicas):
                    spec = {
                        "store_host": store_host,
                        "store_port": int(store_port),
                        "rank": replica_index,
                        "world_size": replicas,
                        "source_rank": 0,
                        "backend": "nccl",
                        "bucket_bytes": weight_sync_bucket_bytes,
                    }
                    refs.append(
                        rollout_spawned.grid[(replica_index, rank_index)][1].set_weight_broadcast.remote(
                            spec,
                            rollout_broadcast.body_names,
                        )
                    )
            ray.get(refs, timeout=R2R_PROCESS_GROUP_READY_TIMEOUT_S)

        _guard_bind(_configure_rollout_broadcast)

    if weight_edge is not None:
        if weight_edge.relationship == "same_process":
            shared_grid = spawned_by_name[weight_edge.policy_actor_group].grid
            _guard_bind(
                lambda: ray.get(
                    [
                        host.validate_shared_weight_identity.remote(
                            weight_edge.policy_names,
                            weight_edge.rollout_names,
                        )
                        for _store_id, host in shared_grid.values()
                    ],
                    timeout=BIND_IDENTITY_TIMEOUT_S,
                )
            )

        def _body_names(actor_name: str) -> tuple[str, ...]:
            names: list[str] = []
            if actor_name == weight_edge.policy_actor_group:
                names.extend(weight_edge.policy_names)
            if actor_name == weight_edge.rollout_actor_group:
                names.extend(weight_edge.rollout_names)
            return tuple(dict.fromkeys(names))

        def _weight_hosts(actor_name: str) -> tuple[Any, ...]:
            grid = spawned_by_name[actor_name].grid
            if (
                rollout_broadcast is not None
                and actor_name == weight_edge.rollout_actor_group
                and actor_name != weight_edge.policy_actor_group
            ):
                return tuple(host for (replica, _rank), (_store_id, host) in grid.items() if replica == 0)
            return tuple(host for _store_id, host in grid.values())

        if weight_hint == "nixl":
            sizes = _guard_bind(
                lambda: ray.get(
                    [
                        host.init_nixl.remote(weight_sync_bucket_bytes, _body_names(actor_name))
                        for actor_name in endpoint_actor_names
                        for host in _weight_hosts(actor_name)
                    ]
                )
            )
            if not any(sizes):
                _cleanup_ray_bind(ray, spawned, all_placement_groups)
                raise RuntimeError(
                    f"NIXL weight transport: no actor allocated a staging buffer (sizes={sizes}); "
                    "no body exposes `.model`, so weight sync has nothing to transfer"
                )
            # A consumer must land the producer's bucket. Mixed-precision policy/rollout models can
            # derive different local plan sizes, and the registered buffer cannot grow after bind.
            largest_staging_buffer = max(sizes)
            _guard_bind(
                lambda: ray.get(
                    [
                        host.reserve_nixl.remote(largest_staging_buffer)
                        for actor_name in endpoint_actor_names
                        for host in _weight_hosts(actor_name)
                    ]
                )
            )
        spec = {
            "relationship": weight_edge.relationship,
            "hint": weight_hint,
            "bucket_bytes": weight_sync_bucket_bytes,
        }
        _guard_bind(
            lambda: ray.get(
                [
                    host.set_weight_backend.remote(spec, _body_names(actor_name))
                    for actor_name in endpoint_actor_names
                    for host in _weight_hosts(actor_name)
                ]
            )
        )

    policy_weight_routes: dict[Any, Any] | None = None
    policy_actor_name: str | None = None
    if weight_edge is not None and weight_edge.policy_actor_group != weight_edge.rollout_actor_group:
        p2r_target_shape = (
            (1, weight_edge.rollout_shape[1]) if rollout_broadcast is not None else weight_edge.rollout_shape
        )
        if rollout_broadcast is not None or weight_edge.policy_shape != p2r_target_shape:
            policy_weight_routes = full_tensor_actor_targets(weight_edge.policy_shape, p2r_target_shape)
            policy_actor_name = weight_edge.policy_actor_group

    for spawned_group in spawned:
        actor_group = spawned_group.resolved
        members = actor_group.actor_group.members
        worker_shapes = {name: (replicas, ranks) for name, replicas, ranks in actor_group.worker_shapes}
        routes = policy_weight_routes if actor_group.name == policy_actor_name else None
        dp_reduce_op = "identical" if any(worker.role == "trainer" for worker in members) else "mean"
        grid_backend = GridBackend(
            spawned_group.grid,
            actor_group.shape,
            dp_reduce_op=dp_reduce_op,
            wg_shapes=worker_shapes,
            weight_routes=routes,
            weight_broadcast_enabled=(
                rollout_broadcast is not None and actor_group.name == rollout_broadcast.actor_group
            ),
            peers=peers,
        )
        for worker in members:
            worker._bind_backend(grid_backend)
