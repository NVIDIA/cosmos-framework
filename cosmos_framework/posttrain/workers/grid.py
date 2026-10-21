# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GridBackend — one replicas×ranks _Host actor mesh per group (DESIGN §1.1/§4.2).

Per ActorGroup: R execution replicas, each with K worker ranks. Every WG method call fans out over
its own resolved sub-shape and aligns arguments through an AlignPolicy. Compact execution-replica
results merge through the Communicator; a policy has R=1 and identity-checks the numeric report its
K-rank Torch model world already reduced. The WG/Handle/loop contract is unchanged.

Arg alignment: a CollectiveHandle arg is aligned per consumer (r, k) and the producer
ObjectRef is forwarded into the actor nested in the args tuple; _Host.invoke resolves it
actor-side (Ray does not auto-resolve refs nested in a container, only direct .remote()
args). So the controller forwards only the tiny ref — never the producer's metadata payload
— and bulk still moves peer-to-peer at materialize (DESIGN §3.4/#7). The remaining efficiency
hooks (async overlap, SQ/CQ engine, real IPC/UCX/NCCL backend) are §3.2 future work.

AlignPolicy is §2.4 dispatch expressed at the mesh level (mesh→mesh rank↔rank mapping); this
version only implements dp_aligned — locality/reshard/gather are left as hooks, to migrate to
control/dispatch + edge(dispatch=) later.
"""

from __future__ import annotations

import os
import site
import sys
import sysconfig
from collections.abc import Mapping
from typing import Any, Callable


def bundle_index(r: int, k: int, k_own: int) -> int:
    """Row-major flat bundle index of cell (r, k) within a group of rank width ``k_own``
    (spec 2026-07-22 §4.2): ``global = r * k_own + k``. Cross-group cells with EQUAL flat index
    share a bundle (⇒ share a GPU); different cells of the SAME group get distinct bundles. Pure —
    the extracted seam the bind (colocate.py) and spawn_grid below both flatten through. At equal
    shapes (``k_own == K_max``) this is bit-identical to the former ``r * K_max + k`` layout.

    Args:
        r: execution-replica index of the cell.
        k: rank (intra-replica) index of the cell.
        k_own: rank width of the cell's group — the row stride.

    Returns:
        Flat row-major bundle index ``r * k_own + k``.
    """
    return r * k_own + k


def dp_aligned(producer_shape: tuple, consumer_shape: tuple, meta: Any = None) -> dict:
    """Default alignment: identity on the replica axis, mirror on the rank axis (fall back
    to rank-0 when the producer has fewer ranks). Returns {consumer (r,k): [producer (r,k),
    ...]}. Same shape ⇒ identity {(r,k): [(r,k)]}."""
    rp, kp = producer_shape
    rc, kc = consumer_shape
    return {(r, k): [(r if r < rp else 0, k if k < kp else 0)] for r in range(rc) for k in range(kc)}


# TODO(Option B — NCCL-broadcast align for large bulk): dp_aligned makes a low-rank producer
# feed a high-rank SPMD consumer by having EVERY consumer rank independently pull producer (r,0)
# (N object-store pulls of the same payload — fine for text/small metadata, wasteful for a big
# multimodal obs, §3.4). This still applies to edges that keep dp_aligned's mirror-on-rank-axis
# shape (e.g. a same-shape identity/Mirror edge with no per-rank selection); it no longer applies
# to the `("trainer", "train", "batch")` edge, which as of Task 8 carries a `select`
# (`workers/dispatch.py:select_chunk`) and materializes only THIS rank's kept share at
# `workers/ray.py:_resolve_dispatched`, not the full mirrored producer payload — so there is
# nothing left to broadcast on that specific edge. Add a `bcast_aligned` variant that still maps
# every consumer (r,k) ← producer (r,0) (so all ranks enter the consumer collective) but TAGS the
# edge so the bulk is delivered by an intra-replica NCCL broadcast (rank0 pulls once, broadcasts
# to its replica's ranks 1..K-1 over the per-replica group formed in spawn_grid Pass 2) instead of
# N pulls. The metadata ref still goes to all ranks (tiny); only the bulk tensors ride the
# broadcast. Inject via GridBackend(align=bcast_aligned); the broadcast itself lands in
# materialize_payload (see the TODO in comm/codec.py + workers/ray.py:_Host.invoke).
AlignPolicy = Callable[[tuple, tuple, Any], dict]

# ray (and the @ray.remote-decorated .replica module) are imported lazily inside the
# functions/methods below so `dp_aligned` above stays importable without Ray installed.
# roles.core.dispatch (registry) and workers.dispatch (routing fns) are both Ray-free/
# torch-free at import (Global Constraints), so module-level import is safe here too.
from cosmos_framework.posttrain.roles.core import dispatch as registry  # noqa: E402
from cosmos_framework.posttrain.workers import dispatch as routing  # noqa: E402
from cosmos_framework.posttrain.workers.worker_group import (  # noqa: E402
    CollectiveHandle,
    Dispatched,
    Handle,
    LocalPeerRef,
    Mirror,
    RankAligned,
    Scatter,
    Sharded,
    WorkerGroup,
    take_control_kwargs,
)


def _kill_grid(ray_module: Any, grid: dict[tuple[int, int], tuple[str, Any]]) -> None:
    """Best-effort kill every actor in a partially spawned grid."""
    for _store_id, host in grid.values():
        try:
            ray_module.kill(host, no_restart=True)
        except Exception:  # noqa: BLE001 -- preserve the failure that triggered rollback
            pass


#: Prefix of the per-GPU shared host-id resource. A named constant because the LAUNCHER must
#: declare the same tags this charges against, and a bind whose demand exceeds the declared
#: supply does not fail -- Ray creates the actor and never schedules it. Two copies of the
#: string, one here and one in the check that verifies the supply, is exactly how that silence
#: gets re-introduced.
HOST_TAG_PREFIX = "node_"


def _interpreter_owned_roots() -> tuple[str, ...]:
    """Directories the running interpreter supplies itself: its stdlib, zipped stdlib and site dirs."""
    scheme = sysconfig.get_paths()
    roots = {scheme[key] for key in ("stdlib", "platstdlib", "purelib", "platlib") if scheme.get(key)}
    roots.update(getattr(site, "getsitepackages", list)())
    roots.add(site.getusersitepackages())
    roots.add(os.path.join(sys.base_prefix, "lib", f"python{sys.version_info.major}{sys.version_info.minor}.zip"))
    return tuple(sorted({os.path.abspath(root) for root in roots}))


def _is_interpreter_owned(entry: str, roots: tuple[str, ...]) -> bool:
    # Editable installs put finder placeholders, not directories, on sys.path from site-packages .pth files.
    if entry.startswith("__editable__.") and entry.endswith(".__path_hook__"):
        return True
    path = os.path.abspath(entry or os.curdir)
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _actor_pythonpath(env_vars: Mapping[str, str] | None) -> tuple[str, ...]:
    """Freeze the driver's import path for actor startup and construction.

    Explicit PYTHONPATH entries (requested, then inherited) always pass through. The driver's other
    ``sys.path`` entries, such as repository roots and external packages, follow them, except what the
    interpreter supplies itself (stdlib, site-packages and their editable-install placeholders): every
    actor's interpreter supplies its own, and prepending the driver's would shadow a separate role
    venv's packages.
    """
    requested = (env_vars or {}).get("PYTHONPATH", "")
    inherited = os.environ.get("PYTHONPATH", "")
    explicit = (
        *(requested.split(os.pathsep) if requested else ()),
        *(inherited.split(os.pathsep) if inherited else ()),
    )
    owned = _interpreter_owned_roots()
    live = (entry for entry in sys.path if not _is_interpreter_owned(entry, owned))
    return tuple(dict.fromkeys(os.path.abspath(entry or os.curdir) for entry in (*explicit, *live)))


def spawn_grid(
    setup,
    replicas: int,
    ranks: int,
    group_id: str,
    gpu_request: float,
    threshold: int,
    materialize_names: tuple,
    pgs: list | None = None,
    k_stride: int | None = None,
    bundles_per_pg: int | None = None,
    init_dist: bool | None = None,
    host_slot: str | None = None,
    host_prefix: str = HOST_TAG_PREFIX,
    cells_per_host: int = 1,
    host_single_thread: bool = False,
    env_vars: Mapping[str, str] | None = None,
    dist_backend: str | None = None,
    dist_scope: str = "replica",
    run_setup: bool = True,
    model_parallelism: tuple[int, int] | None = None,
    enable_tensor_transport: bool = False,
    data_pool_config: dict[str, int] | None = None,
    expose_peer_gpus: bool = False,
) -> dict:
    """Spawn R×K _Host actors addressed by (replica, rank), each running the same
    no-arg setup(). Each actor is assigned a globally unique store_id
    ``{group_id}_r{r}_k{k}``. Returns {(r, k): (store_id, actor)}.

    With a list of placement groups ``pgs`` tiling the union mesh row-major (each PG
    holds ``bundles_per_pg`` bundles), cell (r, k) pins to bundle ``global % bundles_per_pg``
    in PG ``global // bundles_per_pg`` where ``global = r * k_stride + k``. Same-group
    cells get distinct bundles (distinct GPUs); cross-group cells at the same (r, k) share
    a bundle (shared GPU). Multi-node falls out automatically because each PG is
    STRICT_PACK and capacity-saturating. Falls back to unbundled remote() when ``pgs``
    is empty/None.

    **Per-execution-replica rendezvous:** for a GPU mesh with ``ranks > 1``, each execution replica
    forms its own torch.distributed group of size ``ranks`` before ``run_setup``. A rollout/teacher
    engine reuses that group as its vLLM TP world. Policy placement has exactly one execution
    replica; its trainer-owned ``model_parallelism`` factors that one K-rank world into
    ``dp_replicate × dp_shard``.
    ``init_distributed`` sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT/LOCAL_RANK and calls
    ``init_process_group`` on each actor. ``init_dist`` defaults to a grid-wide rendezvous for
    any multi-cell ``dist_scope="grid"`` mesh (NCCL on GPU, gloo on CPU); a replica-scoped mesh
    still requires GPUs and ``ranks>1``. ``grid`` remains a lower-level single-world facility, but
    the live binder does not use it to synthesize policy HSDP from placement R.

    Args:
        setup: no-arg callable run on each actor (via ``run_setup``) to build its role body AFTER rendezvous.
        replicas: ``R`` — execution-replica count of the mesh.
        ranks: ``K`` — worker/process ranks inside each execution replica.
        group_id: prefix for each cell's unique store_id ``{group_id}_r{r}_k{k}``.
        gpu_request: binder-derived GPUs per actor (Ray ``num_gpus`` option); ``0`` ⇒ CPU mesh.
        threshold: bulk-store spill threshold forwarded to ``run_setup``.
        materialize_names: method names whose outputs materialize bulk, forwarded (as a tuple) to ``run_setup``.
        pgs: placement groups tiling the union mesh row-major; empty/None ⇒ unbundled ``remote()``.
        k_stride: rank stride for the flat global index (union mesh width); defaults to ``ranks``.
        bundles_per_pg: bundles per placement group — the PG-selection divisor for the flat index.
        init_dist: force rendezvous; None enables every multi-cell grid-scoped mesh and GPU
            replica-scoped meshes with ``ranks>1``.
        host_slot: custom-resource raylet slot for host-affinity placement (multi-venv); None ⇒ PG/plain placement.
        host_prefix: prefix for the shared pod-affinity resource ``{host_prefix}{idx}``.
        cells_per_host: GPU slots advertised per affinity host/pod; used only to map flat actor
            indexes to the pod hint. Physical GPU identity is proved separately from CUDA UUIDs.
        host_single_thread: use the plain single-threaded host actor so native setup and calls run on
            Ray's process main thread. Independent of placement; default False keeps the training host's
            bulk concurrency group.
        dist_backend: torch.distributed backend passed to ``init_distributed``.
        dist_scope: ``"replica"`` (default) gives each execution replica its own K-rank world.
            ``"grid"`` is a lower-level/testing option forming one ``R*K`` world; it does not imply
            an HSDP factorization, and the live policy binder keeps R=1.
        run_setup: build role bodies before returning (default). The placement binder sets this
            false so it can run a real cross-actor NIXL capability probe after rendezvous but before
            loading large models, then calls ``setup_grid`` explicitly.
        model_parallelism: optional trainer-owned ``(dp_replicate, dp_shard)`` factorization of the
            sole policy execution replica's K-rank world. It never changes actor placement.
        enable_tensor_transport: enable Ray Direct Transport on actors in a NIXL WEIGHT endpoint
            pool and warm each actor's NIXL agent before distributed rendezvous. Kept false for
            ordinary actors so optional NIXL dependencies remain lazy.
        data_pool_config: optional RDT/NIXL page-pool settings forwarded to every host.
        expose_peer_gpus: show every actor its node's other GPUs, its own first, before anything
            initializes CUDA in it (`_HostImpl.expose_peer_gpus`); for engines whose custom
            all-reduce maps tensor-parallel peers' memory.

    Returns:
        ``{(r, k): (store_id, actor)}`` for every spawned cell.
    """
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from cosmos_framework.posttrain.workers.ray import _Host, _HostMainThread

    if dist_scope not in ("replica", "grid"):
        raise ValueError(f"dist_scope={dist_scope!r}: expected 'replica' or 'grid'")

    pythonpath = _actor_pythonpath(env_vars)
    actor_env = dict(env_vars or {})
    actor_env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    runtime_env = {"env_vars": actor_env}
    grid: dict = {}
    # Pass 1: spawn every actor (no setup yet — the model must build AFTER rendezvous).
    for r in range(replicas):
        for k in range(ranks):
            sid = f"{group_id}_r{r}_k{k}"
            opts: dict = {}
            if gpu_request:
                opts["num_gpus"] = gpu_request
            if enable_tensor_transport:
                opts["enable_tensor_transport"] = True
            # Runtime env lets Ray import the actor class even if a raylet missed the launch shell;
            # the constructor also receives this tuple directly because env vars may be applied
            # after the worker interpreter initialized sys.path.
            opts["runtime_env"] = runtime_env
            if host_slot is not None:
                # Host-affinity placement (multi-venv / multi-raylet): pin this actor to its
                # role's raylet slot on a specific pod via custom resources rather than a PG
                # bundle, which cannot span logical Ray nodes/venvs. The shared ``node_*`` tag is
                # only a pod-pool hint; this raw cell order is provisional until bind re-keys shared
                # DeviceGroup grids by CUDA UUID. ``cells_per_host`` maps the flat actor index to the
                # pod-level resource.
                global_idx = bundle_index(r, k, k_stride or ranks)
                opts["resources"] = {host_slot: 1, f"{host_prefix}{global_idx // cells_per_host}": 1}
            elif pgs and k_stride is not None and bundles_per_pg:
                global_idx = bundle_index(r, k, k_stride)
                pg = pgs[global_idx // bundles_per_pg]
                local_idx = global_idx % bundles_per_pg
                opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=local_idx
                )
            # Actor threading is an execution requirement, not a placement property.
            # Native-heavy eval roles opt into the process-main-thread actor explicitly;
            # training keeps the concurrency-group host used by the bulk transfer plane.
            host_cls = _HostMainThread if host_single_thread else _Host
            try:
                host = host_cls.options(**opts).remote(pythonpath)
            except BaseException:  # noqa: BLE001 -- a partially spawned grid owns live actor resources
                _kill_grid(ray, grid)
                raise
            grid[(r, k)] = (sid, host)

    # First call on every actor: GPU visibility is fixed once CUDA starts, and both the NIXL warm-up
    # and the rendezvous below start it.
    if expose_peer_gpus:
        try:
            ray.get([host.expose_peer_gpus.remote() for _, host in grid.values()])
        except BaseException:  # noqa: BLE001 -- a partially prepared grid owns live actor resources
            _kill_grid(ray, grid)
            raise

    # Before the rendezvous: NCCL drags the container's HPC-X UCX in with it, and the UCX that
    # initializes first owns the process-global memory hooks. Losing that race leaves the nixl
    # wheel's UCX without CUDA support, and the first weight sync hangs (_HostImpl.warm_nixl).
    if enable_tensor_transport:
        ray.get([host.warm_nixl.remote() for _, host in grid.values()])

    # Pass 2: per-execution-replica torch.distributed rendezvous BEFORE setup. Each replica r forms
    # its own K-rank group; vLLM reuses it as one engine's TP group, while a sole policy replica may
    # factor it through explicit model_parallelism. All init_distributed calls are dispatched together and awaited in
    # one barrier below so init_process_group (a collective) can rendezvous — serializing per rank
    # would deadlock (rank 0 blocks inside init waiting for rank 1, whose call is never dispatched).
    # A lower-level `grid` scope must rendezvous even at K=1 because its single world may still have
    # R>1 cells. This does not assign model-mesh meaning to R.
    do_init = (
        init_dist
        if init_dist is not None
        else (
            (dist_scope == "grid" and replicas * ranks > 1)
            or (dist_scope == "replica" and bool(gpu_request) and ranks > 1)
        )
    )
    if do_init:
        try:
            init_futs: list = []
            if dist_scope == "grid":
                # ONE world over every cell; flat row-major rank r*ranks+k, master = cell (0,0).
                addr, port = ray.get(grid[(0, 0)][1].rendezvous_endpoint.remote())
                world = replicas * ranks
                for r in range(replicas):
                    for k in range(ranks):
                        init_futs.append(
                            grid[(r, k)][1].init_distributed.remote(addr, port, r * ranks + k, world, dist_backend)
                        )
            else:
                for r in range(replicas):
                    addr, port = ray.get(grid[(r, 0)][1].rendezvous_endpoint.remote())  # this replica's master
                    for k in range(ranks):
                        init_futs.append(grid[(r, k)][1].init_distributed.remote(addr, port, k, ranks, dist_backend))
            ray.get(init_futs)  # barrier: all init_process_group in flight ⇒ the group(s) rendezvous
        except BaseException:  # noqa: BLE001 -- a partial rendezvous must not strand actor resources
            _kill_grid(ray, grid)
            raise

    if run_setup:
        try:
            setup_grid(
                grid,
                setup,
                replicas=replicas,
                ranks=ranks,
                threshold=threshold,
                materialize_names=materialize_names,
                dist_scope=dist_scope,
                model_parallelism=model_parallelism,
                data_pool_config=data_pool_config,
            )
        except BaseException:  # noqa: BLE001 -- direct callers also get resource rollback
            _kill_grid(ray, grid)
            raise
    return grid


#: Seconds between driver progress lines while actor setup (engine builds, model loads) runs.
SETUP_PROGRESS_INTERVAL_S = 60.0
#: Waiting cells named in one progress line; the rest are only counted.
SETUP_PROGRESS_SHOWN = 8


def _cell_hosts(ray_module: Any, grid: dict[tuple[int, int], tuple[str, Any]], cells: tuple) -> dict:
    """Map every cell to its node's hostname, read before setup occupies each single-threaded actor."""
    addresses = ray_module.get([grid[cell][1].host.remote() for cell in cells])
    names = {
        node["NodeManagerAddress"]: node.get("NodeManagerHostname") or node["NodeManagerAddress"]
        for node in ray_module.nodes()
    }
    return {cell: names.get(address, address) for cell, address in zip(cells, addresses, strict=True)}


def _await_setup(ray_module: Any, refs: list, cells: tuple, label: str, hosts: dict) -> list:
    """Collect every cell's setup result in cell order, logging which cells are still building.

    The driver otherwise says nothing until the slowest of possibly a thousand cells finishes. A
    failed cell raises as soon as it completes instead of after every other cell.
    """
    import time

    from cosmos_framework.posttrain.utils.log import logger

    cell_of = dict(zip(refs, cells, strict=True))
    results: dict = {}
    pending = list(refs)
    started = time.monotonic()
    while pending:
        done, pending = ray_module.wait(pending, num_returns=len(pending), timeout=SETUP_PROGRESS_INTERVAL_S)
        for ref in done:
            results[cell_of[ref]] = ray_module.get(ref)
        if pending:
            waiting = sorted(cell_of[ref] for ref in pending)
            shown = " ".join(f"r{r}k{k}@{hosts.get((r, k), '?')}" for r, k in waiting[:SETUP_PROGRESS_SHOWN])
            hidden = len(waiting) - SETUP_PROGRESS_SHOWN
            logger.info(
                f"setup progress group={label} done={len(cells) - len(waiting)}/{len(cells)} "
                f"elapsed_s={time.monotonic() - started:.0f} waiting={shown}" + (f" +{hidden}" if hidden > 0 else "")
            )
    return [results[cell] for cell in cells]


def setup_grid(
    grid: dict[tuple[int, int], tuple[str, Any]],
    setup: Callable[[], dict[str, Any]],
    *,
    replicas: int,
    ranks: int,
    threshold: int,
    materialize_names: tuple[str, ...],
    dist_scope: str,
    model_parallelism: tuple[int, int] | None = None,
    expected_names: tuple[str, ...] | None = None,
    data_pool_config: dict[str, int] | None = None,
) -> None:
    """Build every body after actors/rendezvous and any pre-model transport probes exist."""
    import ray

    # Build concurrently across ALL cells: a body whose setup runs a collective (for example vLLM
    # external_launcher) blocks until every rank in its replica enters. Serial setup deadlocks.
    cells = tuple((replica, rank) for replica in range(replicas) for rank in range(ranks))
    hosts = _cell_hosts(ray, grid, cells)
    refs = [
        grid[(replica, rank)][1].run_setup.remote(
            setup,
            True,
            threshold,
            tuple(materialize_names),
            grid[(replica, rank)][0],
            replicas,
            ranks,
            dist_scope,
            replica,
            rank,
            model_parallelism,
            data_pool_config,
        )
        for replica, rank in cells
    ]
    label = grid[cells[0]][0].rsplit("_r", 1)[0] if cells else "grid"
    built_names = _await_setup(ray, refs, cells, label, hosts)
    if expected_names is not None:
        expected = set(expected_names)
        for cell, names in zip(cells, built_names, strict=True):
            if set(names) != expected:
                raise RuntimeError(
                    f"ActorGroup setup at cell {cell} returned bodies {sorted(names)}, "
                    f"expected exactly {sorted(expected)}"
                )


class GridBackend:
    """WorkerGroup backend: one replicas×ranks _Host actor mesh per group (DESIGN §1.1/§4.2).

    Each WG dispatches over its OWN placement-resolved ``(replicas, ranks)`` sub-shape, not the physical
    mesh: ``wg_shapes[wg_name]`` (registered at bind). So in one shared mesh a rollout WG at
    ranks=TP runs on all TP cells while a data/reward WG at ranks=1 runs ONLY on its (r, 0)
    cell — no redundant execution, no reliance on determinism. Unregistered WGs fall back to
    the mesh shape (direct-construction tests).

    Args (positional and keyword) are aligned per consumer actor via the AlignPolicy (§2.4)
    over the two WGs' OWN shapes (producer shape → consumer shape): a ``CollectiveHandle``
    forwards the aligned producer ObjectRef (controller routes metadata only, §3.4/#7); a peer
    ``WorkerGroup`` becomes a Ray-free ``PeerClient`` to the aligned producer cell (inline
    outbound calls, §1.2 (b)); a ``Scatter`` gives each cell its ``seq[r]`` slice. ``dp_aligned``
    clamp-to-0 broadcasts low→high (every consumer rank ← producer (r, 0)) and gathers high→low
    (consumer (r, 0) ← producer (r, 0)) by construction.

    ``train`` (dp_method) takes each execution replica's rank-0 report; the configured
    ``dp_reduce_op`` either merges typed independent-replica contributions or identity-checks an
    already model-world-global policy report → a scalar Handle.
    All other methods return a CollectiveHandle carrying the WG's own shape.

    submit interface is identical to RayBackend/ReplicaBackend so the WG/Handle/loop
    contract is unchanged.
    """

    def __init__(
        self,
        grid: dict,
        shape: tuple,
        align: AlignPolicy = dp_aligned,
        dp_methods=("train",),
        dp_reduce_op: str = "mean",
        wg_shapes: dict | None = None,
        weight_routes: dict | None = None,
        weight_broadcast_enabled: bool = False,
        peers: dict | None = None,
    ) -> None:
        """Bind an actor mesh (from ``spawn_grid``) as a WorkerGroup backend.

        Args:
            grid: ``{(r, k): (store_id, actor)}`` from ``spawn_grid``; only the actors are retained.
            shape: ``(R, K)`` physical mesh shape (union of the group's WG sub-shapes).
            align: ``AlignPolicy`` mapping producer↔consumer cells (§2.4); default ``dp_aligned``.
            dp_methods: method names reduced over replica leads instead of returning a ``CollectiveHandle``.
            dp_reduce_op: ``"mean"`` for typed execution-replica contributions, ``"identical"``
                when the body's Torch model world already reduced the report, or ``"replica0"``
                for a lower-level diagnostic backend.
            wg_shapes: ``{wg_name: (R, K)}`` per-WG resolved sub-shape; unregistered ⇒ fall back to ``shape``.
            weight_routes: weight-edge overrides consulted only for the gated weight-sync peer edge
                in ``_align_one``. On a POLICY backend (producer-driven ``push_weights``, design
                2026-07-27 §2), this is one total flat-actor map from every policy cell to zero or
                more rollout cells. None ⇒ symmetric ``dp_aligned`` peer edge.
            weight_broadcast_enabled: whether the rollout WG uses the optional post-P2R replica
                broadcast. This is driver-visible mode state; communicators remain actor-local.
            peers: bind-time ``{store_id: actor}`` connection registry. Direct construction defaults
                to this grid's actors.
        """
        from cosmos_framework.posttrain.workers.replica import Communicator

        self.actors = {rk: host for rk, (_sid, host) in grid.items()}  # (r, k) -> actor
        self._peers = dict(peers) if peers is not None else {sid: host for sid, host in grid.values()}
        self.shape = shape  # (R, K) physical mesh (union of the group's WG sub-shapes)
        self.align = align
        self.dp_methods = set(dp_methods)
        if dp_reduce_op not in ("mean", "identical", "replica0"):
            raise ValueError(f"unsupported grid DP reduction {dp_reduce_op!r}")
        self.dp_reduce_op = dp_reduce_op
        # Per-WG resolved sub-shape {wg_name: (R, K)}; a WG dispatches only over its own cells.
        # Unregistered ⇒ fall back to the mesh shape (backward compat for direct construction).
        self.wg_shapes = dict(wg_shapes or {})
        # Weight-edge routing overrides, shipped by colocate.py at bind. Asymmetric shapes carry one
        # total flat-actor target map, including explicit idle routes. Consulted ONLY for the gated
        # weight edge in ``_align_one``; None (every other group / symmetric bind) ⇒ the peer edge
        # takes the byte-identical dp_aligned per-lane PeerClient.
        self.weight_routes = weight_routes
        if not isinstance(weight_broadcast_enabled, bool):
            raise TypeError(f"weight_broadcast_enabled must be bool, got {weight_broadcast_enabled!r}")
        self.weight_broadcast_enabled = weight_broadcast_enabled
        R, _K = shape
        # Per-replica rank-0 actor list = inter-replica Communicator endpoints (§1.1).
        # TODO(v1+, §1.1): real peer-to-peer NCCL/gloo collective inside the step;
        # currently driver-coordinated gather→reduce (same as ReplicaBackend.Communicator).
        self.comm = Communicator([self.actors[(r, 0)] for r in range(R)])

    def _dispatch_from_spec(
        self,
        spec: Sharded | Mirror,
        handle: Any,
        r: int,
        consumer_shape: tuple,
        *,
        kind: str | None,
        method: str | None,
        param: str | None,
        select_kw: dict | None = None,
    ) -> Dispatched:
        """Build a ``Dispatched`` from a ``Sharded``/``Mirror`` spec applied to producer
        ``handle`` for consumer replica ``r`` (DESIGN §2.2/§2.3, Task A4). Units are the
        producer's OWN resolved sub-shape — ``handle.shape`` is already that sub-shape (set
        in ``submit`` from ``wg_shapes``, mirroring the ``WorkerGroup`` branch of
        ``_align_one`` below), flattened row-major (``flat_units``).

        Args:
            spec: the ``Sharded``/``Mirror`` routing spec (call-site marker or registry entry).
            handle: producer ``CollectiveHandle`` the spec applies to.
            r: consumer replica index this ``Dispatched`` is built for.
            consumer_shape: consumer WG's OWN ``(R, K)`` sub-shape.
            kind: producer role / registry ``kind`` (used in error messages).
            method: method name (used in error messages).
            param: parameter name (used in error messages).
            select_kw: resolved ingress-selection kwargs (spec 2026-08-12 §3.4; from
                ``take_control_kwargs`` in ``_actor_args``), carried onto the returned
                ``Dispatched`` alongside ``spec.select`` so the actor can apply it once it
                materializes (Task 8). ``None`` ⇒ no selection — carried through as
                ``Dispatched(select=None, select_kw=None)``, inert.

        Returns:
            A ``Dispatched`` wire descriptor routing producer units onto consumer replica ``r``.
        """
        if not isinstance(handle, CollectiveHandle):
            raise ValueError(
                f"Sharded/Mirror arg (kind={kind!r}, method={method!r}, param={param!r}) must wrap a "
                f"CollectiveHandle (a peer WG's result), got {type(handle).__name__}"
            )
        pshape = handle.shape
        units = routing.flat_units(pshape)
        P = len(units)
        Rc, Kc = consumer_shape
        _Rp, Kp = pshape
        if isinstance(spec, Sharded):
            # A Sharded spec must carry AT MOST ONE of reduce=/split= (they are the P>R_c fan-in and
            # the R_c>P fan-out realizations — never both). Both-set used to silently prefer reduce
            # (route_sharded picks by count), masking a mis-declared registry entry. Fail loud.
            if spec.reduce and spec.split:
                raise ValueError(
                    f"Sharded spec (kind={kind!r}, method={method!r}, param={param!r}) sets BOTH "
                    f"reduce= and split=; a Sharded edge carries at most one (reduce for P>R_c fan-in, "
                    f"split for R_c>P fan-out) — declare exactly one"
                )
            # Fail loud on a MULTI-RANK SPMD producer behind a Sharded edge BEFORE routing (spec
            # 2026-07-22 §2.2). ``flat_units`` counts every producer cell as a unit, but a K_p>1 SPMD
            # producer's rank lanes are MIRRORS of one another — so it contributes K_p duplicate units.
            # The old-canonical uniform shape then either silently K-folds the training batch through
            # the registered reduce (loss corruption, NO error) or raises a misdirected "reduce=
            # required". Neither is recoverable at runtime, so reject the shape at dispatch.
            if Kp > 1 and not spec.rank_partitioned:
                raise ValueError(
                    f"Sharded edge (kind={kind!r}, method={method!r}, param={param!r}) has a multi-rank "
                    f"SPMD producer sub-shape {pshape} (K_p={Kp} > 1): its rank lanes are mirrors, so "
                    f"counting them as {P} units double-counts each replica's output (silently K-folds "
                    f"the registered reduce, corrupting the training batch, or misfires a 'reduce= "
                    f"required'). Declare the SPMD helper at ranks=1 so its mirrored outputs are not "
                    f"double-counted as units — spec 2026-07-22 §2.2"
                )
        if isinstance(spec, Mirror):
            if P != 1:
                raise ValueError(
                    f"Mirror requires a single-unit producer (P == 1), got P={P} for producer shape "
                    f"{pshape} (kind={kind!r}, method={method!r}, param={param!r})"
                )
            ref = handle.ref_at(*units[0])
            return Dispatched(refs=[ref], op="one", fn=None, index=0, n=1, readers=Rc * Kc)
        op, n = routing.route_sharded(P, Rc)
        refs = [handle.ref_at(*units[u]) for u in routing.units_for_replica(op, n, P, r)]
        if op == "reduce":
            if not spec.reduce:
                raise ValueError(
                    f"Sharded reduce= required: P={P} producer units reduce into R_c={Rc} consumer "
                    f"replicas (kind={kind!r}, method={method!r}, param={param!r})"
                )
            fn = spec.reduce
        elif op == "split":
            if not spec.split:
                raise ValueError(
                    f"Sharded split= required: R_c={Rc} consumer replicas split from P={P} producer "
                    f"units (kind={kind!r}, method={method!r}, param={param!r})"
                )
            fn = spec.split
        else:  # op == "one"
            fn = None
        index = routing.split_index(op, n, r)
        # select/select_kw (Task 7 plumbs them onto Dispatched; Task 8 consumes them actor-side in
        # `workers/ray.py:_resolve_dispatched`). None ⇒ no selection, which is both the DPO case
        # (select_kw absent) and every non-selecting Sharded edge.
        select = spec.select if select_kw is not None else None
        # readers = the non-DP group size per kept ref (§9.5): ranks sharing a dp_rank but differing
        # in TP/PP pull the SAME sample, so Kc/dp_world of them count. `select` is opt-in (only
        # GRPO's train passes the kwargs; DPO shares the entry and doesn't), and without it every
        # rank materializes the full slice — so readers stays Kc there or the BulkStore entry pops
        # early. dp_world=None ⇒ Kc, exact today (no TP/PP inside the rank world); readers_for is the seam
        # for when one gains it (a mismatch fails loud there, not silently).
        readers = routing.readers_for(Kc, n, op, select=bool(select), dp_world=None)
        # π_old AHEAD PASS: at n_mini > 1 the driver walks the whole (mini, micro) product TWICE —
        # once recording π_old, once training (entrypoints/grpo.py) — so every kept ref is pulled
        # twice per consumer cell instead of once. `readers` is a REFCOUNT, not a call count: leave
        # it at one pass and the BulkStore entry pops at the end of the record pass, so training
        # dies with `KeyError` on data it never got to see. Both edges ("train" and
        # "compute_old_logprobs") reach this line and must agree on the doubled value, which they do
        # because both carry the same `select_kw`.
        #
        # Read the DRIVER'S OWN decision (`GRPOConfig.needs_cached_old_logprobs`, forwarded as the
        # `ahead_pass` control kwarg) rather than re-deriving it from n_mini here. An earlier version
        # did re-derive it and was WRONG: the predicate is
        # `n_mini > 1 AND NOT use_rollout_logprobs_for_loss`, and under rollout logprobs the
        # objective never consults the cache, so the driver skips the pass — while this line kept
        # doubling and LEAKED every BulkStore entry (refcount never reaching zero). One decision,
        # one owner, forwarded; never two copies of a predicate that must agree.
        if select and bool((select_kw or {}).get("ahead_pass", False)):
            readers *= 2
        return Dispatched(
            refs=refs,
            op=op,
            fn=fn,
            index=index,
            n=n,
            readers=readers,
            on_metadata=spec.on_metadata,
            select=select,
            select_kw=select_kw if select else None,
        )

    def peer_cells(self, peer: WorkerGroup, consumer_shape: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int]]:
        """Resolve ordinary synchronous peers with the same alignment used for live calls."""
        producer = peer._backend
        if producer is None:
            raise RuntimeError(f"peer WG {peer.name!r} passed as arg is not bound to a backend")
        producer_shape = producer.wg_shapes.get(peer.name, producer.shape)
        return {cell: targets[0] for cell, targets in self.align(producer_shape, consumer_shape, None).items()}

    def _align_one(
        self,
        arg: Any,
        consumer_r: int,
        consumer_k: int,
        consumer_shape: tuple,
        *,
        kind: str | None = None,
        method: str | None = None,
        param: str | None = None,
        dispatch_profile: str | None = None,
        select_kw: dict | None = None,
    ) -> Any:
        """Align ONE arg onto consumer cell (r, k) over the consumer WG's OWN shape.

        Registry-driven per-edge routing (DESIGN §2.1/§2.2, Task A4), in precedence order:

        (a) call-site marker: ``RankAligned`` requires identical producer/consumer shapes and
            forwards the exact ``(r, k)`` peer; ``Sharded``/``Mirror`` unwrap ``.arg`` and build
            a ``Dispatched`` from the marker's OWN spec. Both forms override the registry.
        (b) bare ``CollectiveHandle`` + a ``lookup(kind, method, param)`` hit: build a
            ``Dispatched`` from the registry spec, same construction as (a).
        (c) bare ``CollectiveHandle``, no spec: producer/consumer replica counts equal ⇒
            ``dp_aligned``-style forward (``self.align``). A ref that ``self.align`` routes to
            MORE THAN ONE consumer cell (the rank-mirror case, e.g. producer ranks=1 feeding
            consumer ranks=TP) is wrapped ``Dispatched(op="one", readers=<consumer cell count
            for that ref>)`` so the refcounted ``BulkStore`` pops only after every reader has
            pulled; a single-reader ref forwards the raw ref unwrapped (byte-identical to the
            pre-dispatch path). Replica counts differ ⇒ ``ValueError`` naming the unregistered
            ``(kind, method, param)`` and both shapes (the old silent replica-axis clamp is
            gone — DESIGN §2.1 precedence tier 3).
        (d) ``WorkerGroup`` (a peer WG passed to be *called inline*, §1.2 body-mode (b)):
            inject a Ray-free ``PeerClient`` bound to the aligned producer cell's actor. One
            weight-sync sub-case routes BEFORE the general peer form when ``self.weight_routes``
            carries an entry: ``push_weights(rollout)`` on a ``rollout`` arg (producer-driven
            flat-index bijection, design 2026-07-27 §2). A bind without a route entry (symmetric /
            non-weight peer edge) takes the general peer form.
        (e) ``Scatter`` (a per-cell sequence): give this cell its slice ``seq[r]``.
        (f) anything else: broadcast, hardcoded identity routing (not align-delegated).

        ``RankAligned`` in (a)/(b) forwards one exact-rank ref directly; the other marker/registry
        forms build a ``Dispatched``. (d)/(e)/(f) are BYTE-IDENTICAL to before this task
        (WorkerGroup branch changes in Task C1).

        Args:
            arg: the single call argument to align (dispatch marker, bare
                ``CollectiveHandle``, peer ``WorkerGroup``, ``Scatter``, or a plain value).
            consumer_r: consumer cell replica index.
            consumer_k: consumer cell rank index.
            consumer_shape: consumer WG's OWN ``(R, K)`` sub-shape.
            kind: consumer role / registry ``kind`` for the ``(kind, method, param)`` lookup.
            method: method name for the lookup (and the ``push_weights`` weight-edge gate).
            param: parameter name for the lookup.
            dispatch_profile: optional algorithm-specific contract selected by the consumer WG.
            select_kw: resolved ingress-selection kwargs for THIS param (spec 2026-08-12 §3.4,
                from ``take_control_kwargs`` in ``_actor_args``), threaded onto the registry-hit
                (tier 2) ``Dispatched`` only. ``None`` for every other param and for a call-site
                marker (out of scope here — a marker declares its own spec).

        Returns:
            The per-cell aligned arg: a ``Dispatched``, a forwarded producer ObjectRef, a
            ``PeerClient``/``WeightTargetSet``, a ``Scatter`` slice, or the raw value (broadcast).
        """
        if isinstance(arg, RankAligned):
            handle = arg.arg
            if not isinstance(handle, CollectiveHandle):
                raise TypeError(
                    f"RankAligned must wrap a CollectiveHandle when used with GridBackend, got {type(handle).__name__}"
                )
            if handle.shape != consumer_shape:
                raise ValueError(
                    "RankAligned requires identical producer and consumer shapes, "
                    f"got producer={handle.shape}, consumer={consumer_shape}"
                )
            return handle.ref_at(consumer_r, consumer_k)
        if isinstance(arg, (Sharded, Mirror)):
            return self._dispatch_from_spec(
                arg, arg.arg, consumer_r, consumer_shape, kind=kind, method=method, param=param
            )
        if isinstance(arg, CollectiveHandle):
            spec = (
                registry.lookup(kind, method, param, dispatch_profile=dispatch_profile)
                if kind is not None and method is not None and param is not None
                else None
            )
            if isinstance(spec, RankAligned):
                if arg.shape != consumer_shape:
                    raise ValueError(
                        f"dispatch profile {dispatch_profile!r} requires identical producer and consumer shapes "
                        f"for (kind={kind!r}, method={method!r}, param={param!r}), got "
                        f"producer={arg.shape}, consumer={consumer_shape}"
                    )
                return arg.ref_at(consumer_r, consumer_k)
            if spec is not None:
                return self._dispatch_from_spec(
                    spec,
                    arg,
                    consumer_r,
                    consumer_shape,
                    kind=kind,
                    method=method,
                    param=param,
                    select_kw=select_kw,
                )
            full_map = self.align(arg.shape, consumer_shape, None)
            producer_r, producer_k = full_map[(consumer_r, consumer_k)][0]
            producer_replicas, _producer_ranks = arg.shape
            consumer_replicas, _consumer_ranks = consumer_shape
            if producer_replicas != consumer_replicas:
                raise ValueError(
                    f"unregistered edge (kind={kind!r}, method={method!r}, param={param!r}): producer "
                    f"shape {arg.shape} and consumer shape {consumer_shape} have different replica counts "
                    f"({producer_replicas} != {consumer_replicas}) — register a Sharded/Mirror policy in "
                    f"DISPATCH or wrap the arg with Sharded(...)/Mirror(...)"
                )
            ref = arg.ref_at(producer_r, producer_k)  # forward ref; actor resolves it
            readers = sum(1 for v in full_map.values() if v[0] == (producer_r, producer_k))
            if readers > 1:  # this ref feeds more than one consumer cell — needs refcounting
                return Dispatched(refs=[ref], op="one", fn=None, index=0, n=1, readers=readers)
            return ref
        if isinstance(arg, WorkerGroup):
            from cosmos_framework.posttrain.comm.weight_transfer import WeightTargetSet
            from cosmos_framework.posttrain.workers.ray import PeerClient, RoutedWeightPeerClient

            producer_backend = arg._backend  # producer's GridBackend
            if producer_backend is None:
                raise RuntimeError(f"peer WG {arg.name!r} passed as arg is not bound to a backend")

            # ---- Weight-sync edge: policy.push_weights(rollout) — PRODUCER-DRIVEN colocated push
            # (design 2026-07-27 §2). Called ON the policy WG, so this runs on the POLICY body over the
            # ``rollout`` arg. Gated on method + role so ONLY the genuine policy→rollout push edge enters
            # here; no other peer-WG edge is rerouted. ``self.weight_routes`` is the POLICY backend's
            # total flat-actor map: every policy cell owns zero or more rollout cells, independently
            # of either shape's execution-replica boundary. ``producer_backend`` is the peer arg's
            # backend = the ROLLOUT's GridBackend, so ``actors`` are the rollout cells. A SYMMETRIC
            # bind (no routes) falls through to the general peer form below (byte-identical
            # dp_aligned per-lane PeerClient).
            if method == "push_weights" and arg.role == "rollout":
                routes = self.weight_routes
                if routes is not None:
                    if (consumer_r, consumer_k) not in routes:
                        raise RuntimeError(
                            f"weight route map is missing policy cell {(consumer_r, consumer_k)}; "
                            f"bound cells are {sorted(routes)}"
                        )
                    route = routes[(consumer_r, consumer_k)]
                    targets = tuple(route)
                    if not all(
                        isinstance(target, tuple) and len(target) == 2 and all(isinstance(axis, int) for axis in target)
                        for target in targets
                    ):
                        raise TypeError(
                            f"invalid asymmetric weight route for policy cell {(consumer_r, consumer_k)}: {route!r}"
                        )
                    unknown = [target for target in targets if target not in producer_backend.actors]
                    if unknown:
                        raise ValueError(
                            f"weight route for policy cell {(consumer_r, consumer_k)} names unknown rollout "
                            f"cell(s) {unknown}; rollout cells are {sorted(producer_backend.actors)}"
                        )
                    if len(targets) > 1:
                        peer = RoutedWeightPeerClient(
                            tuple((target, producer_backend.actors[target]) for target in targets),
                            arg.name,
                        )
                        return WeightTargetSet((peer,))
                    return WeightTargetSet(
                        tuple(PeerClient(producer_backend.actors[target], arg.name) for target in targets)
                    )

            # ---- General peer form (body-mode (b)): any peer WG passed inline → PeerClient to the
            # dp_aligned producer cell. Serves non-weight inline-peer edges AND direct/symmetric weight
            # edges (which carry no route entry).
            producer_r, producer_k = self.peer_cells(arg, consumer_shape)[(consumer_r, consumer_k)]
            producer_actor = producer_backend.actors[(producer_r, producer_k)]
            consumer_actor = self.actors[(consumer_r, consumer_k)]
            if producer_actor._actor_id == consumer_actor._actor_id:
                return LocalPeerRef(arg.name)
            return PeerClient(producer_actor, arg.name)
        if isinstance(arg, Scatter):
            # NOTE: routing here is hardcoded strict-identity (seq[r]), NOT delegated to
            # self.align — a scatter needs error-on-mismatch, unlike dp_aligned's lenient
            # clamp. When dispatch grows (§8.2 #4 / gather), model this as a virtual
            # producer of shape (len,1) through a pluggable Dispatch strategy.
            consumer_replicas = consumer_shape[0]
            if len(arg.seq) != consumer_replicas:
                raise ValueError(f"scatter length {len(arg.seq)} != replicas {consumer_replicas}")
            return arg.seq[consumer_r]  # per-replica slice (broadcast over the rank axis)
        return arg  # broadcast: plain value, hardcoded identity routing (not align-delegated)

    def _actor_args(
        self,
        r: int,
        k: int,
        consumer_shape: tuple,
        args: tuple,
        kw: dict,
        *,
        kind: str | None = None,
        method: str | None = None,
        dispatch_profile: str | None = None,
    ) -> tuple:
        """Align a full positional+keyword arg set onto consumer cell ``(r, k)`` via ``_align_one``.

        Args:
            r: consumer cell replica index.
            k: consumer cell rank index.
            consumer_shape: consumer WG's OWN ``(R, K)`` sub-shape.
            args: positional call args.
            kw: keyword call args.
            kind: consumer role / registry ``kind``.
            method: method name — with ``kind``, resolves positional param names for the registry lookup.
            dispatch_profile: optional algorithm-specific contract selected by the consumer WG.

        Returns:
            ``(actor_args, actor_kwargs)`` aligned for the cell. When the registry hit for one
            param declares ``select=`` (spec 2026-08-12 §3.4) and the call passes its
            ``select_params`` control kwargs (``mini``/``micro``/``n_mini``/``n_micro``), those
            kwargs are STRIPPED here (via ``take_control_kwargs``) and never reach ``actor_kwargs``
            — the actor body is still invoked as ``train(batch)``. They are carried instead on that
            one param's ``Dispatched.select``/``select_kw`` for the actor to apply once it
            materializes (Task 8); absent control kwargs (the DPO case) leave everything unchanged.
        """
        # Param names: positional args bind via the registry's (kind, method) signature
        # table (Task A3 `bind_param_names`); kwargs already carry their own name.
        pos_names = registry.bind_param_names(kind, method, args, kw)[: len(args)]
        target_name, spec = None, None
        for name in list(pos_names) + list(kw.keys()):
            if name is None:
                continue
            candidate = registry.lookup(kind, method, name, dispatch_profile=dispatch_profile)
            if candidate is not None and getattr(candidate, "select", None):
                target_name, spec = name, candidate
                break
        body_kw, select_kw = take_control_kwargs(spec, kw)
        actor_args = tuple(
            self._align_one(
                a,
                r,
                k,
                consumer_shape,
                kind=kind,
                method=method,
                param=pos_names[i],
                dispatch_profile=dispatch_profile,
                select_kw=select_kw if pos_names[i] == target_name else None,
            )
            for i, a in enumerate(args)
        )
        actor_kwargs = {
            key: self._align_one(
                v,
                r,
                k,
                consumer_shape,
                kind=kind,
                method=method,
                param=key,
                dispatch_profile=dispatch_profile,
                select_kw=select_kw if key == target_name else None,
            )
            for key, v in body_kw.items()
        }
        return actor_args, actor_kwargs

    def hosts(self) -> dict:
        """{(r, k): node_ip} across the mesh — placement introspection read from the
        actors (DESIGN §4.1), so role bodies don't report their own node."""
        import ray

        vals = ray.get([h.host.remote() for h in self.actors.values()])
        return dict(zip(self.actors.keys(), vals))

    def connection_registry(self) -> dict:
        """Return the stable bind-time store-owner registry."""
        return dict(self._peers)

    def gpus(self) -> dict:
        """{(r, k): CUDA_VISIBLE_DEVICES} across the mesh."""
        import ray

        vals = ray.get([h.gpu.remote() for h in self.actors.values()])
        return dict(zip(self.actors.keys(), vals))

    def submit(
        self,
        wg_name: str,
        body: Any,
        method: str,
        args: tuple,
        kw: dict,
        *,
        kind: str | None = None,
        dispatch_profile: str | None = None,
    ) -> Handle:
        """Fan a WG method call out to every cell of the WG's OWN sub-shape, aligning args per cell.

        Args:
            wg_name: WG name; selects its OWN sub-shape from ``wg_shapes`` (fallback: mesh ``shape``).
            body: the role body — accepted for interface parity with RayBackend/ReplicaBackend; unused
                here since actors already hold their body.
            method: method name to invoke on each actor.
            args: positional args, aligned per cell via ``_actor_args``.
            kw: keyword args, aligned per cell.
            kind: WG role, threaded into the dispatch registry ``(kind, method, param)`` lookup.
            dispatch_profile: optional algorithm-specific contract overriding generic dispatch.

        Returns:
            A scalar ``Handle`` carrying the configured replica reduction for a ``dp_methods``
            method, else a ``CollectiveHandle`` carrying the WG's ``(R, K)`` shape.
        """
        import ray

        R, K = self.wg_shapes.get(wg_name, self.shape)  # this WG's OWN sub-shape (fallback: mesh)
        refs = {}
        for r in range(R):
            for k in range(K):
                actor_args, actor_kwargs = self._actor_args(
                    r,
                    k,
                    (R, K),
                    args,
                    kw,
                    kind=kind,
                    method=method,
                    dispatch_profile=dispatch_profile,
                )
                refs[(r, k)] = self.actors[(r, k)].invoke.remote(wg_name, method, actor_args, actor_kwargs)
        if method in self.dp_methods:
            # Policy R=1 uses ``identical`` because its K-rank model world already reduced the
            # report. Independent execution replicas use typed ``mean`` contributions.
            per_replica = [refs[(r, 0)] for r in range(R)]
            return Handle(self.comm.all_reduce(per_replica, op=self.dp_reduce_op))
        return CollectiveHandle(refs, (R, K), getter=ray.get)
