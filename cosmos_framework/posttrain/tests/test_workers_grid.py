# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""dp_aligned mapping + GridBackend mesh fan-out (DESIGN §1.1/§2.4)."""

from __future__ import annotations

import os
import sys
import sysconfig
import time
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.workers.grid import _actor_pythonpath, dp_aligned


@pytest.mark.L0
@pytest.mark.CPU
def test_dp_aligned_identity_same_shape():
    m = dp_aligned((2, 2), (2, 2))
    assert m == {(0, 0): [(0, 0)], (0, 1): [(0, 1)], (1, 0): [(1, 0)], (1, 1): [(1, 1)]}


@pytest.mark.L0
@pytest.mark.CPU
def test_dp_aligned_rank_mirror_when_producer_has_fewer_ranks():
    # consumer 2 ranks, producer 1 rank -> rank axis mirrors to producer rank-0
    m = dp_aligned((2, 1), (2, 2))
    assert m[(0, 1)] == [(0, 0)] and m[(1, 1)] == [(1, 0)]


@pytest.mark.L0
@pytest.mark.CPU
def test_actor_pythonpath_keeps_explicit_entries_and_drops_interpreter_owned_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cosmos_framework.posttrain.workers.grid as grid

    monkeypatch.setattr(grid, "_interpreter_owned_roots", lambda: ("/py/lib", "/venv/site-packages"))
    monkeypatch.setenv("PYTHONPATH", "/repo:/reward:/repo:/venv/site-packages")
    monkeypatch.setattr(
        sys,
        "path",
        [
            "/repo",
            "/py/lib",
            "/py/lib/lib-dynload",
            "/venv/site-packages",
            "__editable__.pkg-1.0.finder.__path_hook__",
            "/driver-only",
        ],
    )

    assert _actor_pythonpath(None) == ("/repo", "/reward", "/venv/site-packages", "/driver-only")
    assert _actor_pythonpath({"PYTHONPATH": "/extra:/repo"}) == (
        "/extra",
        "/repo",
        "/reward",
        "/venv/site-packages",
        "/driver-only",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_actor_pythonpath_omits_this_interpreters_stdlib_and_site_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = sysconfig.get_paths()
    stdlib, purelib = os.path.abspath(paths["stdlib"]), os.path.abspath(paths["purelib"])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [os.getcwd(), stdlib, os.path.join(stdlib, "lib-dynload"), purelib, "/external/rewards/site-packages"],
    )

    assert _actor_pythonpath(None) == (os.getcwd(), "/external/rewards/site-packages")


ray = pytest.importorskip("ray")

from cosmos_framework.posttrain.workers.grid import GridBackend, setup_grid, spawn_grid  # noqa: E402
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup  # noqa: E402


def _pid_setup():
    import os

    class W:
        def whoami(self):
            return os.getpid()

        def echo(self, x):
            return x  # return the received arg unchanged

        def echo_kw(self, x=None):
            return x  # same as echo but received via keyword (exercises kwarg alignment)

        def train(self, x):
            return float(x)  # each actor returns the pid it received (as float)

    return {"w": W()}


def _body_names(host: Any) -> list[str]:
    return sorted(host._bodies)


def _raising_setup() -> dict[str, object]:
    raise RuntimeError("injected setup failure")


def _slow_setup() -> dict[str, object]:
    time.sleep(1.5)
    return {"w": object()}


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_grid_reports_waiting_cells_and_hosts_while_setup_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long setup must be visible from the driver log: which cells are still building, and where."""
    import cosmos_framework.posttrain.workers.grid as grid_module
    from cosmos_framework.posttrain.utils.log import logger

    lines: list[str] = []
    monkeypatch.setattr(grid_module, "SETUP_PROGRESS_INTERVAL_S", 0.2)
    monkeypatch.setattr(logger, "info", lambda message, *args, **kwargs: lines.append(str(message)))
    ray.init(ignore_reinit_error=True, num_cpus=3, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _slow_setup,
            replicas=2,
            ranks=1,
            group_id="progress",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            run_setup=False,
        )
        setup_grid(
            grid,
            _slow_setup,
            replicas=2,
            ranks=1,
            threshold=0,
            materialize_names=(),
            dist_scope="replica",
        )
        assert [ray.get(host.run.remote(_body_names, host)) for _, host in grid.values()] == [["w"], ["w"]]
        node_names = {node["NodeManagerHostname"] for node in ray.nodes()}
    finally:
        ray.shutdown()
    progress = [line for line in lines if line.startswith("setup progress group=progress ")]
    assert progress, lines
    first = progress[0]
    assert "done=0/2" in first
    waiting = first.split("waiting=", 1)[1].split()
    assert [cell.split("@")[0] for cell in waiting] == ["r0k0", "r1k0"]
    assert {cell.split("@")[1] for cell in waiting} <= node_names


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_can_defer_model_setup_until_after_transport_probe() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=3, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _pid_setup,
            replicas=1,
            ranks=1,
            group_id="deferred",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            run_setup=False,
        )
        host = grid[(0, 0)][1]
        assert ray.get(host.run.remote(_body_names, host)) == []
        setup_grid(
            grid,
            _pid_setup,
            replicas=1,
            ranks=1,
            threshold=0,
            materialize_names=(),
            dist_scope="replica",
        )
        assert ray.get(host.run.remote(_body_names, host)) == ["w"]
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_setup_failure_releases_actor_resources() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=3, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        before = float(ray.available_resources().get("CPU", 0.0))
        with pytest.raises(Exception, match="injected setup failure"):
            spawn_grid(
                _raising_setup,
                replicas=1,
                ranks=1,
                group_id="rollback",
                gpu_request=0,
                threshold=0,
                materialize_names=(),
            )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and float(ray.available_resources().get("CPU", 0.0)) < before:
            time.sleep(0.1)
        assert float(ray.available_resources().get("CPU", 0.0)) >= before
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_rejects_invalid_scope_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.posttrain.workers.ray as ray_workers

    class _ForbiddenHost:
        @staticmethod
        def remote() -> object:
            pytest.fail("actor must not spawn")

    monkeypatch.setattr(ray_workers, "_Host", _ForbiddenHost)
    with pytest.raises(ValueError, match="dist_scope"):
        spawn_grid(
            _pid_setup,
            replicas=1,
            ranks=1,
            group_id="bad-scope",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            dist_scope="invalid",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_actor_creation_failure_kills_partial_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.posttrain.workers.ray as ray_workers

    first_actor = object()

    class _FailingHost:
        calls: int = 0

        @classmethod
        def options(cls, **_kwargs: Any) -> type:
            return cls

        @classmethod
        def remote(cls, _pythonpath: tuple[str, ...]) -> object:
            cls.calls += 1
            if cls.calls == 2:
                raise RuntimeError("injected actor creation failure")
            return first_actor

    killed: list[object] = []
    monkeypatch.setattr(ray_workers, "_Host", _FailingHost)
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append(actor))
    with pytest.raises(RuntimeError, match="injected actor creation failure"):
        spawn_grid(
            _pid_setup,
            replicas=1,
            ranks=3,
            group_id="partial",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            run_setup=False,
        )
    assert killed == [first_actor]


class _RecordedCall:
    """One fake actor method: ``.remote()`` logs its name and returns ``result`` (or raises it)."""

    def __init__(self, log: list[str], name: str, result: Any = None) -> None:
        self._log = log
        self._name = name
        self._result = result

    def remote(self, *_args: Any, **_kwargs: Any) -> Any:
        self._log.append(self._name)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _recording_host_class(log: list[str], expose_results: list[Any]) -> type:
    class _RecordingHost:
        @classmethod
        def options(cls, **_kwargs: Any) -> type:
            return cls

        @staticmethod
        def remote(_pythonpath: tuple[str, ...]) -> Any:
            return SimpleNamespace(
                expose_peer_gpus=_RecordedCall(log, "expose", expose_results.pop(0)),
                rendezvous_endpoint=_RecordedCall(log, "rendezvous", ("127.0.0.1", 29500)),
                init_distributed=_RecordedCall(log, "init"),
            )

    return _RecordingHost


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_exposes_peer_gpus_before_anything_starts_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.posttrain.workers.ray as ray_workers

    log: list[str] = []
    monkeypatch.setattr(ray_workers, "_Host", _recording_host_class(log, ["0,1", "1,0"]))
    monkeypatch.setattr(ray, "get", lambda value: value)
    spawn_grid(
        _pid_setup,
        replicas=1,
        ranks=2,
        group_id="peers",
        gpu_request=0,
        threshold=0,
        materialize_names=(),
        init_dist=True,
        run_setup=False,
        expose_peer_gpus=True,
    )
    assert log == ["expose", "expose", "rendezvous", "init", "init"]


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_peer_gpu_failure_kills_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.posttrain.workers.ray as ray_workers

    log: list[str] = []
    killed: list[Any] = []
    monkeypatch.setattr(
        ray_workers, "_Host", _recording_host_class(log, ["0,1", RuntimeError("injected visibility failure")])
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append(actor))
    with pytest.raises(RuntimeError, match="injected visibility failure"):
        spawn_grid(
            _pid_setup,
            replicas=1,
            ranks=2,
            group_id="peers",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            init_dist=True,
            run_setup=False,
            expose_peer_gpus=True,
        )
    assert log == ["expose", "expose"]
    assert len(killed) == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_hands_every_actor_the_filtered_path_in_its_runtime_env_and_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cosmos_framework.posttrain.workers.grid as grid_module
    import cosmos_framework.posttrain.workers.ray as ray_workers

    spawned: list[tuple[dict[str, Any], tuple[str, ...]]] = []

    class _CapturingHost:
        @classmethod
        def options(cls, **opts: Any) -> SimpleNamespace:
            def remote(pythonpath: tuple[str, ...]) -> object:
                spawned.append((opts, pythonpath))
                return object()

            return SimpleNamespace(remote=remote)

    monkeypatch.setattr(ray_workers, "_Host", _CapturingHost)
    monkeypatch.setattr(grid_module, "_interpreter_owned_roots", lambda: ("/py/lib", "/venv/site-packages"))
    monkeypatch.setenv("PYTHONPATH", "/repo:/rewards/site-packages")
    monkeypatch.setattr(sys, "path", ["/repo", "/py/lib", "/venv/site-packages", "/editable/src"])
    spawn_grid(
        _pid_setup,
        replicas=1,
        ranks=2,
        group_id="paths",
        gpu_request=0,
        threshold=0,
        materialize_names=(),
        run_setup=False,
        env_vars={"PYTHONPATH": "/extra", "LD_PRELOAD": "libfoo.so"},
    )

    expected = ("/extra", "/repo", "/rewards/site-packages", "/editable/src")
    assert [pythonpath for _opts, pythonpath in spawned] == [expected, expected]
    for opts, _pythonpath in spawned:
        assert opts["runtime_env"] == {"env_vars": {"PYTHONPATH": os.pathsep.join(expected), "LD_PRELOAD": "libfoo.so"}}


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_fanout_identity_alignment_and_dp_reduce():
    ray.init(ignore_reinit_error=True, num_cpus=6, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _pid_setup, replicas=2, ranks=2, group_id="g0", gpu_request=0, threshold=0, materialize_names=()
        )
        peers = {sid: host for sid, host in grid.values()}
        ray.get([host.set_peers.remote(peers) for _, host in grid.values()])
        gb = GridBackend(grid, shape=(2, 2))

        w = WorkerGroup("w", "test")
        w._bind_backend(gb)

        pids = w.whoami()  # CollectiveHandle: {(r,k): pid ref}
        echoed = w.echo(pids)  # each actor echoes its own aligned pid
        pid_grid = pids.all()
        echoed_grid = echoed.all()

        assert set(pid_grid) == {(0, 0), (0, 1), (1, 0), (1, 1)}  # fan-out to 4 actors
        assert len(set(pid_grid.values())) == 4  # 4 distinct processes
        assert echoed_grid == pid_grid  # identity alignment: each actor got its own pid

        loss = w.train(pids)  # dp reduce: rank-0 per replica, mean across replicas
        expected = (pid_grid[(0, 0)] + pid_grid[(1, 0)]) / 2
        assert loss.result() == pytest.approx(expected)
    finally:
        ray.shutdown()


def _dist_probe_setup():
    # Runs INSIDE each actor AFTER spawn_grid Pass 2 formed this replica's torch.distributed
    # group (per-replica: rank=k, world=ranks). It also does a REAL cross-rank collective
    # (barrier) in setup() — the exact shape that deadlocks if run_setup is dispatched serially
    # (rank 0 blocks in the barrier waiting for rank 1, whose run_setup is never dispatched).
    # Concurrent run_setup lets it complete (mirrors why vLLM external_launcher construction,
    # which runs collectives in setup, needs concurrency).
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    dist.barrier()

    class W:
        def dist_info(self):
            return (rank, world)

    return {"w": W()}


def _grid_dist_probe_setup() -> dict[str, object]:
    """Report the default process group created for a grid-scoped CPU colocation."""
    import torch.distributed as dist

    class W:
        @staticmethod
        def dist_info() -> tuple[int, int]:
            return dist.get_rank(), dist.get_world_size()

    return {"w": W()}


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_per_replica_rendezvous_and_concurrent_setup():
    # Two replicas × 2 ranks, rendezvous forced on (gloo) on CPU. Each replica must form its OWN
    # world of size 2 (rank=k) — NOT one flat world of 4 — proving per-replica rendezvous; the
    # barrier inside setup proves run_setup runs concurrently across a replica's ranks.
    pytest.importorskip("torch")
    ray.init(ignore_reinit_error=True, num_cpus=6, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _dist_probe_setup,
            replicas=2,
            ranks=2,
            group_id="g0",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            init_dist=True,  # force per-replica rendezvous on CPU; GPU meshes get it by default
            dist_backend="gloo",  # hermetic: don't auto-pick nccl on a GPU host
        )
        peers = {sid: host for sid, host in grid.values()}
        ray.get([h.set_peers.remote(peers) for _, h in grid.values()])
        gb = GridBackend(grid, shape=(2, 2))
        w = WorkerGroup("w", "test")
        w._bind_backend(gb)

        info = w.dist_info().all()  # {(r,k): (rank, world)}
        assert {rk: v[1] for rk, v in info.items()} == {
            (0, 0): 2,
            (0, 1): 2,
            (1, 0): 2,
            (1, 1): 2,
        }  # world==ranks, not 4
        assert {info[(0, 0)][0], info[(0, 1)][0]} == {0, 1}  # replica 0 ranks 0,1
        assert {info[(1, 0)][0], info[(1, 1)][0]} == {0, 1}  # replica 1 ranks 0,1 (independent world)
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_spawn_grid_grid_scope_defaults_to_one_gloo_world_on_cpu() -> None:
    """The retained lower-level grid scope forms one generic R*K Gloo world."""
    pytest.importorskip("torch")
    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _grid_dist_probe_setup,
            replicas=2,
            ranks=1,
            group_id="grid_cpu",
            gpu_request=0,
            threshold=0,
            materialize_names=(),
            dist_scope="grid",
        )
        peers = {sid: host for sid, host in grid.values()}
        ray.get([host.set_peers.remote(peers) for _, host in grid.values()])
        backend = GridBackend(grid, shape=(2, 1), dp_reduce_op="identical")
        worker = WorkerGroup("w", "test")
        worker._bind_backend(backend)

        assert worker.dist_info().all() == {(0, 0): (0, 2), (1, 0): (1, 2)}
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_kwarg_alignment():
    # A CollectiveHandle passed as a KEYWORD arg must be dp_aligned just like a positional one.
    ray.init(ignore_reinit_error=True, num_cpus=6, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _pid_setup, replicas=2, ranks=2, group_id="gkw", gpu_request=0, threshold=0, materialize_names=()
        )
        ray.get([host.set_peers.remote({sid: host for sid, host in grid.values()}) for _, host in grid.values()])
        gb = GridBackend(grid, shape=(2, 2))
        w = WorkerGroup("w", "test")
        w._bind_backend(gb)

        pids = w.whoami()
        echoed_kw = w.echo_kw(x=pids)  # kwarg path
        assert echoed_kw.all() == pids.all()  # each actor got its own aligned pid via the kwarg
    finally:
        ray.shutdown()


from cosmos_framework.posttrain.workers.worker_group import scatter  # noqa: E402


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_scatter_distributes_per_cell():
    """A ``scatter(seq)`` arg gives consumer cell (r, 0) its element ``seq[r]`` instead
    of broadcasting the whole sequence (DESIGN §2.4)."""
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles.core import factory as roles

    def setup():
        class W:
            def echo(self, item):
                return item

        return {"w": W()}

    try:
        w = roles.env("w", replicas=3, ranks=1)
        bind_device_groups([DeviceGroup((ActorGroup((w,), setup),), device="cpu")], backend="ray")
        got = w.echo(scatter(["a", "b", "c"])).all()
        assert got == {(0, 0): "a", (1, 0): "b", (2, 0): "c"}
        # length mismatch fails loud
        with pytest.raises(ValueError):
            w.echo(scatter(["only-one"])).all()
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_peer_wg_client_identity_alignment():
    """Passing a peer ``WorkerGroup`` as a method arg injects a ``PeerClient`` bound to
    the dp-aligned producer cell: consumer cell r reaches producer cell r (§1.2 (b))."""
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles.core import factory as roles

    def setup_prod():
        import os

        class P:
            def whoami(self):
                return os.getpid()

        return {"prod": P()}

    def setup_cons():
        class C:
            def ask(self, peer):
                return peer.whoami()  # calls the injected Ray-free peer client

        return {"cons": C()}

    try:
        prod = roles.rollout("prod", replicas=2, ranks=1)
        cons = roles.env("cons", replicas=2, ranks=1)
        bind_device_groups(
            [DeviceGroup((ActorGroup((prod,), setup_prod), ActorGroup((cons,), setup_cons)), device="cpu")],
            backend="ray",
        )
        prod_pids = prod.whoami().all()  # {(r,0): producer pid}
        via_peer = cons.ask(prod).all()  # {(r,0): pid the consumer reached via its peer client}
        assert via_peer == prod_pids  # identity alignment: cell r -> cell r
        assert len(set(prod_pids.values())) == 2  # two distinct producer processes
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_driver_dispatch_shape_scatter_peer_and_as_completed():
    """The eval driver's exact dispatch on toy bodies: ``sim.run(scatter(bins),
    policy=policy_wg)`` — each sim cell gets its bin AND a peer client to policy cell r —
    drained via ``Handle.as_completed()`` (§2.4/§1.2 (b))."""
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles.core import factory as roles

    def setup_policy():
        class P:
            def infer(self, x):
                return x * 10

        return {"policy": P()}

    def setup_sim():
        class S:
            def run(self, item, policy):  # policy is the injected peer client
                return {"item": item, "via_policy": policy.infer(item)}

        return {"env": S()}

    try:
        pol = roles.rollout("policy", replicas=3, ranks=1)
        sim = roles.env("env", replicas=3, ranks=1)
        bind_device_groups(
            [DeviceGroup((ActorGroup((pol,), setup_policy), ActorGroup((sim,), setup_sim)), device="cpu")],
            backend="ray",
        )
        # placement introspection through the WG (no reaching into _backend)
        assert set(sim.hosts()) == {(0, 0), (1, 0), (2, 0)}
        assert set(pol.gpus()) == {(0, 0), (1, 0), (2, 0)}
        got = dict(sim.run(scatter([1, 2, 3]), policy=pol).as_completed())
        assert got == {
            (0, 0): {"item": 1, "via_policy": 10},
            (1, 0): {"item": 2, "via_policy": 20},
            (2, 0): {"item": 3, "via_policy": 30},
        }
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_handle_as_completed_surfaces_per_cell_error():
    """A cell whose body raises yields the Exception as its value — one failure never
    aborts the drain (the driver records it as a colocate error)."""
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles.core import factory as roles

    def setup():
        class W:
            def run(self, x):
                if x == "boom":
                    raise RuntimeError("kaboom")
                return x

        return {"w": W()}

    try:
        w = roles.env("w", replicas=2, ranks=1)
        bind_device_groups([DeviceGroup((ActorGroup((w,), setup),), device="cpu")], backend="ray")
        got = dict(w.run(scatter(["ok", "boom"])).as_completed())
        assert got[(0, 0)] == "ok"
        assert isinstance(got[(1, 0)], Exception)
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_hosts_and_gpus_introspection():
    """Placement introspection is read from the backend, not from role bodies (§4.1)."""
    ray.init(ignore_reinit_error=True, num_cpus=3, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        grid = spawn_grid(
            _pid_setup, replicas=2, ranks=1, group_id="g0", gpu_request=0, threshold=0, materialize_names=()
        )
        peers = {sid: host for sid, host in grid.values()}
        ray.get([host.set_peers.remote(peers) for _, host in grid.values()])
        gb = GridBackend(grid, shape=(2, 1))
        assert set(gb.hosts()) == {(0, 0), (1, 0)}
        assert set(gb.gpus()) == {(0, 0), (1, 0)}
        assert all(isinstance(ip, str) for ip in gb.hosts().values())
    finally:
        ray.shutdown()
