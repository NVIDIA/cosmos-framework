# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NIXL weight transport: availability probing, lazy imports, and the mover surface.

Everything here must pass on a machine with NO ray / nixl / cuda-python / cupy — that is
the training container, and it is the whole point of the lazy-import discipline.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

_NIXL_PY = pathlib.Path(__file__).resolve().parents[1] / "workers" / "nixl.py"
_RAY_PY = pathlib.Path(__file__).resolve().parents[1] / "workers" / "ray.py"


@pytest.mark.L0
@pytest.mark.CPU
def test_module_imports_without_ray_or_nixl():
    # The training container has NO ray, nixl, cuda-python or cupy. Importing this module must
    # still succeed, or every posttrain run breaks the moment workers/ touches it.
    mod = importlib.import_module("cosmos_framework.posttrain.workers.nixl")
    assert hasattr(mod, "nixl_available")


@pytest.mark.L0
@pytest.mark.CPU
def test_no_module_scope_imports_of_optional_deps():
    """The lazy-import discipline, enforced structurally rather than by convention.

    A module-scope `import ray` here would make posttrain unimportable in the training
    container, and the failure would surface far from its cause."""
    tree = ast.parse(_NIXL_PY.read_text())
    optional = {"ray", "nixl", "cupy", "cuda"}
    bad = []
    for node in tree.body:  # module scope ONLY, not ast.walk
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] in optional]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in optional:
                bad.append(node.module)
    assert bad == [], f"optional deps imported at module scope: {bad}"


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_available_reports_rather_than_raises():
    from cosmos_framework.posttrain.workers import nixl as N

    ok, reason = N.nixl_available()
    assert isinstance(ok, bool)
    if not ok:
        assert reason, "unavailability must explain itself"


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_available_checks_every_fabric_allocator_dependency() -> None:
    source = _NIXL_PY.read_text()
    body = source.split("def nixl_available", 1)[1].split("\n\ndef ", 1)[0]
    assert "import cupy" in body
    assert "from cuda.bindings import driver" in body


@pytest.mark.L0
@pytest.mark.CPU
def test_get_fails_loud_when_unavailable():
    from cosmos_framework.posttrain.workers import nixl as N

    ok, _ = N.nixl_available()
    if ok:
        pytest.skip("ray+nixl present; the fail-loud path is not reachable here")
    with pytest.raises(RuntimeError, match="NIXL transport unavailable"):
        N.NixlTransport.get()


@pytest.mark.L0
@pytest.mark.CPU
def test_transport_state_does_not_live_in_the_cloudpickled_actor_module():
    """workers/ray.py is shipped BY VALUE as the actor class, so Ray rebuilds its module
    namespace with the same __name__ but a DIFFERENT dict — which is why current_pdims()
    silently returned None for every Ray-backed setup (workers/ray.py:376-383).

    NixlTransport's singleton must therefore live in workers/nixl.py, which the actor
    reaches through a normal import resolved via sys.modules. Checked textually because
    importing workers/ray.py requires `ray`, which the training container lacks."""
    # AST rather than substrings: `"_singleton" not in src` would fire on any unrelated future
    # use of that name in ray.py and report it as a NIXL architecture violation.
    tree = ast.parse(_RAY_PY.read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "NixlTransport" not in classes, "move NixlTransport back to workers/nixl.py"
    module_globals = {
        t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }
    assert "_singleton" not in module_globals, (
        "NIXL singleton state must not be a module global in the cloudpickled actor module"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_mover_registers_itself_under_the_nixl_hint():
    import cosmos_framework.posttrain.workers.nixl as N
    from cosmos_framework.posttrain.comm.transfer import movers as M

    assert M._HINT_MOVERS.get("nixl") is N.NixlWeightMover


@pytest.mark.L0
@pytest.mark.CPU
def test_mover_surface_matches_the_staged_contract():
    import cosmos_framework.posttrain.workers.nixl as N
    from cosmos_framework.posttrain.comm.transfer.mover import STAGED

    # STAGED, not COLLECTIVE: NIXL is one-sided, so pull_stream_from_peer takes the
    # synchronous per-bucket branch and never enters the broadcast path (design §1).
    assert N.NixlWeightMover.lifetime == STAGED
    assert N.NixlWeightMover.tensors_only is True


@pytest.mark.L0
@pytest.mark.CPU
def test_transport_reuses_recycled_probe_buffer() -> None:
    from cosmos_framework.posttrain.workers.nixl import NixlTransport

    class _Buffer:
        @staticmethod
        def numel() -> int:
            return 1024

    transport = NixlTransport()
    buffer = _Buffer()
    keepalive = object()
    transport.recycle_buffer(buffer, keepalive)
    got_buffer, got_keepalive = transport.new_buffer(512)
    assert got_buffer is buffer
    assert got_keepalive is keepalive
    assert transport._spare_buffer is None


@pytest.mark.L0
@pytest.mark.CPU
def test_transport_releases_probe_buffer_smaller_than_model_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.workers.nixl import NixlTransport

    class _Buffer:
        @staticmethod
        def numel() -> int:
            return 512

    transport = NixlTransport()
    buffer = _Buffer()
    keepalive = object()
    released: list[tuple[object, object]] = []
    transport.recycle_buffer(buffer, keepalive)
    monkeypatch.setattr(transport, "release_buffer", lambda buf, keep: released.append((buf, keep)))
    transport.reserve(1024)
    assert released == [(buffer, keepalive)]
    assert transport._spare_buffer is None


@pytest.mark.L0
@pytest.mark.CPU
def test_transport_rejects_a_second_recycled_probe_buffer() -> None:
    from cosmos_framework.posttrain.workers.nixl import NixlTransport

    transport = NixlTransport()
    transport.recycle_buffer(object(), object())
    with pytest.raises(RuntimeError, match="already holds a recycled probe buffer"):
        transport.recycle_buffer(object(), object())


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_mover_selects_it_by_hint():
    import cosmos_framework.posttrain.workers.nixl as N  # noqa: F401  (registers the hint)
    from cosmos_framework.posttrain.comm.transfer.movers import resolve_mover

    got = resolve_mover(relationship="cross_device", plane="weight", capability="tensor", hint="nixl")
    assert isinstance(got, N.NixlWeightMover)


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_rejects_a_handle_with_no_ref():
    import cosmos_framework.posttrain.workers.nixl as N

    with pytest.raises(ValueError, match="ObjectRef"):
        N.NixlWeightMover().fetch(None)


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_device_still_fails_loud_without_nixl():
    # The existing guidance must survive verbatim when NIXL is not requested -- silently
    # falling back to a comm-less NcclMover would hang at the first push, not error.
    from cosmos_framework.posttrain.control.placement.colocate import _cross_device_transport

    hint, reason = _cross_device_transport(None)
    assert hint is None
    assert "init_weight_plane" in reason


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_device_resolves_to_nixl_when_enabled_and_available():
    from cosmos_framework.posttrain.control.placement.colocate import _cross_device_transport
    from cosmos_framework.posttrain.workers.nixl import nixl_available

    ok, _ = nixl_available()
    hint, reason = _cross_device_transport("nixl")
    if ok:
        assert hint == "nixl" and reason == ""
    else:
        # Requested but unavailable must name the missing piece, not fall back silently.
        assert hint is None and "unavailable" in reason


@pytest.mark.L0
@pytest.mark.CPU
def test_init_nixl_is_an_actor_method_not_driver_side():
    """bind_device_groups runs on the DRIVER and reaches actors only via .remote().
    Constructing the transport driver-side would leave every actor uninitialized and fail
    at the first sync, so init_nixl must live on _HostImpl."""
    src = _RAY_PY.read_text()
    assert "def init_nixl" in src, "init_nixl must be a _Host method"
    from cosmos_framework.posttrain.control.placement.colocate import _cross_device_transport  # noqa: F401

    colo = (_RAY_PY.parent.parent / "control" / "placement" / "colocate.py").read_text()
    assert "init_nixl.remote(" in colo, "bind must invoke init_nixl on each actor via .remote()"


@pytest.mark.L0
@pytest.mark.CPU
def test_warm_nixl_skips_the_cuda_transport_on_cpu_placed_actors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-placed actor warms only Ray's NIXL agent; a GPU-assigned actor also binds the CUDA transport."""
    pytest.importorskip("ray")
    from unittest.mock import MagicMock

    from cosmos_framework.posttrain.workers import data_pool, nixl
    from cosmos_framework.posttrain.workers import ray as ray_workers

    transport = MagicMock(name="NixlTransport")
    monkeypatch.setattr(nixl.NixlTransport, "get", classmethod(lambda cls: transport))
    manager = MagicMock(name="tensor_transport_manager")
    rdt_util = importlib.import_module("ray.experimental.rdt.util")
    monkeypatch.setattr(rdt_util, "get_tensor_transport_manager", lambda backend: manager)
    host = object.__new__(ray_workers._HostImpl)

    monkeypatch.setattr(data_pool, "actor_cuda_accessible", lambda: False)
    host.warm_nixl()
    manager.get_nixl_agent.assert_called_once_with()
    transport.warmup.assert_not_called()

    monkeypatch.setattr(data_pool, "actor_cuda_accessible", lambda: True)
    host.warm_nixl()
    transport.warmup.assert_called_once_with()
    manager.get_nixl_agent.assert_called_once_with()


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_reservation_is_endpoint_scoped_and_guarded() -> None:
    """Only weight endpoints reserve NIXL, and reservation failure uses bind rollback."""
    colo = (_RAY_PY.parent.parent / "control" / "placement" / "colocate.py").read_text()
    reserve_at = colo.index("host.reserve_nixl.remote")
    guard_at = colo.rfind("_guard_bind(", 0, reserve_at)
    spec_at = colo.index("spec =", reserve_at)
    reservation = colo[guard_at:spec_at]

    assert "for actor_name in endpoint_actor_names" in reservation
    assert "for host in _weight_hosts(actor_name)" in reservation
    assert "all_hosts" not in reservation


@pytest.mark.L0
@pytest.mark.CPU
def test_real_nixl_probe_precedes_model_setup() -> None:
    """The capability gate must run after actor rendezvous but before expensive setup functions."""
    colo = (_RAY_PY.parent.parent / "control" / "placement" / "colocate.py").read_text()
    probe_at = colo.index("_run_nixl_probe(", colo.index("def bind_device_groups"))
    setup_at = colo.index("setup_grid(", probe_at)
    assert "run_setup=False" in colo[:probe_at]
    assert "enable_tensor_transport=(" in colo[:probe_at]
    assert probe_at < setup_at

    grid = (_RAY_PY.parent / "grid.py").read_text()
    spawn_at = grid.index("def spawn_grid")
    warm_at = grid.index("host.warm_nixl.remote()", spawn_at)
    rendezvous_at = grid.index(".init_distributed.remote(", warm_at)
    assert "if enable_tensor_transport:" in grid[spawn_at:warm_at]
    assert warm_at < rendezvous_at


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_runtime_contracts_are_pinned_and_launchers_do_not_install() -> None:
    from cosmos_framework.posttrain.utils.nixl_preflight import _runtime_contract

    assert _runtime_contract("cu12") == (
        {
            "cuda-bindings": "12.9.4",
            "cuda-python": "12.9.4",
            "cupy-cuda12x": "13.6.0",
            "fastrlock": "0.8.3",
            "nixl": "1.3.2",
            "nixl-cu12": "1.3.2",
            "ray": "2.56.1",
        },
        "12.8",
    )
    assert _runtime_contract("cu13") == (
        {
            "cuda-bindings": "13.0.3",
            "cuda-python": "13.0.3",
            "cupy-cuda13x": "13.6.0",
            "fastrlock": "0.8.3",
            "nixl": "1.3.2",
            "nixl-cu13": "1.3.2",
            "ray": "2.56.1",
        },
        "13.",
    )
    assert _runtime_contract("cu128") == _runtime_contract("cu12")
    assert _runtime_contract("cu130") == _runtime_contract("cu13")
    with pytest.raises(ValueError, match="unsupported POSTTRAIN_NIXL_CUDA_FLAVOR"):
        _runtime_contract("cu11")
    qualification = (_RAY_PY.parent.parent / "tests" / "run_asymmetric_nixl_qualification.sh").read_text()
    # The whole launcher contract now rides on the one surviving NIXL wrapper: the image supplies the
    # runtime (no install step), the preflight proves it on every node, and no personal path is baked in.
    assert "pip install" not in qualification
    assert "ensure_nixl_overlay" not in qualification
    assert "cosmos_framework.posttrain.utils.nixl_preflight" in qualification
    assert "POSTTRAIN_NIXL_RAYLETS_READY" in qualification
    assert "PREFLIGHT_CMD" in qualification
    assert "UCX_MODULE_DIR" in qualification
    assert "/users/" not in qualification
    # The wrapper runs on the Slurm host; /opt/venv exists only after it enters each container.
    assert '[ -x "$PYTHON_BIN" ]' not in qualification


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_preflight_validates_ucx_declarations() -> None:
    from cosmos_framework.posttrain.utils.nixl_preflight import _ucx_declaration_failures, _ucx_module_failures

    assert _ucx_declaration_failures("rc_mlx5,cuda_copy,tcp", "mlx5_0:1,eth0") == []
    failures = _ucx_declaration_failures("rc_mlx5", "all")
    assert any("tcp" in failure for failure in failures)
    assert any("CUDA transport" in failure for failure in failures)
    assert any("UCX_NET_DEVICES" in failure for failure in failures)
    assert _ucx_module_failures("") == ["UCX_MODULE_DIR must select the baked UCX plugin directory"]


@pytest.mark.L0
@pytest.mark.CPU
def test_nixl_preflight_requires_a_cuda_ucx_plugin(tmp_path: pathlib.Path) -> None:
    from cosmos_framework.posttrain.utils.nixl_preflight import _ucx_module_failures

    assert "does not exist" in _ucx_module_failures(str(tmp_path / "missing"))[0]
    assert "no libuct_cuda.so" in _ucx_module_failures(str(tmp_path))[0]
    (tmp_path / "libuct_cuda.so.0").touch()
    assert _ucx_module_failures(str(tmp_path)) == []


@pytest.mark.L0
@pytest.mark.CPU
def test_stage_waits_before_reassigning_views():
    """The §4 ordering invariant, checked structurally so it runs in CI where the GPU e2e
    cannot.

    `pack` returns a FRESH view each call and `wait_tensor_freed` keys on `id(tensor)`, so
    `_wait_free()` must run BEFORE `self._views` is reassigned — otherwise it waits on the
    object it is about to publish rather than the one still being read, and becomes a silent
    no-op. Swapping those two statements leaves every test green but corrupts weights under a
    one-sided read, so the ordering is asserted directly."""
    tree = ast.parse(_NIXL_PY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NixlWeightMover")
    stage = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "stage")

    wait_at = views_at = None
    for i, node in enumerate(stage.body):
        src = ast.dump(node)
        if wait_at is None and "_wait_free" in src:
            wait_at = i
        if views_at is None and isinstance(node, ast.Assign) and "_views" in ast.dump(node):
            views_at = i
    assert wait_at is not None, "stage() must call _wait_free()"
    assert views_at is not None, "stage() must record the published views"
    assert wait_at < views_at, (
        "_wait_free() must precede the self._views assignment, or it waits on the object being "
        "published instead of the one the peer may still be reading"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_init_nixl_sizes_only_from_staging_bodies():
    """A vLLM rollout body exposes `.model`, but it is an `LLM` engine handle with no
    `state_dict` — `WeightMapper.manifest` raises on it. Sizing must therefore select on the
    STAGING capability, not on `hasattr(body, "model")`, or bind dies with
    `AttributeError: 'LLM' object has no attribute 'state_dict'` (observed running a
    disaggregated NIXL GRPO job).

    Also asserts warmup is unconditional: a consumer-only actor allocates no staging buffer
    but still runs a NIXL receive, and skipping warmup would kill its first transfer with
    NIXL_ERR_BACKEND."""
    src = _RAY_PY.read_text()
    body = src.split("def init_nixl", 1)[1].split("\n    def ", 1)[0]

    assert 'getattr(net, "state_dict", None)' in body, (
        "init_nixl must select staging bodies by state_dict capability, not by hasattr(.model)"
    )
    warm_at = body.index("t.warmup()")
    need_at = body.index("if need == 0")
    assert warm_at < need_at, "warmup() must run before the need==0 early return"


@pytest.mark.L0
@pytest.mark.CPU
def test_bind_threads_one_bucket_bytes_to_both_sizing_and_plan():
    """The staging buffer and the bucket plan are computed at DIFFERENT times (buffer at bind,
    plan at first sync) and must not diverge: a plan with larger buckets than the buffer was
    sized for fails mid-sync with 'staging buffer too small', and the buffer refuses to grow
    because the buffer is registered and pinned. So bind must pass ONE value to both."""
    colo = (_RAY_PY.parent.parent / "control" / "placement" / "colocate.py").read_text()
    assert "weight_sync_bucket_bytes" in colo, "bind must expose a bucket_bytes knob"
    assert "init_nixl.remote(weight_sync_bucket_bytes," in colo, "buffer sizing must take the knob"
    assert '"bucket_bytes": weight_sync_bucket_bytes' in colo, "the plan must take the SAME knob"

    # AST, not substring: the previous form was `X in body or ("bucket_bytes" in <text before the
    # call>)`, whose second branch is satisfied by the `bucket_bytes` in init_nixl's own SIGNATURE.
    # It therefore passed unconditionally -- including for the change it exists to catch, a body
    # that ignores the parameter and hardcodes BUCKET_BYTES. Check the call site's argument.
    tree = ast.parse(_RAY_PY.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "init_nixl")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "bucket_plan"]
    assert calls, "init_nixl must call bucket_plan"
    names = {n.id for c in calls for a in c.args for n in ast.walk(a) if isinstance(n, ast.Name)}
    assert "bucket_bytes" in names, (
        f"bucket_plan must be called with the configured bucket_bytes, not only the default; "
        f"call references {sorted(names)}"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_target_buffer_view_is_persisted_not_passed_inline(monkeypatch):
    """`set_target_for_ref` stores WEAK references, so the tensor handed to it must be kept
    alive by the mover. Passing `self._recv[:need]` inline hands Ray a temporary that nothing
    references — Ray then silently allocates and registers a fresh receive buffer on every
    fetch (no error, 1.3x on GPU / ~9x on host memory). Holding the PARENT is not enough: the
    weak ref is to the view, so the view object itself must be cached.

    Driven through the real `fetch` against a fake `ray`, so the mistake is caught by its actual
    signature — the target dying — rather than by how the call site happens to be spelled.
    `_recv` is pre-seeded with CPU memory purely to keep this off the GPU; the device choice is
    not what is under test, the lifetime of the handed-out object is.
    """
    import gc
    import sys
    import types
    import weakref

    import torch

    from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes, flat_plan
    from cosmos_framework.posttrain.comm.transfer.mover import MoveHandle
    from cosmos_framework.posttrain.workers.nixl import NixlWeightMover

    obj = {"a.weight": torch.zeros(4, 8, dtype=torch.bfloat16), "a.norm": torch.zeros(8)}
    plan = flat_plan(obj)
    need = flat_nbytes(plan)
    landed = torch.zeros(need, dtype=torch.uint8)

    seen: list = []
    fake_ray = types.ModuleType("ray")
    fake_exp = types.ModuleType("ray.experimental")
    fake_ray.get = lambda ref: [landed]
    # Exactly what Ray does: keep a WEAK reference to the target and nothing more.
    fake_exp.set_target_for_ref = lambda ref, targets: seen.append(weakref.ref(targets[0]))
    fake_ray.experimental = fake_exp
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.experimental", fake_exp)

    mover = NixlWeightMover()
    mover._recv = torch.empty(need, dtype=torch.uint8)  # see docstring: keeps the test on CPU
    out = mover.fetch(MoveHandle(kind="nixl", payload=object(), meta={"plan": plan}))
    assert set(out) == set(obj)

    gc.collect()
    assert seen, "fetch must hand a target buffer to set_target_for_ref"
    assert seen[0]() is not None, (
        "the tensor passed to set_target_for_ref was collected the moment fetch returned. Ray "
        "keeps only a weak ref, so it will silently allocate and register a fresh receive buffer "
        "on every fetch — cache the VIEW on the mover instead of slicing inline."
    )
    assert seen[0]() is mover._target, "the persisted view must be the one Ray was given"


@pytest.mark.L0
@pytest.mark.CPU
def test_version_guidance_derives_from_min_ray():
    """Guidance strings must be derived from MIN_RAY, never hardcoded.

    A stale literal can send users toward the wrong dependency fix. The build-time dependency group
    pins the exact floor because safe probe-buffer cleanup requires Ray's deregistration API."""
    from cosmos_framework.posttrain.workers import nixl as N

    floor = ".".join(map(str, N.MIN_RAY))
    for path in (_NIXL_PY,):
        src = path.read_text()
        # Any ray version literal other than the one MIN_RAY renders to is a drift hazard.
        literals = set(re.findall(r"ray>=(\d+\.\d+(?:\.\d+)?)", src))
        assert literals <= {floor}, (
            f"{path.name}: hardcoded ray version(s) {sorted(literals - {floor})} contradict "
            f"MIN_RAY={N.MIN_RAY}; derive with \"ray>={{'.'.join(map(str, MIN_RAY))}}\""
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_each_mover_owns_its_staging_buffer():
    """Two movers in one actor must never write the same bytes.

    `register_nixl_memory` grants addressability, not access control -- a raw pack() into a
    shared region would land on top of bytes a one-sided peer might still be reading for the
    other mover, silently, since each `_wait_free()` only knows its own publications. Owning
    the buffer per mover makes `self._views` the complete story for those bytes.

    Structural: allocating a real fabric buffer needs a GPU + nixl, so assert the ownership
    instead -- the buffer must be per-mover state, and the transport must hand out FRESH
    buffers rather than caching one."""
    tree = ast.parse(_NIXL_PY.read_text())
    mover = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NixlWeightMover")
    init = next(n for n in mover.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assigned = {
        t.attr for st in init.body if isinstance(st, ast.Assign) for t in st.targets if isinstance(t, ast.Attribute)
    }
    assert {"_buf", "_keep"} <= assigned, f"mover must own its buffer; got {sorted(assigned)}"

    transport = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NixlTransport")
    tinit = next(n for n in transport.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    tattrs = {
        t.attr for st in tinit.body if isinstance(st, ast.Assign) for t in st.targets if isinstance(t, ast.Attribute)
    }
    assert "_buf" not in tattrs, "the transport must NOT cache a buffer; movers own theirs"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "new_buffer" for n in transport.body), (
        "transport must expose new_buffer() handing an idle or fresh buffer to each mover"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_probe_buffer_lifecycle_has_reuse_and_real_release() -> None:
    """A pre-model probe must not leave a second registered fabric allocation behind."""
    src = _NIXL_PY.read_text()
    assert "deregister_nixl_memory" in src
    assert "recycle_staging_buffer" in src
    assert "cuMemUnmap" in src and "cuMemRelease" in src


@pytest.mark.L0
@pytest.mark.CPU
def test_mover_implements_consumer_done():
    """`WeightSync.apply_bucket_state` drains the consumer via an OPTIONAL `consumer_done` hook
    (`getattr(self.mover, "consumer_done", None)`), so a mover without one silently skips the
    drain -- there is no error and no signal.

    NixlWeightMover needs it: `carve` returns views into `self._recv`, the sink's copies out of
    those views are async, and the next bucket's `ray.get` writes the same buffer via a
    one-sided transfer that is not ordered against this process's torch stream. Ray does not
    cover it -- `wait_fetch_complete` polls NIXL transfer state, and the only CUDA sync in
    ray.experimental.rdt is on the send path (cuda_ipc_transport, by contrast, orders with
    CUDA events)."""
    import cosmos_framework.posttrain.workers.nixl as N

    assert callable(getattr(N.NixlWeightMover, "consumer_done", None)), (
        "NixlWeightMover must implement consumer_done; apply_bucket_state skips the drain "
        "silently when the hook is absent"
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "version,expected",
    [
        ("2.55.0", (2, 55, 0)),
        ("2.55.1", (2, 55, 1)),
        ("2.56.1", (2, 56, 1)),
        ("2.55.1rc0", (2, 55, 1)),  # rc: filtering used to yield (2, 55) -> "below floor"
        ("2.55.0+cu12", (2, 55, 0)),  # local build: same
        ("2.55", (2, 55, 0)),  # short: 2.55 means 2.55.0
        ("3.0.0.dev0", (3, 0, 0)),
    ],
)
def test_version_parsing_never_shortens_the_tuple(version, expected):
    """Tuple comparison is length-sensitive -- `(2, 55) < (2, 55, 0)` -- so dropping a
    non-numeric component reports a NEWER ray as below MIN_RAY, with a self-contradictory
    message, and `_cross_device_transport` turns that into a bind-time refusal. Every parse must
    yield exactly three components."""
    from cosmos_framework.posttrain.workers.nixl import MIN_RAY, _release

    got = _release(version)
    assert got == expected, f"{version} parsed to {got}, expected {expected}"
    assert len(got) == 3
    if version.startswith(("2.56.1", "3.")):
        assert got >= MIN_RAY, f"{version} must satisfy MIN_RAY={MIN_RAY}, got {got}"
    elif version.startswith("2.55"):
        assert got < MIN_RAY


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_transport_hint_is_rejected_by_name():
    """`weight_sync_transport == "nixl"` folded every typo into "no transport requested", which
    then emitted the NCCL guidance telling the caller to pass `weight_sync_transport='nixl'` --
    advice they had just followed. The hint must be validated as a NAME."""
    from cosmos_framework.posttrain.control.placement.colocate import _check_transport_hint

    _check_transport_hint("nixl")  # registered (the import inside registers it)
    for bad in ("NIXL", "ucx", "nixl "):
        with pytest.raises(ValueError, match="unknown weight_sync_transport"):
            _check_transport_hint(bad)


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_device_transport_takes_the_string_not_a_bool():
    """Signature guard: collapsing to a bool at the call site is what made movers.py's good
    diagnostic unreachable from bind."""
    import inspect

    from cosmos_framework.posttrain.control.placement.colocate import _cross_device_transport

    (param,) = inspect.signature(_cross_device_transport).parameters.values()
    assert "str" in str(param.annotation), f"expected a string hint, got {param.annotation}"
    hint, reason = _cross_device_transport(None)
    assert hint is None and "init_weight_plane" in reason
