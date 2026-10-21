# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Test that _Host.run_setup constructs bodies INSIDE the actor and they share a
resource constructed there (DESIGN §4.2 tier-3, §9 worker-resident)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ray = pytest.importorskip("ray")

import cosmos_framework.posttrain.workers.ray as workers_ray  # noqa: E402
from cosmos_framework.posttrain.workers.ray import (  # noqa: E402
    PEER_GPUS_ENV,
    RayBackend,
    _expose_peer_gpus,
    _Host,
    _HostImpl,
    _prepend_actor_pythonpath,
)


def _setup():
    shared = {"v": 0}

    class A:
        def __init__(self, s):
            self.s = s

        def inc(self):
            self.s["v"] += 1
            return self.s["v"]

    class B:
        def __init__(self, s):
            self.s = s

        def read(self):
            return self.s["v"]

    return {"a": A(shared), "b": B(shared)}  # a and b share `shared`


@pytest.mark.L0
@pytest.mark.CPU
def test_run_setup_builds_shared_bodies_in_actor():
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        names = ray.get(host.run_setup.remote(_setup))
        assert set(names) == {"a", "b"}
        be = RayBackend(host)
        be.submit("a", None, "inc", (), {}).result()
        be.submit("a", None, "inc", (), {}).result()
        # b reads the SAME shared dict mutated via a -> proves shared construction
        assert be.submit("b", None, "read", (), {}).result() == 2
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_rendezvous_endpoint_returns_ip_and_free_port():
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        h = _Host.remote()
        ip, port = ray.get(h.rendezvous_endpoint.remote())  # per-replica TP/FSDP MASTER_ADDR:PORT
        assert isinstance(ip, str) and ip
        assert isinstance(port, int) and 1024 < port < 65536
    finally:
        ray.shutdown()


def _two_model_body_setup():
    # Two model-bearing bodies in one actor; each can report its bound backend's identity.
    class M:
        def __init__(self):
            self.model = object()  # only needs to LOOK like a model (has .model)

        def backend_id(self):
            return id(self._weight_backend)

        def has_backend(self) -> bool:
            return hasattr(self, "_weight_backend")

        def backend_uses_single_buffer_ipc(self) -> bool:
            return bool(getattr(self._weight_backend.mover, "_single_buffer", False))

    return {"x": M(), "y": M()}


@pytest.mark.L0
@pytest.mark.CPU
def test_set_weight_backend_is_per_body_not_shared():
    # Guard: each model-bearing body must get its OWN backend instance. A shared instance would let
    # one body's IpcMover.stage() free another's pinned snapshot (GPU use-after-free).
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_two_model_body_setup))
        ray.get(host.set_weight_backend.remote({"kind": "copy"}))
        be = RayBackend(host)
        idx = be.submit("x", None, "backend_id", (), {}).result()
        idy = be.submit("y", None, "backend_id", (), {}).result()
        assert idx != idy  # distinct instances per body
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_set_weight_backend_scopes_to_declared_body_names() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_two_model_body_setup))
        ray.get(host.set_weight_backend.remote({"kind": "copy"}, ("x",)))
        backend = RayBackend(host)
        assert backend.submit("x", None, "has_backend", (), {}).result() is True
        assert backend.submit("y", None, "has_backend", (), {}).result() is False
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_set_weight_backend_same_gpu_always_uses_single_buffer_ipc() -> None:
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_two_model_body_setup))
        ray.get(host.set_weight_backend.remote({"relationship": "same_gpu"}, ("x", "y")))
        backend = RayBackend(host)
        assert backend.submit("x", None, "backend_uses_single_buffer_ipc", (), {}).result() is True
        assert backend.submit("y", None, "backend_uses_single_buffer_ipc", (), {}).result() is True
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_node_id_answered_by_host_not_body():
    # node_id is host/placement introspection: _Host.invoke routes it to the ACTOR, so a body with
    # NO node_id method still answers it (role bodies stay Ray-agnostic — they must not know Ray
    # node placement). The multinode value test relies on this to check mesh span / co-location.
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_two_model_body_setup))  # bodies "x"/"y" have no node_id
        nid = ray.get(host.invoke.remote("x", "node_id", (), {}))
        assert isinstance(nid, str) and nid  # host answered; no AttributeError on the body
    finally:
        ray.shutdown()


def _peer_body_setup():
    class P:
        def echo(self, x):
            return ("echo", x)

        def call_async(self):  # same NAME as the client method — must NOT shadow the client's
            return "body-call_async"

    return {"p": P()}


@pytest.mark.L0
@pytest.mark.CPU
def test_peer_client_call_async_fires_without_blocking():
    # PeerClient.call_async → future with .result() (the COLLECTIVE branch of pull_stream_from_peer
    # fires the producer's push before entering its own recv — a synchronous call would deadlock).
    # Also pin the shadowing rule: call_async is a REAL method, so it wins attribute lookup and is
    # never routed to a peer-body method of the same name; the sync __getattr__ path still works.
    from cosmos_framework.posttrain.workers.ray import PeerClient

    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_peer_body_setup))
        peer = PeerClient(host, "p")
        fut = peer.call_async("echo", 7)  # non-blocking fire
        assert fut.result() == ("echo", 7)
        assert peer.echo(8) == ("echo", 8)  # sync __getattr__ path unaffected
        # the client's real method won lookup; the body's same-named method is reachable only via it
        assert peer.call_async("call_async").result() == "body-call_async"
    finally:
        ray.shutdown()


class _FakeRemoteMethod:
    def __init__(self, calls: list, refs: list) -> None:
        self._calls = calls
        self._refs = refs

    def remote(self, *args: Any, **kwargs: Any) -> object:
        ref = object()
        self._calls.append((args, kwargs))
        self._refs.append(ref)
        return ref


class _FakeWeightHost:
    def __init__(self, calls: list, refs: list) -> None:
        self.invoke = _FakeRemoteMethod(calls, refs)


@pytest.mark.L0
@pytest.mark.CPU
def test_routed_weight_peer_submits_every_target_before_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.workers.ray import RoutedWeightPeerClient

    calls: list = []
    refs: list = []
    targets = (
        ((0, 0), _FakeWeightHost(calls, refs)),
        ((0, 2), _FakeWeightHost(calls, refs)),
    )
    payload = object()

    def wait(all_refs: list, *, num_returns: int, timeout: float) -> tuple[list, list]:
        assert len(calls) == 2, "all rollout calls must be submitted before policy waits"
        assert all_refs == refs and num_returns == 2 and timeout == 7.0
        return list(all_refs), []

    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", lambda _ref: None)
    client = RoutedWeightPeerClient(targets, "rollout", timeout=7.0)
    client.apply_bucket(payload, 3)

    assert [call[0][1] for call in calls] == ["apply_bucket", "apply_bucket"]
    assert all(call[0][2] == (payload, 3) for call in calls)


@pytest.mark.L0
@pytest.mark.CPU
def test_routed_weight_peer_drains_completed_target_and_bounds_pending_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.workers.ray import RoutedWeightPeerClient

    calls: list = []
    refs: list = []
    targets = (
        ((1, 0), _FakeWeightHost(calls, refs)),
        ((1, 1), _FakeWeightHost(calls, refs)),
    )
    drained: list = []

    def wait(all_refs: list, *, num_returns: int, timeout: float) -> tuple[list, list]:
        assert num_returns == 2 and timeout == 0.25
        return [all_refs[0]], [all_refs[1]]

    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", lambda ref: drained.append(ref))
    client = RoutedWeightPeerClient(targets, "rollout", timeout=0.25)
    with pytest.raises(RuntimeError, match=r"rollout cell \(1, 1\).*TimeoutError"):
        client.mark_weight_version_ready(9)
    assert drained == [refs[0]]


@pytest.mark.L0
@pytest.mark.CPU
def test_routed_weight_peer_drains_all_completed_targets_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.workers.ray import RoutedWeightPeerClient

    calls: list = []
    refs: list = []
    targets = tuple(((2, rank), _FakeWeightHost(calls, refs)) for rank in range(3))
    drained: list = []

    monkeypatch.setattr(
        ray,
        "wait",
        lambda all_refs, num_returns, timeout: (list(all_refs), []),
    )

    def get(ref: object) -> None:
        drained.append(ref)
        if ref is refs[1]:
            raise ValueError("bad target")

    monkeypatch.setattr(ray, "get", get)
    client = RoutedWeightPeerClient(targets, "rollout")
    with pytest.raises(RuntimeError, match=r"rollout cell \(2, 1\).*bad target"):
        client.check_plan_coverage()
    assert drained == refs


def _declaring_body_setup():
    """One model-bearing body that declares a wire dtype, as `LLMTrainer` does."""

    class M:
        def __init__(self):
            self.model = object()  # only needs to LOOK like a model (has .model)
            self.transfer_dtype = "bfloat16"

        def backend_transfer_dtype(self):
            return self._weight_backend.transfer_dtype

    return {"x": M()}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("spec", [{"relationship": "host"}, {"kind": "copy"}], ids=["relationship", "kind"])
def test_bind_carries_the_body_transfer_dtype_into_the_backend(spec: dict):
    """Bind is the ONE crossing from the body's declaration to the backend that builds the plan.

    The wire dtype lives on the body (`LLMTrainer.transfer_dtype`), while `ensure_plan` receives
    only `body.model` -- so a backend constructed without it silently plans at the model's STORAGE
    dtype. With the shipped GSM8K recipe (`master_dtype=float32`, `transfer_dtype=bfloat16`) that
    is an fp32 plan against a bf16 staging buffer, roughly a factor of two, and the buffer cannot
    grow after bind.

    BOTH spec forms, because `set_weight_backend` builds the backend on two different branches and
    real placement only ever takes one of them: `control/placement/colocate.py` always sends a
    `relationship` spec, while `kind` is the legacy direct-test form. A version of this test that
    covered only `kind` passed while the production branch dropped the declaration entirely.

    Reading it off the backend rather than off a plan is deliberate: this pins the HANDOFF, and the
    restatement it enables is pinned separately by
    `test_comm_weight_producer_driven.py::test_producer_push_plan_declares_the_transfer_dtype_not_the_model_dtype`.
    """
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcray")
    try:
        host = _Host.remote()
        ray.get(host.run_setup.remote(_declaring_body_setup))
        ray.get(host.set_weight_backend.remote(spec))
        backend = RayBackend(host)
        assert backend.submit("x", None, "backend_transfer_dtype", (), {}).result() == "bfloat16"
    finally:
        ray.shutdown()


@pytest.mark.L0
@pytest.mark.CPU
def test_host_constructor_applies_the_driver_frozen_import_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "actor_reward_probe.py").write_text("VALUE = 42\n")
    inherited = tmp_path / "inherited"
    inherited.mkdir()
    installed = tmp_path / "site-packages"
    installed.mkdir()
    (installed / "actor_reward_probe.py").write_text("VALUE = 7\n")
    monkeypatch.setattr(
        sys,
        "path",
        [str(installed), *(entry for entry in sys.path if entry not in (str(external), str(inherited)))],
    )

    host = _HostImpl((str(external), str(inherited), str(external)))

    assert sys.path[:2] == [str(external), str(inherited)]
    assert __import__("actor_reward_probe").VALUE == 42
    assert host is not None
    _prepend_actor_pythonpath((str(inherited), str(external)))
    assert sys.path[:2] == [str(inherited), str(external)]


class _FakeNvml:
    """The three NVML calls `_expose_peer_gpus` makes, on a node with ``count`` GPUs."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.calls: list[str] = []

    def nvmlInit(self) -> None:
        self.calls.append("init")

    def nvmlDeviceGetCount(self) -> int:
        return self.count

    def nvmlShutdown(self) -> None:
        self.calls.append("shutdown")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("assigned", "expected"),
    [("0", "0,1,2,3"), ("2", "2,0,1,3"), ("3", "3,0,1,2")],
)
def test_expose_peer_gpus_keeps_the_ray_assigned_gpu_at_ordinal_zero(
    monkeypatch: pytest.MonkeyPatch,
    assigned: str,
    expected: str,
) -> None:
    nvml = _FakeNvml(4)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    monkeypatch.delenv(PEER_GPUS_ENV, raising=False)

    _expose_peer_gpus()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
    assert os.environ[PEER_GPUS_ENV] == "1"
    assert nvml.calls == ["init", "shutdown"]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("assigned", "cuda_initialized", "message"),
    [
        ("0,1", False, "exactly one Ray-assigned GPU index"),
        ("GPU-6f1c0b2a", False, "exactly one Ray-assigned GPU index"),
        ("", False, "exactly one Ray-assigned GPU index"),
        ("4", False, "not one of this node's 4 GPUs"),
        ("1", True, "before this process initializes CUDA"),
    ],
    ids=["two-assigned", "uuid", "none-assigned", "outside-node", "cuda-already-up"],
)
def test_expose_peer_gpus_refuses_a_visibility_it_cannot_reorder(
    monkeypatch: pytest.MonkeyPatch,
    assigned: str,
    cuda_initialized: bool,
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml(4))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialized)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    monkeypatch.delenv(PEER_GPUS_ENV, raising=False)

    with pytest.raises(RuntimeError, match=message):
        _expose_peer_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == assigned
    assert PEER_GPUS_ENV not in os.environ


@pytest.mark.L0
@pytest.mark.CPU
def test_host_exposes_peer_gpus_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml(4))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv(PEER_GPUS_ENV, raising=False)

    host = _HostImpl()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    assert host.expose_peer_gpus() == "2,0,1,3"
    assert os.environ[PEER_GPUS_ENV] == "1"


class _FakeCudaDriver:
    """The CUDA driver calls `gpu_identity` makes; ordinal N reports UUID bytes ``N+1`` repeated."""

    def __init__(self, count: int) -> None:
        self._count = count
        self.ordinals: list[int] = []
        self.cuDeviceGetUuid_v2 = self._get_uuid

    def cuInit(self, _flags: int) -> int:
        return 0

    def cuDeviceGetCount(self, count: Any) -> int:
        count._obj.value = self._count
        return 0

    def cuDeviceGet(self, device: Any, ordinal: int) -> int:
        self.ordinals.append(ordinal)
        device._obj.value = ordinal
        return 0

    def _get_uuid(self, uuid: Any, device: Any) -> int:
        for index in range(16):
            uuid._obj.bytes[index] = device.value + 1
        return 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("count", "exposed", "accepted"),
    [(1, False, True), (4, False, False), (4, True, True)],
    ids=["isolated", "unexpected-peers", "exposed-peers"],
)
def test_gpu_identity_proves_ordinal_zero_and_accepts_peers_only_when_exposed(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    exposed: bool,
    accepted: bool,
) -> None:
    """The colocation proof must name the actor's own GPU even when its TP peers are visible."""
    driver = _FakeCudaDriver(count)
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: driver)
    monkeypatch.setattr(
        workers_ray.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_accelerator_ids=lambda: {"GPU": ["2"]}, get_node_id=lambda: "node-a"),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1,3" if count > 1 else "2")
    if exposed:
        monkeypatch.setenv(PEER_GPUS_ENV, "1")
    else:
        monkeypatch.delenv(PEER_GPUS_ENV, raising=False)
    host = _HostImpl.__new__(_HostImpl)

    if accepted:
        identity = host.gpu_identity()
        assert identity["cuda_uuid"] == bytes([1] * 16).hex()
        assert identity["ray_accelerator_ids"] == ("2",)
        assert driver.ordinals == [0]
    else:
        with pytest.raises(RuntimeError, match="exactly one CUDA device"):
            host.gpu_identity()
