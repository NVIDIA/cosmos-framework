# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The vLLM-Omni weight plane on CPU: exported mapping, direct writes, bitwise witness, brackets, failure, recovery.

The engine is a fake that replies the way vLLM-Omni does (``[stage][rank-0 result]``; a worker exception comes
back as the stage pool's ``{"supported": False, "error"}`` VALUE), but the worker-side code is the REAL
``OmniIpcWorkerExtension`` driven through that reply path. CUDA IPC is replaced by a handle that carries the live
parameter itself (the same aliasing the real handle gives), so the body's mapping IS the fake worker's parameters:
the body's ``copy_`` lands in ``omni.head.*`` exactly as a D2D copy into a mapped storage would. The producer side
is the real ``push_stream_to_peer`` over a ``CopyMover`` backend.
"""

from __future__ import annotations

import os
import sys
import types
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.comm.transfer import ipc
from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
from cosmos_framework.posttrain.comm.weight_digest import tensor_digest_metadata
from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, push_stream_to_peer
from cosmos_framework.posttrain.roles.rollout.vllm_omni import OmniPolicyBody
from cosmos_framework.posttrain.roles.rollout.vllm_omni_worker import OmniIpcWorkerExtension

PREFIX = "action_head."
NAMES = {f"{PREFIX}a", f"{PREFIX}b", f"{PREFIX}c"}
#: The `OmniDiffusionConfig` switches the export checks, all in their GR00T (compatible) state.
_OD_CONFIG_OK = {
    "enable_sleep_mode": False,
    "enable_cpu_offload": False,
    "enable_layerwise_offload": False,
    "custom_pipeline_args": None,
}


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(2, 3))
        self.b = nn.Parameter(torch.zeros(4))
        self.c = nn.Parameter(torch.zeros(5))


class _Pipeline:
    """The live pipeline as the extension reaches it: `policy.model.action_head` (the real one is exercised at L1)."""

    def __init__(self, head: nn.Module) -> None:
        self.policy = SimpleNamespace(model=SimpleNamespace(action_head=head))


class _Ext(OmniIpcWorkerExtension):
    _pids = iter(range(1000, 10_000))

    def __init__(self, pipeline: _Pipeline, od_config: SimpleNamespace) -> None:
        self.model_runner = SimpleNamespace(pipeline=pipeline)
        self.pid = next(_Ext._pids)
        self.od_config = od_config

    def _live_named_parameters(self):
        head = self.model_runner.pipeline.policy.model.action_head
        return ((f"{PREFIX}{name}", p) for name, p in head.named_parameters())

    def describe_device(self) -> dict[str, Any]:  # a distinct pid per worker, like a real respawn
        return {**super().describe_device(), "pid": self.pid}


class _Engine:
    """Replies as vLLM-Omni does: ``[stage][rank-0 result]``; a worker exception becomes the stage pool's value.

    ``override[method]`` replaces a reply: a static value, or a callable given the worker's raw result to tamper with.
    """

    def __init__(self, ext: _Ext, override: dict[str, Any]) -> None:
        self.ext = ext
        self.calls: list[tuple[str, float | None, tuple]] = []
        self.replies: list[Any] = []
        self.override: dict[str, Any] = dict(override)

    def collective_rpc(self, method: str, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None):
        self.calls.append((method, timeout, tuple(args)))
        if method in self.override:
            value = self.override[method]
            reply = value(getattr(self.ext, method)(*args, **(kwargs or {}))) if callable(value) else value
        else:
            try:
                reply = [[getattr(self.ext, method)(*args, **(kwargs or {}))]]
            except Exception as exc:  # noqa: BLE001 -- this IS the executor's re-raise, flattened by the stage pool
                reply = [[{"supported": False, "error": f"RPC {method!r} failed on worker rank(s): {exc}"}]]
        self.replies.append(reply)
        return reply

    def method_calls(self, method: str) -> list[tuple]:
        return [args for m, _timeout, args in self.calls if m == method]


class _Omni:
    def __init__(self, rig: _Rig) -> None:
        self.head = _Head()
        self.ext = _Ext(_Pipeline(self.head), SimpleNamespace(**{**_OD_CONFIG_OK, **rig.od_config}))
        self.engine = _Engine(self.ext, rig.engine_override)
        self.order = rig.order
        self.closed = False

    def generate(self, prompts, params):
        return ["out"]

    def close(self) -> None:
        self.closed = True
        self.order.append("engine")


class _Body(OmniPolicyBody):
    def _engine_kwargs(self) -> dict[str, Any]:
        return {"model": "fake"}

    def _build_request(self, obs_list, seed):
        return ["p"], None

    def _parse_output(self, output, n):
        return [{"action": output}] * n


class _Rig:
    def __init__(self) -> None:
        self.created: list[_Omni] = []
        self.exports: list[torch.Tensor] = []
        self.order: list[str] = []
        self.od_config: dict[str, Any] = {}  # applied to the NEXT engine's worker
        self.engine_override: dict[str, Any] = {}  # installed on the NEXT engine

    def omni(self, **kwargs: Any) -> _Omni:
        omni = _Omni(self)
        self.created.append(omni)
        return omni


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    """Fake `vllm_omni.entrypoints.omni.Omni` and fake CUDA IPC (handle = the live parameter itself)."""
    rig = _Rig()
    omni_module = types.ModuleType("vllm_omni.entrypoints.omni")
    omni_module.Omni = rig.omni  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints", types.ModuleType("vllm_omni.entrypoints"))
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.omni", omni_module)

    def _export(tensor: torch.Tensor):
        rig.exports.append(tensor)
        return ("fake-ipc", tensor)

    # Both production modules call these through the `ipc` module, so one patch point serves the worker and the body.
    monkeypatch.setattr(ipc, "tensor_to_ipc", _export)
    monkeypatch.setattr(ipc, "tensor_from_ipc", lambda handle: handle[1])
    return rig


def _wire(name: str, *shape: int, fill: float) -> tuple[str, torch.Tensor]:
    return f"{PREFIX}{name}", torch.full(shape, fill)


def _plan(entries=(("a", (2, 3)), ("b", (4,)), ("c", (5,))), dtype: str = "torch.float32") -> list[list]:
    return [[(f"{PREFIX}{name}", shape, dtype)] for name, shape in entries]


class _FeedingBackend:
    """A consumer-side backend whose `apply_bucket` hands the sink the payload (an owned state dict) directly."""

    def __init__(self, plan: list[list]) -> None:
        self._plan = plan
        self.mover = CopyMover()  # `prepare_weight_version` reads the mover to tell a stream from a shared edge

    def set_bucket_plan(self, plan: list[list]) -> None:
        self._plan = plan

    def plan(self) -> list[list]:
        return self._plan

    def bucket_count(self) -> int:
        return len(self._plan)

    def apply_bucket(self, model: Any, i: int, payload: Any, sink: Any = None) -> None:
        sink(payload)

    def close(self) -> None:
        pass


def _fed_body(rig: _Rig, plan: list[list] | None = None) -> _Body:
    body = _Body(transfer_dtype="float32", verify_weight_values=True)
    body.bind_weight_backend(_FeedingBackend(plan or _plan()))
    return body


def _open_and_apply(body: _Body, buckets: list[dict[str, torch.Tensor]]) -> None:
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    for i, bucket in enumerate(buckets):
        body.apply_bucket(bucket, i)


def _stream(body: _Body, buckets: list[dict[str, torch.Tensor]], version: int) -> None:
    _open_and_apply(body, buckets)
    body.mark_weight_version_ready(version)
    body.prepare_weight_version(version, None)
    body.commit_weight_version(version)


_BUCKETS = [dict([_wire("a", 2, 3, fill=1.0)]), dict([_wire("b", 4, fill=2.0)]), dict([_wire("c", 5, fill=3.0)])]


@pytest.mark.L0
@pytest.mark.CPU
def test_buckets_land_through_the_worker_exported_mapping_and_the_stream_commits(rig: _Rig) -> None:
    body = _fed_body(rig)
    omni = rig.created[-1]
    assert body.worker_device["pid"] == omni.ext.pid  # device verified through the worker before the mapping
    # One export per live parameter, at construction; the body's views ARE the worker's parameters.
    assert len(rig.exports) == 3
    assert [m for m, _t, _a in omni.engine.calls] == ["describe_device", "export_live_parameters"]
    assert body._live_params[f"{PREFIX}a"].data_ptr() == omni.head.a.data_ptr()
    assert body._mapping["pid"] == omni.ext.pid and body._engine_names == NAMES

    _stream(body, _BUCKETS, version=1)

    assert torch.equal(omni.head.a.detach(), torch.full((2, 3), 1.0))
    assert torch.equal(omni.head.b.detach(), torch.full((4,), 2.0))
    assert torch.equal(omni.head.c.detach(), torch.full((5,), 3.0))
    # Two RPCs per publication -- the bracket -- and none per bucket.
    assert [m for m, _t, _a in omni.engine.calls][2:] == ["begin_weight_update", "finish_weight_update"]
    assert omni.engine.method_calls("begin_weight_update") == [(1,)]
    (epoch, expected), *_ = omni.engine.method_calls("finish_weight_update")
    assert epoch == 1 and set(expected) == NAMES, "verification on: the worker got a digest for every parameter"
    assert body.weight_version() == 1 and body._stream_open is False
    assert body.infer_batch([{"o": 1}], weight_version=None) == [{"action": "out"}]


@pytest.mark.L0
@pytest.mark.CPU
def test_inference_is_refused_while_a_stream_is_open_even_without_a_version(rig: _Rig) -> None:
    body = _fed_body(rig)
    body.set_bucket_plan(body._weight_backend.plan())
    with pytest.raises(RuntimeError, match="refuses to serve"):
        body.infer_batch([{"o": 1}], weight_version=None)
    with pytest.raises(RuntimeError, match="refuses to serve"):
        body.infer_batch([{"o": 1}], weight_version=0)
    assert body.infer_batch([], weight_version=None) == []  # an empty batch is not a request


@pytest.mark.L0
@pytest.mark.CPU
def test_a_bucket_failure_invalidates_the_engine_and_rebuild_re_exports(rig: _Rig) -> None:
    body = _fed_body(rig)
    first = rig.created[-1]
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    body.apply_bucket(_BUCKETS[0], 0)
    with pytest.raises(ValueError, match="but the live parameter is"):
        body.apply_bucket(dict([_wire("b", 5, fill=2.0)]), 1)  # the wrong shape is refused before any copy
    assert torch.equal(first.head.a.detach(), torch.full((2, 3), 1.0)) and torch.equal(
        first.head.b.detach(), torch.zeros(4)
    )

    assert body.engine_invalidated is True and body._pending_weight_version is None
    with pytest.raises(RuntimeError, match="refuses to serve"):
        body.infer_batch([{"o": 1}], weight_version=None)
    with pytest.raises(RuntimeError, match="rebuild_engine"):
        body.set_bucket_plan(body._weight_backend.plan())

    exports_before = len(rig.exports)
    body.rebuild_engine()
    second = rig.created[-1]
    assert first.closed and second is not first
    assert len(rig.exports) == exports_before + 3, "a fresh worker exports a fresh mapping"
    assert body._live_params[f"{PREFIX}b"].data_ptr() == second.head.b.data_ptr()
    assert body.worker_device["pid"] == second.ext.pid != first.ext.pid and body._mapping["pid"] == second.ext.pid
    assert body.engine_invalidated is False and body._stream_open is True and body._engine_names == NAMES
    with pytest.raises(RuntimeError, match="refuses to serve"):  # a rebuilt engine still holds no committed version
        body.infer_batch([{"o": 1}], weight_version=None)

    _stream(body, _BUCKETS, version=1)
    assert torch.equal(second.head.b.detach(), torch.full((4,), 2.0))
    assert body.infer_batch([{"o": 1}], weight_version=1) == [{"action": "out"}]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_corrupted_copy_is_caught_by_the_bitwise_compare(rig: _Rig) -> None:
    """The per-bucket witness reads the mapped memory back: a copy that did not land is a failure, not a receipt."""

    class _Stuck(torch.Tensor):
        def copy_(self, src, non_blocking=False):  # a write that never lands
            return self

    body = _fed_body(rig)
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    body._live_params[f"{PREFIX}b"] = torch.zeros(4).as_subclass(_Stuck)
    body.apply_bucket(_BUCKETS[0], 0)
    with pytest.raises(RuntimeError, match="do not hold the bytes just copied.*action_head.b"):
        body.apply_bucket(_BUCKETS[1], 1)
    assert body.engine_invalidated is True


@pytest.mark.L0
@pytest.mark.CPU
def test_the_bitwise_compare_is_part_of_the_verification_switch(rig: _Rig) -> None:
    """`verify_weight_values` is ONE switch over every byte-level check, so with it off the same corrupted copy
    goes unnoticed: an unverified publication is backed by the structure and the bracket, not by the bytes."""

    class _Stuck(torch.Tensor):
        def copy_(self, src, non_blocking=False):
            return self

    body = _fed_body(rig)
    body.verify_weight_values = False
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    body._live_params[f"{PREFIX}b"] = torch.zeros(4).as_subclass(_Stuck)
    body.apply_bucket(_BUCKETS[0], 0)
    body.apply_bucket(_BUCKETS[1], 1)  # the compare that would have caught it never ran
    assert body.engine_invalidated is False
    assert [t.compare_s for t in body.bucket_timings] == [0.0, 0.0], "no compare, no time attributed to one"
    assert [t.digest_s for t in body.bucket_timings] == [0.0, 0.0] and body._expected_digests == {}
    # And with the switch back on, the very same corruption is caught.
    body.verify_weight_values = True
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    with pytest.raises(RuntimeError, match="do not hold the bytes just copied"):
        body.apply_bucket(_BUCKETS[1], 1)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "reply, match",
    [
        ([[{"epoch": 99, "verified": 3}]], "answered epoch 99"),
        ([[{"epoch": 1, "verified": 2}]], "verified 2 parameters, expected 3"),
        ([[None]], "expected a mapping"),
        ([["not a mapping"]], "expected a mapping"),
        ([[{"x": 1}], [{"x": 2}]], "2 results at nesting depth 0"),
        ([[{"supported": False, "error": "boom"}]], "failed in the Omni worker: boom"),
    ],
)
def test_bad_finish_replies_fail_readiness_and_invalidate(rig: _Rig, reply: Any, match: str) -> None:
    body = _fed_body(rig)
    _open_and_apply(body, _BUCKETS)
    rig.created[-1].engine.override["finish_weight_update"] = reply
    with pytest.raises(RuntimeError, match=match):
        body.mark_weight_version_ready(1)
    assert body.engine_invalidated is True and body._pending_weight_version is None


def _with_world_size(n: int):
    return lambda real: [[{**real, "world_size": n}]]


def _with_shape(name: str, shape: list[int]):
    def tamper(real):
        params = dict(real["params"])
        params[name] = {**params[name], "shape": shape}
        return [[{**real, "params": params}]]

    return tamper


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "override, match",
    [
        (_with_world_size(2), "one worker per engine"),
        (_with_shape(f"{PREFIX}a", [3, 2]), "the worker exported \\(3, 2\\)"),
        ([[{"pid": 1, "device": {}, "world_size": 1, "params": {}}]], "expected the parameter handles"),
    ],
)
def test_bad_export_replies_refuse_the_body_and_close_the_engine(rig: _Rig, override: Any, match: str) -> None:
    rig.engine_override["export_live_parameters"] = override
    with pytest.raises(RuntimeError, match=match):
        _fed_body(rig)
    assert rig.created[-1].closed is True, "a body that failed to construct must not leave its worker behind"
    assert rig.order == ["engine"]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_bad_export_on_rebuild_leaves_the_lane_invalidated(rig: _Rig) -> None:
    body = _fed_body(rig)
    body.engine_invalidated = True  # as a failed stream leaves it
    rig.engine_override["export_live_parameters"] = _with_world_size(2)
    with pytest.raises(RuntimeError, match="one worker per engine"):
        body.rebuild_engine()
    assert body.engine_invalidated is True and body._live_params is None
    with pytest.raises(RuntimeError, match="rebuild_engine"):
        body.set_bucket_plan(body._weight_backend.plan())


@pytest.mark.L0
@pytest.mark.CPU
def test_an_export_missing_a_parameter_is_caught_by_plan_coverage(rig: _Rig) -> None:
    def drop_c(real):
        params = {k: v for k, v in real["params"].items() if k != f"{PREFIX}c"}
        return [[{**real, "params": params}]]

    rig.engine_override["export_live_parameters"] = drop_c
    body = _fed_body(rig)
    body.set_bucket_plan(body._weight_backend.plan())
    with pytest.raises(ValueError, match="unexpected=\\['action_head.c'\\]"):
        body.check_plan_coverage()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "field, value",
    [
        ("enable_sleep_mode", True),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("custom_pipeline_args", {"x": 1}),
    ],
)
def test_export_preconditions_refuse_sleep_mode_offload_and_custom_pipelines(rig: _Rig, field: str, value: Any) -> None:
    rig.od_config[field] = value
    with pytest.raises(RuntimeError, match=field):
        _fed_body(rig)
    assert rig.created[-1].closed is True


@pytest.mark.L0
@pytest.mark.CPU
def test_a_second_export_on_the_same_worker_is_refused(rig: _Rig) -> None:
    body = _fed_body(rig)
    with pytest.raises(RuntimeError, match="already exported"):
        body._worker_rpc("export_live_parameters")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "plan, match",
    [
        (_plan((("a", (2, 3)), ("b", (4,)))), "missing=\\['action_head.c'\\]"),
        (_plan((("a", (2, 3)), ("b", (4,)), ("c", (5,)), ("d", (1,)))), "unexpected=\\['action_head.d'\\]"),
        (_plan((("a", (3, 2)), ("b", (4,)), ("c", (5,)))), "shape_or_dtype=\\[.action_head\\.a: plan \\(\\(3, 2\\)"),
        (_plan(dtype="torch.bfloat16"), "shape_or_dtype="),
    ],
)
def test_check_plan_coverage_rejects_before_any_bytes_move(rig: _Rig, plan: list[list], match: str) -> None:
    body = _fed_body(rig, plan)
    omni = rig.created[-1]
    body.set_bucket_plan(plan)
    with pytest.raises(ValueError, match=match):
        body.check_plan_coverage()
    assert omni.engine.method_calls("finish_weight_update") == [] and torch.equal(
        omni.head.a.detach(), torch.zeros(2, 3)
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_begin_supersedes_an_open_epoch_after_a_coverage_failure(rig: _Rig) -> None:
    """A stream that stops after `begin` (here: coverage) leaves the worker's bracket open on a lane that is NOT
    invalidated; the next stream's `begin` must take over, or the recovery sync could never start."""
    body = _fed_body(rig, _plan((("a", (2, 3)), ("b", (4,)))))
    omni = rig.created[-1]
    body.set_bucket_plan(body._weight_backend.plan())
    with pytest.raises(ValueError, match="missing"):
        body.check_plan_coverage()
    assert body.engine_invalidated is False and omni.ext._weight_plane_state()["open_epoch"] == 1

    body._weight_backend.set_bucket_plan(_plan())
    _stream(body, _BUCKETS, version=1)
    assert omni.engine.method_calls("begin_weight_update") == [(1,), (2,)]
    begin_replies = [r for (m, _t, _a), r in zip(omni.engine.calls, omni.engine.replies) if m == "begin_weight_update"]
    assert [r[0][0]["superseded"] for r in begin_replies] == [None, 1]
    assert omni.ext._weight_plane_state() == {
        "exported": omni.ext._weight_plane_state()["exported"],
        "open_epoch": None,
        "last_epoch": 2,
    }
    assert body.weight_version() == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_a_reallocated_worker_parameter_is_refused_at_begin_and_invalidates(rig: _Rig) -> None:
    body = _fed_body(rig)
    omni = rig.created[-1]
    _stream(body, _BUCKETS, version=1)
    omni.head.b = nn.Parameter(torch.zeros(4))  # the worker re-allocated a parameter behind the mapping
    with pytest.raises(RuntimeError, match="re-allocated.*action_head.b"):
        body.set_bucket_plan(body._weight_backend.plan())
    assert body.engine_invalidated is True


@pytest.mark.L0
@pytest.mark.CPU
def test_finish_with_a_stale_epoch_is_refused(rig: _Rig) -> None:
    body = _fed_body(rig)
    _open_and_apply(body, _BUCKETS)
    body._update_epoch += 1  # the body's counter ran ahead of the bracket it opened
    with pytest.raises(RuntimeError, match="the open update is epoch 1"):
        body.mark_weight_version_ready(1)
    assert body.engine_invalidated is True


@pytest.mark.L0
@pytest.mark.CPU
def test_verification_catches_a_tampered_middle_bucket_tensor(rig: _Rig) -> None:
    """A change behind the sink's back after the bucket landed is invisible to the per-bucket witness; the worker's
    independent re-read at `finish` must catch it."""
    body = _fed_body(rig)
    omni = rig.created[-1]
    _open_and_apply(body, _BUCKETS)
    with torch.no_grad():
        omni.head.b.add_(1.0)
    with pytest.raises(RuntimeError, match="do not hold the staged bytes.*action_head.b"):
        body.mark_weight_version_ready(1)
    assert body._pending_weight_version is None
    assert body.engine_invalidated is True, "unproven live bytes must force rebuild_engine() before the next stream"


@pytest.mark.L0
@pytest.mark.CPU
def test_tamper_after_the_last_bucket_is_caught_only_with_verification_on(rig: _Rig) -> None:
    body = _fed_body(rig)
    body.verify_weight_values = False
    _open_and_apply(body, _BUCKETS)
    with torch.no_grad():
        body._live_params[f"{PREFIX}b"].add_(1.0)  # through the mapping, after the witness ran
    body.mark_weight_version_ready(1)  # by design undetected: the off arm trusts the per-bucket witness
    body.prepare_weight_version(1, None)
    body.commit_weight_version(1)
    assert body.weight_version() == 1

    body.verify_weight_values = True
    _open_and_apply(body, _BUCKETS)
    with torch.no_grad():
        body._live_params[f"{PREFIX}b"].add_(1.0)
    with pytest.raises(RuntimeError, match="do not hold the staged bytes.*action_head.b"):
        body.mark_weight_version_ready(2)


@pytest.mark.L0
@pytest.mark.CPU
def test_verification_requires_a_digest_for_every_live_parameter(rig: _Rig) -> None:
    body = _fed_body(rig)
    _open_and_apply(body, _BUCKETS)
    del body._expected_digests[f"{PREFIX}c"]
    with pytest.raises(RuntimeError, match="digests were staged for 2 of 3"):
        body.mark_weight_version_ready(1)
    assert body.engine_invalidated is True


@pytest.mark.L0
@pytest.mark.CPU
def test_expected_digests_are_the_wire_bytes(rig: _Rig) -> None:
    body = _fed_body(rig)
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    body.apply_bucket(_BUCKETS[0], 0)
    assert body._expected_digests == {f"{PREFIX}a": tensor_digest_metadata(torch.full((2, 3), 1.0))}


@pytest.mark.L0
@pytest.mark.CPU
def test_close_drops_the_mapping_before_the_engine_then_the_backend(rig: _Rig) -> None:
    body = _fed_body(rig)
    omni = rig.created[-1]
    seen: dict[str, Any] = {}
    real_close = omni.close

    def closing() -> None:
        seen["mapping_at_engine_close"] = body._live_params
        real_close()

    omni.close = closing  # type: ignore[method-assign]
    body._weight_backend = SimpleNamespace(close=lambda: rig.order.append("backend"))
    body.close()
    assert seen["mapping_at_engine_close"] is None, "the views alias the worker's memory: released before it dies"
    assert rig.order == ["engine", "backend"]


class _Producer:
    """A trainer stand-in whose publication clock is NOT its optimizer clock, like `PPOPolicyTrainer`."""

    def __init__(self) -> None:
        self.model = nn.Module()
        self.model.action_head = _Head()
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                p.fill_(float(i + 1))
        self._iter = 3

    def publication_version(self) -> int:
        return 7


@pytest.mark.L0
@pytest.mark.CPU
def test_the_real_push_stream_stamps_the_publication_version_not_the_optimizer_step(rig: _Rig) -> None:
    body = _Body(transfer_dtype="float32")
    body.bind_weight_backend(WeightSync(CopyMover()))
    producer = _Producer()
    producer_backend = WeightSync(CopyMover(), bucket_bytes=16)  # tiny cap: several buckets for three tensors

    push_stream_to_peer(producer, body, producer_backend)

    assert producer_backend.bucket_count() >= 2
    omni = rig.created[-1]
    for name, p in producer.model.action_head.named_parameters():
        assert torch.equal(getattr(omni.head, name).detach(), p.detach())
    assert body._pending_weight_version == 7 and body.weight_version() == 0
    with pytest.raises(RuntimeError, match="ready for 7"):
        body.prepare_weight_version(producer._iter, None)  # the optimizer step is NOT the published version
    body.prepare_weight_version(7, None)
    body.commit_weight_version(7)
    assert body.weight_version() == 7 and body.infer_batch([{"o": 1}], weight_version=7) == [{"action": "out"}]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_failed_push_publishes_nothing_and_leaves_the_last_committed_version(rig: _Rig) -> None:
    body = _Body(transfer_dtype="float32")
    body.bind_weight_backend(WeightSync(CopyMover()))
    rig.created[-1].engine.override["finish_weight_update"] = [[{"supported": False, "error": "boom"}]]
    producer_backend = WeightSync(CopyMover(), bucket_bytes=16)
    with pytest.raises(RuntimeError, match="failed in the Omni worker"):
        push_stream_to_peer(_Producer(), body, producer_backend)
    assert body._pending_weight_version is None and body.weight_version() == 0 and body.engine_invalidated is True
    with pytest.raises(RuntimeError, match="not prepared"):
        body.commit_weight_version(7)


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_construction_hides_the_actors_rendezvous_variables(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside a worker actor MASTER_PORT names the body's own process group and is the same on every rollout
    replica; vLLM-Omni would take it as its worker's port and W engines would collide (EADDRINUSE). The engine
    must be built without those variables and the actor must get them back, on the first start, on rebuild,
    and when the start fails."""
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "49363")
    seen: list[dict[str, str | None]] = []
    real_omni = sys.modules["vllm_omni.entrypoints.omni"].Omni
    fail_next = {"value": False}

    def recording_omni(**kwargs):
        seen.append({key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT")})
        if fail_next["value"]:
            fail_next["value"] = False
            raise RuntimeError("engine start failed")
        return real_omni(**kwargs)

    sys.modules["vllm_omni.entrypoints.omni"].Omni = recording_omni
    body = _Body(transfer_dtype="float32")
    assert seen == [{"MASTER_ADDR": None, "MASTER_PORT": None}]
    assert (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]) == ("10.0.0.1", "49363")

    body.rebuild_engine()
    assert seen[-1] == {"MASTER_ADDR": None, "MASTER_PORT": None} and len(seen) == 2
    assert os.environ["MASTER_PORT"] == "49363"

    fail_next["value"] = True
    with pytest.raises(RuntimeError, match="engine start failed"):
        body.rebuild_engine()
    assert os.environ["MASTER_PORT"] == "49363", "a failed start must not leave the actor without its rendezvous"


@pytest.mark.L0
@pytest.mark.CPU
def test_stream_records_sink_and_finish_timings(rig: _Rig) -> None:
    """Every bucket leaves a `BucketSinkTiming` (copy, sync, compare, digest); the bracket's close leaves a
    `VerifyTiming` with the worker's own time and its digest share; both reset when the next stream opens."""
    body = _fed_body(rig)
    _stream(body, _BUCKETS, version=1)
    assert len(body.bucket_timings) == 3
    for timing in body.bucket_timings:
        assert timing.copy_s >= 0.0 and timing.sync_s >= 0.0 and timing.compare_s >= 0.0 and timing.digest_s > 0.0
    verify = body.verify_timing
    assert verify is not None and verify.verified == 3
    assert verify.rpc_s >= verify.worker_s >= verify.digest_s >= 0.0
    body._coverage_reset(0)
    assert body.bucket_timings == [] and body.verify_timing is None


@pytest.mark.L0
@pytest.mark.CPU
def test_verification_can_be_toggled_between_streams_without_stale_state(rig: _Rig) -> None:
    """Off: no digests staged, `finish` carries no expectation, zero digest time. On again: the expectation is rebuilt
    from bucket 0 and the worker re-reads once -- nothing from the off stream leaks into the on stream."""
    body = _Body(transfer_dtype="float32")
    body.bind_weight_backend(_FeedingBackend(_plan()))
    omni = rig.created[-1]
    assert body.verify_weight_values is False
    _stream(body, _BUCKETS, version=1)
    assert omni.engine.method_calls("finish_weight_update") == [(1, None)]
    assert body._expected_digests == {} and body.verify_timing.verified is None and body.verify_timing.digest_s is None
    assert [t.compare_s for t in body.bucket_timings] == [0.0, 0.0, 0.0]
    assert [t.digest_s for t in body.bucket_timings] == [0.0, 0.0, 0.0]
    assert body.weight_version() == 1
    assert all(torch.equal(value, body._live_params[name]) for bucket in _BUCKETS for name, value in bucket.items())

    body.verify_weight_values = True
    _stream(body, _BUCKETS, version=2)
    epoch, expected = omni.engine.method_calls("finish_weight_update")[-1]
    assert epoch == 2 and set(expected) == NAMES
    assert body.verify_timing.verified == 3 and all(t.digest_s > 0.0 for t in body.bucket_timings)
    assert body.weight_version() == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_the_real_push_stream_leaves_consumer_and_producer_timing_records(rig: _Rig) -> None:
    body = _Body(transfer_dtype="float32", verify_weight_values=True)
    body.bind_weight_backend(WeightSync(CopyMover()))
    producer_backend = WeightSync(CopyMover(), bucket_bytes=16)
    push_stream_to_peer(_Producer(), body, producer_backend)
    n = producer_backend.bucket_count()
    assert [t.bucket for t in body._weight_backend.apply_timings] == list(range(n))
    assert all(t.fetch_s >= 0.0 and t.sink_s >= 0.0 for t in body._weight_backend.apply_timings)
    assert len(body.bucket_timings) == n and body.verify_timing is not None and body.verify_timing.verified == 3
    timing = producer_backend.last_push_timing
    assert timing is not None and timing.version == 7 and len(timing.deliver_s) == n


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("verify", [False, True])
def test_rebuild_after_a_committed_stream_requires_full_resync(rig: _Rig, verify: bool) -> None:
    kwargs = {"verify_weight_values": True} if verify else {}
    body = _Body(transfer_dtype="float32", **kwargs)
    body.bind_weight_backend(_FeedingBackend(_plan()))
    assert body.verify_weight_values is verify
    _stream(body, _BUCKETS, version=1)
    expected = {name: tensor.detach().clone() for name, tensor in body._live_params.items()}
    backend, epoch = body._weight_backend, body._update_epoch
    old_engine = rig.created[-1]
    original_close = old_engine.close
    observed = []

    def closing() -> None:
        observed.append(
            (body._stream_open, body.engine_invalidated, body._pending_weight_version, body._prepared_weight_version)
        )
        original_close()

    old_engine.close = closing
    body.rebuild_engine()
    assert observed == [(True, True, None, None)]
    assert body._weight_backend is backend and body._update_epoch == epoch and body.weight_version() == 1
    assert body._stream_open and not body.engine_invalidated
    assert any(not torch.equal(expected[name], tensor) for name, tensor in body._live_params.items())
    for version in (None, 1):
        with pytest.raises(RuntimeError, match="refuses to serve"):
            body.infer_batch([{"o": 1}], weight_version=version)
    with pytest.raises(RuntimeError, match="not prepared"):
        body.commit_weight_version(1)
    _stream(body, [{name: value} for name, value in expected.items()], version=1)
    assert all(torch.equal(expected[name], tensor) for name, tensor in body._live_params.items())
    assert body.infer_batch([{"o": 1}], weight_version=1) == [{"action": "out"}]
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["partial_mapping", "after_start"])
def test_startup_or_binding_failure_releases_imported_views_before_worker_close(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    imported: list[weakref.ReferenceType[torch.Tensor]] = []
    alive_at_close: list[int] = []
    original_close = _Omni.close

    def importing(handle: tuple[str, torch.Tensor]) -> torch.Tensor:
        # A separate Tensor object shares the worker storage, as a real imported view does.
        view = handle[1].view_as(handle[1])
        imported.append(weakref.ref(view))
        return view

    def closing(engine: _Omni) -> None:
        alive_at_close.append(sum(ref() is not None for ref in imported))
        original_close(engine)

    def fail_after_start(self: _Body) -> None:
        raise RuntimeError("startup validation rejected")

    monkeypatch.setattr(ipc, "tensor_from_ipc", importing)
    monkeypatch.setattr(_Omni, "close", closing)
    if failure == "partial_mapping":
        rig.engine_override["export_live_parameters"] = _with_shape(f"{PREFIX}b", [2, 2])
        message = "the worker exported"
    else:
        monkeypatch.setattr(_Body, "_after_engine_start", fail_after_start)
        message = "startup validation rejected"
    with pytest.raises(RuntimeError, match=message) as error:
        body = _Body(transfer_dtype="float32")
        body.bind_weight_backend(_FeedingBackend(_plan()))
    assert error.value.__traceback__ is not None  # retaining the exception must not retain IPC views
    assert bool(imported) is (failure == "partial_mapping")
    assert alive_at_close == [0]
    assert all(ref() is None for ref in imported)
    assert rig.created[-1].closed


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["start", "device", "partial_mapping", "after_start"])
def test_failed_rebuild_closes_new_engine_and_can_retry(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    body = _fed_body(rig)
    _stream(body, _BUCKETS, version=1)
    backend = body._weight_backend

    def reject() -> None:
        raise RuntimeError("rebuild validation rejected")

    with monkeypatch.context() as patch:
        if failure == "partial_mapping":
            patch.setitem(rig.engine_override, "export_live_parameters", _with_shape(f"{PREFIX}b", [2, 2]))
            message = "the worker exported"
        else:
            method = {
                "start": "_start_engine",
                "device": "_verify_worker_device",
                "after_start": "_after_engine_start",
            }[failure]
            patch.setattr(body, method, reject)
            message = "rebuild validation rejected"
        with pytest.raises(RuntimeError, match=message):
            body.rebuild_engine()
    assert all(engine.closed for engine in rig.created)
    assert body._omni is None and body.model is None and body._live_params is None
    assert body._stream_open and body.engine_invalidated and body.weight_version() == 1
    assert body._weight_backend is backend
    for version in (None, 1):
        with pytest.raises(RuntimeError, match="refuses to serve"):
            body.infer_batch([{"o": 1}], weight_version=version)
    body.rebuild_engine()
    _stream(body, _BUCKETS, version=1)
    assert body.infer_batch([{"o": 1}], weight_version=1) == [{"action": "out"}]
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_close_failure_still_releases_the_weight_backend(rig: _Rig) -> None:
    body = _fed_body(rig)
    backend_closed = []
    body._weight_backend = SimpleNamespace(close=lambda: backend_closed.append(True))

    def reject() -> None:
        raise RuntimeError("engine close failed")

    rig.created[-1].close = reject
    with pytest.raises(RuntimeError, match="engine close failed"):
        body.close()
    assert backend_closed == [True]
    assert body._live_params is None and body._omni is None and body.model is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("verify", [False, True])
def test_retained_bucket_exception_releases_views_before_worker_close(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, verify: bool
) -> None:
    refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def importing(handle):
        view = handle[1].detach()
        refs.append(weakref.ref(view))
        return view

    monkeypatch.setattr(ipc, "tensor_from_ipc", importing)
    body = _fed_body(rig)
    body.verify_weight_values = verify
    old_refs = tuple(refs)
    closed_with = []
    old_engine = rig.created[-1]
    original_close = old_engine.close

    def closing():
        closed_with.append(sum(ref() is not None for ref in old_refs))
        original_close()

    monkeypatch.setattr(old_engine, "close", closing)
    body.set_bucket_plan(body._weight_backend.plan())
    body.check_plan_coverage()
    body.apply_bucket(_BUCKETS[0], 0)
    with pytest.raises(ValueError, match="shape|live parameter") as error:
        body.apply_bucket({f"{PREFIX}b": torch.zeros(2)}, 1)
    body.rebuild_engine()
    assert error.value.__traceback__ is not None
    assert closed_with == [0]
    assert all(ref() is None for ref in old_refs)
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("offload", [False, True])
def test_inference_only_constructor_does_not_export_weight_parameters(rig: _Rig, offload: bool) -> None:
    rig.od_config["enable_cpu_offload"] = offload
    body = _Body()
    assert body.infer_batch([{"o": 1}]) == [{"action": "out"}]
    assert not rig.created[-1].engine.method_calls("export_live_parameters")
    assert body._live_params is None
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
def test_binding_exports_once_and_duplicate_initialization_reuses_mapping(rig: _Rig) -> None:
    body = _Body()
    assert body._live_params is None
    backend = _FeedingBackend(_plan())
    body.bind_weight_backend(backend)
    mapping = body._mapping
    body.initialize_weight_receiver()
    body.bind_weight_backend(backend)
    assert body._mapping is mapping
    assert len(rig.created[-1].engine.method_calls("export_live_parameters")) == 1
    _stream(body, _BUCKETS, 1)
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
def test_inference_only_rebuild_resumes_without_an_ipc_mapping(rig: _Rig) -> None:
    rig.od_config["enable_cpu_offload"] = True
    body = _Body()
    old = body.worker_device["pid"]
    body.rebuild_engine()
    assert body.worker_device["pid"] != old
    assert body._live_params is None and not body._stream_open
    assert all(not engine.engine.method_calls("export_live_parameters") for engine in rig.created)
    assert body.infer_batch([{"o": 1}], weight_version=0) == [{"action": "out"}]
    body.close()


@pytest.mark.L0
@pytest.mark.CPU
def test_binding_failure_closes_engine_then_backend_and_can_retry(rig: _Rig) -> None:
    class Backend(_FeedingBackend):
        def close(self):
            rig.order.append("backend")

    rig.od_config["enable_cpu_offload"] = True
    body = _Body()
    with pytest.raises(RuntimeError, match="enable_cpu_offload"):
        body.bind_weight_backend(Backend(_plan()))
    assert rig.order == ["engine", "backend"]
    assert body._weight_backend is None and body._omni is None and body.engine_invalidated
    with pytest.raises(RuntimeError, match="refuses to serve"):
        body.infer_batch([{"o": 1}])
    rig.od_config.clear()
    body.rebuild_engine()
    body.bind_weight_backend(Backend(_plan()))
    _stream(body, _BUCKETS, 1)
    assert body.infer_batch([{"o": 1}], weight_version=1) == [{"action": "out"}]
    body.close()
