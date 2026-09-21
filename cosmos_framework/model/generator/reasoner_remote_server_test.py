# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
import torch

pytest.importorskip("google.protobuf")
grpc = pytest.importorskip("grpc")

from cosmos_framework.model.generator.reasoner_features import (  # noqa: E402
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
)
from cosmos_framework.model.generator.reasoner_remote import (  # noqa: E402
    REASONER_FEATURE_PROTOCOL_VERSION,
    encode_generate_request,
)
from cosmos_framework.model.generator.reasoner_remote_server import (  # noqa: E402
    ReasonerFeatureService,
    create_reasoner_grpc_server,
)
from cosmos_framework.model.generator.reasoner_runtime import ReasonerRuntimeInvariantError  # noqa: E402
from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2 as reasoner_pb2  # noqa: E402
from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2_grpc  # noqa: E402

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def _identity() -> ReasonerFeatureIdentity:
    return ReasonerFeatureIdentity(reasoner="reasoner", tokenizer="tokenizer", framing="framing")


def _signature() -> ReasonerFeatureSignature:
    return ReasonerFeatureSignature(num_layers=1, num_kv_heads=1, head_dim=1, dtype=torch.float32)


def _request(sample_key: str, token_ids: Sequence[int] = (1,)) -> ReasonerFeatureRequest:
    tokens = torch.tensor(token_ids, dtype=torch.int64)
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=tokens,
        position_ids=torch.arange(tokens.numel(), dtype=torch.int64),
        causal_offsets=torch.tensor([0, tokens.numel()], dtype=torch.int64),
        fingerprint=f"{sample_key}-fingerprint",
    )


def _features(requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
    offsets = [0]
    for request in requests:
        offsets.append(offsets[-1] + request.token_ids.numel())
    total_tokens = offsets[-1]
    return ReasonerFeatureBatch(
        cross_k=(torch.zeros(total_tokens, 1, 1, dtype=torch.float32),),
        cross_v=(torch.ones(total_tokens, 1, 1, dtype=torch.float32),),
        causal_offsets=torch.tensor(offsets, dtype=torch.int64),
        fingerprints=tuple(request.fingerprint for request in requests),
    )


class _FakeRuntime:
    def __init__(
        self,
        behavior: Callable[[tuple[ReasonerFeatureRequest, ...]], ReasonerFeatureBatch] | None = None,
    ) -> None:
        self.identity = _identity()
        self.signature = _signature()
        self.max_requests = 8
        self.max_total_tokens = 64
        self.device = torch.device("cpu")
        self._behavior = behavior
        self._calls_lock = threading.Lock()
        self._calls: list[str] = []

    @property
    def calls(self) -> list[str]:
        with self._calls_lock:
            return list(self._calls)

    def execute(self, requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
        requests = tuple(requests)
        with self._calls_lock:
            self._calls.append(requests[0].sample_key)
        if self._behavior is not None:
            return self._behavior(requests)
        return _features(requests)


@contextmanager
def _running_service(
    runtime: _FakeRuntime,
    *,
    max_rpc_workers: int = 2,
) -> Iterator[tuple[ReasonerFeatureService, Any]]:
    service = ReasonerFeatureService(
        runtime,  # type: ignore[arg-type]
        max_queued_tokens=128,
        max_chunk_bytes=64,
        service_instance_id="test-replica",
    )
    server, port = create_reasoner_grpc_server(
        service,
        address="127.0.0.1:0",
        max_rpc_workers=max_rpc_workers,
    )
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = reasoner_features_pb2_grpc.ReasonerFeatureServiceStub(channel)
    try:
        yield service, stub
    finally:
        channel.close()
        server.stop(grace=0).wait(timeout=5)


def _consume(call: Any, result: dict[str, Any]) -> None:
    try:
        result["responses"] = list(call)
    except grpc.RpcError as error:
        result["code"] = error.code()
        result["details"] = error.details()


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not satisfied before timeout")


def test_generate_maps_protocol_and_empty_token_validation_statuses() -> None:
    runtime = _FakeRuntime()
    with _running_service(runtime) as (service, stub):
        info = stub.GetInfo(
            reasoner_pb2.GetInfoRequest(protocol_version=REASONER_FEATURE_PROTOCOL_VERSION),
            timeout=2,
        )
        assert info.max_requests == runtime.max_requests

        wrong_version = reasoner_pb2.GenerateRequest(protocol_version=REASONER_FEATURE_PROTOCOL_VERSION + 1)
        with pytest.raises(grpc.RpcError) as protocol_error:
            list(stub.Generate(wrong_version, timeout=2))
        assert protocol_error.value.code() == grpc.StatusCode.FAILED_PRECONDITION

        empty = encode_generate_request((_request("empty", ()),), runtime.identity)
        with pytest.raises(grpc.RpcError) as empty_error:
            list(stub.Generate(empty, timeout=2))
        assert empty_error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        assert runtime.calls == []
        assert service.metrics().rejected_batches == 2
        assert service.metrics().queued_tokens == 0


def test_maximum_concurrent_rpcs_rejects_before_the_executor_queue() -> None:
    entered = threading.Event()
    release = threading.Event()

    def behavior(requests: tuple[ReasonerFeatureRequest, ...]) -> ReasonerFeatureBatch:
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("test did not release the first RPC")
        return _features(requests)

    runtime = _FakeRuntime(behavior)
    with _running_service(runtime, max_rpc_workers=1) as (service, stub):
        first_result: dict[str, Any] = {}
        first_call = stub.Generate(encode_generate_request((_request("first"),), runtime.identity), timeout=5)
        first_thread = threading.Thread(target=_consume, args=(first_call, first_result))
        first_thread.start()
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(grpc.RpcError) as excess_error:
                list(stub.Generate(encode_generate_request((_request("excess"),), runtime.identity), timeout=2))
            assert excess_error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert runtime.calls == ["first"]
            assert service.metrics().accepted_batches == 1
        finally:
            release.set()
            first_thread.join(timeout=5)
        assert not first_thread.is_alive()
        assert "code" not in first_result


@pytest.mark.parametrize(
    ("cancel_explicitly", "expected_code"),
    ((True, grpc.StatusCode.CANCELLED), (False, grpc.StatusCode.DEADLINE_EXCEEDED)),
    ids=("cancel", "deadline"),
)
def test_cancelled_or_expired_queued_rpc_never_executes(
    cancel_explicitly: bool,
    expected_code: Any,
) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()

    def behavior(requests: tuple[ReasonerFeatureRequest, ...]) -> ReasonerFeatureBatch:
        if requests[0].sample_key == "first":
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise AssertionError("test did not release the first RPC")
        return _features(requests)

    runtime = _FakeRuntime(behavior)
    with _running_service(runtime, max_rpc_workers=2) as (service, stub):
        first_result: dict[str, Any] = {}
        first_call = stub.Generate(encode_generate_request((_request("first"),), runtime.identity), timeout=5)
        first_thread = threading.Thread(target=_consume, args=(first_call, first_result))
        first_thread.start()

        second_result: dict[str, Any] = {}
        second_thread: threading.Thread | None = None
        try:
            assert first_entered.wait(timeout=2)
            second_timeout = 5.0 if cancel_explicitly else 1.0
            second_call = stub.Generate(
                encode_generate_request((_request("second"),), runtime.identity),
                timeout=second_timeout,
            )
            second_thread = threading.Thread(target=_consume, args=(second_call, second_result))
            second_thread.start()
            _wait_until(lambda: service.metrics().accepted_batches == 2)
            if cancel_explicitly:
                assert second_call.cancel()
            second_thread.join(timeout=3)
            assert not second_thread.is_alive()
            assert second_result["code"] == expected_code

            # Wait for the server handler, not just the client, to observe the
            # cancellation and relinquish its token reservation while the first
            # request still owns the execution lock.
            _wait_until(lambda: service.metrics().queued_tokens == 1)
            assert runtime.calls == ["first"]
        finally:
            release_first.set()
            first_thread.join(timeout=5)
            if second_thread is not None:
                second_thread.join(timeout=5)

        assert not first_thread.is_alive()
        assert "code" not in first_result
        assert runtime.calls == ["first"]
        _wait_until(lambda: service.metrics().queued_tokens == 0)


def test_oom_marks_unhealthy_before_queued_request_can_execute() -> None:
    first_entered = threading.Event()
    release_oom = threading.Event()

    def behavior(requests: tuple[ReasonerFeatureRequest, ...]) -> ReasonerFeatureBatch:
        if requests[0].sample_key == "oom":
            first_entered.set()
            if not release_oom.wait(timeout=5):
                raise AssertionError("test did not release the OOM RPC")
            raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM")
        return _features(requests)

    runtime = _FakeRuntime(behavior)
    with _running_service(runtime, max_rpc_workers=2) as (service, stub):
        oom_result: dict[str, Any] = {}
        oom_call = stub.Generate(encode_generate_request((_request("oom"),), runtime.identity), timeout=5)
        oom_thread = threading.Thread(target=_consume, args=(oom_call, oom_result))
        oom_thread.start()

        queued_result: dict[str, Any] = {}
        queued_thread: threading.Thread | None = None
        try:
            assert first_entered.wait(timeout=2)
            queued_call = stub.Generate(encode_generate_request((_request("queued"),), runtime.identity), timeout=5)
            queued_thread = threading.Thread(target=_consume, args=(queued_call, queued_result))
            queued_thread.start()
            _wait_until(lambda: service.metrics().accepted_batches == 2)
            release_oom.set()
            oom_thread.join(timeout=3)
            queued_thread.join(timeout=3)
        finally:
            release_oom.set()
            oom_thread.join(timeout=5)
            if queued_thread is not None:
                queued_thread.join(timeout=5)

        assert not oom_thread.is_alive()
        assert queued_thread is not None and not queued_thread.is_alive()
        assert oom_result["code"] == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert queued_result["code"] == grpc.StatusCode.UNAVAILABLE
        assert runtime.calls == ["oom"]
        metrics = service.metrics()
        assert not metrics.healthy
        assert metrics.failed_batches == 2
        assert metrics.queued_tokens == 0


def test_fatal_runtime_invariant_marks_unhealthy_before_queued_request_can_execute() -> None:
    first_entered = threading.Event()
    release_failure = threading.Event()

    def behavior(requests: tuple[ReasonerFeatureRequest, ...]) -> ReasonerFeatureBatch:
        if requests[0].sample_key == "fatal":
            first_entered.set()
            if not release_failure.wait(timeout=5):
                raise AssertionError("test did not release the fatal runtime failure")
            raise ReasonerRuntimeInvariantError("synthetic output signature mismatch")
        return _features(requests)

    runtime = _FakeRuntime(behavior)
    with _running_service(runtime, max_rpc_workers=2) as (service, stub):
        fatal_result: dict[str, Any] = {}
        fatal_call = stub.Generate(encode_generate_request((_request("fatal"),), runtime.identity), timeout=5)
        fatal_thread = threading.Thread(target=_consume, args=(fatal_call, fatal_result))
        fatal_thread.start()

        queued_result: dict[str, Any] = {}
        queued_thread: threading.Thread | None = None
        try:
            assert first_entered.wait(timeout=2)
            queued_call = stub.Generate(encode_generate_request((_request("queued"),), runtime.identity), timeout=5)
            queued_thread = threading.Thread(target=_consume, args=(queued_call, queued_result))
            queued_thread.start()
            _wait_until(lambda: service.metrics().accepted_batches == 2)
            release_failure.set()
            fatal_thread.join(timeout=3)
            queued_thread.join(timeout=3)
        finally:
            release_failure.set()
            fatal_thread.join(timeout=5)
            if queued_thread is not None:
                queued_thread.join(timeout=5)

        assert not fatal_thread.is_alive()
        assert queued_thread is not None and not queued_thread.is_alive()
        assert fatal_result["code"] == grpc.StatusCode.INTERNAL
        assert "marked unhealthy" in fatal_result["details"]
        assert queued_result["code"] == grpc.StatusCode.UNAVAILABLE
        assert runtime.calls == ["fatal"]
        metrics = service.metrics()
        assert not metrics.healthy
        assert metrics.failed_batches == 2
        assert metrics.queued_tokens == 0

        info = stub.GetInfo(
            reasoner_pb2.GetInfoRequest(protocol_version=REASONER_FEATURE_PROTOCOL_VERSION),
            timeout=2,
        )
        assert not info.ready
        with pytest.raises(grpc.RpcError) as subsequent_error:
            list(stub.Generate(encode_generate_request((_request("subsequent"),), runtime.identity), timeout=2))
        assert subsequent_error.value.code() == grpc.StatusCode.UNAVAILABLE
        assert runtime.calls == ["fatal"]


def test_stream_serialization_failure_is_internal_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cosmos_framework.model.generator.reasoner_remote_server as server_module

    def broken_stream(*_args: object, **_kwargs: object) -> Iterator[Any]:
        yield from ()
        raise ValueError("synthetic stream failure")

    monkeypatch.setattr(server_module, "iter_feature_stream", broken_stream)
    runtime = _FakeRuntime()
    with _running_service(runtime) as (service, stub):
        with pytest.raises(grpc.RpcError) as stream_error:
            list(stub.Generate(encode_generate_request((_request("stream"),), runtime.identity), timeout=2))

        assert stream_error.value.code() == grpc.StatusCode.INTERNAL
        assert "synthetic stream failure" in stream_error.value.details()
        metrics = service.metrics()
        assert metrics.accepted_batches == 1
        assert metrics.completed_batches == 0
        assert metrics.failed_batches == 1
        assert metrics.queued_tokens == 0
