# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest
import torch

import cosmos_framework.model.generator.reasoner_remote as reasoner_remote_module
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
)
from cosmos_framework.model.generator.reasoner_remote import (
    REASONER_FEATURE_PROTOCOL_VERSION,
    GrpcReasonerRemoteTransport,
    RemoteReasonerFeatureProvider,
    RemoteReasonerRPCError,
    encode_identity,
    encode_signature,
)
from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2 as reasoner_pb2

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]

_REQUIRED_CAPABILITIES = (
    "server_streaming",
    "strict_fingerprint",
    "ordered_requests",
    "single_document",
)


def _identity() -> ReasonerFeatureIdentity:
    return ReasonerFeatureIdentity(
        reasoner="reasoner-digest",
        tokenizer="tokenizer-digest",
        framing="framing-digest",
    )


def _signature() -> ReasonerFeatureSignature:
    return ReasonerFeatureSignature(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.bfloat16,
    )


def _service_info(
    *,
    ready: bool = True,
    capabilities: Sequence[str] = _REQUIRED_CAPABILITIES,
    max_requests: int = 4,
) -> Any:
    return reasoner_pb2.GetInfoResponse(
        protocol_version=REASONER_FEATURE_PROTOCOL_VERSION,
        service_instance_id="lifecycle-test-replica",
        identity=encode_identity(_identity()),
        signature=encode_signature(_signature()),
        capabilities=tuple(reasoner_pb2.Capability(name=name, version=1) for name in capabilities),
        max_requests=max_requests,
        max_batch_tokens=64,
        max_queued_tokens=128,
        max_chunk_bytes=32,
        ready=ready,
    )


def _request(
    sample_key: str = "sample",
    *,
    causal_offsets: Sequence[int] = (0, 3),
) -> ReasonerFeatureRequest:
    token_count = causal_offsets[-1]
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=torch.arange(token_count, dtype=torch.int64),
        position_ids=torch.arange(token_count, dtype=torch.int64),
        causal_offsets=torch.tensor(causal_offsets, dtype=torch.int64),
        fingerprint=f"{sample_key}-fingerprint",
    )


class _HandshakeTransport:
    def __init__(
        self,
        *,
        info: Any | None = None,
        get_info_error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.info = info or _service_info()
        self.get_info_error = get_info_error
        self.events = events
        self.close_count = 0
        self.generate_count = 0

    def get_info(self, *, timeout_s: float) -> Any:
        del timeout_s
        if self.events is not None:
            self.events.append("get_info")
        if self.get_info_error is not None:
            raise self.get_info_error
        return self.info

    def generate(self, request: Any, *, timeout_s: float) -> list[Any]:
        del request, timeout_s
        self.generate_count += 1
        raise AssertionError("generate should not be called in this lifecycle test")

    def close(self) -> None:
        self.close_count += 1
        if self.events is not None:
            self.events.append("transport.close")


class _RecordingExecutor:
    instances: list[_RecordingExecutor] = []

    def __init__(self, *args: object, events: list[str], **kwargs: object) -> None:
        del args, kwargs
        self.events = events
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.events.append("executor.init")
        self.instances.append(self)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))
        self.events.append("executor.shutdown")


@pytest.mark.parametrize(
    ("transport", "error_type", "error_match"),
    (
        (_HandshakeTransport(get_info_error=RuntimeError("get-info failed")), RuntimeError, "get-info failed"),
        (_HandshakeTransport(get_info_error=KeyboardInterrupt("stop")), KeyboardInterrupt, "stop"),
        (_HandshakeTransport(info=_service_info(ready=False)), RuntimeError, "not ready"),
        (_HandshakeTransport(info=_service_info(max_requests=0)), ValueError, "request-count limit"),
        (
            _HandshakeTransport(info=_service_info(capabilities=("server_streaming",))),
            ValueError,
            "required capability",
        ),
    ),
)
def test_constructor_failure_closes_transport_then_executor(
    monkeypatch: pytest.MonkeyPatch,
    transport: _HandshakeTransport,
    error_type: type[BaseException],
    error_match: str,
) -> None:
    events: list[str] = []
    transport.events = events
    _RecordingExecutor.instances.clear()
    monkeypatch.setattr(
        reasoner_remote_module,
        "ThreadPoolExecutor",
        lambda *args, **kwargs: _RecordingExecutor(*args, events=events, **kwargs),
    )

    with pytest.raises(error_type, match=error_match):
        RemoteReasonerFeatureProvider(
            "unused.test:1234",
            expected_identity=_identity(),
            expected_dtype=torch.bfloat16,
            transport=transport,
        )

    assert transport.close_count == 1
    assert len(_RecordingExecutor.instances) == 1
    assert _RecordingExecutor.instances[0].shutdown_calls == [(False, True)]
    assert events[-2:] == ["transport.close", "executor.shutdown"]


@pytest.mark.parametrize("missing_capability", _REQUIRED_CAPABILITIES)
def test_handshake_requires_each_v1_capability(missing_capability: str) -> None:
    capabilities = tuple(name for name in _REQUIRED_CAPABILITIES if name != missing_capability)
    transport = _HandshakeTransport(info=_service_info(capabilities=capabilities))

    with pytest.raises(ValueError, match=missing_capability):
        RemoteReasonerFeatureProvider(
            "unused.test:1234",
            expected_identity=_identity(),
            expected_dtype=torch.bfloat16,
            transport=transport,
        )

    assert transport.close_count == 1


class _BlockingTransport(_HandshakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.generate_entered = threading.Event()
        self.channel_closed = threading.Event()

    def generate(self, request: Any, *, timeout_s: float) -> list[Any]:
        del request, timeout_s
        self.generate_count += 1
        self.generate_entered.set()
        if not self.channel_closed.wait(timeout=2):
            raise AssertionError("provider waited for its executor before closing the transport")
        raise RemoteReasonerRPCError("UNAVAILABLE", "test channel closed", retryable=True)

    def close(self) -> None:
        super().close()
        self.channel_closed.set()


def test_close_cancels_inflight_transport_before_waiting_for_executor() -> None:
    transport = _BlockingTransport()
    provider = RemoteReasonerFeatureProvider(
        "unused.test:1234",
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
        request_timeout_s=30,
        request_max_retries=2,
        retry_backoff_s=10,
        transport=transport,
    )
    future = provider.submit((_request(),))
    assert transport.generate_entered.wait(timeout=1)

    started = time.monotonic()
    provider.close()
    elapsed = time.monotonic() - started

    assert elapsed < 1
    with pytest.raises(RemoteReasonerRPCError, match="UNAVAILABLE"):
        future.result(timeout=1)
    provider.close()
    assert transport.close_count == 1
    assert transport.generate_count == 1


class _SlowCloseTransport(_HandshakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.close_entered = threading.Event()
        self.allow_close = threading.Event()

    def close(self) -> None:
        self.close_count += 1
        self.close_entered.set()
        if not self.allow_close.wait(timeout=2):
            raise AssertionError("test did not release transport.close")


def test_concurrent_close_is_idempotent_and_waits_for_the_owner() -> None:
    transport = _SlowCloseTransport()
    provider = RemoteReasonerFeatureProvider(
        "unused.test:1234",
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
        transport=transport,
    )
    errors: list[BaseException] = []

    def close_provider() -> None:
        try:
            provider.close()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=close_provider)
    second = threading.Thread(target=close_provider)
    first.start()
    assert transport.close_entered.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    assert second.is_alive()

    transport.allow_close.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert transport.close_count == 1


class _FakeStatus:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRpcError(Exception):
    def __init__(self, status: _FakeStatus, details: str) -> None:
        super().__init__(details)
        self.status = status
        self.message = details

    def code(self) -> _FakeStatus:
        return self.status

    def details(self) -> str:
        return self.message


class _FakeGrpc:
    RpcError = _FakeRpcError

    class StatusCode:
        UNAVAILABLE = _FakeStatus("UNAVAILABLE")
        RESOURCE_EXHAUSTED = _FakeStatus("RESOURCE_EXHAUSTED")


class _FakeStreamingCall:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.index = 0
        self.cancel_count = 0

    def __iter__(self) -> _FakeStreamingCall:
        return self

    def __next__(self) -> str:
        if self.index == 0:
            self.index += 1
            return "first-response"
        if self.failure is not None:
            raise self.failure
        raise StopIteration

    def cancel(self) -> None:
        self.cancel_count += 1


class _FakeStub:
    def __init__(self, call: _FakeStreamingCall) -> None:
        self.call = call

    def Generate(self, request: Any, *, timeout: float, wait_for_ready: bool) -> _FakeStreamingCall:  # noqa: N802
        del request, timeout, wait_for_ready
        return self.call


class _FakeUnaryStub:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    def GetInfo(self, request: Any, *, timeout: float, wait_for_ready: bool) -> Any:  # noqa: N802
        del request, timeout, wait_for_ready
        raise self.failure


def _grpc_transport(call: _FakeStreamingCall) -> GrpcReasonerRemoteTransport:
    transport = object.__new__(GrpcReasonerRemoteTransport)
    transport._grpc = _FakeGrpc()  # type: ignore[attr-defined]
    transport._stub = _FakeStub(call)  # type: ignore[attr-defined]
    return transport


def _grpc_unary_transport(failure: BaseException) -> GrpcReasonerRemoteTransport:
    transport = object.__new__(GrpcReasonerRemoteTransport)
    transport._grpc = _FakeGrpc()  # type: ignore[attr-defined]
    transport._stub = _FakeUnaryStub(failure)  # type: ignore[attr-defined]
    return transport


def test_grpc_stream_cancelled_when_consumer_closes_generator_early() -> None:
    call = _FakeStreamingCall()
    responses = _grpc_transport(call).generate(object(), timeout_s=1)

    assert next(responses) == "first-response"
    responses.close()

    assert call.cancel_count == 1


@pytest.mark.parametrize("control_flow_error", (KeyboardInterrupt(), SystemExit(3)))
def test_grpc_stream_does_not_convert_control_flow_exceptions(control_flow_error: BaseException) -> None:
    call = _FakeStreamingCall(control_flow_error)
    responses = _grpc_transport(call).generate(object(), timeout_s=1)

    assert next(responses) == "first-response"
    with pytest.raises(type(control_flow_error)):
        next(responses)

    assert call.cancel_count == 1


def test_grpc_stream_only_converts_grpc_rpc_errors() -> None:
    call = _FakeStreamingCall(_FakeRpcError(_FakeGrpc.StatusCode.UNAVAILABLE, "temporarily unavailable"))
    responses = _grpc_transport(call).generate(object(), timeout_s=1)

    assert next(responses) == "first-response"
    with pytest.raises(RemoteReasonerRPCError, match="temporarily unavailable") as error:
        next(responses)

    assert error.value.retryable
    assert call.cancel_count == 1


@pytest.mark.parametrize("control_flow_error", (KeyboardInterrupt(), SystemExit(3)))
def test_grpc_get_info_does_not_convert_control_flow_exceptions(control_flow_error: BaseException) -> None:
    with pytest.raises(type(control_flow_error)):
        _grpc_unary_transport(control_flow_error).get_info(timeout_s=1)


def test_submit_rejects_multi_document_request_before_transport() -> None:
    transport = _HandshakeTransport()
    provider = RemoteReasonerFeatureProvider(
        "unused.test:1234",
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
        transport=transport,
    )
    try:
        with pytest.raises(NotImplementedError, match="one causal document"):
            provider.submit((_request(causal_offsets=(0, 1, 3)),))
    finally:
        provider.close()

    assert transport.generate_count == 0


def test_submit_rejects_request_count_above_advertised_limit_before_transport() -> None:
    transport = _HandshakeTransport()
    provider = RemoteReasonerFeatureProvider(
        "unused.test:1234",
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
        transport=transport,
    )
    try:
        requests = tuple(_request(f"sample-{index}") for index in range(5))
        with pytest.raises(ValueError, match=r"5 samples, service limit is 4"):
            provider.submit(requests)
    finally:
        provider.close()

    assert transport.generate_count == 0
