# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import pytest
import torch

import cosmos_framework.model.generator.reasoner_remote as reasoner_remote_module
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
)
from cosmos_framework.model.generator.reasoner_remote import (
    REASONER_FEATURE_PROTOCOL_VERSION,
    RemoteReasonerFeatureProvider,
    RemoteReasonerRPCError,
    decode_feature_stream,
    decode_tensor_payload,
    encode_generate_request,
    encode_identity,
    encode_signature,
    encode_tensor_payload,
    iter_feature_stream,
)
from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2 as reasoner_pb2

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


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
        head_dim=3,
        dtype=torch.bfloat16,
    )


def _request(sample_key: str, token_ids: Sequence[int]) -> ReasonerFeatureRequest:
    tokens = torch.tensor(token_ids, dtype=torch.int64)
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=tokens,
        position_ids=torch.arange(tokens.numel(), dtype=torch.int64),
        causal_offsets=torch.tensor([0, tokens.numel()], dtype=torch.int64),
        fingerprint=f"{sample_key}-fingerprint",
    )


def _feature_batch(
    requests: Sequence[ReasonerFeatureRequest],
    *,
    signature: ReasonerFeatureSignature | None = None,
) -> ReasonerFeatureBatch:
    signature = signature or _signature()
    total_tokens = sum(request.token_ids.numel() for request in requests)
    shape = (total_tokens, signature.num_kv_heads, signature.head_dim)
    numel = total_tokens * signature.num_kv_heads * signature.head_dim

    def values(offset: int) -> torch.Tensor:
        return (torch.arange(numel, dtype=torch.float32) + offset).reshape(shape).to(signature.dtype)

    offsets = [0]
    for request in requests:
        offsets.append(offsets[-1] + request.token_ids.numel())
    return ReasonerFeatureBatch(
        cross_k=tuple(values(100 * layer_idx) for layer_idx in range(signature.num_layers)),
        cross_v=tuple(values(1000 + 100 * layer_idx) for layer_idx in range(signature.num_layers)),
        causal_offsets=torch.tensor(offsets, dtype=torch.int64),
        fingerprints=tuple(request.fingerprint for request in requests),
    )


def _assert_feature_batches_equal(actual: ReasonerFeatureBatch, expected: ReasonerFeatureBatch) -> None:
    assert actual.fingerprints == expected.fingerprints
    assert torch.equal(actual.causal_offsets, expected.causal_offsets)
    assert len(actual.cross_k) == len(expected.cross_k)
    for actual_k, expected_k, actual_v, expected_v in zip(
        actual.cross_k,
        expected.cross_k,
        actual.cross_v,
        expected.cross_v,
    ):
        assert actual_k.dtype == expected_k.dtype
        assert actual_v.dtype == expected_v.dtype
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)


def _stream(
    requests: Sequence[ReasonerFeatureRequest],
    *,
    max_chunk_bytes: int = 7,
) -> tuple[Any, list[Any], ReasonerFeatureBatch]:
    identity = _identity()
    signature = _signature()
    request = encode_generate_request(requests, identity)
    features = _feature_batch(requests, signature=signature)
    responses = list(
        iter_feature_stream(
            features,
            request_id=request.request_id,
            identity=identity,
            signature=signature,
            service_instance_id="test-replica",
            max_chunk_bytes=max_chunk_bytes,
            queue_ns=11,
            compute_ns=22,
            device_to_host_ns=33,
        )
    )
    return request, responses, features


def _clone_responses(responses: Sequence[Any]) -> list[Any]:
    clones = []
    for response in responses:
        clone = reasoner_pb2.GenerateResponse()
        clone.CopyFrom(response)
        clones.append(clone)
    return clones


def _service_info(
    *,
    identity: ReasonerFeatureIdentity | None = None,
    signature: ReasonerFeatureSignature | None = None,
    max_chunk_bytes: int = 7,
) -> Any:
    return reasoner_pb2.GetInfoResponse(
        protocol_version=REASONER_FEATURE_PROTOCOL_VERSION,
        service_instance_id="fake-replica",
        identity=encode_identity(identity or _identity()),
        signature=encode_signature(signature or _signature()),
        capabilities=(
            reasoner_pb2.Capability(name="server_streaming", version=1),
            reasoner_pb2.Capability(name="strict_fingerprint", version=1),
            reasoner_pb2.Capability(name="ordered_requests", version=1),
            reasoner_pb2.Capability(name="single_document", version=1),
        ),
        max_batch_tokens=64,
        max_queued_tokens=128,
        max_chunk_bytes=max_chunk_bytes,
        ready=True,
        max_requests=4,
    )


def test_tensor_payload_roundtrip_preserves_noncontiguous_bfloat16_bits() -> None:
    source = torch.tensor(
        [[1.0, -2.5, 3.25], [4.5, 5.75, -6.0]],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    source = source.t()
    expected = source.clone()

    decoded = decode_tensor_payload(encode_tensor_payload(source), name="bf16")
    source.zero_()

    assert decoded.dtype == torch.bfloat16
    assert decoded.shape == expected.shape
    assert decoded.is_contiguous()
    assert torch.equal(decoded, expected)


def test_multichunk_feature_stream_roundtrip_preserves_bfloat16_kv_and_timing() -> None:
    requests = (_request("first", [1, 2, 3]), _request("second", [4, 5]))
    request, responses, expected = _stream(requests)
    tensor_chunks = [response.tensor_chunk for response in responses if response.HasField("tensor_chunk")]

    assert len(tensor_chunks) > 2 * _signature().num_layers
    assert all(0 < len(chunk.data) <= 7 for chunk in tensor_chunks)

    actual, timing = decode_feature_stream(
        responses,
        request=request,
        expected_identity=_identity(),
        expected_signature=_signature(),
        max_chunk_bytes=7,
    )

    _assert_feature_batches_equal(actual, expected)
    assert timing.queue_ns == 11
    assert timing.compute_ns == 22
    assert timing.device_to_host_ns == 33
    assert timing.serialization_ns >= 0


def test_stream_serialization_timing_excludes_consumer_backpressure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_clock_ns = 0
    monkeypatch.setattr(reasoner_remote_module.time, "perf_counter_ns", lambda: fake_clock_ns)
    requests = (_request("sample", [1, 2, 3]),)
    request = encode_generate_request(requests, _identity())
    stream = iter_feature_stream(
        _feature_batch(requests),
        request_id=request.request_id,
        identity=_identity(),
        signature=_signature(),
        service_instance_id="timing-test",
        max_chunk_bytes=7,
        queue_ns=1,
        compute_ns=2,
        device_to_host_ns=3,
    )

    responses = []
    for response in stream:
        responses.append(response)
        fake_clock_ns += 1_000_000_000

    assert responses[-1].trailer.timing.serialization_ns == 0


def test_feature_stream_rejects_corrupt_chunk_checksum() -> None:
    request, responses, _ = _stream((_request("sample", [1, 2, 3]),))
    damaged = _clone_responses(responses)
    chunk = next(response.tensor_chunk for response in damaged if response.HasField("tensor_chunk"))
    chunk.data = bytes([chunk.data[0] ^ 0xFF]) + chunk.data[1:]

    with pytest.raises(ValueError, match="checksum mismatch"):
        decode_feature_stream(
            damaged,
            request=request,
            expected_identity=_identity(),
            expected_signature=_signature(),
            max_chunk_bytes=7,
        )


def test_feature_stream_rejects_noncanonical_tensor_order() -> None:
    request, responses, _ = _stream((_request("sample", [1, 2, 3]),))
    damaged = _clone_responses(responses)
    first_chunk = next(response.tensor_chunk for response in damaged if response.HasField("tensor_chunk"))
    first_chunk.kind = reasoner_pb2.TENSOR_KIND_CROSS_V

    with pytest.raises(ValueError, match="tensor order"):
        decode_feature_stream(
            damaged,
            request=request,
            expected_identity=_identity(),
            expected_signature=_signature(),
            max_chunk_bytes=7,
        )


class _RetryingFakeTransport:
    def __init__(self, *, info: Any | None = None) -> None:
        self.info = info or _service_info()
        self.get_info_timeouts: list[float] = []
        self.generate_timeouts: list[float] = []
        self.request_ids: list[str] = []
        self.worker_thread_ids: list[int] = []
        self.second_attempt_entered = threading.Event()
        self.release_second_attempt = threading.Event()
        self.closed = False

    def get_info(self, *, timeout_s: float) -> Any:
        self.get_info_timeouts.append(timeout_s)
        return self.info

    def generate(self, request: Any, *, timeout_s: float) -> list[Any]:
        self.generate_timeouts.append(timeout_s)
        self.request_ids.append(str(request.request_id))
        self.worker_thread_ids.append(threading.get_ident())
        if len(self.request_ids) == 1:
            raise RemoteReasonerRPCError("UNAVAILABLE", "transient test failure", retryable=True)

        self.second_attempt_entered.set()
        if not self.release_second_attempt.wait(timeout=5):
            raise AssertionError("test did not release the fake Reasoner transport")
        requests = tuple(
            _request(str(item.sample_key), range(int(item.token_ids.shape[0]))) for item in request.requests
        )
        features = _feature_batch(requests)
        features = ReasonerFeatureBatch(
            cross_k=features.cross_k,
            cross_v=features.cross_v,
            causal_offsets=features.causal_offsets,
            fingerprints=tuple(str(item.fingerprint) for item in request.requests),
        )
        return list(
            iter_feature_stream(
                features,
                request_id=request.request_id,
                identity=_identity(),
                signature=_signature(),
                service_instance_id="fake-replica",
                max_chunk_bytes=7,
                queue_ns=1,
                compute_ns=2,
                device_to_host_ns=3,
            )
        )

    def close(self) -> None:
        self.closed = True


def test_remote_provider_handshake_async_submit_and_retry() -> None:
    transport = _RetryingFakeTransport()
    provider = RemoteReasonerFeatureProvider(
        "unused.test:1234",
        expected_identity=_identity(),
        expected_dtype=torch.bfloat16,
        connect_timeout_s=1.25,
        request_timeout_s=5.0,
        request_max_retries=1,
        retry_backoff_s=0,
        transport=transport,
    )
    requests = (_request("first", [8, 9, 10]), _request("second", [11, 12]))
    main_thread_id = threading.get_ident()

    try:
        future = provider.submit(requests)
        assert transport.second_attempt_entered.wait(timeout=2)
        assert not future.done()
        transport.release_second_attempt.set()
        actual = future.result(timeout=5)
    finally:
        transport.release_second_attempt.set()
        provider.close()

    _assert_feature_batches_equal(actual, _feature_batch(requests))
    assert transport.get_info_timeouts == [1.25]
    assert len(transport.generate_timeouts) == 2
    assert all(timeout > 0 for timeout in transport.generate_timeouts)
    assert transport.generate_timeouts[1] <= transport.generate_timeouts[0]
    assert len(set(transport.request_ids)) == 1
    assert all(thread_id != main_thread_id for thread_id in transport.worker_thread_ids)
    assert provider.last_server_timing is not None
    assert provider.last_server_timing.compute_ns == 2
    assert transport.closed


def test_remote_provider_rejects_handshake_identity_mismatch_and_closes_transport() -> None:
    wrong_identity = ReasonerFeatureIdentity(reasoner="wrong", tokenizer="tokenizer", framing="framing")
    transport = _RetryingFakeTransport(info=_service_info(identity=wrong_identity))

    with pytest.raises(ValueError, match="identity mismatch"):
        RemoteReasonerFeatureProvider(
            "unused.test:1234",
            expected_identity=_identity(),
            expected_dtype=torch.bfloat16,
            transport=transport,
        )

    assert transport.closed


class _FakeRuntime:
    def __init__(self) -> None:
        self.identity = _identity()
        self.signature = _signature()
        self.max_requests = 4
        self.max_total_tokens = 64
        self.device = torch.device("cpu")

    def execute(self, requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
        return _feature_batch(requests, signature=self.signature)


def test_service_holds_token_admission_until_stream_is_closed() -> None:
    pytest.importorskip("grpc")
    from cosmos_framework.model.generator.reasoner_remote_server import ReasonerFeatureService

    runtime = _FakeRuntime()
    service = ReasonerFeatureService(
        runtime,  # type: ignore[arg-type]
        max_queued_tokens=128,
        max_chunk_bytes=7,
    )
    request = _request("sample", [1, 2, 3])
    stream = service.Generate(encode_generate_request((request,), runtime.identity), context=None)

    first_response = next(stream)
    assert first_response.HasField("header")
    assert service.metrics().queued_tokens == 3

    stream.close()
    metrics = service.metrics()
    assert metrics.queued_tokens == 0
    assert metrics.accepted_batches == 1
    assert metrics.completed_batches == 0


def test_remote_provider_end_to_end_through_in_process_grpc_server() -> None:
    pytest.importorskip("grpc")
    from cosmos_framework.model.generator.reasoner_remote_server import (
        ReasonerFeatureService,
        create_reasoner_grpc_server,
    )

    runtime = _FakeRuntime()
    service = ReasonerFeatureService(
        runtime,  # type: ignore[arg-type]
        max_queued_tokens=128,
        max_chunk_bytes=7,
        service_instance_id="in-process-replica",
    )
    server, port = create_reasoner_grpc_server(service, address="127.0.0.1:0", max_rpc_workers=2)
    server.start()
    provider: RemoteReasonerFeatureProvider | None = None
    requests = (_request("first", [1, 2, 3]), _request("second", [4, 5]))
    try:
        provider = RemoteReasonerFeatureProvider(
            f"127.0.0.1:{port}",
            expected_identity=runtime.identity,
            expected_dtype=runtime.signature.dtype,
            connect_timeout_s=5,
            request_timeout_s=5,
            request_max_retries=0,
        )
        actual = provider.submit(requests).result(timeout=5)
    finally:
        if provider is not None:
            provider.close()
        server.stop(grace=0).wait(timeout=5)

    _assert_feature_batches_equal(actual, _feature_batch(requests))
    metrics = service.metrics()
    assert metrics.accepted_batches == 1
    assert metrics.completed_batches == 1
    assert metrics.rejected_batches == 0
    assert metrics.failed_batches == 0
    assert metrics.queued_tokens == 0
    assert metrics.healthy
