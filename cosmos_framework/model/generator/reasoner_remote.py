# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""gRPC wire codec and asynchronous client for remote Reasoner K/V features."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

try:
    from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2 as reasoner_pb2
except ModuleNotFoundError as error:  # pragma: no cover - exercised only in minimal installations
    raise ModuleNotFoundError(
        "Remote Reasoner conditioning requires protobuf; install cosmos-framework[reasoner-remote]"
    ) from error

from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
)

REASONER_FEATURE_PROTOCOL_VERSION = 1
DEFAULT_MAX_CHUNK_BYTES = 2 * 1024**2
_MAX_ACCEPTED_CHUNK_BYTES = 4 * 1024**2

_TORCH_TO_PROTO_DTYPE = {
    torch.int32: reasoner_pb2.DTYPE_INT32,
    torch.int64: reasoner_pb2.DTYPE_INT64,
    torch.float16: reasoner_pb2.DTYPE_FLOAT16,
    torch.bfloat16: reasoner_pb2.DTYPE_BFLOAT16,
    torch.float32: reasoner_pb2.DTYPE_FLOAT32,
    torch.float64: reasoner_pb2.DTYPE_FLOAT64,
}
_PROTO_TO_TORCH_DTYPE = {value: key for key, value in _TORCH_TO_PROTO_DTYPE.items()}


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("Reasoner feature protocol currently supports little-endian hosts only")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: Sequence[int]) -> int:
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape):
        raise ValueError(f"Tensor shape contains an invalid dimension: {tuple(shape)!r}")
    return math.prod(shape)


def encode_tensor_payload(tensor: torch.Tensor) -> Any:
    """Encode a small tensor while preserving its exact dtype, shape, and bytes."""

    _require_little_endian()
    if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
        raise TypeError("Tensor payload must be a materialized torch.Tensor")
    value = tensor.detach().to(device="cpu").contiguous()
    try:
        dtype = _TORCH_TO_PROTO_DTYPE[value.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported Reasoner protocol dtype: {value.dtype}") from error
    return reasoner_pb2.TensorPayload(
        dtype=dtype,
        shape=list(value.shape),
        data=value.view(torch.uint8).numpy().tobytes(),
    )


def decode_tensor_payload(payload: Any, *, name: str) -> torch.Tensor:
    """Decode a small tensor payload into owned CPU storage."""

    _require_little_endian()
    try:
        dtype = _PROTO_TO_TORCH_DTYPE[int(payload.dtype)]
    except KeyError as error:
        raise ValueError(f"{name} uses unsupported protocol dtype {payload.dtype!r}") from error
    shape = tuple(int(size) for size in payload.shape)
    expected_bytes = _shape_numel(shape) * _dtype_size(dtype)
    if len(payload.data) != expected_bytes:
        raise ValueError(f"{name} has {len(payload.data)} bytes, expected {expected_bytes} for {shape} {dtype}")
    storage = bytearray(payload.data)
    return torch.frombuffer(storage, dtype=dtype).reshape(shape)


def encode_identity(identity: ReasonerFeatureIdentity) -> Any:
    return reasoner_pb2.CacheIdentity(**asdict(identity))


def decode_identity(value: Any) -> ReasonerFeatureIdentity:
    return ReasonerFeatureIdentity(
        reasoner=str(value.reasoner),
        tokenizer=str(value.tokenizer),
        framing=str(value.framing),
    )


def encode_signature(signature: ReasonerFeatureSignature) -> Any:
    try:
        dtype = _TORCH_TO_PROTO_DTYPE[signature.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported Reasoner feature signature dtype: {signature.dtype}") from error
    return reasoner_pb2.FeatureSignature(
        dtype=dtype,
        num_layers=signature.num_layers,
        num_kv_heads=signature.num_kv_heads,
        head_dim=signature.head_dim,
    )


def decode_signature(value: Any) -> ReasonerFeatureSignature:
    try:
        dtype = _PROTO_TO_TORCH_DTYPE[int(value.dtype)]
    except KeyError as error:
        raise ValueError(f"Unsupported Reasoner feature signature dtype {value.dtype!r}") from error
    return ReasonerFeatureSignature(
        num_layers=int(value.num_layers),
        num_kv_heads=int(value.num_kv_heads),
        head_dim=int(value.head_dim),
        dtype=dtype,
    )


def encode_reasoner_request(request: ReasonerFeatureRequest) -> Any:
    return reasoner_pb2.ReasonerFeatureRequest(
        sample_key=request.sample_key,
        token_ids=encode_tensor_payload(request.token_ids),
        position_ids=encode_tensor_payload(request.position_ids),
        causal_offsets=encode_tensor_payload(request.causal_offsets),
        fingerprint=request.fingerprint,
    )


def decode_reasoner_request(value: Any) -> ReasonerFeatureRequest:
    return ReasonerFeatureRequest(
        sample_key=str(value.sample_key),
        token_ids=decode_tensor_payload(value.token_ids, name="token_ids"),
        position_ids=decode_tensor_payload(value.position_ids, name="position_ids"),
        causal_offsets=decode_tensor_payload(value.causal_offsets, name="causal_offsets"),
        fingerprint=str(value.fingerprint),
    )


def compute_remote_request_id(
    requests: Sequence[ReasonerFeatureRequest],
    identity: ReasonerFeatureIdentity,
) -> str:
    """Return a stable id for idempotent retries of one ordered request batch."""

    payload = {
        "protocol_version": REASONER_FEATURE_PROTOCOL_VERSION,
        "identity": asdict(identity),
        "requests": [{"sample_key": request.sample_key, "fingerprint": request.fingerprint} for request in requests],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_generate_request(
    requests: Sequence[ReasonerFeatureRequest],
    identity: ReasonerFeatureIdentity,
) -> Any:
    requests = tuple(requests)
    if not requests:
        raise ValueError("At least one Reasoner feature request is required")
    return reasoner_pb2.GenerateRequest(
        protocol_version=REASONER_FEATURE_PROTOCOL_VERSION,
        request_id=compute_remote_request_id(requests, identity),
        identity=encode_identity(identity),
        requests=[encode_reasoner_request(request) for request in requests],
    )


def decode_generate_request(value: Any) -> tuple[str, ReasonerFeatureIdentity, tuple[ReasonerFeatureRequest, ...]]:
    if int(value.protocol_version) != REASONER_FEATURE_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported Reasoner protocol version {value.protocol_version}; "
            f"expected {REASONER_FEATURE_PROTOCOL_VERSION}"
        )
    identity = decode_identity(value.identity)
    requests = tuple(decode_reasoner_request(request) for request in value.requests)
    if not requests:
        raise ValueError("At least one Reasoner feature request is required")
    expected_id = compute_remote_request_id(requests, identity)
    if value.request_id != expected_id:
        raise ValueError(f"Reasoner request_id mismatch: request={value.request_id!r}, expected={expected_id!r}")
    return str(value.request_id), identity, requests


@dataclass(frozen=True)
class RemoteReasonerServiceInfo:
    instance_id: str
    identity: ReasonerFeatureIdentity
    signature: ReasonerFeatureSignature
    capabilities: dict[str, int]
    max_requests: int
    max_batch_tokens: int
    max_queued_tokens: int
    max_chunk_bytes: int


def decode_service_info(value: Any) -> RemoteReasonerServiceInfo:
    if int(value.protocol_version) != REASONER_FEATURE_PROTOCOL_VERSION:
        raise ValueError(
            f"Remote Reasoner protocol mismatch: service={value.protocol_version}, "
            f"client={REASONER_FEATURE_PROTOCOL_VERSION}"
        )
    if not value.ready:
        raise RuntimeError("Remote Reasoner service is not ready")
    if not value.service_instance_id:
        raise ValueError("Remote Reasoner service returned an empty instance id")
    capabilities = {str(item.name): int(item.version) for item in value.capabilities}
    if any(not name or version <= 0 for name, version in capabilities.items()):
        raise ValueError(f"Remote Reasoner service returned invalid capabilities: {capabilities!r}")
    max_requests = int(value.max_requests)
    max_batch_tokens = int(value.max_batch_tokens)
    max_queued_tokens = int(value.max_queued_tokens)
    max_chunk_bytes = int(value.max_chunk_bytes)
    if max_requests <= 0:
        raise ValueError("Remote Reasoner service returned an invalid request-count limit")
    if max_batch_tokens <= 0 or max_queued_tokens < max_batch_tokens:
        raise ValueError("Remote Reasoner service returned invalid token admission limits")
    if max_chunk_bytes <= 0 or max_chunk_bytes > _MAX_ACCEPTED_CHUNK_BYTES:
        raise ValueError(
            f"Remote Reasoner max_chunk_bytes={max_chunk_bytes} is outside (0, {_MAX_ACCEPTED_CHUNK_BYTES}]"
        )
    return RemoteReasonerServiceInfo(
        instance_id=str(value.service_instance_id),
        identity=decode_identity(value.identity),
        signature=decode_signature(value.signature),
        capabilities=capabilities,
        max_requests=max_requests,
        max_batch_tokens=max_batch_tokens,
        max_queued_tokens=max_queued_tokens,
        max_chunk_bytes=max_chunk_bytes,
    )


@dataclass(frozen=True)
class RemoteReasonerServerTiming:
    queue_ns: int
    compute_ns: int
    device_to_host_ns: int
    serialization_ns: int


class _TensorAssembler:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        total_bytes: int,
    ) -> None:
        expected = _shape_numel(shape) * _dtype_size(dtype)
        if total_bytes != expected:
            raise ValueError(f"Stream tensor declares {total_bytes} bytes, expected {expected} for {shape} {dtype}")
        self.dtype = dtype
        self.shape = shape
        self.total_bytes = total_bytes
        self.storage = bytearray(total_bytes)
        self.next_offset = 0

    @property
    def complete(self) -> bool:
        return self.next_offset == self.total_bytes

    def append(self, *, offset: int, data: bytes) -> None:
        if not data:
            raise ValueError("Remote Reasoner stream contains an empty tensor chunk")
        if offset != self.next_offset:
            raise ValueError(f"Remote Reasoner tensor chunk offset={offset}, expected {self.next_offset}")
        end = offset + len(data)
        if end > self.total_bytes:
            raise ValueError(f"Remote Reasoner tensor chunk ends at {end}, beyond {self.total_bytes}")
        self.storage[offset:end] = data
        self.next_offset = end

    def tensor(self) -> torch.Tensor:
        if not self.complete:
            raise ValueError(f"Remote Reasoner tensor ended at {self.next_offset}, expected {self.total_bytes}")
        return torch.frombuffer(self.storage, dtype=self.dtype).reshape(self.shape)


def decode_feature_stream(
    responses: Iterable[Any],
    *,
    request: Any,
    expected_identity: ReasonerFeatureIdentity,
    expected_signature: ReasonerFeatureSignature,
    max_chunk_bytes: int,
) -> tuple[ReasonerFeatureBatch, RemoteReasonerServerTiming]:
    """Validate and assemble one streamed response entirely on CPU."""

    header: Any | None = None
    trailer: Any | None = None
    offsets: torch.Tensor | None = None
    assemblers: dict[tuple[int, int], _TensorAssembler] = {}
    expected_keys = [
        (layer_idx, kind)
        for layer_idx in range(expected_signature.num_layers)
        for kind in (reasoner_pb2.TENSOR_KIND_CROSS_K, reasoner_pb2.TENSOR_KIND_CROSS_V)
    ]
    tensor_cursor = 0
    next_chunk_index = 0
    received_tensor_bytes = 0
    response_digest = hashlib.sha256()

    for response in responses:
        body = response.WhichOneof("body")
        if body == "header":
            if header is not None or next_chunk_index or trailer is not None:
                raise ValueError("Remote Reasoner stream header must be the first and only header")
            header = response.header
            if int(header.protocol_version) != REASONER_FEATURE_PROTOCOL_VERSION:
                raise ValueError(f"Remote Reasoner response protocol mismatch: {header.protocol_version}")
            if header.request_id != request.request_id:
                raise ValueError(
                    f"Remote Reasoner response request id mismatch: {header.request_id!r} != {request.request_id!r}"
                )
            if decode_identity(header.identity) != expected_identity:
                raise ValueError("Remote Reasoner response identity changed after handshake")
            if decode_signature(header.signature) != expected_signature:
                raise ValueError("Remote Reasoner response signature changed after handshake")
            if not header.service_instance_id:
                raise ValueError("Remote Reasoner response contains an empty service instance id")
            expected_fingerprints = tuple(item.fingerprint for item in request.requests)
            if tuple(header.fingerprints) != expected_fingerprints:
                raise ValueError("Remote Reasoner response fingerprints do not preserve request order")
            offsets = decode_tensor_payload(header.causal_offsets, name="response causal_offsets")
            if offsets.dtype != torch.int64 or offsets.ndim != 1:
                raise ValueError("Remote Reasoner response causal_offsets must be a one-dimensional int64 tensor")
            expected_offsets = [0]
            for item in request.requests:
                expected_offsets.append(expected_offsets[-1] + int(item.token_ids.shape[0]))
            if offsets.tolist() != expected_offsets:
                raise ValueError(f"Remote Reasoner response offsets={offsets.tolist()}, expected={expected_offsets}")
            expected_total_bytes = (
                expected_offsets[-1]
                * expected_signature.num_layers
                * 2
                * expected_signature.num_kv_heads
                * expected_signature.head_dim
                * _dtype_size(expected_signature.dtype)
            )
            if int(header.total_tensor_bytes) != expected_total_bytes:
                raise ValueError(
                    f"Remote Reasoner header bytes={header.total_tensor_bytes}, expected={expected_total_bytes}"
                )
            continue

        if body == "tensor_chunk":
            if header is None or trailer is not None:
                raise ValueError("Remote Reasoner tensor chunk appeared outside the header/trailer envelope")
            chunk = response.tensor_chunk
            if int(chunk.chunk_index) != next_chunk_index:
                raise ValueError(f"Remote Reasoner chunk index={chunk.chunk_index}, expected={next_chunk_index}")
            if not chunk.data or len(chunk.data) > max_chunk_bytes:
                raise ValueError(f"Remote Reasoner chunk size={len(chunk.data)} is outside (0, {max_chunk_bytes}]")
            if hashlib.sha256(chunk.data).digest() != chunk.sha256:
                raise ValueError(f"Remote Reasoner chunk {chunk.chunk_index} checksum mismatch")
            key = (int(chunk.layer_index), int(chunk.kind))
            if tensor_cursor >= len(expected_keys) or key != expected_keys[tensor_cursor]:
                expected_key = None if tensor_cursor >= len(expected_keys) else expected_keys[tensor_cursor]
                raise ValueError(f"Remote Reasoner tensor order={key}, expected={expected_key}")
            try:
                dtype = _PROTO_TO_TORCH_DTYPE[int(chunk.dtype)]
            except KeyError as error:
                raise ValueError(f"Remote Reasoner chunk uses unsupported dtype {chunk.dtype!r}") from error
            if dtype != expected_signature.dtype:
                raise ValueError(f"Remote Reasoner chunk dtype={dtype}, expected={expected_signature.dtype}")
            shape = tuple(int(size) for size in chunk.shape)
            expected_shape = (
                int(offsets[-1]),
                expected_signature.num_kv_heads,
                expected_signature.head_dim,
            )
            if shape != expected_shape:
                raise ValueError(f"Remote Reasoner chunk shape={shape}, expected={expected_shape}")
            assembler = assemblers.get(key)
            if assembler is None:
                assembler = _TensorAssembler(dtype=dtype, shape=shape, total_bytes=int(chunk.total_bytes))
                assemblers[key] = assembler
            elif (
                assembler.dtype != dtype or assembler.shape != shape or assembler.total_bytes != int(chunk.total_bytes)
            ):
                raise ValueError(f"Remote Reasoner metadata changed between chunks for tensor {key}")
            assembler.append(offset=int(chunk.byte_offset), data=chunk.data)
            response_digest.update(chunk.data)
            received_tensor_bytes += len(chunk.data)
            next_chunk_index += 1
            if assembler.complete:
                tensor_cursor += 1
            continue

        if body == "trailer":
            if header is None or trailer is not None:
                raise ValueError("Remote Reasoner stream contains a misplaced or duplicate trailer")
            trailer = response.trailer
            continue

        raise ValueError("Remote Reasoner stream contains an empty response envelope")

    if header is None or trailer is None or offsets is None:
        raise ValueError("Remote Reasoner stream ended without a complete header/trailer envelope")
    if tensor_cursor != len(expected_keys) or any(not item.complete for item in assemblers.values()):
        raise ValueError("Remote Reasoner stream ended with incomplete K/V tensors")
    if int(trailer.total_chunks) != next_chunk_index:
        raise ValueError(f"Remote Reasoner trailer chunks={trailer.total_chunks}, received={next_chunk_index}")
    if int(trailer.total_tensor_bytes) != received_tensor_bytes:
        raise ValueError(
            f"Remote Reasoner trailer bytes={trailer.total_tensor_bytes}, received={received_tensor_bytes}"
        )
    if int(header.total_tensor_bytes) != received_tensor_bytes:
        raise ValueError(f"Remote Reasoner header bytes={header.total_tensor_bytes}, received={received_tensor_bytes}")
    if trailer.response_sha256 != response_digest.digest():
        raise ValueError("Remote Reasoner full-response checksum mismatch")

    cross_k = tuple(
        assemblers[(layer_idx, reasoner_pb2.TENSOR_KIND_CROSS_K)].tensor()
        for layer_idx in range(expected_signature.num_layers)
    )
    cross_v = tuple(
        assemblers[(layer_idx, reasoner_pb2.TENSOR_KIND_CROSS_V)].tensor()
        for layer_idx in range(expected_signature.num_layers)
    )
    features = ReasonerFeatureBatch(
        cross_k=cross_k,
        cross_v=cross_v,
        causal_offsets=offsets,
        fingerprints=tuple(header.fingerprints),
    )
    timing = RemoteReasonerServerTiming(
        queue_ns=int(trailer.timing.queue_ns),
        compute_ns=int(trailer.timing.compute_ns),
        device_to_host_ns=int(trailer.timing.device_to_host_ns),
        serialization_ns=int(trailer.timing.serialization_ns),
    )
    return features, timing


def iter_feature_stream(
    features: ReasonerFeatureBatch,
    *,
    request_id: str,
    identity: ReasonerFeatureIdentity,
    signature: ReasonerFeatureSignature,
    service_instance_id: str,
    max_chunk_bytes: int,
    queue_ns: int,
    compute_ns: int,
    device_to_host_ns: int,
) -> Iterator[Any]:
    """Serialize a CPU feature batch into the canonical response stream."""

    if max_chunk_bytes <= 0 or max_chunk_bytes > _MAX_ACCEPTED_CHUNK_BYTES:
        raise ValueError(f"max_chunk_bytes must be in (0, {_MAX_ACCEPTED_CHUNK_BYTES}]")
    layer = features.layer(0)
    actual_signature = ReasonerFeatureSignature(
        num_layers=features.num_layers,
        num_kv_heads=layer.num_kv_heads,
        head_dim=layer.head_dim,
        dtype=layer.cross_k.dtype,
    )
    if actual_signature != signature:
        raise ValueError(f"Feature stream signature={actual_signature!r}, expected={signature!r}")
    serialization_ns = 0
    serialization_started = time.perf_counter_ns()
    tensors = [tensor for pair in zip(features.cross_k, features.cross_v) for tensor in pair]
    cpu_tensors = [tensor.detach().to(device="cpu").contiguous() for tensor in tensors]
    total_tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in cpu_tensors)
    header = reasoner_pb2.GenerateResponse(
        header=reasoner_pb2.GenerateHeader(
            protocol_version=REASONER_FEATURE_PROTOCOL_VERSION,
            request_id=request_id,
            identity=encode_identity(identity),
            signature=encode_signature(signature),
            fingerprints=list(features.fingerprints),
            causal_offsets=encode_tensor_payload(features.causal_offsets.to(dtype=torch.int64)),
            total_tensor_bytes=total_tensor_bytes,
            service_instance_id=service_instance_id,
        )
    )
    serialization_ns += time.perf_counter_ns() - serialization_started
    yield header

    response_digest = hashlib.sha256()
    chunk_index = 0
    for layer_idx in range(signature.num_layers):
        for kind, tensor in (
            (reasoner_pb2.TENSOR_KIND_CROSS_K, cpu_tensors[2 * layer_idx]),
            (reasoner_pb2.TENSOR_KIND_CROSS_V, cpu_tensors[2 * layer_idx + 1]),
        ):
            raw = tensor.view(torch.uint8).reshape(-1).numpy()
            total_bytes = raw.size
            for offset in range(0, total_bytes, max_chunk_bytes):
                serialization_started = time.perf_counter_ns()
                data = raw[offset : offset + max_chunk_bytes].tobytes()
                response_digest.update(data)
                response = reasoner_pb2.GenerateResponse(
                    tensor_chunk=reasoner_pb2.TensorChunk(
                        chunk_index=chunk_index,
                        kind=kind,
                        layer_index=layer_idx,
                        dtype=_TORCH_TO_PROTO_DTYPE[tensor.dtype],
                        shape=list(tensor.shape),
                        byte_offset=offset,
                        total_bytes=total_bytes,
                        data=data,
                        sha256=hashlib.sha256(data).digest(),
                    )
                )
                chunk_index += 1
                serialization_ns += time.perf_counter_ns() - serialization_started
                yield response
    yield reasoner_pb2.GenerateResponse(
        trailer=reasoner_pb2.GenerateTrailer(
            total_chunks=chunk_index,
            total_tensor_bytes=total_tensor_bytes,
            response_sha256=response_digest.digest(),
            timing=reasoner_pb2.ServerTiming(
                queue_ns=queue_ns,
                compute_ns=compute_ns,
                device_to_host_ns=device_to_host_ns,
                serialization_ns=serialization_ns,
            ),
            cache_hits=0,
            cache_misses=len(features.fingerprints),
        )
    )


class RemoteReasonerRPCError(RuntimeError):
    def __init__(self, code: str, details: str, *, retryable: bool) -> None:
        super().__init__(f"Remote Reasoner RPC {code}: {details}")
        self.code = code
        self.details = details
        self.retryable = retryable


class ReasonerRemoteTransport(Protocol):
    def get_info(self, *, timeout_s: float) -> Any: ...

    def generate(self, request: Any, *, timeout_s: float) -> Iterable[Any]: ...

    def close(self) -> None: ...


class GrpcReasonerRemoteTransport:
    """Thin lazy-imported gRPC transport; tensor validation stays transport-neutral."""

    def __init__(self, endpoint: str) -> None:
        if not endpoint or ":" not in endpoint:
            raise ValueError(f"Remote Reasoner endpoint must be host:port, got {endpoint!r}")
        try:
            import grpc

            from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2_grpc
        except ModuleNotFoundError as error:  # pragma: no cover - exercised only in minimal installations
            raise ModuleNotFoundError(
                "Remote Reasoner conditioning requires grpcio; install cosmos-framework[reasoner-remote]"
            ) from error
        self._grpc = grpc
        self._channel = grpc.insecure_channel(
            endpoint,
            options=(
                ("grpc.max_send_message_length", 16 * 1024**2),
                ("grpc.max_receive_message_length", 8 * 1024**2),
            ),
        )
        self._stub = reasoner_features_pb2_grpc.ReasonerFeatureServiceStub(self._channel)

    def _convert_error(self, error: Any) -> RemoteReasonerRPCError:
        code = error.code()
        return RemoteReasonerRPCError(
            code.name,
            error.details() or str(error),
            retryable=code in {self._grpc.StatusCode.UNAVAILABLE, self._grpc.StatusCode.RESOURCE_EXHAUSTED},
        )

    def get_info(self, *, timeout_s: float) -> Any:
        try:
            return self._stub.GetInfo(
                reasoner_pb2.GetInfoRequest(protocol_version=REASONER_FEATURE_PROTOCOL_VERSION),
                timeout=timeout_s,
                wait_for_ready=True,
            )
        except self._grpc.RpcError as error:
            raise self._convert_error(error) from error

    def generate(self, request: Any, *, timeout_s: float) -> Iterator[Any]:
        call: Any | None = None
        completed = False
        try:
            call = self._stub.Generate(request, timeout=timeout_s, wait_for_ready=True)
            yield from call
            completed = True
        except self._grpc.RpcError as error:
            raise self._convert_error(error) from error
        finally:
            # A decoder may fail closed before consuming the full response.  In
            # that case explicitly cancel the server-streaming RPC rather than
            # leaving its server handler and channel resources alive.  Do not
            # catch BaseException here: GeneratorExit, KeyboardInterrupt, and
            # SystemExit must retain their normal control-flow semantics.
            if call is not None and not completed:
                call.cancel()

    def close(self) -> None:
        self._channel.close()


def _request_snapshot(request: ReasonerFeatureRequest) -> ReasonerFeatureRequest:
    return ReasonerFeatureRequest(
        sample_key=request.sample_key,
        token_ids=request.token_ids.detach().to(device="cpu").contiguous().clone(),
        position_ids=request.position_ids.detach().to(device="cpu").contiguous().clone(),
        causal_offsets=request.causal_offsets.detach().to(device="cpu").contiguous().clone(),
        fingerprint=request.fingerprint,
    )


class RemoteReasonerFeatureProvider:
    """Asynchronously fetch canonical K/V from a fail-closed Reasoner service."""

    def __init__(
        self,
        endpoint: str,
        *,
        expected_identity: ReasonerFeatureIdentity,
        expected_dtype: torch.dtype,
        connect_timeout_s: float = 30.0,
        request_timeout_s: float = 300.0,
        request_max_retries: int = 2,
        retry_backoff_s: float = 0.25,
        transport: ReasonerRemoteTransport | None = None,
    ) -> None:
        # Initialize teardown state before creating either resource.  Besides
        # normal shutdown, this lets every constructor failure use the exact
        # same idempotent cleanup path.
        self._state_lock = threading.Lock()
        self._close_complete = threading.Event()
        self._close_requested = threading.Event()
        self._closed = False
        self._transport: ReasonerRemoteTransport | None = transport
        self._executor: ThreadPoolExecutor | None = None

        self.endpoint = endpoint
        self.expected_identity = expected_identity
        self.last_server_timing: RemoteReasonerServerTiming | None = None

        try:
            connect_timeout_s = float(connect_timeout_s)
            self.request_timeout_s = float(request_timeout_s)
            self.request_max_retries = int(request_max_retries)
            self.retry_backoff_s = float(retry_backoff_s)
            if connect_timeout_s <= 0 or self.request_timeout_s <= 0:
                raise ValueError("Remote Reasoner connect/request timeouts must be positive")
            if self.request_max_retries < 0 or self.retry_backoff_s < 0:
                raise ValueError("Remote Reasoner retry count/backoff must be non-negative")

            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reasoner-remote")
            if self._transport is None:
                self._transport = GrpcReasonerRemoteTransport(endpoint)
            info = decode_service_info(self._transport.get_info(timeout_s=connect_timeout_s))
            if info.identity != expected_identity:
                raise ValueError(
                    f"Remote Reasoner identity mismatch: service={info.identity!r}, expected={expected_identity!r}"
                )
            if info.signature.dtype != expected_dtype:
                raise ValueError(
                    f"Remote Reasoner dtype mismatch: service={info.signature.dtype}, expected={expected_dtype}"
                )
            required_capabilities = (
                "server_streaming",
                "strict_fingerprint",
                "ordered_requests",
                "single_document",
            )
            missing_capabilities = [
                capability for capability in required_capabilities if info.capabilities.get(capability, 0) < 1
            ]
            if missing_capabilities:
                raise ValueError(
                    f"Remote Reasoner service does not advertise required capability v1: {missing_capabilities!r}"
                )
            self.info = info
            self.signature = info.signature
        except BaseException:
            self.close(wait=False)
            raise

    def submit(self, requests: Sequence[ReasonerFeatureRequest]) -> Future[ReasonerFeatureBatch]:
        snapshots = tuple(_request_snapshot(request) for request in requests)
        if not snapshots:
            raise ValueError("At least one Reasoner feature request is required")
        multi_document = [request.sample_key for request in snapshots if request.causal_offsets.numel() != 2]
        if multi_document:
            raise NotImplementedError(
                f"Remote Reasoner service v1 supports one causal document per request; got {multi_document!r}"
            )
        if len(snapshots) > self.info.max_requests:
            raise ValueError(
                f"Reasoner request has {len(snapshots)} samples, service limit is {self.info.max_requests}"
            )
        total_tokens = sum(request.token_ids.numel() for request in snapshots)
        if total_tokens > self.info.max_batch_tokens:
            raise ValueError(
                f"Reasoner request has {total_tokens} tokens, service limit is {self.info.max_batch_tokens}"
            )
        with self._state_lock:
            if self._closed:
                raise RuntimeError("RemoteReasonerFeatureProvider is closed")
            submitted_at = time.monotonic()
            if self._executor is None:  # pragma: no cover - guarded by successful construction
                raise RuntimeError("RemoteReasonerFeatureProvider executor is unavailable")
            return self._executor.submit(self._fetch, snapshots, submitted_at)

    def _fetch(
        self,
        requests: tuple[ReasonerFeatureRequest, ...],
        submitted_at: float,
    ) -> ReasonerFeatureBatch:
        request = encode_generate_request(requests, self.expected_identity)
        deadline = submitted_at + self.request_timeout_s
        last_error: RemoteReasonerRPCError | None = None
        for attempt in range(self.request_max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if last_error is not None:
                    raise TimeoutError("Remote Reasoner request exhausted its absolute deadline") from last_error
                raise TimeoutError("Remote Reasoner request expired before transport execution")
            try:
                responses = self._transport.generate(request, timeout_s=remaining)
                try:
                    features, timing = decode_feature_stream(
                        responses,
                        request=request,
                        expected_identity=self.expected_identity,
                        expected_signature=self.signature,
                        max_chunk_bytes=self.info.max_chunk_bytes,
                    )
                finally:
                    close_responses = getattr(responses, "close", None)
                    if close_responses is not None:
                        close_responses()
                self.last_server_timing = timing
                return features
            except RemoteReasonerRPCError as error:
                last_error = error
                if self._close_requested.is_set() or not error.retryable or attempt >= self.request_max_retries:
                    raise
                sleep_seconds = self.retry_backoff_s * (2**attempt)
                remaining = deadline - time.monotonic()
                if sleep_seconds >= remaining:
                    raise TimeoutError("Remote Reasoner retry backoff exceeds the absolute deadline") from error
                if self._close_requested.wait(timeout=sleep_seconds):
                    raise
        raise AssertionError("unreachable")

    def close(self, *, wait: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                close_complete = self._close_complete
                owns_close = False
            else:
                self._closed = True
                self._close_requested.set()
                close_complete = self._close_complete
                transport = self._transport
                executor = self._executor
                owns_close = True

        if not owns_close:
            if wait:
                close_complete.wait()
            return

        # Closing the channel first cancels an in-flight streaming RPC, allowing
        # its worker to leave promptly before executor shutdown waits for it.
        try:
            if transport is not None:
                transport.close()
        finally:
            try:
                if executor is not None:
                    executor.shutdown(wait=wait, cancel_futures=True)
            finally:
                close_complete.set()

    def __enter__(self) -> RemoteReasonerFeatureProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process-exit cleanup
        try:
            self.close(wait=False)
        except BaseException:
            pass
