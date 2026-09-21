# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded one-replica gRPC service for frozen Reasoner K/V extraction."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

try:
    import grpc

    from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2 as reasoner_pb2
    from cosmos_framework.protos.reasoner_features.v1 import reasoner_features_pb2_grpc
except ModuleNotFoundError as error:  # pragma: no cover - exercised only in minimal installations
    raise ModuleNotFoundError(
        "The Reasoner feature service requires grpcio and protobuf; install cosmos-framework[reasoner-remote]"
    ) from error

from cosmos_framework.model.generator.reasoner_features import ReasonerFeatureBatch
from cosmos_framework.model.generator.reasoner_remote import (
    DEFAULT_MAX_CHUNK_BYTES,
    REASONER_FEATURE_PROTOCOL_VERSION,
    decode_generate_request,
    encode_identity,
    encode_signature,
    iter_feature_stream,
)
from cosmos_framework.model.generator.reasoner_runtime import ReasonerFeatureRuntime, ReasonerRuntimeInvariantError

_EXECUTION_LOCK_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class ReasonerServiceMetrics:
    accepted_batches: int
    rejected_batches: int
    failed_batches: int
    completed_batches: int
    accepted_tokens: int
    queued_tokens: int
    healthy: bool


class _TokenAdmission:
    def __init__(self, max_queued_tokens: int) -> None:
        if max_queued_tokens <= 0:
            raise ValueError("max_queued_tokens must be positive")
        self.max_queued_tokens = int(max_queued_tokens)
        self.queued_tokens = 0
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> bool:
        with self._lock:
            if tokens <= 0 or self.queued_tokens + tokens > self.max_queued_tokens:
                return False
            self.queued_tokens += tokens
            return True

    def release(self, tokens: int) -> None:
        with self._lock:
            self.queued_tokens -= tokens
            if self.queued_tokens < 0:
                raise RuntimeError("Reasoner service token admission accounting underflow")

    def snapshot(self) -> int:
        with self._lock:
            return self.queued_tokens


def _cpu_feature_batch(features: ReasonerFeatureBatch) -> ReasonerFeatureBatch:
    return ReasonerFeatureBatch(
        cross_k=tuple(tensor.detach().to(device="cpu").contiguous() for tensor in features.cross_k),
        cross_v=tuple(tensor.detach().to(device="cpu").contiguous() for tensor in features.cross_v),
        causal_offsets=features.causal_offsets.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        fingerprints=features.fingerprints,
    )


class ReasonerFeatureService(reasoner_features_pb2_grpc.ReasonerFeatureServiceServicer):
    """Serve one runtime with bounded admission and exactly one GPU executor."""

    def __init__(
        self,
        runtime: ReasonerFeatureRuntime,
        *,
        max_queued_tokens: int,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        service_instance_id: str | None = None,
    ) -> None:
        if max_queued_tokens < runtime.max_total_tokens:
            raise ValueError("max_queued_tokens must be at least the runtime max_total_tokens")
        if max_chunk_bytes <= 0 or max_chunk_bytes > 4 * 1024**2:
            raise ValueError("max_chunk_bytes must be in (0, 4 MiB]")
        self.runtime = runtime
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.service_instance_id = service_instance_id or str(uuid.uuid4())
        self._admission = _TokenAdmission(max_queued_tokens)
        self._execution_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._healthy = True
        self._accepted_batches = 0
        self._rejected_batches = 0
        self._failed_batches = 0
        self._completed_batches = 0
        self._accepted_tokens = 0

    def _record_accept(self, tokens: int) -> None:
        with self._state_lock:
            self._accepted_batches += 1
            self._accepted_tokens += tokens

    def _record_reject(self) -> None:
        with self._state_lock:
            self._rejected_batches += 1

    def _record_failure(self, *, unhealthy: bool = False) -> None:
        with self._state_lock:
            self._failed_batches += 1
            if unhealthy:
                self._healthy = False

    def _record_complete(self) -> None:
        with self._state_lock:
            self._completed_batches += 1

    def _is_healthy(self) -> bool:
        with self._state_lock:
            return self._healthy

    def metrics(self) -> ReasonerServiceMetrics:
        with self._state_lock:
            return ReasonerServiceMetrics(
                accepted_batches=self._accepted_batches,
                rejected_batches=self._rejected_batches,
                failed_batches=self._failed_batches,
                completed_batches=self._completed_batches,
                accepted_tokens=self._accepted_tokens,
                queued_tokens=self._admission.snapshot(),
                healthy=self._healthy,
            )

    def GetInfo(self, request: Any, context: Any) -> Any:  # noqa: N802 - gRPC API name
        if int(request.protocol_version) != REASONER_FEATURE_PROTOCOL_VERSION:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"protocol_version={request.protocol_version}, expected={REASONER_FEATURE_PROTOCOL_VERSION}",
            )
        return reasoner_pb2.GetInfoResponse(
            protocol_version=REASONER_FEATURE_PROTOCOL_VERSION,
            service_instance_id=self.service_instance_id,
            identity=encode_identity(self.runtime.identity),
            signature=encode_signature(self.runtime.signature),
            capabilities=(
                reasoner_pb2.Capability(name="server_streaming", version=1),
                reasoner_pb2.Capability(name="strict_fingerprint", version=1),
                reasoner_pb2.Capability(name="ordered_requests", version=1),
                reasoner_pb2.Capability(name="single_document", version=1),
            ),
            max_batch_tokens=self.runtime.max_total_tokens,
            max_queued_tokens=self._admission.max_queued_tokens,
            max_chunk_bytes=self.max_chunk_bytes,
            ready=self._is_healthy(),
            max_requests=self.runtime.max_requests,
        )

    def Generate(self, request: Any, context: Any) -> Any:  # noqa: N802 - gRPC API name
        """Validate, enqueue, execute, stage to CPU, then stream an all-or-nothing result."""

        if not self._is_healthy():
            self._record_reject()
            context.abort(grpc.StatusCode.UNAVAILABLE, "Reasoner replica is unhealthy and requires restart")
        if int(request.protocol_version) != REASONER_FEATURE_PROTOCOL_VERSION:
            self._record_reject()
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"protocol_version={request.protocol_version}, expected={REASONER_FEATURE_PROTOCOL_VERSION}",
            )
        try:
            request_id, identity, requests = decode_generate_request(request)
        except (TypeError, ValueError) as error:
            self._record_reject()
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        if identity != self.runtime.identity:
            self._record_reject()
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Reasoner identity mismatch: request={identity!r}, service={self.runtime.identity!r}",
            )
        multi_document = [item.sample_key for item in requests if item.causal_offsets.numel() != 2]
        if multi_document:
            self._record_reject()
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"Reasoner service v1 supports one causal document per request; got {multi_document!r}",
            )
        token_count = sum(item.token_ids.numel() for item in requests)
        empty_requests = [item.sample_key for item in requests if item.token_ids.numel() == 0]
        if empty_requests:
            self._record_reject()
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Reasoner requests must contain at least one token; empty requests={empty_requests!r}",
            )
        if token_count > self.runtime.max_total_tokens or len(requests) > self.runtime.max_requests:
            self._record_reject()
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Request batch has {len(requests)} requests/{token_count} tokens; limits are "
                f"{self.runtime.max_requests}/{self.runtime.max_total_tokens}",
            )
        if not self._admission.acquire(token_count):
            self._record_reject()
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Reasoner queue token budget {self._admission.max_queued_tokens} is full",
            )
        self._record_accept(token_count)

        cancelled = threading.Event()
        if context is not None:
            context.add_callback(cancelled.set)

        def request_is_active() -> bool:
            # Direct iterator tests use no gRPC context. Production RPCs always
            # supply one and therefore exercise cancellation/deadline checks.
            return context is None or context.is_active()

        try:
            enqueued_ns = time.perf_counter_ns()
            execution_lock_acquired = False
            try:
                while not execution_lock_acquired:
                    if cancelled.is_set() or not request_is_active():
                        self._record_failure()
                        context.abort(grpc.StatusCode.CANCELLED, "Reasoner request was cancelled while queued")
                    if not self._is_healthy():
                        self._record_failure()
                        context.abort(
                            grpc.StatusCode.UNAVAILABLE,
                            "Reasoner replica became unhealthy while the request was queued",
                        )
                    execution_lock_acquired = self._execution_lock.acquire(timeout=_EXECUTION_LOCK_POLL_SECONDS)

                # Cancellation and health can change after the last poll but
                # before the lock acquisition. Re-check both at the exact GPU
                # execution boundary so expired calls never become ghost work,
                # and requests queued behind a fatal CUDA failure do not touch
                # the unhealthy replica.
                if cancelled.is_set() or not request_is_active():
                    self._record_failure()
                    context.abort(grpc.StatusCode.CANCELLED, "Reasoner request was cancelled before execution")
                if not self._is_healthy():
                    self._record_failure()
                    context.abort(
                        grpc.StatusCode.UNAVAILABLE,
                        "Reasoner replica became unhealthy while the request was queued",
                    )

                try:
                    compute_started_ns = time.perf_counter_ns()
                    queue_ns = compute_started_ns - enqueued_ns
                    features = self.runtime.execute(requests)
                    if self.runtime.device.type == "cuda":
                        torch.cuda.synchronize(self.runtime.device)
                    compute_finished_ns = time.perf_counter_ns()
                    d2h_started_ns = compute_finished_ns
                    cpu_features = _cpu_feature_batch(features)
                    if self.runtime.device.type == "cuda":
                        torch.cuda.synchronize(self.runtime.device)
                    d2h_finished_ns = time.perf_counter_ns()
                    del features
                except torch.cuda.OutOfMemoryError as error:
                    # Mark the replica unhealthy before releasing the execution
                    # lock. Otherwise a queued request can acquire the lock and
                    # enter the same broken CUDA context in the intervening race.
                    self._record_failure(unhealthy=True)
                    context.abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"Reasoner replica encountered CUDA OOM and was marked unhealthy: {error}",
                    )
                except (TypeError, ValueError) as error:
                    self._record_failure()
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
                except NotImplementedError as error:
                    self._record_failure()
                    context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
                except ReasonerRuntimeInvariantError as error:
                    # An output contract violation is deterministic for this
                    # loaded replica.  Mark it unhealthy while still holding
                    # the execution lock so queued calls cannot enter the same
                    # broken runtime in the intervening race.
                    self._record_failure(unhealthy=True)
                    context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"Reasoner runtime violated its output contract and was marked unhealthy: {error}",
                    )
                except RuntimeError as error:
                    is_cuda_failure = "CUDA" in str(error) or "cuda" in str(error)
                    self._record_failure(unhealthy=is_cuda_failure)
                    context.abort(grpc.StatusCode.INTERNAL, str(error))
                except Exception as error:
                    self._record_failure()
                    context.abort(grpc.StatusCode.INTERNAL, f"{type(error).__name__}: {error}")
            finally:
                if execution_lock_acquired:
                    self._execution_lock.release()

            try:
                yield from iter_feature_stream(
                    cpu_features,
                    request_id=request_id,
                    identity=self.runtime.identity,
                    signature=self.runtime.signature,
                    service_instance_id=self.service_instance_id,
                    max_chunk_bytes=self.max_chunk_bytes,
                    queue_ns=queue_ns,
                    compute_ns=compute_finished_ns - compute_started_ns,
                    device_to_host_ns=d2h_finished_ns - d2h_started_ns,
                )
            except GeneratorExit:
                self._record_failure()
                raise
            except Exception as error:
                self._record_failure()
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Reasoner response serialization failed: {type(error).__name__}: {error}",
                )
            self._record_complete()
        finally:
            # Keep the token reservation through streaming so slow/cancelled
            # clients cannot accumulate unbounded CPU-resident K/V responses.
            self._admission.release(token_count)


def create_reasoner_grpc_server(
    service: ReasonerFeatureService,
    *,
    address: str,
    max_rpc_workers: int = 16,
) -> tuple[Any, int]:
    """Create a configured server and bind it without starting it."""

    if max_rpc_workers <= 0:
        raise ValueError("max_rpc_workers must be positive")
    server = grpc.server(
        ThreadPoolExecutor(max_workers=max_rpc_workers, thread_name_prefix="reasoner-rpc"),
        maximum_concurrent_rpcs=max_rpc_workers,
        options=(
            ("grpc.max_receive_message_length", 16 * 1024**2),
            ("grpc.max_send_message_length", 8 * 1024**2),
        ),
    )
    reasoner_features_pb2_grpc.add_ReasonerFeatureServiceServicer_to_server(service, server)
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Reasoner feature service to {address!r}")
    return server, bound_port
