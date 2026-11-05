# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reusable fixed-layout tensor broadcast over an injected collective communicator.

The session owns no model or role semantics. Callers provide a stable ordered tuple of live tensor
bindings once, then reuse the session for repeated in-place broadcasts. Contiguous tensors travel
directly. Non-contiguous tensors are packed into size-targeted mixed-dtype byte buckets that reuse
one temporary scratch allocation per broadcast call. A tensor larger than the target occupies one
oversized bucket rather than being copied into another full-tensor intermediate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

import torch

DEFAULT_STAGING_BUCKET_BYTES = 16 * 1024 * 1024


class CollectiveWork(Protocol):
    """Completion handle returned by an asynchronously submitted collective."""

    def wait(self) -> bool | None: ...


class TensorBroadcastComm(Protocol):
    """Minimal communicator surface required by the tensor-broadcast primitive."""

    rank: int
    world_size: int

    def submit_broadcast(self, tensor: torch.Tensor, src: int) -> CollectiveWork:  # tensor: [*wire]
        ...

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        ...

    def all_reduce_max(self, tensor: torch.Tensor) -> None:  # tensor: [*control]
        ...


@dataclass(frozen=True)
class TensorBinding:
    """One stable name and live tensor view supplied by a framework-level adapter."""

    name: str
    tensor: torch.Tensor  # [*live]


@dataclass(frozen=True)
class PackedTensor:
    """One non-contiguous live tensor's aligned slice in a byte bucket."""

    binding: TensorBinding
    offset_bytes: int
    num_bytes: int


@dataclass(frozen=True)
class PackedBucket:
    """A size-targeted collection of non-contiguous tensors carried by one byte broadcast."""

    entries: tuple[PackedTensor, ...]
    num_bytes: int


@dataclass(frozen=True)
class TensorBroadcastPlan:
    """Immutable live references, layout proof, and transfer schedule for one session."""

    bindings: tuple[TensorBinding, ...]
    direct: tuple[TensorBinding, ...]
    packed: tuple[PackedBucket, ...]
    device: torch.device
    manifest_digest: str
    max_staging_bytes: int


class CollectiveBroadcast:
    """Fixed-source broadcast operations over an injected communicator."""

    _comm: TensorBroadcastComm
    _source_rank: int

    def __init__(self, comm: TensorBroadcastComm, *, source_rank: int = 0) -> None:
        if not 0 <= source_rank < comm.world_size:
            raise ValueError(f"broadcast source_rank={source_rank} is outside world_size={comm.world_size}")
        self._comm = comm
        self._source_rank = int(source_rank)

    @property
    def rank(self) -> int:
        return int(self._comm.rank)

    @property
    def source_rank(self) -> int:
        return self._source_rank

    @property
    def world_size(self) -> int:
        return int(self._comm.world_size)

    def submit(self, tensor: torch.Tensor) -> CollectiveWork:  # tensor: [*wire]
        """Submit one in-place broadcast without waiting for completion."""
        return self._comm.submit_broadcast(tensor, src=self._source_rank)

    def broadcast(self, tensor: torch.Tensor) -> None:  # tensor: [*wire]
        """Synchronously broadcast one tensor, used only during session initialization."""
        self._comm.broadcast(tensor, src=self._source_rank)

    def all_reduce_max(self, tensor: torch.Tensor) -> None:  # tensor: [*control]
        """Make every participant observe the elementwise group maximum."""
        self._comm.all_reduce_max(tensor)


def _manifest(bindings: tuple[TensorBinding, ...]) -> list[dict[str, Any]]:
    """Build bounded layout metadata used to reject divergent participants."""
    return [
        {
            "name": item.name,
            "shape": [int(dim) for dim in item.tensor.shape],
            "stride": [int(step) for step in item.tensor.stride()],
            "dtype": str(item.tensor.dtype),
        }
        for item in bindings
    ]


def _manifest_digest(bindings: tuple[TensorBinding, ...]) -> str:
    encoded = json.dumps(_manifest(bindings), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def value_digest(bindings: tuple[TensorBinding, ...]) -> str:
    """Hash exact logical tensor bytes, materializing at most one tensor-sized host buffer."""
    digest = hashlib.sha256()
    for item in bindings:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        contiguous = item.tensor.detach().contiguous()  # [*live]
        byte_view = contiguous.reshape(-1).view(torch.uint8).cpu()  # [N_bytes]
        digest.update(byte_view.numpy().tobytes())
    return digest.hexdigest()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _packed_buckets(
    bindings: tuple[TensorBinding, ...],
    *,
    bucket_bytes: int,
) -> tuple[PackedBucket, ...]:
    """Pack non-contiguous bindings into deterministic aligned byte buckets."""
    buckets: list[PackedBucket] = []
    entries: list[PackedTensor] = []
    used_bytes = 0

    def flush() -> None:
        nonlocal entries, used_bytes
        if entries:
            buckets.append(PackedBucket(entries=tuple(entries), num_bytes=used_bytes))
        entries = []
        used_bytes = 0

    for binding in bindings:
        tensor = binding.tensor  # [*live]
        if tensor.is_contiguous():
            continue
        num_bytes = int(tensor.numel()) * int(tensor.element_size())
        alignment = max(1, int(tensor.element_size()))
        offset = _align_up(used_bytes, alignment)
        if entries and offset + num_bytes > bucket_bytes:
            flush()
            offset = 0
        entries.append(PackedTensor(binding=binding, offset_bytes=offset, num_bytes=num_bytes))
        used_bytes = offset + num_bytes
        if used_bytes >= bucket_bytes:
            flush()
    flush()
    return tuple(buckets)


def build_tensor_broadcast_plan(
    bindings: tuple[TensorBinding, ...],
    *,
    bucket_bytes: int = DEFAULT_STAGING_BUCKET_BYTES,
) -> TensorBroadcastPlan:
    """Validate fixed live references and build their reusable transfer schedule."""
    if not bindings:
        raise ValueError("tensor broadcast requires at least one binding")
    if isinstance(bucket_bytes, bool) or not isinstance(bucket_bytes, int) or bucket_bytes < 1:
        raise ValueError(f"tensor broadcast bucket_bytes must be positive, got {bucket_bytes!r}")
    names = [item.name for item in bindings]
    if len(names) != len(set(names)):
        raise ValueError("tensor broadcast binding names must be unique")
    if any(item.tensor.numel() == 0 for item in bindings):
        raise ValueError("tensor broadcast bindings must be non-empty")
    device = bindings[0].tensor.device
    mismatched = [item.name for item in bindings if item.tensor.device != device]
    if mismatched:
        raise ValueError(
            f"tensor broadcast bindings must share one device {device}, but found mismatches {mismatched[:8]}"
        )
    packed = _packed_buckets(bindings, bucket_bytes=bucket_bytes)
    return TensorBroadcastPlan(
        bindings=bindings,
        direct=tuple(item for item in bindings if item.tensor.is_contiguous()),
        packed=packed,
        device=device,
        manifest_digest=_manifest_digest(bindings),
        max_staging_bytes=max((bucket.num_bytes for bucket in packed), default=0),
    )


def _require_digest_consensus(
    digest: str,
    device: torch.device,
    collective: CollectiveBroadcast,
    *,
    error_message: str,
) -> None:
    """Require every participant to match the source's digest."""
    digest_tensor = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=device)  # [32]
    source_digest = digest_tensor.clone()  # [32]
    collective.broadcast(source_digest)
    mismatch = torch.tensor(
        [int(not torch.equal(digest_tensor, source_digest))], dtype=torch.int32, device=device
    )  # [1]
    collective.all_reduce_max(mismatch)
    if int(mismatch.item()) != 0:
        raise RuntimeError(error_message)


def _bucket_view(scratch: torch.Tensor, entry: PackedTensor) -> torch.Tensor:  # scratch: [N_bytes]
    """Return one typed contiguous view into an aligned byte bucket."""
    raw = scratch.narrow(0, entry.offset_bytes, entry.num_bytes)  # [entry_bytes]
    typed = raw.view(entry.binding.tensor.dtype)  # [entry_elements]
    return typed.reshape(entry.binding.tensor.shape)  # [*live]


class TensorBroadcastSession:
    """Initialize one fixed tensor layout once, then broadcast its changing values repeatedly."""

    _collective: CollectiveBroadcast
    _plan: TensorBroadcastPlan
    _broadcast_lock: Lock
    _value_error_message: str

    def __init__(
        self,
        bindings: tuple[TensorBinding, ...],
        collective: CollectiveBroadcast,
        *,
        bucket_bytes: int = DEFAULT_STAGING_BUCKET_BYTES,
        manifest_error_message: str = (
            "tensor broadcast manifests differ across participants; every rank must provide "
            "the same names, shapes, strides, dtypes, and ordering"
        ),
        value_error_message: str = "tensor values differ after broadcast",
    ) -> None:
        self._collective = collective
        self._plan = build_tensor_broadcast_plan(bindings, bucket_bytes=bucket_bytes)
        self._broadcast_lock = Lock()
        self._value_error_message = value_error_message
        _require_digest_consensus(
            self._plan.manifest_digest,
            self._plan.device,
            collective,
            error_message=manifest_error_message,
        )

    @property
    def plan(self) -> TensorBroadcastPlan:
        return self._plan

    def broadcast(self, *, verify_values: bool = False) -> str:
        """Overwrite followers from the fixed source using the cached transfer plan."""
        if not isinstance(verify_values, bool):
            raise TypeError(f"verify_values must be bool, got {verify_values!r}")
        if not self._broadcast_lock.acquire(blocking=False):
            raise RuntimeError("tensor broadcast sessions are not reentrant")
        try:
            return self._broadcast(verify_values=verify_values)
        finally:
            self._broadcast_lock.release()

    def _broadcast(self, *, verify_values: bool) -> str:
        with torch.no_grad():
            direct_works = [self._collective.submit(item.tensor) for item in self._plan.direct]
            if self._plan.max_staging_bytes:
                scratch = torch.empty(
                    self._plan.max_staging_bytes,
                    dtype=torch.uint8,
                    device=self._plan.device,
                )  # [max_bucket_bytes]
                for bucket in self._plan.packed:
                    wire = scratch.narrow(0, 0, bucket.num_bytes)  # [bucket_bytes]
                    if self._collective.rank == self._collective.source_rank:
                        for entry in bucket.entries:
                            packed = _bucket_view(scratch, entry)  # [*live]
                            packed.copy_(entry.binding.tensor)  # [*live]
                    work = self._collective.submit(wire)
                    work.wait()
                    if self._collective.rank != self._collective.source_rank:
                        for entry in bucket.entries:
                            packed = _bucket_view(scratch, entry)  # [*live]
                            entry.binding.tensor.copy_(packed)  # [*live]
            for work in direct_works:
                work.wait()
        if self._plan.device.type == "cuda":
            torch.cuda.current_stream(device=self._plan.device).synchronize()

        digest = self._plan.manifest_digest
        if verify_values:
            digest = value_digest(self._plan.bindings)
            _require_digest_consensus(
                digest,
                self._plan.device,
                self._collective,
                error_message=self._value_error_message,
            )
        return digest


def broadcast_tensors(
    bindings: tuple[TensorBinding, ...],
    collective: CollectiveBroadcast,
    *,
    verify_values: bool = False,
    bucket_bytes: int = DEFAULT_STAGING_BUCKET_BYTES,
    manifest_error_message: str = (
        "tensor broadcast manifests differ across participants; every rank must provide "
        "the same names, shapes, strides, dtypes, and ordering"
    ),
    value_error_message: str = "tensor values differ after broadcast",
) -> str:
    """One-shot compatibility surface; repeated callers should cache the session instead."""
    session = TensorBroadcastSession(
        bindings,
        collective,
        bucket_bytes=bucket_bytes,
        manifest_error_message=manifest_error_message,
        value_error_message=value_error_message,
    )
    return session.broadcast(verify_values=verify_values)
