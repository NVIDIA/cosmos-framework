# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rollout-model adapter for generic replica-to-replica tensor broadcast.

The primitive is deliberately narrower than ``WeightSync``. ``WeightSync`` maps canonical policy
checkpoint tensors into an engine; this module starts after one complete rollout replica already
holds the desired weights and copies its live parameter/buffer layout to identical replicas. Each
corresponding intra-replica rank joins its own side NCCL group, so TP/EP-sharded live layouts remain
local to that rank and require no checkpoint-name mapping or engine reload.

The operation covers registered live parameter views and persistent-buffer views only. Engine-owned
padding, fused-weight scratch storage, or other bytes outside those registered views are neither
broadcast nor included in optional value verification.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch

from cosmos_framework.posttrain.comm.tensor_broadcast import (
    DEFAULT_STAGING_BUCKET_BYTES,
    CollectiveBroadcast,
    TensorBinding,
    TensorBroadcastPlan,
    TensorBroadcastSession,
    broadcast_tensors,
    build_tensor_broadcast_plan,
    value_digest,
)
from cosmos_framework.posttrain.comm.transfer.weight_comm import TorchDistComm, build_weight_comm


@dataclass(frozen=True)
class ReplicaBroadcastSpec:
    """Serializable membership for one corresponding-rank rollout replica group."""

    store_host: str
    store_port: int
    rank: int
    world_size: int
    source_rank: int = 0
    backend: str = "nccl"
    bucket_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.store_host:
            raise ValueError("replica broadcast store_host must be non-empty")
        if isinstance(self.store_port, bool) or not isinstance(self.store_port, int) or self.store_port < 1:
            raise ValueError(f"replica broadcast store_port must be positive, got {self.store_port!r}")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int) or self.world_size < 1:
            raise ValueError(f"replica broadcast world_size must be positive, got {self.world_size!r}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 0 <= self.rank < self.world_size:
            raise ValueError(f"replica broadcast rank {self.rank!r} is outside world_size={self.world_size}")
        if (
            isinstance(self.source_rank, bool)
            or not isinstance(self.source_rank, int)
            or not 0 <= self.source_rank < self.world_size
        ):
            raise ValueError(
                f"replica broadcast source_rank {self.source_rank!r} is outside world_size={self.world_size}"
            )
        if self.backend != "nccl":
            raise ValueError(f"replica broadcast currently supports only backend='nccl', got {self.backend!r}")
        if self.bucket_bytes is not None and (
            isinstance(self.bucket_bytes, bool) or not isinstance(self.bucket_bytes, int) or self.bucket_bytes < 1
        ):
            raise ValueError(f"replica broadcast bucket_bytes must be None or positive, got {self.bucket_bytes!r}")


LiveTensor = TensorBinding


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the actual local storage view for a Tensor or DTensor-like value."""
    to_local = getattr(tensor, "to_local", None)
    local = to_local() if callable(to_local) else tensor  # [*live]
    if not isinstance(local, torch.Tensor):
        raise TypeError(f"live model tensor resolved to {type(local).__name__}, expected torch.Tensor")
    return local


def _exact_view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify repeated names for the exact same live storage view without merging subviews."""
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(storage.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(step) for step in tensor.stride()),
    )


def _persistent_buffer_names(model: Any) -> set[str]:
    """Return fully-qualified names of buffers that belong to model state, excluding caches."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError(f"live rollout model {type(model).__name__} exposes no named_modules()")
    names: set[str] = set()
    for module_name, module in named_modules(remove_duplicate=False):
        buffers = getattr(module, "_buffers", {})
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for local_name, buffer in buffers.items():
            if buffer is None or local_name in non_persistent:
                continue
            names.add(f"{module_name}.{local_name}" if module_name else local_name)
    return names


def live_model_tensors(model: Any) -> tuple[LiveTensor, ...]:
    """Enumerate unique live parameters and persistent buffers in deterministic name order.

    ``.net`` is the framework's transparent model wrapper. The returned tensors are views into the
    currently executing model, never state-dict copies.
    """
    live_model = getattr(model, "net", model)
    named_parameters = getattr(live_model, "named_parameters", None)
    named_buffers = getattr(live_model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise TypeError(f"live rollout model {type(live_model).__name__} exposes no parameters/buffers")

    persistent_buffers = _persistent_buffer_names(live_model)
    named: list[tuple[str, torch.Tensor]] = []
    named.extend((f"parameter:{name}", tensor) for name, tensor in named_parameters(remove_duplicate=False))
    named.extend(
        (f"buffer:{name}", tensor)
        for name, tensor in named_buffers(remove_duplicate=False)
        if name in persistent_buffers
    )
    seen: set[tuple[Any, ...]] = set()
    result: list[LiveTensor] = []
    for name, value in sorted(named, key=lambda item: item[0]):
        tensor = _local_tensor(value)  # [*live]
        if tensor.numel() == 0:
            continue
        key = _exact_view_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        result.append(LiveTensor(name=name, tensor=tensor))
    if not result:
        raise ValueError(f"live rollout model {type(live_model).__name__} has no non-empty parameters or buffers")
    return tuple(result)


def broadcast_live_tensors(
    tensors: tuple[LiveTensor, ...],
    collective: CollectiveBroadcast,
    *,
    verify_values: bool = False,
) -> str:
    """One-shot compatibility surface; repeated model broadcasts cache a generic session."""
    return broadcast_tensors(
        tensors,
        collective,
        verify_values=verify_values,
        manifest_error_message=(
            "rollout replica live-tensor manifests differ for corresponding ranks; all replicas "
            "must use the same engine, model configuration, and parallel layout"
        ),
        value_error_message="rollout replica live-tensor values differ after broadcast for corresponding ranks",
    )


_CommKey = tuple[str, int, int, int, str]

_COMM_CACHE: dict[_CommKey, TorchDistComm] = {}
_COMM_CACHE_LOCK = Lock()
_LOCAL_PLAN_CACHE: dict[int, TensorBroadcastPlan] = {}
_MODEL_CACHE_LOCK = Lock()
_SESSION_CACHE: dict[tuple[int, ReplicaBroadcastSpec], TensorBroadcastSession] = {}
_SESSION_CACHE_LOCK = Lock()


def _comm_for(spec: ReplicaBroadcastSpec) -> TorchDistComm:
    """Build the side communicator lazily in the process that owns the live model."""
    key = (spec.store_host, spec.store_port, spec.rank, spec.world_size, spec.backend)
    with _COMM_CACHE_LOCK:
        comm = _COMM_CACHE.get(key)
        if comm is None:
            comm = build_weight_comm(
                spec.store_host,
                spec.store_port,
                spec.rank,
                spec.world_size,
                backend=spec.backend,
            )
            _COMM_CACHE[key] = comm
        return comm


def _model_key(model: Any) -> int:
    """Return the stable identity whose live tensor views remain fixed for one session."""
    return id(getattr(model, "net", model))


def _validated_bindings(model: Any) -> tuple[LiveTensor, ...]:
    """Enumerate one fixed set of CUDA live views for a new cached plan or session."""
    bindings = live_model_tensors(model)
    non_cuda = [item.name for item in bindings if item.tensor.device.type != "cuda"]
    if non_cuda:
        raise ValueError(
            "NCCL replica broadcast requires every live parameter/buffer on CUDA, "
            f"but found non-CUDA tensors {non_cuda[:8]}"
        )
    return bindings


def _local_plan_for(model: Any) -> TensorBroadcastPlan:
    """Build and cache the local-only layout proof without creating a communicator."""
    key = _model_key(model)
    with _MODEL_CACHE_LOCK:
        plan = _LOCAL_PLAN_CACHE.get(key)
        if plan is None:
            plan = build_tensor_broadcast_plan(
                _validated_bindings(model),
                bucket_bytes=DEFAULT_STAGING_BUCKET_BYTES,
            )
            _LOCAL_PLAN_CACHE[key] = plan
        return plan


def _session_for(model: Any, spec: ReplicaBroadcastSpec) -> TensorBroadcastSession:
    """Build and cache the generic tensor-broadcast session in the model-owning process."""
    key = (_model_key(model), spec)
    with _SESSION_CACHE_LOCK:
        session = _SESSION_CACHE.get(key)
        if session is None:
            session = TensorBroadcastSession(
                _validated_bindings(model),
                CollectiveBroadcast(_comm_for(spec), source_rank=spec.source_rank),
                bucket_bytes=spec.bucket_bytes or DEFAULT_STAGING_BUCKET_BYTES,
                manifest_error_message=(
                    "rollout replica live-tensor manifests differ for corresponding ranks; all replicas "
                    "must use the same engine, model configuration, and parallel layout"
                ),
                value_error_message=(
                    "rollout replica live-tensor values differ after broadcast for corresponding ranks"
                ),
            )
            _SESSION_CACHE[key] = session
        return session


def broadcast_live_model(model: Any, spec: ReplicaBroadcastSpec, *, verify_values: bool = False) -> str:
    """Broadcast one engine's live tensors across identical rollout replicas."""
    if not isinstance(verify_values, bool):
        raise TypeError(f"verify_values must be bool, got {verify_values!r}")
    if spec.world_size == 1:
        plan = _local_plan_for(model)
        return value_digest(plan.bindings) if verify_values else plan.manifest_digest
    return _session_for(model, spec).broadcast(verify_values=verify_values)


def initialize_replica_broadcast(model: Any, spec: ReplicaBroadcastSpec) -> None:
    """Eagerly cache live references, validate layout, and claim the side process group."""
    if spec.world_size == 1:
        _local_plan_for(model)
        return
    _session_for(model, spec)


def _close_cached_comms() -> None:
    """Best-effort process-exit cleanup for lazily-created side process groups."""
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE.clear()
    with _MODEL_CACHE_LOCK:
        _LOCAL_PLAN_CACHE.clear()
    with _COMM_CACHE_LOCK:
        comms = tuple(_COMM_CACHE.values())
        _COMM_CACHE.clear()
    for comm in comms:
        comm.close()


atexit.register(_close_cached_comms)
