# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact-resume identity for frozen or moving reference networks."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.utils.dist import is_dtensor_sharded, raise_if_any_rank_failed, run_rank_converged

if TYPE_CHECKING:
    import torch

    from cosmos_framework.posttrain.model.protocol import FlowMatchingPolicy

_REFERENCE_IDENTITY_VERSION = 2
_FROZEN_REFERENCE_LOAD_MAPPING = "DiffusionTrainer.copy_mode:model.net->policy.net+policy.net_ref:strict:v1"


def _hash_identity_field(digest: Any, value: str) -> None:
    payload = value.encode("utf-8", errors="surrogateescape")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _tensor_inventory(net_ref: torch.nn.Module) -> list[tuple[str, str, torch.Tensor]]:
    inventory = [("parameter", name, parameter) for name, parameter in net_ref.named_parameters(remove_duplicate=False)]
    inventory.extend(("buffer", name, buffer) for name, buffer in net_ref.named_buffers(remove_duplicate=False))
    return sorted(inventory, key=lambda item: (item[0], item[1]))


def _schema_sha256(net_ref: torch.nn.Module) -> str:
    """Bind the concrete module type and global tensor mapping used by copy mode."""
    digest = hashlib.sha256()
    digest.update(b"posttrain.dpo.frozen-reference-schema.v1\0")
    _hash_identity_field(digest, f"{type(net_ref).__module__}.{type(net_ref).__qualname__}")
    for name, module in net_ref.named_modules(remove_duplicate=False):
        _hash_identity_field(digest, name)
        _hash_identity_field(digest, f"{type(module).__module__}.{type(module).__qualname__}")
    for kind, name, tensor in _tensor_inventory(net_ref):
        _hash_identity_field(digest, kind)
        _hash_identity_field(digest, name)
        _hash_identity_field(digest, str(tensor.dtype))
        _hash_identity_field(digest, json.dumps(list(tensor.shape), separators=(",", ":")))
    return digest.hexdigest()


def _declared_semantics(model: FlowMatchingPolicy) -> dict[str, Any]:
    from cosmos_framework.posttrain.utils.identity import normalize_json_dict

    capability = getattr(model, "reference_semantics_identity", None)
    if not callable(capability):
        raise TypeError(
            f"{type(model).__module__}.{type(model).__qualname__} must implement "
            "reference_semantics_identity() before it can be used with an exact-resume reference"
        )
    return normalize_json_dict(capability(), context="reference_semantics_identity()")


def _require_identical(identity: dict[str, Any], context: str) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils.identity import json_identity_equal

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    identities: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity)
    if any(not json_identity_equal(candidate, identity, context="exact-resume identity") for candidate in identities):
        raise RuntimeError(f"{context} differs across distributed ranks: {identities!r}")


def materialized_reference_sha256(net_ref: torch.nn.Module) -> str:
    """Hash all reference tensor bytes, materializing one global DTensor at a time."""
    import torch
    import torch.distributed as dist

    sharded = is_dtensor_sharded(net_ref)
    if sharded and (not dist.is_initialized() or dist.get_world_size() < 2):
        raise RuntimeError("a DTensor frozen reference requires an initialized multi-rank process group")
    rank = dist.get_rank() if sharded else 0
    digest = hashlib.sha256()
    digest.update(b"posttrain.dpo.materialized-frozen-reference.v1\0")
    inventory = _tensor_inventory(net_ref)
    if sharded:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError as error:
            raise RuntimeError("DTensor support is required to identify a sharded frozen reference") from error
        local_descriptor = [
            (
                kind,
                name,
                str(tensor.dtype),
                tuple(tensor.shape),
                isinstance(tensor, DTensor) or isinstance(getattr(tensor, "data", None), DTensor),
            )
            for kind, name, tensor in inventory
        ]
        descriptors: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(descriptors, local_descriptor)
        if any(descriptor != local_descriptor for descriptor in descriptors):
            raise RuntimeError("frozen-reference tensor inventory differs across distributed ranks")
    else:
        DTensor = ()  # type: ignore[misc,assignment] -- sentinel for the unsharded branch

    for kind, name, tensor in inventory:
        _hash_identity_field(digest, kind)
        _hash_identity_field(digest, name)
        _hash_identity_field(digest, str(tensor.dtype))
        _hash_identity_field(digest, json.dumps(list(tensor.shape), separators=(",", ":")))
        distributed_tensor = tensor if sharded and isinstance(tensor, DTensor) else getattr(tensor, "data", None)
        is_distributed_tensor = bool(sharded and isinstance(distributed_tensor, DTensor))
        detached = distributed_tensor.detach() if is_distributed_tensor else tensor.detach()  # [*P]
        materialization_error: BaseException | None = None
        materialized: torch.Tensor | None = None
        replicated_sha256: str | None = None
        try:
            materialized = detached.full_tensor() if is_distributed_tensor else detached  # [*P_global]
            if materialized.layout != torch.strided:
                materialized = materialized.to_dense()  # [*P_global]
            if rank == 0 or not is_distributed_tensor:
                cpu_tensor = materialized.to(device="cpu").contiguous().reshape(-1)  # [N_elements]
                byte_view = cpu_tensor.view(torch.uint8)  # [N_bytes]
                byte_array = byte_view.numpy()
                payload = memoryview(byte_array)
                if sharded and not is_distributed_tensor:
                    replicated_sha256 = hashlib.sha256(payload).hexdigest()
            if rank == 0:
                digest.update(byte_view.numel().to_bytes(8, "big"))
                digest.update(payload)
        except BaseException as error:  # noqa: BLE001 -- converge before the next DTensor collective
            materialization_error = error
        if sharded:
            raise_if_any_rank_failed(
                materialization_error,
                f"frozen-reference identity materialization for {kind} {name!r}",
            )
        elif materialization_error is not None:
            raise materialization_error
        if sharded and not is_distributed_tensor:
            replicated_digests: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(replicated_digests, replicated_sha256)
            if len(set(replicated_digests)) != 1:
                raise RuntimeError(f"replicated frozen-reference tensor {name!r} differs across ranks")
        if rank == 0 or not is_distributed_tensor:
            del cpu_tensor, byte_view, byte_array, payload
        del materialized
    result: list[str | None] = [digest.hexdigest() if rank == 0 else None]
    if sharded:
        dist.broadcast_object_list(result, src=0)
    if result[0] is None:
        raise RuntimeError("rank zero omitted the frozen-reference content identity")
    return result[0]


class ReferenceIdentity:
    """Own the exact reference identity required by DPO resume."""

    model: FlowMatchingPolicy
    refresh_every_iter: int
    current: dict[str, Any] | None

    def __init__(
        self,
        model: FlowMatchingPolicy,
        *,
        refresh_every_iter: int,
        defer_frozen: bool = False,
    ) -> None:
        self.model = model
        self.refresh_every_iter = int(refresh_every_iter)
        if defer_frozen and (self.refresh_every_iter > 0 or getattr(model, "net_ref", None) is None):
            raise ValueError("defer_frozen_reference_identity requires a frozen net_ref")
        self.current: dict[str, Any] | None = None if defer_frozen else self._build()

    def _build(self) -> dict[str, Any]:
        net_ref = getattr(self.model, "net_ref", None)

        def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
            if net_ref is None:
                base_identity = {"version": _REFERENCE_IDENTITY_VERSION, "mode": "absent"}
                distributed_preflight = {"identity": base_identity, "sharded": False}
            else:
                base_identity = {
                    "version": _REFERENCE_IDENTITY_VERSION,
                    "mode": "moving_checkpointed" if self.refresh_every_iter > 0 else "frozen",
                    "model_type": f"{type(net_ref).__module__}.{type(net_ref).__qualname__}",
                    "schema_sha256": _schema_sha256(net_ref),
                    "declared_semantics": _declared_semantics(self.model),
                }
                distributed_preflight = {"identity": base_identity, "sharded": is_dtensor_sharded(net_ref)}
            return base_identity, distributed_preflight

        base_identity, distributed_preflight = run_rank_converged(
            prepare,
            "reference semantics identity preparation",
        )
        _require_identical(distributed_preflight, "reference semantics identity")
        if net_ref is None:
            return base_identity
        if self.refresh_every_iter > 0:
            identity = {**base_identity, "refresh_every_iter": self.refresh_every_iter}
            _require_identical(identity, "moving reference identity")
            return identity
        reference_sha256 = run_rank_converged(
            lambda: materialized_reference_sha256(net_ref),
            "frozen reference content identity",
        )
        identity = {
            **base_identity,
            "source": {
                "version": _REFERENCE_IDENTITY_VERSION,
                "kind": "materialized_reference",
                "sha256": reference_sha256,
            },
            "load_mapping": _FROZEN_REFERENCE_LOAD_MAPPING,
        }
        _require_identical(identity, "frozen reference identity")
        return identity

    def bind_after_copy(self) -> None:
        """Bind the exact reference bytes populated by copy mode."""
        self.current = self._build()

    def state_dict(self) -> dict[str, Any]:
        if self.current is None:
            raise RuntimeError("frozen-reference identity was deferred for init_from but copy_mode has not completed")
        return copy.deepcopy(self.current)

    def validate_restored(self, restored: Any) -> None:
        from cosmos_framework.posttrain.utils.identity import json_identity_equal

        current = self.state_dict()
        if restored is None:
            if current.get("mode") == "frozen":
                raise RuntimeError(
                    "checkpoint predates the frozen-reference identity journal; exact DPO resume "
                    "cannot prove that net_ref has the original base weights (use rl.init_from to "
                    "start a new tuning run)"
                )
            return
        if not json_identity_equal(restored, current, context="exact-resume identity"):
            raise RuntimeError(f"trainer checkpoint reference identity={restored!r}, current run expects {current!r}")
