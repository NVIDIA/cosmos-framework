# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The transfer primitive: a placement-keyed tensor Mover (design spec §2).

A Mover moves a tensor (or dict of tensors) from a producer placement to a consumer
placement. `stage` (producer) makes the payload movable and returns an opaque `MoveHandle`;
`fetch` (consumer) realizes it at this endpoint.

For the data-plane movers `stage` is cheap and the byte movement happens in `fetch`, which is
what makes filter-before-transfer work (§3.4): an un-fetched handle never moves. The same-GPU
weight mover is NOT one of these — it packs the payload inside `stage` — but every staged weight
bucket is pushed and there is nothing to filter. `resolve_mover` keeps the two planes apart.

Capability flags keep mechanisms honest:
- lifetime: "staged" (shared/copy/ipc — handle persists, fetch anytime) vs "collective"
  (nccl — stage/fetch are a co-scheduled matched pair).
- tensors_only: ipc/nccl require torch tensors; shared/copy accept any object.

`reserve`/`release_staged` are the staging-buffer lifetime pair — §3.5 #3's registered-buffer
pool and §3.5 #4's explicit release. Every mover implements them because the callers
(`WeightSync.set_bucket_plan`, the stream seams) are generic; they are a no-op for movers and
modes that stage nothing worth reusing.

This module is a leaf (no other comm imports) so bulk.py can import BulkRef from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

STAGED = "staged"
COLLECTIVE = "collective"


@dataclass(frozen=True)
class MoveHandle:
    """Opaque descriptor produced by stage(), consumed by fetch(). Contents are
    mechanism-specific: shared -> tensors/ref; copy -> cpu dict or store locator;
    ipc -> reduce_tensor handles; nccl -> rendezvous descriptor (ranks/tag/shapes)."""

    kind: str
    payload: Any = None
    meta: dict | None = None


@dataclass(frozen=True)
class BulkRef(MoveHandle):
    """Data-plane name for a store-locator handle (kept for readability). A MoveHandle subtype
    so movers treat it uniformly, but with its OWN hashable scalar fields — the data plane keys
    BulkRefs in dicts (RayStoreMover's prefetch cache, the per-owner batch map), so a BulkRef
    MUST stay hashable. Storing the locator in the base's dict-valued `meta` would make the frozen
    dataclass unhashable ("unhashable type: 'dict'"); scalar fields keep it hashable."""

    store_id: str = ""
    key: int = -1
    nbytes: int = 0
    dtype: str = ""

    @classmethod
    def for_bulk(cls, store_id: str, key: int, nbytes: int = 0, dtype: str = "") -> "BulkRef":
        return cls(kind="copy", store_id=store_id, key=key, nbytes=nbytes, dtype=dtype)


@runtime_checkable
class Mover(Protocol):
    lifetime: ClassVar[str]
    tensors_only: ClassVar[bool]

    def stage(self, obj: Any) -> MoveHandle: ...
    def fetch(self, handle: MoveHandle) -> Any: ...
    def reserve(self, buckets: list) -> None: ...
    def release_staged(self) -> None: ...
    def close(self) -> None:
        """Idempotent final cleanup of mover-owned resources; shared process resources keep their own lifetime."""
        ...
