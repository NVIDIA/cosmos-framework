# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Worker-local, fail-closed lifecycle for reusable trainer batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from cosmos_framework.posttrain.schema import Trajectory

_T = TypeVar("_T", bound=Trajectory)


class BatchCache(Generic[_T]):
    """Hold exactly one active trajectory batch on one model worker.

    The controller routes only ``batch_id`` and episode IDs after installation;
    trajectory graphs and processor mappings remain inside this cache.
    """

    def __init__(self, role: str, *, digest: Callable[[_T], str]) -> None:
        if not isinstance(role, str) or not role:
            raise ValueError("batch cache role must be a non-empty string")
        if not callable(digest):
            raise TypeError("batch cache digest must be callable")
        self.role: str = role
        self._digest: Callable[[_T], str] = digest
        self._batch_id: str | None = None
        self._items: dict[str, _T] = {}
        self._digests: dict[str, str] = {}
        self._completed: set[str] = set()
        self._aborted: set[str] = set()

    @staticmethod
    def _validate_batch_id(batch_id: str) -> None:
        if not isinstance(batch_id, str) or not batch_id.startswith("sha256:"):
            raise ValueError("batch cache batch_id must be a SHA256 digest")

    def _require_active(self, batch_id: str) -> None:
        self._validate_batch_id(batch_id)
        if self._batch_id is None:
            raise RuntimeError(f"{self.role} batch cache is missing active batch {batch_id!r}")
        if self._batch_id != batch_id:
            raise RuntimeError(
                f"{self.role} batch cache received stale batch {batch_id!r}; active batch is {self._batch_id!r}"
            )

    def _index(self, trajectories: Sequence[_T]) -> tuple[dict[str, _T], dict[str, str]]:
        items: dict[str, _T] = {}
        digests: dict[str, str] = {}
        for item in trajectories:
            if not isinstance(item, Trajectory):
                raise TypeError(f"{self.role} batch cache accepts only Trajectory objects")
            episode_id = item.id
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"{self.role} batch cache received an invalid trajectory ID")
            if episode_id in items:
                raise ValueError(f"{self.role} batch cache received duplicate trajectory ID {episode_id!r}")
            digest = self._digest(item)
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError(f"{self.role} batch cache digest for {episode_id!r} is invalid")
            items[episode_id] = item
            digests[episode_id] = digest
        if not items:
            raise ValueError(f"{self.role} batch cache cannot install an empty batch")
        return items, digests

    def install(self, batch_id: str, trajectories: Sequence[_T]) -> None:
        """Install one new batch; no active or completed ID may be reused."""
        self._validate_batch_id(batch_id)
        if self._batch_id is not None:
            raise RuntimeError(f"{self.role} batch cache already has active batch {self._batch_id!r}")
        if batch_id in self._completed or batch_id in self._aborted:
            raise RuntimeError(f"{self.role} batch cache batch {batch_id!r} already completed")
        items, digests = self._index(trajectories)
        self._batch_id = batch_id
        self._items = items
        self._digests = digests

    def replace(self, batch_id: str, trajectories: Sequence[_T], *, preserve_identity: bool = True) -> None:
        """Atomically replace active overlays while retaining exact episode coverage."""
        self._require_active(batch_id)
        items, digests = self._index(trajectories)
        if set(items) != set(self._items):
            raise ValueError(
                f"{self.role} batch cache replacement IDs differ: "
                f"missing={sorted(set(self._items) - set(items))!r}, "
                f"extra={sorted(set(items) - set(self._items))!r}"
            )
        if preserve_identity and digests != self._digests:
            drifted = sorted(episode_id for episode_id in items if digests[episode_id] != self._digests[episode_id])
            raise ValueError(f"{self.role} batch cache replacement changed trajectory identity for {drifted!r}")
        self._items = items
        self._digests = digests

    def select(self, batch_id: str, episode_ids: Sequence[str]) -> tuple[_T, ...]:
        """Resolve one ID-only schedule entry inside the worker."""
        self._require_active(batch_id)
        copied = tuple(episode_ids)
        if not copied:
            raise ValueError(f"{self.role} batch cache selection cannot be empty")
        if any(not isinstance(episode_id, str) or not episode_id for episode_id in copied):
            raise ValueError(f"{self.role} batch cache selection contains an invalid episode ID")
        if len(set(copied)) != len(copied):
            raise ValueError(f"{self.role} batch cache selection contains duplicate episode IDs")
        missing = [episode_id for episode_id in copied if episode_id not in self._items]
        if missing:
            raise KeyError(f"{self.role} batch cache selection is missing IDs {missing!r}")
        return tuple(self._items[episode_id] for episode_id in copied)

    def digest_map(self, batch_id: str) -> dict[str, str]:
        """Return tiny identity evidence, never cached trajectories."""
        self._require_active(batch_id)
        return dict(self._digests)

    def episode_ids(self, batch_id: str) -> tuple[str, ...]:
        """Return the active batch's episode IDs in install order.

        Install order is the role's own model-DP shard order, which is what lets the role derive its
        update schedule from its cache instead of being told which episodes to train on.
        """
        self._require_active(batch_id)
        return tuple(self._items)

    def clear(self, batch_id: str) -> None:
        """Strictly complete the active batch and tombstone its ID."""
        self._validate_batch_id(batch_id)
        if self._batch_id is None and batch_id in self._completed:
            return
        self._require_active(batch_id)
        self._completed.add(batch_id)
        self._batch_id = None
        self._items = {}
        self._digests = {}

    def abort(self, batch_id: str) -> None:
        """Force-clear the matching active batch; retrying the same abort is safe."""
        self._validate_batch_id(batch_id)
        if self._batch_id is None:
            # Distributed cleanup is all-cell and retry-safe: a peer may have
            # failed before install or may already have completed strict clear.
            # With no active batch there is nothing this ID could accidentally
            # discard, so force-abort converges as a no-op.
            return
        if self._batch_id != batch_id:
            raise RuntimeError(
                f"{self.role} batch cache abort received stale batch {batch_id!r}; active is {self._batch_id!r}"
            )
        self._aborted.add(batch_id)
        self._batch_id = None
        self._items = {}
        self._digests = {}


__all__ = ["BatchCache"]
