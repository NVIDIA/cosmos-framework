# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Traversal over a map-style dataset: order, epoch, shard, position, resume.

IT NEVER READS THE DATA. The cursor answers "which index next", and the caller fetches
`dataset[i]` itself -- `DatasetEnv` and `DatasetPreferenceEnv` both own the source anyway. The
dataset object is still held, for three things that are traversal concerns rather than access:
its LENGTH (which is the index space, and authoritative here so a caller cannot pass a stale
count and silently corrupt sharding), its `cursor_fingerprint()` (checked inside
`load_state_dict`, so a snapshot cannot be restored without proving it points at the same data),
and its optional `cursor_state_dict`/`load_cursor_state_dict` pair (sources that carry their own
internal position). The `__getitem__` requirement in `__init__` is asserted on the CALLER's
behalf: an iterable-only source cannot be sharded or resumed, and failing at construction beats
failing on the first draw.

Owns NO item semantics. `DatasetEnv` turns an item into an `ObsBundle`, `DatasetPreferenceEnv` into
a preference `Trajectory`; both traverse identically, and before this module they each said so in
their own code and disagreed (`DatasetEnv` reshuffled every epoch, `GSM8KEnv` shuffled once).

SHUFFLE THEN SHARD -- A DELIBERATE CHANGE OF SEMANTICS. `DatasetEnv` used to shard at
materialization and then shuffle WITHIN the replica's own rows, using one persistent RNG across
epochs, so a replica owned the same rows forever. This permutes the WHOLE corpus with `seed + epoch`
and strides afterwards, which is what `DistributedSampler` does and what DPO already did. Coverage is
disjoint and complete either way, but two things differ: rows now MIGRATE between replicas across
epochs, and the per-replica sample ORDER is not what it was. The persistent RNG is what made the old
scheme unresumable -- epoch N's order depended on every draw before it -- so this is the change that
buys exact resume. The GSM8K recipe pins `shuffle=False`, so the seeded paired-run evidence on this
branch does NOT exercise this path.

THE SHARD AXIS IS A PARAMETER, NEVER INFERRED. The caller supplies it because only the caller knows
which mesh it is in:

  - DPO passes the FSDP RANK. It is offline; the rank axis is the only data-parallel axis and
    nothing downstream divides it.
  - GRPO passes its data worker-group's REPLICA index. Ranks inside a replica are SPMD mirrors that
    must see the SAME prompts: `HFRollout.generate` runs until every sequence hits EOS or
    `max_new_tokens`, so the decode-step count is data-dependent, each step is a module forward, and
    FSDP hangs its parameter all-gathers off the forward. Different prompts per rank means different
    collective counts, which HANGS rather than errors. GRPO divides its rank axis later, in
    `workers/dispatch.py::select_chunk`, where the batch is static and unequal counts are
    repairable by padding.

Nor may coordinates be defaulted from `replica_coords()`: that publishes the COLOCATION's replica
count (the max over its members), so in a fan-out (`data(replicas=1) -> rollout(replicas=2)`) a
single data actor would keep `items[0::2]` and silently drop half the corpus.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint import CheckpointManager

_CURSOR_VERSION = 1


class CursorStateUnavailable(RuntimeError):
    """The source cannot represent an exact resumable cursor."""


def _require_exact_int(value: Any, *, field: str) -> int:
    """An integer checkpoint field, rejecting bools and refusing coercion.

    Args:
        value: the raw value read from a checkpoint.
        field: name used in the error message.

    Returns:
        `value` unchanged.

    Raises:
        ValueError: `value` is not exactly an `int` (a `bool` is not accepted).
    """
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


class IndexedCursor:
    """Deterministic, shardable, resumable traversal of a map-style dataset.

    Args:
        dataset: map-style source — `__len__` and `__getitem__`. Iterable-only sources are refused:
            they have no stable item identity, hence no checkpointable cursor.
        shard_index: this shard's index. See the module docstring for which axis to pass.
        shard_count: number of shards. `1` disables sharding.
        shuffle: reshuffle the index order at every epoch boundary.
        seed: base seed; epoch `e` permutes with `seed + e`, identically on every shard.
        cursor_resume: `"auto"` | `"strict"` | `"off"` (see `config/rl.py`).
        label: prefix for error and state messages, so a caller's existing messages are preserved.
        coord_names: what to CALL the two coordinates in error messages. DPO shards by FSDP rank
            and its messages (and tests) say `rank`/`world_size`; GRPO shards by replica and says
            `shard_index`/`shard_count`. The axis differs, so the wording must follow the caller.

    Raises:
        TypeError: `dataset` is not map-style.
        ValueError: inconsistent shard coordinates.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        shard_index: int = 0,
        shard_count: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        cursor_resume: str = "auto",
        label: str = "IndexedCursor",
        coord_names: tuple[str, str] = ("shard_index", "shard_count"),
    ) -> None:
        if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise TypeError(
                f"{label} requires a map-style dataset (__len__ + __getitem__); got "
                f"{type(dataset).__name__}. An iterable-only source has no stable item identity and "
                f"therefore no checkpointable cursor."
            )
        idx_name, cnt_name = coord_names
        if shard_count < 1 or not (0 <= shard_index < shard_count):
            raise ValueError(
                f"{label}: {idx_name}={shard_index}, {cnt_name}={shard_count} — "
                f"need 0 <= {idx_name} < {cnt_name} and {cnt_name} >= 1"
            )
        self._dataset = dataset
        self._shard_index = shard_index
        self._shard_count = shard_count
        self._shuffle = shuffle
        self._seed = seed
        self._cursor_resume = cursor_resume
        self._label = label
        self._coord_names = coord_names
        self._epoch = 0
        self._order: list[int] = []
        self._pos = 0
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """Total items yielded since construction (survives resume)."""
        return self._consumed

    def _rebuild_order(self) -> None:
        """Permute indices identically on every shard, then keep this shard's stride.

        Fails loud on a shard that cannot be served rather than padding/duplicating as
        `DistributedSampler` does — padding hides the misconfiguration.

        Raises:
            ValueError: the dataset is empty, or smaller than the shard count.
        """
        n = len(self._dataset)
        if n == 0:
            raise ValueError(f"{self._label}: dataset is empty — nothing to sample.")
        idx_name, cnt_name = self._coord_names
        if n < self._shard_count:
            raise ValueError(
                f"{self._label}: dataset has {n} item(s) but {cnt_name}={self._shard_count} — "
                f"{idx_name} {self._shard_index} would get an empty shard. Provide >= {cnt_name} "
                f"items, or reduce the data-parallel degree."
            )
        idx = list(range(n))
        if self._shuffle:  # identical permutation on every shard, then this shard's stride
            random.Random(self._seed + self._epoch).shuffle(idx)
        self._order = idx[self._shard_index :: self._shard_count]
        self._pos = 0
        self._epoch += 1  # NOTE: incremented AFTER building; restore compensates with epoch - 1

    def next_index(self) -> int:
        """The next dataset index for this shard, wrapping (and reshuffling) at the epoch boundary.

        Returns:
            An index into `dataset`.
        """
        if self._pos >= len(self._order):
            self._rebuild_order()
        i = self._order[self._pos]
        self._pos += 1
        self._consumed += 1
        return i

    def advance(self, count: int) -> None:
        """Skip the first `count` draws so the next `next_index()` is draw `count + 1` of this seed's sequence.

        This is the exact-continuation primitive for a run started from a MODEL-ONLY artifact, which
        carries no cursor state: a fresh cursor with the source run's `seed`, `shuffle` and shard
        coordinates, advanced by the number of draws that shard had consumed, continues the source
        run's prompt sequence instead of re-serving it from its first prompt. Epoch wraps are honoured
        (the skipped draws walk `_rebuild_order` exactly as the source run did), and `consumed` counts
        them, so `state_dict()` after an advance equals the source run's cursor state at that draw.

        Args:
            count: draws to skip; `0` is a no-op.

        Raises:
            ValueError: `count` is negative or not an exact integer (a `bool` is refused).
        """
        if type(count) is not int or count < 0:
            raise ValueError(f"{self._label}: advance count must be a non-negative integer, got {count!r}")
        for _ in range(count):
            self.next_index()

    # ----------------------------------------------------------------- resume
    def _dataset_cursor_hooks(
        self,
    ) -> tuple[Callable[[], Any] | None, Callable[[dict[str, Any]], Any] | None]:
        """The dataset's optional `(save, load)` cursor pair, validated as all-or-nothing.

        Returns:
            `(save, load)`, each callable or None.

        Raises:
            ValueError: exactly one of the two is defined — a source that can snapshot its
                stochastic preprocessing but not restore it cannot resume exactly, and the
                asymmetry is a wiring bug rather than a capability level.
        """
        save = getattr(self._dataset, "cursor_state_dict", None)
        load = getattr(self._dataset, "load_cursor_state_dict", None)
        if callable(save) != callable(load):
            raise ValueError(
                f"{self._label} exact cursor persistence requires both dataset cursor save and restore methods"
            )
        return (save if callable(save) else None), (load if callable(load) else None)

    def _fingerprint(self) -> str:
        """The dataset's identity claim; required for exact resume.

        Returns:
            The source's `cursor_fingerprint()`.

        Raises:
            CursorStateUnavailable: the source cannot identify itself, so a restored position
                could silently point into different data.
        """
        fn = getattr(self._dataset, "cursor_fingerprint", None)
        if not callable(fn):
            raise CursorStateUnavailable(
                f"{self._label} exact cursor persistence requires dataset.cursor_fingerprint()"
            )
        return str(fn())

    def state_dict(self) -> dict[str, Any] | None:
        """Snapshot the exact next-item position.

        Returns:
            The cursor state, or None when `cursor_resume='off'` disables persistence.

        Raises:
            CursorStateUnavailable: the dataset cannot identify itself.
            ValueError: the dataset defines only one half of the cursor save/restore pair, or
                returns a non-dict state.
        """
        if self._cursor_resume == "off":
            return None
        save, _ = self._dataset_cursor_hooks()
        dataset_state = save() if save is not None else None
        if dataset_state is not None and not isinstance(dataset_state, dict):
            raise ValueError(f"{self._label}: dataset.cursor_state_dict() must return a state dictionary")
        return {
            "version": _CURSOR_VERSION,
            # LEGACY KEY NAMES. Persisted in existing DPO checkpoints; the Python parameters are
            # shard_index/shard_count but the on-disk contract is deliberately NOT renamed.
            "rank": self._shard_index,
            "world_size": self._shard_count,
            "shuffle": self._shuffle,
            "seed": self._seed,
            "dataset_len": len(self._dataset),
            "dataset_fingerprint": self._fingerprint(),
            "epoch": self._epoch,
            "position": self._pos,
            "dataset_state": copy.deepcopy(dataset_state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore a snapshot after validating source and sharding identity.

        Args:
            state: a snapshot produced by `state_dict()`.

        Returns:
            True when the cursor was restored; False when `cursor_resume='off'` declined it.

        Raises:
            ValueError: the snapshot describes a different dataset, seed, or shard topology, or
                its counters are malformed.
            CursorStateUnavailable: the dataset cannot identify itself.
        """
        if self._cursor_resume == "off":
            return False
        version = _require_exact_int(state.get("version"), field=f"{self._label} cursor version")
        if version != _CURSOR_VERSION:
            raise ValueError(f"unsupported {self._label} cursor version {version}")
        for key in ("rank", "world_size", "seed", "dataset_len"):
            _require_exact_int(state.get(key), field=f"{self._label} cursor {key}")
        if type(state.get("shuffle")) is not bool:
            raise ValueError(f"{self._label} cursor shuffle must be a boolean")
        expected = {
            "rank": self._shard_index,
            "world_size": self._shard_count,
            "shuffle": self._shuffle,
            "seed": self._seed,
            "dataset_len": len(self._dataset),
            "dataset_fingerprint": self._fingerprint(),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"{self._label} cursor {key}={state.get(key)!r}, expected {value!r}")
        epoch = _require_exact_int(state.get("epoch"), field=f"{self._label} cursor epoch")
        position = _require_exact_int(state.get("position"), field=f"{self._label} cursor position")
        if epoch < 0 or position < 0:
            raise ValueError(f"{self._label} cursor counters must be non-negative")
        if epoch == 0:
            if position != 0:
                raise ValueError(f"{self._label} cursor position must be zero before the first epoch")
            order: list[int] = []
        else:
            full = list(range(len(self._dataset)))
            if self._shuffle:
                # epoch - 1: `_rebuild_order` increments AFTER permuting, so the order in force at
                # `epoch` was built with `seed + epoch - 1`. Off by one here silently resumes into a
                # DIFFERENT permutation while every validity check above still passes.
                random.Random(self._seed + epoch - 1).shuffle(full)
            order = full[self._shard_index :: self._shard_count]
        if position > len(order):
            raise ValueError(f"{self._label} cursor position exceeds its shard order")
        _, load = self._dataset_cursor_hooks()
        dataset_state = state.get("dataset_state")
        if load is not None:
            if not isinstance(dataset_state, dict):
                raise ValueError(f"{self._label} cursor is missing required dataset-local state")
            load(copy.deepcopy(dataset_state))
        elif dataset_state is not None:
            raise ValueError(f"{self._label} cursor has unexpected dataset-local state")
        self._epoch, self._order, self._pos = epoch, order, position
        # Recomputed, never serialized: adding a key to a versioned schema would silently break
        # checkpoints written by the previous format while `version` still claimed 1.
        order_len = len(range(self._shard_index, len(self._dataset), self._shard_count))
        self._consumed = 0 if epoch == 0 else (epoch - 1) * order_len + position
        return True


def restore_env_cursor(
    env: Any,
    checkpoint: str,
    cursor_resume: str,
    *,
    checkpoint_manager: CheckpointManager | None = None,
) -> None:
    """Restore one rank-local env cursor with stable checkpoint provenance."""
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.config.rl import CURSOR_RESUME_OFF, CURSOR_RESUME_STRICT
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged
    from cosmos_framework.posttrain.utils.log import info0, logger
    from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank

    manager = checkpoint_manager or CheckpointManager()

    def restore_cursor() -> None:
        if cursor_resume == CURSOR_RESUME_OFF:
            logger.warning(f"data cursor restore is disabled; source restarts non-exactly from {checkpoint}")
        else:

            def select_cursor_state() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
                data_component = manager.load_component(checkpoint, "data")
                if data_component is None and cursor_resume == CURSOR_RESUME_STRICT:
                    raise ValueError("strict data-cursor resume requires a data checkpoint component")
                data_state = state_for_current_rank(
                    data_component,
                    on_world_size_mismatch=("raise" if cursor_resume == CURSOR_RESUME_STRICT else "skip"),
                )
                if data_state is None and cursor_resume == CURSOR_RESUME_STRICT:
                    raise ValueError("strict data-cursor resume requires a compatible rank state")
                return data_component, data_state

            data_component, data_state = run_rank_converged(
                select_cursor_state,
                "preference source checkpoint selection",
            )
            if data_state is None:
                reason = (
                    "checkpoint has no data cursor"
                    if data_component is None
                    else "data cursor topology differs from the current world size"
                )
                logger.warning(f"{reason}; source restarts non-exactly from {checkpoint}")
            else:
                restored = run_rank_converged(
                    lambda: bool(env.load_state_dict(data_state)),
                    "preference cursor state application",
                )
                restored = rank_identical(restored, "preference cursor restore outcome")
                if restored:
                    info0(f"restored data cursor from {checkpoint}")

    _value, loaded_provenance = manager.load_checkpoint(checkpoint, restore_cursor)
    if loaded_provenance is not None:
        env.set_loaded_checkpoint_provenance(loaded_provenance)
