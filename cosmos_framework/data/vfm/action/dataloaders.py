# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.utils.data

from cosmos_framework.utils import distributed, log
from cosmos_framework.data.vfm.action.unified_dataset import ActionUnifiedIterableDataset
from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn

# DataLoader kwargs that are safe to patch on a per-family basis from
# ``per_family_overrides``.  Anything outside this set would either change
# the dataset identity, the seeding contract, or break PyTorch's
# expectations.
_ALLOWED_PER_FAMILY_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"batch_size", "num_workers", "prefetch_factor", "persistent_workers", "pin_memory"}
)


def _action_worker_init_fn(
    worker_id: int, seed: int = 42, use_deterministic_seed: bool = True, rank: int = 0, world_size: int = 1
) -> None:
    # DataLoader workers use spawn, so parent-process monkey patches do not
    # carry over. Apply the LeRobot decoder-cache cap inside each worker.
    from cosmos_framework.data.vfm.action.cosmos3_action_lerobot import _patch_decoder_cache

    _patch_decoder_cache()

    if use_deterministic_seed:
        worker_seed = seed + rank * 9999 + worker_id
    else:
        worker_seed = int(time.time() * 1000) % (2**32) + rank * 9999 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)

    info = torch.utils.data.get_worker_info()
    assert info is not None
    dataset = info.dataset
    if isinstance(dataset, ActionUnifiedIterableDataset):
        dataset.assign_worker(worker_id, info.num_workers, rank, world_size)


def create_action_worker_init_fn(seed: int = 42, use_deterministic_seed: bool = True) -> Callable[[int], None]:
    """Create a worker_init_fn for Action training with ``ActionUnifiedIterableDataset``.

    Seeds RNGs first, then calls ``dataset.assign_worker()`` to set up
    rank-level dataset assignment and worker-level shard distribution.

    Passed to ``DataLoader`` (or ``InfiniteDataLoader``) as the
    ``worker_init_fn`` parameter.  Only called when ``num_workers > 0``.

    Args:
        seed: Base seed for deterministic worker seeding.  Ignored when
            ``use_deterministic_seed=False`` (time-based seed used instead).
        use_deterministic_seed: If True, use the provided seed for reproducible
            RNG initialization. If False, derive a time-based seed so that
            each resume sees different data. This is preferred for large-scale
            runs that resume frequently, and when ``in_order=False`` already
            makes iteration order non-deterministic.

    Returns:
        A ``worker_init_fn`` suitable for ``torch.utils.data.DataLoader``.
    """
    try:
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    except RuntimeError:
        rank = 0
        world_size = 1

    return functools.partial(
        _action_worker_init_fn,
        seed=seed,
        use_deterministic_seed=use_deterministic_seed,
        rank=rank,
        world_size=world_size,
    )


def _apply_per_family_overrides(
    args: tuple,
    kwargs: dict,
    per_family_overrides: dict[str, dict[str, Any]],
) -> dict:
    """Patch DataLoader kwargs for this rank based on its action family.

    Each global rank serves exactly one action family under Hare-Niemeyer
    allocation (see :meth:`ActionUnifiedIterableDataset.assign_worker`).
    This lets us tune ``batch_size`` / ``prefetch_factor`` / ``num_workers``
    / ``persistent_workers`` / ``pin_memory`` per family, even though they
    share a single top-level ``action_data`` dataloader entry.

    Heavy families (``camera_720``, ``hand_pose``, ``agibotworld_beta``)
    can shrink batch / prefetch to control host RSS, while small-token
    high-IO families (``droid``, ``fractal``, ``bridge``) can expand the
    in-flight buffer to absorb shard / page-cache stalls.

    No-ops (returns ``kwargs`` unchanged with a warning) when:

    * the dataset is not an :class:`ActionUnifiedIterableDataset`
      (overrides are routed by family name, which only that class exposes);
    * ``shard_across_workers=False`` — every worker iterates all families
      via weighted random selection, so per-family override would be
      meaningless;
    * the resolved family has no entry in ``per_family_overrides``.

    Args:
        args: Positional ``__init__`` args of the surrounding DataLoader.
            ``args[0]`` is ``dataset`` when the caller used positional form.
        kwargs: Keyword args of the surrounding DataLoader; mutated via
            ``.update()`` with the resolved overrides.
        per_family_overrides: Mapping from family name (matching
            ``dataset_entry(name=...)``) to a dict of override kwargs.

    Returns:
        The (possibly updated) ``kwargs`` dict.

    Raises:
        ValueError: If an override dict contains keys outside
            :data:`_ALLOWED_PER_FAMILY_OVERRIDE_KEYS`.
    """
    dataset = kwargs.get("dataset", args[0] if args else None)
    if not isinstance(dataset, ActionUnifiedIterableDataset):
        log.warning(
            "InfiniteDataLoader: per_family_overrides ignored - dataset is "
            f"{type(dataset).__name__}, not ActionUnifiedIterableDataset."
        )
        return kwargs
    if not dataset._shard_across_workers:
        log.warning(
            "InfiniteDataLoader: per_family_overrides ignored - "
            "shard_across_workers=False (every worker iterates all families)."
        )
        return kwargs

    try:
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    except RuntimeError:
        rank, world_size = 0, 1

    # Validate override keys up-front so a typo fails loudly instead of
    # silently doing nothing on most ranks.
    for fam, ov in per_family_overrides.items():
        bad = set(ov) - _ALLOWED_PER_FAMILY_OVERRIDE_KEYS
        if bad:
            raise ValueError(
                f"per_family_overrides[{fam!r}] has unsupported keys {sorted(bad)}; "
                f"allowed: {sorted(_ALLOWED_PER_FAMILY_OVERRIDE_KEYS)}"
            )

    known_families = {entry.get("name", f"family_{i}") for i, entry in enumerate(dataset._datasets)}
    unknown = set(per_family_overrides) - known_families
    if unknown and rank == 0:
        log.warning(
            f"InfiniteDataLoader: per_family_overrides has unknown families "
            f"{sorted(unknown)}; known: {sorted(known_families)}"
        )

    if world_size < len(dataset._datasets):
        # Hare-Niemeyer requires world_size >= num_families.  Skip
        # per-family overrides in tiny test runs rather than crashing.
        log.warning(
            f"InfiniteDataLoader: per_family_overrides ignored - "
            f"world_size ({world_size}) < num_families ({len(dataset._datasets)})."
        )
        return kwargs

    _, family_name = ActionUnifiedIterableDataset.resolve_family_for_rank(dataset._datasets, rank, world_size)

    overrides = per_family_overrides.get(family_name)
    if not overrides:
        return kwargs

    base_snapshot = {k: kwargs.get(k) for k in overrides}

    # Guard against PyTorch invariants when an override sets num_workers=0.
    new_num_workers = overrides.get("num_workers", kwargs.get("num_workers", 0))
    if new_num_workers == 0:
        # PyTorch errors if prefetch_factor / persistent_workers are set
        # while num_workers=0; clear them to keep the override safe.
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)

    kwargs.update(overrides)

    log.info(
        f"InfiniteDataLoader: rank={rank}/{world_size} family={family_name!r} "
        f"applying per_family_overrides {overrides} (was {base_snapshot})",
        rank0_only=False,
    )
    return kwargs


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """A dataloader that yields forever with proper seeding for reproducibility.

    All Action datasets are ``IterableDataset`` instances (map-style datasets
    are automatically wrapped by :class:`~.transforms.MapToIterableAdapter`).
    The loader catches ``StopIteration`` and restarts the iterator so that
    iteration never ends.
    """

    def __init__(
        self,
        *args,
        seed: int = 42,
        use_deterministic_seed: bool = True,
        per_family_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Initialize InfiniteDataLoader.

        Args:
            *args: Positional arguments passed to parent DataLoader.
            seed: Random seed for reproducible worker initialization.
                  Default is 42 for reproducibility.
            use_deterministic_seed: If True, use the provided seed for reproducible
                  RNG initialization. If False, derive a time-based seed so that
                  each resume sees different data. This is preferred for large-scale
                  runs that resume frequently, and when ``in_order=False`` already
                  makes iteration order non-deterministic.
            per_family_overrides: Optional ``{family_name: {kwarg: value, ...}}``
                  mapping that patches DataLoader kwargs for whatever action
                  family this rank is assigned to under Hare-Niemeyer.  Only
                  meaningful when the wrapped dataset is an
                  :class:`ActionUnifiedIterableDataset` with
                  ``shard_across_workers=True``.  Allowed keys per family:
                  ``batch_size``, ``num_workers``, ``prefetch_factor``,
                  ``persistent_workers``, ``pin_memory``.

                  Each rank serves exactly one family, so this lets heavy
                  families (``camera_720``, ``hand_pose``) shrink their
                  in-flight buffer to control host RSS, and small-token
                  high-IO families (``droid``, ``fractal``) expand it to
                  hide shard / page-cache stalls.  Family names must match
                  ``dataset_entry(name=...)`` in
                  :func:`~.unified_dataset.wrap_dataset`'s ``list_of_datasets``.

                  Example::

                      per_family_overrides={
                          "camera_720_20260501":       dict(batch_size=1, prefetch_factor=1),
                          "hand_pose_20260501":        dict(batch_size=2, prefetch_factor=1),
                          "agibotworld_beta_20260501": dict(batch_size=2, prefetch_factor=1),
                          "fractal_20260501":          dict(batch_size=8, prefetch_factor=3),
                          "droid_20260501":            dict(batch_size=4, prefetch_factor=3),
                          "bridge_20260501":           dict(batch_size=4, prefetch_factor=3),
                      }
            **kwargs: Keyword arguments passed to parent DataLoader.
        """
        kwargs.pop("shuffle", None)
        kwargs["shuffle"] = False

        # Default to ``custom_collate_fn`` so that variable-length per-sample
        # tensors (e.g. ``text_token_ids``) and multi-item keys (``video``,
        # ``action``, ...) are returned as lists rather than stacked by
        # PyTorch's ``default_collate``.
        if kwargs.get("collate_fn") is None:
            kwargs["collate_fn"] = custom_collate_fn

        # Apply per-family overrides BEFORE PyTorch sees the kwargs so the
        # rank-specific batch_size / prefetch_factor / num_workers take
        # effect at the DataLoader / worker_pool level, not just at runtime.
        if per_family_overrides:
            kwargs = _apply_per_family_overrides(args, kwargs, per_family_overrides)

        if "worker_init_fn" not in kwargs or kwargs["worker_init_fn"] is None:
            kwargs["worker_init_fn"] = create_action_worker_init_fn(seed, use_deterministic_seed=use_deterministic_seed)

        num_workers = kwargs.get("num_workers", 0)
        if num_workers == 0:
            try:
                rank = distributed.get_rank()
            except RuntimeError:
                rank = 0
            if use_deterministic_seed:
                rank_seed = seed + rank * 9999
            else:
                rank_seed = int(time.time() * 1000) % (2**32) + rank * 9999
            random.seed(rank_seed)
            np.random.seed(rank_seed % (2**32))
            torch.manual_seed(rank_seed)

        super().__init__(*args, **kwargs)
        self._stream_iterator: Iterator | None = None

    def __len__(self) -> int:
        # Delegate to DataLoader which calls len(self.dataset).
        # Raises TypeError if the underlying dataset has no __len__.
        return super().__len__()

    def __iter__(self) -> Iterator:
        """Yield batches forever."""
        while True:
            if self._stream_iterator is None:
                self._stream_iterator = super().__iter__()
            try:
                yield next(self._stream_iterator)  # type: ignore[arg-type]
            except StopIteration:
                self._stream_iterator = super().__iter__()
                yield next(self._stream_iterator)  # type: ignore[arg-type]
