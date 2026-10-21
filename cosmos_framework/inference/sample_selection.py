# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic sample selection helpers.

Kept self-contained so this package does not need optional evaluation
dependencies for dataset sampling.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

SampleSelectionStrategy = Literal["first", "deterministic_random"]

DEFAULT_SAMPLE_SELECTION_STRATEGY: SampleSelectionStrategy = "deterministic_random"
DEFAULT_SAMPLE_SELECTION_SEED = 42


def _stable_index_key(*, namespace: str, seed: int, index: int) -> str:
    data = f"{namespace}\0{seed}\0{index}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def select_limited_indices(
    candidates: Iterable[int],
    *,
    limit: int | None,
    strategy: SampleSelectionStrategy = DEFAULT_SAMPLE_SELECTION_STRATEGY,
    seed: int = DEFAULT_SAMPLE_SELECTION_SEED,
    namespace: str = "",
) -> list[int]:
    """Select a bounded set of indices while preserving candidate order.

    ``deterministic_random`` ranks candidates with a stable SHA-256 key, takes
    the best ``limit`` entries, then returns that selected set in the original
    candidate order. This gives a reproducible random subset without changing
    downstream access order more than necessary.
    """
    candidate_list = list(candidates)
    if limit is None:
        return candidate_list
    if limit <= 0:
        return []
    if limit >= len(candidate_list):
        return candidate_list
    if strategy == "first":
        return candidate_list[:limit]
    if strategy != "deterministic_random":
        raise ValueError(f"Unsupported sample selection strategy: {strategy!r}")

    selected = set(
        sorted(
            candidate_list,
            key=lambda index: _stable_index_key(namespace=namespace, seed=seed, index=index),
        )[:limit]
    )
    return [index for index in candidate_list if index in selected]
