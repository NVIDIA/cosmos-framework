# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the ``apply_long_sample_halving`` knob added to
:class:`cosmos_framework.data.vfm.packing_iterable_dataset.PackingIterableDataset` with
the wandb + DataPacker spec (2026-05-21-toml-interface-wandb-datapacker-design.md).
"""

from __future__ import annotations

import torch

from cosmos_framework.data.vfm.packing_iterable_dataset import PackingIterableDataset


class _StubIterable(torch.utils.data.IterableDataset):
    """Trivial finite IterableDataset; the halving tests don't iterate it."""

    def __iter__(self):
        yield from ()


class _StubPacking(PackingIterableDataset):
    """Minimal concrete subclass; ``_max_tokens`` is the SUT, so we just need
    a constructable instance — ``compute_sample_tokens`` isn't called by these
    tests."""

    def compute_sample_tokens(self, sample: dict) -> int:  # pragma: no cover - unused
        return 0


def _make(apply_long_sample_halving: bool = True) -> _StubPacking:
    return _StubPacking(
        datasets_cfg={"default": {"dataset": _StubIterable(), "ratio": 1.0}},
        max_tokens=45056,
        pool_size=16,
        max_batch_size=1,
        long_threshold=6400,
        batching_strategy="prefer_closest",
        apply_long_sample_halving=apply_long_sample_halving,
    )


# ----- halving heuristic ----------------------------------------------------


def test_default_applies_halving_above_threshold():
    """Default behavior: cur_max >= 1000 triggers ``max_tokens // 2``."""
    ds = _make()
    assert ds.apply_long_sample_halving is True
    assert ds._max_tokens(999) == 45056  # below threshold → full budget
    assert ds._max_tokens(1000) == 22528  # at threshold → halved
    assert ds._max_tokens(5000) == 22528  # well above → halved


def test_halving_disabled_keeps_full_budget():
    """``apply_long_sample_halving=False`` returns ``max_tokens`` literally."""
    ds = _make(apply_long_sample_halving=False)
    assert ds.apply_long_sample_halving is False
    assert ds._max_tokens(999) == 45056
    assert ds._max_tokens(1000) == 45056  # would have been halved with default
    assert ds._max_tokens(50_000) == 45056


def test_halving_default_is_true_when_unspecified():
    """Backwards compat: constructing without the new kwarg keeps the original
    (halving-active) behavior bit-for-bit — every existing recipe is unchanged.
    """
    ds = _StubPacking(
        datasets_cfg={"default": {"dataset": _StubIterable(), "ratio": 1.0}},
        max_tokens=10_000,
        pool_size=16,
        max_batch_size=1,
        long_threshold=6400,
        batching_strategy="prefer_closest",
    )
    assert ds.apply_long_sample_halving is True
    assert ds._max_tokens(2000) == 5000
