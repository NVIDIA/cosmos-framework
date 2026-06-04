# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for SimpleBatcher."""

from __future__ import annotations

from cosmos_framework.data.vfm.dataflow.batchers import SimpleBatcher


def _src(n):
    return iter([{"i": i} for i in range(n)])


def test_simple_batcher_groups_in_fixed_size():
    groups = list(SimpleBatcher(batch_size=3).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5], [6]]


def test_simple_batcher_drop_last_discards_partial():
    groups = list(SimpleBatcher(batch_size=3, drop_last=True).batches(_src(7)))
    assert [[s["i"] for s in g] for g in groups] == [[0, 1, 2], [3, 4, 5]]


def test_simple_batcher_empty_source_yields_nothing():
    assert list(SimpleBatcher(batch_size=3).batches(_src(0))) == []
