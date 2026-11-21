# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The rotation helper. Its failures are invisible: a plan that quietly omits whole episodes."""

from __future__ import annotations

from collections import Counter

from failuregen.balance import spread


def _items(primaries, secondaries):
    return [{"p": p, "s": s, "r": 1.0 - p * 0.01 - s * 0.001} for p in primaries for s in secondaries]


def test_every_primary_is_served_before_any_gets_seconds():
    items = _items(range(8), range(11))
    picked = spread(items, 8, primary=lambda i: i["p"], secondary=lambda i: i["s"], rank=lambda i: i["r"])
    assert sorted(i["p"] for i in picked) == list(range(8))


def test_a_short_take_still_reaches_the_last_primary():
    """The bug this exists to prevent: a flat cycle over pairs walks the sorted order and never
    reaches the last groups, so whole host episodes vanish from a dataset with no error."""
    items = _items(range(8), range(11))
    picked = spread(items, 43, primary=lambda i: i["p"], secondary=lambda i: i["s"], rank=lambda i: i["r"])
    counts = Counter(i["p"] for i in picked)
    assert set(counts) == set(range(8))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_secondaries_rotate_so_one_kind_does_not_dominate():
    items = _items(range(8), range(11))
    picked = spread(items, 8, primary=lambda i: i["p"], secondary=lambda i: i["s"], rank=lambda i: i["r"])
    assert len({i["s"] for i in picked}) == 8  # a different secondary leads on each primary


def test_offset_shifts_which_group_leads():
    items = _items(range(4), range(4))
    first = spread(items, 1, primary=lambda i: i["p"], secondary=lambda i: i["s"])
    shifted = spread(items, 1, primary=lambda i: i["p"], secondary=lambda i: i["s"], offset=2)
    assert first[0]["p"] != shifted[0]["p"]


def test_rank_decides_the_representative_of_a_cell():
    items = [{"p": 0, "s": 0, "r": 1.0}, {"p": 0, "s": 0, "r": 9.0}]
    picked = spread(items, 1, primary=lambda i: i["p"], secondary=lambda i: i["s"], rank=lambda i: i["r"])
    assert picked[0]["r"] == 9.0


def test_taking_more_than_exists_returns_everything_once():
    items = _items(range(3), range(2))
    picked = spread(items, 100, primary=lambda i: i["p"], secondary=lambda i: i["s"])
    assert len(picked) == len(items)
    assert len({(i["p"], i["s"]) for i in picked}) == len(items)


def test_degenerate_inputs_do_not_raise():
    assert spread([], 5, primary=lambda i: i) == []
    assert spread([{"p": 1}], 0, primary=lambda i: i["p"]) == []


def test_a_missing_secondary_key_is_tolerated():
    """Controls have no donor, so the secondary key is None and must not break sorting."""
    items = [{"p": 0, "s": None}, {"p": 0, "s": 3}, {"p": 1, "s": None}]
    picked = spread(items, 3, primary=lambda i: i["p"], secondary=lambda i: i["s"])
    assert len(picked) == 3
