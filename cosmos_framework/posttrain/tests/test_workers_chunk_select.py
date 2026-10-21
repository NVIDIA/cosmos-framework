# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The ingress selection predicate (spec §3.4): one owner for the mini, micro and dp axes."""

import itertools
from dataclasses import dataclass

import pytest

from cosmos_framework.posttrain.workers.dispatch import select_chunk


@dataclass
class _Item:
    group_id: str
    id: str
    # One item per episode, the shape text GRPO produces: `select_chunk` falls back to `id` and
    # each item is its own DP unit.
    session: str | None = None


@dataclass
class _LegacyItem:
    """The pre-feature duck type: selection historically required only these two fields."""

    group_id: str
    id: str


def _items(n: int) -> list[_Item]:
    """n fake trajectories with the two fields the canonical key reads.

    `group_id` is zero-padded so the canonical string sort agrees with construction order (unpadded
    ids like "g10" < "g2" sort differently from the order they were built in). That lets a test
    spell the canonical order as `sorted(...)` without restating the key.
    """
    return [_Item(group_id=f"g{i // 4:04d}", id=f"t{i:03d}") for i in range(n)]


def _legacy_select_chunk(
    items: list,
    *,
    mini: int,
    micro: int,
    n_mini: int,
    n_micro: int,
    dp_rank: int,
    dp_world: int,
) -> list:
    """The exact selector from the MR base, kept as the disabled-path oracle."""
    ordered = sorted(items, key=lambda item: (str(item.group_id), str(item.id)))
    mini_len = len(ordered) // n_mini
    chunk = ordered[mini * mini_len : (mini + 1) * mini_len]
    micro_len = mini_len // n_micro
    chunk = chunk[micro * micro_len : (micro + 1) * micro_len]
    if dp_world <= 1:
        return chunk
    return [item for index, item in enumerate(chunk) if index % dp_world == dp_rank]


@pytest.mark.L0
@pytest.mark.CPU
def test_selection_is_an_exact_partition() -> None:
    """Every item claimed by exactly ONE (rank, mini, micro) triple — never twice, never dropped.
    This is the 1/dp_world**2 regression test: a double-shard drops items and fails here."""
    # Both batch sizes are multiples of 64 = max(n_mini) * max(n_micro) * max(dp_world), so
    # `chunk_len % dp_world == 0` (the precondition select_chunk is documented to assume, per
    # spec §5.1) holds for every combination below and "equal shares, always" is achievable.
    cases = itertools.product([64, 128], [1, 2, 4], [1, 2, 4], [1, 2, 4])
    for batch, n_mini, n_micro, dp_world in cases:
        items = _items(batch)
        claimed: list[str] = []
        for rank, mini, micro in itertools.product(range(dp_world), range(n_mini), range(n_micro)):
            got = select_chunk(
                items,
                mini=mini,
                micro=micro,
                n_mini=n_mini,
                n_micro=n_micro,
                dp_rank=rank,
                dp_world=dp_world,
            )
            assert len(got) == batch // (n_mini * n_micro * dp_world)  # equal shares, always
            claimed.extend(item.id for item in got)
        assert sorted(claimed) == sorted(item.id for item in items)
        assert len(claimed) == len(set(claimed))


@pytest.mark.L0
@pytest.mark.CPU
def test_default_dp_axis_matches_the_legacy_rank_mapping() -> None:
    """Without the opt-in, each rank must retain its exact pre-feature item assignment."""
    items = _items(64)
    for dp_world in [1, 2, 4, 8]:
        for rank in range(dp_world):
            kwargs = dict(mini=0, micro=0, n_mini=1, n_micro=1, dp_rank=rank, dp_world=dp_world)
            got = select_chunk(items, **kwargs)
            expected = _legacy_select_chunk(items, **kwargs)
            assert [item.id for item in got] == [item.id for item in expected]


@pytest.mark.L0
@pytest.mark.CPU
def test_default_matches_legacy_for_every_axis_and_item_shape() -> None:
    """The disabled branch ignores new episode metadata and accepts the old duck type."""
    bare = [_LegacyItem(group_id=f"g{i // 4:04d}", id=f"t{i:03d}") for i in range(64)]
    episodic = [_Item(group_id=f"g{i // 4:04d}", id=f"t{i:03d}", session=f"session-{i // 2}") for i in range(64)]
    for items in (bare, episodic):
        for n_mini, n_micro, dp_world in itertools.product([1, 2, 4], repeat=3):
            for rank, mini, micro in itertools.product(range(dp_world), range(n_mini), range(n_micro)):
                kwargs = {
                    "mini": mini,
                    "micro": micro,
                    "n_mini": n_mini,
                    "n_micro": n_micro,
                    "dp_rank": rank,
                    "dp_world": dp_world,
                }
                got = select_chunk(items, **kwargs)
                expected = _legacy_select_chunk(items, **kwargs)
                assert [item.id for item in got] == [item.id for item in expected]


@pytest.mark.L0
@pytest.mark.CPU
def test_the_permutation_is_opt_in_deterministic_and_not_the_sort() -> None:
    """Three things, each of which fails silently on its own.

    Off by default, or an opt-in switch is not opt-in and every existing caller's training order
    moves under it. Deterministic when on, because ranks chunk the same whole batch independently
    and a permutation that differed between them would hand two ranks the same item and drop
    another -- no error, just a quietly wrong batch. And actually a permutation: one that
    degenerated back to the sort would restore the ordering it exists to break (an episode's steps
    share one advantage; consecutive optimizer steps must not all be that episode's), again with
    no symptom but a training curve.
    """
    items = _items(64)
    kwargs = dict(mini=0, micro=0, n_mini=1, n_micro=1, dp_rank=0, dp_world=1)
    canonical = sorted(item.id for item in items)

    # Default: the sort, untouched. This is the contract every caller that does not ask for a
    # permutation still gets, and it is what makes leaving `shuffle` off byte-identical.
    assert [item.id for item in select_chunk(items, **kwargs)] == canonical

    shuffled = [item.id for item in select_chunk(items, shuffle=True, **kwargs)]
    assert shuffled == [item.id for item in select_chunk(list(reversed(items)), shuffle=True, **kwargs)]
    assert shuffled != canonical
    assert sorted(shuffled) == canonical  # a permutation, not a filter


@pytest.mark.L0
@pytest.mark.CPU
def test_enabled_path_keeps_each_episode_on_one_dp_rank() -> None:
    """Episode-first partitioning is part of the opt-in and never leaks into the default path."""
    items = [_Item(group_id="g", id=f"t{i:03d}", session=f"session-{i // 2}") for i in range(8)]
    enabled = [
        select_chunk(
            items,
            mini=0,
            micro=0,
            n_mini=1,
            n_micro=1,
            dp_rank=rank,
            dp_world=2,
            shuffle=True,
        )
        for rank in range(2)
    ]
    owner: dict[str, int] = {}
    for rank, share in enumerate(enabled):
        for item in share:
            prior = owner.setdefault(str(item.session), rank)
            assert prior == rank
    assert set(owner) == {f"session-{i}" for i in range(4)}


@pytest.mark.L0
@pytest.mark.CPU
def test_short_batch_raises_instead_of_silently_dropping_the_tail() -> None:
    """A length not divisible by n_mini*n_micro would floor-divide its tail into oblivion. The
    conservation law keeps the batch exact today, but a short batch is structurally reachable
    (`ExperienceBuffer.select` slices `pool[:size]`) and the §5 dynamic-filter TODO would make it
    live — so the predicate must fail loud, not train on a silent 24/31 subset."""
    items = _items(31)  # 31 = 4*2*3 + 7; mini_len=7, micro_len=3 -> 4*2*3=24 used, 7 vanish
    with pytest.raises(ValueError, match="not a multiple of"):
        select_chunk(items, mini=0, micro=0, n_mini=4, n_micro=2, dp_rank=0, dp_world=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_multiple_does_not_raise() -> None:
    """The guard's boundary: a length that IS a multiple of n_mini*n_micro (here 24) is accepted,
    even when it is not separately a multiple of dp_world — the dp interleave never drops."""
    got = select_chunk(_items(24), mini=0, micro=0, n_mini=4, n_micro=2, dp_rank=0, dp_world=1)
    assert len(got) == 24 // (4 * 2)  # 3 items per chunk, tail-free
