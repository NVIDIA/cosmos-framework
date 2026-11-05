# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor


@pytest.mark.L0
@pytest.mark.CPU
def test_shards_are_disjoint_and_cover_the_dataset():
    """Every replica seeds identically, so the permutation matches and the strides never overlap."""
    rows = list(range(12))

    def _take(shard_index, n=4):
        # ONE cursor per shard, drawn n times. Constructing it inside the draw loop would rebuild
        # the cursor every call and yield its first item n times over.
        cur = IndexedCursor(rows, shard_index=shard_index, shard_count=3, shuffle=True, seed=7)
        return [cur.next_index() for _ in range(n)]

    sets = [set(_take(i)) for i in range(3)]
    assert all(len(s) == 4 for s in sets), f"each shard holds 4 of 12: {sets}"
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), f"shards overlap: {sets}"
    assert set().union(*sets) == set(rows), "shards must cover the dataset"


@pytest.mark.L0
@pytest.mark.CPU
def test_single_shard_is_the_whole_dataset_in_order():
    cur = IndexedCursor(list(range(5)), shard_index=0, shard_count=1, shuffle=False)
    assert [cur.next_index() for _ in range(5)] == [0, 1, 2, 3, 4]


@pytest.mark.L0
@pytest.mark.CPU
def test_epoch_wraps_and_reshuffles():
    """Epoch 2 must not repeat epoch 1's order, or a long run walks one fixed sequence forever.

    Asserts on the INDEX sequence -- the cursor never reads the data, and these sources are
    identity-mapped (`dataset[i] == i`), so the values are the indices.
    """
    cur = IndexedCursor(list(range(8)), shuffle=True, seed=3)
    first = [cur.next_index() for _ in range(8)]
    second = [cur.next_index() for _ in range(8)]
    assert sorted(first) == sorted(second) == list(range(8)), "each epoch covers the dataset once"
    assert first != second, "a new epoch must reshuffle"


@pytest.mark.L0
@pytest.mark.CPU
def test_consumed_counts_every_item_across_epochs():
    cur = IndexedCursor(list(range(3)))
    for _ in range(7):
        cur.next_index()
    assert cur.consumed == 7


@pytest.mark.L0
@pytest.mark.CPU
def test_unservable_shard_and_empty_dataset_are_refused():
    with pytest.raises(ValueError, match="nothing to sample"):
        IndexedCursor([]).next_index()
    with pytest.raises(ValueError, match="would get an empty shard"):
        IndexedCursor([0], shard_index=1, shard_count=2).next_index()
    with pytest.raises(ValueError, match="need 0 <= shard_index < shard_count"):
        IndexedCursor([0, 1], shard_index=2, shard_count=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_iterable_only_dataset_is_refused():
    """No __len__/__getitem__ means no checkpointable cursor; refuse rather than silently unshard."""
    with pytest.raises(TypeError, match="map-style"):
        IndexedCursor(iter([1, 2, 3]))


class _Fingerprinted(list):
    """Map-style AND identity-bearing. `IndexedCursor.state_dict` REQUIRES
    `cursor_fingerprint()` — a bare list raises `CursorStateUnavailable`, which is the
    correct behaviour (DPO's 'auto' mode degrades on exactly that), so resume tests must
    supply a source that can identify itself."""

    def cursor_fingerprint(self) -> str:
        return f"len={len(self)}"


def _fresh(n_items=10, **kw):
    return IndexedCursor(_Fingerprinted(range(n_items)), shuffle=True, seed=5, **kw)


def _drained(n_items=10, taken=3, **kw):
    cur = _fresh(n_items, **kw)
    for _ in range(taken):
        cur.next_index()
    return cur


@pytest.mark.L0
@pytest.mark.CPU
def test_state_dict_requires_an_identifiable_dataset():
    """A source that cannot identify itself cannot be resumed EXACTLY; say so rather than guess."""
    from cosmos_framework.posttrain.roles.env.cursor import CursorStateUnavailable

    with pytest.raises(CursorStateUnavailable, match="cursor_fingerprint"):
        IndexedCursor([1, 2, 3]).state_dict()


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_round_trips_the_position():
    src = _drained()
    rest = list(range(7))
    expected = [src.next_index() for _ in rest]

    restored = _fresh()
    assert restored.load_state_dict(_drained().state_dict()) is True
    assert [restored.next_index() for _ in rest] == expected, "resume must continue, not restart"
    # the INDEX stream is the whole contract here: the cursor hands out positions, and the env
    # is what turns one into a row.
    assert restored.consumed == 10


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_uses_legacy_key_names():
    """`rank`/`world_size` are persisted in existing DPO checkpoints; renaming them breaks resume."""
    state = _drained().state_dict()
    assert {"rank", "world_size"} <= set(state), f"legacy keys must survive: {sorted(state)}"
    assert state["version"] == 1


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "key,value",
    [("seed", 999), ("world_size", 4), ("dataset_len", 99), ("shuffle", False)],
)
def test_resume_rejects_a_changed_topology(key, value):
    """A changed shard count, seed or dataset must fail loud, not resume into a different split."""
    state = {**_drained().state_dict(), key: value}
    cur = _fresh()
    with pytest.raises(ValueError, match=key):
        cur.load_state_dict(state)


@pytest.mark.L0
@pytest.mark.CPU
def test_resume_off_declines_without_raising():
    cur = _fresh(cursor_resume="off")
    assert cur.state_dict() is None
    assert cur.load_state_dict(_drained().state_dict()) is False
