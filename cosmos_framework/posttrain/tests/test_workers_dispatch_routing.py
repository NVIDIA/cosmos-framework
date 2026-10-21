# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.posttrain.workers.dispatch import (
    flat_units,
    route_sharded,
    split_index,
    units_for_replica,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_route_one_to_one():
    assert route_sharded(P=4, R_c=4) == ("one", 1)
    assert units_for_replica("one", 1, 4, 2) == [2]


def test_route_reduce_contiguous():
    assert route_sharded(P=4, R_c=2) == ("reduce", 2)
    assert units_for_replica("reduce", 2, 4, 0) == [0, 1]
    assert units_for_replica("reduce", 2, 4, 1) == [2, 3]


def test_route_split_shared_unit():
    assert route_sharded(P=1, R_c=2) == ("split", 2)
    assert units_for_replica("split", 2, 1, 0) == [0] == units_for_replica("split", 2, 1, 1)
    assert split_index("split", 2, 0) == 0 and split_index("split", 2, 1) == 1


def test_route_split_multi_unit():  # data (1,2) -> rollout (4,*): P=2, R_c=4 => s=2
    assert route_sharded(P=2, R_c=4) == ("split", 2)
    assert units_for_replica("split", 2, 2, 3) == [1]  # replica 3 <- unit 1, shard 1
    assert split_index("split", 2, 3) == 1


def test_route_non_divisible_raises():
    with pytest.raises(ValueError, match="3.*2|2.*3"):
        route_sharded(P=3, R_c=2)


@pytest.mark.parametrize("P, R_c", [(4, 0), (0, 4), (0, 0)])
def test_route_degenerate_counts_raise_not_zerodiv_or_silent(P, R_c):
    # R_c == 0 would ZeroDivisionError on `P % R_c`; P == 0 would satisfy `0 % R_c == 0` and route
    # ("reduce", 0) -> an empty ref list -> a SILENT empty batch. Both must fail loud naming counts.
    with pytest.raises(ValueError, match=r"must both be >= 1"):
        route_sharded(P=P, R_c=R_c)


def test_flat_units_row_major():
    assert flat_units((1, 2)) == [(0, 0), (0, 1)]
    assert flat_units((2, 1)) == [(0, 0), (1, 0)]


# --------------------------------------------------------------------------- §9.5 readers refcount


@pytest.mark.L0
@pytest.mark.CPU
def test_readers_for_select_divides_by_dp_world() -> None:
    """A kept item is pulled by the Kc/dp_world ranks sharing its model-DP coordinate."""
    from cosmos_framework.posttrain.workers.dispatch import readers_for

    assert readers_for(Kc=4, n=1, op="one", select=True) == 1  # dp_world defaults to Kc=4
    assert readers_for(Kc=4, n=2, op="reduce", select=True) == 1  # pure DP: 4 // 4
    # TP=2 in a future K=8 world: dp_world=4 -> each DP sample is shared by two TP ranks.
    assert readers_for(Kc=8, n=2, op="reduce", select=True, dp_world=4) == 2
    assert readers_for(Kc=6, n=1, op="one", select=True, dp_world=2) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_readers_for_rejects_dp_world_not_dividing_Kc() -> None:
    """Kc must be a whole number of DP coordinates; a remainder means the parallelism
    factorization disagree, which would give a fractional refcount (a leak or an early pop)."""
    from cosmos_framework.posttrain.workers.dispatch import readers_for

    with pytest.raises(ValueError, match="dp_world"):
        readers_for(Kc=8, n=1, op="one", select=True, dp_world=3)
