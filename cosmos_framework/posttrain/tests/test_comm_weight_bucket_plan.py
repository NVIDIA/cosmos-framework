# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""bucket_plan partitions a sorted manifest into ≤bucket_bytes buckets, preserving order and covering
every param exactly once (the Phase-0 bucketing for streaming weight sync)."""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.comm.weight_transfer import bucket_plan


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_plan_partitions_by_size_in_order():
    # 4 float32 params (4 B/elem): a,b,c = 100 elems (400 B each), d = 250 elems (1000 B). bucket=800 B.
    m = [
        ("a", (100,), "torch.float32"),
        ("b", (100,), "torch.float32"),
        ("c", (100,), "torch.float32"),
        ("d", (250,), "torch.float32"),
    ]
    plan = bucket_plan(m, bucket_bytes=800)
    names = [[e[0] for e in bk] for bk in plan]
    assert names == [["a", "b"], ["c"], ["d"]]  # a+b=800 closes; c alone (c+d>800); d alone (1000>800)
    assert [e[0] for bk in plan for e in bk] == ["a", "b", "c", "d"]  # every param once, order preserved


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_plan_single_oversized_param_is_own_bucket():
    m = [("big", (1000,), "torch.float32")]  # 4000 B > 800
    plan = bucket_plan(m, bucket_bytes=800)
    assert len(plan) == 1 and [e[0] for bk in plan for e in bk] == ["big"]


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_plan_bf16_half_bytes():
    # bf16 is 2 B/elem: 300 elems = 600 B each; two fit in a 1400 B bucket, third spills.
    m = [("x", (300,), "torch.bfloat16"), ("y", (300,), "torch.bfloat16"), ("z", (300,), "torch.bfloat16")]
    plan = bucket_plan(m, bucket_bytes=1400)
    assert [[e[0] for e in bk] for bk in plan] == [["x", "y"], ["z"]]


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_plan_rejects_unsorted_manifest():
    # Sorted-name order is the collective transfer's correctness precondition (broadcast order is
    # sorted order, recv-buffer order is plan order) — same-shape params (q/k/v) would silently swap.
    m = [("q_proj", (100,), "torch.float32"), ("k_proj", (100,), "torch.float32")]
    with pytest.raises(ValueError, match="name-sorted"):
        bucket_plan(m, bucket_bytes=800)


@pytest.mark.L0
@pytest.mark.CPU
def test_set_bucket_plan_rejects_unsorted_plan():
    # The consumer receives the plan over RPC from its peer — it must not trust ordering blindly.
    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync

    ws = WeightSync(CopyMover())
    bad = [[("b", (10,), "torch.float32")], [("a", (10,), "torch.float32")]]  # unsorted ACROSS buckets
    with pytest.raises(ValueError, match="name-sorted"):
        ws.set_bucket_plan(bad)
    assert not ws.has_plan()  # a rejected plan must not be installed
    ws.set_bucket_plan([[("a", (10,), "torch.float32")], [("b", (10,), "torch.float32")]])
    assert ws.has_plan() and ws.bucket_count() == 2
