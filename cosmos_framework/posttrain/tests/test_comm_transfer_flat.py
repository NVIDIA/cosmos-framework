# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest
import torch

from cosmos_framework.posttrain.comm.transfer.flat import (
    carve,
    flat_nbytes,
    flat_plan,
    pack,
)


def _state():
    # Mixed dtypes and a non-square shape on purpose: a real bucket holds bf16
    # weights next to fp32 norms, and byte-based offsets are what make that work.
    return {
        "b.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "a.bias": torch.ones(4, dtype=torch.bfloat16),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_flat_plan_is_sorted_and_aligned():
    plan = flat_plan(_state())
    assert [e[0] for e in plan] == ["a.bias", "a.weight", "b.weight"]
    for _name, _shape, _dtype, off, _nb in plan:
        assert off % 16 == 0, f"offset {off} is not 16-byte aligned"


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_carve_round_trip():
    state = _state()
    plan = flat_plan(state)
    buf = torch.empty(flat_nbytes(plan), dtype=torch.uint8)
    flat = pack(state, plan, buf)
    got = carve(flat, plan)

    assert set(got) == set(state)
    for k, v in state.items():
        assert got[k].shape == v.shape
        assert got[k].dtype == v.dtype
        assert torch.equal(got[k], v)


@pytest.mark.L0
@pytest.mark.CPU
def test_carve_returns_views_not_copies():
    # The consumer must not pay a copy to split the flat buffer.
    state = _state()
    plan = flat_plan(state)
    buf = torch.empty(flat_nbytes(plan), dtype=torch.uint8)
    flat = pack(state, plan, buf)
    got = carve(flat, plan)
    got["a.bias"].fill_(7)
    assert carve(flat, plan)["a.bias"].eq(7).all()


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_rejects_undersized_buffer():
    state = _state()
    plan = flat_plan(state)
    with pytest.raises(ValueError, match="too small"):
        pack(state, plan, torch.empty(flat_nbytes(plan) - 1, dtype=torch.uint8))


@pytest.mark.L0
@pytest.mark.CPU
def test_pack_rejects_non_contiguous_source():
    # A transposed view would pack garbage; fail loud instead.
    state = {"w": torch.zeros(4, 4)[:, :2]}
    plan = flat_plan(state)
    buf = torch.empty(flat_nbytes(plan), dtype=torch.uint8)
    with pytest.raises(ValueError, match="contiguous"):
        pack(state, plan, buf)


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_sizing_matches_the_real_layout():
    """Bind sizes the staging buffer from metadata, before any bucket exists. If that
    disagreed with what pack() needs, the mismatch would surface mid-sync as 'buffer too
    small' -- so the two must be computed the same way."""
    from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

    state = _state()
    entries = [(n, tuple(t.shape), str(t.dtype)) for n, t in sorted(state.items())]
    assert flat_nbytes_for_manifest(entries) == flat_nbytes(flat_plan(state))


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_sizing_is_exact_for_every_dtype_width():
    # weight_transfer._dtype_bytes is a 2/4/8 heuristic on the dtype string; sizing must use
    # the true element size or a float64 bucket would be under-allocated.
    from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

    for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.uint8):
        t = torch.zeros(8, dtype=dt)
        entries = [("w", (8,), str(dt))]
        assert flat_nbytes_for_manifest(entries) == flat_nbytes(flat_plan({"w": t})), dt
