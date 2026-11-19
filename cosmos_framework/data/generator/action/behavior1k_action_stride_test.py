# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Adaptive training-time action stride: aggregate raw rows into a faster trajectory at
the original fps, per controller type, WITHOUT ever clipping the base velocity.

Mechanism tests pass an explicit uniform ``cap`` so they exercise the algorithm
independently of the dataset constant; two tests lock the real per-component default.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.data.generator.action.behavior1k_action_stride import (
    BASE_SLICE,
    L_GRIP,
    R_GRIP,
    adaptive_stride_groups,
    aggregate_action_groups,
    build_strided_action,
)

DIM = 23


@pytest.mark.L0
@pytest.mark.CPU
def test_max_stride_of_one_is_the_identity_partition():
    bv = np.random.default_rng(0).uniform(-0.5, 0.5, size=(8, 3)).astype(np.float32)
    assert adaptive_stride_groups(bv, max_stride=1, target_len=8, cap=1.0) == [(i, i + 1) for i in range(8)]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_slow_region_merges_up_to_max_stride():
    bv = np.full((16, 3), 0.1, dtype=np.float32)
    assert adaptive_stride_groups(bv, max_stride=4, target_len=4, cap=1.0) == [(0, 4), (4, 8), (8, 12), (12, 16)]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_fast_region_stays_raw():
    """0.6 + 0.6 = 1.2 > cap, so no pair may merge -- every group is a single row."""
    bv = np.full((4, 3), 0.6, dtype=np.float32)
    assert adaptive_stride_groups(bv, max_stride=4, target_len=4, cap=1.0) == [(0, 1), (1, 2), (2, 3), (3, 4)]


@pytest.mark.L0
@pytest.mark.CPU
def test_the_cap_is_per_component():
    """vx, vy could merge but wz saturates first, so the group is cut on wz alone."""
    bv = np.tile(np.array([0.1, 0.1, 0.6], dtype=np.float32), (6, 1))
    groups = adaptive_stride_groups(bv, max_stride=3, target_len=6, cap=1.0)
    assert all(hi - lo == 1 for lo, hi in groups)


@pytest.mark.L0
@pytest.mark.CPU
def test_base_velocity_is_summed_and_never_exceeds_the_cap():
    action = np.zeros((8, DIM), dtype=np.float32)
    action[:, BASE_SLICE] = 0.3
    groups = adaptive_stride_groups(action[:, BASE_SLICE], max_stride=2, target_len=4, cap=1.0)
    out = aggregate_action_groups(action, groups)
    assert np.allclose(out[:, BASE_SLICE], 0.6), "paired sub-cap velocities must add"
    assert np.all(np.abs(out[:, BASE_SLICE]) <= 1.0 + 1e-6), "summed base must stay within the cap"


@pytest.mark.L0
@pytest.mark.CPU
def test_total_base_displacement_is_preserved():
    action = np.zeros((12, DIM), dtype=np.float32)
    action[:, BASE_SLICE] = 0.2
    groups = adaptive_stride_groups(action[:, BASE_SLICE], max_stride=3, target_len=4, cap=1.0)
    out = aggregate_action_groups(action, groups)
    assert out[:, BASE_SLICE].sum() == pytest.approx(action[:, BASE_SLICE].sum())


@pytest.mark.L0
@pytest.mark.CPU
def test_positions_take_the_group_last_value():
    action = np.zeros((4, DIM), dtype=np.float32)
    action[:, BASE_SLICE] = 0.3
    action[:, 7] = [0.0, 1.0, 2.0, 3.0]
    groups = adaptive_stride_groups(action[:, BASE_SLICE], max_stride=2, target_len=2, cap=1.0)
    out = aggregate_action_groups(action, groups)
    assert np.allclose(out[:, 7], [1.0, 3.0]), "arm target must be the group's last, not the mean"


@pytest.mark.L0
@pytest.mark.CPU
def test_grippers_take_the_last_value_and_are_never_averaged():
    action = np.zeros((4, DIM), dtype=np.float32)
    action[:, BASE_SLICE] = 0.3
    action[:, L_GRIP] = [1.0, -1.0, -1.0, 1.0]
    action[:, R_GRIP] = [-1.0, 1.0, 1.0, -1.0]
    groups = adaptive_stride_groups(action[:, BASE_SLICE], max_stride=2, target_len=2, cap=1.0)
    out = aggregate_action_groups(action, groups)
    assert np.array_equal(out[:, L_GRIP], [-1.0, 1.0]), "gripper must be last, not mean (0.0)"
    assert np.array_equal(out[:, R_GRIP], [1.0, -1.0])
    assert not np.any(out[:, [L_GRIP, R_GRIP]] == 0.0), "a 0.0 gripper means averaging crept in"


@pytest.mark.L0
@pytest.mark.CPU
def test_an_adaptive_mix_stays_capped_everywhere():
    bv = np.zeros((12, 3), dtype=np.float32)
    bv[0::2] = 0.1
    bv[1::2] = 0.9
    groups = adaptive_stride_groups(bv, max_stride=4, target_len=6, cap=1.0)
    for lo, hi in groups:
        assert np.all(np.abs(bv[lo:hi].sum(axis=0)) <= 1.0 + 1e-6)


@pytest.mark.L0
@pytest.mark.CPU
def test_too_few_rows_raises_so_the_caller_reserves_the_tail():
    with pytest.raises(ValueError):
        adaptive_stride_groups(np.zeros((3, 3), dtype=np.float32), max_stride=1, target_len=8)


@pytest.mark.L0
@pytest.mark.CPU
def test_wz_throttles_merging_at_the_default_cap():
    """At the real caps, wz (0.3) binds before vx/vy (0.7): a uniform slow stream merges
    far fewer rows than the vx/vy cap alone would permit (~7 rows of 0.1)."""
    bv = np.full((16, 3), 0.1, dtype=np.float32)
    widths = [hi - lo for lo, hi in adaptive_stride_groups(bv, max_stride=16, target_len=4)]
    assert max(widths) <= 3, widths  # |wz sum| <= 0.3 -> ~2-3 rows of 0.1
    assert max(widths) < 7, widths  # strictly tighter than the vx/vy cap would allow


@pytest.mark.L0
@pytest.mark.CPU
def test_build_strided_action_returns_aligned_boundary_frames():
    """T aggregated actions come with T+1 conditioning frames at the group boundaries."""
    action = np.zeros((8, DIM), dtype=np.float32)
    action[:, 0] = 0.3  # vx only: pairs sum to 0.6 (< vx cap 0.7) -> width 2 under default cap
    action[:, 7] = np.arange(8)
    agg, frame_idx = build_strided_action(action, max_stride=2, target_len=4)
    assert agg.shape == (4, DIM)
    assert frame_idx == [0, 2, 4, 6, 8], frame_idx  # anchor + each group's landing row
    assert len(frame_idx) == agg.shape[0] + 1  # T+1 frames for T actions
    assert np.allclose(agg[:, 0], 0.6)  # base summed
    assert np.allclose(agg[:, 7], [1, 3, 5, 7])  # positions last
