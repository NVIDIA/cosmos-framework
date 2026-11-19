# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Adaptive training-time action stride for BEHAVIOR-1K (R1Pro).

Synthesise a faster trajectory at the original frame rate by aggregating consecutive
action rows, reducing per controller type:

* **base velocity** (vx, vy, wz) -> ``sum``. Displacement is the integral of the
  command, so summing keeps the base covering the ground the skipped rows would have.
* **position targets** (arms/torso/head) -> the group's ``last`` row: the absolute
  pose the summed base carries the robot to. The conditioning frame is the group's
  ``first`` row, so first->last spans exactly the aggregated command.
* **grippers** -> ``last``, never averaged (binary; the mean of an open/close
  transition neither opens nor closes).

Both paths aggregate ADAPTIVELY and neither clips: a group grows only while the summed
base stays within the cap, so the sum never saturates and displacement is preserved
exactly. An earlier design summed a FIXED group and clipped to the controller limit;
that is not acceptable as a *training target*, because it teaches the model to
under-drive the base on exactly the fast segments where the clip bites.

Training and inference differ only in which end is fixed. Training has read-ahead, so it
fixes the OUTPUT length (``target_len``) and consumes a variable number of raw rows.
Inference has nothing beyond the emitted chunk, so it fixes the INPUT and the output
length is variable and data-dependent, in ``[len(rows)/max_stride, len(rows)]``;
``target_len=None`` selects that mode.

Anchoring stays on raw frames and grouping runs forward FROM each anchor, so every raw
frame is a distinct valid anchor and no global re-segmentation is needed. Callers must
reserve ``max_stride * target_len`` rows past each anchor so a fully-slow region can
still fill ``target_len`` groups.
"""

from __future__ import annotations

import numpy as np

# R1Pro 23-dim action layout: base(3) trunk(4) l_arm(7) l_grip(1) r_arm(7) r_grip(1).
BASE_SLICE = slice(0, 3)  # normalized velocity command (vx, vy, wz)
L_GRIP, R_GRIP = 14, 22  # gripper channels (saturate at +/-1, effectively binary)

# Maximum accumulated action velocity for each base component (vx, vy, wz).
BASE_VEL_CAP = np.array([0.7, 0.7, 0.3], dtype=np.float64)


def adaptive_stride_groups(
    base_vel: np.ndarray,
    max_stride: int,
    target_len: int | None,
    cap: np.ndarray | float = BASE_VEL_CAP,
) -> list[tuple[int, int]]:
    """Greedily partition rows into ``target_len`` groups of 1..``max_stride`` rows.

    ``target_len=None`` consumes ALL rows instead and returns however many groups result.
    That is the INFERENCE case and it is the mirror image of training: the policy emits a
    fixed 32-row chunk with nothing beyond it to read ahead into, so the input length is
    fixed and the OUTPUT length is variable -- between ``len(rows)/max_stride`` (every
    group merges fully) and ``len(rows)`` (yaw or speed blocks every merge).

    Walks forward accumulating the base velocity; a group grows while every component of
    the running sum stays within ``+/-cap`` (broadcast, so ``cap`` may be a scalar or a
    per-component array) and the group is shorter than ``max_stride``. A single raw
    command is already within the cap, so every group holds at least one row and the
    summed base of a group never exceeds the cap -- no clipping is required.

    Args:
        base_vel: ``[M, 3]`` raw base-velocity rows, from the anchor forward.
        max_stride: Maximum rows merged into one output step (the ``N`` in 1x..Nx).
            ``1`` returns the identity partition ``[(0, 1), (1, 2), ...]``.
        target_len: Number of output groups to produce (the chunk length), or ``None``
            to consume every row and return a data-dependent number of groups.
        cap: Per-component (or scalar) saturation limit for the summed base velocity.

    Returns:
        ``target_len`` half-open ``(lo, hi)`` row spans.

    Raises:
        ValueError: if ``max_stride < 1``, or (when ``target_len`` is not ``None``) fewer
            than ``target_len`` groups fit in ``M`` rows (the caller must reserve
            ``max_stride * target_len`` rows).
    """
    if max_stride < 1:
        raise ValueError(f"max_stride must be >= 1, got {max_stride}")
    cap = np.asarray(cap, dtype=np.float64)
    m = base_vel.shape[0]
    groups: list[tuple[int, int]] = []
    i = 0
    while (target_len is None or len(groups) < target_len) and i < m:
        lo = i
        acc = base_vel[i].astype(np.float64).copy()  # first row always fits (|raw| <= cap)
        i += 1
        while i < m and (i - lo) < max_stride and np.all(np.abs(acc + base_vel[i]) <= cap):
            acc += base_vel[i]
            i += 1
        groups.append((lo, i))
    if target_len is not None and len(groups) < target_len:
        raise ValueError(
            f"only {len(groups)} groups fit in {m} rows for target_len={target_len}, "
            f"max_stride={max_stride}; reserve max_stride*target_len rows past the anchor"
        )
    return groups


def aggregate_action_groups(
    action: np.ndarray,
    groups: list[tuple[int, int]],
    base_slice: slice = BASE_SLICE,
    grip_indices: tuple[int, ...] = (L_GRIP, R_GRIP),
) -> np.ndarray:
    """Reduce raw action rows to one row per group, by controller type.

    Starts from each group's LAST row (so positions and grippers take the landed / last
    value) and overrides the base channels with the group SUM. When ``groups`` come from
    :func:`adaptive_stride_groups` the summed base is within the per-component cap by
    construction, so no clip is applied.
    """
    out = np.stack([action[hi - 1] for (_lo, hi) in groups]).astype(action.dtype, copy=True)
    for gi, (lo, hi) in enumerate(groups):
        out[gi, base_slice] = action[lo:hi, base_slice].sum(axis=0)
    # positions and grippers are already the group's last row from the stack above;
    # grip_indices is kept explicit to document that grippers are never averaged.
    _ = grip_indices
    return out


def build_strided_action(
    raw_action: np.ndarray,
    max_stride: int,
    target_len: int,
    base_slice: slice = BASE_SLICE,
) -> tuple[np.ndarray, list[int]]:
    """Aggregate a raw ``[M, D]`` action window into ``target_len`` faster steps.

    Returns ``(aggregated_action, frame_idx)``: ``aggregated_action`` is ``[target_len, D]``
    and ``frame_idx`` is the ``target_len + 1`` group-boundary rows -- the anchor (0)
    followed by each group's landing row -- for gathering the conditioning frames aligned
    to the aggregated actions (frame ``g`` is the observation before aggregated step ``g``).
    """
    groups = adaptive_stride_groups(raw_action[:, base_slice], max_stride, target_len)
    agg = aggregate_action_groups(raw_action, groups)
    frame_idx = [0] + [hi for (_lo, hi) in groups]
    return agg, frame_idx
