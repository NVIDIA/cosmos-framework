# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The shared segment must hold the observation the renderer actually produces.

Nothing guarded this. `segment_bytes()` sizes the segment from the capabilities handshake with
`channels=3`, while an `ObsBundle` carries 4 -- the renderer emits RGBA and
`the external simulator backend::detach_frames` does not narrow it. `wire.py` documents that gap and calls
the thing that absorbs it "a 0.17 margin, not a design", which is exactly the kind of statement
that should be executable rather than prose.

The anchor is MEASURED, not derived. `describe_obs` on a pod, campaign `round5-c5-smoke`
2026-08-27, reproduced on two units and two tasks:

    declared 5 keys / 3,917,128 B   undeclared 0 keys / 0 B
    camera_channels = {'head': 4, 'left_wrist': 4, 'right_wrist': 4}

which reconciles to the byte: 720^2*4 + 2*480^2*4 = 3,916,800 cameras, 61*f32 = 244 proprio,
3*7*f32 = 84 extrinsics.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.posttrain.embodied.wire import (
    MIN_OBS_BYTES,
    SEGMENT_SLACK,
    Segment,
    WireError,
    segment_bytes,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

#: BEHAVIOR-1K's own camera shapes -- head 720 / wrists 480, a corpus-shape requirement rather
#: than a tunable. Square, as the measured reconciliation above confirms.
B1K_SHAPES = {"head": (720, 720), "left_wrist": (480, 480), "right_wrist": (480, 480)}

#: Proprioception width from the same measurement.
B1K_PROPRIO = 61

#: The observation `describe_obs` actually reported on a pod, in bytes.
MEASURED_OBS_BYTES = 3_917_128


def test_the_segment_holds_the_observation_that_was_actually_measured():
    """The invariant, stated against a number nobody in this process computed."""
    capacity = segment_bytes(B1K_SHAPES, B1K_PROPRIO)
    assert capacity >= MEASURED_OBS_BYTES, (
        f"segment is {capacity} B for an observation measured at {MEASURED_OBS_BYTES} B"
    )


def test_the_floor_is_what_protects_b1k_and_not_the_channel_default():
    """`channels=3` is NOT what makes b1k fit. `MIN_OBS_BYTES` is, and the difference matters.

    `wire.py` explains the 3-vs-4 gap in terms of `SEGMENT_SLACK` being 1.5 against a needed
    4/3 -- a 0.17 margin. At b1k's own resolutions that margin is not load-bearing at all: the
    computed size loses to the floor either way, so raising the default to 4 would not move a
    single byte.

    This test is here so that stops being true LOUDLY. Trim `SEGMENT_SLACK`, or raise the
    resolutions until the computed size clears the floor, and the margin becomes the thing
    holding RGBA -- at which point somebody should be told, rather than finding out from a
    region overflow in a pod log.
    """
    assert segment_bytes(B1K_SHAPES, B1K_PROPRIO, channels=3) == MIN_OBS_BYTES
    assert segment_bytes(B1K_SHAPES, B1K_PROPRIO, channels=4) == MIN_OBS_BYTES


def test_the_floor_itself_covers_the_measured_observation():
    """`MIN_OBS_BYTES` is `int(3_917_128 * 1.5)`. If either half is edited, this says so."""
    assert MIN_OBS_BYTES >= MEASURED_OBS_BYTES
    assert MIN_OBS_BYTES == int(MEASURED_OBS_BYTES * SEGMENT_SLACK)


def test_five_channels_would_outgrow_the_floor():
    """The documented breaking point, pinned rather than asserted in prose.

    `wire.py` says the margin "would stop holding at 5 channels". At 5 the computed size wins
    over the floor, which is the observable form of that claim -- and the reason the channel
    default cannot be dismissed as dead for all time.
    """
    assert segment_bytes(B1K_SHAPES, B1K_PROPRIO, channels=5) > MIN_OBS_BYTES


# ── the claim the "this is benign" argument rests on ─────────────────────────


def test_an_oversized_write_raises_and_names_both_numbers():
    """No silent truncation, and the error has to be diagnosable.

    Everything above only matters because the failure mode is a REFUSAL. A segment that
    truncated instead would put a short observation on the wire with a valid-looking shape, and
    the two numbers in this message are what tell an operator whether the fix is the declared
    `camera_shapes` or the slack. Reading the guard is not the same as it staying there.
    """
    capacity = 4096
    seg = Segment.create(capacity)
    try:
        with pytest.raises(WireError, match=r"needs \d+ B, region is \d+ B"):
            seg.write_arrays("obs", {"too_big": np.zeros(capacity + 1, dtype=np.uint8)})
    finally:
        seg.close()


def test_a_write_that_exactly_fills_the_region_is_allowed():
    """The boundary on the other side: `>` not `>=`, or the last usable byte is unreachable."""
    capacity = 4096
    seg = Segment.create(capacity)
    try:
        descriptors = seg.write_arrays("obs", {"exact": np.zeros(capacity, dtype=np.uint8)})
        assert descriptors[0].nbytes == capacity
    finally:
        seg.close()
