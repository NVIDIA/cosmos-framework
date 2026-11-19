# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The splice invariants. Each of these broke silently at least once.

Silently is the operative word: a broken splice still renders a plausible video, so nothing fails
and the run simply scores as a weak failure. These are the checks that would have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest
from failuregen.splice import continuous, rechunk, splice, windows


def test_starts_at_the_host_pose(profile, host_states, donor_states):
    """No teleport, even though the donor stands a metre away.

    Cross-fading absolute joint values instead of deltas jumps the arm to the donor's position at
    the injection instant. On screen that is a cut, not a slip.
    """
    out = splice(profile, host_states, donor_states, dims="all", src_arm="left", dst_arm="left")
    joints = profile.robot.joint_columns
    np.testing.assert_allclose(out[0, joints], host_states[0, joints], atol=1e-6)
    step = np.abs(np.diff(out[:, joints], axis=0)).max()
    host_step = np.abs(np.diff(host_states[:, joints], axis=0)).max()
    donor_step = np.abs(np.diff(donor_states[:, joints], axis=0)).max()
    assert step <= max(host_step, donor_step) * 1.5


def test_jaw_only_leaves_joints_bit_identical(profile, host_states, donor_states):
    """A jaw-only splice must not move the arm at all, or the comparison against a full
    transplant stops meaning what it is used to mean (6 percent against about 80)."""
    out = splice(profile, host_states, donor_states, dims="jaw", src_arm="left", dst_arm="left")
    joints = profile.robot.joint_columns
    assert np.array_equal(out[:, joints], host_states[:, joints])
    assert not np.array_equal(out[:, 6], host_states[:, 6])


def test_arms_only_leaves_the_jaw_untouched(profile, host_states, donor_states):
    out = splice(profile, host_states, donor_states, dims="arms", src_arm="left", dst_arm="left")
    assert np.array_equal(out[:, profile.robot.jaw_columns], host_states[:, profile.robot.jaw_columns])


def test_jaw_starts_from_the_hosts_own_grip(profile, host_states, donor_states):
    """The excursion is re-based on the host's grip value, not copied.

    Two episodes grip the same object at slightly different apertures, so copying the donor's raw
    number steps the jaw at the injection instant, which reads as a glitch rather than a slip.
    """
    donor = donor_states.copy()
    donor[:, 6] += 0.08  # same slip, a differently-calibrated grip
    out = splice(profile, host_states, donor, dims="jaw", blend_frames=0, src_arm="left", dst_arm="left")
    assert out[0, 6] == pytest.approx(host_states[0, 6], abs=1e-6)
    assert out[-1, 6] > profile.robot.hold_band[1]  # and it still leaves the holding band


def test_amplitude_scales_the_excursion_about_the_hosts_value(profile, host_states, donor_states):
    full = splice(
        profile,
        host_states,
        donor_states,
        dims="jaw",
        blend_frames=0,
        jaw_amplitude=1.0,
        src_arm="left",
        dst_arm="left",
    )
    half = splice(
        profile,
        host_states,
        donor_states,
        dims="jaw",
        blend_frames=0,
        jaw_amplitude=0.5,
        src_arm="left",
        dst_arm="left",
    )
    base = host_states[0, 6]
    assert (half[-1, 6] - base) == pytest.approx((full[-1, 6] - base) * 0.5, rel=1e-5)
    zero = splice(
        profile,
        host_states,
        donor_states,
        dims="jaw",
        blend_frames=0,
        jaw_amplitude=0.0,
        src_arm="left",
        dst_arm="left",
    )
    assert np.allclose(zero[:, 6], base)


def test_the_excursion_goes_to_the_holding_arm_not_the_donors(profile, host_states, donor_states):
    """After a handover the object is in the other hand, and writing the slip to the arm that
    dropped it in the donor releases an empty gripper."""
    host = host_states.copy()
    host[:, 13] = 0.32  # the object has been handed over: the right jaw is the one holding
    out = splice(profile, host, donor_states, dims="jaw", blend_frames=0, src_arm="left", dst_arm="right")
    assert np.array_equal(out[:, 6], host[:, 6])  # the donor's own arm is left alone
    assert out[0, 13] == pytest.approx(host[0, 13], abs=1e-6)  # starts from the host's grip
    assert out[-1, 13] > profile.robot.hold_band[1]  # and leaves the holding band


def test_the_excursion_cannot_leave_the_mechanism_range(profile, host_states, donor_states):
    """An amplitude that would drive the aperture past fully open is clamped, because a jaw
    position the mechanism cannot reach is not a failure the robot could produce."""
    out = splice(
        profile,
        host_states,
        donor_states,
        dims="jaw",
        blend_frames=0,
        jaw_amplitude=5.0,
        src_arm="left",
        dst_arm="left",
    )
    lo, hi = profile.robot.jaw_range
    assert out[:, 6].max() == pytest.approx(hi)
    assert out[:, 6].min() >= lo


def test_without_a_declared_range_the_clamp_follows_the_data(profile, host_states, donor_states):
    """A profile that has not measured the mechanism still must not invent unreachable values."""
    import dataclasses

    loose = dataclasses.replace(profile, robot=dataclasses.replace(profile.robot, jaw_range=None))
    out = splice(
        loose, host_states, donor_states, dims="jaw", blend_frames=0, jaw_amplitude=5.0, src_arm="left", dst_arm="left"
    )
    observed_max = max(host_states[:, 6].max(), donor_states[:, 6].max())
    assert out[:, 6].max() == pytest.approx(observed_max)


def test_blend_reaches_the_donor_and_zero_blend_is_immediate(profile, host_states, donor_states):
    faded = splice(profile, host_states, donor_states, dims="jaw", blend_frames=8, src_arm="left", dst_arm="left")
    hard = splice(profile, host_states, donor_states, dims="jaw", blend_frames=0, src_arm="left", dst_arm="left")
    assert faded[1, 6] == pytest.approx(host_states[1, 6], abs=0.05)
    assert hard[0, 6] != pytest.approx(host_states[0, 6] + 1.0)
    assert faded[-1, 6] == pytest.approx(hard[-1, 6], abs=1e-5)


def test_shape_mismatch_and_bad_dims_raise(profile, host_states, donor_states):
    with pytest.raises(ValueError):
        splice(profile, host_states, donor_states[:-1], dims="all")
    with pytest.raises(ValueError):
        splice(profile, host_states, donor_states, dims="gripper")


def test_windows_are_overlapping_and_round_trip(profile, host_states):
    chunk = 8
    ws = windows(host_states, 0, 3, chunk)
    assert [w.shape for w in ws] == [(chunk + 1, 14)] * 3
    for a, b in zip(ws, ws[1:]):
        np.testing.assert_allclose(a[-1], b[0])  # actions are differences, so states are shared
    back = continuous(ws, chunk)
    np.testing.assert_allclose(back, host_states[: 3 * chunk + 1])
    again = rechunk(back, 3, chunk)
    for a, b in zip(ws, again):
        np.testing.assert_allclose(a, b)


def test_short_tails_are_padded_by_holding_still(profile, host_states):
    ws = windows(host_states, 30, 2, 8)
    assert all(w.shape == (9, 14) for w in ws)
    np.testing.assert_allclose(ws[-1][-1], ws[-1][-2])
