# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Splice a donor's failure into a host trajectory.

The single most important finding behind this package: transplanting the donor's **arm motion**
is what makes a failure render. The same donors with only their gripper channel copied land
about 6 percent of the time; with the arm motion included, about 80. What the model reproduces
is the arm's reaction to losing the object, not a number changing in one channel.

Two details that look cosmetic and are not:

Joints are framewise deltas, so a cross-fade has to happen on the deltas and be integrated from
the host's own pose. Averaging absolute joint values instead teleports the arm to wherever the
donor happened to be standing.

The donor's aperture excursion is re-expressed around the host's own holding value before being
applied, because two episodes grip the same object at slightly different apertures and copying
the raw number reads as a jaw jump rather than a slip.
"""

from __future__ import annotations

import numpy as np

from .config import Profile

Dims = str  # "arms" | "jaw" | "all"


def splice(
    profile: Profile,
    host: np.ndarray,
    donor: np.ndarray,
    dims: Dims = "all",
    blend_frames: int = 4,
    jaw_amplitude: float = 1.0,
    src_arm: str | None = None,
    dst_arm: str | None = None,
) -> np.ndarray:
    """Post-injection states, faded out of ``host`` and into ``donor``.

    ``dims`` selects the channels taken from the donor: ``arms`` its joints, ``jaw`` its
    gripper, ``all`` both. ``jaw_amplitude`` scales the aperture excursion, below one for a
    partial slip. ``src_arm`` is the arm that slipped in the donor and ``dst_arm`` the arm
    holding the object in the host; they differ whenever the object changes hands, and writing
    the excursion to the wrong one releases an empty hand.
    """
    if host.shape != donor.shape:
        raise ValueError(f"host {host.shape} and donor {donor.shape} must match")
    if dims not in ("arms", "jaw", "all"):
        raise ValueError(f"dims must be arms, jaw or all, not {dims!r}")

    joints = profile.robot.joint_columns
    n = len(host)
    out = np.array(host, dtype=np.float32, copy=True)

    ramp = np.ones(n, dtype=np.float32)
    if blend_frames > 0:
        k = min(blend_frames, n)
        ramp[:k] = np.linspace(0.0, 1.0, k, dtype=np.float32)

    if dims in ("arms", "all"):
        d_host = np.diff(host[:, joints], axis=0)
        d_donor = np.diff(donor[:, joints], axis=0)
        w = ramp[1:][:, None]
        blended = (1.0 - w) * d_host + w * d_donor
        out[1:, joints] = host[0, joints] + np.cumsum(blended, axis=0)

    if dims in ("jaw", "all"):
        first = next(iter(profile.robot.arms))
        src = profile.robot.arm(src_arm or first).jaw
        dst = profile.robot.arm(dst_arm or src_arm or first).jaw
        lead = max(1, min(3, n))
        host_hold = float(np.median(host[:lead, dst]))
        donor_hold = float(np.median(donor[:lead, src]))
        profile_curve = host_hold + jaw_amplitude * (donor[:, src] - donor_hold)
        blended = (1.0 - ramp) * host[:, dst] + ramp * profile_curve
        if profile.robot.jaw_range is not None:
            lo, hi = profile.robot.jaw_range
        else:
            observed = np.concatenate([host[:, dst], donor[:, src]])
            lo, hi = float(np.min(observed)), float(np.max(observed))
        out[:, dst] = np.clip(blended, lo, hi)

    return out


def windows(states: np.ndarray, start: int, n_chunks: int, chunk_length: int) -> list[np.ndarray]:
    """``n_chunks`` overlapping windows of ``chunk_length + 1`` states, tail-padded if short.

    Consecutive windows share a state because the actions are differences: chunk k's last state
    is chunk k+1's first. Padding repeats the final state, which holds the arm still -- fine at
    the very end of an episode, wrong in the middle, which is why callers check for room first.
    """
    out = []
    for k in range(n_chunks):
        lo = start + k * chunk_length
        w = states[lo : lo + chunk_length + 1]
        if len(w) < chunk_length + 1:
            if len(w) == 0:
                w = states[-1:]
            w = np.concatenate([w, np.repeat(w[-1:], chunk_length + 1 - len(w), axis=0)])
        out.append(np.asarray(w, dtype=np.float32))
    return out


def continuous(windows_list: list[np.ndarray], chunk_length: int) -> np.ndarray:
    """Flatten overlapping windows back into one sequence, dropping the shared states."""
    span = len(windows_list) * chunk_length + 1
    joined = np.concatenate([w[:-1] for w in windows_list] + [windows_list[-1][-1:]])
    return joined[:span]


def rechunk(states: np.ndarray, n_chunks: int, chunk_length: int) -> list[np.ndarray]:
    return [states[k * chunk_length : k * chunk_length + chunk_length + 1] for k in range(n_chunks)]
