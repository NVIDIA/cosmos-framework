# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Choose hosts and injection points.

A host is a successful episode; the injection point is a moment inside the target phase where
the relevant hand is genuinely holding the object. Both conditions matter and both were learned
by getting them wrong:

The hand must be the one carrying the object, which changes at a handover. Three of eleven smoke
cases were releasing an empty hand before this was checked.

The jaw must be inside the holding band at that instant. The midpoint of a grasp phase is often
before the jaw has closed, so the search nudges later within the phase rather than giving up.

Pose matching is deliberately *not* a filter. Injections landed at every donor-host pose
distance from 0.21 to 1.29 rad with no trend (correlation -0.14), so matching is recorded for
analysis but never used to reject a pairing -- which is what makes wide coverage cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import annotations as ann
from .config import Profile
from .donors import Donor


@dataclass
class Host:
    collection: str
    index: int
    duration: float
    episode: ann.Episode

    def windows(self, phase: str) -> list[ann.Window]:
        return self.episode.windows(phase)


@dataclass
class InjectionPoint:
    host: Host
    phase: str
    arm: str
    t: float
    post_chunks: int
    pose_distance: float
    window_index: int


def find_hosts(
    profile: Profile,
    collection: str,
    require_phases: tuple[str, ...] = (),
    limit: int | None = None,
    diverse: bool = True,
) -> list[Host]:
    """Clean episodes, optionally spread apart in joint space rather than simply the first N.

    Taking consecutive episodes tends to give near-identical takes; spacing them out is what
    makes a generated set look varied.
    """
    episodes = ann.load_episodes(profile, collection)
    cands: list[Host] = []
    signatures: list[np.ndarray] = []
    for index in sorted(episodes):
        ep = episodes[index]
        if not ep.is_clean(profile.dataset):
            continue
        if require_phases and not ep.has_phases(require_phases):
            continue
        st = ann.state(profile, collection, index)
        cands.append(Host(collection, index, len(st) / profile.dataset.source_fps, ep))
        step = max(1, len(st) // 12)
        signatures.append(st[::step][:12].mean(axis=0))

    if limit is None or limit >= len(cands) or not diverse or not cands:
        return cands[:limit] if limit else cands

    picked = [0]
    while len(picked) < limit:
        best, best_d = None, -1.0
        for i in range(len(cands)):
            if i in picked:
                continue
            d = min(float(np.linalg.norm(signatures[i] - signatures[j])) for j in picked)
            if d > best_d:
                best, best_d = i, d
        picked.append(best)
    return [cands[i] for i in picked]


def injection_points(
    profile: Profile,
    host: Host,
    phase: str,
    donor: Donor,
    *,
    pre_chunks: int = 2,
    post_chunks: int = 2,
    strategy: str = "pose",
    onset: float = 0.5,
    max_windows: int = 1,
    allow_pad: bool = False,
) -> list[InjectionPoint]:
    """Candidate injection points in ``phase``, at most one per window of it.

    ``strategy`` picks within a window: ``pose`` takes the instant whose arm pose is closest to
    the donor's at failure onset, ``onset`` takes a fixed fraction through the window. Neither
    predicts success, so the reason to walk several windows is not a better match -- it is that a
    later cycle has objects already stacked, so the same donor yields a visibly different scene.

    Returns the most post chunks the episode affords rather than nothing, because the late phases
    end with the episode and would otherwise be unusable.
    """
    dp = profile.dataset
    arm_name = profile.robot.holding_arm(phase)
    jaw = profile.robot.arm(arm_name).jaw
    donor_arm = profile.robot.arm(donor.arm)

    host_state = ann.state(profile, host.collection, host.index)
    donor_state = ann.state(profile, donor.collection, donor.episode)
    q0 = donor_state[
        min(int(donor.t_start * dp.source_fps), len(donor_state) - 1), donor_arm.joints[0] : donor_arm.joints[1]
    ]
    lo_col, hi_col = profile.robot.arm(arm_name).joints

    out: list[InjectionPoint] = []
    for wi, w in enumerate(host.windows(phase)):
        best: InjectionPoint | None = None
        best_cost = float("inf")
        seed_t = w.start + w.duration * onset
        for i in range(int(w.start * dp.source_fps), min(int(w.end * dp.source_fps), len(host_state)), dp.stride):
            t = i / dp.source_fps
            if t - pre_chunks * dp.chunk_seconds < 0:
                continue
            # The hand must be holding the object for it to be lost. Grasp phases often begin
            # before the jaw has closed, so the search moves later in the window instead of
            # giving up on the phase.
            if not profile.robot.is_holding(host_state[i, jaw]):
                continue
            affordable = next((p for p in range(post_chunks, 0, -1) if t + p * dp.chunk_seconds <= host.duration), None)
            if affordable is None:
                if not allow_pad:
                    continue
                affordable = 1
            dist = float(np.linalg.norm(host_state[i, lo_col:hi_col] - q0))
            cost = dist if strategy == "pose" else abs(t - seed_t)
            if cost < best_cost:
                best_cost = cost
                best = InjectionPoint(host, phase, arm_name, round(t, 2), affordable, round(dist, 4), wi)
        if best is not None:
            out.append(best)
        if len(out) >= max_windows:
            break
    return out
