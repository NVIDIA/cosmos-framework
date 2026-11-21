# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Did the object actually leave the gripper? Measured on the wrist view, against a control.

Two decisions here were each arrived at by getting a wrong answer first.

**Watch the wrist view of the holding hand.** In the wide view a slip is a handful of pixels, and
ordinary autoregressive drift plus the odd hallucinated object looks the same; in the wrist view
the object fills the frame while held and the frame empties when it goes. The hand is read from
the run's record, because after a handover it is the other arm, and scoring the wrong pane
measures a hand that was never going to change.

**Divide by a matched control.** The same clip untouched scores 0.079 in one phase and 0.255 in
another purely because the arm moves faster there. An absolute threshold therefore reports
success where nothing happened; it once produced an apparent 100 percent on exactly the phases
where the injection provably did nothing. Only the ratio against the run's own control is
reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from . import legacy
from .config import Profile

RATIO_THRESHOLD = 1.5  # a judgement call, unlike the pane measure; report it alongside results
PRE_FRAMES = 12
POST_FRAMES = 25


@dataclass
class RunScore:
    name: str
    recipe: str
    mode: str
    arm: str | None
    target_phase: str | None
    pre: float
    post: float
    control: float | None = None
    pose_distance: float | None = None

    @property
    def ratio(self) -> float | None:
        if self.control is None or self.control <= 0:
            return None
        return self.post / self.control

    @property
    def hit(self) -> bool:
        r = self.ratio
        return r is not None and r >= RATIO_THRESHOLD


def wrist_pane(profile: Profile, mp4: Path, arm: str) -> list[np.ndarray]:
    """Frames of one wrist pane, as uint8.

    Kept at 8 bits deliberately: a gallery can hold hundreds of clips and float32 frames run to
    tens of gigabytes, which is enough to take down the machine this is called from.
    """
    order = sorted(profile.cameras.wrist)
    if not order:
        raise ValueError("profile declares no wrist cameras, so there is no pane to score")
    slot = order.index(arm) if arm in order else 0
    n = len(order)
    out = []
    with av.open(str(mp4)) as container:
        for frame in container.decode(container.streams.video[0]):
            img = frame.to_image()
            w, h = img.size
            top = int(round(h * profile.cameras.front_fraction))
            lo = (w // n) * slot
            out.append(np.asarray(img.crop((lo, top, lo + w // n, h)), dtype=np.uint8))
    return out


def _change(frames: list[np.ndarray], ref: np.ndarray, lo: int, hi: int) -> float:
    idx = range(max(0, lo), min(len(frames), hi))
    vals = [float(np.abs(frames[i].astype(np.float32) - ref.astype(np.float32)).mean()) / 255.0 for i in idx]
    return float(np.mean(vals)) if vals else 0.0


def score_run(profile: Profile, run_dir: Path) -> RunScore | None:
    record = legacy.read_record(run_dir)
    if record is None or not (run_dir / "synth.mp4").is_file():
        return None
    fps = int(record.get("fps", profile.dataset.target_fps))
    r = int(round(float(record.get("failure_time_s", 0.0)) * fps))
    arm = record.get("arm") or next(iter(profile.robot.arms))
    frames = wrist_pane(profile, run_dir / "synth.mp4", arm)
    if r <= 1 or r >= len(frames) - 2:
        return None
    ref = frames[r]
    return RunScore(
        name=record.get("name") or run_dir.name,
        recipe=record.get("recipe") or "",
        mode=record.get("mode") or "",
        arm=arm,
        target_phase=record.get("target_phase"),
        pre=round(_change(frames, ref, r - PRE_FRAMES, r), 4),
        post=round(_change(frames, ref, r + 1, r + 1 + POST_FRAMES), 4),
        pose_distance=record.get("pose_distance"),
    )


def score_directory(profile: Profile, root: Path) -> list[RunScore]:
    """Score every run under ``root``, pairing each injection with its matched control."""
    scores: dict[str, RunScore] = {}
    keys: dict[str, tuple] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        s = score_run(profile, run_dir)
        if s is None:
            continue
        record = legacy.read_record(run_dir)
        scores[s.name] = s
        keys[s.name] = (
            record["host"]["episode_index"],
            record.get("target_phase"),
            round(float(record["host"]["t_inject"]), 2),
            record.get("arm"),
        )

    controls = {keys[n]: s.post for n, s in scores.items() if s.mode == "control"}
    out = []
    for name, s in scores.items():
        if s.mode == "control":
            continue
        s.control = controls.get(keys[name])
        out.append(s)
    return sorted(out, key=lambda s: -(s.ratio or 0))


def summarise(scores: list[RunScore]) -> dict[str, dict]:
    """Hit rate and median ratio per recipe, plus how many runs had no control to compare to."""
    groups: dict[str, list[RunScore]] = {}
    for s in scores:
        groups.setdefault(s.recipe or "unknown", []).append(s)
    out = {}
    for recipe, items in groups.items():
        ratios = [s.ratio for s in items if s.ratio is not None]
        out[recipe] = {
            "n": len(items),
            "scored": len(ratios),
            "unscorable": len(items) - len(ratios),
            "hits": sum(1 for s in items if s.hit),
            "median_ratio": round(float(np.median(ratios)), 3) if ratios else None,
            "best_ratio": round(max(ratios), 3) if ratios else None,
        }
    return out
