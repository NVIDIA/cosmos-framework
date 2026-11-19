# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Choose donors: annotated failures whose telemetry actually corroborates the label.

Reviewers mark what they see, and the state channels do not always agree. Of 22 human-reviewed
drops in the reference corpus, only about half show the jaw aperture leaving the holding band;
the rest are real events the gripper channel never registered. Transplanting one of those
carries no signature and produces a clip where nothing happens, which is indistinguishable from
a broken run unless you check.

So a donor must satisfy both: a human said a failure happened, and the jaw moved out of the
holding band while it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import annotations as ann
from .config import Profile


@dataclass
class Donor:
    collection: str
    episode: int
    event_index: int
    t_start: float
    t_end: float
    phase: str | None
    arm: str
    overshoot: float
    outcome: str
    label: str
    terminal: bool = False
    n_events_in_episode: int = 1
    caption: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.episode}:{self.event_index}"

    def to_dict(self) -> dict:
        d = {
            "source": self.collection,
            "episode_index": self.episode,
            "event_index": self.event_index,
            "t_start": round(self.t_start, 2),
            "t_end": round(self.t_end, 2),
            "phase": self.phase,
            "arm": self.arm,
            "overshoot": round(self.overshoot, 3),
            "outcome": self.outcome,
            "label": self.label,
            "terminal": self.terminal,
            "n_events_in_episode": self.n_events_in_episode,
            "prompt": self.caption,
            "tier": "annotated",
        }
        d.update(self.extra)
        return d


def find(
    profile: Profile,
    collection: str,
    *,
    require_signature: bool = True,
    min_overshoot: float = 0.0,
    arms: tuple[str, ...] | None = None,
) -> list[Donor]:
    """Every annotated failure event, strongest signature first.

    ``require_signature`` is on by default and is the difference between donors that work and
    donors that do nothing. Set it off only to inspect what the annotations claim.
    """
    episodes = ann.load_episodes(profile, collection)
    dp = profile.dataset
    want_arms = arms or tuple(profile.robot.arms)
    out: list[Donor] = []
    for index in sorted(episodes):
        ep = episodes[index]
        events = ep.failures(dp)
        if not events:
            continue
        st = ann.state(profile, collection, index)
        terminal = any(s.label not in dp.failure_labels + (dp.normal_label,) + dp.recovery_labels for s in ep.segments)
        for i, seg in enumerate(events):
            lo = max(0, int(seg.start * dp.source_fps))
            hi = min(len(st) - 1, int(seg.end * dp.source_fps))
            if hi <= lo:
                continue
            best: tuple[float, str] | None = None
            for name in want_arms:
                jaw = profile.robot.arm(name).jaw
                pre = float(np.median(st[max(0, lo - int(dp.source_fps // 2)) : lo + 1, jaw]))
                peak = float(np.max(st[lo : hi + 1, jaw]))
                trough = float(np.min(st[lo : hi + 1, jaw]))
                if not profile.robot.is_holding(pre):
                    continue
                # A slip moves the aperture out of the holding band in whichever direction that
                # arm's mechanism allows, so measure the larger excursion either way.
                excursion = max(peak - pre, pre - trough)
                left = profile.robot.hold_band[0]
                right = profile.robot.hold_band[1]
                escaped = peak > right or trough < left
                if require_signature and not escaped:
                    continue
                if excursion < min_overshoot:
                    continue
                if best is None or excursion > best[0]:
                    best = (excursion, name)
            if best is None:
                continue
            overshoot, arm = best
            out.append(
                Donor(
                    collection=collection,
                    episode=index,
                    event_index=i,
                    t_start=seg.start,
                    t_end=seg.end,
                    phase=ep.phase_at(seg.start),
                    arm=arm,
                    overshoot=overshoot,
                    outcome=ep.outcome,
                    label=seg.label,
                    terminal=terminal,
                    n_events_in_episode=len(events),
                )
            )
    out.sort(key=lambda d: -d.overshoot)
    return out


def by_phase(donors: list[Donor]) -> dict[str, list[Donor]]:
    grouped: dict[str, list[Donor]] = {}
    for d in donors:
        grouped.setdefault(d.phase or "unknown", []).append(d)
    return grouped


def write_labels(donors: list[Donor], path: Path) -> None:
    """Persist donors so a run can resolve an id back to a window.

    A manifest names donors by id, and a lookup that misses used to fall through to authoring a
    run from the host's own opening seconds -- 64 runs of a production batch were silently
    wasted that way. Writing the donors alongside the manifest is what keeps them in step.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_events": len(donors), "events": [d.to_dict() for d in donors]}, indent=2))


def load_labels(path: Path) -> list[Donor]:
    """Read a donor file, including one written by hand.

    Curated windows are how the behaviours no annotation vocabulary covers get in -- a gripper
    closing on nothing and then hunting for the object, say. Such a donor needs a window and an
    arm; ``phase`` may be ``any`` because it is not tied to one, and ``outcome`` is free text a
    recipe can select on.
    """
    import json

    raw = json.loads(Path(path).read_text())
    events = raw["events"] if isinstance(raw, dict) else raw
    known = {
        "source",
        "episode_index",
        "event_index",
        "t_start",
        "t_end",
        "phase",
        "arm",
        "overshoot",
        "outcome",
        "label",
        "terminal",
        "n_events_in_episode",
        "prompt",
        "tier",
    }
    out = []
    for e in events:
        for required in ("source", "episode_index", "t_start", "t_end", "arm"):
            if required not in e:
                raise ValueError(f"{path}: donor {e} is missing {required!r}")
        out.append(
            Donor(
                collection=e["source"],
                episode=int(e["episode_index"]),
                event_index=int(e.get("event_index", 0)),
                t_start=float(e["t_start"]),
                t_end=float(e["t_end"]),
                phase=e.get("phase"),
                arm=e["arm"],
                overshoot=float(e.get("overshoot", 0.0)),
                outcome=e.get("outcome", ""),
                label=e.get("label", ""),
                terminal=bool(e.get("terminal", False)),
                n_events_in_episode=int(e.get("n_events_in_episode", 1)),
                caption=e.get("prompt", "") or e.get("caption", ""),
                extra={k: v for k, v in e.items() if k not in known},
            )
        )
    return out
