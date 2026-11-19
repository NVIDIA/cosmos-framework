# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The annotation contract, and the dataset reads that go with it.

Two tracks per episode are required. A **phase** track saying what the robot was attempting at
each instant, which is what makes an injection point meaningful rather than a guess at a
timestamp. And an **outcome/segment** track marking the failures, which is what supplies
donors. Without the second the pipeline has nothing to transplant; see
docs/annotation-format.md.

One structural warning encoded here: phases repeat. An episode with two pick cycles has two
``handover`` windows, and keying them by name keeps only the last, which silently mislabels
every event in the first cycle and hides most donors. Windows are therefore a list throughout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import DatasetProfile, Profile


@dataclass(frozen=True)
class Window:
    name: str
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Segment:
    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Episode:
    collection: str
    index: int
    outcome: str
    reviewed: bool
    phases: list[Window]
    segments: list[Segment]

    def phase_at(self, t: float) -> str | None:
        for w in self.phases:
            if w.contains(t):
                return w.name
        return None

    def windows(self, phase: str) -> list[Window]:
        return [w for w in self.phases if w.name == phase]

    def has_phases(self, names: tuple[str, ...]) -> bool:
        present = {w.name for w in self.phases}
        return all(n in present for n in names)

    def failures(self, profile: DatasetProfile) -> list[Segment]:
        return [s for s in self.segments if s.label in profile.failure_labels]

    def is_clean(self, profile: DatasetProfile) -> bool:
        if self.outcome != profile.clean_outcome:
            return False
        return not [s for s in self.segments if s.label != profile.normal_label]


class AnnotationError(ValueError):
    pass


def load_episodes(profile: Profile, collection: str) -> dict[int, Episode]:
    """Parse the annotation file for one collection, validating as we go."""
    root = profile.dataset.root(collection)
    path = root / profile.dataset.annotations
    if not path.is_file():
        raise AnnotationError(
            f"no annotations at {path}. This pipeline transplants real recorded failures, so it "
            "needs a phase track and a failure track; see docs/annotation-format.md"
        )
    raw = json.loads(path.read_text())
    if "episodes" not in raw:
        raise AnnotationError(f"{path}: expected a top-level 'episodes' object")

    out: dict[int, Episode] = {}
    for name, e in raw["episodes"].items():
        try:
            index = int(str(name).split("_")[-1])
        except ValueError as exc:
            raise AnnotationError(f"{path}: episode key {name!r} does not end in an index") from exc
        phases = [Window(str(p["phase"]), float(p["start"]), float(p["end"])) for p in e.get("phase_segments") or []]
        segments = [Segment(str(s["label"]), float(s["start"]), float(s["end"])) for s in e.get("segments") or []]
        out[index] = Episode(
            collection=collection,
            index=index,
            outcome=str(e.get("outcome", "")),
            reviewed=bool(e.get("reviewed", False)),
            phases=phases,
            segments=segments,
        )
    if not out:
        raise AnnotationError(f"{path}: no episodes")
    return out


@lru_cache(maxsize=256)
def _read_state(root_str: str, data_path: str, episode: int, state_dim: int) -> np.ndarray:
    table = pq.read_table(
        Path(root_str) / data_path.format(episode_chunk=0, episode_index=episode), columns=["observation.state"]
    )
    flat = table["observation.state"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(-1, state_dim)


def info(profile: Profile, collection: str) -> dict:
    return json.loads((profile.dataset.root(collection) / "meta" / "info.json").read_text())


def state(profile: Profile, collection: str, episode: int) -> np.ndarray:
    """Measured state at source rate, shape (frames, state_dim).

    Measured, not commanded: a slip shows up as the jaw aperture moving while the hold command
    stays put, so the command channel does not contain the failure at all.
    """
    meta = info(profile, collection)
    return _read_state(str(profile.dataset.root(collection)), meta["data_path"], episode, profile.robot.state_dim)


def strided_state(profile: Profile, collection: str, episode: int) -> np.ndarray:
    """State resampled to the model's rate."""
    return state(profile, collection, episode)[:: profile.dataset.stride]


def video_path(profile: Profile, collection: str, episode: int, key: str) -> Path:
    meta = info(profile, collection)
    return profile.dataset.root(collection) / meta["video_path"].format(
        episode_chunk=0, video_key=key, episode_index=episode
    )


def duration(profile: Profile, collection: str, episode: int) -> float:
    return len(state(profile, collection, episode)) / profile.dataset.source_fps
