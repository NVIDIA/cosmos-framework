# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Everything about a robot, its cameras and its dataset that the pipeline must not assume.

The original scripts hardcoded a 14-D bimanual UR5e with grippers at columns 6 and 13, jaw
values that mean "free" at opposite ends per arm, and a three-camera canvas at a fixed
aspect. None of that generalises, and getting any of it wrong produces runs that look weak
rather than broken -- injections into an empty hand, or a slip written to the wrong end of the
gripper channel. So it all lives here, declared per robot, and the pipeline reads it.

The one piece of hard-won advice: measure ``jaw.free`` from frame zero of several episodes.
At the home pose the robot holds nothing, so whatever the channel reads there is the free
value, and it is not always the same end for both arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ArmSpec:
    """One arm: where its joints and jaw live in the state vector, and what its jaw values mean."""

    joints: tuple[int, int]  # [start, stop) columns of the revolute joints
    jaw: int  # column of the gripper channel
    free: float  # value the jaw reads while holding nothing (measure at frame zero)
    weak: float | None = None  # barely engaged, for partial-grasp scenarios

    @property
    def joint_columns(self) -> list[int]:
        return list(range(self.joints[0], self.joints[1]))


@dataclass(frozen=True)
class RobotProfile:
    state_dim: int
    arms: dict[str, ArmSpec]
    hold_band: tuple[float, float]  # jaw closed onto an object, rather than free or empty-shut
    phase_arm: dict[str, str]  # which arm holds the object during each phase
    # Physical travel of the gripper channel, from fully shut to fully open. A transplanted
    # excursion is clamped to it, since an aperture outside the mechanism's range is not a
    # failure the robot could produce. Left unset, the splice clamps to what the host and donor
    # actually reach, which is looser but needs no measurement.
    jaw_range: tuple[float, float] | None = None

    @property
    def jaw_columns(self) -> list[int]:
        return sorted(a.jaw for a in self.arms.values())

    @property
    def joint_columns(self) -> list[int]:
        return [c for c in range(self.state_dim) if c not in self.jaw_columns]

    def arm(self, name: str) -> ArmSpec:
        if name not in self.arms:
            raise KeyError(f"unknown arm {name!r}; profile defines {sorted(self.arms)}")
        return self.arms[name]

    def holding_arm(self, phase: str) -> str:
        """The arm carrying the object during a phase, defaulting to the first declared arm."""
        return self.phase_arm.get(phase, next(iter(self.arms)))

    def is_holding(self, value: float) -> bool:
        lo, hi = self.hold_band
        return lo <= float(value) <= hi


@dataclass(frozen=True)
class CameraProfile:
    """Camera feature keys by role, and how they are tiled into the frame the model sees.

    ``front_fraction`` is the share of canvas height taken by the wide view, with the wrist
    views side by side underneath. Scoring reads the wrist pane of the holding hand, because a
    slip is a handful of pixels in the wide view and the whole frame in the wrist view.
    """

    front: str
    wrist: dict[str, str] = field(default_factory=dict)  # arm name -> feature key
    front_fraction: float = 2.0 / 3.0


@dataclass(frozen=True)
class DatasetProfile:
    roots: dict[str, Path]  # collection name -> LeRobot dataset root
    annotations: str  # annotations file, relative to a root
    source_fps: float
    target_fps: int
    chunk_length: int
    clean_outcome: str  # outcome value marking an episode with no anomaly
    failure_labels: tuple[str, ...]  # segment labels that count as a failure event
    recovery_labels: tuple[str, ...] = ()
    normal_label: str = "normal"

    @property
    def stride(self) -> int:
        """Source frames per model frame."""
        s = round(self.source_fps / self.target_fps)
        if s < 1:
            raise ValueError(f"target_fps {self.target_fps} exceeds source_fps {self.source_fps}")
        return s

    @property
    def chunk_seconds(self) -> float:
        return self.chunk_length / self.target_fps

    def root(self, name: str) -> Path:
        if name not in self.roots:
            raise KeyError(f"unknown collection {name!r}; profile defines {sorted(self.roots)}")
        return self.roots[name]


@dataclass(frozen=True)
class Profile:
    robot: RobotProfile
    cameras: CameraProfile
    dataset: DatasetProfile
    model: dict[str, Any] = field(default_factory=dict)  # opaque, handed to the adapter

    @staticmethod
    def from_yaml(path: str | Path) -> "Profile":
        raw = yaml.safe_load(Path(path).read_text())
        return Profile.from_dict(raw, base=Path(path).parent)

    @staticmethod
    def from_dict(raw: dict, base: Path | None = None) -> "Profile":
        def resolve(p: str) -> Path:
            q = Path(p)
            return q if q.is_absolute() or base is None else (base / q).resolve()

        r = raw["robot"]
        arms = {
            name: ArmSpec(
                joints=tuple(spec["joints"]), jaw=int(spec["jaw"]), free=float(spec["free"]), weak=spec.get("weak")
            )
            for name, spec in r["arms"].items()
        }
        jaw_range = r.get("jaw_range")
        robot = RobotProfile(
            state_dim=int(r["state_dim"]),
            arms=arms,
            hold_band=(float(r["hold_band"][0]), float(r["hold_band"][1])),
            phase_arm=dict(r.get("phase_arm", {})),
            jaw_range=(float(jaw_range[0]), float(jaw_range[1])) if jaw_range else None,
        )
        c = raw["cameras"]
        cameras = CameraProfile(
            front=c["front"], wrist=dict(c.get("wrist", {})), front_fraction=float(c.get("front_fraction", 2.0 / 3.0))
        )
        d = raw["dataset"]
        dataset = DatasetProfile(
            roots={k: resolve(v) for k, v in d["roots"].items()},
            annotations=d["annotations"],
            source_fps=float(d["source_fps"]),
            target_fps=int(d["target_fps"]),
            chunk_length=int(d["chunk_length"]),
            clean_outcome=d["clean_outcome"],
            failure_labels=tuple(d["failure_labels"]),
            recovery_labels=tuple(d.get("recovery_labels", ())),
            normal_label=d.get("normal_label", "normal"),
        )
        model = dict(raw.get("model", {}))
        for key in ("normalizer_stats", "config_file", "training_contract_file"):
            if key in model:
                model[key] = str(resolve(model[key]))
        profile = Profile(robot=robot, cameras=cameras, dataset=dataset, model=model)
        _validate(profile)
        return profile


def _validate(p: Profile) -> None:
    """Catch profile mistakes here, where the message can be useful.

    Every one of these corresponds to a bug that cost a full experiment: a jaw column outside
    the state vector, an arm whose free value sits inside the holding band so nothing ever
    reads as a slip, or a wrist camera named for an arm that does not exist.
    """
    for name, arm in p.robot.arms.items():
        if not 0 <= arm.jaw < p.robot.state_dim:
            raise ValueError(f"arm {name}: jaw column {arm.jaw} outside state_dim {p.robot.state_dim}")
        if arm.joints[1] > p.robot.state_dim or arm.joints[0] < 0:
            raise ValueError(f"arm {name}: joints {arm.joints} outside state_dim {p.robot.state_dim}")
        if p.robot.is_holding(arm.free):
            raise ValueError(
                f"arm {name}: free value {arm.free} lies inside hold_band {p.robot.hold_band}, so a "
                "released jaw would still read as holding and no slip could be detected"
            )
    lo, hi = p.robot.hold_band
    if not lo < hi:
        raise ValueError(f"hold_band {p.robot.hold_band} is empty")
    if p.robot.jaw_range is not None:
        jlo, jhi = p.robot.jaw_range
        if not jlo < jhi:
            raise ValueError(f"jaw_range {p.robot.jaw_range} is empty")
        if not (jlo <= lo and hi <= jhi):
            raise ValueError(f"hold_band {p.robot.hold_band} falls outside jaw_range {p.robot.jaw_range}")
        for name, arm in p.robot.arms.items():
            if not jlo <= arm.free <= jhi:
                raise ValueError(f"arm {name}: free {arm.free} outside jaw_range {p.robot.jaw_range}")
    for arm in p.cameras.wrist:
        if arm not in p.robot.arms:
            raise ValueError(f"wrist camera declared for unknown arm {arm!r}")
    for phase, arm in p.robot.phase_arm.items():
        if arm not in p.robot.arms:
            raise ValueError(f"phase_arm[{phase!r}] names unknown arm {arm!r}")
    if not 0.0 < p.cameras.front_fraction <= 1.0:
        raise ValueError(f"front_fraction {p.cameras.front_fraction} must be in (0, 1]")
