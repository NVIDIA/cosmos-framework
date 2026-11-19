# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Turn rendered runs into a LeRobot dataset for training a failure predictor.

Four things happen here that are worth stating up front, because each is a decision rather than
a mechanical conversion.

**Controls ship as labelled negatives.** A model that only ever sees failures cannot learn to
predict them. Every injection here has a matched control -- the same episode at the same instant,
replayed untouched -- and that is the negative that isolates the failure from ordinary motion.

**Real successes can ship alongside them.** Every failure is model-rendered, so a dataset whose
only negatives were real footage could be separated on rendering alone. The controls close that
hole; real episodes then add what controls cannot, which is real pixels and the whole task rather
than a window around an injection. Both are resampled and resized to one geometry, and
``annotation.is_generated`` records which is which so the difference can be measured.

**States and actions are reconstructed, then proved.** The rendered runs stored only normalized
actions, so the physical trajectory is rebuilt from the run's record with the same splice that
drove generation. It is then re-normalized and compared against what the model was actually fed;
a run that does not match exactly is dropped rather than shipped, because a plausible-looking but
wrong trajectory would teach the model dynamics that never happened.

**The task string stays neutral.** The caption used to steer generation names the failure. Putting
it in the task field would hand the model the answer it is supposed to predict, so captions live
in the metadata and every episode carries the same neutral instruction.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from . import annotations as ann
from . import balance, legacy
from .config import Profile
from .scoring import RATIO_THRESHOLD, RunScore, score_directory
from .splice import continuous, rechunk, splice, windows

NEUTRAL_TASK = "Pick the reels from the box and stack them in the black tote."
# 0 is reserved for "no failure", so controls, real successes and unlabelled frames all agree.
TYPE_INDEX = {
    "control": 0,
    "success": 0,
    "pickup_slip": 1,
    "handover_loss": 2,
    "groping": 3,
    "repeated": 4,
    "terminal": 5,
    "partial_slip": 6,
}
# The current filename, and the one the pre-package scripts used; a batch may hold either.
ACTION_FILES = ("actions_normalized.json", "actions.json")


@dataclass
class Episode:
    """One clip about to be written: where it came from, what drove it, and what it means.

    Two sources end up side by side. ``generated`` episodes are rendered by the world model, from
    a run directory. ``real`` episodes are recorded footage of a successful take, decimated to the
    model's frame rate and resized to its pane sizes so both look alike to a trainer.
    """

    name: str
    run_dir: Path | None
    record: dict
    is_failure: bool
    ratio: float | None = None
    control_of: str | None = None  # for a control, an injection it is the yardstick for
    states: np.ndarray | None = None
    actions: np.ndarray | None = None
    verified: bool = False
    source: str = "generated"  # "generated" | "real"
    collection: str | None = None  # for real episodes: which dataset it was recorded in
    episode_index: int | None = None

    @property
    def is_generated(self) -> bool:
        return self.source == "generated"

    @property
    def family(self) -> str:
        return self.record.get("recipe") or ("success" if self.source == "real" else "control")

    @property
    def host_episode(self) -> int:
        return int(self.record["host"]["episode_index"])

    @property
    def donor_id(self) -> str | None:
        d = self.record.get("donor")
        return f"{d['episode_index']}:{d['event_index']}" if d else None

    @property
    def onset_frame(self) -> int:
        fps = int(self.record.get("fps", 5))
        return int(round(float(self.record.get("failure_time_s", 0.0)) * fps))


# -- selection ---------------------------------------------------------------


def select(
    profile: Profile,
    root: Path,
    n: int = 100,
    threshold: float = RATIO_THRESHOLD,
    whole_families: tuple[str, ...] = (),
    per_host: int | None = None,
    with_controls: bool = True,
) -> list[Episode]:
    """The strongest injections, spread over families, hosts and donors, plus their controls.

    ``threshold`` is the quality gate: how many times its own control a clip must change to count
    as a failure at all. Everything above it is eligible.

    When more clips are eligible than ``n``, the surplus is spent on coverage rather than on the
    top of the ranking. Taking the highest ratios alone returns almost nothing but pickup slips,
    since that family is both the largest and the strongest, so families are cycled first, then
    hosts, then donors. ``whole_families`` forces a family to be taken entire before the rest is
    filled, and ``per_host`` caps how much any one episode can contribute.
    """
    eligible = [s for s in score_directory(profile, root) if s.ratio is not None and s.ratio >= threshold]
    if not eligible:
        raise ValueError(f"no run under {root} beats its control by {threshold}x")
    records = {s.name: legacy.read_record(root / s.name) for s in eligible}

    if n >= len(eligible):
        chosen = list(eligible)
    else:
        by_family: dict[str, list[RunScore]] = defaultdict(list)
        for s in eligible:
            by_family[s.recipe or "unknown"].append(s)
        chosen = []
        for fam in whole_families:
            chosen += sorted(by_family.get(fam, []), key=lambda s: -s.ratio)
        chosen = chosen[:n]
        rest = [s for s in eligible if (s.recipe or "unknown") not in whole_families]
        chosen += balance.spread(
            rest,
            n - len(chosen),
            primary=lambda s: s.recipe or "unknown",
            secondary=lambda s: records[s.name]["host"]["episode_index"],
            rank=lambda s: s.ratio,
        )
    if per_host is not None:
        chosen = balance.spread(
            chosen,
            len(chosen),
            primary=lambda s: records[s.name]["host"]["episode_index"],
            secondary=lambda s: s.recipe,
            rank=lambda s: s.ratio,
        )
        kept: list[RunScore] = []
        seen: Counter = Counter()
        for s in chosen:
            host = records[s.name]["host"]["episode_index"]
            if seen[host] < per_host:
                kept.append(s)
                seen[host] += 1
        chosen = kept

    episodes = [
        Episode(name=s.name, run_dir=root / s.name, record=records[s.name], is_failure=True, ratio=s.ratio)
        for s in sorted(chosen, key=lambda s: (s.recipe or "", -s.ratio))
    ]
    if with_controls:
        episodes += _controls_for(root, episodes)
    return episodes


def _controls_for(root: Path, failures: list[Episode]) -> list[Episode]:
    """Every control in the batch, tagged with the injection it is the yardstick for.

    All of them ship, not only the ones paired with a selected clip: an untouched replay is a
    valid negative whether or not its own injection made the cut, and negatives are the scarcer
    half of a failure-prediction dataset.
    """
    paired = {
        (
            e.host_episode,
            e.record.get("target_phase"),
            round(float(e.record["host"]["t_inject"]), 2),
            e.record.get("arm"),
        ): e.name
        for e in failures
    }
    out = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record = legacy.read_record(run_dir)
        if record is None or record.get("mode") != "control":
            continue
        if not (run_dir / "synth.mp4").is_file():
            continue
        key = (
            int(record["host"]["episode_index"]),
            record.get("target_phase"),
            round(float(record["host"]["t_inject"]), 2),
            record.get("arm"),
        )
        out.append(
            Episode(name=run_dir.name, run_dir=run_dir, record=record, is_failure=False, control_of=paired.get(key))
        )
    return out


# -- physical states and actions ---------------------------------------------


def reconstruct(profile: Profile, record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Absolute states and the actions that drove them, in physical units.

    Mirrors the rollout exactly: grounded chunks from the host, then the spliced chunks. States
    are ``(frames, state_dim)``; actions are joint deltas with absolute gripper columns, the same
    representation the model was conditioned on, one row per frame with a held final row.
    """
    dp = profile.dataset
    chunk = dp.chunk_length
    host = record["host"]
    pre, post = int(record["pre_chunks"]), int(record["post_chunks"])
    fps = int(record.get("fps", dp.target_fps))

    host_states = ann.strided_state(profile, host["source"], int(host["episode_index"]))
    inject_idx = int(round(float(host["t_inject"]) * fps))
    start_idx = max(0, inject_idx - pre * chunk)
    host_windows = windows(host_states, start_idx, pre + post, chunk)

    program = list(host_windows[:pre])
    if record.get("mode") == "control" or not record.get("donor"):
        program += list(host_windows[pre:])
    else:
        donor = record["donor"]
        donor_states = ann.strided_state(profile, donor["source"], int(donor["episode_index"]))
        donor_windows = windows(donor_states, int(round(float(donor["t_start"]) * fps)), post, chunk)
        spliced = splice(
            profile,
            continuous(host_windows[pre:], chunk),
            continuous(donor_windows, chunk),
            dims=record.get("dims") or "all",
            blend_frames=int(record.get("blend_frames") or 0),
            jaw_amplitude=float(record.get("jaw_amplitude") or 1.0),
            src_arm=donor.get("arm"),
            dst_arm=record.get("arm"),
        )
        program += rechunk(spliced, post, chunk)

    states = continuous(program, chunk)
    actions = np.zeros_like(states)
    jaws = profile.robot.jaw_columns
    actions[:-1] = states[1:] - states[:-1]
    actions[:, jaws] = states[:, jaws]
    actions[-1, profile.robot.joint_columns] = 0.0  # nothing follows the last frame
    return states.astype(np.float32), actions.astype(np.float32)


def verify(profile: Profile, episode: Episode, normalizer) -> float:
    """Largest disagreement between the reconstruction and what the model was actually fed.

    The rendered runs stored the normalized action chunks they were driven with, so this is a true
    round trip rather than a self-consistency check: reconstruct, re-normalize, compare. Both the
    current and the pre-package filenames are accepted, since a batch can contain either.
    """
    stored_path = next((episode.run_dir / n for n in ACTION_FILES if (episode.run_dir / n).is_file()), None)
    if stored_path is None:
        return math.nan
    stored = np.asarray(json.loads(stored_path.read_text()), dtype=np.float32)
    chunk = profile.dataset.chunk_length
    n_chunks = len(stored)
    rebuilt = np.stack([normalizer(episode.states[k * chunk : k * chunk + chunk + 1]) for k in range(n_chunks)]).astype(
        np.float32
    )
    if rebuilt.shape != stored.shape:
        return math.inf
    return float(np.abs(rebuilt - stored).max())


def prepare(
    profile: Profile, episodes: list[Episode], normalizer=None, tolerance: float = 1e-5
) -> tuple[list[Episode], list[tuple[str, str]]]:
    """Reconstruct every episode, and drop any whose round trip does not match.

    Rejections carry a reason rather than a bare number, because the two causes need different
    responses: a mismatch means the reconstruction is wrong, while nothing to compare against
    means the run was rendered without recording its actions.
    """
    kept, rejected = [], []
    for ep in episodes:
        if not ep.is_generated:
            kept.append(ep)  # recorded states are the ground truth; there is nothing to rebuild
            continue
        ep.states, ep.actions = reconstruct(profile, ep.record)
        if normalizer is None:
            kept.append(ep)
            continue
        err = verify(profile, ep, normalizer, tolerance)
        if math.isnan(err):
            rejected.append((ep.name, f"no {' or '.join(ACTION_FILES)} to verify against"))
            continue
        if err > tolerance:
            rejected.append((ep.name, f"actions differ by {err:.2g}"))
            continue
        ep.verified = True
        kept.append(ep)
    return kept, rejected


def quantile_normalizer(profile: Profile):
    """The model's own action encoder, for the round trip, without loading a checkpoint.

    It has to be the same encoding the adapter uses, or the verification would compare a
    reconstruction against a differently-encoded reference and pass while meaning nothing, so both
    call one function.
    """
    from .adapters.cosmos import CosmosForwardDynamics, encode_chunk

    stats_path = profile.model.get("normalizer_stats")
    if not stats_path:
        raise ValueError("model.normalizer_stats is required to verify reconstructions")
    stats = CosmosForwardDynamics._load_stats(Path(stats_path))
    jaws = profile.robot.jaw_columns
    chunk = profile.dataset.chunk_length

    return lambda states: encode_chunk(states, stats, jaws, chunk)


# -- sources and video --------------------------------------------------------


def real_successes(profile: Profile, collection: str, require_phases: tuple[str, ...] = ()) -> list[Episode]:
    """Recorded successful episodes, as a second source alongside the generated clips.

    They matter for a reason beyond volume. Every failure in an export like this is model-rendered,
    so if the only negatives were real footage a classifier could separate the classes on
    rendering alone and never look at the gripper. The generated controls close that hole; the
    real episodes then add what controls cannot -- footage of the whole task, in real pixels.
    """
    from . import hosts as hosts_mod

    out = []
    for host in hosts_mod.find_hosts(profile, collection, require_phases=require_phases, limit=None):
        states = ann.strided_state(profile, collection, host.index)
        out.append(
            Episode(
                name=f"real_success__{collection}_ep{host.index}",
                run_dir=None,
                is_failure=False,
                source="real",
                collection=collection,
                episode_index=host.index,
                states=states.astype(np.float32),
                actions=_actions_from(profile, states),
                verified=True,  # recorded, so there is nothing to verify against a model
                record={
                    "name": f"real_success__{collection}_ep{host.index}",
                    "recipe": "success",
                    "mode": "real",
                    "arm": None,
                    "target_phase": None,
                    "fps": profile.dataset.target_fps,
                    "failure_time_s": None,
                    "host": {"source": collection, "episode_index": host.index, "t_start": 0.0, "t_inject": None},
                    "donor": None,
                    "duration_s": round(host.duration, 2),
                },
            )
        )
    return out


def _actions_from(profile: Profile, states: np.ndarray) -> np.ndarray:
    """Joint deltas with absolute gripper columns: the same action meaning for every source.

    A real episode also records a commanded-position column, but writing that into ``action``
    would give one feature two different meanings inside a single dataset, which no consumer
    could untangle. It is recomputed from measured state instead, exactly as for a generated clip.
    """
    actions = np.zeros_like(states, dtype=np.float32)
    actions[:-1] = states[1:] - states[:-1]
    actions[:, profile.robot.jaw_columns] = states[:, profile.robot.jaw_columns]
    actions[-1, profile.robot.joint_columns] = 0.0
    return actions


def frames_for(
    profile: Profile, episode: Episode, shapes: dict[str, tuple[int, int, int]]
) -> dict[str, list[np.ndarray]]:
    """The episode's camera streams, whatever its source, at the dataset's pane sizes."""
    if episode.is_generated:
        streams = split_canvas(profile, episode.run_dir / "synth.mp4")
    else:
        streams = _real_streams(profile, episode)
    return {key: [_fit(f, shapes[key]) for f in frames] for key, frames in streams.items()}


def _fit(frame: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if frame.shape == tuple(shape):
        return frame
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.uint8)


def _real_streams(profile: Profile, episode: Episode) -> dict[str, list[np.ndarray]]:
    """Recorded cameras decimated to the model's frame rate.

    Decoded sequentially and sampled by stride rather than by seeking: these are long clips and
    seeking per frame in a temporally compressed stream is both slower and less exact.
    """
    stride = profile.dataset.stride
    cams = profile.cameras
    keys = {cams.front: cams.front}
    for arm, key in cams.wrist.items():
        keys[key] = key
    out: dict[str, list[np.ndarray]] = {}
    for key in keys:
        path = ann.video_path(profile, episode.collection, episode.episode_index, key)
        frames = []
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                if i % stride == 0:
                    frames.append(np.asarray(frame.to_image(), dtype=np.uint8))
        out[key] = frames
    return out


def split_canvas(profile: Profile, mp4: Path) -> dict[str, list[np.ndarray]]:
    """The generated multi-camera canvas, cut back into one stream per camera.

    The model renders every view into a single frame: the wide view on top, the wrist views side
    by side underneath. Training wants them as separate cameras, and the geometry is the same one
    scoring already uses, so the two cannot drift apart.
    """
    cams = profile.cameras
    order = sorted(cams.wrist)
    streams: dict[str, list[np.ndarray]] = {cams.front: []}
    for arm in order:
        streams[cams.wrist[arm]] = []

    with av.open(str(mp4)) as container:
        for frame in container.decode(container.streams.video[0]):
            img = np.asarray(frame.to_image(), dtype=np.uint8)
            h, w = img.shape[:2]
            top = int(round(h * cams.front_fraction))
            streams[cams.front].append(np.ascontiguousarray(img[:top]))
            tile = w // max(1, len(order))
            for i, arm in enumerate(order):
                streams[cams.wrist[arm]].append(np.ascontiguousarray(img[top:, i * tile : (i + 1) * tile]))
    return streams


# -- labels ------------------------------------------------------------------


def labels(episode: Episode, n_frames: int, fps: int) -> dict[str, np.ndarray]:
    """Per-frame supervision: is a failure under way, and how far off is its onset.

    ``time_to_onset`` is positive before the failure and negative after, which is what lets a
    model be trained to anticipate rather than merely to recognise. Controls carry NaN for it,
    not zero, so a horizon loss cannot silently treat "no failure" as "failing right now".
    """
    if episode.is_failure:
        onset = episode.onset_frame
        active = (np.arange(n_frames) >= onset).astype(np.int64)
        time_to = (onset - np.arange(n_frames)).astype(np.float32) / fps
        severity = np.full(n_frames, episode.ratio or np.nan, dtype=np.float32)
        type_index = np.full(n_frames, TYPE_INDEX.get(episode.family, 0), dtype=np.int64)
    else:
        active = np.zeros(n_frames, dtype=np.int64)
        time_to = np.full(n_frames, np.nan, dtype=np.float32)
        severity = np.full(n_frames, np.nan, dtype=np.float32)
        type_index = np.zeros(n_frames, dtype=np.int64)
    return {
        "annotation.failure.active": active,
        "annotation.failure.time_to_onset": time_to,
        "annotation.failure.type_index": type_index,
        "annotation.failure.severity_ratio": severity,
        # Model-rendered or real footage. Carried per frame so the render shortcut can be
        # measured and ablated rather than assumed away.
        "annotation.is_generated": np.full(n_frames, int(episode.is_generated), dtype=np.int64),
    }


def provenance(episode: Episode, index: int, checkpoint: str) -> dict:
    r = episode.record
    donor = r.get("donor") or {}
    return {
        "episode_index": index,
        "run": episode.name,
        "source": episode.source,
        "source_dir": str(episode.run_dir) if episode.run_dir else None,
        "checkpoint": checkpoint if episode.is_generated else None,
        "is_failure": episode.is_failure,
        # "control" for a generated replay, "success" for real footage: both are negatives, but
        # calling recorded footage a control would misdescribe where it came from.
        "failure_type": episode.family,
        "failure_type_index": TYPE_INDEX.get(episode.family, 0) if episode.is_failure else 0,
        "onset_frame": episode.onset_frame if episode.is_failure else None,
        "onset_s": r.get("failure_time_s") if episode.is_failure else None,
        "severity_ratio": episode.ratio,
        "caption": r.get("caption") if episode.is_failure else None,
        "host": {
            "collection": r["host"]["source"],
            "episode_index": episode.host_episode,
            "t_start": r["host"]["t_start"],
            "t_inject": r["host"]["t_inject"],
        },
        "donor": {
            "episode_index": donor.get("episode_index"),
            "event_index": donor.get("event_index"),
            "t_start": donor.get("t_start"),
            "phase": donor.get("phase"),
            "arm": donor.get("arm"),
        }
        if donor
        else None,
        "target_phase": r.get("target_phase"),
        "arm": r.get("arm"),
        "transplant": {
            "dims": r.get("dims"),
            "blend_frames": r.get("blend_frames"),
            "jaw_amplitude": r.get("jaw_amplitude"),
        }
        if episode.is_failure
        else None,
        "pose_distance": r.get("pose_distance"),
        "control_of": episode.control_of,
        "actions_verified": episode.verified,
    }


# -- writing -----------------------------------------------------------------


def source_joint_names(profile: Profile, collection: str) -> list | None:
    """Reuse the source dataset's joint naming so the export is legible next to the real data."""
    try:
        return ann.info(profile, collection)["features"]["action"].get("names")
    except Exception:
        return None


def features(profile: Profile, joint_names) -> dict:
    state_dim = profile.robot.state_dim
    feats = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (state_dim,), "names": joint_names},
        "annotation.failure.active": {"dtype": "int64", "shape": (1,), "names": None},
        "annotation.failure.time_to_onset": {"dtype": "float32", "shape": (1,), "names": None},
        "annotation.failure.type_index": {"dtype": "int64", "shape": (1,), "names": None},
        "annotation.failure.severity_ratio": {"dtype": "float32", "shape": (1,), "names": None},
        "annotation.is_generated": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for key, size in _camera_shapes(profile).items():
        feats[key] = {"dtype": "video", "shape": size, "names": ["height", "width", "channels"]}
    return feats


def _camera_shapes(profile: Profile, canvas: tuple[int, int] = (640, 560)) -> dict[str, tuple]:
    h, w = canvas
    cams = profile.cameras
    top = int(round(h * cams.front_fraction))
    order = sorted(cams.wrist)
    tile = w // max(1, len(order))
    shapes = {cams.front: (top, w, 3)}
    for arm in order:
        shapes[cams.wrist[arm]] = (h - top, tile, 3)
    return shapes


def write(
    profile: Profile,
    episodes: list[Episode],
    out: Path,
    repo_id: str,
    checkpoint: str,
    joint_names=None,
    vcodec: str = "libsvtav1",
) -> dict:
    """Write a LeRobot v3.0 dataset, one episode per selected run."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = profile.dataset.target_fps
    # The generated clips define the pane sizes, and real footage is resized to them. Sizing the
    # other way round would mean upscaling wrist views 2.25x into detail that was never rendered.
    reference = next((e for e in episodes if e.is_generated), None)
    if reference is not None:
        shapes = {k: tuple(v[0].shape) for k, v in split_canvas(profile, reference.run_dir / "synth.mp4").items()}
    else:
        shapes = _camera_shapes(profile)
    feats = features(profile, joint_names)
    for key, shape in shapes.items():
        feats[key]["shape"] = tuple(shape)

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=feats, root=out, robot_type="UR5e", use_videos=True, vcodec=vcodec
    )

    manifest = []
    for index, ep in enumerate(episodes):
        streams = frames_for(profile, ep, shapes)
        n_frames = min(len(ep.states), min(len(v) for v in streams.values()))
        lab = labels(ep, n_frames, fps)
        for t in range(n_frames):
            frame = {
                "observation.state": ep.states[t],
                "action": ep.actions[t],
                "task": NEUTRAL_TASK,
                **{k: np.asarray([v[t]], dtype=v.dtype) for k, v in lab.items()},
                **{key: streams[key][t] for key in streams},
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        entry = provenance(ep, index, checkpoint)
        entry["n_frames"] = n_frames
        manifest.append(entry)

    meta = {
        "description": "Induced robot failures generated with a forward-dynamics world model, "
        "paired with matched controls.",
        "task": NEUTRAL_TASK,
        "type_index": TYPE_INDEX,
        "n_episodes": len(manifest),
        "n_failures": sum(1 for m in manifest if m["is_failure"]),
        "n_controls": sum(1 for m in manifest if not m["is_failure"] and m["source"] == "generated"),
        "n_real_successes": sum(1 for m in manifest if m["source"] == "real"),
        "episodes": manifest,
    }
    (out / "meta" / "failure_labels.json").write_text(json.dumps(meta, indent=2))
    return meta


def summarise(episodes: list[Episode]) -> dict:
    fails = [e for e in episodes if e.is_failure]
    return {
        "episodes": len(episodes),
        "failures": len(fails),
        "controls": sum(1 for e in episodes if not e.is_failure and e.is_generated),
        "real_successes": sum(1 for e in episodes if e.source == "real"),
        "by_family": dict(Counter(e.family for e in fails)),
        "by_host": dict(Counter(e.host_episode for e in fails)),
        "by_phase": dict(Counter(e.record.get("target_phase") for e in fails)),
        "by_arm": dict(Counter(e.record.get("arm") for e in fails)),
        "donors": len({e.donor_id for e in fails if e.donor_id}),
        "ratio_median": round(float(np.median([e.ratio for e in fails])), 3) if fails else None,
        "ratio_range": [round(min(e.ratio for e in fails), 2), round(max(e.ratio for e in fails), 2)]
        if fails
        else None,
    }
