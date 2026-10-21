# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The trajectory sink -- the capability that does not exist on the i4 path today.

The b1k eval path is outcome-only: it records success, steps and metrics, and throws the
trajectory away. Collection needs every frame. This is the seam where that difference lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome
from cosmos_framework.posttrain.schema.trajectory import ObsBundle, Transition

# ── the v1 episode-package contract ─────────────────────────────────────────
#
# Defined HERE, in the core, and imported by both the writer (``sinks/package.py``) and the
# checker (``reconcile.py``) -- BOTH of which are on an archive branch and not in this series, so this
# constant has no second reader in this tree yet. It lived in both, which is a contract
# duplicated between a
# producer and its gate: add a camera and the writer emits six files while the reconciler
# still demands five, so every package in the campaign is quarantined for "unexpected entry"
# -- a clean-looking structural error with no hint that the two lists disagree.
#
# The layering allows only this direction. A backend may import the core; the core may not
# import a backend, or it stops loading in the venvs that have no simulator.

#: The three camera streams, in the order ``to_record_view`` names them.
CAMERA_FILES = ("head.mp4", "left_wrist.mp4", "right_wrist.mp4")

#: Every file a published package must contain, besides its markers.
REQUIRED_FILES = ("episode.json", "frames.npz") + CAMERA_FILES


def package_id_for(key: EpisodeKey) -> str:
    """Directory name for an episode. Derived from identity ALONE.

    In the core, with ``REQUIRED_FILES``, because three components now agree on it: the
    writer names the directory, the reconciler files it under that name, and the census
    turns a planned :class:`EpisodeKey` back into the path where its marker would be. The
    third one is what moved it here -- a census that re-derived the format would be a second
    implementation of the layout, and the failure mode is silent: every planned episode
    reads as missing while the packages sit on disk under the name the writer used.

    The converter requires ``pkg_dir.name == episode.json["package_id"]`` but does not parse
    the id, so the format is free -- and making it content-derived is what turns
    ``already_landed`` into a ``stat()`` instead of a ledger lookup.
    """
    return f"{key.task}_i{key.instance:04d}_s{key.seed}_a{key.attempt}"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streamed, because a package's mp4s do not belong in memory to be hashed."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_frame(
    *,
    step_idx: int,
    view: Mapping[str, Any],
    action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    policy_info: Mapping[str, Any],
    episode_id: str = "",
) -> Transition:
    """Build the framework's :class:`~schema.trajectory.Transition` for one frame.

    One function so the alignment convention is written once, in the type that carries it.
    It is normative and silent-corruption-shaped:

        Frame ``t`` carries the observation captured **before** ``a_t``, the action ``a_t``,
        and the reward / terminated / truncated produced by **executing** ``a_t``. The first
        frame pairs the reset observation with ``a_0``. The final post-episode observation is
        dropped. ``len(frames) == len(executed actions)``. The last frame has exactly one of
        ``terminated | truncated`` true.

    Get it wrong and the data looks fine and trains badly.

    ``terminated`` is not stored: it is ``done and not truncated``. Keeping both would let
    them disagree, and a frame that claims to be terminal-and-truncated is not a state any
    consumer can interpret. :func:`is_terminated` is the reader.

    The observation goes in ``ObsBundle.representations`` as ``to_record_view`` produced it
    -- named camera streams plus proprio arrays. It is NOT copied: the writer contract is
    that a frame passed to ``append`` is consumed before return, so copying here would
    double the peak footprint of the only thing in this pipeline big enough to matter.
    """
    if terminated and truncated:
        raise ValueError(
            f"frame {step_idx}: terminated and truncated are both set. The contract is at "
            f"most one, because a time-limit cut is not a terminal state and bootstrapping "
            f"through the two differs."
        )
    return Transition(
        obs=ObsBundle(representations=dict(view), modality_tags=("rgb", "proprio"), id=episode_id),
        action=action,
        # A named component, per DESIGN 1.5. "final" is left for the RL sink to fill from the
        # outcome; the per-step env reward is not a terminal signal and labelling it "final"
        # would make Trajectory.final_reward report the last step's reward as the return.
        reward={"env": float(reward)},
        policy_info=dict(policy_info),
        step_idx=step_idx,
        done=bool(terminated or truncated),
        truncated=bool(truncated),
        info=dict(info),
    )


def is_terminated(frame: Transition) -> bool:
    """The episode ended by reaching a terminal state, not by hitting a time limit."""
    return bool(frame.done and not frame.truncated)


class EpisodeWriter(Protocol):
    """Accumulates one episode and commits it as one artifact."""

    def append(self, frame: Transition) -> None:
        """Consume one frame.

        MUST NOT retain full-resolution frames. Contract: any frame passed here is consumed
        -- encoded, or queued to a BOUNDED ring -- before return.

        At 720/480 a retained frame is ~1.5 MB per lane; at V=5 and T=15,000 that is well over
        100 GB of anonymous memory. RLinf measured the same defect at ~51 GB/epoch, and a
        collection job is strictly worse: it records EVERY episode rather than the ~5% that
        succeed.
        """

    def commit(self, outcome: Outcome, metrics: Mapping[str, Any]) -> str:
        """Finalize; return an artifact URI.

        Writes ``READY`` **last**, and never ``DONE``. ``DONE`` is a reconciler-written
        validation marker: a writer that certifies its own output is a stage trusting itself.
        """

    def abort(self, reason: str) -> None:
        """Discard the episode.

        Leaves NO marker, so the partial artifact is invisible by definition rather than
        by a downstream filter that someone has to remember to write.
        """


class TrajectorySink(Protocol):
    """Where a worker's episodes go.

    **Two attributes are OPTIONAL and are therefore not declared here.** Declaring them would
    make them structurally required -- `NullSink` omits both and would stop conforming -- so
    the Protocol states only what every sink must implement, and the optional pair is read
    through :func:`sink_info` rather than off the object:

    ``wants_extrinsics: bool`` -- whether this sink CONSUMES per-frame camera extrinsics
    (``robot2cam``). Absent means yes; see :func:`sink_wants_extrinsics` for why the default
    is ``True`` and not ``False``.

    ``sr: Mapping[str, Sequence[int]]`` -- per-task ``[successes, attempts]`` over the
    episodes this sink COMMITTED, or absent. The two sinks in this tree disagree on purpose:
    an eval sink counts outcomes because that is its whole product, a package sink writes
    frames and has no opinion.

    Read through :func:`sink_info`, never with a bare ``getattr`` at the call site -- a
    default spelled twice is a default that can differ.
    """

    def open_episode(self, key: EpisodeKey, meta: Mapping[str, Any]) -> EpisodeWriter:
        """Begin an episode. ``meta`` is provenance, persisted with the artifact."""

    def already_landed(self, key: EpisodeKey) -> str | None:
        """Return the artifact URI if this episode is already complete, else ``None``.

        Content-addressed resume: one ``stat()`` on the ``READY`` marker. This is what
        stops a preempted or bad-card-killed job re-running the instances it already
        finished, and it is checked before the scene build, not after.
        """

    def heartbeat(self, snapshot: Mapping[str, Any]) -> None:
        """Publish liveness/progress.

        The sim actor runs on the host main thread with no concurrency group, so no other
        method can run while a unit is executing -- for hours. The filesystem is the only
        progress protocol available, and it is what distinguishes STUCK from SLOW.
        """

    def close(self) -> None:
        """Flush and release. Must be safe to call more than once."""


def sink_wants_extrinsics(sink: Any) -> bool:
    """Does ``sink`` consume ``robot2cam``? Defaults to ``True`` when it does not say.

    Producing extrinsics costs one base-link pose read plus a ``get_position_orientation()``
    and a ``relative_pose_transform`` per camera on EVERY boxing call -- measured in-pod at
    **2.383 ms/step**, which is ~83% of the whole +2.87 ms/step collector-vs-legacy gap that
    C7 differenced. For a sink that drops the view it is pure waste, and this is what lets
    the session skip it.

    **THE DEFAULT IS ``True``, AND THAT DIRECTION IS THE WHOLE POINT.** An undeclared sink
    under ``True`` pays 2.4 ms/step it did not need -- a throughput regression, which is
    measurable and gets found. Under ``False`` it lands an **all-zero** ``robot2cam`` in a
    package that passes every structural gate the reconciler has, and only a human comparing
    camera poses would ever notice.

    So a sink OPTS OUT explicitly and silence means "give me the data".
    """
    return sink_info(sink).wants_extrinsics


@dataclass(frozen=True)
class SinkInfo:
    """What a sink declares about itself, resolved ONCE by one set of rules.

    Every ``getattr(sink, ..., default)`` spelled at a call site carries its own default, and
    a default written at two call sites can be written differently at each. So the rules live
    here, once, and the reader is a value rather than a convention.

    NOT CACHED, AND THAT IS DELIBERATE. ``sr`` is a running total the sink accumulates as it
    commits episodes; a snapshot taken when the body was built would report zeros. Call this
    where the answer is needed.
    """

    #: See :func:`sink_wants_extrinsics` for why absence means ``True`` and not ``False``.
    wants_extrinsics: bool
    #: ``None`` when the sink has no opinion, which is NOT the same as ``{}``. An eval that
    #: resumed every episode from disk committed nothing and reports ``{}``; emitting that as
    #: a result would print "0 tasks | overall SR 0.0% (0/0)" over a run that wrote every row
    #: correctly. Both collapse to "omit the key", but only because a caller that wanted to
    #: tell them apart could.
    success_rates: Mapping[str, Sequence[int]] | None

    def summary_fields(self) -> dict[str, Any]:
        """The ``sr`` key for a body's return dict, or nothing. One rule, both loops."""
        return {"sr": self.success_rates} if self.success_rates else {}


def sink_info(sink: Any) -> SinkInfo:
    """Resolve the optional parts of the sink contract. Never raises on a partial sink."""
    sr = getattr(sink, "sr", None)
    return SinkInfo(
        wants_extrinsics=bool(getattr(sink, "wants_extrinsics", True)),
        success_rates=sr if isinstance(sr, Mapping) else None,
    )
