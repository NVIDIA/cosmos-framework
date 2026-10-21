# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The simulator seam.

Deliberately **not** the declared ``Env`` Protocol at ``roles/env/base.py`` (``reset()`` /
``step()``), which zero simulators in this repo implement -- its own docstring concedes as
much. The lane-batched, caller-driven shape below is what three independent in-tree
implementations converged on: ``roles/env/the external simulator backend``'s vec loop, the fork's
``eval_vec.VecRunner``, and cosmos-rl's B1K rollout. The convergence is the argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome, WorkUnit
from cosmos_framework.posttrain.schema.trajectory import ObsBundle

#: Bumped when this module's contract changes in a way a backend must notice. Compared at
#: bind time by ``validate_sim_role``, not at every call: the check exists to fail
#: BEFORE a scene build, and a per-call version compare would be pure overhead on a seam
#: that already crosses a process boundary 5,300 times per episode.
#:
#: It is deliberately ONE integer and not a (major, minor) pair: the sim image and the
#: framework image are built from one tree, so nothing here can run two protocol versions
#: at once.
PROTOCOL_VERSION = 1


# -- errors -------------------------------------------------------------------
#
# CLASSIFY, DO NOT RE-ROUTE. Nothing in the driver or the collector changes its behaviour on
# these classes: nobody knows the transient/fatal split, so the distribution is collected
# first and the retry policy is written against measured numbers afterwards.


class SimError(Exception):
    """Base for everything the simulator raises across this seam."""


class SimTransient(SimError):
    """This unit may be re-dispatched unchanged.

    A scene-load flake or a recoverable engine fault: the next attempt has a real chance,
    on this process or another.
    """


class SimFatal(SimError):
    """This PROCESS is bad -- the card, the driver, or Kit itself.

    A different unit dispatched here will fail the same way. The distinction from
    :class:`SimTransient` is about the PROCESS, not about the unit, which is why it cannot
    be inferred from how many times a unit has failed.
    """


class SimProtocolError(SimError):
    """The contract was violated: a capability mismatch, a use-after-close, or a payload
    that is not wire-safe.

    **Never a data condition.** An episode that fails, a task whose goal is unsatisfiable,
    a lane that truncates -- none of those are protocol errors. Raising this for a data
    condition is how a class meant to mean "the code on the far side is wrong" comes to
    mean "something happened".
    """


def classify_error(exc: BaseException) -> str:
    """Coarse bucket for :attr:`~..unit.UnitResult.error_class`. A plain string, always.

    NOT an enum, and not ``type(exc).__name__``. The exception's own class name is already
    recorded verbatim in ``UnitResult.error`` (as ``repr(exc)``), so repeating it here would
    add a second copy of a thing we have rather than the thing we lack -- which is the
    coarse bucket a distribution can be built over.

    ``"unclassified"`` is the honest answer for everything that has not been triaged yet,
    and today that is nearly everything: no in-tree raise site uses these classes. A bucket
    named ``"transient"`` that silently absorbed every unclassified fault would report a
    tidy split that nobody measured.
    """
    if isinstance(exc, SimProtocolError):
        return "protocol"
    if isinstance(exc, SimFatal):
        return "fatal"
    if isinstance(exc, SimTransient):
        return "transient"
    return "unclassified"


# -- wire safety --------------------------------------------------------------

#: Scalars that survive every transport this seam has to cross today (Ray's cloudpickle)
#: and every transport it is meant to reach tomorrow (a length-prefixed frame over a unix
#: socket, with arrays in shared memory). ``bool`` is a subclass of ``int`` and is listed
#: for the reader, not for ``isinstance``.
_WIRE_SCALARS = (str, int, float, bool, type(None))


def assert_wire_safe(obj: Any, where: str) -> None:
    """Raise :class:`SimProtocolError` if ``obj`` holds anything that must not cross the seam.

    Allowed: :class:`numpy.ndarray` (non-object dtype), numpy scalars, ``str``, ``int``,
    ``float``, ``bool``, ``None``, and ``dict`` / ``list`` / ``tuple`` of those. Forbidden:
    torch tensors, engine handles, callables, sets, and any other custom class.

    **Called once per unit on the first observation and once per episode on ``metrics()``.
    NEVER per step.** The walk is over structure rather than bytes -- an array is one node,
    not the measured 3,917,128 -- but the per-step budget on this seam is the whole reason
    the observation is cached.

    An ``object``-dtype ndarray is rejected rather than accepted as "an ndarray": it is a
    box holding arbitrary Python, so a torch tensor smuggled inside one would pass a naive
    type check and then pull the whole engine into the unpickling process. The test is
    ``dtype.hasobject``, not ``dtype == object``: a STRUCTURED dtype with an object field --
    ``np.empty(1, dtype=[("payload", object)])`` -- is not equal to ``object`` and holds one
    just as effectively.

    Containers must be EXACTLY ``dict``/``list``/``tuple``. A subclass walks identically and
    carries state this walk never visits: a ``defaultdict``'s ``default_factory`` is a
    callable, and any subclass may hold attributes in ``__dict__``. cloudpickle serialises
    that hidden state, so a payload of "safe items" can still drag an engine handle across.

    This is what ``sim_decouple.md`` §4.2's "pickle is forbidden" reduces to under today's
    topology, where the transport is still Ray and cannot be swapped: we cannot stop
    cloudpickle from running, so we assert on the payload it would be handed instead.

    An :class:`~schema.trajectory.ObsBundle` is UNWRAPPED rather than rejected: the declared
    rule is about the values reachable from ``representations``, and every caller on this seam
    holds the bundle.
    """
    _walk_wire(obj.representations if isinstance(obj, ObsBundle) else obj, where, ())


def _walk_wire(obj: Any, where: str, path: tuple[str, ...]) -> None:
    if isinstance(obj, np.ndarray):
        # `hasobject`, NOT `dtype == object`. A structured dtype with an object field --
        # `[("payload", object)]` -- compares UNEQUAL to `object` while holding arbitrary
        # Python in exactly the way this rejects, so the equality test admitted the case it
        # was written to catch.
        if obj.dtype.hasobject:
            raise SimProtocolError(
                f"{where}{_at(path)} is an ndarray whose dtype {obj.dtype!r} holds Python "
                f"objects. That is a box holding arbitrary Python, not an array -- whatever "
                f"is inside crosses the seam unchecked. Store the values themselves, or a "
                f"numeric array."
            )
        return
    if isinstance(obj, (np.generic, *_WIRE_SCALARS)):
        return
    # TYPE, not isinstance, for containers. A subclass satisfies the walk and still carries
    # state the walk cannot see: `defaultdict.default_factory` is a callable, and any subclass
    # may hold instance attributes. cloudpickle ships that hidden state, so "every item is
    # safe" stops implying "the object is safe".
    if type(obj) is dict:
        for k, v in obj.items():
            if not isinstance(k, str):
                raise SimProtocolError(
                    f"{where}{_at(path)} has a non-string key {k!r} ({type(k).__name__}). "
                    f"Every transport this seam must reach serialises keys as strings."
                )
            _walk_wire(v, where, (*path, k))
        return
    if type(obj) in (list, tuple):
        for i, v in enumerate(obj):
            _walk_wire(v, where, (*path, str(i)))
        return
    if isinstance(obj, (Mapping, list, tuple)):
        raise SimProtocolError(
            f"{where}{_at(path)} is {type(obj).__module__}.{type(obj).__name__}, a container "
            f"SUBCLASS. Its items may all be wire-safe while the object itself carries state "
            f"this check never sees -- a `default_factory`, an instance attribute -- which the "
            f"transport serialises anyway. Rebuild it as a plain dict/list/tuple at the "
            f"boundary that produced it."
        )
    raise SimProtocolError(
        f"{where}{_at(path)} is {type(obj).__module__}.{type(obj).__name__}, which is not "
        f"wire-safe. Allowed: ndarray, numpy scalar, str, int, float, bool, None, and "
        f"dict/list/tuple of those. Convert it at the boundary that produced it -- a value "
        f"that only survives because both ends happen to share an interpreter is exactly "
        f"what this seam exists to eliminate."
    )


def _at(path: tuple[str, ...]) -> str:
    return f" at {'.'.join(path)}" if path else ""


@dataclass(frozen=True)
class SimCapabilities:
    """What a simulator declares about itself, at ``probe()``, BEFORE any scene build.

    Declared rather than measured, and that is the contract: the whole value of this object
    is that a mismatch costs a handshake instead of an Isaac boot plus a six-minute scene
    build. Anything that can only be known after a scene exists does not belong here -- see
    ``describe_obs`` (SPEC §4) for the measured counterpart.
    """

    protocol_version: int
    backend: str  # the registry name -- "behavior1k"
    backend_version: str  # image digest / engine version, best effort, "" when unknown
    embodiment: str  # MUST equal the EmbodimentMap name the role was built with
    max_lanes: int
    #: Logical camera name -> ``(H, W)`` the backend is CONFIGURED to render. Not the
    #: measured frame shape: ``sinks/package.py`` records that per episode precisely because
    #: a wrapper that ignores the request would otherwise produce a corpus whose provenance
    #: states a resolution it does not have. This is the request; that is the truth.
    camera_shapes: Mapping[str, tuple[int, int]]
    #: The proprio width the embodiment DECLARES it needs -- ``EmbodimentMap.state_dim``,
    #: which ``to_record_view`` already refuses to go under. It is deliberately not the raw
    #: vector's width, which is wider, varies, and is unmeasured on this path.
    proprio_width: int
    extras: tuple[str, ...] = ()
    supports_render_skip: bool = False


@dataclass
class StepOut:
    """The result of advancing one lane by one control step."""

    lane: int
    #: POST-step observation. The session MUST cache this so the next ``observe()`` is
    #: free -- see :meth:`SimSession.observe`, whose invariant 6 says this is the SAME
    #: object that call returns.
    #:
    #: **This field has zero readers today. Do not delete it.** The driver reads the
    #: observation through ``observe()``, so a reader looking for uses finds none and
    #: concludes the field is dead. Under a process split ``observe()`` and ``step()``
    #: collapse into one ``step(request) -> StepResult`` round trip, and this field is the
    #: half that carries the answer back. Deleting it buys one dataclass slot and costs a
    #: second round trip per step, 5,300 times per episode.
    obs: ObsBundle
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class SimSession(Protocol):
    """One scene build, holding ``n_lanes`` concurrent episodes.

    Seven invariants, each silent when broken. They are numbered because the driver cites
    them by number, and each is argued at the member that owns it:

    1. ``observe()`` is CACHED -- it must not step, re-render or re-read sensors.
    2. A lane absent from ``step(actions)`` gets no ``StepOut``.
    3. The session owns TRUNCATION: a lane at ``scenario["max_steps"]`` reports
       ``truncated=True`` and becomes ``done()``.
    4. ``n_lanes`` is the lanes bound by the most recent ``advance_turn()``, and SHRINKS on
       the final turn.
    5. ``outcome()`` and ``metrics()`` are LATCHED terminal values, read once at commit time
       -- possibly hundreds of steps after the lane finished.
    6. ``observe(lanes)[i]`` **IS** the :attr:`StepOut.obs` the last ``step()`` emitted for
       that lane -- the same object, not an equal one. This is what makes ``observe()``
       free.
    7. Every method raises :class:`SimProtocolError` after ``close()``.

       This is the only part of generation fencing that is testable in-process, and it is
       the part that catches the real bug -- a session used after its scene was released.
       The failure it replaces is not an exception: a closed OmniGibson session's arrays
       are stale rather than absent, so the loop keeps running and writes the LAST frame of
       the previous scene over and over, at full length, into a package that passes every
       structural gate.
    """

    #: Lanes bound by the most recent :meth:`advance_turn`. It may SHRINK on the final
    #: turn: a unit of 20 episodes at V=6 ends with a turn of 2, and the driver iterates
    #: ``range(n_lanes)`` to decide which lanes to open writers for. Reporting the vec's
    #: full width on a partial turn would open writers for lanes holding a previous
    #: episode's finished state, and commit them a second time.
    n_lanes: int

    #: Increments once per :meth:`Simulator.open`, i.e. once per scene build, for the life
    #: of the process. Stamped into provenance so a package can be attributed to the scene
    #: that produced it.
    #:
    #: It is NOT a claim generation and must never be reconciled against one. The fleet's
    #: work-claim already has a generation counter in ``campaign/dirlease.py``, arbitrated
    #: over NFS across hosts; this one is process-local and counts scene builds. Two
    #: counters in this subsystem have already produced one false double-claim report.
    generation: int

    def lane_key(self, lane: int) -> EpisodeKey:
        """Which episode this lane is currently running."""

    def observe(self, lanes: Sequence[int]) -> list[ObsBundle]:
        """Return the current observation for each lane. **CACHED.**

        This is THE fixed data interface of the seam, and the reason it is this method
        rather than the recorded frame: ``driver.py``'s step loop forks one ``observe()``
        value into both projections -- ``to_policy_obs`` for the policy and
        ``to_record_view`` for the sink. Typing the trunk types both branches; typing
        either branch types neither the other one nor the value they share. The frame is
        already ``schema.Transition`` and never crosses this boundary at all, because the
        sink runs sim-side.

        MUST NOT step, re-render, or re-read sensors.

        b1k: return the ``obs_list`` that ``vec.step()`` already produced, or the reset
        observation on turn entry. This contract is load-bearing. A naive ``env.get_obs()``
        here adds up to ``n_lanes x max_steps`` full-resolution readbacks (720^2 plus two
        480^2 per lane) that the production loop deliberately avoids by calling
        ``get_obs()`` only for lanes that need a new action chunk. Implementations that
        cannot cache should not implement this protocol.
        """

    def dropped_obs_keys(self) -> tuple[str, ...]:
        """Raw keys the engine emitted that this embodiment does not declare.

        Read once per unit by the driver and published on the progress heartbeat. Today
        these keys still cross the wire inside the observation; making them VISIBLE is the
        prerequisite for dropping them, because nobody currently knows what is in the set
        or how large it is.

        Sorted, so two runs of the same scene produce the same string and a diff means a
        difference.
        """

    def step(self, actions: Mapping[int, Any], *, render: bool = True) -> list[StepOut]:
        """Advance the given lanes by one control step.

        A lane absent from ``actions`` is not advanced by the caller's intent. Simulators
        that can only step the whole vec should pass a zero action for absent lanes and
        MUST NOT emit a ``StepOut`` for them -- the driver counts steps and writes frames
        from the returned list, so an extra entry silently fabricates a training frame.

        ``render=False`` is the opt-in L1 render-skip. The driver forces ``True`` whenever
        a sink is recording, because a render-skipped step has no pixels and therefore
        cannot be a training frame.

        **The session owns truncation.** A lane that reaches ``scenario["max_steps"]`` MUST
        be reported with ``truncated=True`` on that step and MUST become ``done()`` --
        exactly as the production b1k loop does (``steps[i] >= max_steps -> done[i]``). The
        driver bounds its own loop by ``max_steps`` as a guard against a session that never
        stops, but it cannot patch the flag onto a frame it has already handed to a writer,
        so it abandons the turn instead. Every frame must carry its own terminal state.
        """

    def done(self, lane: int) -> bool:
        """Whether this lane's episode has terminated or truncated."""

    def outcome(self, lane: int) -> Outcome:
        """LATCHED final outcome for a done lane.

        The driver opens a writer for every lane at turn start and commits them all after
        the step loop ends, so this is read once, at commit time -- for a lane that may have
        finished hundreds of steps earlier and been stepped with zero actions ever since.
        The value must be the one the lane ENDED on.

        A session that recomputes it live (`SUCCESS if env.task.success else FAILURE`,
        evaluated now) reports whatever the goal predicate says after all that extra
        physics. An episode that satisfied its goal at step 100 and is committed at step 800
        can be labelled FAILURE, at a ~4.4% measured yield where the label IS the product,
        with no exception raised anywhere. Latch it in `step()` when the lane goes done.
        """

    def metrics(self, lane: int) -> dict[str, Any]:
        """LATCHED aggregate per-episode metrics for a done lane.

        Same read-at-commit-time contract as :meth:`outcome`. Metrics that are accumulated
        per step (fed only for lanes the driver named) satisfy it for free; anything derived
        from live engine state at call time does not.

        The driver does NOT take reward from here -- it sums ``StepOut.reward`` itself, so
        the episode's scalar and its per-step column always agree. Whatever this returns is
        recorded verbatim alongside them.
        """

    def advance_turn(self) -> bool:
        """Bind the next batch of this unit's episodes onto the lanes.

        Returns ``False`` when the unit is exhausted. Reuses the built scene -- so the
        number of turns IS ``episodes_per_scene``, i.e. the EPOCHS knob. See
        :class:`~cosmos_framework.posttrain.embodied.unit.WorkUnit`.
        """

    def close(self) -> None:
        """Release the scene. Must be safe to call after a partial or failed turn."""


class Simulator(Protocol):
    """Process-level handle. Built once per worker (Isaac/Kit boot); ``open()``ed per unit."""

    def capabilities(self) -> SimCapabilities:
        """What this backend is, declared without booting a scene.

        An implementation that builds a scene to answer this has inverted it.
        """

    def probe(self) -> None:
        """Raise if this process cannot render.

        NOTE: the PRIMARY card probe belongs in the POD BOOTSTRAP, before ``ray start``,
        and it is already there -- ``utils/launch_multinode_eval.sh`` checks the assigned
        GPU UUIDs against ``isaac_bad_cards.json`` well before either raylet starts. This
        method is the second line of defence, for a card that goes bad between harvests.
        """

    def open(self, unit: WorkUnit) -> SimSession:
        """Build the scene for ``unit`` and return a session bound to it."""

    def wants_restart(self) -> bool:
        """Should the caller tear this process down and build a fresh one? Asked BETWEEN units.

        A QUESTION, never the act. The loop asks it in the one window where it holds no
        lease -- after the previous unit is retired and before the next is claimed -- so an
        engine that answers True is never the reason a claimed unit is stranded. An engine
        with no reason to recycle answers False forever, which is why this is cheap and why
        it must not boot, render, or touch a card to answer.
        """

    def gpu_uuid(self) -> str | None:
        """The physical GPU this process was handed, for bad-card attribution."""

    def close(self) -> None:
        """Shut down the simulator process-level handle."""
