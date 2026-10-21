# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Work-unit contract -- the cross-consumer seam.

RL, collection and eval differ ONLY in ``(UnitSource, TrajectorySink)``; the types here are
what all three agree on. Nothing engine-specific lives in this module, and its only import
outside the standard library is ``schema.identity`` -- itself stdlib-only, and shared rather
than re-implemented so one encoding rule produces every persisted identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from cosmos_framework.posttrain.schema.identity import digest


class Outcome(str, Enum):
    """How an episode ended.

    ``FAILURE`` is **data, not an error**: at round 3's measured 4.43% yield, ~95% of
    episodes legitimately end this way and every one of them is a valid training frame
    sequence. Only ``SIM_ERROR`` (and ``ABANDONED``) means the unit must be re-dispatched.
    Conflating the two is how a pipeline decides a campaign has catastrophically failed
    when it is working exactly as designed.
    """

    SUCCESS = "success"  # the sim's goal predicate was satisfied
    FAILURE = "failure"  # ran to completion, goal not satisfied  <- DATA, not an error
    SIM_ERROR = "sim_error"  # simulator/GPU fault -> the unit MUST be re-dispatched
    POLICY_ERROR = "policy_error"
    ABANDONED = "abandoned"  # lease expired / preempted mid-episode


#: Outcomes that produced a complete, keepable trajectory.
TERMINAL_DATA = (Outcome.SUCCESS, Outcome.FAILURE)


@dataclass(frozen=True)
class EpisodeKey:
    """Content-derived episode identity.

    NOT positional. Two workers that draw the same key must produce interchangeable data,
    and the sink uses :meth:`slug` as the output path -- so a key that encoded *which*
    worker or *when* would make resume impossible and duplicates undetectable. Round 1
    produced 102 duplicate packages precisely because identity was positional.
    """

    suite: str
    task: str
    instance: int
    seed: int
    attempt: int = 0

    def slug(self) -> str:
        """The output PATH, and therefore not a global identity. ``suite`` is deliberately out.

        Two keys differing only by ``suite`` produce the same slug, so a unit holding both
        would be half-covered by landing one. That is real, and it is fixed one layer up:
        :meth:`WorkUnit.make` REFUSES a mixed-suite unit, because the suite is already a field
        of the unit and an episode from another one is not work this unit describes.

        Fixed HERE rather than by widening the slug because this string is a filesystem path
        under the sink's root and a component of ``unit_id``, which names a lease file on NFS.
        Adding a field would re-identify every episode and every unit in every existing
        corpus: resume would stop recognising landed work and re-run it. The collision needs
        the two suites to meet inside one unit, and refusing that costs nothing.
        """
        return f"{self.task}/i{self.instance:04d}_s{self.seed}_a{self.attempt}"


def _snapshot(value: Any) -> Any:
    """Deep-copy into plain containers, so nothing the caller still holds can reach inside.

    Sequences become tuples and mappings become ``dict``. Not ``MappingProxyType``: a
    mappingproxy raises ``TypeError: cannot pickle`` under both ``copy.deepcopy`` and
    ``pickle``, and a ``WorkUnit`` crosses the gateway by those routes.

    ``str``/``bytes`` are excluded from the sequence branch explicitly -- they are Sequences,
    and iterating one would turn a task name into a tuple of characters.
    """
    if isinstance(value, Mapping):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return tuple(_snapshot(v) for v in value)
    return value


@dataclass(frozen=True)
class WorkUnit:
    """One scene build's worth of episodes.

    Sized in **predicted minutes**, not episode count: per-episode cost ranges from ~5 min
    (``turning_on_radio``, 3,224 demo steps) to ~23 min (``setting_mousetraps``, 15,294 at
    1.5x), so a fixed episode count gives 3-4x different overhead per task and makes any
    bin-packing plan meaningless.

    ``scenario["episodes_per_scene"]`` is the EPOCHS knob in disguise. Under EPOCHS>1,
    env poisoning stalled 101 windows / 572 episodes across 6 tasks in round 2 (all later
    recovered), and during that recovery a THIRD failure class surfaced -- an articulation
    -view corpse -- which was explicitly not covered by the gates written for the first
    two. It MUST be pilot-measured (gate G3), never inferred from an amortization curve.
    """

    unit_id: str
    suite: str
    # SIM-SPECIFIC and opaque to the core. b1k: {task, instances, num_envs, max_steps,
    # mode, seed, timeout_mult, episodes_per_scene}.
    scenario: Mapping[str, Any]
    episodes: tuple[EpisodeKey, ...]
    est_minutes: float
    group_id: str | None = None  # GRPO: G episodes sharing one prompt
    tags: Mapping[str, str] = field(default_factory=dict)

    @staticmethod
    def make(
        *,
        suite: str,
        scenario: Mapping[str, Any],
        episodes: Iterable[EpisodeKey],
        est_minutes: float,
        **kw: Any,
    ) -> "WorkUnit":
        """Build a unit whose ``unit_id`` is a digest of its own content.

        Two planners that emit the same work produce the same id, which is what makes a
        re-plan after a crash idempotent rather than a source of duplicate dispatch.

        The encoding lives in ``schema.identity``, not here, so the campaign plane and this
        module cannot drift into two rules that agree until the day they do not. It is FROZEN,
        pinned by ``tests/test_embodied_unit_identity.py``: a unit_id names a lease file on NFS,
        so re-identifying units mid-campaign would orphan every claim.

        **``episodes`` is materialised FIRST.** It is an iterable, and the digest walks it once:
        a generator was exhausted computing the id and the unit then stored ``()`` -- a unit
        whose id is a digest of episodes it does not contain, which every coverage check reads
        as complete. One conversion, used for both.

        **``scenario`` is SNAPSHOT, not stored.** ``WorkUnit`` is frozen; the mapping handed to
        it is not. A caller that mutated their dict afterwards changed ``unit.scenario`` while
        ``unit_id`` -- a digest taken before the mutation -- stayed the same, so a persisted
        lease named different work than it described.

        The snapshot is a deep copy into PLAIN containers rather than a
        ``MappingProxyType`` tree. Immutable proxies read better and cannot survive this seam:
        a mappingproxy cannot be pickled or deep-copied, and units cross the gateway wire by
        exactly those routes. Sequences become tuples, which is free; mappings stay ``dict``.
        Removing the ALIAS is what closes the reported hole -- the caller can no longer reach
        this state at all.
        """
        episodes = tuple(episodes)
        # A unit HAS a suite. An episode from another one is not work this unit describes, and
        # `slug()` cannot tell them apart -- see there for why the slug is not the thing that
        # moves. Checked before the digest so a mixed unit never acquires an id at all.
        if foreign := sorted({e.suite for e in episodes} - {suite}):
            raise ValueError(
                f"WorkUnit(suite={suite!r}) was given episodes from {foreign}. Episode slugs "
                f"omit the suite, so these collide with this unit's own keys and landing one "
                f"would mark both covered. Build one unit per suite."
            )
        scenario = _snapshot(scenario)
        return WorkUnit(
            digest(
                {
                    "suite": suite,
                    "scenario": dict(sorted(scenario.items())),
                    "eps": [e.slug() for e in episodes],
                }
            ),
            suite,
            scenario,
            episodes,
            est_minutes,
            **kw,
        )


@dataclass
class EpisodeResult:
    key: EpisodeKey
    outcome: Outcome
    steps: int
    reward_sum: float = 0.0
    lane: int = 0
    #: 0-based index within this scene build. THE poisoning signal (gate G3): compare
    #: outcome and SIM_ERROR rate at position 0 against position k.
    position_in_scene: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_uri: str | None = None
    error: str | None = None


@dataclass
class UnitResult:
    unit_id: str
    episodes: list[EpisodeResult]
    wall_s: float
    boot_s: float  # scene build -- the term the cost model amortizes
    host: str
    gpu_uuid: str | None = None
    error: str | None = None  # unit-level fault (cell died before any episode committed)
    #: Coarse bucket for ``error``, from ``embodied.sim.classify_error``: one of
    #: ``"transient"``, ``"fatal"``, ``"protocol"`` or ``"unclassified"``. ``None`` when the
    #: unit did not fault.
    #:
    #: A PLAIN STRING, never an ``Outcome`` or any other enum. ``str(Outcome.SUCCESS)`` is
    #: ``'Outcome.SUCCESS'`` and not ``'success'``, which is the shape of bug that survives
    #: review because both sides look right in isolation and only the persisted value is
    #: wrong.
    #:
    #: Nothing routes on it. It exists so the transient/fatal split can be MEASURED before a
    #: retry policy is written against it.
    error_class: str | None = None
    #: Slugs of episodes that were NOT run because they were already on disk, and so have
    #: no :class:`EpisodeResult` here. Only the skip-everything path sets this: an episode
    #: skipped *individually* is still stepped through the turn and reports normally.
    #:
    #: It exists because "did this unit finish" is answered by SLUG COVERAGE, not by a row
    #: count. Without it a unit whose episodes had all landed reports zero episodes, a
    #: source comparing ``len(result.episodes)`` against ``len(unit.episodes)`` reads that
    #: as "nothing landed", and the unit is re-dispatched forever -- burning one generation
    #: per pass until it is declared stuck, while its data has been on disk the whole time.
    skipped_landed: tuple[str, ...] = ()

    def covered(self) -> set[str]:
        """Slugs this run put (or found) on disk. The evidence a source retires a unit on."""
        return {e.key.slug() for e in self.episodes if e.artifact_uri} | set(self.skipped_landed)

    def missing(self, unit: "WorkUnit") -> set[str]:
        """Slugs ``unit`` asked for that this result does not account for.

        THE COVERAGE RULE, IN ONE PLACE. A source and its caller must both answer "did this
        unit finish" from here rather than each recomputing ``wanted - covered``: two copies of
        that predicate disagree invisibly, until a unit is either lost or re-dispatched forever.

        Empty means retire; non-empty means hand back. Note this is NOT ``len(episodes)``
        against ``len(unit.episodes)``: episodes that were already on disk are covered
        without producing a row (see :attr:`skipped_landed`).
        """
        return {e.slug() for e in unit.episodes} - self.covered()
