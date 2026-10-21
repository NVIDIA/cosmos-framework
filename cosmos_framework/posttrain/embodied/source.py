# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The work plane.

Phase 1 ships :class:`StaticUnits`: the driver already holds the whole list, so there is
nothing to coordinate. The protocol is declared now anyway so that Phase 2's
``campaign/dirlease.py`` slots in without touching the role body.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol, Sequence

from cosmos_framework.posttrain.embodied.unit import UnitResult, WorkUnit


@dataclass
class Lease:
    unit: WorkUnit
    #: Fencing token, minted **after** a successful ``O_EXCL`` create, never before. Round 3
    #: minted it first, forked a journal, and 2 identities out of 16,000 produced a 100%
    #: delivery stop.
    token: str
    expires_at: float


class RetireDisposition(str, Enum):
    """What a source DID with a retirement it was handed.

    A source may decline: ``DirLeaseSource.complete`` refuses silently when a successor
    generation already owns the unit, because the work landed but this cell no longer holds
    the claim. Reporting that as a completion would make a cell whose leases were all stolen
    mid-unit look like its most productive one.

    Three members, and each one is an exit that ``DirLeaseSource.complete`` actually takes.
    ``error`` is NOT among them: a faulted unit is abandoned one branch earlier and never
    reaches ``complete`` at all, so it is the collector's verdict rather than a source's.

    ``INCOMPLETE`` is a SECOND-TRUST-BOUNDARY answer, not a duplicate of the collector's own
    coverage check. The collector abandons a short unit before retiring it, so on the normal
    path this never fires; the source re-checks anyway because ``.done`` is the only dispatch
    filter, and writing it for a unit whose episodes are not all on disk retires that unit
    permanently. When the re-check does fire, the source has released the claim and refused
    to retire -- which is neither a completion nor a stolen lease, and reporting it as either
    would misattribute it.
    """

    COMPLETED = "completed"
    REVOKED = "revoked"
    INCOMPLETE = "incomplete"


class UnitSource(Protocol):
    """Where work comes from."""

    def claim(self, *, budget_minutes: float, owner: str, prefer: Sequence[str] = ()) -> list[Lease]:
        """Take up to ``budget_minutes`` of work. May return fewer, or none."""

    def heartbeat(self, leases: Sequence[Lease]) -> list[Lease]:
        """Renew. Returns the leases still held -- a lease may have been revoked."""

    def complete(self, lease: Lease, result: UnitResult) -> RetireDisposition | None:
        """Retire a finished unit, and say whether the retirement was recorded.

        ``| None`` is in the SIGNATURE, not only in the prose below, because the collector
        genuinely accepts it today and a type that hides a supported return is a type a static
        checker will believe. It narrows to ``RetireDisposition`` when the affordance goes.

        Returns :attr:`RetireDisposition.COMPLETED` when this source took the result,
        :attr:`RetireDisposition.REVOKED` when it declined because a successor owns the unit,
        or :attr:`RetireDisposition.INCOMPLETE` when it declined because the result does not
        cover the unit's own episode set.

        A source written before this return existed yields ``None``; the collector treats
        that as "cannot say" and falls back to the counter it used to read, so an
        out-of-series implementation keeps working rather than being silently reported as
        completing everything. That fallback is a MIGRATION AFFORDANCE and is expected to go
        once ``campaign/dirlease.py`` returns the value -- it is not a second supported way
        to answer.
        """

    def abandon(self, lease: Lease, reason: str) -> None:
        """Return the unit to the pool immediately.

        THIS is what converts cell mortality from DATA LOSS into SCHEDULE DELAY. Today the
        driver bin-packs once, scatters once, records a dead cell's error and NEVER
        re-dispatches its bin -- measured as "15 dead cells cost 150 of 400 episodes".
        """

    def status(self) -> dict:
        """Work REMAINING, broken down. Part of the contract, not a convenience.

        A pull worker's loop condition and a campaign's terminal state are both read from
        here, and both need the same distinction: ``unclaimed`` and ``reclaimable`` are work
        this source will still hand out, ``held`` is work in flight elsewhere, and ``stuck``
        is work it will now refuse forever. A source that reported only a done-count would
        make "finished" and "wedged" the same observation -- round 3's C7 sweep sat in a
        fail-fast loop for hours behind exactly that ambiguity.

        Keys every implementation must supply: ``units``, ``done``, ``held``,
        ``reclaimable``, ``stuck``, ``unclaimed``.
        """

    def claimable(self) -> int:
        """Units this source could still hand out. **The pull loop's condition.**"""


class StaticUnits:
    """PHASE 1. A pushed list, with no coordination of any kind.

    Every unit is handed out exactly once, in order, and ``abandon`` genuinely returns work
    to the pool -- which costs nothing here and means the driver's retry path is exercised
    by the same code Phase 2 will replace. What this cannot do is coordinate across
    processes; two workers each constructed with the same list will each run all of it.
    That is the whole reason ``dirlease`` exists, and why Phase 1 scatters disjoint bins.
    """

    def __init__(self, units: Iterable[WorkUnit], *, lease_seconds: float = 3600.0, **kw: Any) -> None:
        del kw  # tolerate extra source_options so one config shape serves every source
        self._pending: list[WorkUnit] = list(units)
        self._total = len(self._pending)
        self._lease_seconds = float(lease_seconds)
        self._outstanding: dict[str, WorkUnit] = {}
        self._done: dict[str, UnitResult] = {}
        self._tokens = itertools.count()

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def outstanding(self) -> int:
        return len(self._outstanding)

    @property
    def results(self) -> list[UnitResult]:
        return list(self._done.values())

    def claim(self, *, budget_minutes: float, owner: str, prefer: Sequence[str] = ()) -> list[Lease]:
        """Take units until the next one would exceed ``budget_minutes``.

        Always yields at least one unit when any is pending, even if that single unit is
        over budget: refusing would deadlock a worker whose budget is smaller than the
        longest task, and the longest b1k task is ~23 min per episode.

        ``prefer`` is accepted and ignored. Honouring it here would mean keeping a scene
        open across claims, which is the cross-episode reuse round 3 settled as forbidden
        (spec §7.3) -- so the parameter exists for protocol conformance and does nothing.
        """
        del prefer  # see docstring: deliberately not honoured
        taken: list[Lease] = []
        spent = 0.0
        while self._pending:
            unit = self._pending[0]
            if taken and spent + unit.est_minutes > budget_minutes:
                break
            self._pending.pop(0)
            spent += unit.est_minutes
            self._outstanding[unit.unit_id] = unit
            taken.append(Lease(unit, f"{owner}:{next(self._tokens)}", time.monotonic() + self._lease_seconds))
        return taken

    def heartbeat(self, leases: Sequence[Lease]) -> list[Lease]:
        """Renew every lease still outstanding; drop any that was completed or abandoned."""
        live = [ln for ln in leases if ln.unit.unit_id in self._outstanding]
        for ln in live:
            ln.expires_at = time.monotonic() + self._lease_seconds
        return live

    def complete(self, lease: Lease, result: UnitResult) -> RetireDisposition:
        """Always records. Nothing else can hold this unit -- there is no other process."""
        self._outstanding.pop(lease.unit.unit_id, None)
        self._done[lease.unit.unit_id] = result
        return RetireDisposition.COMPLETED

    def status(self) -> dict:
        """The same shape ``DirLeaseSource.status`` reports, so one loop reads both.

        ``reclaimable`` and ``stuck`` are always 0: there is no expiry here because there is
        no other process to expire against, and nothing is ever refused. ``held`` is what
        this process has out on lease -- which for a single-process source is also the only
        work "in flight anywhere".
        """
        return {
            "units": self._total,
            "done": len(self._done),
            "held": len(self._outstanding),
            "reclaimable": 0,
            "stuck": 0,
            "stuck_units": [],
            "unclaimed": len(self._pending),
        }

    def claimable(self) -> int:
        return len(self._pending)

    def remaining(self) -> int:
        """Units not yet retired. NOT a loop condition -- see :meth:`claimable`."""
        return self._total - len(self._done)

    def abandon(self, lease: Lease, reason: str) -> None:
        """Put the unit back at the FRONT of the queue.

        Front, not back: an abandoned unit is the one most likely to have partially landed
        episodes, and ``already_landed`` makes a re-run of it cheap. Sending it to the back
        maximises the time before that cheapness is collected, and on a wall-clock-bounded
        run can mean it is never collected at all.
        """
        del reason  # recorded by the caller in UnitResult.error
        unit = self._outstanding.pop(lease.unit.unit_id, None)
        if unit is not None:
            self._pending.insert(0, unit)
