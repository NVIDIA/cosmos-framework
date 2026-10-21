# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Env role -- the collector body.

Bound into a ``_HostMainThread`` actor. Isaac must construct ``SimulationApp`` on the process
main thread (``workers/ray.py``'s ``_HostMainThreadImpl`` exists for that one reason), which
is why this is a role body and not a pipeline stage.

Under ``sim_transport=gateway`` this runs in a SUBPROCESS of the gateway actor rather than on
a ``sim_slot`` raylet, and the main-thread requirement is then satisfied by the subprocess's
own main thread -- so the ``_HostMainThread`` binding above is still used but no longer needed
on that path. Left in place because it is bind-level and not scopeable per-`Colocation`.

The body is deliberately thin. Everything that varies between RL, collection and evaluation
is the ``(sim, policy, sink)`` triple it is constructed with; the loop itself lives once, in
``embodied/driver.py``.

Two entry points, and the difference between them is the whole of Phase 2:

``run_units``
    PUSH. The driver bin-packs once and scatters disjoint bins. A cell that dies takes its
    remaining bin with it -- measured in round 3 as *15 dead cells cost 150 of 400
    episodes*.

``run_from_source``
    PULL. Every cell is given the same plan and claims from a shared
    :class:`~embodied.source.UnitSource`. A dead cell's unit expires and another cell takes
    it, which is what converts cell mortality from DATA LOSS into SCHEDULE DELAY.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from cosmos_framework.posttrain.embodied.driver import run_unit
from cosmos_framework.posttrain.embodied.embodiment import get_embodiment
from cosmos_framework.posttrain.embodied.registry import build_sim, build_sink, build_source
from cosmos_framework.posttrain.embodied.rollout import validate_rollout
from cosmos_framework.posttrain.embodied.sink import sink_info, sink_wants_extrinsics
from cosmos_framework.posttrain.embodied.source import Lease, RetireDisposition
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, UnitResult, WorkUnit
from cosmos_framework.posttrain.roles.env.sim_role import capabilities_as_wire

#: How often, at most, to renew a held lease while a unit is running. The driver's progress
#: callback fires every 200 control steps -- roughly every 12 s at the measured 0.06 s/step
#: -- and a lease renewal is an ``os.utime`` on shared storage, so renewing on every callback
#: would put 40 pods' worth of pointless writes on the mount. The TTL is measured in minutes;
#: 30 s of slack against it is free.
_RENEW_EVERY_S = 30.0

#: Backoff bounds for an idle pull worker. The lower bound is short because an empty claim
#: right after another cell releases a unit is pure contention; the upper bound is long
#: because a full ``claim`` pass is a stat ladder per unit, and 40 idle pods polling a
#: 3,000-unit campaign at the lower bound would be several thousand stats a second against
#: shared storage for no work at all.
_IDLE_POLL_S = 10.0
_IDLE_POLL_MAX_S = 120.0


class CollectorBody:
    """Runs work units on one cell, recording through a sink."""

    def __init__(
        self,
        *,
        sim: str = "behavior1k",
        sink: str = "package",
        embodiment: str = "r1pro_b1k",
        sim_options: dict | None = None,
        sink_options: dict | None = None,
        force_render: bool = True,
        render_every: int = 1,
    ) -> None:
        # SINK FIRST, AND THE ORDER IS LOAD-BEARING: the simulator is built with a flag read
        # off the sink, so the sink has to exist. `sink_wants_extrinsics` carries what
        # `robot2cam` costs and why silence there means "give me the data".
        #
        # An explicit `sim_options["emit_extrinsics"]` WINS over the sink's declaration. That
        # is for the one case the declaration cannot express: forcing extrinsics on under a
        # dropping sink to time or debug their cost. Turning them OFF under a sink that
        # CONSUMES them is refused below rather than honoured.
        self._sink = build_sink(sink, **(sink_options or {}))
        sim_options = dict(sim_options or {})
        wanted = sink_wants_extrinsics(self._sink)
        if "emit_extrinsics" in sim_options and wanted and not sim_options["emit_extrinsics"]:
            raise ValueError(
                f"sim_options['emit_extrinsics'] is False but the {sink!r} sink CONSUMES "
                "extrinsics. That combination writes an all-zero `robot2cam` into an "
                "otherwise-valid v1 package -- correct file count, valid hevc, a hash chain "
                "that verifies -- and nothing downstream would flag it. Drop the override, "
                "or use a sink that declares `wants_extrinsics = False`."
            )
        sim_options.setdefault("emit_extrinsics", wanted)
        self._sim = build_sim(sim, embodiment=embodiment, **sim_options)
        # AFTER `build_sim`, not before, and the order is the whole fix. Building the sim is
        # what imports the backend module, and importing the backend is what registers its
        # `EmbodimentMap` -- so resolving the embodiment first made the advertised lazy plugin
        # path order-dependent: in a fresh sim-server process `CollectorBody(sim="example",
        # embodiment="example")` died on a bare `KeyError: 'example'` unless the caller already
        # knew to preload the backend. `get_embodiment` rather than `MAPS[...]` for the second
        # half of the same failure: it names what IS registered, which a KeyError does not.
        self._emb = get_embodiment(embodiment)
        self._force_render = bool(force_render)
        self._render_every = int(render_every)
        self._owner = f"{os.uname().nodename}-{os.getpid()}"
        #: Every lease claimed and not yet retired, under ``run_from_source``. ``_heartbeat``
        #: is called from inside the rollout loop and renews ALL of them, not just the one
        #: currently running: a source is allowed to hand out several at once, and a lease we
        #: are holding but not yet working on ages out exactly like one we abandoned. Empty
        #: under ``run_units``, where the renewal is skipped rather than guessed at.
        self._source: Any = None
        self._held: list[Lease] = []
        self._last_renew = 0.0
        #: Leases that came back revoked from a renewal -- someone judged us dead and took
        #: the unit. Counted ONCE per lease: a revoked lease is still held (the teardown path
        #: needs it), so it comes back revoked on every renewal for the rest of the unit, and
        #: a naive increment would report one stolen lease as one per 30 seconds of runtime.
        self._revoked = 0
        self._revoked_ids: set[str] = set()
        #: Retirements the source refused, observed at ``complete`` time. NOT the same event
        #: as above and deliberately not merged with it: one is what we noticed while
        #: running, the other is what we found when we tried to finish, and a single counter
        #: would report one stolen lease as two.
        self._declined = 0

    def run_window(self, units: Sequence[Any] = (), *, policy: Any, version: int, decisions: int) -> dict:
        """Collect decision feedback through the optional chunk-simulator capability."""
        from cosmos_framework.posttrain.embodied.window import collect_window

        if units:
            raise ValueError("window collection does not accept episode work units")
        if "decision_chunks" not in self._sim.capabilities().extras:
            raise ValueError("simulator does not provide decision_chunks")
        return collect_window(self._sim.open_window(version), policy, version=version, decisions=decisions)

    def run_evaluation(
        self,
        units: Sequence[Any] = (),
        *,
        policy: Any,
        cases: list[dict[str, int | str]],
        version: int,
        seed: int,
        decisions: int,
    ) -> dict:
        from cosmos_framework.posttrain.embodied.window import evaluate_episodes

        if units or "fixed_case_evaluation" not in self._sim.capabilities().extras:
            raise ValueError("simulator does not support fixed-case evaluation without episode work units")
        import random

        import numpy as np

        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        before = self._sim.online_state_digest()
        try:
            result = evaluate_episodes(self._sim, policy, cases=cases, version=version, seed=seed, decisions=decisions)
            if self._sim.online_state_digest() != before:
                raise RuntimeError("evaluation changed the online simulator state")
            return {**result, "training_session_unchanged": True}
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def run_units(
        self,
        units: Sequence[Any],
        *,
        policy: Any = None,
        wall_s: float = 0.0,
        unit_deadline_s: float | None = None,
    ) -> dict:
        """PHASE 1 SHAPE: units are pushed (scatter).

        ``wall_s`` is checked BEFORE starting a unit, never during one. Stopping mid-unit
        would abandon a scene build that has already been paid for, and the driver's own
        ``unit_deadline_s`` is the right tool for a unit that runs long.
        """
        if policy is None:
            raise ValueError("run_units requires a policy client (the peer WG arg)")
        validate_rollout(policy, expect_action_dim=self._emb.action_dim, expect_embodiment=self._emb.name)

        t0 = time.monotonic()
        out, skipped = [], 0
        total = len(units)
        for i, raw in enumerate(units, 1):
            unit = _as_unit(raw)
            if wall_s and time.monotonic() - t0 > wall_s:
                skipped += 1
                continue
            # THE UNIT BOUNDARY, ON THE DIRECT CHANNEL. Without these two lines `wall_s` and
            # `boot_s` are computed on every run and never reach a log.
            #
            # Three things become measurable only because this line exists:
            #
            #   * WHERE TIME GOES. `boot_s` against `wall_s - boot_s`, per unit, separates a
            #     fixed per-unit cost from a per-step one.
            #   * MEMORY ACROSS UNITS. The `[mem]` heartbeat is a pod-level trace with no unit
            #     boundaries in it: segmenting by sample count made a CORRECT run (RZ-3, one
            #     unit) look like it lost 68.9 GiB. These two lines are the boundaries, on the
            #     same clock.
            #   * A FAULT THAT COMMITTED NOTHING. `run_unit` catches a fault into
            #     `UnitResult.error` and this method reports DONE, so a failed eval unit is
            #     otherwise indistinguishable from one that ran.
            #
            # THIS PROCESS IS THE SIMULATOR, not a Ray actor, so these land on the pod's own
            # stdout with a usable clock. The gateway's lines are forwarded by Ray and can flush
            # at teardown -- `[gateway] run_units done` cannot be used to time anything.
            #
            # Unconditional, not env-gated: an instrument that must be armed is an instrument
            # someone forgets to arm, and the cost is two lines per UNIT, not per step.
            print(
                f"[collector] unit {i}/{total} START {unit.unit_id} "
                f"task={unit.scenario.get('task', '?')} episodes={len(unit.episodes)} "
                f"t+{time.monotonic() - t0:.1f}s",
                flush=True,
            )
            r = run_unit(
                unit,
                self._sim,
                policy,
                self._sink,
                self._emb,
                force_render=self._force_render,
                render_every=self._render_every,
                deadline_s=unit_deadline_s,
                progress=self._heartbeat,
            )
            print(
                f"[collector] unit {i}/{total} END {unit.unit_id} "
                f"wall_s={r.wall_s:.1f} boot_s={r.boot_s:.1f} "
                f"step_s={max(0.0, r.wall_s - r.boot_s):.1f} "
                f"steps={sum(e.steps for e in r.episodes)} episodes={len(r.episodes)} "
                f"fault={r.error_class or 'none'} t+{time.monotonic() - t0:.1f}s",
                flush=True,
            )
            out.append(r)
        # Report what was dropped. A wall-clock stop that silently returns fewer results
        # reads as "this cell finished its bin" to every consumer above it.
        if skipped:
            print(f"[collector] wall_s={wall_s} reached; {skipped} unit(s) not started", flush=True)
        # `unit_faults` is a DISTINCT key and never a top-level `error`. `entrypoints/collect.py`
        # does `if r.get("error"): errors.append(r); continue` and `live = [r for r in results
        # if not r.get("error")]`, so one `error` key on a cell that DID return episodes drops
        # every one of them from the census -- trading a silent eval failure for a silent
        # collection one. Nor `unit_errors`, which is already an int on the pull path.
        #
        # Why it exists at all: on this push path `run_unit` catches a fault into
        # `UnitResult.error` and the colocation reports DONE, so a failed eval unit is
        # indistinguishable from a successful one. This is the evidence a caller needs to
        # tell them apart.
        faults = [
            {"unit_id": r.unit_id, "error": r.error, "error_class": r.error_class or "unclassified"}
            for r in out
            if r.error
        ]
        # `sr` comes from the SINK, not from the UnitResults: the sink counts COMMITTED
        # episodes and skips `already_landed` hits (`_LandedWriter.commit` returns the URI
        # without appending a record), which is exactly the rule the path this replaces states
        # -- landed episodes were reported by whichever cell produced them, and counting them
        # again inflates the summary rather than completing it. PackageSink has no `sr`, so a
        # collection run's return dict is provably unchanged by this.
        #
        # TWO LIMITS, stated rather than discovered later. (1) The sink is built once in
        # `__init__` and under gateway the body is long-lived, so a SECOND `run_units` call
        # returns the CUMULATIVE `sr` and double-reports the first; the path this replaces
        # built its merged dict fresh per call. (2) Omitting the key when `sr` is empty means a
        # fully-resumed eval reports no tasks at all, where the old path returned
        # `{task: [0, 0]}`.
        return {
            "results": [asdict(r) for r in out],
            "units_not_started": skipped,
            "unit_faults": faults,
            "host": os.uname().nodename,
            "gpu": self._sim.gpu_uuid(),
            **sink_info(self._sink).summary_fields(),
        }

    # ── PHASE 2: pull ───────────────────────────────────────────────────────
    def run_from_source(
        self,
        units: Sequence[Any],
        *,
        policy: Any = None,
        source: str = "dirlease",
        source_options: dict | None = None,
        wall_s: float = 0.0,
        unit_deadline_s: float | None = None,
        drain_s: float = 0.0,
        poll_s: float = _IDLE_POLL_S,
        poll_max_s: float = _IDLE_POLL_MAX_S,
        max_consecutive_errors: int = 3,
        reserve_minutes: float | None = None,
    ) -> dict:
        """Claim, run, retire -- until there is nothing left to claim.

        Every cell is handed the SAME ``units`` list. That is not a redundancy: ``unit_id``
        is a digest of the unit's own content, so identical plans produce identical ids and
        the id is a valid cross-pod claim key by construction. ``DirLeaseSource`` refuses to
        join a campaign whose plan digest differs from its own, which turns a pod that
        planned differently into a startup error instead of a duplicated corpus.

        **One unit at a time**, by passing ``budget_minutes=0``: both sources yield at least
        one unit and then stop at any budget. Claiming a batch would mean holding leases on
        units we have not started -- for hours, on a 30-minute TTL -- so either they age out
        and someone else runs them while we run them too, or they do not and the tail of the
        campaign starves behind one cell's queue. Neither is worth the handful of stats a
        second claim pass costs.

        **The stop conditions**, all four of which report themselves in ``stopped``:

        ``wall``
            ``wall_s`` minus a reserve big enough for the largest unit in the plan. Checked
            before CLAIMING, so a unit is never claimed that cannot be finished -- claiming
            and then handing it back is not free, because each handover consumes one of the
            unit's ``MAX_GENERATIONS`` and enough of them mark it stuck forever.
        ``complete``
            the source reports every unit retired.
        ``drained``
            nothing was claimable for ``drain_s``. With the default of 0 the cell exits at
            the first empty claim; raising it keeps cells alive to reclaim units whose
            holders die late, at the cost of idle GPU-minutes. Neither is universally right,
            so it is a knob and the report says which one fired.
        ``errors``
            ``max_consecutive_errors`` units in a row failed to RETIRE -- whether from a
            unit-level fault (a dead card, a wedged Isaac) or from episodes that never
            reached disk (a full filesystem, a dead encoder). Either way this cell is the
            common factor, and a pull worker that keeps claiming will chew through the whole
            campaign's generation budget one unit at a time -- strictly worse than the push
            model it replaced. Exiting hands the units straight back. The two causes are
            counted apart in the result (``unit_errors`` vs ``units_incomplete``) because a
            card and a filesystem call for different responses.
        """
        if policy is None:
            raise ValueError("run_from_source requires a policy client (the peer WG arg)")
        validate_rollout(policy, expect_action_dim=self._emb.action_dim, expect_embodiment=self._emb.name)

        plan = [_as_unit(u) for u in units]
        opts = dict(source_options or {})
        sink_root = getattr(self._sink, "root", None)
        if "root" in opts and sink_root is not None and os.path.abspath(opts["root"]) != os.path.abspath(sink_root):
            raise ValueError(
                f"source root {opts['root']!r} != sink root {sink_root!r}. The claim plane "
                f"lives under the campaign root because `.done` attests to packages in it: "
                f"split them and `already_landed` looks in one campaign for episodes claimed "
                f"in another, so every re-dispatch re-runs work that is already on disk."
            )
        opts.setdefault("root", sink_root)
        if opts["root"] is None:
            raise ValueError(
                f"no campaign root: sink {type(self._sink).__name__} exposes none and "
                f"source_options carries none. `str(None)` would give every pod a claim "
                f"directory called 'None' in its own container, so 40 pods would each claim "
                f"and run the entire plan."
            )
        src = build_source(source, units=plan, owner=self._owner, **opts)

        reserve_s = 60.0 * (
            max((u.est_minutes for u in plan), default=0.0) if reserve_minutes is None else float(reserve_minutes)
        )
        if wall_s and wall_s <= reserve_s:
            # Say it, rather than exiting immediately and looking like an empty campaign.
            print(
                f"[collector] wall_s={wall_s:.0f}s is not larger than the reserve "
                f"({reserve_s:.0f}s = the longest unit in the plan). This cell will claim "
                f"nothing: there is no unit it could finish inside its budget.",
                flush=True,
            )
        t0 = time.monotonic()
        out: list[UnitResult] = []
        claimed = completed = abandoned = incomplete = errors = 0
        consecutive = 0
        idle_since: float | None = None
        last_status = 0.0
        backoff = float(poll_s)
        st: dict = {}
        stopped = "drained"

        self._source = src
        try:
            while True:
                now = time.monotonic()
                if wall_s and now - t0 > max(0.0, wall_s - reserve_s):
                    stopped = "wall"
                    break

                leases = src.claim(budget_minutes=0.0, owner=self._owner)
                if leases:
                    idle_since, backoff = None, float(poll_s)
                    # Held from the moment they are claimed, not from the moment they start
                    # running: an unstarted lease ages out exactly like an abandoned one.
                    self._held.extend(leases)
                    for lease in leases:
                        claimed += 1
                        res = self._run_leased(lease, policy, unit_deadline_s)
                        out.append(res)
                        verdict = self._retire(src, lease, res)
                        if verdict == "completed":
                            completed += 1
                        elif verdict == "error":
                            abandoned += 1
                        elif verdict == "revoked":
                            pass  # counted by self._revoked; the work landed, we just lost the claim
                        else:
                            incomplete += 1
                        # The breaker counts consecutive NON-COMPLETIONS, not just faults: a
                        # unit whose episodes cannot publish fails identically on every unit
                        # this cell touches, and the question is not whose fault it is, it is
                        # whether continuing is doing any good.
                        errors += int(verdict == "error")
                        consecutive = 0 if verdict in ("completed", "revoked") else consecutive + 1
                    if consecutive >= max_consecutive_errors:
                        stopped = "errors"
                        break
                    # THE ONLY LEASE-FREE WINDOW IN THIS LOOP, and the reason the planned
                    # restart is asked here rather than taken where it is wanted. `_retire`
                    # has dropped every lease above; the next `claim` has not run yet.
                    self._planned_restart(src)
                    continue

                # Nothing claimable at this instant: `claim` scanned every unit to say so.
                now = time.monotonic()
                if idle_since is None:
                    idle_since = now
                    st, last_status = src.status(), now
                    if int(st.get("done", 0)) >= int(st.get("units", 0)):
                        stopped = "complete"
                        break
                elif now - last_status >= poll_max_s:
                    st, last_status = src.status(), now
                    if int(st.get("done", 0)) >= int(st.get("units", 0)):
                        stopped = "complete"
                        break
                if now - idle_since >= drain_s:
                    break
                nap = min(backoff, drain_s - (now - idle_since))
                if wall_s:
                    nap = min(nap, max(0.0, wall_s - reserve_s - (now - t0)))
                if nap > 0:
                    time.sleep(nap)
                backoff = min(backoff * 2.0, float(poll_max_s))
        finally:
            # A lease still held here belongs to a unit whose iteration did not finish --
            # only KeyboardInterrupt and SystemExit get out of `_run_leased`, and those are
            # exactly the cases where the unit must go back to the pool immediately rather
            # than sit unavailable for the whole TTL while this process is already gone.
            self._release_held(src, "worker exit")
            self._source = None

        if not st:
            try:
                st = src.status()
            except OSError as exc:
                # The episode results are in hand; losing them to a failed stat at the very
                # end would trade the run's whole output for a status line.
                st = {"error": repr(exc)}
        print(
            f"[collector] {stopped}: claimed={claimed} completed={completed} "
            f"abandoned={abandoned} incomplete={incomplete} errors={errors} "
            f"revoked={self._revoked} declined={self._declined} | "
            f"campaign {st.get('done')}/{st.get('units')} done, "
            f"{st.get('held')} held, {st.get('reclaimable')} reclaimable, "
            f"{st.get('stuck')} stuck",
            flush=True,
        )
        # `sr` on the PULL path too, for the same reason and from the same place as
        # `run_units`: the eval driver's `===== RESULTS =====` block is `_log_final_summary`
        # summing what each body RETURNED, not the jsonl glob that builds results.csv. Omit
        # it and a pull eval prints "0 tasks | overall SR 0.0% (0/0)" on a run that wrote
        # every row correctly -- the exact shape of the two-channel disagreement C7 already
        # had to state once.
        return {
            "results": [asdict(r) for r in out],
            **sink_info(self._sink).summary_fields(),
            "units_claimed": claimed,
            "units_completed": completed,
            "units_abandoned": abandoned,
            "units_incomplete": incomplete,
            "unit_errors": errors,
            "leases_revoked": self._revoked,
            "retirements_declined": self._declined,
            "stopped": stopped,
            "source_status": st,
            "wall_s": round(time.monotonic() - t0, 1),
            "host": os.uname().nodename,
            "gpu": self._sim.gpu_uuid(),
        }

    def _run_leased(self, lease: Lease, policy: Any, deadline_s: float | None) -> UnitResult:
        """One unit, with ``_heartbeat`` renewing every held lease while it runs.

        The lease is deliberately NOT dropped from ``self._held`` on the way out. It is
        dropped by :meth:`_retire`, once the unit has actually been handed back or retired --
        so a ``KeyboardInterrupt`` or ``SystemExit`` out of the rollout loop leaves it
        visible to the outer ``finally``, which is the only thing that returns it to the
        pool. Dropping it here would leak exactly the lease the teardown path exists for.

        ``_last_renew`` is reset so the first progress callback of a unit renews
        unconditionally: a scene build is 1.5-6 minutes of silence at the START of a unit,
        which is the worst moment to be running on a lease someone else has already aged out.
        """
        self._last_renew = 0.0
        return run_unit(
            lease.unit,
            self._sim,
            policy,
            self._sink,
            self._emb,
            force_render=self._force_render,
            render_every=self._render_every,
            deadline_s=deadline_s,
            progress=self._heartbeat,
        )

    def _retire(self, src: Any, lease: Lease, res: UnitResult) -> str:
        """Complete or abandon, decided HERE on evidence.

        Returns one of ``"completed"``, ``"revoked"`` (the work landed but a successor owns
        the unit now), ``"error"`` (a unit-level fault -- the cell is implicated) or
        ``"incomplete"`` (the unit ran, some episode did not reach disk). The caller's
        circuit breaker treats the first two as progress and the last two as not, and reports
        the last two under separate names, so the four cannot be collapsed.

        The decision is not delegated to the source, even though ``DirLeaseSource.complete``
        makes the same check before writing ``.done``. Two reasons, and the second is the
        real one: the loop knows *why* (a unit-level fault reads differently from a lane that
        published nothing), and a rule that lives only inside one implementation of
        ``UnitSource`` is a rule the other implementation silently does not have -- so a test
        driving this loop against ``StaticUnits`` would exercise a retire policy the
        production source does not share.

        An episode-level ``FAILURE`` is **data** and retires normally. Only a unit-level
        fault, or an episode that did not reach disk, sends the unit back.
        """
        try:
            if res.error:
                src.abandon(lease, res.error)
                # PRINTED, and this is the only place `UnitResult.error_class` becomes
                # evidence: `DirLeaseSource.complete` writes `result.error` into the `.done`
                # journal, but a unit with an error never reaches `complete` -- it is
                # abandoned, one branch above.
                print(
                    f"[collector] unit {lease.unit.unit_id} faulted ({res.error_class or 'unclassified'}): {res.error}",
                    flush=True,
                )
                return "error"
            # ONE coverage rule, `UnitResult.missing`, shared with the source rather than
            # recomputed here from the same two objects. Two copies of the predicate that
            # decides whether work is handed back is how a caller and its source come to
            # disagree about whether a unit is done -- invisibly, until it is either lost or
            # re-dispatched forever.
            missing = res.missing(lease.unit)
            if missing:
                src.abandon(lease, f"{len(missing)}/{len(lease.unit.episodes)} episodes did not land")
                print(
                    f"[collector] unit {lease.unit.unit_id} returned without landing "
                    f"{sorted(missing)[:4]}{'...' if len(missing) > 4 else ''}; handed back",
                    flush=True,
                )
                return "incomplete"
            # A source may DECLINE the retirement -- because a successor generation owns the
            # unit, or because its own coverage re-check refused. ``RetireDisposition`` carries
            # WHICH, and that is why the answer is returned rather than inferred from a private
            # counter: the two refusals need different verdicts, and a boolean could only say
            # that one of them happened.
            disposition = self._retired_as(src, lease, res)
            if disposition is RetireDisposition.COMPLETED:
                return "completed"
            self._declined += 1
            if disposition is RetireDisposition.REVOKED:
                return "revoked"
            # INCOMPLETE. Not "revoked": nothing was stolen, and "revoked" is the one verdict
            # the loop treats as harmless -- it neither counts the unit as incomplete nor trips
            # the consecutive-failure breaker, because the work DID land under another holder.
            # A source that refused for coverage means the work did not land, and reporting it
            # as revoked would hide exactly the failure the breaker exists to catch.
            return "incomplete"
        finally:
            self._drop(lease)

    def _retired_as(self, src: Any, lease: Lease, res: UnitResult) -> RetireDisposition:
        """Retire through the source and report what it DID.

        The typed answer is authoritative. The counter fallback below is the migration
        affordance described on ``UnitSource.complete``: a source written before this return
        existed yields ``None``, and is read through the two counters it does publish rather
        than being reported as completing everything.

        **The fallback cannot distinguish all three answers, and that is the argument for the
        return value rather than against the fallback.** ``revoked`` and ``incomplete`` are
        separate attributes on ``DirLeaseSource``, so both are reachable by counter -- but only
        because that one source happens to publish both. Any other implementation is read as
        COMPLETED by default, which is the failure this return type removes.
        """
        before_revoked = int(getattr(src, "revoked", 0))
        before_incomplete = int(getattr(src, "incomplete", 0))
        disposition = src.complete(lease, res)
        if disposition is not None:
            return RetireDisposition(disposition)
        if int(getattr(src, "revoked", 0)) > before_revoked:
            return RetireDisposition.REVOKED
        if int(getattr(src, "incomplete", 0)) > before_incomplete:
            return RetireDisposition.INCOMPLETE
        return RetireDisposition.COMPLETED

    def _drop(self, lease: Lease) -> None:
        self._held = [ln for ln in self._held if ln.unit.unit_id != lease.unit.unit_id]

    def _planned_restart(self, src: Any) -> None:
        """Ask the simulator whether it wants a fresh process, and give it one if so.

        Asked ONLY from the window between retiring one unit and claiming the next, because
        the answer is acted on with ``os._exit`` and that runs no ``finally``. The simulator
        used to both decide and act, at the top of its own ``open()`` -- which the loop reaches
        holding the next unit's lease, so the exit stranded a unit nobody else could reclaim
        for a full ``lease_seconds`` and spent one of its ``MAX_GENERATIONS`` doing it. Moving
        the ACT here and leaving the DECISION with the engine is what makes the exit safe;
        neither half is sufficient alone.

        ``exit 9`` and not a clean 0: this must look to the supervisor EXACTLY like the fault
        it already restarts from, and a clean exit is never restarted -- 0 would end the cell.

        ``_release_held`` first, and it is not belt-and-braces. It is the invariant this
        method's safety rests on, made true rather than assumed: today ``claim`` is called with
        ``budget_minutes=0`` so ``_held`` is empty here, and the day someone claims a batch this
        keeps returning units instead of silently stranding them again.

        THREE BUDGETS MUST AGREE or this silently does nothing, or ends the cell: the
        supervisor's ``--max-restarts`` must cover the planned count and the gateway's
        ``restart_budget`` must cover the re-issued calls -- they count DIFFERENT things and both
        default to 0. Only ``run_from_source`` is resumable, which is why this lives here and
        ``run_units`` never asks: under the push path a restart would re-run finished units.
        """
        if not self._sim.wants_restart():
            return
        self._release_held(src, "planned restart")
        print(
            f"[collector] planned restart requested by {type(self._sim).__name__}; exiting 9 "
            f"so the supervisor rebuilds. Every lease was handed back first, so no unit waits "
            f"out a dead holder's TTL and no generation is spent.",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(9)

    def _release_held(self, src: Any, reason: str) -> None:
        """Hand back every lease still held. Called from the loop's ``finally``."""
        held, self._held = self._held, []
        for lease in held:
            try:
                src.abandon(lease, reason)
            except Exception as exc:  # noqa: BLE001 -- teardown must not mask the original fault
                print(f"[collector] WARNING: could not abandon {lease.unit.unit_id}: {exc!r}", flush=True)

    def capabilities(self) -> Any:
        """What this body's simulator can do, WITHOUT the card probe.

        Split out of :meth:`probe` because the two are wanted at different moments by different
        callers. ``probe`` is a bind-time card check that happens to carry the capabilities;
        ``sim_server`` needs only the capabilities, before any card exists to check -- the
        gateway cannot size its shared segment until it knows the embodiment. Calling ``probe``
        there would run a GPU probe to answer a question about geometry.
        """
        return self._sim.capabilities()

    def probe(self) -> dict:
        """Second-line card check AND the capability handshake, callable at bind time.

        The capabilities ride as a plain dict rather than a ``SimCapabilities``: this return
        value crosses an actor boundary and ``embodied/sim.py``'s wire rule forbids custom
        classes on that hop. ``roles/env/sim_role.py::validate_sim_role`` rebuilds and checks
        it.

        Decision-window bindings call this before collection to validate their optional
        capabilities. Episode-work-unit bindings keep their existing handshake-only card
        check and unit-level failure handling; this method does not change that policy.
        """
        self._sim.probe()
        return {
            "host": os.uname().nodename,
            "gpu": self._sim.gpu_uuid(),
            "capabilities": capabilities_as_wire(self.capabilities()),
        }

    def close(self) -> None:
        try:
            self._sink.close()
        finally:
            self._sim.close()

    # ── internals ───────────────────────────────────────────────────────────
    def _heartbeat(self, snapshot: Mapping[str, Any]) -> None:
        """Publish progress, and renew every lease this worker is holding.

        The lease renewal has to happen HERE, from inside the rollout loop, and that is not
        an optimisation. A unit is up to ~46 minutes of episodes against a 30-minute default
        TTL, so a worker that renewed only between units would age out mid-unit and have its
        work claimed by a second cell while the first is still running it. Round 3 hit the
        same shape from the other side: takeovers fired on LIVE processes because the
        holder's only heartbeat was the work it was not currently doing.

        Neither the sink write nor the renewal may raise into the rollout loop. A failed
        heartbeat is diagnostics; a failed renewal is a lease we will lose. Losing an episode
        to either would be trading data for telemetry.
        """
        try:
            self._sink.heartbeat(snapshot)
        except Exception as exc:  # noqa: BLE001
            print(f"[collector] heartbeat failed: {exc!r}", flush=True)

        src = self._source
        if not self._held or src is None:
            return
        now = time.monotonic()
        if now - self._last_renew < _RENEW_EVERY_S:
            return
        self._last_renew = now
        try:
            live = src.heartbeat(list(self._held))
        except Exception as exc:  # noqa: BLE001
            print(f"[collector] lease renewal failed: {exc!r}", flush=True)
            return
        kept = {ln.unit.unit_id for ln in live}
        for lease in self._held:
            if lease.unit.unit_id in kept or lease.unit.unit_id in self._revoked_ids:
                continue
            self._revoked_ids.add(lease.unit.unit_id)
            # Revoked mid-unit: someone judged us dead and took the unit. Keep running to the
            # end anyway. The episodes still land, `already_landed` and the publish-time
            # duplicate check make the overlap harmless, and `complete` will decline to write
            # `.done` over the successor's claim. Stopping here would throw away a scene
            # build and the episodes already committed in this turn.
            self._revoked += 1
            print(
                f"[collector] lease REVOKED for {lease.unit.unit_id} while we hold it; "
                f"finishing the unit, retirement will be declined",
                flush=True,
            )
        # Keep holding the revoked ones. Dropping them here would take them out of the
        # teardown path's reach, and a lease we still believe we own is exactly what the
        # teardown has to hand back -- `abandon` on a lease someone else owns is a no-op,
        # because it only expires markers we created.


def _as_unit(raw: Any) -> WorkUnit:
    """Accept a :class:`WorkUnit` or the dict Ray round-trips it to.

    Ray serialises dataclasses by value but a driver that built units from JSON hands over
    plain dicts, and a body that only accepted one shape would fail after the scatter --
    i.e. after every cell has already booted Isaac.
    """
    if isinstance(raw, WorkUnit):
        return raw
    if isinstance(raw, Mapping):
        eps = tuple(e if isinstance(e, EpisodeKey) else EpisodeKey(**dict(e)) for e in raw["episodes"])
        return WorkUnit(
            unit_id=raw["unit_id"],
            suite=raw["suite"],
            scenario=dict(raw["scenario"]),
            episodes=eps,
            est_minutes=float(raw["est_minutes"]),
            group_id=raw.get("group_id"),
            tags=dict(raw.get("tags") or {}),
        )
    raise TypeError(f"cannot interpret {type(raw).__name__} as a WorkUnit")
