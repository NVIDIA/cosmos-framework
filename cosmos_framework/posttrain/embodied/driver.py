# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The one rollout loop.

:func:`run_unit` is the only place a trajectory is recorded and the only place the
frame-alignment convention lives. Centralizing it converts N independent bugs into one bug
with N-fold blast radius, which is a real cost -- gate G1 (a differential run against the
then-existing ``run_tasks`` path) is what bought it back. **G1 RAN AND PASSED, 2026-08-29**:
correctness equivalent at 400/400 (gap -2.37 pp) and throughput +2.87 ms/step measured with
both arms on one card. The ``run_tasks`` path was deleted the same day, so the differential
cannot be taken again. SPEC §0.16-§0.17.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cosmos_framework.posttrain.embodied.embodiment import (
    EmbodimentMap,
    check_obs,
    describe_obs,
    to_policy_obs,
    to_record_view,
)
from cosmos_framework.posttrain.embodied.rollout import Rollout
from cosmos_framework.posttrain.embodied.sim import SimProtocolError, Simulator, assert_wire_safe, classify_error
from cosmos_framework.posttrain.embodied.sink import EpisodeWriter, TrajectorySink, make_frame
from cosmos_framework.posttrain.embodied.unit import EpisodeResult, Outcome, UnitResult, WorkUnit

#: How often, in control steps, to publish a heartbeat. The sim actor is single-threaded
#: for hours at a time, so this is the only liveness signal that exists.
_PROGRESS_EVERY = 200


def render_decision(*, forced: bool, render_every: int, live_steps: Sequence[int], will_query: bool) -> bool:
    """Should this step render? ONE expression, shared by the eval loop and this one.

    They had two, and they disagreed on both terms: eval decided AFTER its step counter
    advanced and included "the next iteration will query the policy"; this loop decided
    BEFORE and had no such term. At ``render_every=2`` over a 5-step episode they rendered
    ``{1,3}`` against ``{0,2,4}`` -- no step agreed.

    **EVAL'S EXPRESSION WINS, and not merely because it is the reference.** ``will_query`` is a
    CORRECTNESS term: ``SimSession.observe`` hands back the bundle boxed by the LAST step, so
    skipping the render before a policy query feeds the model a stale image. ``config/eval.py``
    already states that contract; this loop did not implement it. The phase is then forced to
    match, because a term is only correct at the phase it was written for.

    NOTE FOR ANYONE READING A GREEN TEST HERE: at ``render_every=1`` every branch returns True,
    and ``force_render=True`` is hardcoded for every collection campaign, so **neither
    production regime can tell the old expression from this one**. No score already on record moves. A
    differential run at the default proves nothing; test it at ``render_every > 1`` or do not
    claim to have tested it. The first campaign that raises ``render_every`` measures a new
    distribution regardless -- the RTX path denoises with temporal accumulation -- so do not
    attribute a delta on such a run to the transport or to the loop.

    ``live_steps`` is taken over LIVE lanes only. Including a finished lane freezes ``min()``
    at that lane's last count, ``% render_every`` stops advancing, and the skip never fires
    again once any lane terminates -- the speed lever quietly stops working while everything
    still looks correct.
    """
    if forced or render_every <= 1:
        return True
    if will_query:
        return True
    if not live_steps:
        # Every lane is done; the vec still has to be stepped, and there is nothing to skip
        # rendering FOR. Matches eval's `not live` term.
        return True
    return min(live_steps) % render_every == 0


class _LandedWriter:
    """A writer for an episode that is already on disk.

    Dispatch is per unit, so a unit with *some* episodes already landed is re-run in full:
    the encode and the write are recovered, the simulation is not.

    What this DOES buy is correctness: without it, the re-run would call ``commit()`` and
    overwrite a package that already carries a ``READY`` marker, replacing a validated
    artifact with a fresh one that no reconciler has seen. Returning the existing URI
    leaves the landed artifact untouched.
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self.appended = 0

    def append(self, frame) -> None:
        del frame
        self.appended += 1

    def commit(self, outcome: Outcome, metrics: Mapping[str, Any]) -> str:
        del outcome, metrics
        return self._uri

    def abort(self, reason: str) -> None:
        del reason


def _abort_all(writers: Mapping[int, EpisodeWriter], done: set, reason: str) -> None:
    """Abort every writer that has not already been committed.

    Aborting a committed writer would be a request to unpublish a ``READY`` artifact, which
    is not what a failed sibling lane means.
    """
    for lane, w in writers.items():
        if lane in done:
            continue
        try:
            w.abort(reason)
        except Exception:  # noqa: BLE001 -- an abort that fails must not mask the original fault
            pass


def run_unit(
    unit: WorkUnit,
    sim: Simulator,
    rollout: Rollout,
    sink: TrajectorySink,
    emb: EmbodimentMap,
    *,
    force_render: bool = True,
    render_every: int = 1,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    deadline_s: float | None = None,
) -> UnitResult:
    """Run every episode in ``unit``, recording each through ``sink``.

    Never raises for a simulator or policy fault: those are returned as
    ``UnitResult.error`` so the caller can re-offer the unit. ``KeyboardInterrupt`` and
    ``SystemExit`` DO propagate -- swallowing them would make a worker unkillable -- but the
    in-flight turn's writers are aborted first, so a torn-down process still leaves no
    half-written artifact carrying a ``READY`` marker.
    """
    t0 = time.monotonic()
    host = socket.gethostname()
    eps: list[EpisodeResult] = []
    pinfo = rollout.describe()
    gpu = sim.gpu_uuid()

    # Skip-if-landed BEFORE the scene build: a re-dispatched unit whose episodes all landed
    # costs a stat(), not a ~6-minute scene build plus twenty episodes.
    landed: dict[str, str] = {}
    for key in unit.episodes:
        uri = sink.already_landed(key)
        if uri is not None:
            landed[key.slug()] = uri
    if len(landed) == len(unit.episodes):
        # Report WHICH episodes were skipped, not just that there were none to run. A caller
        # that retires a unit on evidence needs the slugs: an empty episode list with no
        # error is otherwise indistinguishable from a unit that ran and landed nothing.
        return UnitResult(
            unit.unit_id,
            [],
            time.monotonic() - t0,
            0.0,
            host,
            gpu,
            skipped_landed=tuple(sorted(landed)),
        )

    # Emit liveness BEFORE the build, not only at turn start.
    #
    # `turn_start` below bounds the silent window to one turn -- but the FIRST window is the
    # build itself, and nothing precedes it. Measured on the first real pod: a unit claimed at
    # 14:58:30 whose heartbeat was still untouched four minutes later, because no lane had
    # taken a step yet. The renewal is what a holder's liveness is judged on, so the phase
    # most likely to hang (Isaac) was also the phase that published nothing. The beat also
    # gives an operator a timestamp to measure the build against.
    if progress is not None:
        progress(
            {
                "unit": unit.unit_id,
                "pos": 0,
                "event": "scene_build",
                "lanes": len(unit.episodes),
                "host": host,
                "t": time.time(),
            }
        )

    # Read the scenario BEFORE the scene build. It does not need a session, and a malformed
    # scenario read afterwards escaped this function as a bare KeyError -- past the guard
    # below, so the built scene was never closed. `run_unit` returns a UnitResult for a
    # unit-level fault and does not raise; a missing key is a unit-level fault like any other.
    try:
        max_steps = int(unit.scenario["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        return UnitResult(
            unit.unit_id,
            [],
            time.monotonic() - t0,
            0.0,
            host,
            gpu,
            error=f"scenario needs an integer 'max_steps': {exc!r}",
            error_class=classify_error(exc),
        )

    # The scene build is itself a fault surface -- a dead card fails here, not later -- so
    # it is inside the guard. The spec draft had `sim.open` outside the try, which would
    # have let a bad card kill the whole bin instead of returning one re-dispatchable unit.
    boot0 = time.monotonic()
    try:
        sess = sim.open(unit)
    except Exception as exc:  # noqa: BLE001
        return UnitResult(
            unit.unit_id,
            [],
            time.monotonic() - t0,
            time.monotonic() - boot0,
            host,
            gpu,
            error=repr(exc),
            error_class=classify_error(exc),
        )
    boot_s = time.monotonic() - boot0

    scene_pos = 0
    # The contract check is per UNIT, not per step. Once the first live lane's observation
    # satisfies the embodiment, every later one in the same scene comes from the same sensors
    # through the same flatten -- so a per-step check would re-answer a settled question
    # 5,300 times and buy nothing.
    contract_checked = False
    # Payloads that would not survive a transport other than this one. COUNTED, NOT FATAL --
    # see where it is filled.
    wire_faults: list[str] = []

    try:
        while sess.advance_turn():
            lanes = list(range(sess.n_lanes))
            # Emit liveness at TURN START, not only every _PROGRESS_EVERY steps.
            #
            # Round 3's lease takeovers fired on LIVE processes because the holder's only
            # heartbeat was the work it was not currently doing: it heartbeat on journal
            # appends, then spent ~110 s in ffprobe under the lease and aged out at 91.4 s
            # against a 90 s threshold. The same shape is here -- a scene build is 1.5-6
            # minutes during which no lane takes a step, so a step-driven heartbeat is silent
            # exactly when the worker is busiest. Emitting here bounds the silent window to
            # one turn rather than one unit.
            if progress is not None:
                beat: dict[str, Any] = {
                    "unit": unit.unit_id,
                    "pos": scene_pos,
                    "event": "turn_start",
                    "lanes": len(lanes),
                    "host": host,
                    "t": time.time(),
                }
                if scene_pos == 0:
                    # Published on the HEARTBEAT, deliberately not into episode provenance:
                    # `meta` is written verbatim into the package, so adding a key there
                    # changes the bytes of every package produced from here on. The heartbeat
                    # is free-form JSON that nothing gates on.
                    #
                    # Read once per unit rather than per turn: a scene build is what
                    # `generation` counts, and the engine's key space does not change between
                    # turns of one build.
                    beat["sim_generation"] = sess.generation
                    beat["dropped_obs_keys"] = list(sess.dropped_obs_keys())
                progress(beat)
            writers: dict[int, EpisodeWriter] = {}
            committed: set = set()
            try:
                for lane in lanes:
                    key = sess.lane_key(lane)
                    uri = landed.get(key.slug())
                    writers[lane] = (
                        _LandedWriter(uri)
                        if uri is not None
                        else sink.open_episode(
                            key,
                            {
                                "unit_id": unit.unit_id,
                                "policy": pinfo.name,
                                "weight_version": pinfo.weight_version,
                                "group_id": unit.group_id,
                                "position_in_scene": scene_pos,
                                # Which lane of the turn ran this episode. In the package so
                                # that "does lane 0 differ from lane 1" can be asked of the
                                # corpus without reconstructing the turn and assuming the
                                # binding order -- which is what such a question suspects.
                                "lane": lane,
                                "host": host,
                                "gpu": gpu,
                            },
                        )
                    )

                chunks: dict[int, Any] = {}
                idxs: dict[int, int] = {}
                steps = {lane: 0 for lane in lanes}
                # Summed HERE, from the same StepOut.reward that lands in the package's
                # reward[T] column, so the scalar and the column cannot disagree. It was
                # `metrics.get("reward_sum", 0.0)` -- and no real session produces that key
                # (only the test fake did), so every b1k episode would have reported
                # reward_sum=0.0, forever, with nothing raised. A session is free to publish
                # its own aggregate; it stays in `metrics` under its own name.
                rewards = {lane: 0.0 for lane in lanes}

                for _ in range(max_steps):
                    if deadline_s is not None and time.monotonic() - t0 > deadline_s:
                        raise TimeoutError(f"unit {unit.unit_id} exceeded {deadline_s}s")

                    live = [lane for lane in lanes if not sess.done(lane)]
                    if not live:
                        break

                    # observe() is CACHED -> one call, zero simulator cost.
                    pre_obs = dict(zip(live, sess.observe(live)))

                    if not contract_checked:
                        contract_checked = True
                        first = pre_obs[live[0]]
                        # MANDATORY, and this is the declared site the contract names.
                        # Both properties it asserts are already fatal further downstream --
                        # a missing key inside `_adapt_obs` on the far side of the wire,
                        # after an 8B checkpoint load; a wrong camera rank inside the hevc
                        # encoder, mid-episode. Raising here moves them to a point where the
                        # unit has cost one scene build and no physics.
                        check_obs(first, emb)
                        # The observation is wire-safe BY CONSTRUCTION on the real path --
                        # `flatten_obs` converts every leaf off the engine's tensors -- so
                        # this raising means something genuinely new arrived, which is worth
                        # the unit. Contrast the metrics check below.
                        assert_wire_safe(first, "observe()")
                        # THE MEASUREMENT. Once per unit, on the first observation that
                        # exists -- which is the earliest moment the answer is knowable and
                        # the only one that costs nothing.
                        #
                        # Printed AND put on the heartbeat. The print is what a person reads
                        # in a pod log; the heartbeat is what survives the pod, because pod
                        # logs are per-run and reaped with the container while the campaign
                        # root is not. Publishing it in only one of the two places is how a
                        # measurement gets taken and then lost.
                        shape = describe_obs(first, emb)
                        print(
                            f"[driver] obs {unit.unit_id}: "
                            f"declared {shape['n_declared']} keys / {shape['declared_bytes']} B, "
                            f"undeclared {shape['n_undeclared']} keys / {shape['undeclared_bytes']} B, "
                            f"camera_channels={shape['camera_channels']}",
                            flush=True,
                        )
                        if progress is not None:
                            # ITS OWN EVENT, not a rider on the periodic beat.
                            #
                            # The periodic beat fires on `min(steps) % _PROGRESS_EVERY == 0`
                            # with _PROGRESS_EVERY=200, so on any unit shorter than 200 steps
                            # it never fires at all -- and the measurement would be taken,
                            # printed to a pod log that gets reaped, and never reach the
                            # campaign root. Measured: a 4-step unit published it zero times.
                            progress(
                                {
                                    "unit": unit.unit_id,
                                    "pos": scene_pos,
                                    "event": "obs_shape",
                                    "obs_shape": shape,
                                    "host": host,
                                    "t": time.time(),
                                }
                            )

                    need = [lane for lane in live if lane not in chunks or idxs[lane] >= len(chunks[lane])]
                    if need:
                        out = rollout.act(
                            [to_policy_obs(pre_obs[lane], emb, task=unit.scenario["task"]) for lane in need]
                        )
                        if len(out) != len(need):
                            raise ValueError(
                                f"rollout.act returned {len(out)} chunks for {len(need)} observations; "
                                f"a mismatched batch silently pairs a lane with another lane's actions"
                            )
                        for slot, lane in enumerate(need):
                            chunks[lane], idxs[lane] = out[slot], 0

                    acts = {}
                    for lane in live:
                        acts[lane] = np.asarray(chunks[lane][idxs[lane]])
                        idxs[lane] += 1

                    # `force_render` is a RENDER POLICY and nothing else -- it was called
                    # `record`, which read as "are we recording", and it never gated recording:
                    # the append below is unconditional, and what decides whether a frame is
                    # persisted is the SINK (PackageSink writes, EvalJsonlWriter drops).
                    #
                    # DO NOT be tempted to gate `writers[...].append` on it. EvalJsonlWriter
                    # counts every frame it drops and `commit` reports that count as the row's
                    # `steps`, so a gate there would silently write RENDERED steps where every
                    # consumer reads CONTROL steps.
                    #
                    # A recording sink still forces render=True, because a render-skipped step
                    # has no pixels and therefore cannot be a training frame. Everything else
                    # about the decision -- the phase, and the term for "the next iteration
                    # queries the policy" -- lives in `render_decision`, shared with eval.
                    #
                    # `steps[lane] + 1` because eval decides AFTER its counter advances and
                    # this loop decides before. Same instant, two clocks.
                    render_now = render_decision(
                        forced=force_render,
                        render_every=render_every,
                        live_steps=[steps[lane] + 1 for lane in live],
                        will_query=any(idxs[lane] >= len(chunks[lane]) for lane in live),
                    )

                    for so in sess.step(acts, render=render_now):
                        steps[so.lane] += 1
                        rewards[so.lane] += so.reward
                        writers[so.lane].append(
                            make_frame(
                                step_idx=steps[so.lane] - 1,
                                view=to_record_view(pre_obs[so.lane], emb),
                                action=acts[so.lane],
                                reward=so.reward,
                                terminated=so.terminated,
                                truncated=so.truncated,
                                info=so.info,
                                policy_info={"weight_version": pinfo.weight_version},
                                episode_id=sess.lane_key(so.lane).slug(),
                            )
                        )

                    # Trigger on the POST-step count, which is also the count reported; the
                    # pre-step minimum fires one step late and publishes 201 under a rule
                    # that says 200.
                    if progress is not None and min(steps[lane] for lane in live) % _PROGRESS_EVERY == 0:
                        progress(
                            {
                                "unit": unit.unit_id,
                                "pos": scene_pos,
                                "steps": dict(steps),
                                "host": host,
                                "t": time.time(),
                            }
                        )

                # The session owns termination AND truncation -- `step()` must report
                # truncated at the scenario's max_steps, exactly as the production b1k loop
                # does (`steps[i] >= max_steps -> done[i] = True`). The `range(max_steps)`
                # above is a guard against a session that never stops, not the mechanism.
                #
                # Reaching it with lanes still live means the last frame written carries
                # `terminated=False, truncated=False`, violating "the last frame has exactly
                # one of terminated | truncated". That frame cannot be patched -- it is
                # already encoded -- so the turn is abandoned rather than committed. A
                # silently-unflagged final frame is exactly the kind of defect that trains
                # badly with nothing downstream to catch it.
                stuck = [lane for lane in lanes if not sess.done(lane)]
                if stuck:
                    raise RuntimeError(
                        f"session ran {max_steps} steps without terminating lanes {stuck}; "
                        f"the simulator must truncate at scenario['max_steps']"
                    )

                # Read AFTER the whole step loop, for every lane -- including one that
                # ended hundreds of steps before its siblings. `outcome()` and `metrics()`
                # must therefore be LATCHED terminal values (SimSession invariant 5). A
                # session that recomputes them live is reporting the state the lane drifted
                # into while it was being stepped with zero actions, not the state it ended
                # in, and a mislabelled SUCCESS->FAILURE is silent everywhere downstream.
                for lane in lanes:
                    outcome, metrics = sess.outcome(lane), sess.metrics(lane)
                    # MEASURED, NOT ENFORCED -- and the asymmetry with the observation check
                    # above is the whole point.
                    #
                    # The observation is wire-safe by construction. These are not: they are
                    # whatever OmniGibson's AgentMetric and TaskMetric aggregates put in a
                    # dict, and `sinks/package.py::_jsonable` ends in a `str(obj)` catch-all
                    # precisely because unknown objects are known to arrive -- which means
                    # today they are already being written into episode.json as
                    # `<... object at 0x7f...>`, an address that differs every run.
                    #
                    # So the payload has never been characterised, and a check that RAISED
                    # here would cost a whole turn's committed episodes to report something
                    # we have no measurement of. Collect the distribution first. Promote this
                    # to a raise once a real run says the set is empty -- and if it is not
                    # empty, the finding is the fix, not this line.
                    try:
                        assert_wire_safe(metrics, f"metrics(lane={lane})")
                    except SimProtocolError as exc:
                        wire_faults.append(str(exc))
                    uri = writers[lane].commit(outcome, metrics)
                    committed.add(lane)
                    if sess.lane_key(lane).slug() in landed:
                        # This lane was BACKED BY A LANDED PACKAGE: it was re-run because a
                        # sibling in the unit had not landed, and `_LandedWriter.commit`
                        # discarded this run's outcome to leave the published artifact
                        # untouched. So the outcome below is THIS run's and the package on
                        # disk holds the original one, and the two may differ -- the policy
                        # is not reproducible (measured across all 385 round-3 pairs: one
                        # pair with identical inputs gave a failure and a success).
                        #
                        # Marked rather than suppressed, because the number is real and the
                        # rollout happened. Consumers that need what is ON DISK read the
                        # census, which is the campaign's authoritative count either way.
                        metrics = {**metrics, "rerun_of_landed": True}
                    eps.append(
                        EpisodeResult(
                            key=sess.lane_key(lane),
                            outcome=outcome,
                            steps=steps[lane],
                            reward_sum=rewards[lane],
                            lane=lane,
                            position_in_scene=scene_pos,
                            metrics=metrics,
                            artifact_uri=uri,
                        )
                    )
            finally:
                _abort_all(writers, committed, "turn aborted")
            scene_pos += 1

    except Exception as exc:  # noqa: BLE001 -- any sim/policy fault is a re-dispatchable unit
        # The in-flight turn's writers were aborted above, so nothing carries a READY
        # marker and those episodes are invisible by definition. The UnitSource re-offers
        # the unit; episodes that DID commit are skipped by already_landed().
        return UnitResult(
            unit.unit_id,
            eps,
            time.monotonic() - t0,
            boot_s,
            host,
            gpu,
            error=repr(exc),
            error_class=classify_error(exc),
        )
    finally:
        try:
            sess.close()
        except Exception:  # noqa: BLE001 -- a failed close must not mask the real fault
            pass

    if wire_faults:
        # One line per unit, not per episode: the same metric key from the same aggregate
        # produces the same fault on every lane, so per-episode logging would report one
        # defect twenty times and hide how many DISTINCT ones there are.
        print(
            f"[driver] unit {unit.unit_id}: {len(wire_faults)} metrics payload(s) are not "
            f"wire-safe; first: {sorted(set(wire_faults))[0]}",
            flush=True,
        )
    return UnitResult(unit.unit_id, eps, time.monotonic() - t0, boot_s, host, gpu)
