# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The retire verdict, the coverage rule, and the sink's optional facts.

Three things the review asked to make typed, and all three have the same failure mode: they
are decided by a rule spelled at more than one call site, so the copies can disagree without
anything going red. These tests are the differential that would notice.

``CollectorBody.__init__`` builds a simulator and a sink, so these drive ``_retire`` on an
instance made with ``object.__new__`` and given only the two attributes it touches. That is a
deliberate trade: the alternative is a fake simulator, and a test that needs a simulator to
check a bookkeeping verdict tends to get skipped in the venv where it matters.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.embodied.sink import SinkInfo, sink_info, sink_wants_extrinsics
from cosmos_framework.posttrain.embodied.source import Lease, RetireDisposition, StaticUnits
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, EpisodeResult, Outcome, UnitResult, WorkUnit
from cosmos_framework.posttrain.roles.env.collector import CollectorBody

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _unit(n: int = 2) -> WorkUnit:
    return WorkUnit.make(
        suite="example",
        scenario={"task": "demo", "max_steps": 4, "num_envs": 1},
        episodes=tuple(EpisodeKey(suite="example", task="demo", instance=i, seed=i) for i in range(n)),
        est_minutes=1.0,
    )


def _result(unit: WorkUnit, *, landed: int | None = None, error: str | None = None) -> UnitResult:
    landed = len(unit.episodes) if landed is None else landed
    episodes = [
        EpisodeResult(
            key=k, outcome=Outcome.SUCCESS, steps=1, artifact_uri=f"file:///{k.slug()}" if i < landed else None
        )
        for i, k in enumerate(unit.episodes)
    ]
    return UnitResult(unit.unit_id, episodes, 1.0, 0.1, "host", error=error)


class _Src:
    """A source that answers however the test says, and records what it was asked."""

    def __init__(self, *, disposition=None, revoked_bump=0, incomplete_bump=0):
        self._disposition, self._bump = disposition, revoked_bump
        self._incomplete_bump = incomplete_bump
        self.revoked = 0
        self.incomplete = 0
        self.completed: list[str] = []
        self.abandoned: list[tuple[str, str]] = []

    def complete(self, lease, result):
        self.completed.append(lease.unit.unit_id)
        self.revoked += self._bump
        self.incomplete += self._incomplete_bump
        return self._disposition

    def abandon(self, lease, reason):
        self.abandoned.append((lease.unit.unit_id, reason))


def _body() -> CollectorBody:
    body = object.__new__(CollectorBody)
    body._declined = 0
    body._held = []
    return body


def _lease(unit: WorkUnit) -> Lease:
    return Lease(unit, "tok", 0.0)


# ── the retire verdict ──────────────────────────────────────────────────────


def test_a_source_that_records_the_retirement_reads_as_completed():
    unit, body, src = _unit(), _body(), _Src(disposition=RetireDisposition.COMPLETED)
    assert body._retire(src, _lease(unit), _result(unit)) == "completed"
    assert src.completed == [unit.unit_id]
    assert body._declined == 0


def test_a_source_that_declines_reads_as_revoked_without_inspecting_a_counter():
    """The counter stays at zero: the verdict comes from the RETURN, not from private state."""
    unit, body, src = _unit(), _body(), _Src(disposition=RetireDisposition.REVOKED)
    assert body._retire(src, _lease(unit), _result(unit)) == "revoked"
    assert src.revoked == 0, "the typed answer must not need the counter"
    assert body._declined == 1


def test_the_bare_string_a_json_round_trip_leaves_behind_is_still_understood():
    """``RetireDisposition`` is a ``str`` Enum and a gateway hop may flatten it to its value."""
    unit, body, src = _unit(), _body(), _Src(disposition="revoked")
    assert body._retire(src, _lease(unit), _result(unit)) == "revoked"


def test_an_answer_the_protocol_does_not_define_is_refused_not_guessed():
    unit, body, src = _unit(), _body(), _Src(disposition="mostly")
    with pytest.raises(ValueError):
        body._retire(src, _lease(unit), _result(unit))


# ── the migration affordance, and the reason it exists ──────────────────────


def test_a_source_predating_the_typed_return_still_reports_a_refusal():
    """A source written before the typed return yields None; its refusals must not read as wins.

    `DirLeaseSource` was the motivating case and now answers in the new shape, so this
    covers any out-of-series implementation rather than a specific one.
    """
    unit, body, src = _unit(), _body(), _Src(disposition=None, revoked_bump=1)
    assert body._retire(src, _lease(unit), _result(unit)) == "revoked"
    assert body._declined == 1


def test_a_source_predating_the_typed_return_that_did_not_refuse_reads_as_completed():
    unit, body, src = _unit(), _body(), _Src(disposition=None, revoked_bump=0)
    assert body._retire(src, _lease(unit), _result(unit)) == "completed"
    assert body._declined == 0


def test_the_shipped_source_answers_in_the_new_shape():
    """`StaticUnits` is the in-series implementation; it must not need the fallback."""
    unit = _unit()
    src = StaticUnits([unit])
    lease = src.claim(budget_minutes=99.0, owner="t")[0]
    assert src.complete(lease, _result(unit)) is RetireDisposition.COMPLETED


# ── the coverage rule, shared rather than recomputed ────────────────────────


def test_a_unit_missing_an_episode_is_handed_back_not_retired():
    unit, body, src = _unit(), _body(), _Src(disposition=RetireDisposition.COMPLETED)
    assert body._retire(src, _lease(unit), _result(unit, landed=1)) == "incomplete"
    assert src.completed == []
    assert src.abandoned and "1/2 episodes did not land" in src.abandoned[0][1]


def test_a_faulted_unit_is_abandoned_before_coverage_is_even_asked():
    unit, body, src = _unit(), _body(), _Src(disposition=RetireDisposition.COMPLETED)
    assert body._retire(src, _lease(unit), _result(unit, landed=0, error="boom")) == "error"
    assert src.abandoned == [(unit.unit_id, "boom")]


def test_episodes_already_on_disk_count_as_covered():
    """The rule is slug coverage, not row count -- a fully-resumed unit retires."""
    unit = _unit()
    res = UnitResult(unit.unit_id, [], 1.0, 0.1, "host", skipped_landed=tuple(k.slug() for k in unit.episodes))
    assert res.missing(unit) == set()
    body, src = _body(), _Src(disposition=RetireDisposition.COMPLETED)
    assert body._retire(src, _lease(unit), res) == "completed"


def test_the_collector_and_the_coverage_helper_cannot_disagree():
    """Whatever `_retire` hands back is exactly what `UnitResult.missing` names."""
    unit = _unit(3)
    res = _result(unit, landed=2)
    assert res.missing(unit) == {unit.episodes[2].slug()}
    body, src = _body(), _Src(disposition=RetireDisposition.COMPLETED)
    body._retire(src, _lease(unit), res)
    assert "1/3" in src.abandoned[0][1]


# ── the sink's optional facts ───────────────────────────────────────────────


class _Sink:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.mark.parametrize(
    "sink,expected",
    [
        (_Sink(), {}),
        (_Sink(sr={}), {}),
        (_Sink(sr={"demo": [1, 2]}), {"sr": {"demo": [1, 2]}}),
    ],
    ids=["no-opinion", "committed-nothing", "committed-something"],
)
def test_one_rule_decides_whether_a_run_reports_success_rates(sink, expected):
    assert sink_info(sink).summary_fields() == expected


def test_absent_and_empty_stay_distinguishable_even_though_both_omit_the_key():
    """`{}` means "resumed everything"; absence means "not that kind of sink"."""
    assert sink_info(_Sink()).success_rates is None
    assert sink_info(_Sink(sr={})).success_rates == {}


def test_a_sink_that_says_nothing_still_gets_extrinsics():
    """The asymmetric default: silence must not land an all-zero robot2cam."""
    assert sink_info(_Sink()).wants_extrinsics is True
    assert sink_wants_extrinsics(_Sink()) is True
    assert sink_wants_extrinsics(_Sink(wants_extrinsics=False)) is False


def test_the_named_predicate_and_the_typed_value_are_the_same_answer():
    for sink in (_Sink(), _Sink(wants_extrinsics=False), _Sink(wants_extrinsics=True)):
        assert sink_wants_extrinsics(sink) is sink_info(sink).wants_extrinsics


def test_a_junk_sr_is_no_opinion_rather_than_a_crash_at_the_end_of_a_run():
    """A body returns this dict after the work is done; raising here would discard it."""
    assert sink_info(_Sink(sr=["not", "a", "mapping"])).success_rates is None
    assert isinstance(sink_info(_Sink()), SinkInfo)


# ── the third disposition, which a boolean could not carry ──────────────────


def test_a_source_that_refuses_on_coverage_reads_as_incomplete_not_revoked():
    """The two refusals mean opposite things about whether the data exists.

    `revoked` says the work landed under a successor's claim -- the loop treats it as harmless
    and does not advance its consecutive-failure breaker. `incomplete` says nothing landed.
    Collapsing them, which is all a boolean could do, hides exactly what the breaker reads.
    """
    unit, body, src = _unit(), _body(), _Src(disposition=RetireDisposition.INCOMPLETE)
    assert body._retire(src, _lease(unit), _result(unit)) == "incomplete"
    assert body._declined == 1


def test_the_incomplete_disposition_survives_a_json_round_trip_too():
    unit, body, src = _unit(), _body(), _Src(disposition="incomplete")
    assert body._retire(src, _lease(unit), _result(unit)) == "incomplete"


def test_a_source_predating_the_typed_return_can_still_report_incomplete():
    """Only because this shape publishes a SECOND counter -- which is the argument for the type.

    The fallback can separate the two refusals here solely because `DirLeaseSource` happens to
    expose both `revoked` and `incomplete`. Any other implementation has one counter or none,
    and reads as COMPLETED by default. That default is the failure the return value removes.
    """
    unit, body, src = _unit(), _body(), _Src(disposition=None, incomplete_bump=1)
    assert body._retire(src, _lease(unit), _result(unit)) == "incomplete"
    assert body._declined == 1
