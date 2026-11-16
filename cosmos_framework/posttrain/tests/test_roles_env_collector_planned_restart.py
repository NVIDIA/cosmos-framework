# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The planned restart must not strand the unit the loop has already claimed.

The defect these cover shipped as a comment asserting its own safety. `the external simulator backend` decided
AND acted at the top of `open()`, reasoning that the previous unit's session was closed and its
result committed -- true, and about the wrong unit. `run_from_source` reaches `open()` holding
the NEXT unit's lease, and `os._exit` runs no `finally`, so `_release_held` never ran and the
claim outlived the process that owned it.

What that cost is measurable rather than theoretical. The claim plane's `_alive` proves a holder dead only
within its own host and pid namespace, so the pod that died could reclaim at once but every other
pod had to wait out the full `lease_seconds`; and each reclaim spent one of the unit's finite
generations, which at the eval default of `sim_units_per_generation=1` is one burnt generation
per unit.

Driven against ``StaticUnits`` -- the reference ``UnitSource``, which is what lets these exercise
the REAL loop with no filesystem and no second process. What is under test is the loop's ordering,
and that is source-independent by construction: any source that hands out leases is stranded by an
exit taken while one is held.

So the tests here are ordering tests, not restart tests: the question is never "did it exit", it
is "what did it hold when it did".
"""

from __future__ import annotations

import inspect
import os

import pytest

from cosmos_framework.posttrain.embodied.unit import (
    EpisodeKey,
    EpisodeResult,
    Outcome,
    UnitResult,
    WorkUnit,
)
from cosmos_framework.posttrain.roles.env import collector as collector_mod
from cosmos_framework.posttrain.roles.env.collector import CollectorBody

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Exited(BaseException):
    """Stands in for `os._exit`, which does not return.

    A recording no-op would let the loop run on past the restart and claim again -- three
    exits from a three-unit plan -- which is a test artefact, not the production shape, and
    it would hide whether anything is claimed AFTER the decision. `BaseException` so the
    loop's own `except Exception` handlers cannot swallow it, exactly as `os._exit` cannot
    be swallowed.
    """

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class _Sim:
    """The engine half of the contract: it ANSWERS, it never exits."""

    def __init__(self, wants: bool = False) -> None:
        self._wants = wants
        self.asked_while_holding: list[int] = []
        self.body: CollectorBody | None = None

    def wants_restart(self) -> bool:
        # Record what the loop was holding at the instant it asked. This is the whole test:
        # the answer is acted on with `os._exit`, so a non-empty `_held` here is a stranded unit.
        if self.body is not None:
            self.asked_while_holding.append(len(self.body._held))
        return self._wants

    def gpu_uuid(self) -> str | None:
        return "GPU-test"


class _Sink:
    def __init__(self, root: str) -> None:
        self.root = root


class _Emb:
    """Only what `run_from_source` reads off it before the first claim."""

    name = "test"
    action_dim = 7


def _units(n: int) -> list[WorkUnit]:
    return [
        WorkUnit.make(
            suite="b1k",
            scenario={"task": f"t{i}", "instances": [i], "seed": i},
            episodes=(EpisodeKey(suite="b1k", task=f"t{i}", instance=i, seed=i),),
            est_minutes=1.0,
        )
        for i in range(n)
    ]


def _body(sim: _Sim, root: str) -> CollectorBody:
    """A real `CollectorBody` with only the parts this loop does not exercise left out.

    `__init__` builds a sim, a sink and an embodiment from the registry; none of the three is
    what is under test, and constructing them would need a rendering backend. Everything the
    loop itself touches -- `_held`, `_owner`, `_retire`, `_release_held`, `_planned_restart` --
    is the production code.
    """
    body = object.__new__(CollectorBody)
    body._sim = sim
    body._sink = _Sink(root)
    body._emb = _Emb()
    body._owner = f"{os.uname().nodename}-{os.getpid()}"
    body._source = None
    body._held = []
    body._last_renew = 0.0
    body._revoked = 0
    body._revoked_ids = set()
    body._declined = 0
    body._force_render = False
    body._render_every = 1
    sim.body = body
    return body


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Drive the REAL loop: real `DirLeaseSource`, real claim/retire, only the unit stubbed."""
    monkeypatch.setattr(collector_mod, "validate_rollout", lambda *a, **k: None)
    exits: list[int] = []

    def _exit(code):
        exits.append(code)
        raise _Exited(code)

    monkeypatch.setattr(collector_mod.os, "_exit", _exit)

    def make(sim: _Sim, n_units: int = 3):
        body = _body(sim, str(tmp_path))
        plan = _units(n_units)

        def _ran(lease, policy, deadline_s):
            # A unit that RETIRES cleanly: `_retire` checks every declared episode landed,
            # so `artifact_uri` has to be set or the loop hands the unit back and the
            # restart window is never reached.
            eps = [
                EpisodeResult(key=k, outcome=Outcome.FAILURE, steps=1, artifact_uri=f"file:///{k.slug()}")
                for k in lease.unit.episodes
            ]
            return UnitResult(lease.unit.unit_id, eps, 1.0, 0.0, "h", "GPU-test")

        body._run_leased = _ran  # type: ignore[method-assign]
        return body, plan

    return make, exits, tmp_path


# ── the ordering, which is the defect ───────────────────────────────────────────


def test_the_loop_holds_no_lease_at_the_moment_it_asks(loop):
    """The question is asked in the lease-free window, not while a unit is claimed.

    `os._exit` runs no `finally`, so the ONLY safe moment to ask is after `_retire` has
    dropped the last lease and before the next `claim`. Asking at the top of `open()` -- where
    this used to live -- is asking while holding exactly one.
    """
    make, _exits, _root = loop
    sim = _Sim(wants=False)
    body, plan = make(sim, n_units=3)

    body.run_from_source(plan, policy=object(), source="static")

    assert sim.asked_while_holding, "the loop never asked -- the hook is not wired"
    assert set(sim.asked_while_holding) == {0}, (
        f"asked while holding {sim.asked_while_holding}; a non-zero count is a lease that "
        f"`os._exit` would strand, which is the whole defect"
    )


def test_a_planned_restart_leaves_nothing_held(loop):
    """After the restart the loop holds no lease -- there is nothing for a successor to wait out."""
    make, exits, _root = loop
    sim = _Sim(wants=True)
    body, plan = make(sim, n_units=3)

    with pytest.raises(_Exited):
        body.run_from_source(plan, policy=object(), source="static")

    assert exits == [9], f"expected one exit(9) (the fault injector's convention), got {exits}"
    assert body._held == [], (
        "a lease survived the restart: `os._exit` runs no `finally`, so this one is held by a "
        "process that no longer exists"
    )


def test_the_restart_is_not_taken_when_the_engine_does_not_ask(loop):
    """Positive control: the loop exits 9 only because the ENGINE said so."""
    make, exits, _root = loop
    sim = _Sim(wants=False)
    body, plan = make(sim, n_units=3)

    out = body.run_from_source(plan, policy=object(), source="static")

    assert exits == [], "exited without being asked to"
    assert out["units_completed"] == 3, out


def test_held_leases_are_handed_back_before_the_exit(loop, monkeypatch):
    """`_release_held` runs FIRST, so a batch claim can never re-introduce the defect.

    Today `claim(budget_minutes=0)` yields one unit at a time and `_retire` has already
    dropped it, so `_held` is empty here and the release is a no-op. That is exactly why it is
    worth pinning: the release is what makes the exit safe, and nothing else would notice if a
    later change started claiming batches.
    """
    make, exits, _root = loop
    sim = _Sim(wants=True)
    body, plan = make(sim, n_units=1)

    order: list[str] = []
    real_release = body._release_held

    def _release(src, reason):
        order.append(f"release:{reason}")
        return real_release(src, reason)

    def _exit(code):
        order.append(f"exit:{code}")
        raise _Exited(code)

    monkeypatch.setattr(body, "_release_held", _release)
    monkeypatch.setattr(collector_mod.os, "_exit", _exit)

    with pytest.raises(_Exited):
        body.run_from_source(plan, policy=object(), source="static")

    assert "exit:9" in order, order
    assert order.index("release:planned restart") < order.index("exit:9"), f"the exit ran before the release: {order}"


# ── the engine half: it decides, it does not act ─────────────────────────────────


def test_open_no_longer_exits_the_process():
    """The decision left `open()`. If it comes back, this is the test that says so."""
    b1k = pytest.importorskip("cosmos_framework.posttrain.embodied.sims.behavior1k")
    src = inspect.getsource(b1k.Behavior1KSimulator._release_previous_stage)
    assert "_exit" not in src, (
        "`_release_previous_stage` runs from the top of `open()`, which the loop reaches "
        "holding the next unit's lease. Exiting there strands it."
    )
    assert "_exit" not in inspect.getsource(b1k.Behavior1KSimulator.open)


@pytest.mark.parametrize(
    "generation, per_gen, want",
    [
        (0, "1", False),  # nothing built yet: there is no degradation to reset
        (1, "1", True),
        (2, "1", True),
        (1, "3", False),
        (3, "3", True),
        (4, "3", False),
        (5, "0", False),  # the knob off
        (5, "", False),
        (5, "three", False),  # unparseable is off, not a crash at a unit boundary
    ],
)
def test_wants_restart_counts_finished_units(monkeypatch, generation, per_gen, want):
    b1k = pytest.importorskip("cosmos_framework.posttrain.embodied.sims.behavior1k")
    monkeypatch.setenv("B1K_SIM_UNITS_PER_GENERATION", per_gen)
    sim = object.__new__(b1k.Behavior1KSimulator)
    sim._generation = generation
    assert sim.wants_restart() is want


def test_wants_restart_is_off_when_the_knob_is_absent(monkeypatch):
    b1k = pytest.importorskip("cosmos_framework.posttrain.embodied.sims.behavior1k")
    monkeypatch.delenv("B1K_SIM_UNITS_PER_GENERATION", raising=False)
    sim = object.__new__(b1k.Behavior1KSimulator)
    sim._generation = 7
    assert sim.wants_restart() is False
