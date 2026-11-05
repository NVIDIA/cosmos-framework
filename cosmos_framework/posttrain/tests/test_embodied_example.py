# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The integration template, tested as the thing an external team copies.

`sims/example.py` had no tests at all, which is the wrong place to have none: it is the file
an integrator reads first and imitates, so a defect in it is copied rather than hit. Three of
the assertions below pin behaviour that was WRONG when first written and was found by running
the template rather than reading it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

from cosmos_framework.posttrain.embodied import embodiment as emb_mod
from cosmos_framework.posttrain.embodied.driver import run_unit
from cosmos_framework.posttrain.embodied.registry import build_sim, build_sink
from cosmos_framework.posttrain.embodied.sim import SimProtocolError
from cosmos_framework.posttrain.embodied.sims.example import ZeroRollout
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome, WorkUnit
from cosmos_framework.posttrain.roles.env.collector import CollectorBody


def _unit(n: int, *, max_steps: int, scenario: dict | None = None) -> WorkUnit:
    eps = tuple(EpisodeKey(suite="demo", task="demo", instance=i, seed=i, attempt=0) for i in range(n))
    return WorkUnit(
        unit_id="u0",
        suite="demo",
        episodes=eps,
        est_minutes=1.0,
        scenario={"task": "demo", "max_steps": max_steps} if scenario is None else scenario,
    )


def _run(n_eps: int, lanes: int, max_steps: int, **kw):
    sim = build_sim("example", n_lanes=lanes, max_steps=max_steps)
    return run_unit(
        _unit(n_eps, max_steps=max_steps, **kw),
        sim,
        ZeroRollout(7),
        build_sink("null"),
        emb_mod.get_embodiment("example"),
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_documented_run_produces_the_documented_output():
    """`docs/reference/embodied.md` prints this exact run. A doc nobody executes drifts."""
    res = _run(4, lanes=2, max_steps=6)
    assert res.error is None
    got = [(e.key.instance, e.outcome, e.steps) for e in res.episodes]
    assert got == [(0, Outcome.SUCCESS, 3), (1, Outcome.FAILURE, 6), (2, Outcome.SUCCESS, 3), (3, Outcome.FAILURE, 6)]


@pytest.mark.L0
@pytest.mark.CPU
def test_the_last_turn_is_partial_because_n_lanes_is_per_turn():
    """`n_lanes` describes the CURRENT turn, not the session.

    Three episodes over two lanes means turn 2 binds ONE lane. A backend that returns a
    constant here makes the driver ask for a lane that was never bound -- which is a
    `KeyError` from inside the driver, and is what the first version of the template did.
    """
    res = _run(3, lanes=2, max_steps=6)
    assert res.error is None
    assert len(res.episodes) == 3, "the partial final turn dropped an episode"


@pytest.mark.L0
@pytest.mark.CPU
def test_outcome_is_latched_and_does_not_contradict_the_frame():
    """An episode that terminates ON its last permitted step is a SUCCESS.

    Deriving the outcome from `steps < max_steps` reads that as "ran out the budget". At
    `max_steps=1` EVERY successful episode terminates on its last permitted step, so the
    template reported `FAILURE` for a frame carrying `terminated=True, reward=1.0`. The
    outcome must be latched where the episode ends, not recomputed later from a counter.
    """
    res = _run(1, lanes=1, max_steps=1)
    assert res.error is None
    (ep,) = res.episodes
    assert ep.reward_sum == 1.0, "precondition: this episode did reach its goal"
    assert ep.outcome is Outcome.SUCCESS, (
        f"outcome {ep.outcome} contradicts the frame (reward={ep.reward_sum}). "
        f"The outcome is being derived from the step count instead of latched."
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_an_unregistered_embodiment_raises_instead_of_silently_substituting():
    """The failure mode this replaces was silent and misattributed.

    `MAPS.get(name) or EXAMPLE_EMBODIMENT` handed the example's camera and proprio keys to
    any backend whose embodiment was not registered -- including when the name was passed
    explicitly -- and the eventual `MissingObsKey` named an embodiment the author never asked
    for. The error must name what IS registered and how to register.
    """
    with pytest.raises(KeyError) as ei:
        emb_mod.get_embodiment("my_isaaclab_robot")
    msg = str(ei.value)
    assert "my_isaaclab_robot" in msg
    assert "register_embodiment" in msg, "the error does not say how to fix it"
    assert "example" in msg, "the error does not say what IS registered"


@pytest.mark.L0
@pytest.mark.CPU
def test_registering_an_embodiment_needs_no_edit_to_a_core_module():
    """The claim `docs/reference/embodied.md` makes: adding a backend touches nothing in `embodied/`.

    It was false while `MAPS` was a dict literal with no hook -- an IsaacLab author had to
    edit a core module, which is the coupling this package exists to remove.
    """
    name = "test_only_probe_embodiment"
    probe = emb_mod.EmbodimentMap(
        name=name,
        cameras={"head": "probe::head"},
        proprio_key="probe::proprio",
        action_dim=3,
        state_dim=3,
    )
    try:
        emb_mod.register_embodiment(probe)
        assert emb_mod.get_embodiment(name) is probe
        emb_mod.register_embodiment(probe)  # identical re-registration is a no-op
        clash = emb_mod.EmbodimentMap(
            name=name,
            cameras={"head": "other::head"},
            proprio_key="other::proprio",
            action_dim=9,
            state_dim=9,
        )
        with pytest.raises(ValueError, match="already registered"):
            emb_mod.register_embodiment(clash)
    finally:
        emb_mod.MAPS.pop(name, None)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("bad", [{"task": "demo"}, {"task": "demo", "max_steps": "abc"}])
def test_a_malformed_scenario_returns_a_result_and_builds_no_scene(bad):
    """`run_unit` returns a `UnitResult` for a unit-level fault; it does not raise.

    The scenario read sat AFTER `sim.open()` and outside its guard, so a missing key escaped
    as a bare `KeyError` and the built scene was never closed -- a leaked Isaac scene per
    malformed unit. Reading it first also means a bad scenario costs no scene build at all.
    """
    built = []
    sim = build_sim("example", n_lanes=1, max_steps=4)
    original = sim.open
    sim.open = lambda u: (built.append(1), original(u))[1]

    res = run_unit(
        _unit(1, max_steps=4, scenario=bad),
        sim,
        ZeroRollout(7),
        build_sink("null"),
        emb_mod.get_embodiment("example"),
    )
    assert res.error is not None and "max_steps" in res.error
    assert built == [], "a malformed scenario still paid for a scene build"


@pytest.mark.L0
@pytest.mark.CPU
def test_the_template_satisfies_the_session_invariants_it_teaches():
    """`sim.py` numbers seven invariants; the template broke three of them.

    Each is silent when broken, which is why they are worth a test rather than a comment:

    * **1 and 6** -- `observe()` must be CACHED and must return the SAME OBJECT `step()`
      emitted, not an equal one. The template rebuilt the array on every call. That is the
      invariant `SimGatewayBody` relies on to avoid shipping the observation twice per step,
      so a copied backend that rebuilds doubles the wire traffic with every test still green.
    * **7** -- every method raises after `close()`. A closed OmniGibson session does not
      raise; its arrays go stale, so the loop writes the last frame of a released scene over
      and over into a package that passes every structural gate.

    `generation` is checked separately below because it is wrong in a different way: it
    reads correctly in a single-unit run.
    """
    sim = build_sim("example", n_lanes=2, max_steps=6)
    sess = sim.open(_unit(2, max_steps=6))
    assert sess.advance_turn()

    assert sess.observe([0])[0] is sess.observe([0])[0], (
        "invariant 1/6: observe() rebuilt the observation instead of returning the cached one"
    )
    outs = sess.step({0: [0.0] * 7, 1: [0.0] * 7})
    assert sess.observe([0])[0] is outs[0].obs, "invariant 6: observe()[i] must BE StepOut.obs, not merely equal to it"

    sess.close()
    for name, call in [
        ("observe", lambda: sess.observe([0])),
        ("step", lambda: sess.step({0: [0.0] * 7})),
        ("done", lambda: sess.done(0)),
        ("outcome", lambda: sess.outcome(0)),
        ("lane_key", lambda: sess.lane_key(0)),
        ("advance_turn", lambda: sess.advance_turn()),
    ]:
        with pytest.raises(SimProtocolError):
            call()


@pytest.mark.L0
@pytest.mark.CPU
def test_generation_counts_scene_builds_not_turns():
    """`sim.py` defines `generation` as incrementing once per `Simulator.open()`.

    The template returned the TURN index within a session, which is indistinguishable from
    correct until a cell runs a second unit -- and then every package is attributed to the
    wrong scene.
    """
    sim = build_sim("example", n_lanes=2, max_steps=6)
    first = sim.open(_unit(2, max_steps=6))
    second = sim.open(_unit(2, max_steps=6))
    assert second.generation == first.generation + 1, (
        f"generation went {first.generation} -> {second.generation} across two open() calls; "
        f"it must count scene builds, not turns."
    )
    # and it must NOT move as turns advance within one session
    g = first.generation
    first.advance_turn()
    assert first.generation == g, "generation moved on advance_turn(); it counts scene builds"


# ── the lazy plugin path must not be order-dependent ─────────────────────────


@pytest.mark.L0
@pytest.mark.CPU
def test_a_backend_supplies_its_own_embodiment_without_a_preload():
    """`CollectorBody(sim=X, embodiment=<X's own>)` must work in a process that knows nothing.

    A backend registers its `EmbodimentMap` when its MODULE is imported, and the module is
    imported lazily by `build_sim`. Resolving the embodiment before that call therefore failed
    with a bare `KeyError` unless the caller already knew to preload the backend -- which makes
    the advertised lazy plugin path order-dependent, and is exactly the shape a fresh
    `sim_server` process hits.

    RUN IN A SUBPROCESS, and that is not ceremony. `MAPS` is a process-global dict, so any
    earlier test in this session that touched the example backend has already registered
    `example` and this assertion would pass for a reason unrelated to what it checks -- the
    same way `tests/test_roles_core_registry.py` passed for years against an ambient registry.
    Only a fresh interpreter can answer the question.
    """
    probe = textwrap.dedent(
        """
        from cosmos_framework.posttrain.embodied.embodiment import MAPS
        assert "example" not in MAPS, f"not a fresh interpreter: {sorted(MAPS)}"
        from cosmos_framework.posttrain.roles.env.collector import CollectorBody
        body = CollectorBody(sim="example", sink="null", embodiment="example")
        assert body._emb.name == "example", body._emb.name
        print("OK")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[4]),
    )
    assert done.returncode == 0, f"stdout={done.stdout!r}\nstderr={done.stderr[-1500:]}"
    assert "OK" in done.stdout


@pytest.mark.L0
@pytest.mark.CPU
def test_an_unregistered_embodiment_names_the_ones_that_exist():
    """The other half: `MAPS[name]` raised a bare `KeyError`, `get_embodiment` explains.

    A `KeyError: 'nope'` is indistinguishable from the ordering bug above, which is how that
    bug survived -- both spellings produce the same two words.
    """
    with pytest.raises(KeyError, match="no embodiment named 'nope'"):
        CollectorBody(sim="example", sink="null", embodiment="nope")
