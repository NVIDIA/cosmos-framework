# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A simulator backend in ~150 lines, and a runner that drives it.

**This file is the integration template.** To put a new engine behind this framework you
implement two Protocols -- :class:`sim.Simulator` and :class:`sim.SimSession` -- register the
class under a name, and :func:`driver.run_unit` drives it. Nothing else about the framework
has to change, and nothing here imports Ray, torch or a config system.

`the external simulator backend` is the same two Protocols over Isaac Sim 5.1 + OmniGibson.

Run it::

    python -m cosmos_framework.posttrain.embodied.sims.example              # this fake, on CPU
    python -m cosmos_framework.posttrain.embodied.sims.example --sim behavior1k --task <task>

Both go through ``registry.build_sim(name)``, which is the point: the runner does not know
which engine it got.

TWO RULES, both enforced by ``tests/test_import_discipline.py``:

1. **Importing this module must stay free.** Boot the engine in ``Simulator._boot()`` or
   ``open()``, never at module scope -- ``build_sim()`` runs inside a Ray actor constructor,
   so a module-level ``import <engine>`` moves the boot to bind time and any ImportError
   surfaces as the unhelpful "no sim named ...".
2. **Do not import the framework from here.** The dependency runs framework -> plugin and
   never back.
"""

from __future__ import annotations

import numpy as np

from cosmos_framework.posttrain.embodied.embodiment import (
    EmbodimentMap,
    register_embodiment,
    to_sim_obs,
)
from cosmos_framework.posttrain.embodied.registry import register_sim
from cosmos_framework.posttrain.embodied.sim import SimCapabilities, SimProtocolError, StepOut
from cosmos_framework.posttrain.embodied.unit import EpisodeKey, Outcome, WorkUnit

#: An embodiment is a plain dataclass: which observation keys are cameras, which is proprio,
#: and how wide an action is. Declare yours next to your `Simulator` and REGISTER it at module
#: scope, exactly as below -- the call is what makes `--embodiment <name>` and
#: `SimCapabilities.embodiment` resolve. Nothing in `embodied/` has to be edited to add one.
EXAMPLE_EMBODIMENT = register_embodiment(
    EmbodimentMap(
        name="example",
        cameras={"head": "example::head::rgb"},
        proprio_key="example::proprio",
        action_dim=7,
        state_dim=7,
    )
)

_H, _W = 64, 64


class ExampleSession:
    """One open scene holding a batch of concurrent episodes.

    THE TURN CONTRACT, and it is the part a new backend gets wrong first. The driver runs
    ``while sess.advance_turn():`` and re-reads ``sess.n_lanes`` INSIDE the loop, so:

    * nothing is bound in ``__init__`` -- the first ``advance_turn()`` loads the first batch;
    * ``n_lanes`` is a property describing THIS turn, not a fixed width, because the last
      batch of a unit is usually partial. Returning a constant makes the driver ask for a lane
      that was never bound.
    """

    def __init__(self, unit: WorkUnit, *, n_lanes: int, max_steps: int, generation: int) -> None:
        self._unit = unit
        self._max_steps = max_steps
        self._generation = generation
        self._closed = False
        #: The observation handed out for each lane's CURRENT step. `observe()` returns this
        #: object rather than building a new one -- see invariant 6.
        self._obs_cache: dict[int, object] = {}
        width = max(1, n_lanes)
        eps = list(unit.episodes)
        self._turns = [tuple(eps[i : i + width]) for i in range(0, len(eps), width)]
        self._turn = -1
        self._bound: tuple[EpisodeKey, ...] = ()
        self._steps: dict[int, int] = {}
        self._done: dict[int, bool] = {}
        self._outcome: dict[int, Outcome | None] = {}

    @property
    def n_lanes(self) -> int:
        return len(self._bound)

    @property
    def generation(self) -> int:
        """Scene builds so far in this PROCESS -- not turns within this session.

        `sim.py` defines it as incrementing once per `Simulator.open()`. Returning a turn
        index instead looks right in a single-unit run and silently mis-attributes every
        package once a cell runs more than one unit.
        """
        return self._generation

    def _check_open(self) -> None:
        """Invariant 7: every method raises after `close()`."""
        if self._closed:
            raise SimProtocolError("session is closed")

    # -- SimSession ----------------------------------------------------------
    def lane_key(self, lane: int) -> EpisodeKey:
        self._check_open()
        return self._bound[lane]

    def _render(self, lane: int):
        """Build this lane's observation for its CURRENT step and cache it.

        Called from `advance_turn()` and from `step()` -- the two places the world actually
        changes -- and never from `observe()`.
        """
        rng = np.random.default_rng(self._bound[lane].seed + self._steps[lane])
        obs = to_sim_obs(
            {
                "example::head::rgb": rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8),
                "example::proprio": rng.standard_normal(7).astype(np.float32),
            }
        )
        self._obs_cache[lane] = obs
        return obs

    def observe(self, lanes):
        """Invariants 1 and 6: CACHED, and the SAME OBJECT `step()` emitted.

        Not `equal to` -- the same object. Rebuilding here is the tempting shortcut and it
        breaks the seam `SimGatewayBody` rests on: it relies on identity to avoid shipping
        the observation twice per step, so a session that rebuilds silently doubles the wire
        traffic while every test still passes. Re-rendering here would also make `observe()`
        advance the world, which is invariant 1.
        """
        self._check_open()
        return [self._obs_cache[lane] for lane in lanes]

    def dropped_obs_keys(self) -> tuple[str, ...]:
        """Keys the engine produced that this session does not forward. Nothing here."""
        return ()

    def step(self, actions, *, render: bool = True) -> list[StepOut]:
        self._check_open()
        outs = []
        for lane in sorted(actions):
            self._steps[lane] += 1
            truncated = self._steps[lane] >= self._max_steps
            # A stand-in for a task predicate. Deterministic on the episode so a run shows BOTH
            # terminal shapes: even instances reach their goal early, odd ones run out the
            # budget. A real backend asks the engine whether the task predicate holds.
            reaches_goal = self._bound[lane].instance % 2 == 0
            terminated = reaches_goal and self._steps[lane] >= max(1, self._max_steps // 2)
            self._done[lane] = terminated or truncated
            # LATCH the outcome at the moment the episode ends, rather than re-deriving it
            # later from the step count. The two disagree whenever an episode terminates ON
            # its last permitted step -- at `--max-steps 1` every successful episode does --
            # and a step-count rule then reports FAILURE for a frame that says
            # `terminated=True, reward=1.0`.
            if self._done[lane] and self._outcome[lane] is None:
                self._outcome[lane] = Outcome.SUCCESS if terminated else Outcome.FAILURE
            outs.append(
                StepOut(
                    lane=lane,
                    # the SAME object `observe()` will hand back -- invariant 6
                    obs=self._render(lane),
                    reward=1.0 if terminated else 0.0,
                    terminated=terminated,
                    truncated=truncated and not terminated,
                )
            )
        return outs

    def done(self, lane: int) -> bool:
        self._check_open()
        return self._done[lane]

    def outcome(self, lane: int) -> Outcome:
        self._check_open()
        # There is no TRUNCATED outcome: running out the step budget IS the failure, which is
        # how the b1k backend reads it too. `Transition.truncated` carries the WHY, per step.
        # Read the LATCH set in `step()`; a lane the driver never stepped is a failure.
        return self._outcome[lane] or Outcome.FAILURE

    def metrics(self, lane: int) -> dict:
        return {"steps": self._steps[lane]}

    def advance_turn(self) -> bool:
        """Bind the next batch onto the lanes. False when the unit is exhausted.

        Called BEFORE the first turn, so this is where binding happens -- not `__init__`.
        """
        self._check_open()
        self._turn += 1
        if self._turn >= len(self._turns):
            self._bound = ()
            return False
        self._bound = self._turns[self._turn]
        self._steps = {i: 0 for i in range(len(self._bound))}
        self._done = dict.fromkeys(self._steps, False)
        self._outcome = dict.fromkeys(self._steps, None)
        # Prime the cache: the driver calls `observe()` BEFORE the first `step()` of a turn,
        # and invariant 1 forbids rendering from inside `observe()`.
        self._obs_cache = {}
        for lane in range(len(self._bound)):
            self._render(lane)
        return True

    def close(self) -> None:
        self._closed = True
        self._bound = ()
        self._obs_cache = {}


class ExampleSimulator:
    """The engine handle. One per cell; `open()` is per unit."""

    def __init__(self, *, n_lanes: int = 1, max_steps: int = 8, **_ignored) -> None:
        self._n_lanes = n_lanes
        self._max_steps = max_steps
        #: Scene builds so far, for the life of this handle.
        self._generation = 0

    def capabilities(self) -> SimCapabilities:
        return SimCapabilities(
            protocol_version=1,
            backend="example",
            backend_version="0",
            embodiment=EXAMPLE_EMBODIMENT.name,
            max_lanes=self._n_lanes,
            camera_shapes={"head": (_H, _W)},
            proprio_width=EXAMPLE_EMBODIMENT.state_dim,
        )

    def probe(self) -> None:
        """Fail HERE if the engine cannot come up -- before a caller pays for `open()`."""

    def open(self, unit: WorkUnit) -> ExampleSession:
        self._generation += 1
        return ExampleSession(
            unit,
            n_lanes=self._n_lanes,
            max_steps=self._max_steps,
            generation=self._generation,
        )

    def wants_restart(self) -> bool:
        """Never. Nothing here degrades, so there is nothing a fresh process would fix.

        Implemented rather than omitted because it is part of ``Simulator`` and this file is
        the template: an engine that DOES recycle (``behavior1k`` restarts on a measured
        per-unit slowdown) answers True here and does nothing else. Taking the restart is the
        loop's job, in the one window where it holds no lease -- an engine that calls
        ``os._exit`` itself strands whatever the loop had already claimed.
        """
        return False

    def gpu_uuid(self) -> str | None:
        return None

    def close(self) -> None:
        pass


register_sim("example", ExampleSimulator)


class ZeroRollout:
    """The smallest thing that satisfies :class:`rollout.Rollout`.

    A real one loads a checkpoint and returns an action CHUNK per observation; the driver
    consumes ``horizon`` actions per query, which is why the seam is a chunk and not a step.
    """

    def __init__(self, action_dim: int, horizon: int = 1) -> None:
        self._dim, self._horizon = action_dim, horizon

    def act(self, obs_batch):
        return [np.zeros((self._horizon, self._dim), dtype=np.float32) for _ in obs_batch]

    def describe(self):
        from cosmos_framework.posttrain.embodied.rollout import RolloutInfo

        return RolloutInfo(
            name="zero",
            action_dim=self._dim,
            horizon=self._horizon,
            embodiment=EXAMPLE_EMBODIMENT.name,
        )


def _main(argv=None) -> int:
    """Bring a simulator up, run a unit through the real loop, print what came back.

    Deliberately NOT an entrypoint: no Ray, no config tree, no campaign. `run_unit` takes the
    four seams as plain objects, so this is all it takes to prove a backend is wired.
    """
    import argparse

    from cosmos_framework.posttrain.embodied.driver import run_unit
    from cosmos_framework.posttrain.embodied.embodiment import get_embodiment
    from cosmos_framework.posttrain.embodied.registry import build_sim, build_sink

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", default="example", help="registry name: example | behavior1k")
    ap.add_argument("--task", default="demo", help="task name (a real suite needs a real one)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--embodiment", default=None, help="defaults to the sim's own")
    ap.add_argument("--mode", default=None, help="task split, for suites that have one (b1k: public_test)")
    args = ap.parse_args(argv)

    # "example" is already registered: this module ran its `register_sim` on import. Any other
    # name is resolved by the registry's lazy `__import__`.
    sim = build_sim(args.sim, n_lanes=args.lanes, max_steps=args.max_steps)

    caps = sim.capabilities()
    print(f"[example] backend={caps.backend} v{caps.backend_version} embodiment={caps.embodiment}")
    print(f"[example] max_lanes={caps.max_lanes} cameras={dict(caps.camera_shapes)} proprio={caps.proprio_width}")
    sim.probe()

    # `get_embodiment` RAISES on an unknown name. The `.get(...) or EXAMPLE_EMBODIMENT` this
    # replaces silently handed the example's map to anyone whose backend named an embodiment it
    # had not registered -- including when the name was passed explicitly on the command line.
    emb = get_embodiment(args.embodiment or caps.embodiment)
    # `scenario` is the per-unit dict the DRIVER reads, not just the backend: it requires
    # `task` (driver.py:367, for the policy-side observation) and `max_steps` (driver.py:210,
    # the truncation budget the simulator must honour). A backend adds its own keys alongside
    # them -- b1k also reads `mode`, `seed`, `num_envs`, `env_wrapper` and `robot_config`,
    # all of which have defaults except the two above.
    scenario = {"task": args.task, "max_steps": args.max_steps, "num_envs": args.lanes}
    if args.mode:
        scenario["mode"] = args.mode

    unit = WorkUnit.make(
        suite=args.sim,
        scenario=scenario,
        episodes=tuple(EpisodeKey(suite=args.sim, task=args.task, instance=i, seed=i) for i in range(args.episodes)),
        est_minutes=0.1,
    )

    result = run_unit(unit, sim, ZeroRollout(emb.action_dim), build_sink("null"), emb)

    print(f"[example] unit={result.unit_id} wall={result.wall_s:.2f}s boot={result.boot_s:.2f}s")
    for ep in result.episodes:
        print(f"[example]   {ep.key.slug()}  {ep.outcome.name:<9} steps={ep.steps} reward={ep.reward_sum}")
    if result.error:
        print(f"[example] ERROR {result.error_class}: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
