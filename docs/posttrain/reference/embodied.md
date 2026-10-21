# `embodied/` — one rollout loop for RL, collection and evaluation

A simulator **fills the environment role** but does not implement
`roles/env/base.py::Env`. It is declared as a role in a Manifest, and its runtime protocol
is `SimSession`.

The package is three layers, and they land as three merge requests so each can be argued on
its own terms:

|                  | what it is                                                                                     | modules                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **the contract** | the protocol, the vocabulary it speaks, and the framework-side surface a Ray handle answers to | `sim.py`, `unit.py`, `roles/env/sim_role.py`                                                                |
| **the loop**     | the one obs→act→step driver, the four seams it takes, and the in-process role that runs it     | `driver.py`, `rollout.py`, `sink.py`, `source.py`, `embodiment.py`, `registry.py`, `roles/env/collector.py` |
| **an engine**    | a real backend behind the contract, and the transport that lets it live in its own container   | `wire.py`, `roles/env/sim_gateway.py`, `roles/env/sim_server.py`                                            |

Nothing above a layer is needed to read it, and nothing below it is optional. An integrator
writing a new backend touches only the third row, and only the parts of it under `sims/`.

## Why a simulator fills the Env role without implementing `Env`

**Two things are called `Env` here, and keeping them apart is the whole of this section.**
DESIGN §1.3's *Env-as-source* is a ROLE — the thing a rollout draws observations from.
`roles/env/base.py::Env` is a Python Protocol with a Gym signature. A simulator fills the
first and does not implement the second; what it implements is `SimSession`.

`roles/env/base.py` declares a Gym-shaped `Env` — `reset() -> obs`, `step(action) -> (obs,
reward, done, info)`. `DatasetEnv` and `ArithmeticEnv` really do satisfy it. **No simulator in
this repo ever has**, and the Protocol's own docstring conceded as much before this change.

The shape three independent in-tree implementations converged on instead is lane-batched and
caller-driven: the simulator holds `n_lanes` concurrent episodes, is stepped in a batch by its
caller, and owns truncation. That convergence — `roles/env/the external simulator backend`'s vec loop, the
fork's `eval_vec.VecRunner`, and cosmos-rl's B1K rollout — is the argument. Forcing it into
`reset()/step()` would mean either one lane per process, or a `step()` whose `action` is
secretly a batch and whose `done` is secretly a vector.

**This is a request for a decision on the shape, not a claim that the shape is enforced.**
`SimRole` is a Protocol that nothing subclasses, and `validate_sim_role` has no caller at this
commit — a Ray handle answers to any attribute name, so `isinstance(handle, SimRole)` is
unconditionally true and the check has to be an explicit call at bind. It becomes load-bearing
where the collector binds its simulator and where the gateway implements the surface; both
arrive later in this series. `sim_manifest()` is declare-only for the same reason, and says so.

## Where it sits in the framework

| DESIGN element               | how `embodied/` relates                                                                                                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §1.3 Env-as-source           | **Adopted, in the role sense.** `SimSession` is the runtime protocol that fills it — lane-batched and caller-driven rather than Gym `reset()/step()`, so it does not implement `roles/env/base.py::Env`.                  |
| §1.4 Trajectory / Transition | **Adopted.** `Transition` gains `truncated` and `info`, both additive with defaults: a time-limit cut is not a terminal state, and bootstrapping `V(s')` through the two differently is what `done` alone cannot express. |
| §4.0 naming (normative)      | `Rollout` is the generation seam; `policy` stays reserved for the trainer/learner role.                                                                                                                                   |
| §4.2 colocation tiers / §7   | sim↔policy colocation is a placement instance, not a mechanism here: `bind_colocated_groups(..., colocate="host_affinity")`, unchanged.                                                                                   |
| §9 Shared library bucket     | `embodied/` imports `schema` and nothing else outside stdlib+numpy — enforced by gate G4.                                                                                                                                 |
| roles taxonomy               | No new role kind. `sim_manifest()` declares `env` and `policy`, matching `eval_system.md` §8.                                                                                                                             |

## The import discipline (gate G4)

Every module directly under `embodied/` imports **stdlib, numpy and `schema` only**. That is
what lets one core load in the Isaac venv (py3.11 + Isaac 5.1, no torch, no i4), the policy
venv (torch + i4, no omnigibson) and the trainer venv alike.

**It is also why this is a top-level package rather than a subtree of `roles/`.**
`roles/` is the framework's role taxonomy; this is a library that two different venvs load.
The acute symptom was that `roles/env/__init__.py` eagerly re-exported `wfm`, which imports
torch at module scope, so importing anything under `roles.env` dragged the training stack in.
That barrel is lazy as of this MR and `test_the_roles_env_barrel_is_lazy` keeps it that way;
measured, the closure went from ten packages to `["projects"]`. But a lazy import is one line
away from being eager again, and a directory boundary is not. The boundary is the statement;
heavy imports live in backend subpackages, reached lazily by name.

`tests/test_import_discipline.py` measures it as a **delta over a numpy-only interpreter** —
importing the whole core must add exactly `["projects"]`. A blacklist of known-heavy packages
was tried first and rejected: it silently stops covering anything added after it is written,
which is the drift the gate exists to catch. The gate walks the directory rather than naming
its modules, for the same reason.

The same file carries the entrypoint rule: **no entrypoint imports `ray` at module scope.**
CI's unit-test image has no Ray, and `policy_eval.py` imported it at the top — which took out
the tests that drive the real eval aggregator with the real sink's output. They failed on the
import line, having never reached the code they exist to test, and passed locally because a
dev host has Ray installed. A pure function that cannot be imported without a cluster library
is untestable by construction.

## Layout

| module                  | what it is                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `sim.py`                | the simulator seam: `Simulator` / `SimSession`, `SimCapabilities`, the error classes                                   |
| `unit.py`               | the work-unit vocabulary — `WorkUnit`, `EpisodeKey`, `Outcome`. Zero imports                                           |
| `driver.py`             | the ONE loop. Everything else here exists so this file has one shape to speak to                                       |
| `embodiment.py`         | the observation contract: `obs_spec()` / `check_obs()`, and the maps between engine, policy and record views           |
| `rollout.py`            | the generation seam. `policy` stays reserved for the trainer role                                                      |
| `sink.py` / `source.py` | `TrajectorySink` and `UnitSource` — the two things the three uses differ in                                            |
| `registry.py`           | lazy `build_sim` / `build_rollout` / `build_sink`, by name. This is what keeps the heavy imports out of the core       |
| `plugin_surface.py`     | what ships into the simulator container, declared in code rather than prose                                            |
| `wire.py`               | the framing for the two-container split: one length-prefixed control channel over `AF_UNIX`, bulk arrays in `/dev/shm` |

`roles/env/collector.py::CollectorBody` is the **in-process implementation of `SimRole`** — it
owns a driver, a sink and a source and answers the two calls a Ray handle makes. `SimRole` stops
being a declaration here.

## Running it

The example backend is a complete `Simulator`/`SimSession` in ~180 lines with no engine behind
it. It is the integration template, and it runs:

```bash
python -m cosmos_framework.posttrain.embodied.sims.example --episodes 4 --lanes 2 --max-steps 6

[example] backend=example v0 embodiment=example
[example] max_lanes=2 cameras={'head': (64, 64)} proprio=7
[example]   demo/i0000_s0_a0  SUCCESS   steps=3 reward=1.0
[example]   demo/i0001_s1_a0  FAILURE   steps=6 reward=0.0
[example]   demo/i0002_s2_a0  SUCCESS   steps=3 reward=1.0
[example]   demo/i0003_s3_a0  FAILURE   steps=6 reward=0.0
```

The same runner drives any registered backend — it resolves through `registry.build_sim` and
never learns which one it got.

### What a backend author has to get right

> Integrating a simulator from OUTSIDE this repo? Start at
> [`sim-integration.md`](../howto/sim-integration.md) — the task-oriented version of this section, with
> the two runnable commands and the CI bar. What follows is the design rationale behind it.

Four things are not obvious from the Protocols, and all four are what the example gets wrong if
written from the type signatures alone:

|                     | the contract                                                                                                                                             | decided at                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `advance_turn()`    | the driver runs `while sess.advance_turn():` and calls it BEFORE the first turn, so binding happens there and **not** in `__init__`                      | `driver.py:222`                  |
| `n_lanes`           | a **property** describing the CURRENT turn. The last batch of a unit is usually partial; a constant makes the driver ask for a lane that was never bound | `driver.py:223`                  |
| `Outcome`           | there is no `TRUNCATED`. Running out the step budget IS the failure; `Transition.truncated` carries the why, per step                                    | `unit.py::Outcome`               |
| `WorkUnit.scenario` | read by the DRIVER, not just the backend: `task` and `max_steps` are both required                                                                       | `driver.py:367`, `driver.py:210` |

### The one structural exception

Two rules hold for every module under `sims/`, both enforced by
`tests/test_import_discipline.py`: **importing the module must stay free** (the engine boots in
`Simulator._boot()`, not at module scope), and **a backend must not import the framework** (the
dependency runs framework → plugin and never back).

An adapter that subclasses an engine-owned type may require a reviewed lazy-import
exemption because the base class is needed at class-creation time.

Keep such a module off the registry path and import it at call time from a backend that has
already booted its engine. `the external simulator backend` does this; the cost of getting it wrong is an
`ImportError` three packages away, surfacing as the unhelpful `no sim named 'behavior1k'`.

## Two containers, one loop

OmniGibson pins python 3.11 and needs `SimulationApp` on the process main thread. The framework
venv has torch and i4 and pins its own python. Neither constraint bends, so the simulator runs
in its own image — and the loop has to reach across.

**The split is at `SimRole`, not at `Simulator`.** That choice is the whole design:

```
Ray actor (framework venv)
  SimRole                      <-- the container boundary is HERE
    |- CollectorBody           in-process: driver + sink + source + Simulator, one card
    `- SimGatewayBody          forwards over AF_UNIX + /dev/shm
                                    `- sim_server -> CollectorBody -> driver -> Simulator
```

`SimGatewayBody` is a drop-in for `CollectorBody`: one surface, two implementations, and the
driver never learns which it got. Splitting one layer lower — at `Simulator`, forwarding
`step()` — would put a **~94 MB chunk** on the wire where the observation is **3.9 MB**.

What this buys is not performance; it is the deletion of a dependency. Under
`sim_transport: direct` the simulator's raylet joins the GCS, and Ray compares python
**patch-for-patch** before admitting a node — which is why `docker/Dockerfile.behavior1k` pins
python to the framework image. Under `gateway` there is no sim raylet, so the pin buys nothing.
`test_the_sim_side_roots_import_without_ray` is what makes deleting it a safe commit to write
rather than a hopeful one: it imports every module in `plugin_surface.sim_side_roots()` with
`ray` blocked at the meta-path and requires success.

**The claim is Ray, not torch.** torch IS in the sim-side closure and belongs there — omnigibson
needs it, and the image asserts `SIM_TORCH`.

`sim_gateway.py` is **framework-side and engine-agnostic**: every behavior1k mention in it is a
default value, an env-var name or a socket filename. `plugin_surface` names it under `GATEWAY`
rather than `LOOP` precisely because the two files sit next to each other with nearly the same
name, and "which one ships into the image" is the question a reader will get wrong.

Restart budgets and the supervisor process are deliberately **not here** — that is leak
mitigation, a different subject on a different schedule. There is no `supervise` switch to
find: the parameter, the module constant and the branch that used them are absent rather than
present-and-refusing, so nothing in this file describes a path it cannot take. They arrive with
the process that implements them, and with the restart tests that would make them arguable.
