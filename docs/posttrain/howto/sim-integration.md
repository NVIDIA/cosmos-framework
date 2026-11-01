# Integrating a simulator

For a team that owns a simulator and wants it driven by this repo's rollout loop — RL,
data collection and evaluation all run the *same* loop, so wiring it once wires it for all three.

`embodied/sims/example.py` is the reference implementation: ~270 lines, no engine, runnable
locally. This document is the task-oriented version of it. Read them together; where they
disagree, the code is right.

---

## 0. The ownership boundary

**You implement three things**, all on the engine side, all reached by name:

| you write       | where it lives | what it is                                                          |
| --------------- | -------------- | ------------------------------------------------------------------- |
| `EmbodimentMap` | your module    | which obs keys are cameras, which is proprio, how wide an action is |
| `Simulator`     | your module    | process-level handle: built once, `open()`ed per unit               |
| `SimSession`    | your module    | one scene build, N lanes: `observe` / `step` / `done`               |

**You must not need to touch any of these**, and if you find yourself editing one, that is a
bug report rather than an integration step:

- `roles/env/sim_role.py` — the framework-facing contract
- `roles/env/sim_gateway.py` — its framework-side body
- `embodied/wire.py` — the shared-memory + UDS transport
- `comm/` — framework transports
- `embodied/plugin_surface.py` — see §2; a new backend needs **no** edit here

You also do not implement the loop, the sink, the claim plane, retries, or the policy call.
`driver.run_unit` owns the obs→act→step loop and calls your session.

**Where your code runs.** The loop executes *inside the simulator process*, so your `Simulator`
is constructed in your own venv, with your own engine. The policy is reached through a proxy the
gateway answers; you never see Ray, and **Ray is not installed in the simulator venv** — importing
it at module scope will fail in production and is checked in CI (§5).

---

## 1. What to implement

### 1.1 `EmbodimentMap` — declare it, register it at module scope

```python
from cosmos_framework.posttrain.embodied.embodiment import EmbodimentMap, register_embodiment

MY_EMBODIMENT = register_embodiment(
    EmbodimentMap(
        name="my_robot",                       # must equal SimCapabilities.embodiment
        cameras={"head": "myrobot::head::rgb"}, # logical name -> YOUR flattened raw key
        proprio_key="myrobot::proprio",
        action_dim=7,                           # width of one action row
        state_dim=7,                            # proprio width the policy is guaranteed
    )
)
```

`cameras` maps the **logical** name the policy and the sink use to the **raw flattened key** your
simulator emits. That indirection is the entire point: your engine keeps its own key names.

Optional, only if you publish per-frame camera extrinsics: `extrinsics_key` (an
`[n_cameras, 7]` array of `(xyz, quat)` in the robot-base frame) and `extrinsics_order`. Leave
them unset and the sink records no extrinsics — that is a supported configuration, not a gap.

`register_embodiment` is what makes `--embodiment my_robot` and `SimCapabilities.embodiment`
resolve. `get_embodiment` **raises** on an unknown name; do not paper over it with
`MAPS.get(n) or DEFAULT`, which silently hands out the wrong map for an explicitly-named
embodiment.

### 1.2 `Simulator` — six methods

```python
class MySimulator:
    def capabilities(self) -> SimCapabilities: ...
    def probe(self) -> None: ...
    def open(self, unit: WorkUnit) -> "MySession": ...
    def wants_restart(self) -> bool: ...
    def gpu_uuid(self) -> str | None: ...
    def close(self) -> None: ...
```

- **`capabilities()` must be cheap and must not build a scene.** It is called at bind time and
  its whole value is that a wrong embodiment or a stale protocol version costs a handshake
  instead of an engine boot plus a multi-minute scene build. An implementation that builds a
  scene to answer has inverted it.

  ```python
  SimCapabilities(
      protocol_version=PROTOCOL_VERSION,
      backend="my_robot_sim",          # your registry name
      backend_version="",              # image digest / engine version, best effort
      embodiment="my_robot",           # MUST equal the EmbodimentMap name
      max_lanes=8,
      camera_shapes={"head": (224, 224)},  # the resolution you are CONFIGURED to render
      proprio_width=7,                     # = EmbodimentMap.state_dim
      supports_render_skip=False,
  )
  ```

  `camera_shapes` is the **request**, not the measured frame. The sink records the truth per
  episode, precisely because a wrapper that ignores the request would otherwise produce a corpus
  whose provenance claims a resolution it does not have.
- **`probe()`** raises if this process cannot render. Second line of defence; the primary card
  check is in the pod bootstrap.
- **`open(unit)`** builds the scene and returns a session. This is where the expensive work goes.
- **`wants_restart()`** answers whether this process should be torn down and rebuilt — it is a
  **question, never the act**. The loop asks it between units, in the one window where it holds
  no lease, and takes the restart itself. If you call `os._exit` from inside `open()` you will
  strand the unit the loop has already claimed. An engine with nothing to recycle returns `False`
  forever, which is what the example does.
- **`gpu_uuid()`** — the physical GPU this process was handed, for bad-card attribution.
- **`close()`** — shut the process-level handle down.

**A constructor that raises leaves nobody to close it.** If you start the engine from `__init__`,
a failed handshake orphans it on the card. Prefer a lazy `_boot()` on first use, as the b1k
backend does.

### 1.3 `SimSession` — one scene, N lanes

```python
    def lane_key(self, lane: int) -> EpisodeKey: ...
    def observe(self, lanes: Sequence[int]) -> list[ObsBundle]: ...
    def dropped_obs_keys(self) -> tuple[str, ...]: ...
    def step(self, actions: Mapping[int, Any], *, render: bool = True) -> list[StepOut]: ...
    def done(self, lane: int) -> bool: ...
    def outcome(self, lane: int) -> Outcome: ...
    def metrics(self, lane: int) -> dict[str, Any]: ...
    def advance_turn(self) -> bool: ...
    def close(self) -> None: ...
```

The shape is **lane-batched and caller-driven** — `observe(lanes)` / `step({lane: action})` — not
Gym `reset()/step()`. A vec-env steps N episodes together and the *caller*, not the env, owns the
chunked policy query. That is why a simulator fills the environment role without implementing
`roles/env/base.py::Env`.

Invariants worth calling out, because they are cheap to violate and expensive to debug:

- `observe()` after a `step()` must return **the same object** `StepOut.obs` carried. Cache it;
  do not re-render.
- Every method raises `SimProtocolError` after `close()`.
- `outcome(lane)` distinguishes `SUCCESS` / `FAILURE` / `SIM_ERROR`. **`FAILURE` is data, not an
  error** — only `SIM_ERROR` (and `ABANDONED`) means the unit must be re-dispatched. Conflating
  them makes a working pipeline look catastrophically broken.
- `advance_turn()` returns whether another turn exists; single-turn suites return `False`.

---

## 2. Registration, lazy imports, and `preload`

One line, at module scope:

```python
from cosmos_framework.posttrain.embodied.registry import register_sim

register_sim("my_robot_sim", MySimulator)
```

`build_sim("my_robot_sim", **kw)` resolves a name by importing
`cosmos_framework.posttrain.embodied.sims.my_robot_sim` and then looking the name up. So:

- **Put your module in `embodied/sims/`** and name the file after the registry name.
- **Your module may depend on the engine. Importing it may not.** `sims/` is exempt from the
  *core* import rule — you are allowed to name your engine — but the import itself, and the
  boot, belong inside `Simulator._boot()`/`open()`. The module must import cleanly on a machine
  where the engine is not installed, because that is where the registry walks it. §5's
  `test_the_backends_import_without_their_engines` is that rule, and it is the one an author
  hoisting `import my_engine` to module top passes every *other* gate against.
- **The one exemption is structural, and it is declared, not inferred.** A module that

  Engine-subclass adapters may require an explicit, reviewed lazy-import
  exemption because their base class is needed at class-creation time.
- **`embodied/` core modules import stdlib + numpy + `schema` only.** That is gate G4 and it is
  what lets the core load in the simulator venv, the policy venv and the trainer venv alike. Your
  backend is *not* core; the exemption is the subpackage.

**`plugin_surface.py` needs no edit.** `SUBPACKAGES = ("sims", "rollouts", "sinks", "sources")` is
already the declaration, and `core_modules()` is a **walk of the directory, not a literal** — so a
new file under `sims/` is covered without touching anything. If you find yourself editing that
file to add a backend, you are in the wrong directory.

**`preload`** is a list of module names the *simulator* imports before building. Under the gateway
they are imported by the **child** — the interpreter that builds your engine — and are passed
through, never imported framework-side. Use it for engine modules that must be resolved before
the first scene build.

---

## 3. Observations, actions, and `WorkUnit.scenario`

### 3.1 Observation

Build a flat dict keyed by *your* raw key names and box it:

```python
from cosmos_framework.posttrain.embodied.embodiment import to_sim_obs

obs = to_sim_obs({
    "myrobot::head::rgb": head_frame,   # np.uint8 [H, W, C]
    "myrobot::proprio": proprio,        # np.float32 [>= state_dim]
})
```

`to_sim_obs` **adopts the dict without copying**. Build a fresh one each step and hand over
ownership; do not reuse a buffer you will mutate.

The framework flattens it into the policy's key space with `to_policy_obs(obs, emb, task=...)`.
You never call that — it is the driver's, and it is the only translator.

Camera arrays are `[H, W, C]` `uint8`. Proprio must be at least `state_dim` wide; the record view
refuses to go under.

### 3.2 Action

The policy returns one chunk per observation, shaped **`[H, action_dim]`**. The driver takes
**one row per control step** and calls:

```python
session.step({lane: row})     # row: np.ndarray [action_dim]
```

There is **no action adapter** — the row reaches you exactly as the model emitted it. Whatever
your engine needs it to mean, it must mean, which is why `action_dim` is checked against the
policy at bind time (`validate_rollout`) rather than per step. That check compares the embodiment
**name** as well as the width, because two embodiments with the same `action_dim` pass every
numeric check while serving the wrong weights.

### 3.3 `WorkUnit.scenario`

`scenario` is your per-unit dict and mostly opaque to the core — but **the driver reads two keys
and they are required**:

| key         | read by                     | meaning                                             |
| ----------- | --------------------------- | --------------------------------------------------- |
| `task`      | the policy-side observation | task name / instruction handle                      |
| `max_steps` | the truncation budget       | the driver enforces it; your session must honour it |

Everything else is yours. (b1k additionally uses `mode`, `seed`, `num_envs`, `episodes_per_scene`
and others, all with defaults.)

```python
scenario = {"task": "open_drawer", "max_steps": 800, "num_envs": 1}
```

> These two keys living inside a dict the contract calls sim-specific is a known wart, and it is
> the one place a second engine is forced to speak the first one's dialect. If you are adding that
> second engine, expect them to be promoted into the declared contract.

---

## 4. Running it

### 4.1 Local, in-process — no Ray, no config tree, no campaign

The fastest proof a backend is wired. `run_unit` takes the four seams as plain objects:

```bash
python -m cosmos_framework.posttrain.embodied.sims.example \
    --sim my_robot_sim --task demo --lanes 1 --max-steps 8 --episodes 2
```

It prints the declared capabilities, runs a real unit through the real loop, and reports what came
back. If this works, your `Simulator`/`SimSession`/`EmbodimentMap` triple is correct. Everything
after this point is transport and placement, which is our side.

### 4.2 Isolated, behind the gateway — two processes, two venvs

Your simulator runs as its own process, in **your** interpreter, and the framework reaches it over
a unix socket plus a shared `/dev/shm` segment.

Start the simulator, owning the rendezvous (`--listen` makes the *simulator* the host, so it
survives a restart of the framework side):

```bash
export B1K_SIM_IPC_DIR=/dev/shm            # where the socket lives; must be short (see below)
python -u -m cosmos_framework.posttrain.roles.env.sim_server \
    --socket "$B1K_SIM_IPC_DIR/sim-cell0.sock" --listen
```

It will bind the socket and wait. **Your backend is not named here** — `sim_server` takes no
`--sim` flag. It announces itself, then the framework sends a `configure` frame carrying the
backend name, the embodiment and the sink, and only then is your `Simulator` constructed. So in
gateway mode the selection lives in the run config (`sim:`), and §4.1 is where you prove the
backend itself before any of this is involved.

Now the other half, in **our** interpreter, in a second shell. It connects to the socket you
just bound, configures your backend across it, and runs one unit with a zero-action policy:

```bash
python -u -m cosmos_framework.posttrain.roles.env.sim_gateway \
    --socket /dev/shm/sim-cell0.sock \
    --sim my_robot_sim --embodiment my_robot --action-dim 7 \
    --preload my_package.my_robot_sim --episodes 2
```

Substituting the shipped `example` backend for `my_robot_sim`, that pair prints:

```
[gateway] orphan sweep: 0 shared segment(s) reclaimed (armed check, not silence)
[gateway] child READY ... backend='example' segment=5875692B rendezvous='connect'
[gwdemo] attached: {... 'backend': 'example', 'embodiment': 'example', 'max_lanes': 1,
                    'camera_shapes': {'head': [64, 64]}, 'proprio_width': 7 ...}
[gateway] run_units done: crossings_verified=12 (cumulative 12)
[gwdemo] unit=4fb4460d8dba3bc6 wall=0.01s error=None
[gwdemo]   demo/i0  success   steps=4
[gwdemo]   demo/i1  failure   steps=8
```

and on the simulator side, the loop you already proved in §4.1 — plus the receipt that matters
most here:

```
[sim] framework isolation @session-end: ray absent in sys.modules after a full session
```

`tests/test_roles_env_gateway_two_process.py` runs exactly this pair on CPU, so the commands
above stay runnable rather than staying written down.

Three things about that command are worth more than the transcript:

- **`--preload` is passed through, never imported here.** It names modules for *your*
  interpreter. The framework half must be able to run on a machine where your engine does not
  exist, which is the same rule §2 puts on your module and the reason the split is worth having.
- **`--action-dim` is asked for rather than guessed.** This side needs exactly one number about
  your robot — the action width — and the only honest ways to get it are for your
  `EmbodimentMap` to be registered framework-side too, or for you to say it. A default here
  would produce a rollout of the wrong shape that your simulator accepts and the corpus records.
  Omit it with an unregistered embodiment and the command refuses, with that message.
- **`num_envs` is pinned at 1 and is not a flag.** Scene tiling was measured corrupting
  physical particle systems — V=2 scored 0/316 where V=1 scored 5/40 — and V>1 is a standing
  refusal, not a default. b1k declares `max_lanes=1`, so §5's bind check refuses a run that asks
  for more. If your engine genuinely tiles, raise your own ceiling in `capabilities()` first.
- **This is a demo driver, not the entrypoint.** Production builds the same `SimGatewayBody`
  from the run config and hands it a Ray policy handle instead of the zero policy; what the two
  share is every line between the handshake and the result, which is the part you are proving.

The framework side needs to know which interpreter you are and where to dial:

| variable              | meaning                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `SIM_PY`              | absolute path to **your** python — the gateway spawns/dials with it, never its own                            |
| `SIM_LD_LIBRARY_PATH` | your engine's libs. **Not inherited**: the policy venv's CUDA libraries make some engines segfault at startup |
| `B1K_SIM_IPC_DIR`     | directory holding the socket and shared segment                                                               |
| `B1K_SIM_SOCKET`      | the exact socket path, required when the simulator owns it                                                    |

**GPU assignment.** Under the default (`sim_attach: true`) the *launcher* assigns the card before
the framework exists, and your process reads its own `CUDA_VISIBLE_DEVICES` — the gateway passes
no card and refuses to, because a value there would be an intent it never acted on. Your process
reports its own device in the handshake and the framework **refuses the bind** if it disagrees
with the policy's card.

**One practical trap:** `AF_UNIX` paths are capped at 107 bytes and the failure is a bare `OSError`
naming neither the limit nor the path. Keep `B1K_SIM_IPC_DIR` short.

---

## 5. The minimum you must pass

Run the following mapped tests before publishing a simulator integration:

```bash
python3 -m pytest -q \
    cosmos_framework/posttrain/tests/test_import_discipline.py \
    cosmos_framework/posttrain/tests/test_embodied_example.py \
    cosmos_framework/posttrain/tests/test_roles_env_gateway_two_process.py
```

What they enforce, and why each one exists:

| test                                                                  | what it catches                                                                                               |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `test_the_backends_import_without_their_engines`                      | your module must import with the engine **absent** — the registry walks it on a machine that has no simulator |
| `test_the_sim_side_roots_import_without_ray`                          | installs a hook that raises on `import ray`, the way an interpreter without it would                          |
| `test_the_plugin_surface_never_reaches_outward`                       | the declared surface, walked from the directory, cannot reach outside itself                                  |
| `test_the_core_set_is_derived_from_the_directory_and_not_written_out` | your new file is covered without anyone remembering to add it                                                 |
| `test_the_roles_env_barrel_is_lazy`                                   | no re-export drags the training stack into the simulator venv                                                 |
| `test_the_documented_two_process_pair_actually_rendezvous`            | §4.2's two commands still rendezvous and still return episodes                                                |
| `test_the_framework_half_never_imports_the_backend`                   | `--preload` reaches the child's interpreter and not ours                                                      |

**Add your backend's own tests to that inventory.** Being present in the repo is not being
executed: the CI path filter decides whether the job *runs*, and the node-ID list decides *what*
it runs. A test file that is green locally and absent from the inventory runs nowhere while
reading as protection.

The bar for a new backend, concretely:

1. §4.1 runs clean end to end.
2. Import discipline passes with your engine uninstalled.
3. `capabilities()` is answered without a scene build.
4. `observe()` after `step()` returns the cached object.
5. Every method raises `SimProtocolError` after `close()`.
6. Your tests are in the required inventory, with `@pytest.mark.L0` and `@pytest.mark.CPU`.

---

## Where to look next

- `embodied/sims/example.py` — the whole contract, working, in one file.
- `docs/reference/embodied.md` — the package's own design notes and the import discipline.
- `docs/design/DESIGN.md` §1.3 — who owns which seam, and why the loop and sink stay on your side.
