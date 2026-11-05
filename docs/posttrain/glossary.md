# Glossary

The terms below are prerequisites for reading anything else. Each cites the code that defines it by
symbol rather than line number, so the reference stays valid as files change.

## Placement and process structure

**WorkerGroup (WG)** — `workers/worker_group.py::WorkerGroup`. The placement-agnostic proxy for
every process of one role. Calling `wg.method(...)` returns a `Handle`; whether that ran in this
process or fanned out across a Ray mesh is invisible to the caller. This is the abstraction that
lets one loop drive single-GPU and multinode alike.

**Execution replica (`R`)** — an independent working instance inside a WG. One rollout replica is
one independent inference engine. Configured through `grid.num_replicas` or `grid.shapes`.

**rank (`K`)** — the process lane inside one execution replica. For rollout, `K` equals the vLLM
tensor-parallel width.

**ActorGroup** — `control/placement/colocate.py::ActorGroup`. WorkerGroups that share one OS
process at every physical cell. This is the **only** truth about process sharing.

**DeviceGroup** — `control/placement/colocate.py::DeviceGroup`. ActorGroups mapped onto one CPU or
GPU placement domain. This is the **only** truth about device sharing: one GPU ActorGroup owns an
exclusive pool; several GPU ActorGroups in one DeviceGroup share cards by flat actor index;
different GPU DeviceGroups get disjoint pools.

**colocated / disaggregated** — `grid.gpu_placement`. Whether rollout and policy share one GPU
DeviceGroup or own disjoint pools. Disaggregated additionally requires a cross-device weight
transport (`rl.weight_sync_transport=nixl`).

## The naming rule that causes the most confusion

**policy** is the training role. **rollout** is the generation role. **actor** is reserved for Ray
actors and never means the policy.

**`dp_replicate` and `dp_shard`** are the model's own data-parallel factorization inside the
policy's `K` ranks, owned by FSDP2/DeviceMesh. They are **not** the placement axis `R`.

> Never abbreviate `dp_replicate` to "replica" and never derive it from placement `R`. Policy
> placement requires `POLICY_REPLICAS == 1` — `entrypoints/grpo.py` rejects anything else — and
> HSDP width is configured as `dp_replicate * dp_shard` inside `POLICY_RANKS` instead.
> `config/build.py::resolve_policy_parallelism` performs that resolution.

## Data flow

**Handle / CollectiveHandle** — `workers/worker_group.py`. A pending result. `Handle` wraps one
value or remote reference; `CollectiveHandle` holds one per `(replica, rank)` cell.
`resolve()` returns cell `(0,0)` only; **`resolve_all()` returns every cell** and is required
whenever the value is per-cell state, so one cell's failure cannot be silently dropped.

**Trajectory / Transition / ObsBundle** — `schema/`. The unified data contract. A `Transition` may
carry both `reward` (the RL signal) and `targets` (the supervised signal), which is what lets one
trajectory train a policy and a world model.

**needs_bulk** — a consumer-declared port property on a WorkerGroup. True means "my body computes
on the large tensors, so materialize my inputs"; false means the large payloads flow through as
references. Rollout and trainer set it; env, reward, and buffer do not.

**BulkRef / BulkStore** — `comm/bulk.py`, `comm/codec.py`. Large fields are split out of the
metadata and pulled peer-to-peer only where `needs_bulk` is set, so the controller never handles a
byte of tensor payload.

## Algorithm and batching

**Objective** — `algorithm/primitives/objective.py::Objective`. A callable plus a declared
`Reduction`. The trainer honors the reduction without inspecting the algorithm.

**Reduction** — `MEAN` divides by the gradient-accumulation count (DPO, NFT). `TOKEN_SUM`
normalizes a local numerator by the **data-parallel-global** valid-unit count, making the loss
invariant to DP world size (GRPO, OPD).

**batch_per_replica / n_mini / n_micro** — one collected batch is consumed by `n_mini` optimizer
steps, each split into `n_micro` forward/backward chunks. Mini-batching changes training dynamics
on purpose; micro-batching is a pure memory split that must not change the result. Full treatment:
[`howto/batching.md`](howto/batching.md).

**pi_old ahead pass** — at `n_mini > 1` the policy moves within one collected batch, so the clipped
surrogate needs a frozen reference distribution captured before the first update. Without it the
ratio is `exp(0)` on every step and the epsilon clip never engages.

**group_size** — rollouts sampled per prompt. GRPO's advantage is computed within a group, so a
group whose rewards all tie contributes exactly zero gradient signal.

## Validation — three different things

The word appears in three unrelated places. They are not variants of one feature:

| Term                      | What it is                                                                                  | Where                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Manifest preflight**    | GPU-free check that the wiring, fields, and transport are consistent, before any allocation | [`howto/preflight.md`](howto/preflight.md)                         |
| **Train-time validation** | Held-out evaluation during training, on an explicit held-out source                         | [`howto/train-time-validation.md`](howto/train-time-validation.md) |
| **Sim-eval**              | Closed-loop evaluation on simulator suites                                                  | not included in this release                                       |
