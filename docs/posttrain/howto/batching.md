# Sizing mini-batches and micro-batches

**Come here when a run runs out of memory, or when you want several optimizer steps per collected
batch.** Two independent knobs split one collected batch:

- **Micro-batching** (`rl.micro_batch_per_replica`) — split each optimizer step into several
  forward/backward chunks that accumulate gradients and step once. **A pure memory play: the result
  is meant to be identical to the un-chunked step.** This is what fixes an OOM.
- **Mini-batching** (`rl.mini_batch_per_replica`, giving `n_mini`) — take several optimizer steps
  per collected batch. This **changes training dynamics on purpose**, and at `n_mini > 1` it
  requires the frozen pi_old ahead pass described below.

Used by [`../guides/grpo.md`](../guides/grpo.md) and [`../guides/opd.md`](../guides/opd.md). Knob
definitions live in [`../reference/config.md`](../reference/config.md); what the resulting metrics
mean is in [`../reference/metrics.md`](../reference/metrics.md). If you arrived here from an OOM,
[`../troubleshooting.md`](../troubleshooting.md) lists the recovery order.

---

## The problem

The GRPO policy trains on a **collected batch** — the trajectories from one round of rollout — in a
single forward/backward pass. That was fine until a VLM run (`Qwen3-VL-2B`) ran out of memory in two
ways. At `group_size=4` it OOM'd **partway in** — step 4 of 200, when *that* step happened to draw
longer completions — so the memory is **data-dependent**: steps 1–3 fit, step 4 did not. And at
`group_size=8` it OOM'd **outright**, trying to allocate 7+ GiB for the log-softmax over 2048-token
completions. The workaround was to cap `max_tokens` at 512 — i.e. to train on shorter completions than
we wanted.

The batch itself is fine; the problem is consuming it *all at once*. Two independent ways to split it:

- **Mini-batching** — take several optimizer steps per collected batch instead of one. This changes
  the training dynamics (more, smaller updates) on purpose.
- **Micro-batching** — split each optimizer step into several forward/backward chunks that accumulate
  gradients, stepping once at the end. This is a pure memory play: the result is meant to be identical
  to the un-chunked step, just with a smaller activation footprint.

Micro-batching is what fixes the OOM. Mini-batching is a training knob that came along for free once
the machinery existed.

---

## The core idea

One collected batch, consumed as `n_mini` optimizer steps of `n_micro` chunks each:

```
collected batch
   ├─ mini-batch 0  →  optimizer step
   │     ├─ micro 0  →  forward/backward (accumulate)
   │     └─ micro 1  →  forward/backward (accumulate) → step()
   └─ mini-batch 1  →  optimizer step
         ├─ micro 0  →  forward/backward (accumulate)
         └─ micro 1  →  forward/backward (accumulate) → step()
```

The loop that drives this lives in the entrypoint and passes **indices only** — never data. Which
chunk each index refers to is resolved by the dispatch layer (see the *dispatch* design doc): the
driver says "mini 0, micro 1," and each policy rank materializes exactly that chunk of its own
data-parallel share. The trainer's method is still plain `train(batch)`.

The genuinely hard part is not the loop. It is **keeping the normalization correct** once the loss is
computed over chunks instead of the whole batch. That is most of this document.

---

## A worked example

A batch of 32 trajectories, with `mini_batch_per_replica = 16` and `micro_batch_per_replica = 8`. From
those the loop derives:

- `n_mini = 32 / 16 = 2` optimizer steps,
- `n_micro = 16 / 8 = 2` chunks per step.

So the driver runs a 2×2 loop:

```python
for mini in range(2):
    for micro in range(2):
        train(batch, mini=mini, micro=micro, n_mini=2, n_micro=2)
```

Four `train` calls. The trainer accumulates gradients across the two micro calls of each mini-batch
and steps on the second. `weight_sync`, the reward report, and checkpointing all fire once **per
collected batch** — not per optimizer step — because the *batch* is the loop unit, not the step. That
is why splitting the batch does not multiply the (expensive) weight sync.

If neither knob is set (`mini = micro = 0`), `n_mini = n_micro = 1`, and the four-call loop collapses
to the single `train(batch)` it was before — byte-for-byte the old behavior.

---

## The config surface

Three replica-global knobs describe one policy replica's step, independent of how many ranks that
replica has:

| field                     | default | meaning                                                                                                                                            |
| ------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch_per_replica`       | `32`    | trajectories the policy trains on, per replica, per loop step — the primary knob (not a `0`-sentinel; must be a positive multiple of `group_size`) |
| `mini_batch_per_replica`  | `0`     | trajectories per optimizer step; `0` = the whole batch (one step)                                                                                  |
| `micro_batch_per_replica` | `0`     | trajectories per forward/backward; `0` = one chunk (no accumulation)                                                                               |

The loop counts are *derived*, never configured: `n_mini = batch / mini`, `n_micro = mini / micro`.
Both are **DP-invariant** — `batch / mini` equals `batch_per_dp / mini_per_dp` — so every rank
computes the same counts from config alone and agrees without communicating. Config-load validation
requires exact divisibility at each level (a ragged final chunk would change the token-mean of one
step, the same class of bug as everything below).

---

## The hard part: deferred normalization

Here is the subtlety that makes micro-batching more than a `for` loop.

The GRPO loss is a **token mean**: sum the per-token loss over the batch, divide by the total number
of valid tokens `N`. If you naively split the batch into chunks and average the chunks' losses, you
get a **mean of means** — each chunk normalized by *its own* token count, not the batch's. On
unequal-length chunks that is simply the wrong number, and it is exactly why gradient accumulation was
*rejected* under this loss before this work.

The obvious fix — "compute `N` for the whole batch up front, then normalize each chunk by that" —
does not work either, and the reason is instructive:

- `N` is not `len(batch)` or anything you can read off metadata. It counts **masked completion
  positions after shaping** — the packer's masks, the shared-prefix ignore, empty-completion
  filtering, and for a VLM the image-token expansion. Computing it requires *shaping* every item.
- Shaping a VLM item needs its **image**. So computing `N` up front would mean materializing every
  image — which is precisely the bulk that micro-batching exists to avoid pulling.
- And `N` is a data-parallel **mean** (an all-reduce with `AVG`), so it composes with FSDP's
  mean-reduced gradients without a world-size factor. Any externally supplied `N` would have to be
  that same quantity, computed with that same collective — so passing one in buys nothing.

**So don't compute `N` ahead of time. Defer the division to the optimizer step.** The objective keeps
returning its normal per-chunk normalized loss, and additionally reports the divisor it used. The
trainer:

- for `n_micro == 1`, backpropagates the loss as-is — byte-for-byte today's path;
- for `n_micro > 1`, backpropagates each chunk's *un-normalized sum* (recovered by multiplying the
  chunk's loss by its divisor), accumulates the divisors, and on the last chunk **scales the gradient
  once** by `1 / max(Σ divisors, 1)`, *then* clips, *then* steps.

This is exact because gradients are linear: `Σ_chunks ∇(chunk sum) / N` equals `∇(Σ chunk sums / N)`.
And no extra collective is needed at step time — the per-chunk all-reduce the objective already does,
summed, *is* the batch divisor (the DP-mean is linear over the chunk index).

Three details that are easy to get wrong and are therefore load-bearing:

- **Order: scale, then clip, then step.** Clipping an un-normalized gradient clips at the wrong scale
  and silently changes the update.
- **`max(Σ, 1)`, not `Σ`.** An all-empty batch has `Σ = 0` (the objective's own guard returns a zero
  divisor there); an unguarded `1/0` would turn a legitimately zero gradient into NaN, which would
  surface as a spurious "non-finite gradient" skip rather than an obvious crash.
- **`n_micro == 1` deliberately keeps the old expression.** `(Σaᵢ)·s` and `Σ(aᵢ·s)` differ in the
  last bits, so routing existing single-chunk configs through the deferred path would break the
  bit-identity guarantee for a change that is supposed to be a no-op for them.

---

## `grad_accum`: three of them, one retired

"Retire `grad_accum`" was too blunt — the name lives at three layers with different owners:

| layer                                           | fate                                                                                                                                                                                    |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the shared `RunConfig.grad_accum` field         | **stays** — DPO genuinely uses it (its experiment configs set 32 and 4)                                                                                                                 |
| `rl.grad_accum` as a **GRPO** user knob         | **retired** — GRPO rejects a non-default value; the accumulation count is now derived as `n_micro`                                                                                      |
| the trainer's `grad_accum` constructor argument | **stays, unrenamed** — at that boundary it already means "how many `train` calls per optimizer step," which *is* `n_micro`. GRPO passes `n_micro` where it used to pass `rl.grad_accum` |

The old GRPO meaning of `grad_accum` — "that many *fresh rollouts* per step" — genuinely disappears
once the batch is drawn once and split, so retiring it as a GRPO knob is correct.

---

## The poisoned-window countdown

A subtle interaction with non-finite handling, worth calling out because no clean run ever hits it.

When a chunk's loss goes non-finite, the trainer discards the current accumulation window (zeroes
grads, resets the accumulator). At `n_micro == 1` that is harmless — every call is its own window. But
at `n_micro > 1` the driver keeps issuing the *remaining* chunks of the abandoned mini-batch; without
intervention they would accumulate into the *next* window, and the optimizer would step on a mixture
of two mini-batches under a divisor matching neither.

The fix is self-contained because the trainer knows its own window length: on a skip it records how
many chunks of this window remain, and discards exactly that many subsequent calls before starting a
fresh window. The accumulated divisor resets alongside. The seeded equivalence gate never exercises
this (it only fires on a non-finite loss mid-window), so it is easy to miss — which is exactly why it
gets its own note here and its own test.

---

## Metric roll-up: token-weighted, not averaged

The driver logs one set of `grpo_*` metrics per collected batch, but it collected them per `train`
call. Rolling them up needs **two different reductions**, and using one for both is wrong:

- `grpo_loss` is a token-mean already normalized by *each chunk's own* divisor, so a plain average
  over calls is a mean-of-means (the same error the gradient path defers to avoid). Weight each
  chunk's loss by that same divisor and it telescopes back to the batch loss exactly:
  `Σ(loss_i · denom_i) / Σ denom_i`.
- The count metrics (`grpo_n`, valid tokens, the divisor) are **counts** — they must *sum*. Dividing
  them by the call count would report a per-chunk average and understate the batch.

This one was caught by an acceptance run, not by a unit test: the plain-mean version and the
token-weighted version agree on *equal*-length chunks, so only an unequal-length batch reveals the
difference.

---

## Prerequisite: reproducibility

None of the "did I change the answer?" gates mean anything if the baseline is not reproducible, and
getting a bit-identical baseline turned out to require two fixes that had nothing to do with batching
per se:

- **Deterministic prompt ids.** Ids were random (`uuid4`), and the data-parallel partition sorts by
  `(group_id, id)` — so every run handed each rank a *different* subset of the batch, and two seeded
  runs diverged from step 2 onward. (The first diagnosis — "non-deterministic backward kernels" — was
  wrong; enabling full torch determinism made it *worse*.) Ids are now a deterministic, shard-scoped,
  monotonic function, so the sort is canonical across runs while staying unique across replicas.
- **A determinism mode.** `rl.deterministic` sets the cuDNN / `use_deterministic_algorithms` /
  flash-attention knobs, with one real trap baked into the helper: `CUBLAS_WORKSPACE_CONFIG` must
  come from the *launch* environment (cuBLAS reads it when it creates its handle, and Ray actors are
  separate processes), so the helper *raises* rather than warns if CUDA is already up.

With those, the acceptance gates hold: bit-identical at `n=1`, chunked-equivalence within `~5e-8` on a
real batch, and the OOM that started all this — 256 sequences at 2048 tokens — fails un-chunked and
completes micro-batched.

---

## Why it's built this way — the invariants

Each guards a failure that is **silent** (no crash, just a wrong model):

1. **Normalization is deferred, not recomputed.** The divisor is a post-shaping DP-mean; computing it
   up front would materialize the very bulk micro-batching avoids. Defer the division to the step.
2. **Token-weighted roll-up, not mean-of-means.** Both for the gradient (via the deferred divisor)
   and for the logged loss (via `Σ loss·denom / Σ denom`).
3. **`n_micro == 1` is byte-for-byte the old path.** So the bit-identity gate proves the refactor is
   a true no-op for existing configs.
4. **A non-finite chunk abandons its *whole* mini-batch,** not just itself — else the optimizer steps
   on a mixture of two mini-batches.
5. **Exact divisibility at every level.** A ragged final chunk changes one step's token-mean.
6. **The batch boundary is the unit.** `weight_sync` / report / save fire once per collected batch,
   so splitting it multiplies neither the transfer nor the loop count.

---

## π_old must be frozen across the window — the ahead pass

`n_mini > 1` does not just split a forward. It makes the batch drive **`n_mini` sequential optimizer
steps**, and that changes what the clipped surrogate means.

The objective's default π_old is `cur.detach()`. At `n_mini == 1` that is exact — nothing moves
between the forward and the update, the ratio is `exp(0) == 1`, and there is nothing to clip. At
`n_mini > 1` it is **wrong**: π_old is recomputed from the drifting policy on every mini-batch, so
the ratio returns to 1 every time and the epsilon-clip **never binds**. The run takes `n_mini`
unconstrained steps on one batch of experience, with no trust region at all.

This is not theoretical. Qwen2.5-3B / GSM8K, `n_mini=32`, identical recipe either side:

|                            | outcome                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| posttrain, no frozen π_old | reward → **0.0 at step 29/100**, 53 zero steps, policy emitted token soup |
| posttrain, `n_mini=1`      | trained cleanly to **0.940**                                              |
| cosmos-rl, 32 updates/loop | trained cleanly to **0.938**                                              |

**The fix** (`algorithm/grpo.py::OldLogprobCache`, driven by `entrypoints/grpo.py`): when
`GRPOConfig.needs_cached_old_logprobs` holds (`n_mini > 1` AND not `use_rollout_logprobs_for_loss`
— both of cosmos-rl's clauses; under rollout logprobs the objective never consults the cache) AND
the objective reports it can consume one (`policy.needs_old_logprobs()` — OPDConfig inherits the
config property but its direct objective has no ratio), walk the whole `(mini, micro)` product ONCE before training it — model in `eval()`,
`no_grad()` — recording π_old per chunk, then replay those values during training. Every optimizer
step in the window scores against the policy that actually collected the batch, so the ratio departs
from 1 and the clip binds. This is cosmos-rl's design: it gates the same pass on
`batch_size > per_optimize_batch_size` (`TrainerPhase.OLD_LOGP_COMPUTE`).

Two implementation details worth knowing before touching this:

- **Keyed by call order, not by `(mini, micro)`.** The dispatch ingress *consumes* the control
  kwargs, so the trainer body only ever sees `train(batch)`. Both passes walk the same product in
  the same order on the same rank, so chunk N pairs with chunk N. cosmos-rl keys it the same way
  (`old_per_token_logps[local_mini_step]`).
- **The bulk refcount doubles.** The batch is materialized twice per consumer cell instead of once,
  so `workers/grid.py` doubles `readers` under the same `n_mini > 1` condition. These two gates MUST
  stay in sync: a driver that skips the pass while grid still doubles **leaks** BulkStore entries;
  the reverse dies with `KeyError` on data already popped.

Cost: one extra no-grad forward over the batch per loop step (~⅓ of a step). `n_mini == 1` skips it
entirely and is byte-identical to before.

**The determinism probe is a one-shot screen, not a monitor.** On the first ahead chunk of a run
the trainer re-scores that chunk in `train()` mode and compares it with the `eval()` record at
unchanged weights. It gates on the mesh-wide fraction of tokens whose logprob moved past a
per-token bound, under the tolerance the model declares (`model/protocol.py::ForwardNoisePolicy`;
a model that declares none gets the strict gate: `0.05` nats on any token). It runs once and never
again for that run, so it catches a stochastic layer that fires on the first chunk, not one that
fires later. On a sparse-MoE policy the natural routing churns between two identical forwards, so
the probe fails at `n_mini > 1` unless the dispatch is pinned; see the routing-replay paragraph in
[`../reference/config.md`](../reference/config.md).

## Colocated offload: what the trainer holds while the rollout runs

On a colocated card (`grid.gpu_placement=colocated`, the default) the loop hands the GPU back and
forth every step, so the trainer's footprint during generation and the engine's footprint during
the update are not the whole story anymore:

```text
generate -> rollout.sleep -> trainer.onload -> train
         -> rollout.wake_up(weights) -> weight_sync
         -> trainer.offload -> rollout.wake_up -> next generate
```

- During generation the trainer's parameters, gradients and optimizer state are on the host. Under
  Ray the policy actor is paused as a whole through `torch_memory_saver` when that package is
  installed; otherwise, and under the local backend, the tensors are moved with `Module.to` and the
  optimizer state replaced by pinned host copies. What stays on the card is the process baseline
  (CUDA context, NCCL communicators) and the weight-sync staging buffer.
- During the update the engine's weights and KV cache are released. `rollout.sleep_level=1` (the
  default) backs weights up to host. Level 2 discards them instead and is for memory-tight GRPO
  runs whose mandatory full weight sync rewrites every engine parameter before the next draw.
  Weight sync wakes only the weights; the KV cache comes back after publication.
- Every scheduled, validation, and emergency checkpoint gets an exclusive trainer-memory window.
  A scheduled save at a boundary that does not validate runs inside the update's own window,
  before publication: the engine is still asleep and the trainer resident, so the step publishes
  once. Validation and emergency saves come after publication: rollout sleeps again before
  checkpoint extraction restores trainer state, and after a successful level-2 save the loop
  performs another complete policy-to-rollout sync before waking the KV cache; otherwise
  generation would resume from discarded or stale engine weights. A failed save leaves rollout
  asleep so run-level finalization can drain safely.
- Exceptional exits use the same cleanup rule. Rollout sleep is idempotent when fully asleep and
  re-releases weights after a weights-only wake, while an active training/setup error remains the
  primary exception if cleanup also fails.

Startup is unchanged for a fresh run: engine and trainer still build side by side, so
`rollout.gpu_memory_utilization` is still bounded by that peak. A resumed run also restores optimizer
state onto the card, so setup puts the new engine to sleep before the trainer loads its checkpoint;
the first hand-off wakes it and publishes the restored policy before any sampling. What the hand-off
buys is headroom inside the step: the KV cache and sampling batch during generation, and the
micro-batch during the update.

Measured on the 3B GSM8K recipe (2 GPUs, TP=2, `gpu_memory_utilization=0.25`): the card holds
about 7 GB when both sides are released, about 11 GB after the trainer offload with the engine's
weights awake, and about 52 GB at the steady-state peak (down from 65 GB with both resident).
Offload/onload cost 0.3 to 0.9 s per step; the exact numbers are in the `grpo_step` profile record
(`POSTTRAIN_PROFILE_LOOP=1`), which samples device memory at every phase boundary.

Disaggregated placement performs none of this: disjoint pools have nothing to share.

`rl.colocated_offload=false` turns the cycle off on a colocated card as well: the engine is built
without sleep mode and stays resident, the trainer is never offloaded, and the step is shorter by
the hand-off cost (about 6 s per step, 20 s down to 14 s, on a 1-node TP4/FSDP4 ~31B MoE arm at
`gpu_memory_utilization=0.15`). Both footprints then stay on the card for the whole step, so it is
only for runs where engine and trainer fit side by side; the bind-time memory checks and CUDA OOM
are what surface one that does not. The knob is inert under disaggregated placement.

## What's not done (follow-ups)

1. **Adaptive (token-budgeted) micro-batching.** The chunk count is fixed, chosen for the worst-case
   completion length. A fixed count cannot react when completions come back long — which is the exact
   shape of the step-4 OOM. Adapting it means a data-dependent `n_micro`, which needs either a
   self-terminating driver loop or moving the micro loop inside the trainer. Deferred by decision
   (fixed count first); this is where the design knowingly differs from cosmos-rl, whose
   `rearrange_mini_batches` both adapts to length and balances chunks by token cost.

2. **`mu_iterations > 1` (inner PPO epochs).** Re-using one collected batch across epochs is PPO
   work, only correct if `pi_old` stays frozen at the rollout policy. **That is no longer the
   blocker** — the ahead pass above provides exactly that tensor, and inner epochs would reuse it
   unchanged. What remains is the bookkeeping: mu>1 breaks "one loop step == one optimizer step",
   which resume/checkpoint/weight-sync key off (`n_mini` already generalized that, so this is a
   smaller job than it was). Still rejected at config load today.

   Note this entry used to imply the frozen-π_old requirement was specific to inner epochs. It was
   not: `n_mini > 1` reaches the identical "N unconstrained steps on one batch" state, was never
   guarded, and destroyed a real run before the ahead pass landed. Both doors, one requirement.

*(Both are tracked in the design spec's §3.7 and §6.)* <!-- rumdl-disable-line MD036 -->

---

## Reference

### The loop at a glance

```
collect batch (one rollout round) ─────────────────────────────────────────┐
for mini in range(n_mini):            #  n_mini = batch // mini_batch        │
    for micro in range(n_micro):      #  n_micro = mini_batch // micro_batch │  one collected
        train(batch, mini, micro, …)  #  indices only; dispatch picks the chunk │  batch
        # trainer: accumulate; on the last micro of a mini-batch:              │
        #   scale grad by 1/max(Σ denom, 1) → clip → step()                    │
weight_sync / reward report / save    #  ONCE, at the batch boundary ──────────┘
```

### Vocabulary

| term                       | meaning                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| **collected batch**        | the trajectories from one rollout round; `batch_per_replica` per policy replica          |
| **mini-batch**             | the slice consumed by one optimizer step (`n_mini` per collected batch)                  |
| **micro-batch / chunk**    | a forward/backward within a mini-batch; gradients accumulate (`n_micro` per step)        |
| **divisor / `N`**          | the DP-mean count of masked completion tokens the objective normalized by (`grpo_denom`) |
| **deferred normalization** | backprop the un-normalized chunk sum; divide once, at the optimizer step                 |
| **poisoned window**        | a mini-batch abandoned mid-accumulation by a non-finite chunk                            |

### Component → file

| concern                                                                 | file                                                                                     |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| the two driver loops; `n_mini`/`n_micro` wiring                         | `entrypoints/grpo.py`                                                                    |
| the three config fields, derived counts, validation                     | `config/rl.py` (`GRPOConfig`)                                                            |
| the objective's reported divisor (`grpo_denom`)                         | `algorithm/grpo.py`                                                                      |
| the `backward(loss, reduction, grad_accum, denom)` contract             | `algorithm/primitives/objective.py` (imported as `reduce_backward` in the trainer)       |
| deferred normalization + poisoned-window countdown in the trainer       | `roles/trainer/llm.py`, `roles/trainer/finite.py`                                        |
| the per-rank chunk selection (how a `(mini, micro)` index becomes data) | `roles/core/dispatch.py`, `workers/dispatch.py`, and the internal dispatch design record |
| deterministic ids; determinism mode                                     | `roles/env/dataset.py`, `utils/determinism.py`                                           |

### Relationship to the dispatch mechanism

This document covers the *consumption* side — how a collected batch becomes several normalized
optimizer steps. The *delivery* side — how a `(mini, micro)` index turns into exactly the tensors one
rank needs, without pulling the rest — is the **data dispatch mechanism** (`docs/design/data-dispatch.md`).
The two meet at one line: the driver's `train(batch, mini, micro, …)`, whose control kwargs the
dispatch layer consumes so the trainer body still sees a plain `train(batch)`.
