# Checkpointing & resume

One checkpoint system for every engine trainer and model backend. The split of
responsibilities (mirrors cosmos-rl's ModelWrapper/Checkpointer split):

- **`checkpoint/`** owns the layout and mechanics: `CheckpointManager` in `manager.py` owns format,
  logical component selection, primary store, initialization-artifact reader selection,
  cross-option policy, rank/async coordination, validation publication, archive-then-retention orchestration, and
  finalization; the storage/IO primitives perform saves,
  `worker.py` owns the rank-zero async slot, `io.py` handles components and resume paths, `provenance.py` and
  `external_copy.py` protect exact promotion. `retention.py` owns local selection and deletion;
  `archive/publisher.py` owns both full-checkpoint archives and the single `publish_artifact`
  operation used by trainer and offline callers. `archive/reader.py` verifies and fetches archives,
  and `archive/marker.py` owns local receipts and remote inventory declarations. Offline archive,
  merge and HF materialization commands live in `tools/`; the reusable HF converter lives in
  `conversion/hf.py` ([`checkpoint-artifacts.md`](../howto/checkpoint-artifacts.md)).
- **Engine trainers** own only STATE EXTRACTION — FSDP2 full-state gather for sharded native or HF
  policies, and plain `state_dict` for unsharded policies
  (`roles/trainer/{diffusion,llm}.py`). Both inherit `BaseTrainer`, which owns the single
  `CheckpointManager` instance, shared lifecycle properties, update clocks, and finalizer.

Config dataclasses validate schema shape, `CheckpointLifecycle` lowers those values and expands the
run-relative default prefix, and concrete store construction validates backend fields. Compatibility
rules across format, primary store, upload, and retention are normalized once by `CheckpointManager`.

A new engine/backend reuses this package unchanged, so checkpoints look identical across
the framework. The mechanism-level truth lives in those files' docstrings; this doc is the
user/ops view.

## Layout

```
<IMAGINAIRE_OUTPUT_ROOT>/<project>/<group>/<name>/<RUN_TS>/checkpoints/
  iter_000000100/
    model.pt      # canonical posttrain weights: net (+ maintained ema when enabled), directly
                  #   reloadable by posttrain (~1/6 of the bytes under AdamW full-param training)
    old.pt        # maintained NFT moving-old policy when it differs from policy;
                  #   omitted at refresh-aligned policy-derived checkpoints
    ref.pt        # ONLY for a moving reference (rl.ref.update_every > 0) — see below
    optim.pt      # optimizer state — the big one; prunable without losing weights.
    optim/        #   A DTensor-sharded LLM trainer writes it as per-rank DCP shards even under
                  #   checkpoint_format=consolidated: the gathered fp32 moments (~244 GB for a
                  #   30 B-parameter policy) do not fit rank 0's host next to the gathered model.
                  #   Resuming a sharded trainer from the older all-.pt layout (model.pt + optim.pt)
                  #   copies only each rank's own chunk out of the mmap-backed files
                  #   (checkpoint/consolidated.py), never the whole dict through the
                  #   full_state_dict setters: that path put world-size copies of the checkpoint
                  #   on one host (4 x 281 GB > 900 GB cgroup) and killed every resume attempt.
    trainer.pt    # update clocks / boundary invariant / scheduler / skip counters /
                  #   old storage mode + cadence + timing / per-rank RNG /
                  #   validation transaction — tiny
    data.pt       # optional per-rank preference cursor or WFM structural row clock — tiny
  latest_checkpoint.txt   # atomic pointer; updated only after ALL component files land

<IMAGINAIRE_OUTPUT_ROOT>/<project>/<group>/<name>/<RUN_TS>/checkpoint_artifacts/
  iter_000000100/
    model.pt | model/     # example model-only artifact; other partial plans use the same names
    .posttrain-checkpoint-artifact.json  # commit-last component/format manifest
```

`RUN_TS` is the run-attempt timestamp (`YYYYmmddHHMMSS`, upstream convention) minted at the
FIRST launch of an attempt and reused by every resumed window: the directory name IS the
wandb run id and the `$RUN_TS` the shared eval tooling takes. Re-running the same
experiment fresh (`rl.resume=off`) creates a SIBLING attempt dir — nothing is ever
overwritten. Resolution rules live in `checkpoint.io.resolve_run_ts` (rank-0
resolves, torchrun broadcasts; ray workers get it pinned via `rl.run_ts`). Legacy jobs
whose `<name>/checkpoints/` already exists keep the old flat layout.

The sibling `checkpoint_artifacts/` tree is deliberately outside the resumable `checkpoints/`
root. Partial saves never update `latest_checkpoint.txt`, are never selected by `resume=auto`, and
are not counted by training-checkpoint retention. For direct object-store primary storage the same
sibling layout is used remotely. For local primary plus `rl.upload`, a partial artifact is archived
to the remote sibling `checkpoint_artifacts/` prefix and its artifact manifest is uploaded last.

Synchronous PPO publishes one atomic multi-role boundary. Its `iter_*` directory contains
`policy/` and `critic/` sub-checkpoints plus a top-level `_SUCCESS` marker; the source cursor is
stored with the policy role. `resume=auto` sees the boundary only after both roles and the marker
commit. This layout currently requires the local primary store. The `iter_*` number is the PPO
loop's outer iteration, which is not the actor's `policy_version` when `rl.critic_warmup_iterations`
is set. The marker records the optimizer counter each role committed and restore checks every role
against its own value, because the roles are not required to share one: during critic warmup the
actor's counter stands still while the critic's advances. Markers written before that field are
still validated by the earlier cross-role agreement rule.

The marker carries one further field, `loop_state`: an opaque mapping of JSON scalars supplied by
the loop that committed the boundary, recorded verbatim and returned verbatim by
`committed_loop_state()`. The coordinator never interprets it; it exists so a loop can persist a
configured choice whose effect no role counter can express and compare it directly on resume. PPO
stores `{"critic_warmup_iterations": N}` there, because a never-published actor reports version 0
under every warmup at least as long as the boundary's iteration, making the value unrecoverable
from the roles alone. The field is additive and optional, so the marker `version` is unchanged and a
boundary written without it still restores; a mismatch between a recorded entry and the current
configuration is the loop's to reject.

On the 8B DPO experiment: `model.pt` ≈ 85G, `optim.pt` ≈ 26G (train_keys subset),
`trainer.pt` ≈ 4K. The old monolithic `state.pt` (net+ref+ema+optim in one file) was
≈ 140G and had to be downloaded whole for eval.

## Two formats (`rl.checkpoint_format`)

|                         | `consolidated` (default)                                                                                                                                                    | `dcp`                                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| on disk                 | one portable `.pt` per component                                                                                                                                            | `model/` `optim/` and optional `old/`/`ref/` DCP dirs (per-rank `.distcp` shards); `trainer.pt` stays a file                         |
| write path              | FULL state gathered to rank-0 (CPU-offloaded under FSDP2)                                                                                                                   | every rank writes its own shards in parallel — no gather, no host-RAM spike                                                          |
| async (`rl.save_async`) | yes — disk write on a background thread; the latest pointer advances only after a COMPLETE write, so a crash mid-write leaves `auto`-resume at the last complete checkpoint | model/optimizer DCP writes stay synchronous; only the small trainer/latest/publication tail may run in the rank-zero background slot |
| resume world size       | same world only                                                                                                                                                             | may reshard compatible state; model-owned padding can still require the original topology                                            |
| posttrain load/eval     | `model.pt` is directly consumable by posttrain                                                                                                                              | load via DCP, or consolidate first                                                                                                   |

These are TRAINING checkpoints, not Hugging Face artifacts. In particular, an HF-backed `model.pt`
does not contain the `config.json`, tokenizer/processor, or safetensors layout expected by
`save_pretrained` consumers. `HFModel.export()` is a separate conversion API; the reserved
`rl.export_dir`/`rl.export_every` fields do not currently trigger it. DCP can reshard ordinary FSDP
state, but an LLM parameter whose producer owns world-size-dependent padding must resume with the
same topology; rank-local RNG also restores exactly only at the same world size.

Async discipline: one in-flight write per trainer role (the next save joins the previous); the
entry joins every trainer/critic role through `CheckpointLifecycle.finalize_units`, so the join
happens in whichever process owns each writer (each Ray worker / this rank). Before
selecting the async branch, rank 0 proves the complete component snapshot is CPU-only and shares no
storage with live model, optimizer, scheduler, trainer, or data state, then broadcasts that one
decision to the complete policy world. An unproven snapshot writes synchronously.

## Logical component selection

Checkpoint selection uses three backend-neutral logical names:

| logical component | physical state                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `model`           | policy weights and any maintained EMA weights in `model.pt` or `model/`                                                   |
| `optimizer`       | optimizer tensors in `optim.pt` or `optim/`                                                                               |
| `training_state`  | trainer clocks/RNG/scheduler plus applicable data cursor and moving old/reference state (`trainer`, `data`, `old`, `ref`) |

`[model, optimizer, training_state]` is the full plan and is the only plan that creates an
exact-resume checkpoint. There are two independent request surfaces:

- `rl.checkpoint_components` controls regular cadence and normal-final saves. Its default is the
  full plan.
- `rl.validation.checkpoint_components` controls a validation-requested save. Its default is also
  the full plan, but it is inactive unless `rl.validation.save_checkpoint=true`; that switch
  defaults to `false`.

Any non-empty proper subset is accepted as an explicitly non-resumable artifact. The checkpoint
manager emits a `UserWarning` when that plan becomes active as the regular run plan or for a
validation save; an inactive validation plan does not warn. Empty selections, duplicate names,
non-string names, and names outside the three listed above fail during config validation. Partial
plans are canonicalized into the table order.

When regular saving and validation are due at the same model step, the controller unions their
component plans and performs one save. A full union uses the durable validation checkpoint
transaction. For a partial union that includes an independently due cadence/final save, the artifact
is committed before ephemeral validation starts, so validation failure cannot discard the due save.
Only a partial plan requested solely by validation evaluates first and commits its artifact after
success. With the defaults this means a cadence/final boundary remains a full checkpoint, while
validation alone evaluates without forcing a save. `save_every=0` disables regular cadence and
normal-final saves.

Emergency saves are different by default: preemption and timeout recovery request the full plan,
regardless of either configured selection (`rl.emergency_checkpoint_components=full`). A partial
artifact therefore cannot suppress the resumable checkpoint needed for requeue. A run whose full plan
cannot be written inside the scheduler's grace opts out with `rl.emergency_checkpoint_components=run`:
the emergency save then writes the run's `checkpoint_components` plus `model` (an emergency save
always carries the weights; for a model-only plan the two coincide) through the cadence path — the
location and commit marker of the save the loop would have written at its next `save_every` boundary.
It is skipped only when every component of that plan is already on disk for the step: a step the
cadence already saved is not saved twice, but a validation-only artifact of the step (an off-cadence
validation with `save_checkpoint`) does not count as "already checkpointed" unless it carries every
component of that plan (the run's `checkpoint_components` plus `model`). Because a committed artifact
is never re-committed, such an artifact missing any of those components would leave only the full
plan to write, so the config refuses `run` together with a saving validation plan that does not cover
every component in `checkpoint_components` plus `model` (`RunConfig.__post_init__`; a run plan
`[model, training_state]` with a validation plan `[model]` is rejected, not only a plan without
`model`): the validation artifact of a step always stands in for the emergency save, and the
coordinator's escalation to the full plan is a logged fallback no accepted configuration reaches. The trade is the one the manager's `UserWarning` states for the
cadence plan: a partial emergency artifact is not exact-resume capable and `rl.resume=auto` will not
adopt it; continuing from it is a warm start that re-initialises the optimizer. Every single-role loop
(GRPO, OPD, DPO, NFT, SFT) honours the knob through `CheckpointCoordinator.save_emergency`; the durable
validation transaction (validation pending at the boundary) always writes the full plan because its resume
needs exact state; the PPO and joint-PPO loops always write the full plan at every boundary, and their config
validation refuses `run` instead of ignoring it (`config/rl.py::require_full_emergency_plan`).

Partial saves use `<run>/checkpoint_artifacts/iter_*/` and commit
`.posttrain-checkpoint-artifact.json` last. They do not write the full-checkpoint control manifest,
do not advance `latest_checkpoint.txt`, and are rejected as `rl.resume` sources. The manifest makes
the selected format and logical/physical component set explicit. Because a partial artifact has no
content identity proving that replayed training reproduced the same tensors, any second save to its
committed step fails before entering a writer; its already-written archive may be retried separately.
The focused contract is
the internal component-selection design record.

## Primary checkpoint store versus archive upload

`rl.checkpoint_store` and `rl.upload` answer different questions:

- `checkpoint_store` selects where the **primary, resumable** checkpoint lives;
- `upload` optionally archives a completed **local-primary** full checkpoint, or the selected
  payload of a partial artifact, to another store.

| mode                   | `checkpoint_store.backend` | `upload.enabled` | durable-copy behavior                                                 |
| ---------------------- | -------------------------- | ---------------- | --------------------------------------------------------------------- |
| local only (default)   | `local`                    | false            | save and resume locally                                               |
| local + S3/GCS archive | `local`                    | true             | local primary, then ordinary best-effort archive upload               |
| direct S3/GCS DCP      | `object_store`             | false            | DCP shards are written and resumed directly; no local checkpoint copy |

The two remote-looking modes are intentionally not aliases. Local+archive retains a complete local
copy when ordinary upload fails. Direct-primary has no such fallback, so every component, manifest,
latest-pointer, and finalization failure is fatal.

“No local checkpoint copy” applies to the `iter_*` checkpoint tree. The ordinary run directory,
resolved config, logs, and `run_ts` selection still use `IMAGINAIRE_OUTPUT_ROOT`; unchanged-command
`resume=auto` uses that run-attempt identity to select the matching remote checkpoint root.

Local primary with archive upload:

```yaml
rl:
  checkpoint_format: dcp       # consolidated is also supported in this mode
  checkpoint_store:
    backend: local
  upload:
    enabled: true
    bucket: <checkpoint-experiments-bucket>
    credentials: credentials/<credential>.secret
    prefix: ""                 # mirror project/group/name/run_ts/checkpoints
```

Direct object-store primary:

```yaml
rl:
  checkpoint_format: dcp
  checkpoint_store:
    backend: object_store
    bucket: <checkpoint-experiments-bucket>
    credentials: credentials/<credential>.secret
    prefix: ""                 # same upstream-compatible mirrored layout
  upload:
    enabled: false
```

Equivalent direct-mode dotlist:

```bash
rl.checkpoint_format=dcp \
rl.checkpoint_store.backend=object_store \
rl.checkpoint_store.bucket=<checkpoint-experiments-bucket> \
rl.checkpoint_store.credentials=credentials/<credential>.secret
```

The selection rule is literal: setting `upload.enabled=true` never changes where training writes
its primary checkpoint. To avoid a local `iter_*` copy, users must select
`checkpoint_store.backend=object_store`; in that mode `upload.enabled` must remain false.

`checkpoint_store.prefix` names the checkpoint **root**, not one iteration. For example,
`prefix=cosmos3_vfm/posttrain_grpo/my_run/20260901120000/checkpoints` produces
`.../checkpoints/iter_000000100/{model,optim,...}` and
`.../checkpoints/latest_checkpoint.txt`. An empty prefix derives that root from the same
`project/group/name/run_ts/checkpoints` layout used locally.

Direct mode rejects `checkpoint_format=consolidated`, `upload.enabled=true`, and explicit non-zero
`keep_optim_last`/`keep_model_last`. The default `keep_optim_last=null` resolves to `0` for a direct
primary because remote retention is disabled. Consolidated direct streaming, secondary archive from
a remote primary, and remote retention are separate features rather than silent no-ops.

`rl.save_async` does not put direct DCP bulk writes in the background. The direct save call returns
only after collective shard writers have finalized their per-component DCP metadata, the posttrain
control generation and pointer are durable, and (when requested) the fixed latest pointer has been
written. Normal and final teardown therefore do not leave an object-store transaction running after
the trainer exits.

S3 and GCS use the same `s3://` URI and S3-compatible API. The endpoint in the credential selects the
provider. Direct mode needs the same object permissions as DCP: the writer finalizes
`.metadata.tmp`, and later checkpoints replace `latest_checkpoint.txt`. Small trainer/data objects
and their pointer use EasyIO with the same configured backend.

The remote bulk layout matches upstream (`iter_*/model`, `iter_*/optim`, and a root
`latest_checkpoint.txt`). Posttrain adds only the control state that its trainers need:

```text
iter_000000100/
  model/                                  # DCP shards + .metadata
  optim/                                  # DCP shards + .metadata
  .posttrain/generations/<id>/trainer.pt  # append-only small state
  .posttrain/generations/<id>/data.pt     # optional append-only source state
  .posttrain-manifest.json                # current bulk names + control descriptors
latest_checkpoint.txt
```

The control manifest is deliberately small. It names the DCP components and active control
generation, but does not duplicate DCP's shard inventory or bind the checkpoint to one endpoint,
bucket, or prefix. Model/optimizer integrity is checked by the DCP reader from each component's
`.metadata`; trainer/data bytes are checked against their manifest descriptors.

Save order is DCP components, append-only control generation, control manifest, then
`latest_checkpoint.txt`. A failure before the manifest may leave a partial `iter_*` prefix, as it
does in upstream pre-training, but it cannot advance auto-resume to that prefix. The framework neither
cleans partial remote iterations nor claims a prefix with owner markers.

The first valid manifest makes that `iter_*` an immutable full-save commit. A same-iteration retry
validates the bulk/control component plan and referenced metadata, then only completes or confirms
`latest_checkpoint.txt`; it never re-enters DCP writers beneath a visible manifest. This covers an
ambiguous latest PUT that succeeded server-side but returned an error. Plan drift fails closed, and
later validation/control changes use append-only successor generations rather than another full save.

Exactly one live trainer/controller may write a configured direct-primary root. Users must choose a
run-specific prefix and must not point two jobs at the same one; posttrain does not provide a lease
or protect unrelated data already present there. Explicit resume of a half-written iteration fails
when its control manifest or a requested DCP component cannot be read.

A native DCP checkpoint has compatible model/optimizer paths but lacks posttrain trainer/control
state, so it remains a model-initialization source rather than an exact posttrain continuation. A
complete copy of a posttrain direct checkpoint, including its hidden control objects, is an exact
resume source at the destination because the control manifest contains only relative paths.

## What is (and isn't) saved

The list below describes the physical expansion of the full logical component plan. A partial
artifact extracts only the families selected in [Logical component selection](#logical-component-selection).

- **`net` (+ a maintained `net_ema` when EMA updates are enabled)** → `model.pt`. A model-provided
  EMA with trainer updates disabled is neither initialized from `rl.init_from` nor saved/restored;
  rank 0 warns instead of publishing a plausible but permanently stale EMA artifact. Maintained EMA
  is an explicit optional `EMAFlowMatchingPolicy` capability rather than an inferred part of every
  flow policy. It advances only after the applied optimizer update and any configured scheduler update, before the
  optimizer-step commit, using OmniMoT's zero-based pre-commit update index (`0`, then `1`, ...).
  The adapter owns the complete persisted EMA state, including persistent buffers that its EMA worker
  does not update. OmniMoT has `ema_beta(0) == 0`, so the first applied update fully replaces the
  initialization copy with the post-step policy; `rl.init_from` does not make EMA a frozen base copy.
  Its enabled/disabled mode is checkpointed and must match on exact resume; a legacy checkpoint
  cannot resume with EMA enabled because it cannot prove that EMA was maintained.
- **Maintained moving-old policy (`rl.old.update_every > 0`)** → `old.pt` / `old/`
  when it differs from policy. DiffusionNFT refreshes it only after a committed
  optimizer update at the configured cadence. A refresh-aligned checkpoint omits
  the redundant tensors and records a versioned `derived_from_policy` marker in
  `trainer.pt`; exact resume loads policy and then reconstructs old through the
  adapter's validated refresh operation. Off-cadence and emergency checkpoints
  store the full old component. The storage marker, cadence, step, and post-commit
  timing must agree; malformed, contradictory, or off-cadence derived state fails
  before continuation. `rl.init_from` remains model-only initialization: it copies
  the source policy into policy, moving old, frozen reference, and maintained EMA
  without restoring optimizer or clocks.
- **Frozen reference (`rl.ref.update_every = 0`) is NOT saved.** Setup rebuilds the ref
  anchor *before* the resume load on every start: `build_ref` copies the freshly-built
  net, and `rl.init_from` (when set) then loads the base weights into net, old, ref, and maintained EMA —
  **including on resumed windows** (resume does not skip `init_from`; skipping it leaves
  the ref at the raw experiment init and silently corrupts the objective — the 2026-07-06
  relaunch bug). The resume load never touches a frozen ref, so the anchor survives
  exactly. Exact resume also verifies the materialized reference tensors, module schema, load
  mapping, and adapter-declared forward semantics, even when validation is disabled. Copying
  net→ref on resume instead would corrupt the anchor: net has moved. A **moving** reference
  (`update_every > 0`) IS
  training state and is saved to `ref.pt` / `ref/`. This mode requires a policy
  adapter implementing `refresh_ref()`. Cosmos3 refreshes the complete persistent module state
  (parameters plus persistent buffers), excludes runtime-only nonpersistent buffers, and validates
  the full inventory and distributed layout before mutating the reference. The currently shipped
  Cosmos3 DPO recipes use a frozen reference (`update_every = 0`).
- **Cosmos3 reference semantics identity v2** binds whether the post-training adapter uses native
  RMSNorm execution or its recognized-module fp32 RMSNorm contract, plus the exact recognized
  module inventory. An older exact-resume checkpoint carrying the v1 identity fails closed rather
  than continuing under changed policy/reference numerics. Use that artifact as `rl.init_from` for
  a new tuning run when model-only migration is intended; `rl.init_from` does not claim continuation
  of optimizer, reference, or validation state.
- **Per-rank RNG streams** (python/numpy/torch/cuda) → `trainer.pt`; restored exactly on a
  same-world resume. Missing legacy state and topology changes are skipped; malformed
  rank inventories or payloads fail rather than being mistaken for a valid legacy checkpoint.
- **Update-control state** → `trainer.pt`: applied `optimizer_step`/iteration,
  `attempt_id`, the quiescent accumulation-boundary marker, and total/consecutive
  non-finite rejection counts. Gradients are not checkpointed, so save and resume both
  require `accum = 0`; an open accumulation window fails instead of claiming exact
  continuation. The clocks are restored together so telemetry and fail-fast
  diagnostics do not silently restart. Legacy checkpoints default `attempt_id` to the
  restored optimizer step and both skip counters to zero. Conflicting clocks,
  malformed counters, or an incomplete attempt fail rather than being normalized.
- **Never-stepped optimizer marker** (`optimizer_pristine`) → `trainer.pt`, present only while the
  optimizer still holds no per-parameter state. It exists because a checkpoint cannot otherwise
  record that condition: under `checkpoint_format="dcp"` (and the sharded consolidated path), state
  extraction goes through `get_optimizer_state_dict`, which MATERIALIZES an untouched optimizer by
  running a zero-gradient `step()` at zero learning rate. Parameters do not move, so neither the
  model state nor `optimizer_step` records anything — but a stateful optimizer's own step counter
  advances, and AdamW's bias correction reads it, so the first real update would be corrected as
  though it were the second. Save rolls that initialization back and writes an empty optimizer
  state. Absence means "not recorded" and preserves the pre-marker behavior, so existing
  checkpoints resume unchanged. This is what makes PPO's critic-only actor warmup
  (`rl.critic_warmup_iterations`) independent of `rl.save_every`.

  "Pristine" is decided over EVERY place a step leaves a trace, not just `optimizer.state`.
  `utils/fused_adam.py::FusedAdam` — which `config_ppo.py::_validate_optimizer` accepts — keeps one
  shared counter on the param group and builds `param_groups_master` lazily inside `step()`, so a
  group of frozen parameters trains with an empty per-parameter dictionary. Reading only that
  dictionary would both mark a stepped optimizer pristine and leave the probe's group counter in
  place to accumulate across saves. The rollback removes group keys the probe created and restores
  the master-weight attribute, and the written state is stripped of them too.

  Restore does NOT simply skip the optimizer. An empty per-parameter state does not mean there is
  nothing to restore: `param_groups` still carries the learning rate, betas and weight decay, and
  `lr_scheduler.load_state_dict` restores the scheduler's bookkeeping WITHOUT writing the learning
  rate back into the optimizer. A pristine restore therefore resets the optimizer to never-stepped
  and applies the saved group options, without materializing moments — otherwise a resume into a
  live trainer would keep whatever hyperparameters that trainer happened to hold, and train
  differently from a fresh process loading the same checkpoint.

  Order matters on that path. The reset runs BEFORE the DCP load target is shaped, because shaping
  it runs the same materializing extraction and DCP's planner is strict: a target field the
  checkpoint intentionally omits — the probe's group `step` — fails the load outright, before any
  later cleanup could remove it. The target is sanitized by the same rule the save applies, and the
  rollback covers the failure path so a rejected load does not leave the optimizer initialized.
- **Scheduler state and construction identity** → `trainer.pt`. The pinned WFM schedule records a
  versioned identity containing its warmup/cycle factors and post-update step timing separately from
  PyTorch's mutable scheduler state. Exact resume validates that identity before loading model,
  optimizer, or scheduler state; a changed schedule, unknown identity version, or legacy checkpoint
  without the identity fails closed. Use `rl.init_from` when intentionally starting a new run under
  different scheduler semantics. Legacy schedulers that do not opt into an identity retain their
  previous state-only resume behavior.
- **Model-validation identity and transaction state** → `trainer.pt`: enabled/source/config
  identity and evaluation/publication state. Exact resume rejects a changed identity.
- **Moving-reference timing** → `trainer.pt` when present. The current contract refreshes
  only after an applied update, so a rejected accumulation window cannot mutate the
  reference. A legacy checkpoint without the timing marker is converted at a due
  refresh boundary; an unknown timing marker fails loudly.
- **Moving-old timing and cadence** → `trainer.pt`. Unlike the bounded legacy conversion
  for an older moving reference, the first runnable NFT contract has no accepted
  marker-less exact-resume format: use `rl.init_from` for model-only initialization.
- **Source cursor** → optional `data.pt`; preference sources use the exact/non-exact
  policy below. Online WFM stores committed rows, their optimizer-step/accumulation
  geometry, and the resolved source fingerprint. This is a structural streaming-loader
  boundary, not a byte-exact WebDataset-worker snapshot. Iterable
  or out-of-tree sources without a stable cursor remain usable in `auto`/`off` mode
  without making an exact-resume claim.
- **Online WFM source identity** additionally persists the rollout seed contract:
  base seed, source-row versus optimizer-step clock, stride, and global group size.
  T2V uses the source-row clock; the qualified T2I recipe uses the optimizer-step
  clock, with its prompt SHA/order/geometry in the source fingerprint. Exact online-NFT
  resume rejects a version-1 state or any changed seed/source contract before moving
  the cursor; `rl.init_from` remains model-only and has no source-continuation claim.

The T2I `rl.init_from` copy policy is deliberately narrower than resume. It reads only
source `net`, requires every configured source-only sound key to exist and be absent
from the target, and permits target-only LoRA only as a complete cold A/B inventory
when the source contains no LoRA. Any unrelated key or shape difference rejects before
policy, old, reference, or maintained-EMA mutation. Exact consolidated and DCP resume
never apply those exceptions and do not migrate prototype parity checkpoints.

## Update outcomes and clocks

Both `DiffusionTrainer` and `LLMTrainer` return one explicit outcome for each
microbatch. The consensus covers the sole policy execution replica's complete K-rank
`torch.distributed` model world, including both `dp_replicate` and `dp_shard`. Multiple policy
execution replicas are rejected, so there is no second framework-level gradient/update consensus
axis. A disagreement within the model world fails before optimizer mutation:

- `accumulating`: the optimizer attempt remains open; no optimizer clock advances;
- `applied`: the full accumulation window was finite, the optimizer and configured
  scheduler stepped, and `optimizer_step` advanced;
- `skipped`: rank consensus found a non-finite loss or gradient, the entire
  accumulation window was discarded and gradients were cleared. Within durable
  optimizer-control state, `attempt_id` and total/consecutive skip counters advance
  while the optimizer, scheduler, callback train-step, and weight version do not.

An upstream online-NFT transaction failure is not a numerical `skipped` outcome. If
it interrupts an open diffusion accumulation window, the controller clears gradients
and rewinds exactly one unpublished attempt, including a failure after the attempt
opened but before the first backward pass. A quiescent or repeated abort is a no-op,
and retry reopens the same attempt ID; no non-finite skip counter advances. Incomplete
windows remain uncheckpointable, so this recovery changes no persisted schema.

A mixed local gradient-device inventory is not a numerical outcome: the same status
collective makes every rank raise a placement error, rather than reporting
`nonfinite_gradient` or allowing one rank to exit alone.

`optim/nonfinite_loss` and `optim/nonfinite_gradient` are dense binary event
indicators: healthy steps emit zero and a rejected step sets its matching reason to
one. `optim/nonfinite_skips_consecutive` belongs to the reusable trainer contract and
can exceed one for callers that continue after rejection; the shipped fail-fast DPO
and GRPO entries terminate on the first rejection and therefore externally observe
only zero or one. In `trainer.pt`, `iter` is retained as the legacy alias of the
explicit `optimizer_step` clock, and restore requires them to agree when both are
present.

The DPO, GRPO, PPO, OPD, and NFT entries are deliberately fail-fast on the first rejected update.
They report the reason and clocks, then terminate instead of starting another accumulation attempt
inside the same loop step. Online entries fail before trainer-to-rollout weight publication or
checkpoint save. The trainer-level outcome remains explicit so
a future bounded-continuation policy can be added without changing optimizer
atomicity.
The DPO objective converges rank-local encode, conditioning, and structural-validation
failures before any rank enters its first policy/reference forward. Exceptions raised
inside a model's collective forward remain fatal and are not converted into a
recoverable outcome; only expected numerical non-finiteness uses the `skipped` path.

Fail-fast termination does not write a new checkpoint. Restart therefore restores the
last committed policy checkpoint. For DPO, it also restores the matching preference
cursor and replays source items consumed after that boundary. A future in-process
continuation policy must explicitly choose whether rejected items are consumed or
rewound; the current code makes no exactly-once claim for such a policy.

Online-WFM source continuation has a narrower transactional guarantee. Within one
live process, the source ID is reserved before rank-0 fetch, and a loader row is cached
immediately on acquisition, before validation or T2V conversion. Preparation failure
retains the raw row; successful preparation replaces it with the typed recursive
snapshot. `abort()` leaves either transaction pending, so retry reuses exactly that
row and ID. Its versioned conditioning-template identity combines that ID with the
source fingerprint and conversion-key contract; it does not hash tensor payloads.
Successful microbatches may be staged, but only a complete
`commit_window()` advances the durable optimizer-window boundary. The online
control loop must call it only after an applied update. This source layer's quiescent
`state_dict()` contains committed rows, optimizer-window geometry, and a source fingerprint
when the loader supplies one. Before each distributed fetch, every source rank proves the
same committed/staged/window/source-identity/pending transaction signature. The entry
publishes this state only at a quiescent applied-update boundary. The contract guarantees
exact same-process source row/ID retry, not exact rollout or reward replay, and restart
reinitializes the streaming loader at a structural boundary rather than restoring an exact
WebDataset-worker snapshot. Controlled numerical comparison still requires frozen source
artifacts.

## Resume (`rl.resume`)

| value                                                | behavior                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto` (**default** — upstream/cosmos-rl convention) | latest under this job's out dir; fresh start if none. An UNCHANGED command survives preemption/requeue with zero config. Every resume logs `[dpo] resumed from ... (iter N)`. When the adopted attempt dir holds committed `checkpoint_artifacts/iter_*` but no resumable checkpoint (a model-only plan), or only a full checkpoint older than the newest committed artifact, the start is refused instead of restarting into a save collision — see [checkpoint-artifacts.md](../howto/checkpoint-artifacts.md).                |
| `off` (or `""`)                                      | force a fresh run — mints a NEW `<RUN_TS>` sibling dir; older attempts are never overwritten (legacy flat-layout jobs excepted).                                                                                                                                                                                                                                                                                                                                                                                                 |
| `<iter dir \| legacy .pt>`                           | resume that checkpoint; MUST exist (fails loud — never silently trains from scratch). Direct-primary mode also accepts an explicit `s3://.../iter_*` carrying posttrain control state. When the adopted attempt dir holds committed `checkpoint_artifacts/iter_*`, the checkpoint must be at or beyond the newest one: its iteration is the `iter_*` segment or, for a legacy path without one, its validated trainer state (`optimizer_step`/`iter`); behind the artifact or with no derivable iteration the resume is refused. |

Load runs inside each model-owning process at the end of setup. DPO resumes directly from
`policy.iteration()`. GRPO checkpoints only at collected-batch boundaries and resumes batch
`policy.iteration() // n_mini`; its `iter_*`, validation, and report steps remain optimizer/model
steps while `save_every` remains a collected-batch cadence. Reaching `rl.max_steps` exits cleanly.
The local format (`pt` file / component dir / DCP dir) is auto-detected. Direct-primary resume reads
the small control manifest, then opens requested DCP components in place through the DCP reader.
A complete direct-primary checkpoint tree copied to another object-store prefix can resume from its
explicit `s3://` location because its control descriptors are relative.

### Pending model validation

Validation does not force a checkpoint by default. With `validation.save_checkpoint=false` and no
regular save due at the same step, the held-out pass is ephemeral: it writes metrics but no durable
validation journal or checkpoint. This applies to initial, periodic, final, and zero-step-final
validation. A failed process therefore restarts from the latest regular full checkpoint rather than
from the just-validated model step.

Setting `validation.save_checkpoint=true` activates its component plan. The default full plan saves
a complete primary `iter_k` marked evaluation-pending before reading held-out data. That checkpoint
advances auto-resume but is not archived, retained, or announced until validation succeeds. Both
engine families then move successful evaluation to publication-pending and clear it only after the
shared publication transaction completes. With configured upload, publication is strict, so resume
can retry it without rerunning the completed objective or rollout/reward validation. A pure
fresh-run before-training pass with remaining training steps is always checkpoint-free, even when
this switch is enabled.

A partial plan requested solely by validation evaluates first and then writes a non-resumable
sibling artifact; it carries no durable pending-validation recovery claim. If a regular partial save
is due at the same step, the regular and validation plans are unioned, but their one artifact is
committed before ephemeral validation. The regular save therefore survives validation failure. A
full union gets the durable behavior above. A termination request observed before validation always
writes a full pending checkpoint, irrespective of configured plans. For a pure fresh-run
before-training pass with remaining training steps, termination exits without saving so the next
fresh start retries it.

A local external full pending checkpoint is staged and promoted without overwrite. A copied direct
full checkpoint with pending validation can continue publication in place only when
`checkpoint_store.prefix` selects the copied root; otherwise the coordinator refuses to mutate an
external source.
Provenance mismatch or unsupported local atomic promotion still fails closed. See
[`model_validation.md`](../howto/train-time-validation.md#checkpoint-resume-and-failure).

### Preference-data cursor (`rl.data.cursor_resume`)

The DPO source snapshot is taken at the same quiescent step boundary as the policy
checkpoint. It stores only the non-derivable cursor (`epoch`, rank-shard `position`,
source-local RNG) plus semantic identity guards (source fingerprint, seed, sharding,
world size, and preprocessing settings). Deterministic order, next index, and emitted
count are recomputed on restore.

| value                | behavior                                                                                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `auto` (**default**) | Restore the exact next preference item when the source and world size match. A legacy/missing cursor, unsupported source, or world-size change logs a non-exact source restart while allowing model/trainer resume to continue. A malformed cursor or genuine source fingerprint mismatch still fails. |
| `strict`             | Require a supported source cursor at save and a compatible cursor at restore. Missing state or any topology/source incompatibility fails before policy construction.                                                                                                                                   |
| `off`                | Omit and ignore source state explicitly. Model/trainer resume may continue, but source continuity is not exact.                                                                                                                                                                                        |

The source fingerprint hashes the ordered, filtered JSONL rows and cursor-relevant
preprocessing configuration. Relative media paths resolve against the JSONL directory,
not the launch CWD. Filesystem mtimes and per-media `stat` calls are deliberately absent:
the fingerprint is a semantic manifest identity, not an implicit bytewise scan. If rows
carry immutable media or manifest digests, those fields are covered automatically.
Loose-media bytes are therefore not independently verified unless the rows carry such
digests. Tar-backed sources additionally fingerprint one normalized JSONL-relative tar
path and file size per distinct archive because their row offsets depend on the archive
byte layout. Omitted, empty, absolute, and equivalent relative `tar_path` spellings
normalize to the effective archive's JSONL-relative identity.

Model and source topology are separate contracts. DCP may reshard compatible model/optimizer state
to a different world size; an `auto` resume then restarts only the rank-sharded source.
Use `strict` when the run must preserve exact next-example continuity.

### GRPO prompt-source continuation

GRPO restores model, optimizer, trainer update-control state, and trainer-rank
framework RNG. It does not restore the JSONL cursor, source-local shuffle RNG,
rollout-actor RNG, replay contents, or next group identity. A resumed GRPO run
therefore continues optimizer step numbering but does **not** claim exact next-prompt
or online trajectory continuation. Do not infer the DPO preference-cursor guarantee
for GRPO.

## Preemption / timeout (emergency save)

See `control/termination.py` for the full Slurm signal timeline. Key facts:

- Slurm sends SIGUSR1 to the **batch shell** — it does NOT reach pyxis/enroot containers.
  The proven mechanism is a **sentinel file**: the batch script's trap writes
  `$SLURM_LOG_DIR/SIGUSR1_RECEIVED` on the shared FS; the loop polls it at each step
  boundary. Direct SIGUSR1/SIGTERM are also trapped (flag-only) for non-container runs.
- There are ≥ 300 s of grace (`--signal=B:SIGUSR1@600` for timeout, `GraceTime` for
  preemption) — enough for one save at a step boundary — PROVIDED the job asked for the user
  signal: on a preempting QOS a job without `--signal=` hears SIGTERM first (see "Preemption on a
  QOS-preempting cluster" below).
- **Rank consistency**: under torchrun every rank runs the loop and the emergency save is
  a collective — `TerminationWatch.should_stop()` all-reduces the decision so no rank
  diverges. On the Ray grid the loop runs only on the driver (single decision point).
- On trigger: save a **full** `[model, optimizer, training_state]` checkpoint if this step does not
  already have one, even when regular or validation component selection is partial. Log the reason,
  exit cleanly (exit 0, `DONE` carries `preempted: ...`), and requeue with the unchanged command →
  `resume=auto` picks the emergency checkpoint up.
- `rl.emergency_checkpoint_components=run` (every single-role loop: GRPO, OPD, DPO, NFT, SFT; the PPO loops refuse it) replaces that forced full plan with the run's own
  `checkpoint_components` plus `model` through the cadence path — the save that fits the grace when two fp32 Adam
  moments make the full plan TiB-scale (about 5x the bf16 weight bytes in total). A model-only
  emergency artifact lands under `checkpoint_artifacts/iter_*` like a cadence save; the requeued run
  does not exact-resume from it (`resume=auto` ignores artifacts) — the continuation is the same warm
  start a partial cadence save allows. The skip is by component coverage of the step, not by step
  number: a validation artifact of the step counts only if it carries the model, and the config
  requires exactly that of a saving validation plan under `run` (a committed artifact is never
  re-committed, so the alternative would be the full plan). See "Checkpoint components" above.
- If model validation is due at that boundary, the emergency checkpoint records it as pending and
  remains unpublished until the resumed validation succeeds.
- With `save_every > 0`, normal completion saves its final step even off cadence. `save_every = 0`
  disables cadence and the normal-final save; emergency saves and configured validation saves remain active.

Submit-side wiring is built into `utils/launch_multinode.py`'s sbatch template
(`--signal=B:SIGUSR1@600` + trap + backgrounded launcher, double `wait`). This is
posttrain-only: the upstream VFM submit path deliberately does not request SIGUSR1, because
its `WallClockCheckpoint` callback already bounds lost work by wall-clock time.

### Preemption on a QOS-preempting cluster (`PreemptType=preempt/qos`, `PreemptMode=REQUEUE`)

What Slurm sends when a higher-priority job takes the nodes (slurm.conf(5) of Slurm 25.11,
`GraceTime` and `PreemptParameters`): once the job is selected, its end time becomes now +
`GraceTime`; the job's **tasks** are signalled immediately; at the new end time SIGCONT, SIGTERM and
SIGKILL go to the tasks and the batch script. Which signal arrives at selection time:

| cluster `PreemptParameters` | job submitted with     | signal at selection time                                                        |
| --------------------------- | ---------------------- | ------------------------------------------------------------------------------- |
| `send_user_signal`          | `--signal=B:SIGUSR1@N` | SIGUSR1 to the **batch shell** (the timeout path; the trap writes the sentinel) |
| `send_user_signal`          | no `--signal=`         | **SIGTERM to every task process at once** — no warning                          |
| unset                       | anything               | **SIGTERM to every task process at once** — no warning                          |

The two SIGTERM rows are fatal for a Ray-grid run, and the framework cannot change that from inside.
`proctrack/cgroup` delivers the signal to every process of every `srun` step: the per-node shell, the
`ray start` node process (`ray/_private/node.py`: its SIGTERM handler kills every Ray child process and
exits 1), each Ray worker process hosting a policy or rollout actor (`ray/_private/worker.py`,
`main_loop`: its SIGTERM handler exits 1), and the driver. The driver survives — `TerminationWatch`
installs its flag-only handler after `ray.init`, replacing the `sys.exit(signum)` handler Ray puts on a
driver — but when it reaches the next step boundary the actors it would save through are gone. Actor
code cannot install a flag-only handler of its own: `_Host` actor methods run on Ray executor threads
(`workers/ray.py`), and Python accepts `signal.signal` only on the main thread. So on such a cluster
the run's ONLY chance is the first row, and it is submit-side wiring:

```bash
#SBATCH --signal=B:SIGUSR1@<N>            # N >= one step + one emergency save (600 in the template)
export SLURM_LOG_DIR=<shared FS dir>       # must be visible INSIDE the container: mount it, pass the variable
mkdir -p "$SLURM_LOG_DIR"
trap 'touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED"' USR1
bash <launcher that runs srun ... the training entry> &   # the srun must NOT be the batch shell's
_lp=$!                                                    # foreground child: bash defers traps while
wait $_lp                                                 # one runs (scancel(1), --batch); the first
wait $_lp                                                 # wait returns on USR1, the second collects
```

plus `rl.emergency_checkpoint_components=run` when the full plan does not fit the grace (above). With
`send_user_signal` this is the same wiring the timeout path already needs, and preemption becomes: SIGUSR1
at selection → sentinel → the driver sees it at the next step boundary → one emergency save → exit 0 →
the SIGKILL at the end of the grace hits a finished process → `--requeue` restarts from that save. The
budget is one step plus one save inside `GraceTime`, because the sentinel is polled at step boundaries
only. The save is the small part: the Ultra arm's step-50 DCP model-only save of 2026-09-29 wrote
440 GiB over 32 ranks in at most 41 s (arm log: last rollout line 07:34:05, shard mtimes 07:34:42–44,
`saved checkpoint` 07:34:46). The step is the budget: with 8–12 min steps and a 20 min grace the save
fits whenever the signal lands with more than one step plus one minute of grace left, so a step that
runs longer than the grace minus one save is the only way to miss it — shorten the step or ask for a
longer grace.

Drill before trusting it: `scancel --signal=USR1 --batch <jobid>` delivers SIGUSR1 to the batch shell
only (scancel(1), `-b`) — the `B:` delivery exactly — without ending the job; the log must show the
sentinel reason and the save. `scancel --signal=TERM <jobid>` (tasks only, no `--full`) reproduces the
fatal rows.

Fallback for a cluster WITHOUT `send_user_signal` (verified on a local process tree only, not under
Slurm): a signal blocked in a process's mask is inherited across `exec`, `bash` and `subprocess.Popen`,
and a blocked signal is never delivered even when the receiving program installs its own handler — so
`ray start` launched as `python3 -c 'import os,signal,sys; signal.pthread_sigmask(signal.SIG_BLOCK,
{signal.SIGTERM}); os.execvp(sys.argv[1], sys.argv[1:])' ray start ...` keeps Ray's node, raylet and
worker processes alive through the tasks' SIGTERM (SIGKILL at the end of the grace still ends them),
the per-node task shell must `trap '' TERM` so the step is not torn down when the shell would exit,
and the driver's own SIGTERM flag (the `signal SIGTERM` reason) drives the save. Whether the raylet's
worker spawn preserves the mask has not been checked; drill it with `scancel --signal=TERM` on a canary.

## Local retention (`rl.keep_optim_last`, `rl.keep_model_last`)

`keep_optim_last=null` is the backend-aware default: it resolves to `1` for a local primary and `0`
for a direct object-store primary. Thus local runs retain optimizer state only for the newest full
iteration unless they explicitly set `0` (keep all) or another non-negative count.
`keep_optim_last=N` prunes `optim.pt`/`optim/` from all but the newest N iter dirs —
weights and trainer state are kept for EVERY iter, so posttrain model-load/eval history stays intact
while the disk hog is bounded. A training resume therefore requires an iter dir whose
optimizer component survived retention; an older model-only dir remains valid for
posttrain eval or DPO `rl.init_from`, but not as `rl.resume`; HF artifact export still requires
the separate conversion path. `keep_model_last=N` goes
further: ENTIRE iter dirs beyond the newest N are deleted — local disk becomes the
working set (resume only ever needs the latest). With upload enabled (below) a dir is
pruned ONLY after its `.uploaded` marker exists — the bucket is the archive; the only
copy is never deleted. Both prunes ride the save's write job (ordering guaranteed,
async included).

Retention is a pluggable policy: `prune_local_optimizer_state(ckpt_root, keep_last,
keep_fn=None, require_uploaded=False)` where
`keep_fn(iter_names) -> set` controls which iters keep optimizer state. Best-by-validation retention
is not implemented. With upload enabled, the manager applies `require_uploaded=True` per candidate;
a later successful upload cannot prune optimizer state from an earlier failed archive attempt.

These policies apply only to full checkpoints under a local `checkpoints/` root. The sibling
`checkpoint_artifacts/` namespace is not counted or pruned by them: the trainer never deletes an
artifact. Release is explicit —
`python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli prune-verified --paths <iter_dir>,...`
unlinks an artifact's payload files only when its `.uploaded` marker binds the destination the given
credentials address, the local bytes still match the digests the marker records, and every recorded
object is live at the destination with that size and digest; anything less keeps the copy with a
reason ([`checkpoint-artifacts.md`](../howto/checkpoint-artifacts.md)). Direct-primary mode resolves
the automatic optimizer policy to zero and rejects explicit non-zero values because deleting
committed remote checkpoints is not implemented.

## Local-checkpoint archive upload (`rl.upload.*`)

With `checkpoint_store.backend=local`, `rl.upload.enabled=true` pushes each just-saved iter dir to
`s3://<bucket>/<prefix>/iter_*/` on the checkpoint write job (rank-0, idempotent,
same-size skip). Ordinary upload is best-effort: a failure is logged and does not fail the run, but
the call or its background slot can still wait for client retries/timeouts before the next save or
finalization joins it.

A completed archive writes `.uploaded` beside the local iter dir. It is a JSON marker
(`posttrain.archive-marker.v1`, `checkpoint/archive/marker.py`) recording the non-secret destination
identity (endpoint authority, bucket, prefix) and the archived inventory: every file's size, plus a
SHA-256 for the files the archive path hashed. Full-checkpoint bulk (`model.pt`, `optim.pt`, DCP
shards) is not hashed on the training path, so the full-checkpoint marker is destination-bound; the
retention gates above test only that the marker exists. Markers written by earlier releases are one
line of text: they stay valid for retention but bind no destination and no digest, so nothing can be
digest-verified through them.

Validation completion is the exception: configured upload is a strict publication boundary, and
failure leaves DPO or GRPO/OPD publication pending instead of continuing training when a full
validation checkpoint exists. Partial artifacts use the same `archive.publisher.publish_artifact`
operation as the offline archive tool; the trainer catches failures and retains the local copy.
Every artifact payload, including DCP shards, is hashed. An immutable `.posttrain-archive.json`
declaration binds the complete inventory before payload upload: a matching retry may resume, while
a different inventory is refused before touching payloads. The native artifact manifest is committed
last, and `.uploaded` is written only after remote verification. Both the declaration and commit use
conditional creation; unsupported conditional writes fail closed. The declaration remains after
an interrupted attempt and cannot be reused for different contents. Artifact markers live under
`checkpoint_artifacts/` and do not gate full-checkpoint pruning. The local `.uploaded.tmp` temporary
is excluded from provenance so an interrupted receipt write cannot invalidate external-copy recovery.
The default prefix mirrors the local layout (`<project>/<group>/<name>/<RUN_TS>/
checkpoints`), which is exactly the reference-experiment bucket structure the shared
eval tooling consumes. Retention semantics live on the BUCKET, not in config: the
default experiments bucket auto-expires (~2 months); promote milestones to the
permanent bucket explicitly (`utils/object_store.py copy`). Remote is never pruned by
the framework.

## Config knobs (all under `rl.`)

| knob                                         | default                                               | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `save_every`                                 | 0                                                     | checkpoint cadence in the entry's published step axis (for example DPO optimizer steps, GRPO collected batches, or PPO outer policy versions); 0 disables cadence and normal-final saves, but not emergency saves or an explicit validation save                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `resume`                                     | `auto`                                                | see Resume above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `data.cursor_resume`                         | `auto`                                                | `auto` \| `strict` \| `off`; see preference-data cursor above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `run_ts`                                     | `""`                                                  | run-attempt timestamp; auto-resolved (see Layout) — users don't set it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `checkpoint_format`                          | `consolidated`                                        | `consolidated` \| `dcp`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `checkpoint_components`                      | `[model, optimizer, training_state]`                  | logical components for cadence and normal-final saves; when used, any proper subset warns and creates a non-resumable `checkpoint_artifacts/iter_*` artifact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `emergency_checkpoint_components`            | `full`                                                | plan of the termination-triggered (preemption / timeout) emergency save: `full` upgrades to the exact-resume plan whatever `checkpoint_components` says; `run` writes the run's own cadence plan plus `model` instead — for a policy whose full plan is TiB-scale and cannot finish inside the grace (a partial emergency artifact is not exact-resume capable); skipped only when every component of that plan is already on disk for the step; with `validation.save_checkpoint` the validation plan must COVER `checkpoint_components` plus `model` (a superset: a run plan `[model, training_state]` with a validation plan `[model]` is rejected). Honoured by every single-role loop (GRPO, OPD, DPO, NFT, SFT); the PPO loops always write the full plan and their config validation refuses `run` |
| `checkpoint_store.backend`                   | `local`                                               | primary resumable store: `local` \| `object_store`; direct object store requires DCP                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `checkpoint_store.bucket/credentials/prefix` | experiments bucket / repo secret / mirror local       | direct-primary address; ignored for a local primary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `save_async`                                 | `true`                                                | immutable CPU-offloaded consolidated snapshots, full or partial, may write in the background; DCP bulk stays synchronous (local DCP may background only its small post-collective tail, while direct-primary commit stays synchronous)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `keep_optim_last`                            | `null`                                                | backend-aware auto: local keeps optimizer state for the newest full iter (`1`), direct object store disables remote pruning (`0`); explicit `0` keeps all local optimizer state                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `keep_model_last`                            | 0                                                     | prune entire iter dirs beyond the newest N (0 = keep all; upload-gated)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `upload.enabled/bucket/credentials/prefix`   | off / experiments bucket / repo secret / mirror local | post-save upload (see above)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `validation.save_checkpoint`                 | `false`                                               | whether validation itself requests a save; a coincident regular save is still honored and merged                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `validation.checkpoint_components`           | `[model, optimizer, training_state]`                  | logical validation-save plan, used only when `validation.save_checkpoint=true`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Moving checkpoints (S3/GCS)

The checkpoint buckets are GCS behind an S3-compat layer with several sharp edges
(no server-side multipart copy, no streaming signatures, rejects boto3>=1.36 default
checksums). Use `utils/object_store.py` — it has the workarounds built in (cosmos-rl
parity) and is idempotent; run BY FILE PATH on login nodes:

```bash
# pull a base checkpoint to shared storage (for rl.init_from)
python3 cosmos_framework/posttrain/utils/object_store.py download s3://<bucket>/<prefix> /path/to/shared-storage
# preserve an expiring checkpoint (experiments bucket = 2-MONTH retention)
python3 cosmos_framework/posttrain/utils/object_store.py copy s3://<src> s3://<dst>
```

Checkpoint artifacts (`checkpoint_artifacts/iter_*`, produced by a partial component plan or by the
expert merge) have their own digest-verified lifecycle; these commands import the checkpoint package
and therefore run inside the training container
([`checkpoint-artifacts.md`](../howto/checkpoint-artifacts.md)):

```bash
# archive an artifact produced outside the trainer (a merge) under the run's upload prefix
python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli publish \
  --artifact-dir <run>/<ts>/checkpoint_artifacts/iter_000300 \
  --upload-prefix <project>/<group>/<run>/<ts>/checkpoints [--credentials <json>]
# stage an archived artifact: manifest first, every payload proven by size + sha256
python -m cosmos_framework.posttrain.checkpoint.tools.archive_cli fetch \
  --artifact-uri s3://<bucket>/<...>/checkpoint_artifacts/iter_000300 --dst <stage-dir>
```

These commands move or stage existing artifacts. A direct-primary DCP exact resume instead opens its
remote DCP components through the configured checkpoint store and does not download an `iter_*` tree
to a local cache first. Copy the complete iteration subtree, including `.posttrain/` and
`.posttrain-manifest.json` plus any `.validation-publication-*.json` visibility commit referenced by
that manifest, to preserve exact continuation; an explicit destination `s3://.../iter_*` then
resumes directly. Copying only `model/` and `optim/` preserves a model artifact but not trainer,
optimizer-clock, validation, or data-source continuation.

## Deferred

Full-DCP async (cosmos-rl's pinned-memory background process), remote-retention policy, secondary
archive from an object-store primary, score-based retention policy, Range-based remote DCP reads,
and `weights_only=True` loading.
