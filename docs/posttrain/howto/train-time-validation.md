# Train-time model validation

**Come here when you want held-out numbers while training, not after.** Validation evaluates a
committed model step against an explicit held-out source; posttrain never takes an implicit split
from the training data.

DPO evaluates the held-out preference objective, while
[`../guides/grpo.md`](../guides/grpo.md)-family and [`../guides/opd.md`](../guides/opd.md) runs
perform held-out rollout and reward. Not to be confused with the preflight
([`preflight.md`](preflight.md))).

Model validation evaluates a committed model step on an explicit held-out source. It is opt-in
under `rl.validation.*` and supports:

- Diffusion DPO: run the training preference objective over held-out pairs.
- LLM GRPO-family and OPD runs: run held-out prompts through rollout and reward
  without replay, teacher, or policy training.

This is not GPU-less [manifest validation](preflight.md) or checkpoint-driven benchmark
evaluation. DPO validation does not generate media or run external quality scorers.

## Configuration

Validation never derives an implicit split from training data. Enabling it requires a separate
`rl.validation.data` source.

| field                   |                              default | meaning                                                                                                                  |
| ----------------------- | -----------------------------------: | ------------------------------------------------------------------------------------------------------------------------ |
| `enabled`               |                              `false` | Enable held-out validation                                                                                               |
| `before_training`       |                              `false` | Validate fresh or `init_from` weights at step 0 without checkpointing when training steps remain                         |
| `every_steps`           |                                 `20` | Periodic cadence; `0` disables only periodic runs                                                                        |
| `save_checkpoint`       |                              `false` | Ask validation to save its model step; a regular cadence/final save due at the same step is independent and still runs   |
| `checkpoint_components` | `[model, optimizer, training_state]` | Components requested when `save_checkpoint=true`; when saved, a proper subset warns and creates a non-resumable artifact |
| `batch_size`            |                                  `1` | Physical validation batch size                                                                                           |
| `max_samples`           |                               `null` | Full set by default; positive smoke/debug cap                                                                            |
| `seed`                  |                                  `0` | Validation-only deterministic seed                                                                                       |

These fields are below `rl.validation`. Once enabled, validation of the final committed step is
mandatory; coincident periodic and final triggers are deduplicated. Checkpoint saving is not
mandatory: with the default `save_checkpoint=false`, validation alone evaluates and reports without
creating a checkpoint.

### DPO

```text
rl.validation.enabled=true
rl.validation.before_training=true
rl.validation.every_steps=20
rl.validation.batch_size=2
rl.validation.data.jsonl_path=<held-out-pairs.jsonl>
rl.validation.data.shuffle=false
rl.validation.data.cfg_dropout_rate=0
```

DPO accepts the same explicit preference sources as training. It requires
`rl.ref.build_ref=true`, `shuffle=false`, and `cfg_dropout_rate=0`. Training and validation pair
members must not overlap; a media archive may be shared when their pair inventories are disjoint.

The trainer runs the already-bound objective in inference mode with deterministic per-sample noise.
It restores RNG, model mode, and supported runtime caches afterward. Uneven FSDP shards use masked
padding so every rank executes the same forwards while padding contributes no metric weight.

Held-out DPO rejects `keep_encoder_cache=true` and `pad_for_cuda_graphs=true` because their retained
process-global state cannot be isolated safely.

### GRPO-family and OPD

```text
rl.validation.enabled=true
rl.validation.every_steps=20
rl.validation.batch_size=8
rl.validation.data.path=<held-out-prompts.jsonl>
rl.validation.data.prompt_column=question
rl.validation.data.response_column=answer
rl.validation.data.shuffle=false
rl.validation.reward=exact_match
```

An empty `rl.validation.reward` inherits `rl.reward`; answer-checking rewards require a response
column. Validation sampling is independent of training rollout sampling:

| `rl.validation.sampling.*` | default | meaning                                          |
| -------------------------- | ------: | ------------------------------------------------ |
| `temperature`              |   `0.0` | Deterministic by default                         |
| `top_p`                    |  `null` | Resolve to `1.0`                                 |
| `top_k`                    |     `1` | `null` disables top-k using the backend sentinel |
| `repetition_penalty`       |   `1.0` | No penalty                                       |
| `n_generation`             |     `1` | Generations per logical prompt                   |
| `max_tokens`               |  `null` | Inherit the training rollout limit               |

The validation request is pinned to the scheduled model step. A mismatched returned weight version
is fatal. Validation trajectories bypass replay, `policy.train`, advantage annotation, and training
filters. Effective sampling and model/tokenizer/packer semantics are part of exact-resume identity.
The packer declares its semantics through `PackerSemanticsIdentity`
([`custom-datapacker.md`](custom-datapacker.md)); on Qwen3-VL that identity binds the video
processor, so pending validations persisted before that binding refuse to resume, by design.
Shared mapped JSONL prompts are rejected; distinct lazy Hugging Face splits are trusted without a
full scan.

## Schedule and metrics

The public step is the number of committed optimizer updates: step 0 is the fresh or `init_from`
model, and step `k` is the model after update `k` (and GRPO weight synchronization). For four steps
with `before_training=true` and `every_steps=2`, validation runs at `[0, 2, 4]`; for three steps it
runs at `[0, 2, 3]` because final validation is unconditional.

A pure before-training pass on a fresh run with at least one configured training step is ephemeral:
it emits the normal validation frame but creates no `iter_000000` checkpoint or durable validation journal.
If termination is already requested, the process exits without evaluating or saving and the next
fresh start retries the pass. This pure initial exception remains checkpoint-free even when
`save_checkpoint=true`. When the configured run has zero training steps, step 0 is also the final
validation, but it still remains checkpoint-free unless `save_checkpoint=true`.

Validation and regular checkpoint cadence are independent requests. When both are due at one model
step, their logical component sets are unioned and written once. The logical names are `model`,
`optimizer`, and `training_state`; all three are required for exact resume. If the union is partial,
the independently due regular artifact is committed before validation, then validation runs
ephemerally. It remains under the sibling `checkpoint_artifacts/iter_*` namespace, never advances
the training latest pointer, and is not a resume source. See
[checkpoint.md](../reference/checkpoint.md#logical-component-selection).

DPO `max_steps` already counts optimizer updates. GRPO-family and OPD `max_steps` count collected
batches, each committing `n_mini` model steps, so the validation total is `max_steps * n_mini` and
`every_steps` must be divisible by `n_mini`.

Validation is synchronous at optimizer boundaries. DPO rejects validation during gradient
accumulation. A future asynchronous loop must drain in-flight work and pin one weight version first.

Metrics use canonical namespaces: `train/*`, `rollout/*`, `val/*`, `optim/*`, and `timer/*`. The
controller is the single reporter; training and validation at the same step form one logical
console/W&B frame. Counts exclude masked padding and weighted/moment reductions are merged globally.

GRPO-family and OPD runs always report `val/reward_avg`, `val/reward_std`, `val/reward_min`,
`val/reward_max`, `val/rollout_count`, `val/step`, and `val/train_total_steps`. Reward components may add
`val/<metric>` and `val/<metric>_count`; absent components are not filled with zero.
`val/rollout_count` must equal held-out prompts times `n_generation`.

DPO reports held-out loss, reward margin/winner/loser, accuracy, policy/reference errors and raw
losses, plus `val/pair_count`, `val/batch_count`, and `timer/validation_s`.

## Checkpoint, resume, and failure

Resume behavior depends on whether the boundary produced a full checkpoint:

- completed validation recorded in a full checkpoint is not repeated, and `before_training` is not
  replayed after that exact checkpoint resume;
- a fresh restart before the first optimizer update reruns the ephemeral before-training pass;
- `init_from` starts a new run at step 0;
- interrupted evaluation backed by a full pending checkpoint restarts deterministically before the
  next update;
- validation completed before an interrupted full-checkpoint archive retries publication without
  rerunning the DPO objective or the GRPO/OPD rollout/reward pass.

With `save_checkpoint=false` and no regular save due, initial, periodic, final, and zero-step-final
validation are ephemeral. They still use the in-memory begin/complete lifecycle, but persist no
journal. A process failure resumes the latest regular full checkpoint; it may therefore replay work
since that checkpoint before rerunning validation.

When validation requests the full component plan, or its plan unions with a coincident regular save
to become full, the transaction records config/source identity and a durable evaluation-pending
marker before reading held-out data. DPO, GRPO-family, and OPD then move successful evaluation to a
publication-pending phase. Configured archive upload is a strict boundary that must succeed before
training continues; without upload the same transaction atomically commits the local completion.

A partial plan requested solely by validation evaluates first and commits a non-resumable artifact
afterward. It has no durable pending-evaluation or publication-retry guarantee, and its optional
archive upload is best-effort. When an independent regular partial save is due at the same step, the
merged artifact is committed first and remains present even if the following ephemeral validation
fails. Empty, duplicate, or unknown component selections fail config validation. The checkpoint
manager emits `UserWarning` when an active proper subset reaches checkpoint planning; configuring
an inactive validation subset does not warn. A termination detected at a post-update validation boundary
overrides both configured plans and writes a full pending checkpoint. The fresh step-zero
before-training exception still exits without saving so a fresh start can retry it.

`ValidationTransaction` owns the in-memory begin/complete lifecycle for every trainer and evaluator;
evaluators do not create or clear journal state implicitly. Checkpoint-free validation uses that
lifecycle transiently and persists neither phase. For full durable phases the checkpoint is
authoritative; external metric delivery is an at-most-once best-effort attempt. See
[checkpoint.md](../reference/checkpoint.md#pending-model-validation) for checkpoint mechanics.

Data, model identity, metric-schema, non-finite metric, and weight-version failures are fatal and
converge across ranks. A full durable-phase failure leaves the pending journal, emits no canonical
validation frame, and prevents another optimizer update. An ephemeral or partial-artifact phase also
emits no frame or update on evaluation failure but has no durable journal; restart begins from the
latest full checkpoint. A reporter transport failure disables only that sink.

## Current limits

- DPO performs held-out objective validation, not generation or external benchmark evaluation.
- GRPO-family and OPD runs support the synchronous local rule-reward path; validation-only teacher,
  advantage, and filter metrics do not exist because those training annotations are skipped.
- `max_samples` is a deterministic cost/debug cap; there is no mesh-dependent `max_batches` or
  automatic split.
- VLM GRPO, diffusion RL, VLA, and SFT need modality-specific adapters before they are runnable.
- Best-checkpoint retention and early stopping are not implemented.
