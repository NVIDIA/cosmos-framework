# PPO

Shared-backbone actor/value training uses the separate `joint_ppo` entry. See
[public Joint PPO instructions](../howto/run-posttrain.md#8-embodied-ppo). The text PPO entry below retains its
independent policy/reference/critic roles and shares only common PPO primitives and infrastructure.

The PPO entrypoint adds only the roles and algorithm logic that differ from GRPO:

- `reference`: frozen token-probability scoring;
- `critic`: value prediction and value-function optimization;
- `policy`: PPO policy optimization through the shared `LLMTrainer` objective seam.

Everything else uses the existing posttrain components: `DatasetEnv`, `RewardWorker`,
`ExperienceBuffer`, `WorkerGroup`, `ActorGroup`, `DeviceGroup`, `bind_device_groups`,
the shared dispatch/data plane, multi-role checkpoint coordinator, and `weight_sync`.

## Flow

Each iteration is a plain imperative role sequence:

```text
data → rollout → reward → replay → reference.score → critic.predict → policy.prepare_batch
policy.prepare_batch → critic.install_targets
policy.train_batch + critic.train_batch
policy → rollout weights
```

The roles pass one `Trajectory` handle along the chain. Reward, reference, and critic
append PPO-owned fields to the trajectory; the controller does not inspect or move its
bulk data. Model-DP ingress is declared in the common dispatch registry and selected from
each consuming role's live `ParallelDims`. Because each trainable role therefore sees only its
own shard, the batch identity it publishes names the COMPLETE batch, not that shard: the
controller compares whole receipts across ranks and then hands one rank's `batch_id` back to
every rank as the key into its own shard cache.

The policy role computes GAE and immutable PPO targets, then the policy and critic each
run the configured epoch/minibatch schedule. Weight synchronization uses the same
`weight_sync(rollout, policy)` call as GRPO and OPD.

The policy and critic share one PPO update runtime over the existing `LLMTrainer`; optimizer
execution remains in the common engine. PPO specializes only vLLM request/output handling and
reuses the common rank-converged generation lifecycle.

## Critic warmup

`rl.critic_warmup_iterations` (default `0`, a non-negative integer counted in OUTER PPO iterations)
runs a leading critic-only phase. A freshly initialized value function is not yet calibrated to the
task's reward distribution, so its estimates make noisy GAE advantages; updating the actor on them
reinforces unhelpful actions and shows up as early reward fluctuation and larger policy-KL
excursions. During iterations `1..N` the loop still collects rollouts, scores rewards and the frozen
reference, prepares targets and trains the critic; the actor's parameters, optimizer state and LR
scheduler are left exactly as they were. The first actor update is iteration `N+1`. `0` reproduces
the previous behaviour exactly, including the update sequence and the metric step.

Two clocks come apart while warmup is active, and only one of them is the actor's:

- the OUTER ITERATION is the loop's durable step — the checkpoint boundary name, the metric step,
  and the seed offset in the shared update plan that keeps the critic's minibatch shuffle moving;
- `policy_version` counts PUBLISHED actor updates and is the rollout's staleness contract, so it
  stands still for the whole warmup phase.

`weight_sync` is therefore skipped during warmup: the actor is byte-unchanged, the rollout already
advertises exactly `policy_version`, and collection's staleness guard is an equality check that
already holds, so a re-push would add a full model transfer — and a transport failure surface — to
an iteration with no weight delta. Both roles still receive the SAME update plan; warmup never
varies the epoch/minibatch shape, which each role pinned into its checkpoint identity.

Metrics carry the phase: `train/actor_warmup` is `1.0` on a critic-only iteration, and that
iteration publishes no policy receipt, so the `train/policy_*` and `optim/policy_optimizer_step`
keys are absent rather than repeated (see [`../reference/metrics.md`](../reference/metrics.md)).

Checkpointing a warmup iteration does not touch the actor either, which takes an explicit
mechanism rather than following from the actor being frozen. Under `checkpoint_format="dcp"` — the
production default — optimizer state is extracted with `get_optimizer_state_dict`, and that helper
MATERIALIZES a never-stepped optimizer by running a zero-gradient `step()` at zero learning rate.
No parameter moves and `optimizer_step` stays `0`, so nothing visible records it; but AdamW's own
step counter advances, and it is what bias-corrects the first REAL actor update. Left alone, that
would make `rl.save_every` — an operational knob — change the weights a run produces. The trainer
therefore rolls the initialization back, writes an empty optimizer state, and marks the checkpoint
`optimizer_pristine`; resume reads the marker before touching the optimizer, because extracting
the load TARGET has exactly the same side effect.

Two details matter for `rl.optimizer.name`. The rollback covers group-owned state, not just
`optimizer.state`, because `fusedadam` keeps its counter on the param group — see
[`../reference/checkpoint.md`](../reference/checkpoint.md). And a pristine resume still restores
the saved `param_groups`: the learning rate lives there, and the scheduler's own state does not
write it back into the optimizer.

Resume preserves warmup progress, including a run that ends inside it: the committed boundary is
named by the outer iteration, and the actor's own publication clock restores from its trainer
state. The boundary also records the configured `rl.critic_warmup_iterations`, and resume compares
that recorded value DIRECTLY; a run configured with a different warmup fails closed instead of
silently renumbering the actor's published versions. The value has to travel with the boundary
because it cannot be recovered from the actor's clock: an actor that never published reports
version 0, so `max(0, iteration - N)` reads the same under every `N` at least as long as the
boundary's iteration count, and a lengthened warmup would otherwise be accepted and slide the first
actor update later with nothing reported. That derived relation is still checked, as an independent
cross-check on the restored actor's own counter.

A boundary committed before the marker carried the warmup recorded none. Absence is treated as
unknown rather than as a mismatch, so such a boundary keeps exactly the guard it shipped with — the
derived cross-check alone, which cannot discriminate a lengthened warmup — and the resume logs that
the configured value could not be verified against it.

## Launch

Set up the container and Posttrain environment from
[sections 1 and 2 of the run guide](../howto/run-posttrain.md#1-start-the-container). The public
profile uses Qwen3-0.6B from Hugging Face: two colocated policy/rollout GPUs, one frozen-reference
GPU, and one critic GPU. Validate the typed config without allocating workers, then run it:

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.ppo \
  --config cosmos_framework/posttrain/configs/ppo_qwen06b_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.ppo \
  --config cosmos_framework/posttrain/configs/ppo_qwen06b_public_4gpu.py -- \
  job.name=ppo_public_4gpu
```

The profile runs 1000 steps; append `rl.max_steps=5` after `--` for a short check (see the
[run guide Q&A](../howto/run-posttrain-qa.md)).

The same role flow supports text, image, and video prompts; the selected
DataPacker owns model-specific tokenization and multimodal tensor preparation. Set
`data_packer.factory` to use the same custom packer-factory seam as GRPO/OPD; `rl.data`
also accepts a Hugging Face name, a local file of rows, or an importable dataset factory.
Configured rows use the generic `project_prompt_items` adapter over the existing
`DatasetEnv`/cursor path; PPO defines no separate prompt-dataset implementation. The prompt
column holds either a non-empty string (one user turn) or a conversation list — a non-empty
list of `{"role", "content"}` objects, each with a non-empty role and a non-empty string or
list content, the image or video inside the content parts — the same two shapes the GRPO
loader accepts, passed through to the DataPacker untouched. A local `rl.data.path` is read as
written by the data roles' own reader (`JsonlRows`, shared with the GRPO loader) in the shapes the
Hugging Face reader accepted -- one JSON object per line or any whitespace-separated stream of
objects (one object pretty-printed over several lines is one row; several, one row each; consecutive
objects need at least one whitespace character between them, so `}{` is refused with its line and
column), one top-level JSON array (a file whose first non-blank line starts with `[`), any of them
gzip-compressed (gzip only, detected by the magic bytes; not bz2 / xz / zst) -- not through
Hugging Face `datasets`; a bad row is refused with its position, in file order (a byte outside UTF-8
further down never hides an earlier bad row):
Arrow would infer one struct type over every content block and null-fill the keys a block lacks
(which the Qwen chat template renders as
spurious image placeholders), null-fill a nested reference column the same way, and refuse a
conversation column whose turns mix string and list content. The projection refuses content parts
that carry the null-fill signature -- a null under a payload key (`image: null` on either a text
part or an image part with another handle) or a null `type` -- whatever reader produced them, when
the data role is built (row 0 is projected at construction); a null under a part's own optional key
(`fps: null` on a video part) is left to the DataPacker's contract.

## Reward

`rl.reward.terminal` selects the episode-level verifier. It takes either of PPO's own matchers —
`arithmetic_exact_match` (default) or `none` — or the name of any shared rule reward, the same
registry GRPO selects from with `rl.reward`: `arithmetic`, `gsm8k`, `countdown`, `exact_match`,
`nonempty`, `constant`.

Prefer a shared rule unless you need PPO's exact-format matcher. `arithmetic_exact_match` requires
the completion to be *nothing but* an integer, so a correct answer written as prose scores zero;
the shared `arithmetic` rule reads the first integer anywhere in the completion.

Two consequences of naming a shared rule:

- Its declared `needs` carry through to the run. A rule needing `reference` (every one above except
  `nonempty` and `constant`) makes the framework populate `RewardContext.reference`, and makes the
  manifest require a ground-truth source — so `--validate-manifest` rejects a run whose data cannot
  supply one instead of failing on the cluster.
- The reward stays on PPO's scale (`correct_reward` / `incorrect_reward`, default `1.0` / `0.0`)
  and is emitted under PPO's own term, so the objective and metrics are unchanged.

`terminal_callable` remains the escape hatch for an importable `package.module:callable`. It
overrides PPO's own matcher and is rejected in combination with a shared rule name, since the two
would otherwise compete silently. `rl.reward.process` (`linear` | `none`, default `none`) is the
separate per-token shaping term.

## Current runtime scope

The current synchronous PPO entry requires `grid.backend=ray` and `grid.grouping=per_role`.
Rollout uses vLLM and produces one completion per prompt. Policy, frozen reference, and critic
may use supported HF or native model construction; training data may be text, image, or video.
Multi-role checkpoint publication currently requires the local checkpoint store.

## Placement

PPO lowers its role declarations through the shared placement binder.

- `gpu_placement=colocated` places policy and rollout ActorGroups in one GPU DeviceGroup;
  reference and critic each use their own DeviceGroup.
- `gpu_placement=disaggregated` gives policy, rollout, reference, and critic independent
  DeviceGroups; set `rl.weight_sync_transport=nixl` to use the shared NIXL weight transport.

Role shape and model parallelism remain separate: WorkerGroup placement owns
`(replicas, ranks)`, while each model role owns its `(dp_replicate, dp_shard)`
factorization.

## Checkpoints

Policy and critic save through their existing `CheckpointManager` instances, writing straight
into the boundary directory alongside the `DatasetEnv` cursor. The shared multi-role coordinator
then publishes one top-level `_SUCCESS` marker — the commit point, written only after every role
is verified resumable — followed by the `latest_checkpoint.txt` pointer. An uncommitted boundary
is invisible to `resume`, and a partial role save is never resumable. Multi-role publication currently supports the local checkpoint
store; config validation rejects the direct object-store backend before worker allocation.

The marker also records the optimizer counter each role committed, and restore checks every role
against its own recorded value. Roles in one boundary are not required to share a counter — during
critic warmup the actor's stands still while the critic's advances — so cross-role agreement is no
longer the invariant; a boundary written before this field is still validated by the old agreement
rule.

Alongside those counters the marker carries `loop_state`: one OPAQUE run-level slot the entrypoint
owns and the shared coordinator never interprets, read back verbatim through
`committed_loop_state()`. PPO records `{"critic_warmup_iterations": N}` there. The coordinator keeps
no schema for it on purpose — it is shared with the single-role loops, and a warmup field on it
would bind an algorithm-neutral boundary to one algorithm. Values must be JSON scalars so the
read-back comparison is exact; the marker `version` is unchanged, because the field is additive and
optional exactly as `role_steps` was.
