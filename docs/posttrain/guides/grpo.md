# GRPO

Online reinforcement learning: sample a group of completions per prompt, score them, and train on
the group-relative advantage. Facts here are read from `config/rl.py::GRPOConfig`,
`entrypoints/grpo.py`, `algorithm/grpo.py`, and the recipes under `configs/`.

## 1. When to choose it

Use GRPO when you can **score a completion** — with a rule, a verifier, a judge, or a model — but
you have no preference pairs to train on directly. It needs a live rollout engine and a reward
that discriminates between samples of the same prompt.

It is the wrong choice when your reward gives every sample in a group the same score. The advantage
is computed **within a group**, so a group whose rewards tie contributes exactly zero gradient. If
your reward is nearly always 0 or nearly always 1, fix the reward before tuning the optimizer. If
you already have preference pairs, [`dpo.md`](dpo.md) is cheaper — it needs no rollout engine. If
you have a stronger model to imitate rather than a reward to maximize, use
[`opd.md`](opd.md).

## 2. Quick start

Set up the container and Posttrain environment from
[sections 1 and 2 of the run guide](../howto/run-posttrain.md#1-start-the-container), then run the
public recipe: Qwen2.5-3B on GSM8K, policy FSDP and rollout TP colocated across four GPUs, rule
reward.

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py -- \
  job.name=grpo_public_4gpu
```

The recipe runs 300 steps; append `rl.max_steps=5` after `--` for a short check (see the
[run guide Q&A](../howto/run-posttrain-qa.md)).

Dr-GRPO and GSPO are config variants of GRPO and run through the same `entrypoints.grpo`. **Neither
has a public recipe yet**; a ready-to-run example will follow. Until then, build your own: copy
`grpo_qwen3b_gsm8k_public_4gpu.py`, keep its `make_config()`, and set the variant's fields on the
returned config.

- **Dr-GRPO**: `algorithm = "dr_grpo"`, `rl.unbiased_advantage = True` and
  `rl.loss_type = "seq-mean-token-sum-fixed"`, as in `dr_grpo_qwen3b_gsm8k.py`.
- **GSPO**: `algorithm = "gspo"` and the fields that `apply_gspo_knobs()` in `gspo_qwen3b_gsm8k.py`
  sets: a clipped surrogate with sequence-level importance sampling, the paper's epsilons, a
  `seq-mean-token-mean` reduction and a mini-batch cadence that fits your rank count.

Run `--validate-manifest` on the new recipe before training. §5 and
[`custom-objective.md`](../howto/custom-objective.md) describe each field.

## 3. Data contract

`rl.data` selects the prompt source. Leave it unset and the built-in arithmetic generator runs.
Otherwise: a JSONL file, a Hugging Face dataset, or an importable custom factory
([`../howto/custom-data.md`](../howto/custom-data.md)).

`rl.reward` picks what scores the completions. `arithmetic` and `exact_match` check against a
ground-truth answer; `gsm8k` applies the `#### N` verifier; `nonempty` checks only that the
completion is non-empty.

> `nonempty` is a connectivity smoke signal, not a training objective. Every completion scores 1,
> every group ties, and the group-relative advantage is zero throughout.

A reward whose declared input the source does not carry is rejected **before** Ray placement and
model construction. Runtime-null fields and malformed trajectories fail loudly rather than being
converted into a numeric-zero reward — zero is a legitimate score for a well-formed failed attempt,
so it must not double as "missing".

Custom scorers: [`../howto/custom-reward.md`](../howto/custom-reward.md).

## 4. Topology and launch

Two independent axes:

| Axis                 | Values                         | Meaning                                                                                                                 |
| -------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `grid.backend`       | `local` \| `ray`               | `local` is one torchrun world (TP=1, the test path). `ray` is required for `tensor_parallel_size > 1` and for multinode |
| `grid.gpu_placement` | `colocated` \| `disaggregated` | Whether rollout and policy share one GPU pool or own disjoint pools                                                     |

Under `ray`, shapes are explicit: `grid.shapes.policy=[1,POLICY_RANKS]` and
`grid.shapes.rollout=[ROLLOUT_REPLICAS,ROLLOUT_RANKS]`, where `ROLLOUT_RANKS` must equal
`rollout.tensor_parallel_size` — the entry rejects a mismatch at startup rather than deadlocking
later.

**`POLICY_REPLICAS` must be 1.** Policy placement has no replica axis; configure data parallelism
as `dp_replicate * dp_shard` inside `POLICY_RANKS` on `model_hf.parallelism` instead. This is the
single most common configuration mistake — see [`../glossary.md`](../glossary.md) for why
`dp_replicate` and placement `R` are different things.

`disaggregated` additionally needs a cross-device weight transport
(`rl.weight_sync_transport=nixl`). Multiple nodes require operator-provided orchestration.

## 5. Key configuration

The complete table is [`../reference/config.md`](../reference/config.md). In routine use:

| Knob                                 | What it controls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rl.group_size`                      | Completions sampled per prompt. Larger groups give a better advantage estimate and cost proportionally more rollout                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `rl.batch_per_replica`               | Trajectories per collected batch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `rl.micro_batch_per_replica`         | Memory-only split of one optimizer step. Does not change the result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `rl.mini_batch_per_replica`          | Several optimizer steps per collected batch. **Does** change training dynamics, and requires the frozen pi_old pass                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `rl.epsilon_low` / `rl.epsilon_high` | The clip bounds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `rl.kl_beta`                         | KL penalty toward a frozen reference (the initial policy weights). `0.0` disables it; a positive value also needs `rl.ref.build_ref=true`, which builds a second sharded copy of the policy on the trainer ranks (scored on every training chunk, not on the pi_old ahead pass; the build refuses a policy with live dropout). `rl.ref.model_dir=<dir>` anchors that reference to another checkpoint: a warm restart from a model-only save starts the policy from the save and keeps the KL anchor at the original base. The content of that directory is part of the checkpointed objective identity, so an exact resume cannot silently switch the anchor                                                                                                         |
| `rl.kl_coeff_clamp`                  | Bound on the KL term's per-token gradient coefficient (nats; `5.0` recommended with any positive `rl.kl_beta`). The logged `train/kl_mean` / `train/kl_max` stay the exact k3; only the gradient beyond `\|z\| = c` is held at the boundary slope. Unset (`null`) keeps the exact, unbounded gradient and the config warns                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `rl.temperature`                     | Policy-logit temperature for the loss. Keep it equal to the rollout temperature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `rl.grpo_surrogate`                  | `clipped` (production) or `legacy_pg` (no ratio, no clip — the arithmetic quick start)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `rl.importance_sampling_level`       | `token` (default) or `sequence`. `sequence` takes the clipped surrogate's importance ratio per response rather than per token — this is what makes a run GSPO rather than GRPO                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `rl.mask_truncated_completions`      | DAPO overlong filtering. `false` (default) trains on every completion; `true` drops the completions the vLLM engine cut at `rollout.max_tokens` (`policy_info["finish_reason"] == "length"`) from the surrogate, the KL term and their divisors. The mask removes the cut completion from the loss only: advantages were computed over the whole group before dispatch (the kept members keep their full-group values) and the group still counts toward the informative-group gate. Watch `train/rollout_truncated_frac` first: a cut completion is scored on an answer it never finished and, under token-sum reductions, outweighs a median response by `max_tokens / median_length`                                                                              |
| `rl.score_centering.enabled`         | Score centering under the sampler's distribution (`false` default). `true` makes the vLLM rollout return its top-k head per generated token (`rl.score_centering.top_k`, 128) in processed-logprob mode and the objective subtract the sampler-expected score, removing the train/inference-mismatch drift from the gradient without changing the KL term or the reduction. `rl.score_centering.compose` = `none` (pure centering) / `tis` / `mis` composes truncated or masked importance sampling. Watch `train/sc_head_mass_q` (≥ 0.99 expected), `train/sc_rho_max` and `train/sc_kl_head`                                                                                                                                                                       |
| `rl.optimizer.eps`                   | Adam's denominator floor. The shipped `token-mean` recipes carry the cosmos-rl TOML's `1e-6`, sized for `token-mean` gradients. Under `rl.loss_type=seq-mean-token-sum-fixed` (Dr-GRPO) every sequence's loss is divided by the fixed `rollout.max_tokens`, the gradients shrink by tens of times and fall below that eps, and the update degenerates to `(lr/eps) · m` — `lr` stops naming the step size and does not port across `max_tokens` or `eps`. The config warns for that pair (refuses it under `rl.recipe_guard_severity=error`); the eps belongs to the reduction, so the Dr-GRPO Qwen3-VL RLVR layer (`the Dr-GRPO recipe layer`) selects `1e-8` together with the fixed reduction, and lr still wants a re-tune ([reference](../reference/config.md)) |
| `rl.optimizer.name`                  | `adamw`, `fusedadam` (what shipped recipes use) or `muon`. Muon reuses the shared `MuonWithAuxAdamW`: supported hidden Linear weights and explicitly declared expert stacks take Newton-Schulz; embeddings, heads, routers, norms, biases, convolutions and ambiguous fused 2-D weights stay on auxiliary AdamW. Expert-axis FSDP shards are processed locally and fused expert gate/up halves are split. Requires FP32 storage with BF16 FSDP compute. `muon_rms_match=0.2` is an initial LR-scale heuristic, not a promise that an Adam-tuned LR carries over; re-tune LR. Knobs: `muon_momentum=0.95`, `muon_ns_steps=5`, `muon_rms_match=0.2` ([reference](../reference/config.md))                                                                              |
| `rollout.gpu_memory_utilization`     | GPU fraction reserved for the colocated engine. Lower it when the policy OOMs in `log_softmax`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `rl.seed` / `rollout.seed`           | The prompt-order seed and the vLLM sampling seed — two different things; see the paragraph below                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `rl.data_seed_follows_rollout_seed`  | `false` (default) orders prompts with `rl.seed`; `true` orders them with `rollout.seed`, so one per-arm seed varies both                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `rl.recipe_guard_severity`           | `warn` (default) keeps the two recipe-hygiene checks — a literal `rl.min_informative_groups` below half of `groups_per_step`, and `seq-mean-token-sum-fixed` with `rl.optimizer.eps > 1e-7` — as the warnings they always were; existing fields and default runtime behavior remain unchanged. `error` collects every hazard found and refuses the config in one error that lists them all, for a recipe that wants a mis-pairing caught at `--validate-manifest`; the Dr-GRPO Qwen3-VL RLVR layer sets it                                                                                                                                                                                                                                                           |

Sizing batches and diagnosing OOM: [`../howto/batching.md`](../howto/batching.md).

### What a seed replicate varies

Two seeds reach a run. `rl.seed` orders the prompt source: with `rl.data.shuffle=true` the data
cursor permutes the corpus with `rl.seed + epoch`, and the built-in arithmetic source draws from
`rl.seed`. `rollout.seed` seeds the vLLM engine and decides which completions are sampled for a
given prompt. An arm set that gives each arm its own `rollout.seed` and leaves `rl.seed` at its
default therefore reads the **identical prompt sequence** in every arm and varies only the
sampling — a sampling replicate, not a data-order replicate — and preflight warns about that
combination ("all arms share one prompt order"). To make one per-arm seed vary both, set
`rl.data_seed_follows_rollout_seed=true` (the data cursor is then keyed on `rollout.seed`, which
must be an integer), or give each arm its own `rl.seed` as well. The same fact governs
continuations: the rows a shuffled run has consumed are a seeded random subset of the corpus, not
a file prefix, so a "tail" pool built by removing leading rows does not remove what the parent
actually saw. The resolved data seed is recorded in the cursor checkpoint, and an exact resume
refuses a run whose data seed changed, so decide this before step 0.

### Dynamic sampling (DAPO)

A group whose rewards all tie has zero advantage for every member: it contributes no gradient but
still occupies batch capacity. `rl.min_informative_groups > 0` turns on DAPO-style dynamic sampling
(arXiv 2503.14476 §3.2): the loop keeps drawing whole prompt batches until the replay cells together
hold that many informative groups, trains on those first, and discards the rest. The count is global:
`control/loop/batching.py::informative_quota_per_cell` splits it exactly across the data cells (the
first `N mod R` cells owe one group more), so `min_informative_groups=1` means one group whether the
run has one data replica or eight.

The flip side of a global count is that it does not scale with the batch either. The capacity it is
a share of is `groups_per_step = batch_per_replica // group_size` (placement-independent: the data
cells split that one batch), and a literal tuned for one layout — `8`, every group of `b32g4` —
copied onto a wider one — `b256g4`, 64 groups — quietly becomes a 12 % floor while the run still
reports dynamic sampling ON. Spell the quota as a share instead: `rl.min_informative_groups_frac = f`
(`0 < f ≤ 1`) resolves to `ceil(f × groups_per_step)` groups, so `1.0` is full DAPO on any layout.
The pair is resolved in one place, `GRPOConfig.effective_min_informative_groups`, which is the only
value the loop reads; setting both knobs is rejected, a literal count below half of
`groups_per_step` warns at config time (the message names `groups_per_step`, the count and the
fraction key; `rl.recipe_guard_severity=error` refuses it), and the run log prints the resolved quota
and `groups_per_step` when dynamic sampling is on.

- **Who decides.** The reward stage stamps every complete group's reward spread while it holds the
  group (`algorithm/annotators/advantage.py`); the replay buffer only compares that stamp against a
  declared threshold. The decision is therefore per group and does not depend on how trajectories
  are spread over cells. The reward report's `grpo_informative_group_count` stays a diagnostic.
- **Budget.** `rl.dynamic_sampling_max_extra_rollouts` caps the extra ROUNDS per step (a round = one more
  draw of `prompts_per_data_cell` prompts on each participating cell); how many `rollout.generate` calls those
  rounds take is the refill dispatch below, counted by `train/grpo_dynamic_sampling_refill_dispatches`. When
  it is spent and a cell is still short, `rl.dynamic_sampling_on_budget_exhausted`
  decides: `raise` (default) aborts the run, because a reward that cannot produce informative groups
  within budget is a recipe fault to fix, not to train through; `proceed` keeps a fixed batch shape
  by filling with tied groups and logs the shortfall.
- **Refill dispatch.** `rl.dynamic_sampling_refill_dispatch` says how many rounds one `rollout.generate`
  call carries once the base draw leaves a cell short (which cells draw is a different axis): `per_round`
  (default) issues one extra round per call and re-counts the cells between rounds, so a step that needs K
  rounds pays K round-trips, each waiting for the slowest engine's longest completion; `whole_budget`
  issues the whole remaining budget in ONE call (each participating cell serves `budget ×
  prompts_per_data_cell` prompts while a cell idled by `short_cells` serves none; its engine request is `budget ×
  prompts_per_data_cell × group_size` sequences wide when cells map 1:1 to engines), so those K round-trips
  become one. The selection remains informative first and first-seen in draw order per cell. The trained set
  is identical when no refill was needed or the
  budget is exhausted under both values, and whenever the quota equals the per-cell share (full DAPO,
  below); below full DAPO, when `per_round` would have stopped early, `whole_budget` replaces tied filler
  with informative groups from the rounds `per_round` never drew. The data cursor advance is identical iff
  no refill was needed or the budget is exhausted under both values; otherwise `whole_budget` advances each
  participating cell by `budget − k` rounds more (k = the extra rounds `per_round` took). With
  `refill_mode=short_cells`, the participating set is the cells short before the wide call; a cell satisfied
  midway still receives the remaining rounds, while cells already satisfied remain idle. Completions are equal in
  distribution only (engine-level seed). `train/grpo_dynamic_sampling_refill_dispatches` counts the calls.
- **Full DAPO** is `rl.min_informative_groups_frac = 1.0` (equivalently the literal
  `rl.min_informative_groups = batch_per_replica // group_size`); anything lower leaves tied filler
  in the batch by design.
- **Cost.** Every extra draw is a full rollout plus reward pass; watch
  `train/grpo_dynamic_sampling_extra_rollouts` and `..._dropped_trajectories`. Dropped draws are
  released from the replay boundary as metadata only: their bulk leases are dropped at the owning
  store without a fetch (and, on the pooled data plane, their pages return at the step barrier), so a
  run that tops up on every step neither leaks rollout memory nor moves the bulk it discards.

Dynamic sampling is GRPO-only; `OPDConfig` rejects the knob because distillation trains on every
trajectory regardless of reward spread.

### Sparse-MoE policies: train/infer routing mismatch

On a sparse MoE the router's top-k over near-tied scores flips between two
forwards at unchanged weights, and between the vLLM sampler and the FSDP training forward, so the
training forward routes some tokens through experts the sampler never used. At
`n_mini == 1` the importance ratio cannot see this (it is identically 1); at `n_mini > 1` it trips
the pi_old determinism probe. `rl.routing_replay=rollout` (Rollout Routing Replay) makes the engine
return the routing it used and pins every training forward to it. With `rl.kl_beta > 0` that includes
the frozen reference's forward, which prepares the same batch: the KL anchor is the reference under
the rollout's expert dispatch, so `train/kl_mean` measures weight drift rather than routing churn.
Constraints:

| Constraint                                                                                                                                                 | Why                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native Qwen3-VL-MoE policy with the vLLM rollout                                                                                                           | the only `ForwardReplayPolicy`; assembly refuses dense and HF policies, OPD refuses the knob                                                                      |
| `grpo_surrogate=clipped` (the default)                                                                                                                     | only the clipped objective prepares its batch through the model capability that pins; `legacy_pg` refuses the knob at construction                                |
| Any `n_mini`, including 1                                                                                                                                  | at `n_mini == 1` it is the only mechanism that closes the routing gap; at `n_mini > 1` the pinned pi_old ahead pass passes the probe under the MoE's relaxed gate |
| Pairs with `rl.use_rollout_logprobs_for_loss=true`                                                                                                         | the natural pairing: pi_old is the sampler's own logprob and the training forward routes as the sampler did                                                       |
| Every rollout-logprob mode needs `rollout.temperature == rl.temperature == 1.0`, `rollout.top_p=1.0`, `rollout.top_k<=0`, `rollout.repetition_penalty=1.0` | vLLM returns raw, untempered logprobs, so any warping makes them a different distribution from the policy term; the config check refuses other values             |
| Fresh runs only                                                                                                                                            | the knob and the model's `replay_identity()` join the exact-resume identity, the engine flag the rollout identity                                                 |

Watch `train/routing_replay_pinned_tokens`: with the Qwen3-VL packer (which masks padding) it must
equal the collected batch's prompt + completion token count. Mechanism:
[`../reference/config.md`](../reference/config.md), "Routing replay on a sparse-MoE policy".

## 6. Is it training correctly

Watch, in this order:

1. **`train/grpo_informative_group_count`** — groups whose rewards were not all equal. If this is
   near zero, nothing else matters; the reward is not discriminating.
2. **`train/reward_mean`** — the task metric.
3. **`train/clip_frac`** — how often the clip bound engaged. **Exactly 0.0 by construction when
   `n_mini == 1`**, because the policy has not moved since sampling.
4. **`optim/zero_grad_skips_total`** — updates skipped on an exactly-zero gradient. Rising counts
   are expected when groups tie, not a fault.
5. **`train/grad_norm`, and with `rl.kl_beta > 0` `train/kl_mean` and `train/kl_max`** — all three
   are on the per-step console line (`step N/M train/loss_avg … train/grad_norm …`). `grad_norm` is
   the total norm `clip_grad_norm_` measured BEFORE clipping, so it shows the size of the update the
   clip had to cut down. A step where `kl_max` is orders of magnitude above `kl_mean` and `grad_norm`
   jumps by orders of magnitude is one token's KL pull, not learning: the k3 gradient coefficient
   `1 − exp(logπ_ref − logπ_θ)` is unbounded inside the ±20 nat clamp, and on a sparse-MoE policy the
   reference and policy forwards disagree on a few tokens through expert-routing nondeterminism
   alone. Use `rl.kl_coeff_clamp`
   (5.0) bounds that coefficient at `exp(c) − 1` while the logged values stay the exact k3; leaving
   it unset with a positive `kl_beta` is allowed and warns at config load.
6. **`train/rollout_truncated_frac`** with `train/response_len_mean` / `train/response_len_max` —
   how often the engine cut a completion at `rollout.max_tokens`, and how long responses run. A
   fraction that climbs during training is length inflation meeting the budget; raise the budget or
   set `rl.mask_truncated_completions=true` so the cut completions stop steering the update.

Full table with aggregation semantics: [`../reference/metrics.md`](../reference/metrics.md).

## 7. Checkpoints

Shared mechanics are in [`../reference/checkpoint.md`](../reference/checkpoint.md). GRPO specifics:
`rl.save_every` counts **collected batches**, not optimizer steps, and resume requires the restored
optimizer step to be an exact multiple of `rl.n_mini` — a checkpoint that does not sit on a
collected-batch boundary is rejected rather than silently resumed mid-batch.

### Provenance

Which framework code trained a run is not recoverable from the run directory by itself: one
`<run_ts>` directory accumulates every requeue and warm continuation, and the tree a batch script
puts on `PYTHONPATH` need not be the checkout whose `git log` you are reading. The GRPO entry
therefore stamps every attempt on rank 0, right after `config_resolved.yaml`
(`utils/entry.py::dump_resolved(..., provenance=True)` → `utils/provenance.py`):

- `<run_dir>/provenance.<attempt_ts>.json` — one file per attempt, never overwritten (a same-second
  collision gets a `-1`, `-2`, … suffix);
- the same mapping embedded as the top-level `provenance` key of `config_resolved.yaml`, which each
  attempt rewrites, so the YAML always describes the latest attempt.

Fields: `tree_hash` — sha256 over the sorted `(relative path, sha256)` list of every `*.py` under
`cosmos_framework/posttrain`, the identity of the code that actually ran, independent of git — with
`tree_files`; `git_commit` and `git_dirty` (`git -C <repo root> rev-parse HEAD` and
`status --porcelain`, each time-boxed to 5 s; `null` when git cannot answer, e.g. a broken or
absent checkout); `python_version`, `torch_version`, `argv`, `cwd`, `env` (`I4_ROOT` and
`PYTHONPATH` when set), `hostname`, `slurm_job_id`, `timestamp`, `attempt_ts`, `repo_root`,
`posttrain_root`. A probe that fails leaves `null` in its field and the reason under `errors`.

The stamp is best effort: it cannot fail or block a run. The source-tree walk and file reads run in
an isolated subprocess with an outer timeout, the git probes have their own timeouts, and all probes
share an 8 s budget. A stamp that could not be written is a warning in the log, not an error. To
check whether a checkout is the code that trained a run, compare its hash with the record:

```bash
python -c "from cosmos_framework.posttrain.utils import provenance as p; print(p.hash_tree(p.POSTTRAIN_ROOT))"
```

## 8. Failure semantics and limits

- **A non-finite update is fatal, not skipped.** The loop raises on
  `optim/update_outcome == "skipped"` rather than continuing.
- **Policy cells must agree.** If replicas disagree on whether the objective needs the frozen
  pi_old pass, the run aborts rather than training some cells clipped and others unclipped.
- **`POLICY_REPLICAS > 1` is unsupported** (see §4).
- **`legacy_pg` cannot score multimodal inputs.** Use the clipped surrogate for VLM runs.
- **Rollout logprobs are opt-in for the loss** (`rl.use_rollout_logprobs_for_loss`) and off by
  default. The vLLM rollout records raw logprobs (temperature 1, before any top-p/top-k warp), so
  the mode additionally requires `rollout.temperature == rl.temperature == 1.0` on top of the
  no-warping rule every on-policy GRPO run gets ([`reference/config.md`](../reference/config.md));
  even then the ratio pairs the inference engine's forward with the training forward.
  `train/logprob_gap_*` reports that train/infer gap
  whenever rollout logprobs are recorded ([`reference/metrics.md`](../reference/metrics.md)).
