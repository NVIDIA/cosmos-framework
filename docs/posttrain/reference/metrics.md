# Metrics reference

**The primary metrics** posttrain reports, what emits them, and how they aggregate. This page covers
the objective metric adapters, trainer outcome keys, PPO role receipts, and the GRPO reward-boundary
report — the numbers you read to decide whether a run is healthy.

It is deliberately **not exhaustive**. Not covered here: NFT's `rollout/reward_group_*` family and
`timer/source_*` stage timings (`entrypoints/nft.py`), and the reward-role provider provenance,
which is log output rather than a metric. For anything absent, the adapters named below are the
authoritative list. Keys are extracted from the
code that defines them, not from prose: the objective metric adapters
([`algorithm/grpo.py`](../../../cosmos_framework/posttrain/algorithm/grpo.py), [`dpo.py`](../../../cosmos_framework/posttrain/algorithm/dpo.py),
[`opd.py`](../../../cosmos_framework/posttrain/algorithm/opd.py), [`nft.py`](../../../cosmos_framework/posttrain/algorithm/nft.py),
[`ppo/objective.py`](../../../cosmos_framework/posttrain/algorithm/ppo/objective.py), [`flow_bc.py`](../../../cosmos_framework/posttrain/algorithm/flow_bc.py)), the trainer
([`roles/trainer/base.py`](../../../cosmos_framework/posttrain/roles/trainer/base.py),
[`llm.py`](../../../cosmos_framework/posttrain/roles/trainer/llm.py)), and the reward reporter
([`utils/batch_stats.py`](../../../cosmos_framework/posttrain/utils/batch_stats.py)).

## How a metric gets its name

Objectives emit algorithm-local names (`grpo_loss`). A `MetricAdapter`
([`algorithm/primitives/objective.py`](../../../cosmos_framework/posttrain/algorithm/primitives/objective.py)) maps each to a
public leaf and prefixes it with a phase, producing `train/loss_avg` or `val/loss_avg`. The adapter
declares:

| Field             | Meaning                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `internal_prefix` | The algorithm-local prefix stripped before publication (`grpo_`, `dpo_`, `opd_`, `nft_`, `ppo_`, `flow_bc_`)               |
| `names`           | Source name to public leaf. Leaves may not contain `/` — the phase supplies it                                             |
| `phases`          | Subset of `{"train", "val"}`. DPO and flow-BC publish both                                                                 |
| `weighted_by`     | Which metric weights this one when several chunks are reduced into one window                                              |
| `max_reduced`     | Source names whose contributions compose by MAX across ranks and chunks (a per-shard maximum); disjoint from `weighted_by` |
| `ignored`         | Source keys deliberately not published                                                                                     |

**Aggregation is not always a mean.** A key listed in `weighted_by` is reduced as
`sum(value * weight) / sum(weight)` across chunks and data-parallel ranks, so the reported number
is a true global figure rather than a mean of per-rank fractions. A key absent from `weighted_by`
is a plain mean, except the integer counts `train/valid_sequence_count`, `train/valid_token_count`,
`train/logprob_gap_token_count` and `train/response_count` (`SUMMED_TRAIN_COUNT_KEYS`), which are
summed across ranks ([`roles/trainer/llm.py`](../../../cosmos_framework/posttrain/roles/trainer/llm.py), `_reduce_metrics`) and
across the chunks of an optimizer window together with every weight key
([`utils/report.py`](../../../cosmos_framework/posttrain/utils/report.py), `merge_training_metric_window`), and the maxima — the
keys an adapter lists in `max_reduced`, the one declaration point (GRPO's `train/kl_max` and
`train/response_len_max`) — which take the MAX over ranks (one extra `all_reduce(MAX)` per step,
only when such a key exists, packed in float64 so an integer extremum comes back an int) and over
the chunks of the window.

## Shared across algorithms

| Key                          | Meaning                                                                                                                                              | Aggregation                                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `train/loss_avg`             | The objective's reduced loss on DPO/GRPO/OPD/NFT/flow-BC paths; PPO publishes separate policy and critic losses below                                | Weighted by `weight_sum` for GRPO and OPD; plain mean for DPO, NFT, and flow_bc                                |
| `train/weight_sum`           | The DP-global valid-unit divisor the objective normalized by                                                                                         | Sum                                                                                                            |
| `train/valid_sequence_count` | Sequences that carried learnable signal                                                                                                              | Sum                                                                                                            |
| `train/valid_token_count`    | Completion tokens that carried learnable signal                                                                                                      | Sum                                                                                                            |
| `train/grad_norm`            | Total gradient norm as `clip_grad_norm_` measured it BEFORE clipping (the value the clip compares against `max_grad_norm`); on the GRPO console line | Last chunk of the window (the applied update); emitted only when `max_grad_norm > 0` or `skip_zero_grad` is on |
| `train/learning_rate`        | The LR the applied update used                                                                                                                       | Plain mean                                                                                                     |
| `timer/e2e_step_s`           | Wall time for one collected batch, end to end                                                                                                        | Plain mean                                                                                                     |

## GRPO

Source: `_GRPO_PUBLIC_METRICS` and `_GRPO_METRIC_ADAPTER`.

| Key                                             | Meaning                                                                                                                                                                                                                                                                                                                                   | Aggregation                                                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `train/clip_frac`                               | Fraction of tokens where the epsilon clip bound the update                                                                                                                                                                                                                                                                                | **Token-weighted** by `valid_token_count`, so the aggregate is total clipped tokens over total tokens |
| `train/ratio_absdev`                            | Mean absolute deviation of the importance ratio from 1                                                                                                                                                                                                                                                                                    | Token-weighted                                                                                        |
| `train/seq_clip_frac`                           | Fraction of **responses** the clip touched — the sequence-level counterpart of `clip_frac`, added with GSPO                                                                                                                                                                                                                               | Weighted by `valid_sequence_count`                                                                    |
| `train/seq_log_ratio_abs`                       | Mean absolute sequence-level log ratio                                                                                                                                                                                                                                                                                                    | Weighted by `valid_sequence_count`                                                                    |
| `train/routing_replay_pinned_tokens`            | Token rows dispatched through the pinned routing: every prompt + completion position (the last one through the filler selection, since the engine never fed it back), padding excluded when the packer supplies `attention_mask` (Qwen3-VL packer) and included otherwise; emitted by `LLMTrainer` only under `rl.routing_replay=rollout` | Sum over ranks and over the chunks of a collected batch                                               |
| `train/routing_replay_pinned_blocks`            | Sparse-MoE blocks pinned per token (a constant of the model); same emission rule                                                                                                                                                                                                                                                          | Plain mean                                                                                            |
| `train/logprob_gap_mean`                        | Mean over compared tokens of \|log π_train − log π_rollout\| in nats: the training forward's temperature-1 logprob of each sampled completion token against the logprob the rollout engine recorded for it                                                                                                                                | Weighted by `logprob_gap_token_count`                                                                 |
| `train/mean_rank_logprob_gap_p99`               | 99th percentile of the same per-token gap on each rank and chunk. Quantiles do not compose, so the published value is the token-weighted mean of those p99s, which the name spells out                                                                                                                                                    | Weighted by `logprob_gap_token_count`                                                                 |
| `train/logprob_gap_gt0p5_frac`                  | Fraction of compared tokens whose gap exceeds 0.5 nats                                                                                                                                                                                                                                                                                    | Weighted by `logprob_gap_token_count`                                                                 |
| `train/logprob_gap_token_count`                 | Tokens the gap was measured over: every loss-masked completion token that carried a finite recorded rollout logprob. 0 when the producer records none                                                                                                                                                                                     | Sum                                                                                                   |
| `train/kl_mean`                                 | Token mean of the unscaled K3 penalty `exp(z) − z − 1`, `z = logπ_ref − logπ_θ`, toward the frozen reference (`rl.kl_beta > 0`); exactly `0.0` without one. The value is the exact k3 whether or not `rl.kl_coeff_clamp` bounds the gradient                                                                                              | Token-weighted by `valid_token_count`                                                                 |
| `train/kl_max`                                  | Largest per-token K3 penalty in the batch, same scale as `kl_mean`. Orders of magnitude above `kl_mean` together with a `grad_norm` jump is the one-token-owns-the-step signature that motivated `rl.kl_coeff_clamp`                                                                                                                      | **Max** over ranks and chunks (`max_reduced`)                                                         |
| `train/kl_beta`                                 | The configured `rl.kl_beta`, echoed so a curve says which anchor strength produced it                                                                                                                                                                                                                                                     | Plain mean (constant)                                                                                 |
| `train/response_count`                          | Completions the trained batch handed the objective, counted before an empty completion or `rl.mask_truncated_completions` drops one                                                                                                                                                                                                       | Sum                                                                                                   |
| `train/rollout_truncated_frac`                  | Fraction of those completions the rollout engine cut at its token budget — `policy_info["finish_reason"] == "length"`, stamped by the vLLM rollout; `0.0` when the producer stamps no finish reason (HF and local rollouts)                                                                                                               | Weighted by `response_count`, so the aggregate is cut completions over all completions                |
| `train/response_len_mean`                       | Mean generated length of those completions in tokens (`policy_info["completion_token_count"]`)                                                                                                                                                                                                                                            | Weighted by `response_count`                                                                          |
| `train/response_len_max`                        | Longest generated completion of the step in tokens; it sits at `rollout.max_tokens` whenever anything was cut                                                                                                                                                                                                                             | **Max** over ranks and chunks (`max_reduced`), never a mean of maxima                                 |
| `train/sc_token_count`                          | Loss-masked completion tokens the score-centering correction covered (`rl.score_centering.enabled`); 0 with zero-valued `sc_*` keys when centering is off                                                                                                                                                                                 | Sum                                                                                                   |
| `train/sc_head_mass_q` / `train/sc_head_mass_p` | Mean over those tokens of the sampler's / the trainer's probability mass inside the sampler's top-k head, `Σ_H q` / `Σ_H p`. How much of the sampling distribution the head form sees (the reference reports ≥ 0.99 at k = 128)                                                                                                           | Weighted by `sc_token_count`                                                                          |
| `train/sc_rho_mean` / `train/sc_rho_max`        | Tail-mass ratio `ρ = q_tail / p_tail` (both floored at 1e-6): its token mean and the largest value of the step. `1.0` when sampler and trainer agree; a large max is a confident trainer against an unsure sampler, where the floor bites                                                                                                 | Weighted by `sc_token_count` / **Max** (`max_reduced`)                                                |
| `train/sc_kl_head`                              | Token mean of `Σ_H q log(q/p)`: the sampler-vs-trainer disagreement on the head at the SAME temperature — the mismatch the correction removes                                                                                                                                                                                             | Weighted by `sc_token_count`                                                                          |
| `train/sc_correction_abs`                       | Token mean of the absolute correction term `\|Σ_H sg[q w − α p] log p\|`; exactly 0 when `q == p`                                                                                                                                                                                                                                         | Weighted by `sc_token_count`                                                                          |
| `train/sc_token_weight`                         | Token mean of the composed sampled-token weight `f(p_{y_t}/q_{y_t})`; exactly 1.0 under `compose=none`                                                                                                                                                                                                                                    | Weighted by `sc_token_count`                                                                          |
| `train/sc_out_of_head_frac`                     | Fraction of sampled tokens that fall outside the sampler's own top-k head (their logprob is still used, per the reference's rule)                                                                                                                                                                                                         | Weighted by `sc_token_count`                                                                          |

The `seq_*` pair is what you watch under `rl.importance_sampling_level=sequence` (GSPO), where the
ratio is formed per response rather than per token; the per-token `clip_frac` still reports, but the
sequence-level rate is the one the objective acts on.

**The `logprob_gap_*` family is the train/infer consistency instrument**
([`algorithm/primitives/logprob_gap.py`](../../../cosmos_framework/posttrain/algorithm/primitives/logprob_gap.py)). It is emitted
whenever the trajectories carry `policy_info["rollout_logprobs"]` (the vLLM rollout always records
them; the HF rollout only when constructed with `HFRollout(capture_logprobs=True)`, which the config
does not expose), independently of `rl.use_rollout_logprobs_for_loss`. Without recorded logprobs the
three gap keys are `0.0` and carry zero weight, and `logprob_gap_token_count` is `0`; read the count
to tell "no gap" from "no data". Presence is all-or-nothing per batch: a batch where only some
trajectories carry a record, or a record shorter than its completion, is a producer bug and fails
the step on every rank, exactly as the loss mode does.
The rollout engine records raw logprobs (temperature 1, before top-p/top-k), so the training side
is compared at temperature 1 even when `rl.temperature` is not 1: the metric measures the
discrepancy between the two forwards, not the sampling temperature. On a dense model the gap is
kernel and precision noise; a rising gap on an MoE policy is the routing-flip collapse signature
reported by GSPO (§5.3) and Rollout Routing Replay (§3), and a large gap is what makes rollout
logprobs unusable as π_old. The HF rollout records processed scores, so with `rollout.temperature`
away from 1 its gap includes the temperature effect.

**The `response_*` / `rollout_truncated_frac` family is the length-budget instrument.** A rising
`train/rollout_truncated_frac` means responses are running into `rollout.max_tokens`; each cut
completion is scored on an answer it never finished and, under the token-sum reductions, carries
`max_tokens / median_length` times a median response's gradient mass. The family counts every
completion the objective was handed, so it reads the same with `rl.mask_truncated_completions` on
(the cut completions leave the loss) or off (they train), and tells the two apart only through
`train/valid_sequence_count`, which excludes them under the mask.

**`clip_frac` and `ratio_absdev` are 0.0 by construction at `n_mini == 1`** — with one optimizer
step per collected batch the policy has not moved since sampling, so the ratio is identically 1 and
nothing clips. A nonzero value there would indicate a bug, not tighter training. See
[`howto/batching.md`](../howto/batching.md) for why `n_mini > 1` requires the frozen pi_old pass.

## Reward and grouping (GRPO reporting path)

Source: `utils/batch_stats.py`, emitted when the entry calls `RewardWorker.score_with_report`.

| Key                                                              | Meaning                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train/reward_mean` / `reward_std` / `reward_min` / `reward_max` | Sample-weighted reward statistics for the batch                                                                                                                                                                                           |
| `train/rollout_count`                                            | Trajectories scored                                                                                                                                                                                                                       |
| `train/grpo_group_count`                                         | Groups scored this step; with dynamic sampling on (`rl.min_informative_groups > 0` or `rl.min_informative_groups_frac` set) this covers every draw of the step (base plus extra rollouts), not only the groups that were trained on       |
| `train/grpo_informative_group_count`                             | Groups whose rewards were **not** all equal — the only groups that produce a nonzero advantage                                                                                                                                            |
| `train/grpo_tied_group_count`                                    | Groups whose rewards tied                                                                                                                                                                                                                 |
| `train/grpo_all_zero_group_count` / `grpo_all_one_group_count`   | Groups scoring uniformly 0 or 1                                                                                                                                                                                                           |
| `train/grpo_dynamic_sampling_extra_rollouts`                     | Extra refill rounds the step spent topping up informative groups; present only when dynamic sampling is on. This is the per-shard cursor advance in rounds under both dispatch values.                                                    |
| `train/grpo_dynamic_sampling_refill_dispatches`                  | `rollout.generate` calls beyond the base draw: up to the round budget under `per_round`, and 0 or 1 under `whole_budget`; present only when dynamic sampling is on.                                                                       |
| `train/grpo_dynamic_sampling_idle_cells`                         | Sum of deliberately idle cells across extra refill rounds; zero for the default `all` mode                                                                                                                                                |
| `train/grpo_dynamic_sampling_batch_informative_groups`           | Informative groups in the trained batch: each cell's informative-first share, summed; compare with the resolved quota                                                                                                                     |
| `train/grpo_dynamic_sampling_avoided_prompts`                    | Prompt draws skipped by selective refills: idle cell-rounds times the per-cell prompt count; this counts avoided work, not measured time savings                                                                                          |
| `train/grpo_dynamic_sampling_dropped_trajectories`               | Scored trajectories the step collected but did not train on (tied groups first, then over-collected informative ones); present only when dynamic sampling is on (`rl.min_informative_groups > 0` or `rl.min_informative_groups_frac` set) |

`train/grpo_informative_group_count` is one of the headline keys the GRPO loop prints on its
per-step console line (with `train/loss_avg`, `train/reward_mean` and the pre-clip `train/grad_norm`,
plus `train/kl_mean` and `train/kl_max` when `rl.kl_beta > 0`). It is the metric to watch first: a batch whose
groups all tie contributes no gradient signal at all, because a group with equal rewards has an
advantage of exactly zero for every member.

With dynamic sampling on, the grouping counts span every draw of the step (base plus top-ups), so they
describe what was generated, not what was trained; the `grpo_dynamic_sampling_*` keys give the difference.

## PPO

Source: `algorithm/ppo/objective.py`, `algorithm/ppo/batch.py`, and
`roles/trainer/ppo.py`. PPO publishes one frame after both policy and critic update windows finish.

| Keys                                                                                                                     | Meaning                                                                                  | Aggregation                                              |
| ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `train/policy_total_loss`, `train/policy_loss`                                                                           | Total and clipped-surrogate policy losses                                                | Token-weighted across every microbatch and model-DP rank |
| `train/clip_fraction`, `train/ratio_mean`                                                                                | PPO policy clipping rate and mean importance ratio                                       | Token-weighted                                           |
| `train/entropy`, `train/jsd_loss`                                                                                        | Optional entropy and reference-JSD terms                                                 | Token-weighted                                           |
| `train/value_loss`, `train/critic_loss`, `train/value_clip_fraction`                                                     | Unscaled value loss, coefficient-scaled critic loss, and critic clipping rate            | Token-weighted                                           |
| `train/reward_mean`, `train/advantage_mean`, `train/advantage_std`                                                       | Prepared batch reward and normalized policy-advantage statistics                         | Over valid action tokens in the immutable prepared batch |
| `train/value_advantage_mean`, `train/return_mean`, `train/value_mean`                                                    | GAE/value-target and old-value statistics                                                | Over valid action tokens                                 |
| `train/reference_kl_mean`, `train/valid_action_tokens`                                                                   | Sampled reference KL and number of valid action tokens                                   | Batch mean and count, respectively                       |
| `train/behavior_logprob_delta_*`, `train/initial_ratio_*`, `train/initial_clip_fraction`, `train/initial_importance_ess` | Pre-update policy-versus-rollout consistency diagnostics                                 | Reduced across the complete policy model-DP world        |
| `train/reference_behavior_logprob_delta_*`, `train/policy_reference_logprob_delta_*`                                     | Reference-versus-rollout and policy-versus-reference consistency diagnostics             | Reduced across the complete policy model-DP world        |
| `train/policy_lr`, `train/critic_lr`                                                                                     | Post-step learning rates left for the next optimizer update                              | Final value from each role window                        |
| `train/actor_warmup`                                                                                                     | `1.0` on a critic-only warmup iteration (`rl.critic_warmup_iterations`), `0.0` otherwise | Exact per-iteration phase flag                           |
| `optim/policy_optimizer_step`, `optim/critic_optimizer_step`                                                             | Durable minibatch optimizer counters                                                     | Final exact counter from each role receipt               |

PPO's outer `policy_version` advances once after all configured policy and critic minibatches for a
collected batch succeed. The two optimizer counters therefore need not equal the outer version, but
policy and critic counters must advance by the same configured number of minibatches.

The frame's step is the OUTER ITERATION, not `policy_version`; the two coincide only when
`rl.critic_warmup_iterations` is `0`. While `train/actor_warmup` is `1.0` the actor runs no update
window at all, so that iteration publishes no policy receipt: every `train/policy_*` key,
`train/clip_fraction`, `train/ratio_mean`, `train/entropy`, `train/jsd_loss` and
`optim/policy_optimizer_step` are ABSENT from the frame rather than repeated or zero-filled. The
critic keys and the prepared-batch/consistency diagnostics are published exactly as usual.

## DPO

Source: `_DPO_METRIC_NAMES`. Like flow-BC below, it publishes both `train/` and `val/`.

| Key                                                                         | Meaning                                               |
| --------------------------------------------------------------------------- | ----------------------------------------------------- |
| `train/dpo_reward_margin`                                                   | Implicit reward gap between winner and loser          |
| `train/dpo_reward_winner` / `dpo_reward_loser`                              | The two implicit rewards separately                   |
| `train/dpo_accuracy`                                                        | Fraction of pairs the policy ranks correctly          |
| `train/dpo_err_pol_w` / `dpo_err_ref_w` / `dpo_err_pol_l` / `dpo_err_ref_l` | Denoising error for policy and reference on each side |
| `train/dpo_raw_model_loss` / `dpo_raw_ref_loss`                             | Unreduced losses before the preference combination    |

`dpo_accuracy` is the readable health signal: it starts near 0.5 and should rise.
`OBJECTIVE_WEIGHT_SUM` is in this adapter's `ignored` set, so DPO publishes no `weight_sum`.

## OPD

Source: `_OPD_PUBLIC_METRICS` plus two adapters — `_OPD_DIRECT_METRIC_ADAPTER` for same-token
direct OPD and `_OPD_ADVANTAGE_METRIC_ADAPTER` for the advantage form. The direct form publishes no
advantage or teacher-credit keys, because it forms neither.

| Key                        | Meaning                   | Adapter        | Aggregation    |
| -------------------------- | ------------------------- | -------------- | -------------- |
| `train/opd_divergence`     | Policy-teacher divergence | Both           | Token-weighted |
| `train/opd_advantage`      | Advantage term            | Advantage only | Token-weighted |
| `train/opd_teacher_credit` | Teacher-supplied credit   | Advantage only | Token-weighted |

## NFT

Source: the adapter built in `make_objective` in `algorithm/nft.py`. Phase `train` only; reduction
is `MEAN`, not `TOKEN_SUM`.

| Key                                                        | Meaning                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| `train/policy_loss` / `train/kl_loss`                      | The two loss components                                             |
| `train/advantage_mean`                                     | Mean advantage                                                      |
| `train/optimality_mean` / `train/optimality_second_moment` | First and second moment of the optimality ratio                     |
| `train/optimality_std`                                     | Standard deviation of the optimality weights (`entrypoints/nft.py`) |
| `train/positive_error` / `train/negative_error`            | Error on the positive and negative branches                         |
| `train/reward_mean`                                        | Mean reward                                                         |

## Flow-BC

Source: `_METRIC_ADAPTER` in `algorithm/flow_bc.py` (GR00T-N1.7 behaviour cloning). Phases `train` and `val`, same
keys under `val/`.

| Key                             | Meaning                                                                                                                                                                                                                                        | Aggregation |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `train/loss_avg`                | The model's masked-mean flow-matching loss rescaled by this rank's supervised count over the DP-mean supervised count (`out.loss * (masked / denom)`): equal shards stay bitwise identical, unequal shards normalize to the global masked mean | Plain mean  |
| `train/flow_bc_masked_elements` | Supervised action elements on this rank, the sum of `action_mask`                                                                                                                                                                              | Sum         |

Flow-BC does not publish `train/weight_sum`: the objective's `OBJECTIVE_WEIGHT_SUM` (the DP-mean count, the
trainer-facing divisor) sits in the adapter's `ignored` set because publicly it would only duplicate
`train/flow_bc_masked_elements`.

## Optimizer outcome keys

Source: `BaseTrainer._outcome_metrics` and `_skip_nonfinite`. These explain a run that looks
stalled, and are the first place to look when the loss stops moving.

| Key                                                           | Values / meaning                                                                                                                                                                    |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optim/update_outcome`                                        | `accumulating` (mid gradient accumulation) · `applied` (optimizer stepped) · `skipped` (rejected, see `skip_reason`) · `discarded` (chunk dropped by the poisoned-window countdown) |
| `optim/skip_reason`                                           | `nonfinite_loss` or `nonfinite_gradient`                                                                                                                                            |
| `optim/step_skipped`                                          | Whether this attempt skipped                                                                                                                                                        |
| `optim/attempt_id`                                            | Monotonic id of the update attempt                                                                                                                                                  |
| `optim/optimizer_step`                                        | Optimizer steps committed so far                                                                                                                                                    |
| `optim/nonfinite_skips_total` / `nonfinite_skips_consecutive` | Non-finite rejections, cumulative and in a row                                                                                                                                      |
| `optim/zero_grad_skips_total`                                 | Updates skipped because the gradient was exactly zero                                                                                                                               |
| `optim/finite_gradient_check_enabled`                         | Whether `force_finite_grads` is on                                                                                                                                                  |

**A rising `optim/zero_grad_skips_total` is expected in GRPO, not a fault.** A group whose rewards
tie yields a zero advantage for every member, and stepping on a zero gradient would still apply
weight decay and advance the Adam moments — moving the weights on a batch that said nothing. The
trainer therefore skips it. Read this key together with
`train/grpo_informative_group_count`: if informative groups are near zero, the reward function is
not discriminating between samples, which is a data or reward problem, not an optimizer problem.

`optim/update_outcome == "skipped"` at the GRPO entry is fatal rather than tolerated — the loop
raises rather than continuing on a rejected update.

## Validation phase

Source: `control/validation/`. Emitted only when `rl.validation` is configured; see
[`howto/train-time-validation.md`](../howto/train-time-validation.md).

| Key                                 | Meaning                                       |
| ----------------------------------- | --------------------------------------------- |
| `val/loss_avg`                      | Held-out objective loss (DPO)                 |
| `val/reward_avg` / `val/reward_std` | Held-out rollout reward (GRPO-family and OPD) |

## Healthy ranges

Only two values are established by the code or by a recorded measurement:

- `train/clip_frac` and `train/ratio_absdev` are exactly `0.0` when `n_mini == 1`.
- Computing logprobs in bf16 rather than fp32 was measured to drop the GRPO clip rate from 0.01493
  to 0.00089 at `n_mini=8` — a 17x difference — which is why `logprob_dtype` defaults to fp32.

One reference trajectory is recorded, from the qualified 300-step Qwen2.5-3B GSM8K GRPO run on two
80 GB H100s: `train/reward_mean` went from **0.8400** averaged over the first
50 steps to **0.9264** over the last 50. That is one model on one task, so treat it as a sanity
reference for the shape of the curve, not as a target.

For every other key, no healthy range is established. Compare against your own baseline run rather
than an absolute number.
