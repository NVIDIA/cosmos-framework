# Algorithm objective customization

**Come here when the change you need is mathematical rather than data or reward.** An objective is
the trainer-facing definition of an algorithm's loss.

Read the decision example first: Dr-GRPO changes GRPO's advantage and reduction and is still
*config-selected*, not a new objective. Most variants land there. Used by
[`../guides/grpo.md`](../guides/grpo.md) and [`../guides/opd.md`](../guides/opd.md).

An objective is the trainer-facing definition of an algorithm's loss. It consumes a model and a
prepared batch, returns a scalar loss and metrics, and declares how that loss composes across data
parallel ranks and gradient-accumulation chunks.

Customizing an objective does not require a new trainer when the existing trainer, data contract,
and execution recipe already provide everything the loss needs. The algorithm entry selects an
objective builder; the framework continues to own model construction, optimizer setup,
checkpointing, placement, data transport, validation, and the control loop.

This guide first describes the general objective contract, then shows the current online
objective-builder seam. Dr-GRPO is the decision example for a variant that stays config-selected
because the shared GRPO objective already owns all of its math. GSPO is the second: it changes the
clipped surrogate's ratio granularity through a sub-knob on the surrogate that owns the ratio.

## Mental model and ownership

The normal path is:

```text
entry + resolved config
        |
        v
objective builder ---------------> Objective
                                      |
prepared Trajectory batch ------------+--> trainer --> backward/optimizer
```

Keep each concern at its owning layer:

| Concern                                                               | Owner                                                                                                                                                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensor loss computation                                               | `algorithm/` objective or reusable loss primitive                                                                                                                                 |
| Config-to-objective composition                                       | Entry-level objective builder, optionally delegating to `config/build.py`                                                                                                         |
| Group statistics and annotations                                      | Reward/data stage before dispatch                                                                                                                                                 |
| Model, optimizer, checkpoint, precision                               | Trainer assembly                                                                                                                                                                  |
| Model forward configuration (routing replay, forward-noise tolerance) | A model capability in `model/protocol.py`, consumed from the policy batch by the model adapter and read by the trainer; never an `Objective` field or an import of a model module |
| Role topology and data/weight edges                                   | Entry manifest and placement assembly                                                                                                                                             |
| Phase ordering                                                        | Entry control loop                                                                                                                                                                |

The objective is not a second control plane. It must not launch roles, move weights, fetch remote
data, save checkpoints, or decide physical placement.

## The `Objective` contract

The shared contract is
[`algorithm/primitives/objective.py`](../../../cosmos_framework/posttrain/algorithm/primitives/objective.py):

```python
Objective(
    fn,
    reduction=...,
    old_logp_cache=...,
    metric_adapter=...,
    identity=...,
)
```

`fn(model, batch)` returns:

```python
loss, metrics
```

- `loss` is a scalar tensor connected to the model graph.
- `metrics` contains algorithm-local diagnostics plus any control metric required by the declared
  reduction.
- `reduction` tells the trainer how to combine loss across accumulation chunks.
- `metric_adapter` maps algorithm-local metric names to stable `train/*` and `val/*` names.
- `old_logp_cache` is optional online-policy state used when a behavior policy must remain frozen
  across multiple updates from one collected batch.
- `reference_scorer` is an optional scorer of a frozen reference's per-token logprobs (GRPO's KL
  anchor). The objective consumes scores through it only; the network behind it is built by the
  entry and handed to the trainer for device residency (`LLMTrainer(auxiliary_models=...)`), so the
  trainer never reads a module off the objective.
- `identity` records objective-specific values that should participate in objective provenance.

### Reduction is part of the algorithm

The two current reduction modes are:

- `Reduction.MEAN`: the loss is already a sample mean; the trainer scales it by gradient
  accumulation.
- `Reduction.TOKEN_SUM`: the loss is a numerator divided by an objective-defined global weight.
  The objective must report that weight as `OBJECTIVE_WEIGHT_SUM`. The trainer recovers each
  chunk's numerator, accumulates numerators and weights, then normalizes once at the optimizer
  boundary.

`TOKEN_SUM` is a historical name. Its weight may count tokens, sequences, or another valid unit.
What matters is that the loss and weight use the same unit. Returning a changed loss while retaining
an incompatible weight produces a silent mean-of-means error under micro-batching.

### Metrics are part of the public behavior

If a wrapper changes a reported loss, it must update the corresponding metric. Preserve the base
objective's metric adapter unless the new objective deliberately defines another public metric
contract. A new internal metric prefix must be completely mapped or explicitly ignored; otherwise
metric adaptation fails rather than silently leaking inconsistent names.

### Objectives consume prepared data

An objective may validate required annotations but should not derive group- or source-level state
from its local batch. Dispatch may split a rollout group across ranks, mini-batches, or
micro-batches. Group-relative advantage, shared-prefix filtering, and similar computations must run
while the complete group is still present, before trainer dispatch. See
[`algorithm/annotators/advantage.py`](../../../cosmos_framework/posttrain/algorithm/annotators/advantage.py).

The same rule applies to large payloads: metadata determines routing, while tensor/media bulk is
materialized through the data plane before the objective consumes it. The objective should not
open a parallel transport path.

## Where customization code belongs

Use the smallest layer that honestly owns the change:

| Change                                                 | Implementation shape                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------- |
| Existing reduction or annotator combination            | Resolved recipe config                                            |
| Small transform not expressible by existing primitives | Small entry-level wrapper                                         |
| New reusable loss math                                 | Function/module under `algorithm/`; entry builder delegates to it |
| New objective assembled from existing primitives       | Builder in the algorithm entry                                    |
| New user-configurable value                            | Typed config field with one default and validation owner          |
| Different choice among existing annotators             | Resolved entry config                                             |
| New group-wide annotation                              | New/reused reward-stage annotator plus its assembly wiring        |
| New teacher or model-bearing role                      | Variant-specific entry topology, as in OPD                        |
| Different phase ordering                               | Variant-specific loop that reuses compatible components           |

Do not add a config class merely to rename existing fields, and do not place substantial algorithm
math in an entry. Entries should remain readable config-to-component compositions.

## Current entry integration surfaces

The `Objective` contract is framework-wide, but objective selection is currently explicit in each
entry rather than selected from a global registry by name.

### Online GRPO-family entries

The shared online trainer accepts a top-level builder with this signature:

```python
def build_objective(rl: Any, model_config: Any) -> Objective:
    ...
```

The builder receives the resolved algorithm config and a trainer-owned context containing the
effective policy/tokenizer identity. The GRPO builder also consumes the rollout response bound from
that context when its fixed-response reduction is selected.
`grpo.run` carries it through worker setup to the shared `build_trainer`:

```python
grpo.run(
    config,
    config_path,
    opts,
    out_dir,
    build_objective_fn=build_objective,
)
```

The builder is required. There is no algorithm-specific runner flag and no fallback that silently
selects GRPO when an entry forgets its own objective.

Keep the builder at module scope. Ray workers are separate processes; nested functions, driver
monkeypatches, and import-time registry side effects are not reliable worker customization. A
module-level function, optionally specialized with `functools.partial` over primitive values, is
serializable into policy setup.

The current users of this seam are:

- [`entrypoints/grpo.py`](../../../cosmos_framework/posttrain/entrypoints/grpo.py): selects the configured GRPO objective.
- [`entrypoints/opd.py`](../../../cosmos_framework/posttrain/entrypoints/opd.py): uses the same trainer builder seam while owning an
  additional teacher role and data edge.

### Built-in selectors and migration

GRPO and OPD select different concepts, so they have different fields:

| Config       | Field               | Values                 | Default     |
| ------------ | ------------------- | ---------------------- | ----------- |
| `GRPOConfig` | `grpo_surrogate`    | `clipped`, `legacy_pg` | `clipped`   |
| `OPDConfig`  | `distillation_mode` | `advantage`, `direct`  | `advantage` |

`clipped` is the production GRPO surrogate. It owns frozen π_old, importance ratios, epsilon and
dual clipping, loss temperature, and the configured loss reduction. `legacy_pg` is the minimal
plain policy-gradient path retained for toy/parity use. The default arithmetic root explicitly
pins `legacy_pg`; production recipes such as GSM8K and custom-provider recipes explicitly use `clipped`.

The granularity of that importance ratio is `clipped`'s own sub-knob, `importance_sampling_level`
(`token`, the default, or `sequence` for GSPO), not a third `grpo_surrogate` value; see the GSPO
decision example below.

The selector migration is intentionally one-way:

```text
rl.objective=degenerate       -> rl.grpo_surrogate=legacy_pg
rl.objective=clipped          -> rl.grpo_surrogate=clipped
rl.objective_mode=direct      -> rl.distillation_mode=direct
rl.objective_mode=advantage   -> rl.distillation_mode=advantage
```

The old keys are unknown configuration keys. There are no aliases or precedence rules, so a stale
recipe fails instead of silently selecting a different objective.

### Preflight rejects semantically inactive settings

For the clipped on-policy surrogate, `rollout.temperature` and `rl.temperature` must be equal. The
entry checks this before placement or model construction because sampling from one distribution
and scoring the loss under another is not the declared on-policy algorithm. `legacy_pg` does not
consume `rl.temperature`, so it instead requires `rollout.temperature=1.0` and leaves
`rl.temperature` at its default.

Configuration also fails when a selected objective would ignore a non-default clip, temperature,
loss-reduction, or advantage knob. Direct OPD emits an explicit warning that task reward and group
advantage are not training signals for its divergence loss; reward configuration still controls
reward-stage metrics, held-out evaluation, and optional shared-prefix masking. These checks keep a
resolved config from claiming an algorithm setting that the actual loss never reads.

### Other entries

DPO uses the same `Objective` contract but currently constructs its diffusion objective directly in
`entrypoints/dpo.py::build_trainer`; it does not use the GRPO online builder seam. The general rules
for loss, reduction, metrics, config ownership, and data dependencies still apply, but a DPO
variant must integrate at DPO's actual assembly boundary rather than calling `grpo.run`.

Likewise, an algorithm that changes topology or loop ordering needs its own entry assembly. Reuse
the compatible trainer, objective primitives, role bodies, and loop phases; do not distort a loss
builder into owning orchestration.

## General online customization workflow

### 1. State the mathematical and data differences

Before writing code, list exactly what changes:

- Does the loss formula change?
- Does its normalization unit change?
- Does it need fields already present on `Trajectory`?
- Does it need a new per-sample, per-token, group-level, or teacher-produced annotation?
- Does it require another role or a different phase order?

Only the first three are normally objective-only changes. This classification prevents an
apparently convenient objective from depending on a complete group or an undeclared data edge.

### 2. Reuse primitives before adding math

Look first in:

- `algorithm/primitives/policy_gradient.py` for clipped policy-gradient machinery.
- `algorithm/grpo.py`, `algorithm/opd.py`, and `algorithm/dpo.py` for existing objectives.
- `config/build.py` for the existing config-to-objective mappings.
- `algorithm/annotators/` for annotations already computed before dispatch.

If the variant is a small transformation of a complete existing objective, wrap it. If it has
substantial new math or more than one caller, put that math under `algorithm/` and keep the entry
builder thin.

### 3. Make config the single source of truth

Workers reload `config_path` with the entry's dotlist. Mutating only the driver-side config object
does not change what reward or policy workers later load.

Put fixed algorithm invariants in the entry's worker opts before loading config:

```python
opts = [
    *opts,
    "rl.some_required_mode=value",
]
config = load_config(config_path, ["--", *opts])
```

Append invariants after user options when overriding them would make the entry no longer represent
the named algorithm. Genuine user knobs belong in typed config and remain user-overridable. Avoid a
second field for a value already owned elsewhere, such as a rollout generation bound.

### 4. Build and inject the objective

For an objective with the standard online signature:

```python
def build_objective(rl: Any, model_config: Any) -> Any:
    from cosmos_framework.posttrain.algorithm.my_variant import make_objective

    return make_objective(rl, model_config.name_or_path)
```

Then pass it explicitly:

```python
grpo.run(
    config,
    config_path,
    opts,
    _out_dir(config),
    build_objective_fn=build_objective,
)
```

Do not copy `grpo.make_setup`, `grpo.run`, or `grpo.loop` merely to select a loss.

## Decision example: Dr-GRPO is config-selected

Dr-GRPO keeps the GRPO online topology and control loop. It changes two mathematical choices.

For a completion group with rewards `r_i`, standard GRPO uses:

```text
A_i = (r_i - group_mean) / (group_std + eps)
```

Dr-GRPO uses:

```text
A_i = r_i - group_mean
```

It also normalizes each completion's token-loss sum by the fixed generation bound
`rollout.max_tokens`, rather than its realized response length.

### Reuse existing behavior through config

The reward-stage estimator already supports mean-centred advantages. The clipped GRPO objective
also owns a fixed-response sequence-sum reduction, so the complete variant is a recipe choice:

```python
cfg = build_grpo_config()
cfg.algorithm = "dr_grpo"
cfg.job.name = "qwen3b_bf16_dr_grpo_seed42"
cfg.rl.grpo_surrogate = "clipped"
cfg.rl.unbiased_advantage = True
cfg.rl.loss_type = "seq-mean-token-sum-fixed"
```

The distinct `algorithm` and `job.name` keep resolved config, output paths, W&B provenance, and
automatic resume separate from the base GRPO recipe. No new advantage estimator, trainer, role,
config class, objective wrapper, or loop is needed.

### Keep the fixed divisor inside the reduction primitive

`algorithm/primitives/policy_gradient.py::reduce_policy_gradient_loss` implements
`seq-mean-token-sum-fixed` as:

```text
sum(sequence token losses) / rollout.max_tokens / number of sequences
```

The shared objective still reports the DP-global sequence count as `OBJECTIVE_WEIGHT_SUM`, so split
and unsplit micro-batches use the same numerator and normalization unit. Metrics, π_old caching, and
the metric adapter remain on the single production objective path instead of being copied by a
wrapper.

### Pass the existing rollout bound without duplicating config

`rollout.max_tokens` already owns the fixed response bound. Bind it to the module-level builder
context instead of adding another field under `rl`:

```python
objective_context = SimpleNamespace(
    name_or_path=effective_model_id(config),
    max_response_length=config.rollout.max_tokens,
)
objective = build_objective_fn(rl, objective_context)
```

`RolloutConfig` and the driver preflight both reject bool, fractional, zero, or negative bounds;
the objective factory keeps the same validation as a worker-side backstop.

The complete recipe is
[`configs/dr_grpo_qwen3b_gsm8k.py`](../../../cosmos_framework/posttrain/configs/dr_grpo_qwen3b_gsm8k.py). Run it through the shared
GRPO entry:

```bash
WANDB_MODE=disabled PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/dr_grpo_qwen3b_gsm8k.py -- \
  rl.max_steps=1 rl.save_every=0 rl.resume=off
```

This is intentionally not a customer of the objective-injection seam: a different choice among
existing annotators and reductions belongs in resolved config. OPD remains the concrete seam user
because its teacher signal and objective are genuinely different while trainer assembly is shared.

## Decision example: GSPO is config-selected

GSPO (Group Sequence Policy Optimization) keeps the GRPO topology, control loop, frozen π_old,
epsilon clip, and loss reduction. It changes the granularity of the importance ratio. Token-level
GRPO scores each token of response `i` with its own ratio:

```text
r_t = exp(log π_θ(y_t) − log π_old(y_t))
```

GSPO gives all `L_i` loss-masked tokens of that response one shared ratio:

```text
z_i = (1/L_i) Σ_t (log π_θ(y_t) − log π_old(y_t))
s_i = exp(z_i)
```

so the epsilon clip admits or drops whole responses.

### Keep the ratio inside the shared primitive

`algorithm/primitives/policy_gradient.py::clipped_policy_gradient_loss` accepts
`importance_sampling_level` and `token_counts`; at `"sequence"` it replaces the per-token log-ratio
with `sequence_log_ratio`, the GSPO-token stop-gradient form:

```text
l_t = sg[z_i] + (c_t − sg[c_t])        c_t = log π_θ(y_t)
```

The value is `z_i` on every token of the response, and the gradient reaches only that token's own
logprob. For the one scalar advantage per response that GRPO's estimator produces, this equals the
paper's objective in value, clip decision, and parameter gradient; the two forms differ only for
per-token advantages, which the sequence level refuses. The log-ratio is clamped from above at 10
and not from below: the epsilon clip already zeroes the gradient of a far-off-policy positive
advantage, and with no dual clip this cap is the only bound on a negative one. The token path is
unchanged, so `"token"` is byte-identical to the surrogate before the knob existed, and on-policy
(`π_old == π`) the two levels agree exactly in value and gradient under every `loss_type`.

### Why a sub-knob and not a new `grpo_surrogate` value

`clipped` is defined above as the owner of the importance ratio, so the ratio's granularity belongs
to it. A `grpo_surrogate="gspo"` value would instead have to be threaded through every
`== "clipped"` decision: the π_old attestation gate (`_objective_consults_pi_old`) and the
`kl_beta` guard in `config/rl.py`, the objective and identity assembly in `config/build.py`, the
temperature contract in `control/temperature.py`, and the native-policy preflight in
`entrypoints/grpo.py`. A missed site fails open: the attestation gate, for example, fires only for
`clipped`. The sub-knob touches none of them, so π_old caching, the attestation gate, the
temperature contract, and entry preflight are inherited unchanged. As with Dr-GRPO, the recipe
records a distinct `algorithm` and `job.name` for provenance:

```python
cfg = build_grpo_config()
cfg.algorithm = "gspo"
cfg.job.name = "qwen3b_bf16_gspo_seed42"
apply_gspo_knobs(cfg.rl, mini_batch_per_replica=32)
```

`apply_gspo_knobs` in [`configs/gspo_qwen3b_gsm8k.py`](../../../cosmos_framework/posttrain/configs/gspo_qwen3b_gsm8k.py) is the one
place that turns a clipped `rl` section into the paper's objective: `grpo_surrogate="clipped"`,
`importance_sampling_level="sequence"`, `lower_bound_ratio=0`, `epsilon_low/high=3e-4/4e-4`,
`loss_type="seq-mean-token-mean"`, the caller's `mini_batch_per_replica`, and
`certified_deterministic_forward=True`. No config class, entry, trainer, or objective wrapper is
introduced.

### Rejected versus warned

The rule is the one this document already applies: a knob the selected loss never reads is
rejected (the dual clip, `legacy_pg`, OPD, per-token advantages), while a knob that is honored but
inert is warned (an on-policy ratio at `n_mini == 1`, token-scale epsilons, bf16 logprobs). The
concrete list, the conditional identity key, and the two response-level diagnostics are documented
once, in the `importance_sampling_level` row and the GSPO paragraph of [`config.md`](../reference/config.md).

Run the recipe through the shared GRPO entry:

```bash
WANDB_MODE=disabled PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/gspo_qwen3b_gsm8k.py -- \
  rl.max_steps=1 rl.save_every=0 rl.resume=off
```

## When objective customization is not enough

Do not force these requirements into an objective builder:

- **New group-wide estimator:** add or reuse a reward-stage annotator and wire it before dispatch.
  A policy rank may hold only part of a group.
- **New teacher signal:** declare the producer role and data edge, then let the objective consume
  the prepared signal. OPD is the existing example.
- **New tensor/media input:** extend the declared data contract and bulk transport path; do not load
  it ad hoc from the objective.
- **Different rollout/reward/train ordering:** own that ordering in a variant entry loop.
- **Different physical placement:** express it through placement/config, not through the loss.

These are algorithm-recipe or framework extensions, even when they ultimately feed a new
objective.

## Testing an objective customization

Add focused L0/CPU tests under `tests/` that cover:

- **Mathematics:** the new objective differs from its base exactly where intended.
- **Reduction:** split and unsplit batches produce the same weighted numerator/gradient semantics.
- **Contract preservation:** reduction, control weight, metric adapter, old-logprob cache, and
  identity are correct.
- **Data preconditions:** missing required annotations fail loudly.
- **Entry wiring:** the selected builder reaches worker launch when the variant genuinely owns a
  new objective.
- **Existing behavior:** entries that reuse the changed seam still select their original objectives
  and retain their topology and loop.

Dr-GRPO's math and recipe tests are in
[`tests/test_algorithm_grpo_clipped.py`](../../../cosmos_framework/posttrain/tests/test_algorithm_grpo_clipped.py) and
[`tests/test_configs_dr_grpo_qwen3b_gsm8k.py`](../../../cosmos_framework/posttrain/tests/test_configs_dr_grpo_qwen3b_gsm8k.py);
GSPO's sequence-level primitive, objective, config, and recipe tests are in
[`tests/test_algorithm_grpo_gspo.py`](../../../cosmos_framework/posttrain/tests/test_algorithm_grpo_gspo.py) and
[`tests/test_configs_gspo_qwen3b_gsm8k.py`](../../../cosmos_framework/posttrain/tests/test_configs_gspo_qwen3b_gsm8k.py):

```bash
pytest -q cosmos_framework/posttrain/tests/test_algorithm_grpo_clipped.py \
  cosmos_framework/posttrain/tests/test_configs_dr_grpo_qwen3b_gsm8k.py \
  cosmos_framework/posttrain/tests/test_algorithm_grpo_gspo.py \
  cosmos_framework/posttrain/tests/test_configs_gspo_qwen3b_gsm8k.py
```

## Review checklist

- The objective receives all required data through declared, existing channels.
- Group/source computations happen before dispatch, not on a rank-local trainer batch.
- Loss and `OBJECTIVE_WEIGHT_SUM` use the same normalization unit.
- Changed metrics are updated and mapped to stable public names.
- Stateful objective features such as the old-logprob cache are preserved or deliberately replaced.
- The builder is module-level and worker-serializable.
- Config has one owner and one default for each user-visible value.
- The entry selects the objective explicitly without algorithm-specific infra flags.
- Existing objective, trainer, role, placement, checkpoint, and loop components are reused where
  their contracts still apply.
- New topology or phase requirements are implemented at the entry boundary, not hidden in the
  objective.
