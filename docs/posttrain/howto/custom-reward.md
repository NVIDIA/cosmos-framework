# Reward providers

**Come here when a rule reward cannot express what "better" means for your task.** A reward
provider scores one group of candidate completions and is the main extension point for online RL.

Primarily used by [`../guides/grpo.md`](../guides/grpo.md) and
[`../guides/opd.md`](../guides/opd.md); [`../guides/nft.md`](../guides/nft.md) uses the image and
video reward contracts instead. Provider implementations are supplied as operator-owned extension modules.

Posttrain owns reward execution, not customer reward semantics. A dataset supplies a canonical
prompt and an optional opaque structured `reference`; the provider selected for the run interprets
that reference after rollout has produced a completion.

Raw dataset conversion stays outside the reward framework. Configure `rl.data.prompt_column` and
`rl.data.response_column` to map source columns onto the canonical `prompt` and `reference` fields,
or use the existing schema-preserving `rl.data.items_fn` row map for an HF source. `DataPacker`
remains responsible for model chat templating, tokenization, media shaping, packing, and collation;
it is not a benchmark-schema adapter.

## Data contract

Each prepared row needs one prompt column and, when the selected provider declares
`needs=("reference",)`, one non-null JSON-compatible reference column. Column names are customer
choices; config maps them into the framework vocabulary. For example:

```json
{
  "messages": [{"role": "user", "content": "Follow the supplied constraints."}],
  "reward_input": {"constraint_ids": ["customer_rule"], "arguments": {"count": 3}},
  "source_id": "stable-customer-row-id"
}
```

```text
rl.data.path=/path/to/prepared.jsonl
rl.data.prompt_column=messages
rl.data.response_column=reward_input
```

The Env renames `messages` to canonical `prompt` and `reward_input` to opaque `reference`. The local
JSONL loader retains only those mapped fields, so provenance or other provider inputs must live
inside the structured reference. The HF `_ColumnMapped` path additionally preserves unmapped row
columns in `obs.representations`. Core validates presence/cardinality but never interprets the
reference schema. The customer provider reads the completion plus whichever canonical fields it
declared in `RewardProviderSpec.needs`.

For a local JSONL, prepare this shape offline. For an HF source, `rl.data.items_fn` may deterministically
rewrite row values before column mapping, but it must preserve the row's keys. Dataset-specific
downloads, filtering, joins, deduplication, and schema interpretation remain customer-owned.

## Group-native online provider contract

The online GRPO/OPD reward stage has one synchronous runtime contract. Every provider receives one
immutable request containing every ordered candidate for exactly one complete `group_id`; there is
no trajectory/group scope switch in `RewardWorker`:

```python
class MyReward:
    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=RewardProviderResult(
                        terms={"task": float(bool(candidate.completion.strip()))}
                    ),
                )
                for candidate in request.candidates
            ),
        )
```

Each `RewardCandidate` carries `trajectory_id`, completion text, an optional rollout-authored
`completion_token_count`, and an optional rollout-authored `finish_reason` (the engine's verbatim
verdict on how the completion ended, `policy_info["finish_reason"]`; `None` when the producer
recorded none). The `candidate.is_length_truncated` property answers whether that verdict is exactly
`"length"`, i.e. the engine cut the completion at its token budget; `None` reads as "not known to be
truncated", never as "stopped". It delegates to `schema.is_length_verdict`, the same comparison
`schema.is_length_truncated(policy_info)` uses, so the reward boundary and the trainer's truncation
mask can never disagree. `RewardGroupRequest.context` carries the group's common declared
`prompt` and/or `reference`. Before calling custom code, the framework-owned bound adapter rejects an
incomplete group, duplicate or missing identity, missing completion, `obs.id`/`group_id`
disagreement, or inconsistent common inputs. Every result item must repeat the exact ordered
trajectory ID and contain one valid `RewardProviderResult`. Candidate work may run concurrently
inside a provider, but `evaluate` remains one synchronous group barrier.

Common-input identity is type-strict and structural after freezing: for example, JSON integer `1`,
boolean `true`, and floating-point `1.0` are distinct. A malformed candidate therefore cannot borrow
another candidate's superficially equal prompt or reference.

Common inputs are detached from trajectory storage and deeply frozen: mappings are read-only and
JSON sequences become tuples. A scorer that needs mutable containers must make its own private copy;
mutating provider inputs can never mutate a trajectory or another provider's prepared request.

Training validates group cardinality against `rl.group_size`; held-out validation uses its own
`n_generation`. Provider-name and optimized-term collisions fail at Worker construction. Scoring,
terminal reward writes, group-relative annotation, and report construction form one rollback-safe
transaction; a failure restores the batch's prior reward-stage state.

### Pointwise scorer authoring

A rule that scores candidates independently does not implement a second provider kind. Write a
typed candidate scorer and lift it through the framework's `map_candidates` adapter:

```python
def score_candidate(
    candidate: RewardCandidate,
    context: RewardContext,
) -> RewardProviderResult:
    reference = context.reference
    if not isinstance(reference, str):
        raise RewardContractError("score_candidate needs a string reference")
    passed = float(reference.casefold() in candidate.completion.casefold())
    return RewardProviderResult(terms={"task": passed})


provider = map_candidates(score_candidate)
```

The adapter maps the function over the request in candidate order and produces explicitly
correlated `RewardCandidateResult` items. Built-in arithmetic, GSM8K, countdown, exact-match,
nonempty, and constant rewards all use this path. Comparative judges and the NVIDIA Inference Hub
rubric provider implement the same group-native `evaluate` method directly. This gives custom code
two concise authoring styles while preserving one catalog contract and one Worker execution path.

Only finite declared terms enter `RewardCombiner` and terminal `reward["final"]`.
Reporting-only diagnostics are provider-local `MetricContribution` values and never enter reward or
advantage computation. This contract is specific to the online GRPO/OPD reward stage; env-native
step rewards and future process-reward placement are separate framework contracts.

### Migrating an older pointwise provider

The former `scope="trajectory"`, `mode="sync"`, `evaluate(Trajectory)` authoring contract is not
retained as a second runtime path. Move its candidate-local logic into a
`(RewardCandidate, RewardContext) -> RewardProviderResult` function, declare the same `needs`, term,
and diagnostic schemas in `RewardProviderSpec`, and use `partial(map_candidates, scorer)` as the
factory. Code that genuinely compares candidates should instead implement group-native `evaluate`
and return correlated result items. Catalog names and `rl.reward` selection do not otherwise
change.

## Run-bound catalog

Custom providers are supplied by a zero-argument, importable catalog factory. Each
`RewardProviderSpec` declares the provider factory, required canonical inputs, revision, term names,
fixed diagnostic names, and an optional JSON-compatible semantic identity. The catalog is
resolved from the same dotted path during driver preflight and reward-actor setup; custom entries
may add names but cannot override built-ins.
The small rule scorers remain built in and are lifted with `map_candidates`; they are not a precedent
for adding benchmark evaluator libraries to core.

The module containing the catalog must be importable on the driver and every reward actor. An
unknown provider, import failure, factory contract mismatch, missing required input, undeclared term
or diagnostic, or non-finite output fails before publishing a partially scored batch. Driver
preflight constructs the selected training provider and, when enabled, the selected validation
provider before model or rollout setup. Both the zero-argument catalog factory and provider
factories must therefore be deterministic, lightweight, CPU-only, and side-effect free; model/GPU
setup, network calls, and scoring belong outside construction. Reward actors independently resolve
the same persisted catalog path and construct their own provider instances.
Change the declared revision whenever provider code changes. The catalog also derives a canonical
identity SHA-256 from `semantic_identity`; remote providers use it for judge revision, complete
judge-content hash, timeout/retry/concurrency settings, credential reference, and failure policy. Held-out
validation persists the catalog path, selected provider name, revision, and identity hash so a
pending result cannot silently resume under different reward semantics. A credential reference is
identity; the resolved secret value never is.

Provider diagnostic metrics use the unambiguous hierarchy
`train/reward_provider/<provider>/<diagnostic>` (or the corresponding `val/` namespace). The
framework-owned sample count is `<namespace>/reward_provider/<provider>/sample_count`; provider and
diagnostic names are separately validated path segments.

Training and held-out validation resolve the same catalog. Validation may select another provider
name from that catalog through `rl.validation.reward`, but it cannot inject a second implementation
under the same name.

## Rank-local decoded-video adapter

Online WFM has one specialized provider adapter for the deployed DanceGRPO video service. It keeps
the canonical group request/result boundary, fixed `RewardProviderResult` term/diagnostic schema,
and provider-revision provenance. Its bound adapter maps each candidate through the singular scorer;
it does not batch candidates into one service request or claim comparative group scoring. In the
initial distributed WFM topology, every model rank owns exactly one trajectory, so production makes
one service request per rank while the complete reward group exists only after the distributed
advantage estimator gathers one scalar per rank.

The adapter receives a model-owned decoder and decodes the final UniPC latent locally under
`torch.no_grad()`. The injected decoder must explicitly attest `output_range="neg_one_to_one"`;
the adapter validates one floating, finite, materially in-range `[1,3,T,H,W]` result before mapping
it to CPU `uint8`. It sends `media_type="video"`, explicit FPS, and receives one positional score;
the service does not echo or authenticate a client sample identity. The client instead retains the
non-empty service UUID returned by enqueue and sends that exact UUID with every pull. Correlated
`202` pending responses must carry that exact UUID and `status="pending"`; the only accepted legacy
`400` pending shape is the exact `Not found scores for <uuid>` message. Every other `400` is terminal
and is not polled again. A successful `200` contains only the singleton positional score. End-to-end
client identity or idempotency therefore requires a future service-protocol extension and is not
claimed by this adapter. The reward actor must provide `curl`, endpoint URLs, and its bearer token.
Enqueue is never retried because the service exposes no
idempotency key, so an ambiguous transport failure can leave orphaned remote work. Pull is the only
retryable operation, and the service exposes no server-side cancellation. Transient pull failures
use capped backoff and retry until the absolute evaluation deadline. Each transport process and the
whole decode/enqueue/poll/result sequence have hard wall-clock bounds. A result is accepted only
before the absolute deadline and only when its singleton score is finite. Video payloads are streamed
from a private per-rank spool file. Set `VideoRewardConfig.spool_dir` to capacity-managed actor-local
storage in production; `None` uses the platform temporary directory. Capacity must cover one decoded
uint8 `[1,3,T,H,W]` sample per active rank plus the NumPy header. The production `121x704x1280`
contract is about 312 MiB per rank (about 2.44 GiB across eight concurrent ranks), before unrelated
runtime temporary storage.

`VideoRewardConfig` is the dependency-light value contract for this adapter, including the required
provider revision, transport bounds, FPS, and the exact fp32 clamp-then-scale transform. The
online-WFM entrypoint constructs the client and decoder in each owning rank, injects distributed
failure convergence, and publishes the mergeable provider report together with globally complete
advantage telemetry. Its recipe intentionally carries a rejected sentinel until the operator
overrides it with the immutable deployed scorer revision.

The qualified T2I recipe adds `ImageRewardConfig` and a separate image adapter while
reusing the same group-native bound-provider, correlation, deadline, diagnostic, and
provenance contracts. Image conditioning must decode to finite floating
`[1,3,1,H,W]` in `[-1,1]`; the adapter converts it to CPU uint8 `[1,H,W,3]`,
serializes with `numpy.save(..., allow_pickle=False)`, sends `media_type=image`, and
omits `video_infos`. The video request schema and default remain unchanged. See
[public NFT run guide](run-posttrain.md#7-nft) for the image reward service contract.

## What belongs outside core

Benchmark-specific checkers, copied third-party evaluator source, external model judges, their
dependencies and credentials, source converters, suite configs, and full parity matrices belong to
the customer module or an optional integration package. Core CI uses small fake providers to verify
the catalog, binding, reporting, GRPO, and DPCA contracts without depending on a benchmark package.
