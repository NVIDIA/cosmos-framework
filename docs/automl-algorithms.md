# AutoML algorithm contracts

Install the local core 1.0.0.dev0 wheel as described in [AutoML](automl.md). Choose
`search.algorithm` in the same native reasoner manifest; there are no competing
launchers or target selectors. All modes use explicit objective direction,
declared parameter domains, recorded native jobs and bounded sequential execution.

## Capabilities

| Manifest value | Proposal/allocation behavior                                                                    | Additional contract                                                             |
| -------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `random`       | Seeded independent domain sampling                                                              | Fixed training budget                                                           |
| `bayesian`     | GP + expected improvement; three successful random observations first                           | Fixed training budget                                                           |
| `bfbo`         | RBF GP + decaying UCB with local penalization                                                   | Fixed training budget                                                           |
| `hyperband`    | Successive halving over multiple brackets                                                       | Cumulative step fidelity and native DCP resume                                  |
| `hyperband_es` | Hyperband plus confidence-guarded power-law curve pruning between stages                        | Enough rung observations and a full-budget peer; otherwise no curve pruning     |
| `bohb`         | Hyperband with good/bad KDE (TPE-style) proposals                                               | Same-resource observations; sparse/singular KDE explicitly falls back to random |
| `asha`         | Promotion quotas without a synchronous bracket barrier                                          | Same fidelity contract; executor still runs one stage at a time                 |
| `dehb`         | Hyperband with DE/rand/1 and binomial crossover                                                 | Same-resource successful population; random warmup                              |
| `pbt`          | Periodic population exploit/explore, donor checkpoint continuation and perturbation             | LR/weight-decay axes only, fixed topology/batch/accumulation                    |
| `llm`          | Structured, domain-validated candidate proposals                                                | Reviewed endpoint/model and request/token budget                                |
| `hybrid`       | Bounded numerical phases with narrowed domains, selected by an LLM or explicit research program | Phase allowlist/budgets; fidelity if any allowed phase needs it                 |
| `autoresearch` | Base-recipe baseline, bounded proposals, local validation, LLM prescreen, measured keep/discard | Base values inside domains; two LLM requests per post-baseline trial            |

The original engine exposed the eleven modes other than standalone `random`.
It did not expose standalone grid search or standalone TPE; BOHB contains the
density-based proposal behavior. These are algorithm-family ports, not a claim
of identical random trajectories or binary state compatibility with TAO brains.
Framework's current four SFT axes remain authoritative; model-specific TAO spec
generation, SDK execution and automatic network discovery are not imported.

GEPA is separate from SFT hyperparameter search; see [Auto-Prompter](automl-prompts.md).

## Multi-fidelity training

Start from [reasoner-multifidelity.yaml](../examples/automl/reasoner-multifidelity.yaml).
`fidelity.minimum`/`maximum` are **cumulative optimizer steps**, not epochs.
Maximum must equal the base recipe's `trainer.max_iter`. Explicit
`scheduler.cycle_lengths` must cover the full horizon and are retained at every
stage. Epoch-based stopping and checkpointing remain unsupported.

Hyperband/BOHB/DEHB use geometric rungs and multiple brackets. PBT uses intervals
of `minimum` steps, capped at `maximum`; `reduction_factor` does not set PBT's
interval. `plan` reports bracket sizes or the population schedule's stage cap.
`max_trials` counts **every evaluated stage**, including promotions, failed
stages and each population member, not just fresh configurations. Too small a
cap may produce no full-budget result and therefore no winner. ASHA's resource
stream is bounded by that cap and the elapsed deadline.

Initial stages load the declared base weights. A promotion requires a successful
parent, a trial-local committed DCP at its recorded step, and nonempty model,
optimizer, scheduler and trainer checkpoint components. Each component needs
nonempty metadata and at least one nonempty shard; all shard files must be
regular files contained within the checkpoint. Native DCP deduplicates replicated
state onto the lowest rank, so empty shards on other ranks are valid. The CPU
planner checks this structure without deserializing metadata; native loading
still validates the serialized state. It mounts the donor
read-only, sets native `checkpoint.load_training_state=true`, and records a
per-rank proof of restored checkpoint/iteration. Stage completion also requires
native `SUCCESS`, the exact target checkpoint and held-out evaluation. Missing
proof fails collection instead of treating a warm start as continuation.

PBT retains donor moments and progress. Framework's optimizer/scheduler
containers preserve the **new recipe's** per-group LR multipliers and weight
decay (including no-decay groups); a callback verifies this before the first
update. Changing batch/accumulation would change data replay semantics and is
not allowed for PBT. Architecture changes and partial tensor-loading filters
are not a supported population transplant contract.

`best.json` selects only successful stages at `fidelity.maximum`. Different
fidelities are never ranked together to select a final winner. Failed stages
are not donors. Approved [infrastructure retries](automl-slurm.md#bounded-failure-recovery)
repeat the same logical stage from its original base or completed parent; they
do not restore an unvalidated partial checkpoint or consume another stage slot.
Promotion resumes a **completed** lower-fidelity stage in a new allocation,
including when that parent's successful execution was a retry attempt.

Options live in `search.options`: BFBO `kappa`, `kappa_decay`, `kappa_min`,
`penalization_radius`; BOHB `kde_samples`, `min_points_in_model`, `top_fraction`;
DEHB `mutation_factor`, `crossover_probability`; PBT `population_size`,
`perturbation_factor`, `exploit_fraction`; curve pruning `min_curve_points`,
`curve_confidence`, `early_stop_margin`. Defaults are emitted by `plan`.

## LLM and research controllers

Start from [reasoner-llm.yaml](../examples/automl/reasoner-llm.yaml). The endpoint
is an OpenAI-compatible base URL; requests append `/chat/completions`. HTTPS is
required except for loopback mock services. Set only the credential variable
name in `api_key_env`; export its value privately in the controller environment.
It does not need to enter the training container.

Review `max_calls`, `max_output_tokens` **per call**, `max_prompt_chars`, request
timeout and total experiment timeout. These are usage limits, not a monetary
price guarantee. The service receives parameter domains/values, scalar scores,
statuses and supplied research instructions—not raw datasets, model weights or
training logs. Do not put secrets in categorical values or instructions.

Responses may change only declared, in-domain configuration fields. They cannot
alter commands, paths, topology, budgets or endpoint settings. JSON/tool-call,
domain and workload checks fail closed. Each API intent is saved before I/O;
completed responses are reused on resume. An incomplete/error receipt stops
instead of silently making another potentially paid request. No redirects or
automatic request retries are used.

The controller requests `response_format: {type: json_object}` for structured modes.
The endpoint must support that format; an HTTP rejection does not trigger a
fallback request. One complete JSON object is accepted, either bare or enclosed
in one `json`/unlabelled Markdown code fence. Prose, multiple objects, incomplete
JSON, duplicate keys and nonfinite constants are rejected. Truncated completions,
tool calls and refusals fail closed; syntax normalization does not bypass domain
or Framework workload validation. GEPA reflection retains its text-output format.

Error receipts include a safe `error_kind` and, when available, HTTP status,
finish reason, token usage and content length/format. They do not retain raw
error bodies, credentials, prompts or underlying exception strings. Validate a
bounded CPU-only proposal request before a GPU rerun. Old 0.2.0 receipts lack
these diagnostics; do not guess their cause or erase them to force a retry.

`hybrid` allows only `llm.phase_algorithms`, at most `llm.phase_trials` stages
per phase. Domains may narrow but not expand/change type. It can also execute
an explicit program without strategy API calls:

```yaml
algorithm: hybrid
llm:
  # Include the explicit endpoint/model/usage envelope as in reasoner-llm.yaml.
  phase_algorithms: [random, bayesian]
  phase_trials: 4
  research_program:
    - {name: exploration, algorithm: random, trials: 4, carry_forward: base}
    - name: refinement
      algorithm: bayesian
      trials: 4
      carry_forward: best
      domains:
        optimizer.lr: {kind: float, low: 0.000002, high: 0.00005, log: true}
```

This is a fragment of `search`, not a full manifest; raise the approved overall
trial cap to cover the program. Comparable observations inside the narrowed
domains seed later fixed-budget phases. Population/stage phases require their
own adequate stage budgets. `autoresearch` accepts structured hypotheses and
keeps only measured improvements; its prescreen is not evidence of performance.
Arbitrary repository edits, generated shell commands and model-specific source
rewrites are deliberately outside this bounded controller.

## Validation boundary

CPU operational runs exercised all twelve modes with actual tiny child
processes, promotion lineage, durable state and a loopback mock LLM. Existing
regression tests are retained; no new unit-test suite was added. A later bounded
SLURM smoke exercised native training/evaluation and real GEPA reflection; see
the [recorded outcomes and remaining gaps](automl.md#bounded-slurm-algorithm-verification-2026-10-06).
The corrected image subsequently passed all six promotion modes on eight GPUs,
including native continuation and PBT donor transfer with a perturbed learning
rate; see the [promotion rerun](automl.md#bounded-promotion-rerun-2026-10-07).
That run checked every rank's restore receipt and PBT optimizer configuration.
Hyperband-ES curve pruning, structured LLM controllers and production-scale
search quality remain outside that GPU smoke. A subsequent
[real-endpoint CPU check](automl.md#live-structured-llm-verification-2026-10-08)
reproduced the original client's fenced-JSON failure and verified corrected LLM,
hybrid and autoresearch proposals, including native Framework configuration
validation. A subsequent [GPU rerun](automl.md#structured-controller-gpu-rerun-2026-10-08)
passed first-round training but scheduler-query timeouts caused cancellation
during evaluation. The subsequent [v1 reruns and full-data outcomes](automl.md#v1-slurm-results-reviewed-2026-10-12)
record completed LLM/hybrid/autoresearch GPU smokes and six completed full-data
numerical/population searches. Live infrastructure recovery, Hyperband-ES curve
pruning and search-quality comparisons remain unverified.
Review changes in topology/workload before larger runs; do not mutate or upgrade
a paused run.
