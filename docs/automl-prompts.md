# GEPA Auto-Prompter

GEPA optimizes a reasoner's **system prompt**, optionally jointly with bounded
frame-count/pixel-budget choices, using a fixed checkpoint. It does not train
weights and is not an SFT search algorithm. The reusable implementation lives
in `automl_core`; Framework supplies native annotations, evaluation and metrics.
Users need no TAO source checkout or SDK.

Install the supplied core 1.0.0.dev0 wheel and Framework's `automl-prompts` extra:

```bash
python -m pip install '/path/to/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl[prompts]'
python -m pip install -e '.[automl-prompts,reasoner-eval]'
cosmos-automl prompts --help
```

The controller extra pins `gepa==0.1.2`. The workload image needs the matching
Framework prompt evaluator and its normal GPU inference dependencies, not GEPA
or an LLM credential. Existing training-only images are not implicitly upgraded.

## Data and checkpoint preparation

Start with [reasoner-prompts.yaml](../examples/automl/reasoner-prompts.yaml) and
[prompt-evaluation.toml](../examples/automl/prompt-evaluation.toml). Set a fixed
HF/native-compatible evaluation checkpoint in `model.model_name`; DCP/LoRA
evaluation follows the existing reasoner checkpoint preparation contract and
requires its matching model config/base model. No training checkpoint is chosen
automatically. Record immutable checkpoint/data/code revisions in
`workload.provenance`; external weights/media are not content-hashed for you.

Provide three roles, with the final one optional:

- `reflection`: examples available for prompt improvement.
- `validation`: selects the final candidate by the full aggregate metric.
- `test`: scored only after validation selects the winner; omit to keep it untouched.

Annotations are a JSON array (or `items` array) with one `question`/`references`
pair, or a two-turn `conversations` question/answer pair, per record. `image`
or `video` references resolve against that role's `media_dir` and must exist.
Tasks must consistently match the native evaluation TOML: `mcq`, `bcq`, or
`binary`. Roles must be media-disjoint; text-only roles must be query-disjoint.
Do not split different questions from the same video across roles. Paths are
checked, but aliases/copies of the same content require your own dataset audit.

The adapter creates batch-local native annotations with stable IDs, checks
prediction/query alignment and exact coverage, and reuses native model loading,
decoding and answer parsing. `accuracy` is a fraction; binary `macro_f1` is the
mean F1 of yes/no (zero for an undefined class). Invalid predictions count as
errors, not dropped examples. The batch metric specialization supports
single-class reflection minibatches; it does not invoke a balanced-accuracy
diagnostic that requires both classes.

## Plan, approve, observe

```bash
cosmos-automl prompts plan --manifest prompts.yaml --workspace outputs/automl
# After replacing placeholders and reviewing resources, disclosure and budget:
cosmos-automl prompts run --manifest prompts.yaml --workspace outputs/automl --approve
cosmos-automl prompts status --experiment-id reasoner-prompts --workspace outputs/automl
cosmos-automl prompts logs --experiment-id reasoner-prompts --workspace outputs/automl
cosmos-automl prompts resume --manifest prompts.yaml --workspace outputs/automl --approve
cosmos-automl prompts cancel --experiment-id reasoner-prompts --workspace outputs/automl --approve
```

Planning checks inputs and displays an example native command without GPU work,
network requests or output files. Choose venv, Docker or SLURM explicitly as in
[AutoML](automl.md); each uncached batch uses the same durable job lifecycle.
For SLURM, prepare the same image/template/record hooks but use a prompt-specific
bootstrap, not the SFT bootstrap. Prompt evaluation uses one GPU per batch;
allocating more does not parallelize this implementation.

Review both evaluation and LLM spend:

- `max_metric_calls` bounds GEPA's search counter. Also set
  `max_evaluated_examples`, covering all uncached inference examples, **including
  final validation reranking and optional test**. GEPA checks its own counter
  at iteration boundaries; the outer example cap is checked before each batch.
- `max_batches`, `max_candidate_proposals`, reflection minibatch size, batch
  timeout and total elapsed timeout are independent caps. Budget enough final
  validation for every retained candidate. Insufficient budget stops with an
  error; it does not hand off a partially scored winner.
- `llm` declares endpoint/model, credential **name**, request timeout, maximum
  requests, output tokens per request and prompt characters. Request intents
  precede API I/O; ambiguous calls are not automatically repeated.

`allow_reflection_data` starts false. Set it true only after reviewing disclosure:
the reflection endpoint receives queries, generated answers and sanitized
correctness/guidance feedback. The adapter does not add gold labels, sample IDs,
media paths or images to those records. Free-form queries/outputs can themselves
contain sensitive data; the sanitizer cannot make arbitrary text safe. Approve
that data use explicitly. The reflection template asks for transferable
reasoning, not memorized example answers.

## Outputs and preserved capabilities

`prompt.json` records identity, budgets and child batches; `best.json` contains
the final prompt/config, each candidate's validation metrics, selected index,
and optional winner-only test metric. Upstream GEPA state and request receipts
are retained. Completed child batches and API responses are reused on resume;
the deadline and experiment identity do not reset. Joint discrete settings use
UCB proposals with persisted, deduplicated scoring history. Only
`dataset.system_prompt`, `vision.num_frames` and `vision.max_pixels` are exposed;
joint choices are positive integer strings and frame-count aliases are rejected.

The shared Python API preserves the original batch-runner callbacks, visual
reflection hooks, per-item/objective feedback, joint discrete settings, final
aggregate reranking, optional candidate-cost penalties and winner-only test
scoring. Framework's declarative adapter deliberately does **not** expose
arbitrary callback imports, candidate-cost code, image upload or optimization
of unrelated model settings. Those need explicit workload/privacy contracts;
they are not hidden unsupported flags. The TAO consumer retains its public API
names as imports of the shared implementation.

Real GEPA plus tiny CPU batches and a loopback mock reflection service have
been smoke-tested. A later bounded SLURM run also completed native GPU prompt
evaluation and one real remote reflection request; see the
[recorded results](automl.md#bounded-slurm-algorithm-verification-2026-10-06).
The candidate tied the baseline, so no WTS prompt-quality improvement was shown.
