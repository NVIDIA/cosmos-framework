# AutoML

Cosmos Framework owns the workload integration; `nvidia-automl-core` owns the
repository-neutral search and execution controller. Users do not need a TAO
checkout, SDK, model registry or skills installation.

AutoML tunes **reasoner SFT** against held-out native evaluation, with fixed-budget,
multi-fidelity, population and bounded LLM controllers. **GEPA Auto-Prompter**
optimizes evaluation prompts without training model weights. Both use the same
venv, Docker, SLURM, Kubernetes, Brev or Lepton execution lifecycle. See
[algorithms](automl-algorithms.md), [prompt optimization](automl-prompts.md), and
[SLURM execution](automl-slurm.md) and [cloud execution](automl-platforms.md). This is not a general optimizer for every
Framework training/inference workflow; the workload contracts remain explicit.

### Execution coverage and controller placement

The original TAO runner exposes six SDK platform factories. The independent
core implements all six without importing those SDK factories. Platform
implementation coverage is not equivalent to live GPU validation.

| Execution platform | Core support | Framework evidence / limits |
| --- | --- | --- |
| Docker | Implemented | Local Docker WTS GPU training/evaluation exercised. |
| Virtualenv | Implemented | CPU lifecycle verified; not equivalent to the recorded Framework GPU matrix. |
| SLURM | Implemented | On-cluster CPU controller submits and monitors separate single-node GPU allocations. Recorded bounded SFT/GEPA matrix results are below. |
| Kubernetes | Implemented; live validation pending | Single-pod Jobs, explicit context/PVCs, native deadline and UID-safe suspension. |
| Brev | Implemented; live validation pending | Docker on an existing instance, explicit daemon identity and shared/controller-local instance storage; no provisioning or teardown. |
| Lepton | Implemented; live validation pending | Workspace Jobs API, dedicated node groups and shared NFS; worker deadlines require a live controller for queue/startup cancellation. |

Controller location and workload platform are separate concerns, but the
current implementations have locality requirements: Docker/venv share their
controller host's filesystem, and managed adapters need the same persistent
filesystem visible to controller and workers. See the [cloud storage contract](automl-platforms.md#controller-and-storage-contract).
A local/service-hosted brain talking to remote SLURM through SSH still
needs a transport and artifact/validation contract. Changing `execution.kind`
does not provide one. An experiment's execution identity is fixed on resume;
cross-platform migration and mixed-platform trial routing are not implemented.

For SLURM CPU allocations shorter than an experiment, the opt-in
[bounded controller handoff](automl-slurm.md#bounded-cpu-controller-handoff)
preserves active jobs and the original deadline. This is not an implementation
of local-to-remote orchestration or a cloud controller service.

## Installation and release status

The independently built `nvidia-automl-core` 1.0.0.dev0 wheel must be supplied by the
maintainer: **this change does not publish it to an index**. Install the wheel
first, then install the extra into your already-prepared Framework environment:

```bash
python -m pip install /path/to/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl
python -m pip install -e '.[automl,reasoner-eval]'
cosmos-automl --help
```

The core is maintained in `tao-automl/packages/automl-core` and built independently;
that is a maintainer source location, not a consumer installation prerequisite.
Current delivery is **local wheels only**. To install a supplied Framework wheel
instead of an editable checkout, install the core wheel above, then:

```bash
python -m pip install '/path/to/cosmos_framework-1.2.2-py3-none-any.whl[automl,reasoner-eval]'
python -m pip check
cosmos-automl --help
```

If Framework 1.2.2 is already installed from a different build, resolve the extras
above, then replace just that package with `pip install --force-reinstall --no-deps
/path/to/cosmos_framework-1.2.2-py3-none-any.whl`. The wheelhouse is not an offline
mirror of third-party dependencies.

A bare index install and normal `uv` lock resolution do not automatically find
these local wheels. The checked-in lock does not yet include the unpublished
core dependency. A distributable dependency location and refreshed resolver lock
remain a release gate; no source checkout or machine-specific wheel path is
silently added to work around it.

### Development version and build identity

AutoML is in its initial **v1 development phase**, not a released package series.
Keep `nvidia-automl-core` at **`1.0.0.dev0`** throughout the port. Bug fixes and local
wheel builds do not increment the patch version or development counter. Change
the package version only as part of an explicitly approved release; building a
local validation wheel is not a release. Cosmos Framework's existing package
version remains **1.2.2**, independent of the AutoML development label.

Store each build in a separate immutable wheelhouse directory and record its
wheel SHA256 and source revision. Do not overwrite an earlier same-named wheel.
When intentionally selecting another same-version wheel for a new environment
or run, install the exact supplied artifact; an existing installation may require
`pip install --force-reinstall --no-deps /path/to/the/wheel.whl` after resolving
dependencies. A package version alone does not identify a development build.

New SFT and GEPA identities include `engine_build`, a SHA256 fingerprint of the
installed core's relative Python file names and contents. Identical source and
wheel installs match regardless of their installation paths; changed runtime
code fails resume identity checks even with the same package version. This is
not a signature or a substitute for dataset, dependency, consumer-wheel and
image provenance. Keep the exact original wheels/manifests for paused runs.
Historical 0.1.x/0.2.x labels below identify preserved experimental artifacts,
not releases, and must not be rewritten or migrated into old ledgers.

New stage and prompt helpers must also be installed in the workload image;
installing only a new controller wheel does not upgrade an existing image.

The controller is CPU-only. The execution environment must separately have
Framework's training and reasoner evaluation prerequisites, GPU resources,
checkpoints and datasets. See [setup](setup.md), [post-training](../.agents/skills/cosmos3-post-training/SKILL.md)
and the native [video-QA recipes](../examples/README.md).
The trial handoff also uses `tomli-w` (included by the AutoML extra). A container
only needs these workload dependencies, not the controller or TAO SDK.

Preflight both training **and the selected evaluator** in that environment.
The example evaluation selects `vision.video_decoder = "torchcodec-cuda-on-demand"`,
which uses Framework's CUDA TorchCodec preprocessor. Omitting the decoder selects
the evaluator's PyNvVideoCodec path and requires its separate dependencies;
a successful TorchCodec probe does not validate that path. For NVIDIA Docker
GPU video decoding, expose `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`
in the image or export it and include its name in `execution.environment`.
Also verify that the non-root container has a valid user identity (for example,
forward `USER` and `LOGNAME`) and writable per-trial caches.

## Prepare and review

Start with [reasoner.yaml](../examples/automl/reasoner.yaml),
[training.toml](../examples/automl/training.toml) and
[evaluation.toml](../examples/automl/evaluation.toml). These are **planning
templates**, not launch-ready data/checkpoint selections. Replace the training
recipe with one already prepared for your dataset and matching checkpoint;
the example preset selects LLaVA data, not an arbitrary local JSONL.

Specify:

- Metric path and direction, e.g. `overall.accuracy` / `maximize`. Confirm that
  the selected task actually emits that metric. Validation data must be
  disjoint from training; keep a final test split untouched by search.
- Search domains and seed. Current supported axes are `optimizer.lr`,
  `optimizer.weight_decay`, `trainer.grad_accum_iter`, and
  `dataloader_train.max_samples_per_batch`. Include each field in the base TOML.
  Other axes need explicit adapter support; accepted TOML fields are not
  automatically safe search knobs (some are ignored for VLM).
- Positive full-budget `trainer.max_iter` and `checkpoint.save_iter` that divides
  it. Epoch-based training/checkpoint schedules are not part of this adapter.
  The evaluator refuses an earlier checkpoint even after training exits zero.
- Initial weights follow native VLM semantics: set `checkpoint.load_path` to
  a DCP checkpoint for a warm start, or explicitly set it to `""` to load HF
  weights from an explicit `model.backbone.model_name` (and, when separate,
  `model.backbone.safetensors_path`). An HF directory is not a DCP load path.
- Maximum trials, per-trial timeout (including checkpoint export/evaluation),
  total elapsed timeout, chosen GPU IDs and `nproc_per_node`. The total clock
  starts at experiment creation, includes downtime, and is not reset on resume.
  Failed logical trials count against the budget; approved execution retries
  have separate per-trial and total caps. Only `autoresearch` runs an explicit
  base-recipe baseline; other modes have no implicit baseline.
  `search.stop_on_failure: true` stops after the first unsuccessful trial once
  its approved infrastructure retries are exhausted or inapplicable.
  `search.pause_after_trials: [1]` pauses after the first trial for review;
  `resume --approve` acknowledges the boundary without resetting the clock.
  Set `workload.expected_validation_records` for a full-data run: collection
  then requires that exact count and native training `SUCCESS`.
- Native evaluation TOML for `mcq`, `bcq`, `binary`, `its_directionality`, or
  `metropolis_sgd`. Set training `model.lora_enabled` explicitly so evaluation
  never guesses the mode inherited from a preset. The first adapter scores a nonempty classification task;
  freeform/captioning metrics and video-generator quality need separate result
  contracts. LoRA recipes retain native LoRA handling and must supply
  evaluation `model.base_model_path`. For promotion/PBT checkpoint contracts,
  see [multi-fidelity training](automl-algorithms.md).
- Immutable training/validation/code revisions in `search.provenance`.
  The manifest binds recipe/evaluation contents, but external datasets, recipe
  Python code, checkpoints and mutable credential-backed resources are not
  automatically content-hashed. Do not change them during a search.

Manifest recipe/evaluation/cwd/mount paths resolve relative to the manifest.
Paths *inside* the native TOMLs retain Framework semantics (usually execution
cwd); use absolute mounted paths for assets. Hydra `${oc.env:NAME}` stays in
the recipe; declare NAME in `execution.environment` and export it in the
session. Do not put credentials in manifests or command arguments. Environment
values are not saved in the ledger. Logs remain the workload's responsibility.

```bash
cosmos-automl plan --manifest examples/automl/reasoner.yaml --workspace outputs/automl
# After resolving template fields and reviewing the plan/budget:
cosmos-automl run --manifest examples/automl/reasoner.yaml --workspace outputs/automl --approve
```

`python -m cosmos_framework.scripts.automl` is the equivalent module entrypoint.
Planning validates the structured SFT schema and domain endpoints without
writing files or importing training/GPU modules. It does **not** Hydra-compose
the preset, validate all cross-parameter combinations, check model memory,
check dataset overlap, download assets, or establish GPU compatibility.
Each actual candidate is validated again before launch. Native training still
performs its own full configuration and dependency checks.

If using Docker, replace the execution block explicitly:

```yaml
execution:
  kind: docker
  python: /opt/venv/bin/python
  image: your-reviewed-framework-image@sha256:your-digest
  gpu_devices: '0'
  mounts: [/absolute/checkpoints, /absolute/datasets]
  environment: [BASE_CHECKPOINT_PATH]
```

The image must already exist locally. No implicit registry login or pull occurs.
The Framework cwd and extra assets are mounted read-only at identical paths;
only the trial directory is writable. Containers run as the controller's UID/GID
with a writable per-trial HOME, so output ownership permits host-side collection;
the prepared image/interpreter must support that UID. Venv execution is not a sandbox. Both
adapters require trusted foreground commands. SLURM is a separate, explicit
execution kind; do not submit jobs from a venv/Docker trial command.
`nproc_per_node` must agree with
the recipe's parallelism and allocated GPUs. Evaluation currently uses one GPU.

## Results, monitoring and recovery

Each trial runs native `torch.distributed.run -m cosmos_framework.scripts.train
--sft-toml=...`, with a separate output root and fixed training seed. It starts
fresh from the declared initial checkpoint (`load_training_state=false`) for
fixed-budget candidates and initial stages. Promotions and PBT donors restore
native training state, including optimizer and scheduler progress; the adapter
verifies lineage and iteration on every rank. The
trial then locates the committed DCP pointer, checks the expected iteration,
and invokes the existing reasoner evaluator. Native checkpoint preparation
handles DCP-to-HF export; no new exporter or inference implementation is added.

```bash
cosmos-automl status --experiment-id reasoner-lr-search --workspace outputs/automl
cosmos-automl logs --experiment-id reasoner-lr-search --workspace outputs/automl
cosmos-automl resume --manifest search.yaml --workspace outputs/automl --approve
cosmos-automl cancel --experiment-id reasoner-lr-search --workspace outputs/automl --approve
```

Resume uses the same manifest and ID. Inspection/cancellation can use just the
workspace and ID even if source manifests have moved. `status` consults native
process/container state for active trials; logs prints log paths. Cancellation
stops execution and prevents future trials. Resume checks manifest identity and
the venv package inventory or Docker image ID and reconciles durable worker
receipts. It never guesses “latest experiment.” An `UNKNOWN` launch stops the
controller rather than submitting another job. Inspect its log and native
process/container; if unrecoverable, stop it explicitly and use a new ID.

`<workspace>/<experiment_id>/experiment.json` is authoritative; `best.json`
is the best successful **full-budget** observation, with sampled config and paths to its
native TOML, DCP and metrics. A trial needs successful commands, a finite metric,
nonempty evaluation and valid trial-local artifacts. No successful trial means
`best=null` and a nonzero CLI exit. A best observed validation score is not final
test evidence. Training configs, checkpoints, containers and logs are retained;
no automatic cleanup deletes them.

For SLURM, [bounded failure recovery](automl-slurm.md)
adds optional same-candidate retries and worker/progress watchdogs without
changing algorithm history, promotion lineage or paid proposal counts. Review
the extra allocation caps in `plan`. An `attempt_id` can differ from the logical
trial `id`; `attempts` preserves failed executions and their native identities.
Unconfirmed cancellation blocks replacement rather than creating duplicate work.

## Scope and checks

The shared engine ports the original eleven algorithm/controller modes and
GEPA, while retaining random search. See the [capability table and intentional
differences](automl-algorithms.md). No TAO SDK, model selector or target-runtime
dispatch is required. Execution remains sequential, including ASHA; simultaneous
trials, multi-node trials, arbitrary code-editing agents, and video-generator
objectives are not supplied by this adapter.

The 0.2.0 modes have CPU operational checks (including real GEPA with a local
mock reflection endpoint) and the bounded SLURM results below. GPU continuation,
PBT donor transfer and perturbed-LR restoration passed the promotion rerun.
Real-endpoint LLM/hybrid/autoresearch proposals passed the CPU-only check below.
Their subsequent GPU rerun completed first-round training but was interrupted
by scheduler-query timeouts during evaluation. Hyperband-ES curve pruning and
completed GPU searches driven by those structured controllers remain verification
gaps; these checks do not establish model quality or validate a production search.

Run adapter tests with an installed core wheel (no GPU imports required):

```bash
python -m pytest -o addopts='' --confcutdir=tests/automl tests/automl
python -m ruff check cosmos_framework/automl cosmos_framework/scripts/automl.py tests/automl
```

These validate planning, native paths, schema guards, discovery and checkpoint
handoffs using fixtures. They are not end-to-end GPU training/evaluation tests.

### Bounded local verification (2026-10-05)

A separate real Docker run used Cosmos3-Nano-VLM and disjoint WTS training and
validation data on four A100 80 GB GPUs (four-rank training, single-GPU evaluation).
Four sequential trials each completed 10 LoRA steps, saved a native final-step
DCP, exported it for evaluation, and scored the same 32 held-out questions.
All four scored 21/32; the best record selected the first tied trial. This verifies
the workflow, **not model-quality improvement or a full-dataset benchmark**.

The fourth candidate exercised GP/EI after three successful warm-up observations.
With tied scores, its acquisition retained the seeded initial candidate and the
GP fit emitted convergence warnings. Controller interruption/resume collected
existing workers/results without duplicate trials. A fifth four-GPU trial was
canceled during training; its container and child processes stopped, with no
successful result recorded. No GPU driver upgrade was needed.

The initial smoke trained successfully but failed evaluation because its omitted
decoder selected unavailable PyNvVideoCodec dependencies. The explicit TorchCodec
example and evaluator preflight guidance above address that configuration mismatch.
GPU UUID allocation also avoided a host-specific Docker ordinal-selection failure;
the non-compute display GPU was excluded.

The two project wheels passed a fresh CPU-environment install, dependency check,
CLI help and planning outside both source checkouts, with no TAO packages installed.
The core wheel also completed a small standalone four-trial CPU example. Those
packaging checks do not establish GPU compatibility. Separately, the 0.1.1
SLURM path completed one full WTS trial (5,555 training records, 695 steps;
2,676 validation records, 2,355 correct). The search paused after that trial;
this does not validate the new 0.2.0 modes or establish an AutoML improvement.
Other models/tasks and distributed evaluation remain unverified here.
A tokenizer-regex warning in the workload logs also needs investigation before
using these runs as model-quality evidence; this validation did not alter the
supplied checkpoint or tokenizer.

### Bounded SLURM algorithm verification (2026-10-06)

Independent experiments used the same WTS subset (64 training records and 16
validation records), Cosmos3-Nano-VLM, eight GPUs per SFT stage and at most five
cumulative LoRA steps. Up to four experiments ran concurrently; execution within
each experiment remained sequential. GEPA used a single GPU per evaluation batch.
All 26 submitted SFT stages and four GEPA batches completed, but the matrix
controller correctly reported failure because nine modes did not finish:

| Modes | Observed outcome |
| --- | --- |
| Random, Bayesian, BFBO | Completed their smoke budgets; Bayesian reached its guided GP proposal |
| Hyperband, Hyperband-ES, BOHB, ASHA, DEHB, PBT | Initial stages passed; the adapter rejected the first promotion/continuation before GPU submission |
| BOHB, DEHB proposal branches | KDE and differential-evolution candidates passed before the promotion failure |
| LLM, hybrid, autoresearch | Structured API request/response failed; only autoresearch's training baseline completed |
| GEPA | Completed four evaluation batches, one real remote reflection request and candidate evaluation |

The six promotion failures shared an adapter bug: it rejected zero-length
rank-local scheduler shards even though Framework's DCP writer deduplicates
replicated state onto rank zero. The corrected shared validator accepts empty
rank-local files while requiring metadata, nonempty data in every required
component, regular files and checkpoint-local paths. Full-state restore,
iteration/lineage checks, per-rank receipts and PBT optimizer verification remain
in place. A read-only replay on 2026-10-07 reproduced the original failure and
accepted corrected plans for all six saved eight-GPU parent checkpoints, without
changing old runs or submitting jobs. This was **planning verification, not
successful DCP deserialization or GPU continuation**. The subsequent GPU rerun
below closes the continuation/PBT gap; Hyperband-ES curve pruning remains
unverified on GPU.

The older `automl-wts-cu128-algorithms-20261006` image/SQSH and matching Framework
wheel are unchanged and contain the bug. Use the corrected artifacts below for
new runs: editing source does not upgrade an installed image. Preserve existing
experiments, wheels and sealed inputs instead of modifying them in place.

The structured LLM failures were separate from checkpoint promotion. Their sanitized receipts
do not distinguish HTTP, JSON parsing and response-shape errors; GEPA's successful
request to the same endpoint/model does not validate those structured controllers.
The live reproduction, corrected proposal paths and interrupted GPU rerun are
recorded below; the three full GPU searches have not yet completed successfully.

Every SFT stage scored 8/16; GEPA's baseline and candidate also tied at 8/16,
so the original prompt was retained. This was neither full-data training nor a
quality comparison, and no final test split was evaluated. The tokenizer warning
noted above remains unresolved.

### Bounded promotion rerun (2026-10-07)

After explicit approval, a new Framework wheel and Docker image containing the
checkpoint guard fix were built. Only `automl/evaluate_trial.py` and
`automl/workload.py` differed within the installed Framework package; the core
0.2.0 wheel and native DCP loader were unchanged. Image:
`nvcr.io/nvstaging/tao/cosmos-framework:automl-wts-cu128-promotions-20261007`,
registry digest `sha256:6bc4f17c47ddc27e7b4319a799a2cbf3fc2818987b386aa57df01c366fd4b9a3`.
CPU conversion job **36105691** completed in 27m11s and verified all 897 installed
package files. SQSH conversion and GPU startup performed no package installs
or source patches. The separate CPU controller used the same new Framework wheel.

Fresh experiment IDs retained the same WTS 64-training/16-validation subset,
seed 42, eight GPUs per stage, maximum five cumulative optimizer steps, and
at most four independent algorithms concurrently. Controller **36106826**
completed in **44m07s**, with native `COMPLETED`, exit `0:0`:

| Mode | Completed stages | Successful promotions/continuations |
| --- | ---: | ---: |
| Hyperband | 3 | 1 |
| Hyperband-ES | 3 | 1 |
| BOHB | 6 | 1 |
| ASHA | 3 | 1 |
| DEHB | 6 | 1 |
| PBT | 4 | 2 |

All **25 GPU jobs** completed with exit `0:0`, native training `SUCCESS`, exact
target-step DCPs and evaluation of all 16 validation records. All seven
continuations had eight matching restore receipts (**56 rank receipts**), at
step 1 for BOHB/DEHB or step 3 for the other modes, before finishing at step 5.
PBT exercised both self-continuation and donor transfer to another population
member with a perturbed learning rate; both stages passed native optimizer
configuration verification on all ranks. BOHB's KDE and DEHB's differential-
evolution proposal branches also ran. Every mode produced a full-budget winner.

No failed stage or automatic retry occurred; no LLM or GEPA request was made.
The rerun's 32 sealed inputs and the original matrix's 49 sealed inputs retained
their checksums. Original runs, checkpoints and the paused full-data run were
preserved. No further workload code changes or new unit tests were needed.

Every stage scored 8/16. This validates the bounded promotion workflows, not
full-data quality, search superiority or production-scale convergence.
Hyperband-ES's curve-pruning branch was not reached by this tiny schedule. This
run did not investigate the separate structured LLM failures or tokenizer warning.

### Structured LLM client correction (2026-10-07)

Inspection of the deployed core 0.2.0 wheel confirmed that its client omitted the
original client's explicit JSON-output request and Markdown-fence handling. It
called `json.loads` directly on response text. An offline replay with that exact
wheel reproduced rejection of a complete, valid JSON object inside a code fence.
Its catch-all error handler also erased the distinction between HTTP failure,
malformed response envelopes and invalid JSON content. **The original three
service responses were not retained, so this replay does not establish their
historical root cause.** At this stage no new live API request had been made;
the subsequent real-endpoint comparison is recorded below.

Core **0.2.1** requests JSON mode for structured controllers and accepts one
complete bare or fenced object. It rejects truncated completions, tool calls,
refusals, duplicate keys and nonfinite JSON; domain and native Framework recipe
checks remain mandatory. GEPA keeps its text format. Safe receipt metadata now
identifies the failure category, HTTP status, finish reason and available usage
without logging credentials, raw error bodies or exception strings. There are
still no automatic retries or format fallbacks.

Offline operational checks passed all three controller proposal paths, including
autoresearch proposal/prescreen, using the saved WTS search inputs and native
Framework validation. The 36 checks also covered cached/no-retry behavior,
malformed/truncated replies, transport/HTTP failures, secret non-disclosure and
unchanged GEPA text handling. Existing tests passed: 28 core, 14 Framework adapter,
and 23 original LLM/hybrid/autoresearch tests. No new unit tests were introduced.
These are CPU/loopback checks, **not real-endpoint or GPU search validation**.

New local core and Framework wheels carry the fix and updated dependency minimum.
The client executes on the CPU controller; this change does not require modifying
the workload image/SQSH. Existing installed controller environments, images, old
wheels and failed runs are unchanged. Install the new core in a separate controller
environment and use fresh experiment IDs after reviewing bounded service calls.
Never erase an error receipt or upgrade an old ledger in place to force a retry.

### Live structured LLM verification (2026-10-08)

After approval for at most six CPU-only requests (2,048 requested output tokens
and a 120-second network timeout per call), five calls used the same endpoint,
model and saved WTS search inputs as the failed matrix. Only parameter domains,
scalar observations and research instructions were disclosed; no videos, weights
or training logs were sent. No GPU job was launched and no call was retried.

The original **0.2.0 wheel** reproduced the failure: HTTP **200**, finish reason
`stop`, and a complete JSON object enclosed in a Markdown `json` fence. Direct
JSON parsing failed at character zero, while removing that single fence produced
a valid object. This confirms the ported client's format-handling regression in
a live reproduction, not an authentication, timeout or token-exhaustion failure.
The exact discarded responses from October 6 cannot be recovered.

The corrected **0.2.1 client** then passed all three proposal paths:

| Mode | API calls | Verified result |
| --- | ---: | --- |
| LLM | 1 | In-domain candidate passed native Framework recipe validation |
| Hybrid | 1 | Approved BFBO phase and generated candidate passed validation |
| Autoresearch | 2 | Candidate proposal, domain/workload checks and prescreen selection passed |

All four corrected requests returned HTTP 200, finish reason `stop`, and bare
JSON objects using explicit JSON-output mode. Provider-reported usage across
all five diagnostic calls was **931 prompt tokens + 1,528 completion tokens =
2,459 total tokens**. The unused sixth call was not sent. No further core code
change or wheel rebuild was needed after the October 7 correction.

This validates real-endpoint request handling and proposal acceptance against the
saved Framework configuration. It is **not** a rerun of the three full GPU AutoML
searches, a fresh baseline evaluation, or evidence of model-quality improvement.
Autoresearch reused its recorded successful baseline as controller input. Failed
runs, old receipts, wheels, images/SQSH and deployed controller environments were
preserved. The separately approved GPU rerun with fresh IDs is recorded below.

### Structured-controller GPU rerun (2026-10-08)

Controller **36214405** used a fresh Python 3.12 environment with core **0.2.1**
and the existing corrected promotion image/SQSH. Three searches ran concurrently,
each requesting eight A100 80GB GPUs, on the unchanged WTS smoke subset (64
training records, 16 validation questions, five LoRA steps). The reviewed budget
was one LLM trial, two hybrid phases and two autoresearch trials, with at most
five API calls; no image rebuild, automatic retries or full-data training occurred.

| Mode | First GPU job | Observed result |
| --- | --- | --- |
| LLM | 36214489 | Structured proposal passed; five training steps succeeded; evaluation canceled |
| Hybrid | 36214490 | First BFBO phase proposal passed; five training steps succeeded; evaluation canceled |
| Autoresearch | 36214485 | Fresh baseline completed five training steps; evaluation canceled before proposal/prescreen |

All three controllers raised `subprocess.TimeoutExpired` when native `squeue`
queries exceeded the executor's **30-second scheduler-command timeout**. This
was not the 120-second LLM request timeout, output-token budget, trial workload
timeout or overall experiment deadline. Independent login-node queries also
stalled. The underlying scheduler delay was not diagnosed. The executor let the
polling exception escape before its `sacct` fallback; the operational wrapper
then requested cancellation, and the GPU workers honored their cancellation
markers during evaluation. Native accounting ultimately reported all four
allocations `FAILED`, exit `1:0`; worker receipts identify cooperative cancellation,
not a training failure. All allocations are terminal and the controller credential
sidecar was removed.

Only **two API calls** were made, both HTTP 200 / finish reason `stop`, totaling
**348 prompt + 566 completion = 914 tokens**. No evaluation finished,
`best.json` contains `trial: null`, and neither hybrid's second phase nor autoresearch's
proposal/candidate was reached. Training checkpoints, logs, request receipts,
sealed inputs and prior experiments are preserved. The interrupted experiment
ledgers retain their last `RUNNING` state; consult terminal scheduler/worker and
matrix receipts rather than treating those stale ledgers as live jobs.

The JSON correction passed again, but this run does **not** establish completed
GPU AutoML search or model-quality improvement. The scheduler correction below
does not change these failed runs. Do not erase cancellation markers or receipts,
upgrade an old ledger, or blindly retry paid work to bypass this failure.

### Scheduler observation recovery correction (2026-10-08)

Core **0.2.2** separates unavailable scheduler observations from failed trials.
Timed-out/nonzero `squeue` calls now reach the `sacct` fallback. When neither
query establishes status, the controller persists safe diagnostics, retains the
native job identity and retries only read-only queries with capped backoff, within
the original elapsed budget. No training, submission or LLM request is retried.
Missing/conflicting identities and invalid completion receipts still fail closed.
Cancellation interrupts backoff; if terminal cancellation cannot be confirmed,
the controller records `ERROR` with an explicit reconciliation reason instead
of claiming the native job stopped. See the time, review boundaries, and failure
handling section in the [SLURM guide](automl-slurm.md).

Validation passed **65 existing tests** (28 core, 14 Framework adapter, 23 original
LLM/hybrid/autoresearch) and **26 CPU operational checks**. The latter exercise
actual tiny child processes through simulated scheduler CLIs, including real
subprocess timeouts with a scaled test timeout, accounting fallback, both-query
outage/recovery, read-only inspection, cancellation, deadline expiry, identity
and receipt guards, pause/resume and no duplicate submission. All simulated jobs
were terminal. No new unit-test suite, cluster allocation or paid API call was
introduced. This is **not live SLURM end-to-end validation** of the fix.

Fresh local core and Framework wheels were installed in an isolated Python 3.12
controller environment. The core dependency check, installed CLI/structural plan,
42 core/adapter tests and the same 26 operational checks passed outside the source
checkouts. Framework was installed with `--no-deps` for planning only; this is
not a complete GPU environment dependency check. Lint, focused format checks,
skill validation and whitespace checks passed.

Only the core's version, scheduler/runner modules and packaged skill changed
relative to 0.2.1. All **897 Framework package files** match the previous wheel;
its dependency metadata now requires core 0.2.2. The existing workload image/SQSH
needs no rebuild for this controller fix. Old wheels, deployed environments,
failed runs and checkpoints were not modified. Install the supplied local wheels
in a new controller environment and review a fresh bounded run before claiming
that the three structured GPU searches pass. No registry or package-index release
was performed.

### Bounded infrastructure recovery verification (2026-10-08)

The v1 development controller now supports opt-in, capped same-candidate
infrastructure retries and phase-aware worker/progress monitoring; see
[SLURM recovery](automl-slurm.md). This fixes waiting solely on native RUNNING
and stopping immediately on every infrastructure failure. It does not classify
bare NCCL errors or DRAIN alone as automatically retryable.

Validation passed **65 existing tests** (28 core, 14 Framework adapter, 23
original structured-controller tests), **26 existing CPU lifecycle/outage
checks**, and **54 recovery operational checks**. The final installed-wheel
matrix used **89 simulated allocations with real tiny CPU child processes**;
all were terminal. It exercised all twelve SFT controller modes, retries without
additional loopback LLM requests, promotion/PBT lineage through successful retry
attempts, bounded exhaustion/deadlines, stalled workers, unconfirmed cancellation,
and linked native Framework job records. GEPA's batch execution path also passed
retry collection and global retry/example accounting checks; this was not a new
full GEPA reflection search. No new unit-test suite was introduced.

Fresh local core **1.0.0.dev0** and Framework **1.2.2** wheels passed independent
CPU installation, CLI/structural planning, and the 42 core/adapter tests outside
the source checkouts. Core dependency checks passed before Framework's deliberate
planning-only `--no-deps` installation. All 18 core and 822 Framework Python files
in the wheels match source. A separate Framework structural promotion check used
a successful retry's checkpoint directory while preserving logical parent lineage;
it did not deserialize DCP tensors. Skill validation, focused lint/format and
whitespace checks passed. The installed engine build is
`c00f243857a430fd65e8ea06d6838f678d4f089f8b25901f5ac14c855f5cdeb1`.

These are CPU/simulated-scheduler checks, **not live SLURM recovery validation**.
Existing jobs, installed controllers, images/SQSH, manifests and ledgers were not
upgraded or retried. No GPU allocations, external LLM calls, registry pushes or
package releases were made for this correction. Package versions remain unchanged;
use exact new wheel hashes for a separately reviewed deployment.

### V1 SLURM results (reviewed 2026-10-12)

The following results supersede the earlier pending structured-controller GPU
validation. Native SLURM accounting and saved experiment, worker, checkpoint and
evaluation artifacts were checked on October 12. No jobs remained running or
queued. These experiments used core **1.0.0.dev0**, engine build
`a43b199b4dce126bdebb2b93bdb98fb70291b7c38e831ca7f69d995d2821f391`,
before the opt-in infrastructure-recovery build above. Automatic trial retries
were disabled; the later recovery changes were not applied to these runs.

The three structured-controller smokes completed native training, checkpoint
export and held-out evaluation on eight GPUs per trial, with 64 training records,
16 validation questions and five LoRA steps:

| Mode | Completed trials | Native GPU jobs | API calls |
| --- | ---: | --- | ---: |
| LLM | 1 | 36216399 | 1 |
| Hybrid | 2 | 36216406, 36216671 | 2 |
| Autoresearch | 2 | 36216357, 36216678 | 2 |

All five GPU jobs reported `COMPLETED`, exit `0:0`, and all scores tied at 8/16.
The five API calls reported 3,551 total tokens. This establishes bounded workflow
completion, not model-quality improvement or a real scheduler-outage recovery.

The separate full-data searches used **5,555 training records and 2,676 validation
records**, eight GPUs per stage, and a full budget of **695 optimizer steps**.
Every completed winner evaluated all 2,676 validation records. Counts below are
successful logical trials/stages versus the approved cap; promotions also count.

| Mode | Successful / approved | Final search state | Best full-budget validation |
| --- | ---: | --- | --- |
| Random | 1 / 1 | COMPLETE | 2355 / 2676 (88.00%) |
| BFBO, separately approved retry | 2 / 2 | COMPLETE | 2355 / 2676 (88.00%) |
| Hyperband-ES | 3 / 3 | COMPLETE | 2357 / 2676 (88.08%) |
| ASHA | 3 / 3 | COMPLETE | 2357 / 2676 (88.08%) |
| PBT | 4 / 4 | COMPLETE | 2357 / 2676 (88.08%) |
| BOHB | 6 / 6 | COMPLETE | 2364 / 2676 (88.34%) |
| Bayesian | 1 / 4 | ERROR | 2355 / 2676 (88.00%), from the completed first trial |
| Hyperband | 1 / 3 | ERROR | No full-budget result |
| DEHB | 3 / 6 | ERROR | No full-budget result |

Bayesian's second allocation **36219715** reported NCCL collective errors and
eventually `TIMEOUT` after four hours; training last reported 600/695 steps.
Hyperband's second allocation **36219331** reported no training progress beyond
startup and also timed out after four hours. DEHB's fourth initial-stage
allocation **36224004** ended in `NODE_FAIL`; its three successful initial stages
did not reach a full-budget promotion. These observations do not establish a
hardware root cause for the NCCL errors or Hyperband startup stall.

The original BFBO allocation **36216360** also ended in `NODE_FAIL`. Its separately
approved fresh two-trial search completed under controller **36220473**, with
GPU allocations **36220556** and **36223154**. This was a manual, reviewed rerun,
not automatic failure recovery. The main matrix controller **36216292** ended
`FAILED` because some searches failed; that aggregate state does not invalidate
the successful independent searches.

Six of nine full-data searches completed. BOHB's score is the highest observed
validation result in this bounded batch, not proof of algorithm superiority or
final-test performance. Bayesian, Hyperband and DEHB still need separately
reviewed reruns. Live validation of the newer infrastructure-recovery build and
Hyperband-ES curve pruning remain outstanding. No jobs were restarted during
this status review.

### Focused #277 companion regression (2026-10-12)

The focused AutoML change in [PR #288](https://github.com/NVIDIA/cosmos-framework/pull/288)
is based directly on #277's head, without the broader workflow/skills migration.
It supersedes the fork-only PR #4; #277 and #288 retain their coordinated merge
hold. The following results validate this split itself, not just its predecessor.

Runtime commit `14505ff0933dd303fc2793319cb6f8c6e95f2924` was packaged into a fresh
Framework wheel and image. The previous Framework installation was removed before
installation; all 854 installed Framework files and 19 core files matched the
reviewed wheels, including absence of omitted workflow modules. A separate Python
3.12 CPU controller imported the adapter without Torch. Package versions remain
Framework **1.2.2** and core **1.0.0.dev0**; no release or version bump occurred.

| Artifact | Identity |
| --- | --- |
| Framework wheel SHA256 | `c600ccb7bcff213fac5d0718eeead0bd1c126c6de0ceecec063bead0340eee34` |
| Core wheel SHA256 | `ed279d7329de333380b9d649c541b6b75d30d9500ca42f3b3ee8c35d8ee3305f` |
| Validation image | `nvcr.io/nvstaging/tao/cosmos-framework:automl-wts-cu128-focused-20261012` |
| Registry digest | `sha256:a4dd7851d3af16bcc89deb6729ae1417681ed128158a06f5629ea1550595f2a9` |
| SQSH SHA256 | `0f162751b85e480a77d99e937ee7c093d7bb4958ff792814b94c0e2ce7e0293a` |

CPU conversion job **36427241** completed in 27m59s and verified the installed
packages inside the SQSH. GPU startup used this immutable image without source
overlays or package installation. The existing checks passed: **14 adapter tests,
103 shared-helper tests and 28 core tests**, plus focused Ruff lint/format, skill
discovery/validation, documentation links, import closure and whitespace checks.
No new unit tests were added. All 41 retained Python files have identical syntax
trees to the previously validated broader stack.

Independent algorithms ran concurrently on SLURM using the same seeded WTS
**64-training/16-validation** subset, eight GPUs per SFT stage, at most five
cumulative optimizer steps, and one GPU per GEPA batch. The four-hour matrix
deadline, 37 SFT-stage cap and six API-call cap were preserved. Automatic
allocation retries were disabled.

| Mode | Completed SFT stages / evaluation batches | Result |
| --- | ---: | --- |
| Random | 1 | COMPLETE |
| Bayesian | 4 | COMPLETE, including GP/expected-improvement proposal |
| BFBO | 2 | COMPLETE |
| Hyperband | 3 | COMPLETE, including promotion |
| Hyperband-ES | 3 | COMPLETE, including promotion |
| BOHB | 6 | COMPLETE, including KDE proposal and promotion |
| ASHA | 3 | COMPLETE, including promotion |
| DEHB | 6 | COMPLETE, including differential-evolution proposal and promotion |
| PBT | 4 | COMPLETE, including continuation and donor transfer |
| LLM | 1 | COMPLETE |
| Hybrid | 2 | COMPLETE |
| Autoresearch | 2 | COMPLETE |
| GEPA | 4 evaluation batches | COMPLETE, including live reflection and candidate evaluation |

All **41 GPU allocations** reported native `COMPLETED`, exit `0:0`. All 37 SFT
stages had native training/evaluation success, valid model/optimizer/scheduler/
trainer DCP components, exact target iterations and all 16 evaluation records.
Seven promotions/continuations had **56 matching rank restore receipts**; both
PBT continuations verified optimizer configuration on all eight ranks. Every SFT
mode produced a winner, at resource five where multi-fidelity applies. All six
API calls completed with HTTP 200 and reported **5,997 total tokens**. The 53
sealed input files retained their hashes, controller credential sidecars were
removed, and no tracked allocations remained running or queued.

The initial matrix **36429196** ended `FAILED`, exit `1:0`, after 38m16s because
Bayesian, BFBO and PBT's first `sbatch` calls exceeded the existing core's
30-second command timeout. Their unknown submission outcomes were reconciled
through repeated queue/accounting queries on the submission host and two reachable
peers, with no matching jobs, worker receipts or native logs. A fourth configured
login name did not resolve. No blind resubmission occurred. Fresh searches under
controller **36429728** used only the ten unspent stages and the original absolute
deadline, completing in 26m11s. Final accounting again found no late jobs for the
three original names. Original error receipts and stale ledgers were preserved;
the aggregate failure must not be relabeled as a clean first-attempt pass.

This is successful bounded algorithm verification with **manual submission
reconciliation**, not validation of automatic submission recovery. All SFT stages
scored 8/16 and GEPA's best validation score was also 8/16; this is not a quality
comparison or full-data convergence result. Hyperband-ES curve pruning and live
fault recovery remain unverified. The existing GEPA entrypoint calls the evaluator
directly and does not emit its declared `evaluation/status.json` progress file;
its completion was checked using worker exits, metrics and exact per-example IDs,
not that missing signal. GEPA progress publication/no-progress monitoring remains
a follow-up gap, not a regression introduced by this split. Independent non-author
validation, core distribution/dependency review and release integration are still
required before the coordinated merge.

### Automatic submission reconciliation correction (2026-10-12)

The lost `sbatch` replies above exposed a missing controller capability, not an
algorithm failure. A new local core build implements
[bounded submission reconciliation](automl-slurm.md#submission-reconciliation):
adopt an existing job, or atomically fence an unobserved attempt before using the
approved retry allowance. Conflicting identities, unavailable scheduler evidence
or an unavailable fence return a blocked error without blanket cancellation or
another launch. A late fenced batch cannot start the workload or overwrite its
receipts, although it may consume a short native allocation. Both retry caps
still default to zero; the earlier matrix's zero-retry policy would not authorize
automatic replacements unchanged.

The local core wheel remains **1.0.0.dev0**, SHA256
`8ea4df261d92cecb2a5579fc10c04a111851c95fc3cd388bc4e30974e07265c0`, with
`engine_build=7cdb3a60e88fd91322595b4f53d0307c6f04e85137045f871f2821a53f266ab3`.
All 19 runtime Python files matched the source and the installed wheel. A fresh
Python 3.12 controller imported the unchanged Framework **1.2.2** adapter without
Torch; this planning-only installation is not a complete GPU training environment.
No version or dependency changes were made for this correction.

Against the installed wheel, **28 existing core tests**, **14 existing adapter
tests**, and **112 CPU operational checks** passed:

- **32 submission checks:** real subprocess acknowledgement timeouts against a
  simulated scheduler, delayed visibility/adoption, absence fencing, blocked
  outages/conflicts, late and duplicate batch rejection, bounded lock contention,
  independent-process claim/fence races, restart recovery, retry-linked Framework
  records, exhausted/disabled caps, cancellation, LLM proposal reuse and GEPA
  evaluation-batch retry accounting.
- **54 existing recovery checks:** all numerical/LLM controller modes and GEPA,
  terminal infrastructure failures, successful-parent checkpoint selection,
  watchdogs, original deadlines and global retry/example caps; all 89 simulated
  allocations reached terminal states.
- **26 lifecycle/outage checks:** pause/resume, cancellation, deadlines, startup,
  scheduler visibility, identity/receipt validation and bounded query outages.

Ruff lint/format, both AutoML skill validations, discovery/local documentation
links and whitespace checks passed. No new unit tests were added; the existing
documentation-link check was corrected to distinguish external links and local
anchors. The future matrix wrapper preserves a returned blocked state instead
of converting it into an exception and blanket cancellation. Historical wrappers,
ledgers, wheels and completed runs were not changed.

These checks used local CPU subprocesses, a simulated scheduler and loopback LLM
responses, **not live SLURM, GPUs or external API calls**. Cross-node shared
filesystem `flock` semantics and live fault recovery still need validation. The
core stages its supervisor/fence from the controller wheel, so this correction
does not require a training-image rebuild; no existing cluster controller was
upgraded. The earlier GPU results used the earlier core hash and do not validate
this new build. The separate GEPA progress-file gap above remains open, as do
the core distribution/dependency review and coordinated merge gates.
