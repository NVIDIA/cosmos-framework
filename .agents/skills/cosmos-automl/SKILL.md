---
name: cosmos-automl
description: Optimize Cosmos Framework reasoner SFT hyperparameters or evaluation prompts with bounded AutoML. Use for numerical, multi-fidelity, population or LLM-guided search, GEPA Auto-Prompter, and monitoring or resuming those experiments.
---

# Cosmos Framework AutoML

Read [docs/automl.md](../../../docs/automl.md) before planning a search. It owns
the installation/release prerequisites, manifest contract, result semantics
and current limits. Use the native `cosmos-automl` CLI; do not call a TAO runner
or introduce model/runtime selection. The engine is an optional wheel, not a
source-checkout dependency.

Reuse **cosmos3-setup** for environment preparation and
**cosmos3-post-training** for a prepared reasoner recipe and dataset/checkpoint
compatibility. AutoML starts from that explicit recipe, not a guessed model ID.

For SFT, read [algorithm contracts](../../../docs/automl-algorithms.md) for the
selected mode. For GEPA, read [prompt optimization](../../../docs/automl-prompts.md)
and use `cosmos-automl prompts <verb>`; it evaluates a fixed checkpoint, not SFT.
Keep this one skill entrypoint for both workflows.

- Collect held-out data, objective **and direction**, domains, fixed training
  budget (or cumulative step fidelity), seed, immutable data/code revisions, GPU allocation, and both trial
  count/time limits. Check training/validation separation. Keep final test data
  out of the optimization loop.
- Ask among supported installed venv, Docker, SLURM, Kubernetes, Brev and Lepton
  environments when not selected; none is a default. Images
  and assets must be prepared separately with approval. Do not silently pull,
  download, increase resources, or fall back to another execution environment.
- For Kubernetes, Brev or Lepton, read the [cloud execution guide](../../../docs/automl-platforms.md).
  Require persistent same-path storage and cross-client POSIX locks, an explicit
  cluster/instance/workspace identity, and reviewed GPU/image/Secret references.
  Use one GPU per SFT rank or exactly one for GEPA. These executors do not
  provision infrastructure or transfer data. Brev instance billing continues
  after its container stops. Keep the CPU controller alive for queue/startup
  cancellation; Lepton/Brev worker deadlines are not native allocation limits.
  Native training retries are disabled; only read-only submission adoption and
  observation retries are automatic. Live cloud GPU validation remains pending.
- Run `cosmos-automl plan --manifest <yaml> --workspace <directory>`. This is a
  CPU-only structural check, not a full Hydra/GPU preflight. Review the actual
  native commands, resource allocation, mounts, metric, budgets and prerequisites.
  Check the selected evaluator's decoder and export dependencies as well as
  training; a training-only GPU probe does not establish evaluation readiness.
- For SLURM, read the [per-trial execution guide](../../../docs/automl-slurm.md)
  for controller, image and credential preparation. Use the user's configured partitions,
  check their live wall-time limits, and keep the controller on CPU. Never move
  a search to a different queue just to fit one long allocation. Queue time and
  trial runtime have separate budgets; the experiment deadline includes both.
  A first-trial review can use `search.pause_after_trials: [1]`; resume only
  after reviewing full-data results and timing. Pausing releases GPU resources
  but does not reset the experiment clock.
- Use the user's private SLURM node-fault history when preparing new launches;
  follow [node exclusions](../../../docs/automl-slurm.md#private-node-history-and-exclusions).
  Review historical incidents against current inventory and seal the selected
  names in `execution.slurm.exclude_nodes`. Keep cluster/node details outside
  public skills and examples. Do not infer whole-block faults or permanent
  quarantine from closed incidents, or edit a live experiment to add exclusions.
- Supervise the native CPU allocation before the brain starts; trial retries
  cannot handle bootstrap/import/prolog failures. Follow the SLURM guide's
  [startup contract](../../../docs/automl-slurm.md#before-the-search-controller-starts):
  arm the cross-node observer automatically before submission, preserve startup
  receipts, and mark the boundary before any algorithm/API/job fan-out. Only
  confirmed pre-fan-out infrastructure/observer failures may use approved CPU
  startup retries. Unknown submission outcomes block replacement. A missing
  experiment ledger is not success; report unlaunched modes as `NOT_STARTED`.
- If the CPU allocation can expire first, review the SLURM guide's opt-in
  `--controller-max-allocations`, allocation seconds and grace in both plan and
  run. CPU handoff requeues the same CPU job and adopts existing GPU jobs; it
  adds no trials or LLM calls and never extends the experiment deadline.
  `SUSPENDED` may retain an active GPU trial; it is not the `PAUSED` review gate.
  Preserve the shared environment and credentials across allocations. Do not
  blanket-cancel or delete the secret sidecar on every batch exit. A lost
  requeue acknowledgement needs native observation, not another submission.
  This requires the matching new wheel and a reviewed requeue-enabled CPU
  wrapper; do not retrofit a queued/live matrix or claim live rollover proof
  from the CPU simulation.
  Combine startup supervision with handoff using the startup helper's `probe`
  verb, a fresh observed attempt per native job/restart generation, and `fanout`
  before directly execing the CLI. Do not put the handoff CLI inside `run`:
  it must receive scheduler signals itself. An early bootstrap signal must
  stop the batch, and a missing ledger after fan-out is not retry permission.
- Obtain approval before `run ... --approve` or `resume ... --approve`.
  Approval covers the reviewed search budget, not unbounded trial retries.
- Review the SLURM guide's bounded recovery policy and worst-case allocation
  count in `plan`. Same-candidate retries need both per-trial and total caps;
  watchdog thresholds must accommodate native progress-report gaps and loading.
  Node failure/preemption can retry, but DRAIN alone or a bare NCCL timeout cannot.
  Confirm termination before replacement. After the cancellation window, the new
  core keeps the brain alive with `cancellation_reconciliation.state: WAITING`,
  observing the same allocation within the original deadline and CPU handoff caps.
  Do not restart a waiting brain; native confirmation enables the existing retry
  policy or next candidate. Identity conflicts and expired limits still block.
  For heartbeat failures, distinguish worker/batch publication from training
  progress and correlate node events, storage errors and job logs; exclusions
  are placement safeguards, not proof of node health. Preserve failed `attempts` and use
  the successful `attempt_id` for artifacts and promotion parents. Do not upgrade
  live controllers or old ledgers to enable recovery.
- For promotions/PBT, inspect stage counts, full-horizon scheduler, donor DCP
  state and topology. Each stage counts as a trial; partial-budget scores cannot
  win. PBT may mutate LR/weight decay, not batch/accumulation or architecture.
  Native DCP deduplication can leave empty rank-local shards; require nonempty
  component data, not nonempty files on every rank. Check continuation on the
  intended topology and inspect every rank's restore receipt before a larger run;
  successful initial stages or read-only promotion plans do not prove restoration.
- For LLM modes, review endpoint/model, data disclosure and call/token caps.
  Credentials are environment names. Proposals are bounded configuration, not
  executable code. Incomplete API receipts are not permission for paid retries.
  Inspect `error_kind`, HTTP status, finish reason and usage before reviewing a
  new bounded call. Structured modes request JSON output; GEPA requests text,
  so successful reflection does not prove proposal compatibility. Validate the
  request/proposal path on the CPU controller before another GPU search.
- GEPA needs media-disjoint reflection/validation/test roles. Review queries,
  model outputs and sanitized feedback disclosed to the reflection endpoint
  before enabling `allow_reflection_data`; no image upload is enabled by the
  Framework adapter. Budget final aggregate reranking and optional test scoring
  as well as GEPA's internal metric calls. Hand off the prompt and metrics,
  not a newly trained checkpoint.
- Use `status` / `logs` for evidence. `cancel ... --approve` stops native jobs.
  An `UNKNOWN` launch is an investigation gate, never permission to resubmit.
  Lost/invalid SLURM submission replies now enter bounded core reconciliation;
  review `execution.slurm.submission_reconcile_seconds` (default 120). Existing
  jobs are adopted. Unobserved attempts require a shared POSIX-lock launch fence
  before replacement, and consume the reviewed retry caps even if they appear
  late only to be rejected. Conflicts/unavailable evidence remain blocked.
  Do not blanket-cancel a blocked result or repeat sbatch outside the controller.
  A `scheduler_unavailable` observation is not a failed trial: the controller
  retries read-only SLURM queries within the original deadline. Do not wrap a
  transient status error in blanket cancellation or restart a paid request.
  If cancellation is unconfirmed, reconcile native jobs and worker receipts;
  neither a stale ledger nor a stop marker proves that GPUs have been released.
- Hand off `best.json`, the winning native TOML, DCP and held-out metrics, or
  report that no trial succeeded. Do not call validation improvement final test
  performance or describe fixture checks as GPU end-to-end verification.

Execution within each experiment is sequential (also for ASHA), with single-node
trials. Consult the current validation results in [AutoML](../../../docs/automl.md):
bounded SLURM smokes passed fixed-budget search, GEPA, eight-rank checkpoint
continuation and PBT donor transfer. Real-endpoint LLM/hybrid/autoresearch
searches also passed bounded GPU smokes. Six of nine full-data numerical/population
searches completed; Bayesian, Hyperband and DEHB did not complete their approved
budgets. Live infrastructure recovery and Hyperband-ES curve pruning remain
unverified; smoke success is not production-scale or model-quality validation.
Keep the core at `1.0.0.dev0` during the port; local builds are not releases.
Use wheel hashes and `engine_build` to distinguish builds, not version bumps.
Preserve exact old wheels/manifests for paused searches; changed builds fail resume
identity checks even when package versions match.
Route ordinary training/evaluation to the existing skills, and do not infer
video-generator objectives or arbitrary-code research support.
