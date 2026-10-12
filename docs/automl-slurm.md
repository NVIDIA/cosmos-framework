# AutoML on SLURM

The local core wheel provides a single-node SLURM executor. The controller
runs on a CPU node **on the cluster**, with `sbatch`, `squeue`, `sacct`,
`scontrol`, `scancel`, `srun`, a compatible Python environment, and shared
storage. It submits one GPU allocation per candidate, waits, collects the
native result, and feeds it to the selected search controller. There is no SSH
executor, per-trial TAO runner, or independent one-trial search workaround.

The 0.1.1 path completed one full-data WTS trial on eight A100 GPUs and paused
for review; it has not completed the full four-trial search. That evidence does
not validate 0.2.0 promotion/PBT or GEPA GPU workflows. CPU checks, simulated
scheduler checks, and `sbatch --test-only` are not GPU end-to-end validation.
For the later 0.2.0 algorithm matrix, see [bounded SLURM verification](automl.md#bounded-slurm-algorithm-verification-2026-10-06).
The corrected image passed native continuation/PBT in the subsequent
[six-mode promotion rerun](automl.md#bounded-promotion-rerun-2026-10-07).

## Environment and review

Use an existing authorized SSH connection to the cluster; verify the host key
and use batch-mode authentication. Credentials come from the session environment
or a user-approved env file, never chat, command-line arguments or job records.
Check `sinfo`, `scontrol show partition`, account access, shared-storage capacity,
and Pyxis/Enroot availability before preparing a run. Conversion to SQSH belongs
in a separate CPU allocation; do not spend GPU training time installing packages
or importing an image. Verify its installed Framework files against the exact
wheel before submission. Do not overlay an unrelated source checkout at launch.

If the CPU controller needs an API key, forward only its environment-variable
name in manifests. An approved sidecar must be mode 600, outside synced results,
and removed after the controller terminates. Never place it in the image or PR.
Bind a parent job record before submitting the controller, record its native
job ID, and monitor `squeue`/`sacct` and logs until terminal or explicit handoff.
Preserve existing runs and use fresh experiment IDs for a changed build.

Use the user's `SLURM_PARTITION` and account. Resolve them into the reviewed
manifest; never fall back to another partition to obtain a longer wall time.
The executor rechecks each named partition's `State` and `MaxTime` before a run.
It also uses `sbatch --test-only` before each actual submission. Site QoS,
account access, Pyxis availability and placement remain cluster prerequisites.

Prepare the image as a shared `.sqsh` before GPU submission. Pin its SHA256,
and list shared model/data paths as read-only `execution.mounts`. Only the
individual trial directory is mounted writable. Mount destinations retain their
declared spelling, including shared-storage aliases used in recipes; do not
replace them with symlink-resolved paths that differ inside the image. Do not forward
`CUDA_VISIBLE_DEVICES`, `SBATCH_*` or `SLURM_*` in `execution.environment`:
allocation identity and GPU visibility belong to the scheduler. Forward
`NVIDIA_DRIVER_CAPABILITIES` by name after exporting `compute,utility,video`
for CUDA video decoding; credentials, if needed, are names only as well.

The CPU controller needs the Framework and core wheels plus their planning
dependencies, not GPU training packages. Verify its Python version/imports,
native scheduler CLIs, and shared interpreter path on compute nodes. Do not
assume the system Python or the interpreter inside the training image is a
suitable host-side controller environment.

For an isolated planning-only controller, install the core wheel with its
dependencies, then the Framework wheel with `--no-deps`:

```bash
python -m pip install /shared/wheels/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl
python -m pip install --no-deps /shared/wheels/cosmos_framework-1.2.2-py3-none-any.whl
python -m cosmos_framework.scripts.automl --help
```

Use the explicit user-local interpreter for these commands. This deliberately
is not a complete Framework installation: a whole-environment `pip check` will
report omitted Framework training/inference dependencies. Verify the core's
dependencies before installing Framework and exercise the installed adapter's
`plan` command outside both source checkouts. Never use this limited environment
for training; the prepared image supplies that stack. Point Python installer
caches at shared storage when the login home has a small quota.

## Manifest settings

Keep the native `workload.recipe`, `workload.evaluation`, `nproc_per_node`,
seed, domains, data revisions and objective from the prepared search. Add
`workload.expected_validation_records` when the whole validation set must be
scored. Use [reasoner-slurm.yaml](../examples/automl/reasoner-slurm.yaml) as a
planning template; replace every site/asset placeholder before approval.

`execution.slurm` describes one node:

- Explicit partition(s), account, optional site QoS, CPU/GPU counts, optional
  memory, exclusivity, allocation wall time, startup cap, cleanup allowance,
  and scheduler polling interval. Scheduler requeue remains off; bounded
  controller retries require the explicit recovery policy below.
- The existing [single-node template](../tools/cosmos_workflows/templates/slurm/singlenode.sbatch.tmpl)
  is the Pyxis step envelope. A small host-side supervisor wraps it to bound
  startup and preserve the actual step exit. It resolves granted `NumCPUs`
  and passes them to `srun`, including on exclusive nodes.
- `template`, optional `bootstrap`, and their required helpers are sealed in
  `supporting_files` as absolute shared paths mapped to SHA256 hashes. Files
  are verified before execution and exposed read-only to the container.
- `record_open` and `record_mark` are trusted argv templates for the existing
  Framework job-record writer. Open binds the trial output directory and mints
  the job name before `sbatch`. Mark records native identity and transitions;
  `COMPLETE` is recorded only after workload artifact validation. Retry-enabled
  searches also provide `record_retry_open`, the open argv with
  `--retry-of {retry_of}` (the previous job-record ID). Stage the
  helper's schema/redaction dependencies too. A parent record is opened in the
  normal launch review before submitting the CPU controller.

For the native Qwen3-VL video-QA recipe, the optional
[bootstrap](../examples/automl/slurm_bootstrap.sh) orchestrates the
[operational preflight](../cosmos_framework/automl/preflight.py) and existing
[NCCL probe](../tools/cosmos_workflows/scripts/nccl_allreduce_probe.py).
Stage and seal these scripts; set `AUTOML_PREFLIGHT_PY` and
`AUTOML_NCCL_PROBE_PY` to their shared paths and forward the names. They validate
the image's installed code rather than patching it. This probe specifically
checks the native video-QA dataset environment, disjoint IDs/video references,
Qwen3-VL, TorchCodec CUDA decoding, FlashAttention, actual candidate dryrun,
and the allocated GPU topology. Use an appropriate reviewed bootstrap for
other workloads instead of claiming these checks cover them.

## Time, review boundaries and failure handling

`search.budget.trial_timeout_seconds` includes training, checkpoint export and
evaluation **after bootstrap**. Startup + trial runtime + cleanup must fit the
allocation. The total experiment clock starts at creation and includes queue
waits, pauses and controller downtime. `sbatch --deadline` prevents late
allocations from outliving that limit; the batch supervisor also checks the
absolute deadline without relying on a live controller. It refuses to begin
when there is insufficient time for the complete allocation.

For a four-hour queue, one reviewed division is 20 minutes startup/preflight,
3 hours 30 minutes workload and 10 minutes cleanup. These are caps, not a claim
that a particular full-data trial will finish. Keep full data/step budgets;
report a timeout rather than silently shortening them.

Set `search.stop_on_failure: true` and `search.pause_after_trials: [1]` for a
first-trial review. `PAUSED` means no active trial, and the CPU controller can
exit. Inspect the final DCP, native `SUCCESS`, validation count and timings
before `resume --approve` from a CPU controller. A resume uses the same identity
and elapsed deadline. It neither restarts trial zero nor silently extends time.
The controller's own allocation must leave time for shutdown before its hard
wall limit; configure its terminal signals and timeout in the launch envelope.
Keep the original engine wheel for existing experiments: changing the engine
build changes its identity even while the development version stays fixed.
See [development version and build identity](automl.md#development-version-and-build-identity).

Pending jobs remain pending. Submission intent is durable before the scheduler
RPC; a lost response enters automatic bounded reconciliation by unique job name
and owner. An existing job is adopted without resubmission. Multiple matches or
unresolved evidence block the search, not authorize another launch. Native `COMPLETED`
requires exit `0:0`, matching worker/batch job IDs, successful receipts, and
native workload artifacts. A receipt from another node is not checked as if
its PID belonged to the CPU controller.

The controller treats scheduler observation failures separately from workload
failures. A timed-out or nonzero `squeue` query falls back to `sacct`. If neither
can establish status, the controller records `observed_state: UNKNOWN` with
`observation_error.reason: scheduler_unavailable`, retaining the last confirmed
trial state and native ID. It retries **read-only queries**, starting at
`poll_seconds` and backing off to at most 300 seconds, within the original
experiment deadline. `_scheduler_status.json` records safe query diagnostics
and recovery. No raw scheduler error text, replacement submission, LLM request
or training retry is introduced. Successful queries with missing/conflicting
identity or mismatched completion receipts still stop at the `UNKNOWN` gate.

The 30-second per-command timeout remains separate from API, trial and elapsed
budgets. User cancellation interrupts the backoff. If cancellation or the
original deadline arrives during an outage, the controller writes the cooperative
stop marker and attempts native cancellation. If terminal status cannot be
confirmed, it ends with experiment `ERROR`, an explicit reconciliation reason,
and the active trial's `observed_state: UNKNOWN`; it does not label the native
job complete or canceled. Worker and scheduler deadlines remain effective.
Inspect native status before resuming with the same engine/identity; never erase
cancellation markers or automatically submit a replacement.

### Submission reconciliation

`execution.slurm.submission_reconcile_seconds` defaults to **120** (1–900).
A timeout, transport/exit error or invalid `sbatch` acknowledgement keeps the
candidate reserved while the controller queries both queue and accounting.
The reconciliation window is persisted across resume and capped by the original
experiment deadline; in-flight scheduler calls retain their bounded timeouts.
Read-only reconciliation needs no additional allocation or LLM-call allowance.

Repeated healthy observations with no matching job can permit a replacement
only after a shared-filesystem launch fence is closed atomically. The staged
batch supervisor claims the same fence before modifying worker/execution files
or launching the training/evaluation step. If it has already claimed the fence,
the controller waits for its identity instead of replacing it. A late batch for
a closed fence exits without running a workload or overwriting its receipts.
This protocol requires shared POSIX `flock` semantics on the controller and
compute nodes; a missing, invalid or unavailable fence blocks replacement.
Worker evidence or conflicting identities also block it. Empty scheduler output
alone is never proof that the submission failed.

A fenced, unobserved attempt uses the existing recovery caps below. Every
uncertain submission counts conservatively toward the allocation envelope,
including one that later appears only to be rejected by the fence. Fencing
prevents duplicate workloads, not a possible short late scheduler allocation;
the original native deadline still bounds that allocation. With retry caps zero,
the attempt ends in error rather than allocating again. A replacement retains
the candidate, seed, promotion source and original deadline, gets fresh paths
and a retry-linked record, and does not repeat an LLM proposal.

`_submission_reconciliation.json` records observations and decisions. When the
window expires with unavailable or conflicting evidence, the controller returns
experiment `ERROR` with an explicit blocked reason, retaining the unresolved
trial rather than claiming it terminated. Verification wrappers must not turn
this result into blanket cancellation. Use the same manifest/build when resuming;
never clear an old fence, receipt or cancellation marker. New local core wheels
stage the supervisor/fence helpers automatically; this core-only correction does
not require rebuilding the training image. Historical runs are not migrated.

`cancel --approve` requests native cancellation of owned queued/running jobs.
Catchable controller termination also requests cancellation; uncatchable
crashes remain bounded by scheduler and worker deadlines. Reconcile native
terminal state after cancellation. Partial trials are not resumed across
allocations and are not treated as successful observations. Preserve all
records, logs and checkpoints.

For multi-fidelity searches, each completed stage is a separate allocation.
Only successful, fully committed stage checkpoints may be promoted; this is
not recovery of a preempted/partial allocation. Budget each stage's evaluation
cost and mount its parent checkpoint read-only. Use a workload image containing
the new stage callback. For GEPA, each uncached evaluation batch is an allocation;
use a prompt-specific bootstrap, not the SFT/NCCL bootstrap above. Planning and
reflection occur on the CPU controller; any external LLM endpoint needs its own
reviewed disclosure and call/token limits.

## Bounded failure recovery

New searches can opt into recovery in the shared controller; this is independent
of the Bayesian, Hyperband, PBT or LLM proposal algorithm. For example:

```yaml
search:
  # Merge with the prepared search; this is not a complete manifest.
  recovery:
    max_retries_per_trial: 1
    max_retries_total: 2
    backoff_seconds: 30
    cancellation_timeout_seconds: 120
    heartbeat_timeout_seconds: 180
    progress_timeout_seconds: 3600
```

Both retry caps default to zero. Watchdog timeouts default to unset, not guessed
from dataset size. `plan` includes the policy and worst-case allocation count:
`max_trials + min(max_retries_total, max_trials * max_retries_per_trial)`.
Review that additional compute before launch. Every attempt has the same resource
request and workload timeout. Backoff doubles up to 300 seconds; queue, retry,
backoff and controller downtime all consume the **original** elapsed deadline.
A retry is refused unless a whole allocation still fits. No time, GPU, dataset,
step or LLM-call budget is silently increased.

Automatic retries cover terminal `NODE_FAIL`, `BOOT_FAIL` and `PREEMPTED`, and
unobserved submissions fenced before workload start as described above.
For a stalled active allocation, the controller can classify a node as unresponsive
only after matching the owned job to a node with native `DOWN` or
`NOT_RESPONDING` evidence. `DRAIN` alone, an NCCL timeout, OOM, code/configuration
errors and elapsed/workload timeouts are **not** automatic-retry evidence.
Scheduler-query outages still retry only read-only observations. Conflicting
identities always block replacement; an unobserved submission requires the
closed, unclaimed launch fence described above.

The staged worker and batch supervisor publish independent heartbeats. A live
worker heartbeat does not establish training progress: the Framework adapter
declares training and evaluation `status.json` files for their respective command
phases. Phase changes reset the progress clock; worker heartbeats and ordinary
log noise do not. Set the no-progress timeout above legitimate model loading,
checkpoint export, data preparation and sparse progress-report intervals. It is
a watchdog threshold, not a promise that every silent interval indicates failure.
Structured workload failure or a stale configured watchdog triggers cancellation
and native reconciliation, **not immediate resubmission**. If termination cannot
be confirmed within `cancellation_timeout_seconds` (plus in-flight bounded
scheduler calls), the controller returns an explicit recovery-blocked error with
the old job identity retained. It does not keep waiting indefinitely or overlap
a replacement allocation. Worker child cleanup after SIGKILL is bounded too.

A retry preserves the logical candidate, seed, proposal and promotion/PBT source.
It gets a new job-record/native job ID and a sibling directory such as
`trial-00000-attempt-01`; `attempt_id` identifies the current execution and
`attempts` retains prior failed executions. `retry_of` links both attempt and
native record histories. Failed attempts do not become extra search observations,
fake low scores or consumed promotion stages. `stop_on_failure` applies after
retries are exhausted or a failure is non-retriable; pause boundaries count logical
trials. LLM proposals are reused without another API call. Promotions use the
validated **successful attempt** of their completed parent; partial failed
checkpoints are retained for diagnosis but not automatically resumed.

GEPA uses the same policy under `prompts.recovery` (the prompt settings block).
Its total retry cap spans evaluation batches, and attempted examples, including
retries, count conservatively toward `max_evaluated_examples`. Reflection API calls
are never retried by this mechanism. Automatic infrastructure retries currently
require SLURM; local venv/Docker failure classification is not inferred.

These changes require new exact local controller wheels, still core
`1.0.0.dev0`; they do not upgrade already-running controllers or old experiment
identities. Do not edit a live manifest/ledger or replace its installed wheel to
enable recovery. The worker is staged from the controller; the Framework changes
declare existing native status files, not new training-image entrypoints.
CPU fault-injection checks are not live cluster recovery validation.
