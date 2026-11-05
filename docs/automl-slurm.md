# AutoML on SLURM

The core package provides a single-node SLURM executor. The controller
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
and removed after the experiment terminates. With CPU handoff enabled, keep it
available to the bounded successor allocations; a single allocation ending is
not experiment completion. Never place it in the image or PR.
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

The CPU controller needs the Framework and core packages plus their planning
dependencies, not GPU training packages. Verify its Python version/imports,
native scheduler CLIs, and shared interpreter path on compute nodes. Do not
assume the system Python or the interpreter inside the training image is a
suitable host-side controller environment.

For an isolated planning-only controller, install the core from the selected
source revision with its dependencies, then the Framework wheel with `--no-deps`.
Use `main` by default or the user's explicit development branch, as described in
[source installation](automl.md#installation-from-source). Resolve and record the
commit during preparation, never reinstall from a moving branch at job startup:

```bash
AUTOML_CORE_REF=main  # Or the selected development branch / recorded commit SHA.
python -m pip install \
  "nvidia-automl-core @ git+https://github.com/NVIDIA-TAO/tao-automl.git@${AUTOML_CORE_REF}#subdirectory=packages/automl-core" \
  'nvidia-automl-core==0.1.0'
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
For GEPA controllers, also install `gepa==0.1.2` before verifying imports; the
planning-only `--no-deps` Framework install does not install its extras. A wheel
built from the recorded core revision remains an alternative for offline nodes.

## Private node history and exclusions

Use the user's existing private cluster records during launch review. SLURM's
node-event history is stored in its accounting database, queried with
`sacctmgr show event`; it is not a skill-owned faulty-node file. A site's
private incident register can supplement it. Keep both outside the public
repository, along with site identities and the resolved launch manifest.

On the selected cluster, these are read-only queries (replace the placeholders
with reviewed nodes and the historical time window; the end precedes launch):

```bash
TZ=UTC sacctmgr -r -n -P show event where Nodes=<node-list> \
  Start=<history-start-UTC> End=<launch-review-UTC> \
  Format=NodeName,Start,End,State,Reason
scontrol show nodes -o
```

Distinguish fault events from normal provisioning and use event end times and
repair evidence: a closed incident does not prove a node is currently faulty.
For a failed multi-node allocation, check `sacct`'s `FailedNode`, not just its
`NodeList`. Do not infer a block-wide exclusion from a few affected nodes.
Review recent/repeated failures and current diagnostic comments with the user's
exclusion policy. A conservative exclusion can remain after a closed incident
when explicitly selected in that review; no universal expiry is imposed.

Put the reviewed set in the private manifest's
`execution.slurm.exclude_nodes` as a list of explicit node names. Expand any
approved hostlist against current inventory first. The default is an empty
list; shell expressions, wildcards, ranges and duplicate names are rejected.
Record the history source, review time and selection rationale in the private
launch review. The list is visible in `plan.identity.execution.slurm` and bound
to the experiment identity. Run preflight checks all names against live
inventory and fails on retired/missing names; reconcile them in a new reviewed
configuration rather than silently dropping exclusions. Scheduler-query errors
do not authorize running without the list.

The executor emits `#SBATCH --exclude` on the actual outer batch for every SFT
trial, promotion, retry and GEPA evaluation allocation, and records the list in
`_slurm.json`. Its existing per-submission `sbatch --test-only` checks still
apply. It does not automatically grow the list during an experiment or bypass
native termination/retry-budget safeguards. GPU exclusions do not change the
separately submitted CPU controller's placement. `SBATCH_EXCLUDE` is not a
substitute: the executor filters scheduler environment overrides.

This requires a core controller build with node-exclusion support; the training
image need not change. Do not upgrade a live controller or mutate its sealed
manifest. Existing searches keep their original placement policy; select the
new build and exclusions when preparing a fresh experiment.

## Manifest settings

Keep the native `workload.recipe`, `workload.evaluation`, `nproc_per_node`,
seed, domains, data revisions and objective from the prepared search. Add
`workload.expected_validation_records` when the whole validation set must be
scored. Use [reasoner-slurm.yaml](../examples/automl/reasoner-slurm.yaml) as a
planning template; replace every site/asset placeholder before approval.

`execution.slurm` describes one node:

- Explicit partition(s), account, optional site QoS, CPU/GPU counts, optional
  memory, exclusivity, allocation wall time, startup cap, cleanup allowance,
  and scheduler polling interval. GPU allocation requeue remains off; bounded
  trial retries require the explicit recovery policy below. CPU-controller
  continuation is a separate opt-in policy, described next.
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

### Before the search controller starts

Cross-node startup probing is **opt-in** in
[slurm_controller.sbatch](../examples/automl/slurm_controller.sbatch): leave
`AUTOML_STARTUP_HELPER` unset for the original direct-CLI launch path. It needs
only `AUTOML_CONTROLLER_PYTHON`, `AUTOML_MANIFEST`, `AUTOML_WORKSPACE`, and
`AUTOML_CONTROLLER_MAX_ALLOCATIONS`; no startup observer, base directory, or
separate deadline variable is required. The search/controller limits still apply,
but this mode does not provide cross-node bootstrap validation or automatic
recovery for failures before the controller starts. Monitor native CPU state
through the site's existing submission service.

The allocation cap **includes the initial allocation**. Native restart generation
is zero-based: cap `3` allows generations `0`, `1`, and `2`; generation `3` or
higher exits `2` before a probe or controller runs. Both values must be decimal
integers (cap positive, generation nonnegative); zero padding such as `003` and
`0002` is accepted without octal interpretation. Missing `SLURM_RESTART_COUNT`
means generation `0`. Malformed values now fail early rather than reaching the
CLI. The shell compares normalized copies but passes the original cap argument
unchanged. With the helper unset and `AUTOML_ACTION` unset, valid launches keep
the original CLI argv and direct `exec` signal delivery.

This is the same count/generation boundary as the core's `controller.Session.run`
check, not a replacement for it: the shell reads `SLURM_RESTART_COUNT`, while the
core independently reads native `scontrol` `Restarts` and checks ownership,
resource shape, ledger continuity, and remaining time. Passing the shell check
does not guarantee core acceptance. A stale or inconsistent scheduler environment
must be reconciled, not used to bypass either guard.

To enable the supervised path below, additionally set `AUTOML_STARTUP_HELPER`
to the sealed shared helper path, `AUTOML_STARTUP_BASE` to the registered shared
attempt base, and `AUTOML_DEADLINE` to the original absolute Unix deadline.
Once opted in, missing settings or an absent observer fail closed; there is no
fallback to an unprobed launch. Use this path when startup recovery is part of
the reviewed launch policy.

Trial recovery and CPU handoff run inside Python. They cannot recover a failed
batch prolog, missing interpreter, failed environment import or bootstrap that
never reaches the search controller. The submitting process must therefore
supervise the **native CPU job**, starting from its durable submission intent.
Observe `squeue`/`sacct` even when no experiment ledger exists; a missing ledger
is not proof of success or permission to submit again.

The stdlib-only [startup helper](../tools/cosmos_workflows/scripts/slurm_controller_startup.py)
wraps the CPU command and records bootstrap and child exit receipts without
importing Framework or the engine. Give every CPU attempt a fresh shared
directory and the same absolute experiment deadline:

```bash
python3 slurm_controller_startup.py run --attempt-dir /shared/run/startup-00 \
  --deadline <approved-unix-deadline> -- bash /shared/run/controller-command.sh
```

An already-running external supervisor must automatically invoke `observe
--attempt-dir /shared/run/startup-00 --job-id <native-id>` from another node.
It acknowledges lock contention and then claim visibility in two separate,
idempotent calls. Arm that supervisor before submission and keep it alive while
the CPU job is queued. This is not a manual preflight to leave for the next chat
turn. The helper times out without starting the child if the observer is absent.
It verifies filesystem semantics, not model/GPU readiness.

The child calls `fanout --attempt-dir "$AUTOML_STARTUP_ATTEMPT"` **before** any
algorithm process, submission or paid request. That durable boundary prevents
startup recovery from replaying a partially launched suite. The external
supervisor may use `retry_allowed` only after matching native terminal identity,
checking the entire workspace for existing work, and enforcing separately
approved CPU attempt/time caps. Call `close_before_retry` under its shared
fan-out lock before submitting the replacement; a failed/conflicting fence
blocks recovery. Only infrastructure failures or the explicit
observer timeout qualify; configuration/import errors, cancellation and unknown
submission outcomes do not. Each replacement has fresh probe receipts, while
the absolute deadline, GPU budgets and API budgets remain unchanged.

The helper does not submit jobs, transport SSH, or replace the search CLI.
The site's submitting service owns durable native-ID reconciliation, observer
scheduling, native terminal verification, retry caps and final job-record
updates. Keep credentials across permitted startup retries and remove them only
after the final owned CPU allocation is confirmed terminal. Report every mode
that never launched as `NOT_STARTED`, even if Python failed before creating its
ledger. Never label CPU-check simulations as live cross-node validation.

Use `run` with a **non-requeue CPU batch**. For bounded handoff, use `probe`
instead: it performs the same check but returns without starting a child.
Then mark `fanout` and **exec** the handoff CLI from the batch shell, so the
controller directly receives SLURM's `SIGUSR1`. In that shell, trap termination
and `USR1` to exit while bootstrap is running; never ignore an early signal.
Use a fresh attempt directory for each native job ID and `SLURM_RESTART_COUNT`
generation. Validate the generation against the approved allocation cap before
probing. The external observer must watch each generation automatically.
The batch template diagnoses invalid restart/allocation values and exhausted
caps before installing signal traps or running helpers. Bootstrap signals and
helper failures exit with a diagnostic and status 2, not `EX_TEMPFAIL` (75).
They do not request native requeue; the external supervisor still owns terminal
reconciliation and approved startup retries. Site-enforced restarts can still
occur, but an out-of-budget generation cannot probe or start the controller.

#### Persistent observer service

Use [slurm_controller_watch.py](../tools/cosmos_workflows/scripts/slurm_controller_watch.py)
on an approved always-on host with the same shared storage and native SLURM
commands, **outside the CPU allocations it observes**. It automatically selects
`<attempt-base>/<job-id>-g<Restarts>` for each owned running job. Scheduler outages
retry read-only; the registry, deadline and observations survive process restarts.
The watcher never submits, requeues, cancels, or authorizes replacement jobs.

Before submitting CPU controllers:

```bash
python3 /shared/tools/slurm_controller_watch.py init \
  --state-dir /shared/run/observer --deadline <original-unix-deadline>
# Review/install slurm_observer.service on the approved service host first.
systemctl --user start slurm_observer.service
python3 /shared/tools/slurm_controller_watch.py ready --state-dir /shared/run/observer
```

Replace paths in the [user-service template](../examples/automl/slurm_observer.service).
It restarts the watcher after unexpected exits; the watcher exits normally at the
original deadline. The user manager must survive logout and the whole run. User
lingering, login-node residency and service-host availability are site prerequisites,
not settings to enable implicitly. Occasional `tick` calls or `nohup` alone are not
a supervised service. `ready` requires a fresh heartbeat **and** a held observer
lock. Stage the watcher and its sibling startup helper together; both are stdlib-only.

After an acknowledged CPU submission, register its exact ID:

```bash
python3 /shared/tools/slurm_controller_watch.py register \
  --state-dir /shared/run/observer --job-id <native-id> \
  --attempt-base /shared/run/startup --max-allocations <remaining-approved-cap>
```

Use that attempt base and original deadline in the reviewed
[CPU batch template](../examples/automl/slurm_controller.sbatch). Register every
acknowledged controller, including cold continuations; native requeues need no
manual registration. A lost submission reply still needs the submitter's native-ID
reconciliation, not a guessed ID or replacement. Registered entries and the deadline
are immutable. The submitting service still owns terminal monitoring and startup
retry policy; this watcher supplies the cross-node handshake, not another search
controller. Restart it on the same host; moving hosts mid-handshake fails closed.

Probe success only means the storage check passed; it does not prove that the
brain started. Preserve `fanout` before exec and use the native CLI's
`controller.json` and experiment ledger for controller status. Never interpret
a failed exec or missing ledger after fan-out as permission for a fresh job.
Native handoff owns same-job restart validation; startup retry still requires
no fan-out or work in **any** prior generation. Do not wrap the handoff CLI
inside `run`, reuse probe receipts, or reset the absolute deadline on requeue.

### Experiment and allocation limits

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
See [source revision, package version and build identity](automl.md#source-revision-package-version-and-build-identity).

### Explicit cold recovery migration

Ordinary resume requires the original build. To apply a reviewed recovery fix
to a **stopped SLURM SFT search**, install the new core wheel into a separate
controller environment. Do not modify environments used by active controllers.
Preserve the exact original wheel and review the changed recovery modules; the
migration's file allowlist cannot prove semantic compatibility by itself.

```bash
cosmos-automl migrate --manifest continuation.yaml --workspace /shared/runs \
  --from-wheel /shared/original/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl
# Only after reviewing the migration output:
cosmos-automl migrate --manifest continuation.yaml --workspace /shared/runs \
  --from-wheel /shared/original/nvidia_automl_core-1.0.0.dev0-py3-none-any.whl --approve
```

This does not launch training. It verifies the original wheel's engine hash,
old CPU/trial ownership and terminal evidence, resolved LLM receipts, unchanged
workload/environment and remaining original time. Only the engine's permitted
recovery files and additive explicit node exclusions may change. Search/proposal
code, model/data, resources and all search/retry/LLM budgets remain unchanged.
An already-failed uncertain attempt needs a closed launch fence and no observed
active allocation; its fence and consumed retry remain intact. Unresolved
ownership, paid requests or active jobs block migration.

Completed trials/results, configurations, failed reserved slots and attempt/API
receipts stay in place. Partial checkpoints are not replayed. The old ledger and
CPU history are archived in `migrations/recovery-v1.json`; `migration.json` records
the transaction. A root stop marker requires `--acknowledge-cancel` and is archived,
not silently ignored. Never remove attempt cancellation markers or fences. An
interrupted transaction blocks resume; rerun the same approved migration.

The resulting ledger is `SUSPENDED`. Review `plan` with the new manifest/build,
then submit a fresh CPU batch using `resume --approve`. Preserve its previous CPU
resource shape, wall time and grace, but set `--controller-max-allocations` to no
more than the migration's **remaining** allocation count. Prior allocations are
deducted, and bypassing the CPU session is rejected. The original absolute deadline
still applies, including controller downtime. Ordinary identity checks are not
disabled. A previously migrated, stopped SFT search requires explicit
`--continue-migration` with its **current exact wheel**. Each transaction retains
its predecessor in `migrations/recovery-v<N>.json`, validates CPU history against
the preceding remaining envelope, and deducts allocations again; it never restores
the initial cap. Unacknowledged native handoffs still require reconciliation,
including CPU generations that failed before entering Python. This narrow operation
does not support GEPA, other executors or arbitrary engine upgrades.

### Bounded CPU-controller handoff

There are three independent limits: each CPU controller allocation, each GPU
trial allocation, and the overall experiment deadline. Reaching the last one
ends the search; obtaining another CPU allocation must not restart that clock.
Without the opt-in below, controller termination requests trial cancellation
and records the observation. An unconfirmed native cancellation is `ERROR`,
not proof that the GPU allocation ended. Independent worker deadlines still
apply after controller loss.

For a CPU wall limit shorter than the approved experiment, use the new core
build's shared CLI handoff flags for **both plan and run**:

```bash
cosmos-automl plan --manifest search.yaml --workspace /shared/automl \
  --controller-max-allocations 4 --controller-allocation-seconds 3600 \
  --controller-grace-seconds 300
```

These example settings permit at most four controller-requested one-hour CPU
allocations, including the initial one. They add no GPU trials, retries or LLM
calls. Review CPU/memory resources and the original experiment deadline too;
grace and time spent queued reduce useful controller time. For GEPA use
`cosmos-automl prompts plan/run` with the same flags. Grace must cover an
in-flight LLM request plus at least 60 seconds.

Prepare [slurm_controller.sbatch](../examples/automl/slurm_controller.sbatch),
replace its site placeholders, and set the documented environment variables.
`AUTOML_WORKFLOW` is `sft` (default) or `prompts`; `AUTOML_ACTION` is `run`
(default) or `resume` for a reviewed continuation. Other actions are rejected
before any probe or approved CLI invocation. Startup probing remains optional
as described [above](#before-the-search-controller-starts).
The example starts one experiment per CPU job, using `exec` so the Python
process receives `--signal=B:USR1@300`. Review before `sbatch`; the example is
not a ready-to-launch site configuration. It is not a supervisor for an
existing multi-experiment matrix wrapper.

Before CPU expiry, a cooperative safe point saves `SUSPENDED` and requests
`scontrol requeue` for the **same CPU job ID**. The successor validates native
ownership, resources, restart count, build and policy, then resumes the same
experiment. It adopts recorded GPU jobs and reuses completed results and LLM
receipts. `SUSPENDED` here is a brain state that may have an active GPU trial;
it is distinct from `PAUSED`, the manual-review boundary with no active trial.
`controller.json` records allocation generations, handoff intent and the
original absolute deadline. No in-flight API request or submission is
interrupted just to checkpoint midway through its durable receipt.

When the CPU allocation cap is exhausted, the search stops and requests native
cancellation. Requeue is disabled on the final permitted CPU allocation;
scheduler-initiated restarts also consume the cap. A site administrator or
forced prolog requeue can override user settings, so such an unexpected extra
generation is refused rather than starting more work. SLURM's requeue rules
are documented in [sbatch](https://slurm.schedmd.com/sbatch.html#OPT_requeue)
and [scontrol](https://slurm.schedmd.com/scontrol.html#OPT_requeue).

A lost requeue acknowledgement is recorded as `REQUEUE_UNCONFIRMED`. An actual
native successor may resume, but the old process never repeats the requeue RPC
or submits a replacement CPU job blindly. Inspect native state if no successor
appears. Hard node/process loss can bypass a cooperative checkpoint; durable
submission intents and independent GPU deadlines remain safeguards, not a
guarantee that arbitrary paid API calls can be replayed.

Use an unchanged shared interpreter, manifest, assets, credentials and workspace
across allocations. Do not put trial cancellation or secret deletion in an
unconditional batch `EXIT` trap. Review the underlying filesystem's cross-node
POSIX locking before GPU fan-out. This feature does not change installed wheels
or retrofit existing queued/running controllers. CPU operational checks cover
handoff and adoption; live site requeue/locking and GPU continuation still need
a separately reviewed smoke test.

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

Timed-out read-only preparation checks are separate from uncertain submissions.
The controller retries cluster/node/partition checks, batch syntax checks and
`sbatch --test-only`, backing off up to 300 seconds within the original deadline
and CPU handoff envelope. `_scheduler_check.json` identifies the operation,
timeout and `WAITING`/`COMPLETE` state without saving argv, stderr or credentials.
Nonzero exits, invalid configuration, ownership conflicts and record-hook errors
are not silently classified as transient scheduler timeouts.

Before submission, the core seals the complete staged attempt's files. A handoff
during these checks resumes **that reserved candidate** after checking the seal,
open launch fence, owner, absence of worker/stop receipts and unchanged deadline.
It does not repeat a proposal or consume a training retry. Older or incomplete
staging cannot be replayed automatically. The preflight clock is durable even
before the first ledger is created. The CPU session also retries timed-out native
identity reads before search startup; unverified ownership still stops safely
when its bounded preparation time expires.

Once `submission_attempted` is recorded, only the uncertain-submission protocol
below applies. Failures after the ledger opens now persist an explicit terminal
error and cleanup observation instead of leaving a stale `RUNNING` ledger.

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
not require rebuilding the training image. Historical runs are not automatically
migrated; use the explicit cold migration above only for its supported case.

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
scheduler calls), the brain stays alive and records the trial's
`cancellation_reconciliation.state: WAITING`. It switches from active cancellation
to read-only observation of that same allocation. SLURM `COMPLETING` still owns
resources even when accounting already says `CANCELLED`; neither that accounting
row nor a worker stop receipt permits an overlapping replacement.

This wait uses the original experiment deadline and the approved CPU-controller
handoff/allocation caps, not another timeout that resets on resume. Scheduler
outages back off without ending the brain early; cooperative handoff preserves
the pending cancellation. No trial, retry or LLM proposal is consumed while
waiting. Confirmed native termination records `CONFIRMED` and lets the existing
retry policy and `stop_on_failure` determine the next step. Node health is
rechecked while the owned allocation is active so later `DOWN`/`NOT_RESPONDING`
evidence is not lost. An already-released node's later fault is not evidence
against the old job. Missing/conflicting native identity still blocks the search.
User cancellation, elapsed deadline or CPU cap exhaustion still ends bounded
monitoring; if release is unconfirmed, report `ERROR` and reconcile before resume.
Worker child cleanup after SIGKILL is bounded too.

Both heartbeats depend on shared-storage reads/writes. A hung node, driver or
filesystem can stop publication even when training has no Python traceback.
`heartbeat_stale` records which publisher was stale and its timestamp/age; worker
and batch exceptions preserve numeric `error_errno` without logging secret-bearing
exception text. Correlate these with native node events and bounded job log tails.
A fresh heartbeat with an unchanged declared progress file is `progress_stalled`,
not a heartbeat failure. Node exclusions reduce placement risk but do not repair
hardware/storage or guarantee future health; do not disable watchdogs to hide it.

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

These changes require a controller build with bounded recovery support; selecting
a new source revision does not upgrade already-running controllers or old experiment
identities automatically. Do not edit a live manifest/ledger or replace its installed package to
enable recovery. The worker is staged from the controller; the Framework changes
declare existing native status files, not new training-image entrypoints.
CPU fault-injection checks are not live cluster recovery validation.
