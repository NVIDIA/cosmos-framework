<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos Workflow Launch Intake

> **Execution setup:** Use `cosmos3-setup` to resolve this checkout's `tools/cosmos_workflows` root and select the execution platform. The existing framework skills own this workflow; no extra plugin is required.

This reference extends `cosmos3-setup` for managed model/data actions. Native
framework development, ordinary native recipes, and read-only planning do not
become managed launches merely because the repository contains these helpers.

## Quick Start

Use the requested supported platform, or ask once among compatible installed
options. Read its platform reference and check only required credential presence.

## Non-Negotiable Launch Gate

This gate is model-agnostic within the bundle. Apply it to the selected model,
data action, or application before launching side-effecting work.

Do **not** create runner scripts, launch scripts, compatibility shims,
workspace folders, state files, logs, or dependency-install side effects until
the launch preflight passes.

Preflight passes only after all of these are true:

1. The execution platform is selected from the supported installed options.
2. Platform credentials and required credential groups are satisfied.
3. Model-specific credentials are satisfied.
4. The default container image is resolved from packaged model/action metadata,
   shown to the user, and either confirmed or replaced by an explicit
   `image=<override>`.
5. The platform access check succeeds from the launch host.
6. Dataset inputs are mapped to concrete spec keys and verified from the
   selected platform's point of view.
7. Required compute shape fields from the model/workflow skill are known.
8. Required local tools for the selected data/platform path are present, or the
   user approved installing the smallest missing dependency and preflight was
   rerun.
9. A launch review with image, platform, datasets, compute shape, expected
   runtime, and any generated/default configuration changes has been shown and
   confirmed by the user.

If any item is missing, ask for the missing input and stop before generating
artifacts. This applies to the managed train/evaluate/inference/export and data
actions.

When preflight work clears a blocker, keep track of the original user request.
After the fix, rerun the relevant preflight and continue toward that request;
do not stop at "blocker fixed" unless the user explicitly asked only for the
repair.

## The Four-Verb Execution Contract

Once the launch gate passes and the producing model/data skill has authored the
spec-bundle (schema: `execution/artifacts/guide.md`), execution is exactly four verbs. Every
platform reference implements them over its native CLI — Docker, SLURM,
Kubernetes, Brev, and virtualenv under `execution/platforms/`. Any
externally installed platform skill joins the same contract (§ External
platform skills); nothing else is platform-specific.
Source `tools/cosmos_workflows/env.sh` from the repository root to set
`COSMOS_WORKFLOWS_ROOT` to this checkout’s helper directory.

- **submit(spec-bundle)** — resolve the data question first: if the inputs are
  **already readable from the compute frame** (a local path, an existing mount —
  tier A in place, the common local and the only air-gapped case), there is
  nothing to stage — record tier A and move on. Invoke `execution/data-io/guide.md` only on a
  **frame mismatch** (remote URIs, cross-host paths, PTM fetches, tier-C result
  uploads). Then lint the assembled command with `redact_secrets.py lint` and
  **open the record and launch, in that order**:

  ```bash
  JOB_ID=$("$COSMOS_WORKFLOWS_ROOT/scripts/cosmos_job_record.py" open --platform <p> --image <img> \
    --network-arch <arch> --action <action> --storage-tier <A|B|C> --results-root <root>)
  # <native launch, naming the backend object after $JOB_ID>
  "$COSMOS_WORKFLOWS_ROOT/scripts/cosmos_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref <ref>
  ```

- **status(id)** — poll the native backend, map to the fixed vocabulary
  `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`; the native sub-state
  (`ImagePullBackOff`, `PENDING`-resources, slurm `COMPLETING`) rides in the
  transition `message`. Never read "what's running" from records — poll the backend.
- **logs(id, tail)** — native log fetch.
- **cancel(id)** — native cancel + orphan teardown, then `mark <id> --state CANCELED`.

**Record-then-launch is the ordering invariant.** `open` mints the id and binds
`results_dir` *before* any launch, and the id it returns is the only handle the
launch can use — a submit that skipped the gate or the open has no id, so it
cannot launch. This is what keeps a run recoverable across a context break:
`results_dir` is recorded before the backend object (which K8s TTL or docker
`--rm` may later delete) ever exists.

When the producing spec-bundle declares `execution`, preserve it as model-owned
action semantics across every application that reuses that model skill. The
selected platform consumes the lifecycle; an application must not copy its
commands into a private launcher. Platform-independent pre/post commands,
runtime attestations, helper dependencies, distributed intent, and completion
evidence belong in the producer's spec-bundle. Scheduler syntax, mounts,
secrets, timeouts, ranks, and child-exit preservation remain platform-owned.

### External platform skills

No registry, no interface file: a platform skill **declares the contract by
documenting the four verbs, and you verify by reading** before first use. A
skill with only native primitives may be used by **inferring** the mapping
(workflow invariants still bind; the mapping goes in the launch review; persist
what worked). Rules and the no-equivalent hard floor:
`references/external-platforms.md`.

### Failure analysis & retry

When `status` reaches `ERROR`, read the log tail and **classify before any
retry** — infrastructure faults are retriable (new record, `--retry-of`, up
to 10), program faults never are. Full criteria, the two judgment calls
(device-side asserts, downstream tracebacks), and the post-turn poller
rules: `references/failure-analysis-retry.md`.

## Initial Questions

Use the user's requested supported platform. Otherwise, identify the compatible
installed choices from `execution/platforms/*/references/skill_info.yaml` and
their `guide.md` procedures. These are references owned by `cosmos3-setup`, not
additional skill entrypoints. Ask once when several platforms fit; a workflow
with only one supported platform has already selected it. External platform
integrations must satisfy `references/external-platforms.md` before use.

Then ask:

- Which supported platform should run this workflow?
- Should I monitor the run in this chat? Monitoring means I keep polling the
  backend/job logs after launch and report progress until the job finishes,
  fails, or you ask me to stop, even if the job stays queued for hours or days.
  If disabled, I launch the job, give you the job id/log path, and stop
  polling. Default: monitor in chat.
- How often should I post status? Default: every 5 minutes. Use 1-2 minutes for
  smoke tests, 5 minutes for normal training, or 10-15 minutes for long runs.

Use `long_running_enabled=true` and `status_interval_minutes=5` when the user
accepts the defaults.

When monitoring is enabled, do not send a final summary just because several
polls have elapsed or the job is still `PENDING`. Keep the turn attached and
emit status every `status_interval_minutes` until a terminal state or explicit
user stop/detach request. If the runtime environment cannot keep the chat turn
open, say that clearly and leave a durable watcher/log path; do not imply that
chat updates will continue after the turn ends.

Final-answer rule: a `final` response ends chat-side monitoring. While
`long_running_enabled=true` and any launched job is non-terminal, status
messages must be sent as in-progress updates and the agent must continue
polling. Only send a final response when the workflow reaches terminal state,
the user explicitly asks to detach/stop monitoring, or the runtime genuinely
cannot keep the turn open; in that last case, say it is a runtime limitation
and provide the exact durable status command/log path.

## Missing-Input Prompt Shape

When intake inputs are missing, ask with the exact prompt shape in
`references/intake-prompts.md` (one consolidated ask, concrete examples,
no invented defaults).

## Model action contract

Read the resolved model's `references/skill_info.yaml` for the action command,
configuration format, supported platforms, and declared inputs and outputs.
Use its planner when provided. The reasoner planner at
`models/cosmos3-reasoner/scripts/cosmos_workflow.py` generates training TOML,
environment bindings, image and dataset fingerprints, and preflight commands.
Checkpoint and evaluation planners consume that sealed training contract.

## Container Image Confirmation

Before creating specs, runner scripts, workspaces, logs, state files, or
submitting a job, resolve the image for the selected model/action:

```bash
${COSMOS_WORKFLOWS_ROOT:?}/scripts/resolve_cosmos_image.py \
  --workflow-root ${COSMOS_WORKFLOWS_ROOT:?} \
  --model <network> --action <action> --format text
```

If the helper is unavailable, read `models/<network>/references/skill_info.yaml`
directly. Resolve image fields in this order:

1. `actions.<action>.container_image`
2. top-level `container_image`

Check that the action is supported before resolving its image.

Show the exact image and ask:

```text
Container image for <network>/<action>:
default=<resolved image>

Use this image, or provide image=<override>?
```

If the user accepts, pass the resolved image as the job `image`. If the user
overrides, require a non-empty image reference and pass that value instead.
Do not silently launch on the default image. This confirmation applies to
training, evaluation, inference, export, and data workflows that submit
containers.

## Credential Filtering

After the user chooses a platform, get the credential list for **only that
platform** from the chosen skill itself — its `## Credentials` section and, if
present, `references/skill_info.yaml` (`required_credentials`, `credential_groups`,
`optional_credentials`). The launch preflight (`check_cosmos_launch_preflight.py`)
reads that same per-skill `skill_info.yaml` to enforce the credential gate; a
credential-free platform (e.g. Docker) may ship only prose, in which case rely on
its Preflight section.

Ask only for credentials that platform actually needs, plus model-specific
credentials from the selected model skill. Do not ask for Brev credentials on
SLURM, Kubernetes, or Docker. Do not ask for SLURM credentials on Brev,
Kubernetes, or Docker. Ask S3 credentials only when the selected
platform and the dataset/result URIs require `s3://` access.
Credentials may already be present in the process environment or in a
user-approved secret env file at the path provided by the user;
source such files only when needed and never print,
grep, cat, paste, or log their contents. Verify only variable presence.

For initial launch intake, ask for required credentials and required credential
groups only. Treat the helper's optional credentials/settings section as
reference material; do not request those values unless their `only_when`
condition applies, the selected workflow cannot proceed without them, or the
user asks to customize that setting.

When the helper output includes a "Required credential groups" section, satisfy
one credential from each group before proceeding. Explain each requested value
using the helper's description and "How to get it" text.

For SLURM, user-facing prompts should ask for `SSH_KEY_PATH` first. Mention
`SSH_AUTH_SOCK` only if the user says they already use an SSH agent.

## Dependency Remediation

If a required CLI/library is missing, say exactly what is missing and why it is
needed, then ask before installing. Examples:

- S3 dataset or results path -> require an S3-capable client such as `aws`.
- Local Docker path -> require the Docker CLI and the configured Docker
  network.

After user approval and installation, rerun the same preflight. Do not create
runner files or launch jobs between the failed check and the rerun.

## Dataset Intake

Accept dataset inputs in either mode:

- **Dataset root mode:** the user gives train/eval/calibration roots, and the
  model skill maps required files by convention. For Framework training, pass
  `--train-annotation <root>/annotations.json --train-media-root <root>` to the
  reasoner planner.
- **Direct spec mode:** the user gives exact spec-key paths when annotations,
  media archives, videos, or image folders live in different places. Preserve
  those keys directly, for example
  `dataset.annotation_path=<EVAL_ANNOTATION_PATH>`
  and `dataset.media_dir=<EVAL_MEDIA_PATH>` for the evaluator.

Ask for dataset examples that match the selected platform:

- SLURM: explicit shared cluster paths supplied by the user and verified from
  the allocated compute node; the skill has no site-specific storage default.
- Brev, Kubernetes: usually `s3://bucket/path/train` and
  `s3://bucket/path/eval` unless the platform profile mounts shared storage.
- Local Docker: local paths visible to the Docker host, such as
  `/data/cosmos/<model>/train`, or direct spec paths visible inside the planned
  container mount.
- Remote Docker: absolute paths visible on the remote Docker host named by
  `DOCKER_HOST`, not paths on the local agent machine.

Do not assume "dataset root" is the only acceptable input. When direct spec
paths are supplied, validate the exact spec paths rather than appending default
filenames.

## Platform Preflight

Run the selected platform's preflight checks before any launch artifact is
created — prefer the packaged helper `scripts/check_cosmos_launch_preflight.py`
(`--platform <p> --container-image <img> --path <label>=<path> ...`). It verifies
credentials, client tools, platform/cluster/object-store access, dataset paths
from the compute frame, GPU/runtime health, and image-architecture fit; treat any
failure as blocking. Never use `--skip-platform-access` for a real launch.

See `references/platform-preflight.md` for the full per-platform detail (SLURM
SSH/key setup + resource defaults, docker/remote-docker GPU + bind-mount checks,
Brev/Kubernetes API + object-store checks, annotation content-field checks, and
data staging).

## Runtime And Configuration Review

Before any side-effecting launch, show a concise review:

- selected platform and exact container image
- GPU ids/count and nodes, including any GPUs avoided because they are already
  occupied
- dataset roots or direct spec paths, with sample counts when available
- important model/workflow overrides that differ from template defaults
- estimated runtime and the assumptions behind it
- monitoring interval and whether chat-side monitoring will stay attached

Ask for confirmation after this review. If the user supplied a time limit, flag any plan that exceeds it
and offer concrete reductions before launch.

Never end a successful launch review with only “nothing was launched.” End
with one direct action prompt, for example: `Ready to materialize the sealed
plan and submit the job. Reply "launch", "go ahead", or "yes" to proceed.`
The next unambiguous affirmative chat message authorizes materialization,
job-record creation, submission, and the previously reviewed monitoring mode;
execute immediately without another intake or confirmation round.

## Structured Training Metrics

When the model contract declares a structured status path or metric extractor,
poll it alongside the native backend. Scheduler/container completion is not a
successful training result by itself: require the model's terminal structured
success record, collect concrete checkpoint events, and return final train
loss plus every epoch validation-complete loss. Do not promote validation
heartbeat/batch metrics or a train-loss line to epoch validation loss. If the
process fails before its native logger exists, invoke the packaged status
finalizer or report the real process exit failure; use raw log parsing only as
a fallback.
