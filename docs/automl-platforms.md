# AutoML on Kubernetes, Brev and Lepton

These are execution adapters in the optional `nvidia-automl-core` package, used by
the existing `cosmos-automl` and `cosmos-automl prompts` commands. They do not
call a TAO runner, require a source checkout, or introduce another search loop.
All twelve SFT modes and GEPA use the same workload, scoring and promotion
contracts. The adapters are implemented; **live cloud GPU validation is pending**.

## Controller and storage contract

The CPU controller can run wherever the platform credentials/CLI and the
declared persistent filesystem are accessible. Training runs in a separate
single-node container. The Framework package, model, data, experiment workspace,
metrics and promotion checkpoints must be visible at the **same absolute paths**
to controller and worker. A directory with the same name but different contents
on two machines is not shared storage.

Typical placements are a CPU pod mounting Kubernetes PVCs, a Lepton CPU dev pod
mounting the selected NFS volumes, or the existing Brev instance itself. A local
controller is also possible with an actual mount of the same storage; this port
does not upload/download data, synchronize checkpoints or provision an NFS mount.
Plain SSH access is insufficient. Brev does not supply shared storage between
your laptop and an instance automatically.

Use disjoint volume roots. Declare inputs read-only and a separate output root
writable. Do not put a source checkout or dataset inside the writable output
volume. All `execution.<platform>.volumes` paths are explicit absolute paths;
`sub_path` is relative to the persistent volume root. The output root must support
atomic rename and **cross-client POSIX flock**. Verify these with a two-client
lock probe before a run. The worker reads its staged unique launch intent and
claims it once before executing any command; a missing/wrong volume or duplicate
platform worker fails closed. A successful local `plan` does not prove remote
mount permissions or distributed locking.

Artifacts and worker logs remain in this storage after job termination. The
existing Framework evaluator checks full validation coverage; stage collection
checks native DCP completeness and rank-local restore receipts. Promotion inputs
are mounted read-only. There is no weaker cloud-specific metric/checkpoint path.

## Kubernetes

Requires an authenticated `kubectl`, an explicitly selected context/namespace,
bound PVCs, an existing service account and GPU device plugin. Read-only
preflight checks namespace UID, PVC state, service account and matching-node GPU
capacity. The controller needs permissions to read namespace/PVC/service-account/
node metadata, create/get Jobs, list Pods and patch its Jobs. Worker pods do not
receive the service account token. Current support is one pod, not Indexed Jobs,
gang scheduling or a training operator. Prepare any required scheduling labels
in the cluster; arbitrary pod templates and tolerations are not exposed.

Replace the execution block in [reasoner.yaml](../examples/automl/reasoner.yaml):

```yaml
execution:
  kind: kubernetes
  python: /opt/venv/bin/python
  image: registry.example/framework@sha256:REPLACE_WITH_64_HEX_DIGEST
  environment: []
  kubernetes:
    context: reviewed-cluster
    namespace: ml-jobs
    service_account: automl-worker
    gpus: 8
    cpus: 32
    memory_gib: 256
    shared_memory_gib: 8
    image_pull_secrets: [registry-pull]
    node_selector: {accelerator: h100}
    volumes:
      - {source: framework-inputs, path: /shared/inputs, read_only: true}
      - {source: automl-results, path: /shared/results, read_only: false}
```

Set workload cwd, model/data paths and workspace under these roots, and set
`workload.nproc_per_node: 8` with matching recipe parallelism. For GEPA use one
GPU and the prompt manifest. Images must use immutable digests. Kubernetes can
pull that explicitly approved image with `IfNotPresent`; private image-pull
Secrets must already exist. The executor does not create Secrets or log in.
For workload credentials declare names in `execution.environment` and map each
to `kubernetes.secrets: {HF_TOKEN: {name: model-access, key: token}}`. Values never
appear in the generated manifest. Keep LLM controller credentials separate.

Jobs have one completion, `restartPolicy: Never`, `backoffLimit: 0`, and
`activeDeadlineSeconds` bounded by remaining experiment time. A durable shared
claim also fences rare duplicate/replacement pods. Cancellation uses a UID-tested
suspend patch, not deletion by a possibly reused name; the controller waits for
owned pods to become terminal. Jobs/PVCs are retained for diagnosis. Manual
cleanup after artifact retention is a separate approved operation.

## Brev

Use an **existing, running** instance with working authenticated `brev exec`,
Docker, NVIDIA runtime and a pre-pulled reviewed image. The controller may run
on that instance; it still needs a working Brev CLI connection to the selected
instance. Provisioning, login, starting/stopping/deleting instances and image
pulls are separate approved preparation steps. Do not send login tokens on argv.

Read the stable daemon identity with:

```bash
brev exec reviewed-instance 'docker info --format "{{.ID}}"'
```

```yaml
execution:
  kind: brev
  python: /opt/venv/bin/python
  image: registry.example/framework:reviewed-local-build
  environment: []
  brev:
    instance: reviewed-instance
    daemon_id: REPLACE_WITH_DOCKER_DAEMON_ID
    gpus: 8
    gpu_devices: [0, 1, 2, 3, 4, 5, 6, 7]
    cpus: 32
    memory_gib: 256
    shared_memory_gib: 8
    volumes:
      - {source: /shared/inputs, path: /shared/inputs, read_only: true}
      - {source: /shared/results, path: /shared/results, read_only: false}
```

`brev exec` takes the entire remote command as **one argument**. The adapter
quotes argv accordingly, verifies the Docker daemon ID on operations and tracks
the container's immutable ID plus unique ownership label. It uses
`docker run --pull=never --restart=no`, the specified CPU/memory/GPU limits and
controller UID/GID. Prepare matching filesystem permissions and a compatible
image. Workload environment names refer to variables already available in the
instance's command environment; local secrets are not copied across the wire.

Cancel stops only the owned container. It does not remove the container, delete
artifacts or stop/delete a shared instance. **Instance billing continues after
training or cancellation** until the owner explicitly tears it down. Preserve
instance-local results before deleting an instance.

## Lepton

Uses the workspace's v1 Jobs HTTP API, checked against NVIDIA's published
`leptonai` 0.27.3 client contract. No Lepton Python SDK is required at runtime.
Supply a full HTTPS workspace API URL, token environment **name**, explicit
resource shape and dedicated node-group IDs. There is no mutable current-workspace
selection or fallback to on-demand placement.

```yaml
execution:
  kind: lepton
  python: /opt/venv/bin/python
  image: registry.example/framework@sha256:REPLACE_WITH_64_HEX_DIGEST
  environment: []
  lepton:
    api_url: https://YOUR_WORKSPACE_HOST/api/v1
    token_env: LEPTON_WORKSPACE_TOKEN
    resource_shape: YOUR_REVIEWED_8_GPU_SHAPE
    node_groups: [YOUR_NODE_GROUP_ID]
    gpus: 8
    shared_memory_mib: 8192
    image_pull_secrets: [registry-pull]
    volumes:
      - {source: 'node-nfs:shared-fs', sub_path: inputs, path: /shared/inputs, read_only: true}
      - {source: 'node-nfs:shared-fs', sub_path: results, path: /shared/results, read_only: false}
```

The token remains on the CPU controller. Workload credentials use separate
pre-created Lepton Secrets: `lepton.secrets: {HF_TOKEN: model-access}` paired
with `execution.environment: [HF_TOKEN]`. HTTP redirects are rejected; errors
record status codes, not response bodies or credentials. Preflight verifies
workspace shape GPU count and node-group access. If the workspace does not
advertise the chosen shape, stop and investigate; do not guess a shape/GPU count.

Jobs are private with one worker/completion and both native retry limits zero.
Cancellation sets `spec.stopped` on the owned job ID and waits for native terminal
evidence. Exact-name reconciliation includes archived jobs and refuses conflicts;
an archived/deleted state alone is not successful-training evidence. Native job
retention is seven days after finishing; durable artifacts remain on NFS. Image
pulls use the explicitly reviewed digest and existing registry Secret.

## Launch, limits and verification

Use the normal plan/run/status/logs/resume/cancel commands. `plan` checks volume
coverage of cwd/workspace and displays GPU, reconciliation and deadline limits;
it makes no platform requests. `run --approve` performs read-only preflight
before submission. Resource/image/storage preparation is not included implicitly.
Do not launch a training command that itself submits an untracked remote job.

Launch intent, submitted spec and platform scope are persisted before submission.
After a lost reply, the controller performs bounded read-only adoption of the
same owned job, never a second create. Healthy absence beyond
`submission_reconcile_seconds` closes the shared launch gate and reports
`UNKNOWN`; a late allocation cannot begin training. Conflicting identity or an
unavailable platform requires reconciliation, not a replacement experiment.
Transient observation outages back off without consuming trials or LLM calls.

Worker runtime begins when its Python supervisor starts; queue time still counts
toward the original absolute experiment deadline. Kubernetes additionally has a
native Job deadline. **Lepton and Brev have no native allocation wall-time cap in
this adapter**: keep the controller alive to cancel queued/startup work. The
worker exits when it eventually starts after the absolute deadline, but this
does not bound a stuck image pull or a host that cannot start Python. Do not
claim an unattended cloud-spend ceiling from worker timeouts. The existing
SLURM CPU requeue supervisor is not a generic cloud controller service.

Automatic infrastructure *training* retries remain SLURM-only; nonzero
`search.recovery` retry caps are rejected on these platforms. Watchdogs and user
cancellation can stop an owned job but cannot manufacture native terminal proof.
Budgets, algorithms, completed trials, paid API receipts and promotions keep the
same core semantics. Cross-platform migration of an existing experiment is not
supported.

Validation uses existing core/Framework checks and CPU operational simulations,
not live cloud jobs. Before claiming GPU readiness, separately approve a tiny
real-platform smoke covering credentials, registry access, shared-lock proof,
one training/evaluation trial, cancellation and (for multi-fidelity) checkpoint
promotion. Do not infer live scheduling/storage compatibility from fixtures.

Primary platform references: [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/),
[Brev connectivity](https://docs.nvidia.com/brev/cli/connectivity),
[Lepton API and storage](https://docs.nvidia.com/dgx-cloud/lepton/reference/api/),
and [Lepton job lifecycle](https://docs.nvidia.com/dgx-cloud/lepton/reference/cli/lep_job/).
