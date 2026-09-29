# Cosmos workflow support

These are supporting references and CPU-only orchestration helpers for the
framework's existing skills. Native Python
training/inference implementations remain in `cosmos_framework/`.

## Ownership and routing

| User task                                                            | Skill entrypoint           | Supporting resources here                           |
| -------------------------------------------------------------------- | -------------------------- | --------------------------------------------------- |
| Recipe SFT, reasoner video-QA SFT/evaluation, checkpoint preparation | `cosmos3-post-training`    | `models/cosmos3-reasoner/`                          |
| Native generation, reasoner inference, container endpoints           | `cosmos3-inference`        | `inference-service/`, reasoner evaluation contracts |
| Installation, managed launch, credentials, platform/GPU checks       | `cosmos3-setup`            | `execution/`, `scripts/`, `templates/`              |
| Environment and managed-job failures                                 | `cosmos3-env-troubleshoot` | Execution retry and reasoner error references       |

No new skills are added; the four existing skill files are extended in both
`.agents/skills` and `.claude/skills`. Setup and launch references are read on demand,
not loaded for unrelated code edits or ordinary native recipe questions.

The helpers prepare Framework model actions for Docker, SLURM, Kubernetes,
Brev, and virtualenv. Each action's contract declares its supported platforms;
the reasoner training planner supports Docker and SLURM.

## Helpers

From the repository root:

```bash
source tools/cosmos_workflows/env.sh
python "$COSMOS_WORKFLOWS_ROOT/scripts/resolve_cosmos_model.py" \
  --model nvidia/Cosmos3-Nano --action train
```

Shared command paths in the references are relative to this directory. Use
`cd "$COSMOS_WORKFLOWS_ROOT"` for those examples. A component's `scripts/`,
`references/`, and `assets/` paths are relative to that component. Helpers find
their own resources from `__file__`; they need no home-directory skill install.
Framework-owned helpers use Cosmos names, with job records under
`~/.cosmos/jobs` by default (`COSMOS_STATE_DIR` overrides the state root).
Render commands take `--cosmos-job-id`; workflow metadata uses `cosmos_job_id`
and `terminal_runtime_status`. Kubernetes selection uses `COSMOS_K8S_CONTEXT`
and `COSMOS_K8S_NAMESPACE`. Existing state directories are not moved or deleted.
External image URIs, package imports, and service protocols keep their
upstream identifiers.

The launch reference owns review/approval, nested specs, record-before-submit
ordering, and backend monitoring. Planning does not authorize jobs, registry
operations, downloads, or host mutations. Dataset validation/conversion,
reasoner actions, checkpoint handling, service adapters, and lifecycle
logging now have implementations in `cosmos_framework/`. Install the optional
`workflows` extra for these capabilities. GPU dependencies remain separate from
the CPU orchestration helpers.

Build images from this checkout: the root `Dockerfile` for Framework,
`docker/quantize.Dockerfile` for isolated quantization dependencies.
The `*:local` values in `versions.yaml` are local build targets, not published
images. Remote platforms require a user-built registry image/digest or an
explicitly materialized SLURM SQSH.

## Scope

Runtime request/job schemas and tests stay with the helpers that use them.
License headers and metadata remain authoritative; see
`LICENSE` and `NOTICE`. Skill-support file attribution is recorded in
`migration.json`; native runtime attribution lives independently in
[`cosmos_framework/licenses/workflows/`](../../cosmos_framework/licenses/workflows/).

The planner uses native CUDA TorchCodec decoding. The source's precomputed
decoder-compatibility artifact workflow and its always-disabled Framework
branch were omitted; Framework's independent video-override utilities remain
available in the native package. Inference uses the native CLI, without a
second TOML adapter or archive-to-spec substitution layer.

## CPU validation

```bash
cd tools/cosmos_workflows
python -m pip install -r requirements-test.txt
python -m pytest --confcutdir=. -q
python scripts/stamp_versions.py --check --strict-strays
```

Tests cover model/action/image resolution, sealed plans, checkpoint/evaluation
handoffs, execution contracts, relocation, and agent integration without model
weights, GPUs, registry access, or framework GPU fixtures. They do not substitute
for live training or inference validation.

Additional runtime contract tests are in `tests/workflow_runtime/`; run them
with `pytest --confcutdir=tests/workflow_runtime -c /dev/null tests/workflow_runtime`.
Framework owns the native dataset, validation-cache, TorchCodec, and status
implementations. GPU builds, distributed execution,
quantization, model serving, and paid API calls require separate validation.

## AutoML execution helpers

These checkout-local assets support [AutoML on SLURM](../../docs/automl-slurm.md):

- `scripts/cosmos_job_record.py` binds output paths before submission and records
  native job identities and terminal states. Its sibling `redact_secrets.py`
  prevents credential persistence; the record schema is under `execution/artifacts/`.
- `templates/slurm/singlenode.sbatch.tmpl` supplies the Pyxis execution envelope.
  AutoML renders it with scheduler requeue disabled and enforces its own budgets.
- `scripts/nccl_allreduce_probe.py` checks the requested GPU topology before training.
- `scripts/slurm_controller_startup.py` records CPU bootstrap/child failures and
  supplies the cross-node lock observer and pre-fan-out marker. The submitting
  service must schedule observation automatically and reconcile native terminal
  state before any bounded startup retry; see the SLURM guide's startup contract.
- `scripts/slurm_controller_watch.py` persistently observes registered CPU jobs
  and each native restart generation. Run it as an independently supervised,
  deadline-bounded service; its `ready` check gates submission. It never launches
  or replaces jobs. See `examples/automl/slurm_observer.service` at repository root.

The search controller is `cosmos-automl`, not a workflow wrapper in this directory.
Stage these assets from this repository, seal their hashes in the manifest, and
keep records outside synchronized results. Native scheduler queries are the source
of truth for whether jobs are running.

### AutoML helper regression checks

From the repository root, using a test environment with `pytest` and `jsonschema`:

```bash
python -m pytest -o addopts= --confcutdir=tools/cosmos_workflows \
  -o 'python_files=test_*.py *_test.py' tools/cosmos_workflows
```

The AutoML helpers CI job runs this suite without installing the GPU stack.
Observer tests cover registry identity, readiness/locking, native state and
generation checks, and scheduler/storage errors. Batch-process tests check
legacy argv/exec, decimal allocation boundaries, and opt-in bootstrap failures.
Scheduler calls and controller children are mocked; these are not live SLURM,
cross-node filesystem, or GPU-training verification.
