# Shared execution helpers

These checkout-local assets support [AutoML on SLURM](../../docs/automl-slurm.md):

- `scripts/cosmos_job_record.py` binds output paths before submission and records
  native job identities and terminal states. Its sibling `redact_secrets.py`
  prevents credential persistence; the record schema is under `execution/artifacts/`.
- `templates/slurm/singlenode.sbatch.tmpl` supplies the Pyxis execution envelope.
  AutoML renders it with scheduler requeue disabled and enforces its own budgets.
- `scripts/nccl_allreduce_probe.py` checks the requested GPU topology before training.

The search controller is `cosmos-automl`, not a workflow wrapper in this directory.
Stage these assets from this repository, seal their hashes in the manifest, and
keep records outside synchronized results. Native scheduler queries are the source
of truth for whether jobs are running.

The helpers retain their original Apache-2.0 license and attribution in
[LICENSE](LICENSE) and [NOTICE](NOTICE). Annotation, serving, quantization, data
conversion, and other platform workflows are not part of this focused AutoML change.
