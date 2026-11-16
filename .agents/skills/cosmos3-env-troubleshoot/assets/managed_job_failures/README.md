# Managed-job failure bundles (eval fixtures)

Synthetic artifacts of two failed two-node SLURM reasoner training jobs, in the
formats the managed workflow writes: the Cosmos job record
(`tools/cosmos_workflows/scripts/cosmos_job_record.py`), the Framework
`status.json` lifecycle lines (`cosmos_framework/utils/workflow_status.py`), the
job's stdout, `sacct` output, and (second bundle) the node epilog's `dmesg`
capture. Host names, job ids and timestamps are fictional.

- `device_assert/`: job `cosmos3-reasoner-train-7c41d2`, SLURM `4815162`.
- `gpu_fell_off_bus/`: job `cosmos3-reasoner-train-a19f03`, SLURM `4815390`.
