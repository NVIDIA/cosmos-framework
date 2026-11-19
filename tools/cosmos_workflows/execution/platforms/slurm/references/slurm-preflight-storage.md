# SLURM storage and job preflight

Use [SSH and credentials](slurm-ssh-credentials.md) for passwordless access and
approved Enroot authentication. Keep the login/submission host distinct from
compute nodes: a path visible on the login host is not proof of compute access.

## Shared paths

Supply absolute shared-filesystem paths for annotations, media, checkpoint,
cache, SQSH, specs, and results. Resolve logical storage URIs through the
[data staging guide](../../../data-io/guide.md) before invoking the planner.
Do not pass `lustre://`, `slurm://`, or `file://` to native Framework paths.

Inspect each selected annotation and media root from the login host, then
confirm mounts and permissions inside the requested allocation before training.
For example (substitute the approved identity and paths):

```bash
ssh -o BatchMode=yes -i <SSH_KEY_PATH> <SLURM_USER>@<login-host> \
  'test -r <ANNOTATION_PATH> && test -d <MEDIA_ROOT> && test -w <RESULTS_ROOT>'
```

For reasoner SFT, supply `--train-annotation`, `--train-media-root`,
`--validation-annotation`, and `--validation-media-root`. The planner translates
them into the native recipe's environment bindings; there is no
`custom.train_dataset` payload. Stage missing data before GPU submission.
Do not copy an existing corpus or download it again on each retry.

## Records and artifacts

Open the record with `scripts/cosmos_job_record.py open --platform slurm`
before submission; its `results_dir` is the output authority. Record the
scheduler ID returned by `sbatch --parsable` as `backend_ref`. Poll that ID
with `squeue`/`sacct`, read logs over SSH, and cancel with `scancel`.

The reasoner planner binds `COSMOS_STATUS_FILE` and the child-exit file to the
job's results. Framework's DCP checkpoints live separately under
`IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints`. Retain the saved
config and base checkpoint identity alongside them for native export.
