# Run the README workflow on a Slurm cluster

This example runs Jupyter and inference on one compute node with **four GB200
GPUs**. Your laptop connects through an SSH tunnel. Downloads, environment setup,
and GPU work take place inside the compute allocation.

The container commands assume Slurm with the Pyxis/Enroot container integration.
Replace all placeholder paths, account names, partition names, and SSH hosts
with your site's settings. If your cluster uses another container runtime, adapt
the container launch to provide the same framework, Python environment, GPUs,
and shared filesystem.

## Configure the shared workspace

Place this cookbook, a compatible Cosmos framework checkout, and a compatible
ARM64 Cosmos container image on storage accessible to both login and compute
nodes. Use framework revision `cf5d68c00d97ccd2480a2320ed652b92dec63102`, matching
the checkpoint's training contract.

On the login node, set:

```bash
export ROOT=/path/to/shared/workspace
export COOKBOOK="$ROOT/repos/cosmos-framework/.agents/skills/cosmos3-failuregen-yam-bimanual/assets/failuregen"
export FAILUREGEN_FRAMEWORK="$ROOT/repos/cosmos-framework"
export CONTAINER_IMAGE="$ROOT/containers/cosmos-notebook.sqsh"
export PYTHON_BIN=/path/inside/container/to/python
export SLURM_ACCOUNT=YOUR_ACCOUNT
export SLURM_PARTITION=YOUR_GB200_PARTITION
export JUPYTER_PORT=8888
export JUPYTER_IP=127.0.0.1
mkdir -p "$ROOT/logs"
```

`PYTHON_BIN` must point to the container's Cosmos Python interpreter, with
PyTorch/CUDA, the cookbook dependencies, `jupyterlab`, `ipykernel`, and
`huggingface_hub` available. The container also needs `ffmpeg`.

The launcher mounts `ROOT` at the same path inside the container. Keep the
cookbook, framework, outputs, and any symlink targets under that mount. Model
and example-data downloads use `$ROOT/.cache/huggingface`; allow approximately
63.6 GB for the inference bundle plus space for generated outputs.

## Option A: interactive allocation

Use an interactive allocation for first-time setup and troubleshooting:

```bash
salloc --job-name=failuregen-notebook \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --nodes=1 --ntasks=1 --gpus-per-node=4 \
  --cpus-per-task=32 --mem=0 --exclusive --time=04:00:00
```

GPU allocation flags vary by site. Some use `--gres=gpu:4` or require a GPU type;
others expose all four GPUs through a whole-node allocation without a separate
GPU flag. Adjust the request so one task sees exactly four GB200 GPUs. CPU,
memory, exclusivity, and time requests must also fit your site's policy.

Once the allocation is granted, enter the container:

```bash
srun --nodes=1 --ntasks=1 --cpu-bind=none --pty \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$ROOT:$ROOT" \
  --container-workdir="$COOKBOOK" \
  --container-writable \
  --no-container-entrypoint bash
```

Inside the container, activate its Cosmos environment as needed and verify the
interpreter and GPU visibility:

```bash
"$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -c 'import jupyterlab, ipykernel, huggingface_hub'
```

The GPU count should be four. If notebook dependencies are missing, install the
cookbook's notebook extras into this interpreter inside the allocation:

```bash
"$PYTHON_BIN" -m pip install -e "$COOKBOOK[notebook]"
```

If the environment uses `uv` instead of pip, use
`uv pip install --python "$PYTHON_BIN" -e "$COOKBOOK[notebook]"`.
To reuse changes in later jobs, save a separate container image using your site's
container tooling; writable-session changes are not automatically persistent.

Start Jupyter from the same container shell:

```bash
bash "$COOKBOOK/scripts/slurm_notebook.sbatch" --in-container
```

This starts Jupyter in the current allocation without submitting another job.
The terminal prints the compute-node name and authenticated server URL. Follow
the SSH instructions below. Outputs default to
`$ROOT/outputs/failuregen-notebook/job<JOB_ID>`.

## Option B: submit a prepared notebook container

Use this alternative once your container already contains the required
packages. Start from the login-node environment configured above. Explicit
account and partition arguments override the launcher's defaults:

```bash
umask 077
sbatch --parsable --export=ALL \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --gpus-per-node=4 \
  --output="$ROOT/logs/failuregen-notebook-%j.out" \
  "$COOKBOOK/scripts/slurm_notebook.sbatch"
```

Apply the same site-specific GPU allocation adjustments as for `salloc`. The
launcher requests one exclusive node, one task, 32 CPUs, and four hours; override
these resource settings with `sbatch` options if needed while retaining access
to four GPUs. It starts Jupyter but does not automatically run notebook cells.

Use the returned job ID to find the node and server URL:

```bash
squeue -j JOB_ID -o '%.18i %.9T %.30N'
tail -f "$ROOT/logs/failuregen-notebook-JOB_ID.out"
```

## Connect through the login node

Choose the method supported by your cluster. Keep Jupyter token authentication
enabled in both cases.

### Compute-node SSH is available through a jump host

Leave `JUPYTER_IP=127.0.0.1`. On your laptop, substitute your login SSH alias,
username, and allocated compute-node name:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -J YOUR_LOGIN_ALIAS \
  -L 8888:127.0.0.1:8888 YOUR_USER@COMPUTE_NODE
```

### Only the login node can reach the compute-node port

If compute-node SSH is unavailable but login-to-compute TCP connections are
allowed, set `JUPYTER_IP=0.0.0.0` before starting Jupyter or submitting the job.
For an already running interactive server, stop Jupyter with Ctrl-C, then restart
it in the same container:

```bash
export JUPYTER_IP=0.0.0.0
bash "$COOKBOOK/scripts/slurm_notebook.sbatch" --in-container
```

On your laptop, forward through the login node:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8888:COMPUTE_NODE:8888 YOUR_LOGIN_ALIAS
```

This binds the authenticated server to the compute-node network; use it only
where your cluster permits that access. Use the new token if Jupyter restarted.

## Open and run the notebook

Open `http://127.0.0.1:8888/lab` on your laptop and enter the token from the server
terminal or job log. If laptop port 8888 is busy, change the first port in `-L`
to 8889 and open `http://127.0.0.1:8889/lab` instead.

Jupyter serves the shared workspace. Create a new notebook in
`repos/cosmos-framework/.agents/skills/cosmos3-failuregen-yam-bimanual/assets/failuregen/`
and copy the Python blocks from the [recipe README](../../../README.md#commands)
into it in order. Select the **FailureGen Cluster** kernel, which uses the
container's Cosmos interpreter.

In a JupyterLab terminal, authenticate with an account that can read the private
`nvidia/failure-gen-yam-bimanual` repository:

```bash
"$PYTHON_BIN" -c 'from huggingface_hub import login; login()'
```

Run the notebook cells in order. The download cell fetches the pinned inference
bundle, including example inputs, the adapted checkpoint, VAE, and tokenizer.
Leave the optional FD-training cell commented. The inference cell generates the
three supplied examples; no annotation document or full-dataset download is
needed.

## End the session

For an interactive session, stop Jupyter, exit the container, and exit the
allocation shell. For a submitted job, cancel it when finished:

```bash
scancel JOB_ID
```

Closing the browser or SSH tunnel alone does not release the GPUs.
