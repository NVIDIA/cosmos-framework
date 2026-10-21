# Launch training on two four-GPU nodes

The reference parameter-shard degree is eight. On four-GPU GB200 trays, use
**two nodes and four training processes per node**.
`torchrun --standalone --nproc_per_node=8` requires one eight-GPU node.

Use this launcher with the
[local Stage 0–3 commands](local-training.md).
Run Bash from `FRAMEWORK_DIR` inside a Slurm allocation of two four-GPU nodes.
Request your cluster's partition, account and QoS when allocating.

Use the same checkout, Python environment, local assets and dataset paths on
both nodes. Set `COSMOS_INTERNAL=0`, `COSMOS_TRAINING=1`, `COSMOS_SMOKE=0` and
`COSMOS_DEVICE=cuda` as the stage guide specifies. Export the native
library paths and `CUDA_HOME` from the [environment setup](../../shared/environment.md).
Set `IMAGINAIRE_OUTPUT_ROOT` to a writable shared directory at the same absolute
path on both nodes. Checkpoints and tokenizer AOT packages are written beneath
it. A node-local output directory prevents ranks from exchanging compiled
packages.

Run the selected stage's native `--dryrun` command from one controlling shell
before starting distributed training. Then replace the training guide's
`TRAIN=(torchrun ...)` assignment with the multi-node array below.

## Launch one torchrun agent per node

Choose a free rendezvous port reachable from both nodes. Keep the same run
identity, parent paths and data selection on both nodes.

```bash
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${IMAGINAIRE_OUTPUT_ROOT:?Set the shared output directory}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
export TRANSFER_MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export TRANSFER_MASTER_PORT=29500
TRAIN=(
  srun
  --nodes=2
  --ntasks=2
  --ntasks-per-node=1
  --kill-on-bad-exit=1
  bash -c '
    exec torchrun \
      --nnodes=2 \
      --nproc_per_node=4 \
      --node_rank="$SLURM_PROCID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$TRANSFER_MASTER_ADDR:$TRANSFER_MASTER_PORT" \
      --rdzv_id="$SLURM_JOB_ID" \
      -m cosmos_framework.scripts.train_experiment "$@"
  ' transfer-node
)
```

Run the chosen stage block with this `TRAIN` array. Do not append
`--standalone`. The array forwards the stage's configuration flags and Hydra
overrides unchanged. In a script that uses a complete literal training
command instead, replace its `torchrun ... -m cosmos_framework.scripts.train_experiment` prefix with
`"${TRAIN[@]}"`.

The launch uses eight distributed ranks. Stage 3 runs one student update and
five score-model updates per six-step cycle. For resume, keep the shared output
root and run identity unchanged and use the same launcher in a fresh process.
