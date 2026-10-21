#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# In-allocation multinode torchrun bringup (the torchrun counterpart of launch_multinode.sh). The
# per-node container is already up (the sbatch two-step bring-up); this runs ONE `torchrun` per node
# via --overlap, with a c10d rendezvous at the head — nodes self-assign ranks, so there is NO Ray,
# NO barrier, and NO --node-rank math (torchrun's rendezvous waits for all nodes). The Python
# launcher (utils/launch_multinode.py --mode torchrun) submits the sbatch job and, in waiting mode,
# backstops this with an sacct timeout + scancel. This stays a thin bring-up with no defensive control logic.
#
# Env overrides: REPO_ROOT, CONTAINER_NAME, GPUS_PER_NODE, RAY_PORT (reused as the rendezvous port),
# RUN_CMD (the torchrun payload, e.g. "-m cosmos_framework.posttrain.entrypoints.dpo -- experiment=..."),
# VENV_ACTIVATE, POSTTRAIN_RUNTIME_ENV (launcher-generated literal exports after venv activation).
# See operator-provided multinode documentation for the full tutorial.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
CONTAINER_NAME="${CONTAINER_NAME:-posttrain_mn}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RDZV_PORT="${RAY_PORT:-29500}"
RUN_CMD="${RUN_CMD:?set RUN_CMD to the torchrun payload, e.g. -m ...entrypoints.dpo -- experiment=...}"
# Same env rules as the Ray launcher's run(): one python everywhere (venv if configured), and
# NODE-LOCAL compile caches. Both inductor AND triton default to shared paths ($TMPDIR on $HOME,
# ~/.triton/cache on $HOME): concurrent jobs then race/replace each other's kernels — truncated
# .so ("ImportError: ....main.so: file too short") or, when a sibling job is cancelled mid-read,
# "InductorError: OSError: [Errno 116] Stale file handle" (observed 2026-07-06, job 1568874).
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

nnodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
# One routable IPv4 for the head, resolved on the batch/head node via DNS (same as the Ray path).
head_ip=$(getent ahostsv4 "$head" | awk 'NR==1{print $1}')
rdzv_id="${SLURM_JOB_ID:-torchrun}"  # captured before the unset below; must match across nodes

# One torchrun per node (--ntasks-per-node=1, --overlap into the already-up container). c10d
# rendezvous at the head forms the global group across all nnodes x GPUS_PER_NODE workers; the same
# command runs on every node (no --node-rank). torchrun sets MASTER_ADDR/RANK/WORLD_SIZE/LOCAL_RANK
# for the entry (e.g. entrypoints/dpo.py _run_single, which reads them via init_process_group).
srun --overlap --ntasks-per-node=1 --container-name "$CONTAINER_NAME" bash -c "
  export POSTTRAIN_SLURM_JOB_ID=\"\${SLURM_JOB_ID:-}\"  # the entry's provenance stamp reads it after the unset (R125-D-04)
  unset SLURM_JOB_ID SLURM_NODELIST SLURM_JOBID
  ${VENV_ACTIVATE:+source '$VENV_ACTIVATE' && }cd '$CONTAINER_WORKDIR' || exit 1
  export PYTHONPATH='$CONTAINER_WORKDIR' HF_HOME=\"\${HF_HOME:-\$HOME/.cache}\" \
    TORCHINDUCTOR_CACHE_DIR=\"\${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_\$USER}\" \
    TRITON_CACHE_DIR=\"\${TRITON_CACHE_DIR:-/tmp/triton_\$USER}\"
  ${POSTTRAIN_RUNTIME_ENV:-}
  # CUDA JIT does not create its cache directory; keep it node-local and job-scoped.
  export PYTORCH_KERNEL_CACHE_PATH=\"\${PYTORCH_KERNEL_CACHE_PATH:-/tmp/posttrain_kernels_$rdzv_id}\"
  mkdir -p \"\$PYTORCH_KERNEL_CACHE_PATH\" || exit 1
  torchrun --nnodes=$nnodes --nproc-per-node=$GPUS_PER_NODE \
    --rdzv-backend=c10d --rdzv-endpoint=$head_ip:$RDZV_PORT --rdzv-id=$rdzv_id \
    $RUN_CMD
"
