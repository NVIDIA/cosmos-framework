#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# In-allocation multinode Ray bringup: start a Ray cluster across the job's nodes inside the i4
# container, export RAY_ADDRESS so the driver joins it (control.placement._ensure_ray), then run
# RUN_CMD on the head. Cluster teardown = job end. The Python launcher (utils/launch_multinode.py)
# submits the sbatch job and, in waiting mode, backstops it with an sacct timeout + scancel. This stays a
# thin bring-up with no defensive control logic of its own.
#
# Env overrides: REPO_ROOT, CONTAINER_NAME, GPUS_PER_NODE, RAY_PORT, RUN_CMD, VENV_ACTIVATE,
# RAY_RUNTIME_ENV (trusted shell fragment exported before every Ray daemon and the driver),
# PREFLIGHT_CMD (optional command that must pass on every node before any Ray daemon starts),
# PREFLIGHT_TIMEOUT_SECONDS, POSTTRAIN_RUNTIME_ENV (launcher-generated literal exports after venv activation).
# See operator-provided multinode documentation for the full tutorial.
set -uo pipefail

# Default to the repo root inferred from this script's own location
# (utils/ -> posttrain -> cosmos3 -> projects -> <repo root>); the Python launcher overrides it.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$REPO_ROOT}"
CONTAINER_NAME="${CONTAINER_NAME:-posttrain_mn}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_READY_ATTEMPTS="${RAY_READY_ATTEMPTS:-60}"
RAY_READY_TIMEOUT_SECONDS="${RAY_READY_TIMEOUT_SECONDS:-15}"
PREFLIGHT_TIMEOUT_SECONDS="${PREFLIGHT_TIMEOUT_SECONDS:-120}"
# The launcher is entry-agnostic, so there is no sensible default: required, like the torchrun sibling.
RUN_CMD="${RUN_CMD:?set RUN_CMD to the head command, e.g. python -m ...entrypoints.grpo --config <recipe>}"
# Optional venv activate script (e.g. /opt/venv/cosmos_rl/bin/activate). Sourced by run() for
# EVERY in-container command — Ray daemons AND the driver must resolve the same `ray`/`python`,
# or the driver fails to join the cluster (Ray version mismatch) / misses deps entirely.
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
PREFLIGHT_CMD="${PREFLIGHT_CMD:-}"
[[ "$RAY_READY_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "RAY_READY_ATTEMPTS must be a positive integer" >&2; exit 1; }
[[ "$RAY_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "RAY_READY_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
}
[[ "$PREFLIGHT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PREFLIGHT_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
n=$(echo "$nodes" | wc -l)
head=$(echo "$nodes" | head -1)
# One routable IPv4 for the head, resolved on the launcher node via DNS (`ip` isn't in the image;
# the first getent record avoids `hostname --ip-address` concatenating addresses).
head_ip=$(getent ahostsv4 "$head" | awk 'NR==1{print $1}')
addr="$head_ip:$RAY_PORT"

run() {  # run <node> <cmd...> in the persistent per-node container (venv sourced if configured)
  # Compile caches (inductor, triton AND vLLM's) default NODE-LOCAL: shared defaults ($TMPDIR on
  # $HOME/NFS, ~/.triton/cache on $HOME) make ranks on different nodes/jobs race the same
  # cache -> truncated kernels ("ImportError: ....main.so: file too short") or, when a
  # sibling job is cancelled mid-read, "OSError: [Errno 116] Stale file handle"
  # (2026-07-06, job 1568874). vLLM re-points inductor under VLLM_CACHE_ROOT (default
  # ~/.cache/vllm), so it needs its own default: engines on 256 nodes compiling there at once
  # left an entry without its graph -> FXGraphCacheMiss (2026-09-30, job 721209).
  # Node-local is always coherent.
  srun --overlap --nodes=1 --ntasks=1 -w "$1" --container-name "$CONTAINER_NAME" \
    bash -c "${VENV_ACTIVATE:+source '$VENV_ACTIVATE' && }cd '$CONTAINER_WORKDIR' && export PYTHONPATH='$CONTAINER_WORKDIR' HF_HOME=\"\${HF_HOME:-\$HOME/.cache}\" TORCHINDUCTOR_CACHE_DIR=\"\${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_\$USER}\" TRITON_CACHE_DIR=\"\${TRITON_CACHE_DIR:-/tmp/triton_\$USER}\" VLLM_CACHE_ROOT=\"\${VLLM_CACHE_ROOT:-/tmp/vllm_\$USER}\"; ${RAY_RUNTIME_ENV:+$RAY_RUNTIME_ENV; }${POSTTRAIN_RUNTIME_ENV:+$POSTTRAIN_RUNTIME_ENV; }${*:2}"
}

run_timed() {  # run_timed <node> <seconds> <cmd...>; bounds srun/container attach too
  local node="$1" seconds="$2"
  shift 2
  timeout "$seconds" srun --overlap --nodes=1 --ntasks=1 -w "$node" --container-name "$CONTAINER_NAME" \
    bash -c "${VENV_ACTIVATE:+source '$VENV_ACTIVATE' && }cd '$CONTAINER_WORKDIR' && export PYTHONPATH='$CONTAINER_WORKDIR' HF_HOME=\"\${HF_HOME:-\$HOME/.cache}\" TORCHINDUCTOR_CACHE_DIR=\"\${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_\$USER}\" TRITON_CACHE_DIR=\"\${TRITON_CACHE_DIR:-/tmp/triton_\$USER}\" VLLM_CACHE_ROOT=\"\${VLLM_CACHE_ROOT:-/tmp/vllm_\$USER}\"; ${RAY_RUNTIME_ENV:+$RAY_RUNTIME_ENV; }${POSTTRAIN_RUNTIME_ENV:+$POSTTRAIN_RUNTIME_ENV; }$*"
}

if [ -n "$PREFLIGHT_CMD" ]; then
  for node in $nodes; do
    if ! run_timed "$node" "$PREFLIGHT_TIMEOUT_SECONDS" "$PREFLIGHT_CMD"; then
      echo "ERROR: runtime preflight failed on node $node; Ray was not started" >&2
      exit 1
    fi
  done
fi

# Head: --block keeps the srun step (and thus the Ray daemons, under Slurm's cgroup proctrack) alive.
run "$head" "ray start --head --node-ip-address=$head_ip --port=$RAY_PORT \${RAY_TMPDIR:+--temp-dir=\"\$RAY_TMPDIR\"} \${RAY_PLASMA_DIRECTORY:+--plasma-directory=\"\$RAY_PLASMA_DIRECTORY\"} --num-gpus=$GPUS_PER_NODE --block" &
head_pid=$!
# Workers can only connect once the head's GCS is up — the one wait we actually need.
ready=0
for ((attempt = 1; attempt <= RAY_READY_ATTEMPTS; attempt++)); do
  if ! kill -0 "$head_pid" 2>/dev/null; then
    wait "$head_pid"
    head_rc=$?
    echo "ERROR: Ray head exited before becoming ready (rc=$head_rc)" >&2
    exit 1
  fi
  if run_timed "$head" "$RAY_READY_TIMEOUT_SECONDS" "RAY_ADDRESS=$addr ray status" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "ERROR: Ray head was not ready after $RAY_READY_ATTEMPTS attempts" >&2
  kill "$head_pid" >/dev/null 2>&1 || true
  wait "$head_pid" >/dev/null 2>&1 || true
  exit 1
fi
for w in $(echo "$nodes" | tail -n +2); do
  run "$w" "ray start --address=$addr \${RAY_TMPDIR:+--temp-dir=\"\$RAY_TMPDIR\"} \${RAY_PLASMA_DIRECTORY:+--plasma-directory=\"\$RAY_PLASMA_DIRECTORY\"} --num-gpus=$GPUS_PER_NODE --block" &
done
# No barrier: the driver's placement groups pend until the workers' GPUs register, so Ray does the
# "wait for all nodes" for us. Export NNODES/GPUS_PER_NODE for the entry's detect_mesh().
run "$head" "export RAY_ADDRESS=$addr NNODES=$n GPUS_PER_NODE=$GPUS_PER_NODE; $RUN_CMD"
