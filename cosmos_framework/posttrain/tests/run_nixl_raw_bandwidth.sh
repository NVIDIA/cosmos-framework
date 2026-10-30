#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Launch test_workers_nixl_raw_bandwidth.py as 2*PAIRS ranks in one of three topologies, inside
# running Slurm allocations that already hold the posttrain container. No Ray is involved.
#
#   card  <job> <node>                  producers and consumers on one node (PAIRS=1: GPU0 -> GPU1)
#   node  <job> <node0> <node1>         two nodes of one allocation (same NVL72 rack when the
#                                       allocation is rack-local)
#   rack  <job0> <node0> <job1> <node1> one node from each of two allocations in different racks
#                                       (CONTAINER1 names the second allocation's container)
#
# PAIRS (default 1) producers sit on the first node's GPUs 0..PAIRS-1 and pull into the second
# node's GPUs 0..PAIRS-1; PAIRS=4 gives the node-to-node aggregate. CONTAINER (required)
# names the pyxis container; REPO_ROOT is the cosmos-framework checkout; EXTRA_PYTHONPATH
# is appended inside the container (e.g. a pytest install).
set -euo pipefail
MODE="${1:?mode: card | node | rack}"; shift
PAIRS="${PAIRS:-1}"
CONTAINER="${CONTAINER:?set CONTAINER to your pyxis container name}"
CONTAINER1="${CONTAINER1:-$CONTAINER}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PORT="${MASTER_PORT:-29611}"
TEST="cosmos_framework/posttrain/tests/test_workers_nixl_raw_bandwidth.py"
WORLD=$((2 * PAIRS))

node_ip() {  # job node -> first IPv4 of that node, read from inside the allocation
  srun --overlap --jobid "$1" -N1 -w "$2" --container-name "$CONTAINER" bash -c "hostname -I | awk '{print \$1}'" 2>/dev/null | tail -1
}
rank_cmd() {  # rank-expr master_addr gpu-expr -> the command one rank runs inside the container
  echo "cd $REPO_ROOT && RANK=$1 WORLD_SIZE=$WORLD MASTER_ADDR=$2 MASTER_PORT=$PORT LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$3 \
GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME:-eth0} PYTHONPATH=$REPO_ROOT${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH} \
python -m pytest -p no:cacheprovider -q -s --tb=short --L1 -m L1 $TEST"
}

case "$MODE" in
  card)
    JOB="$1"; NODE="$2"
    [ "$PAIRS" -eq 1 ] || { echo "card mode supports PAIRS=1 (GPU0 -> GPU1)" >&2; exit 2; }
    srun --overlap --jobid "$JOB" -N1 -w "$NODE" --container-name "$CONTAINER" bash -c \
      "( $(rank_cmd 0 127.0.0.1 0) ) > /tmp/nixl_raw_rank0.log 2>&1 & $(rank_cmd 1 127.0.0.1 1); wait; grep -E 'cross_|size|MiB' /tmp/nixl_raw_rank0.log" ;;
  node)
    JOB="$1"; NODE0="$2"; NODE1="$3"; MASTER=$(node_ip "$JOB" "$NODE0")
    # Slurm numbers tasks node-major, so ranks 0..PAIRS-1 land on NODE0 (producers) and the rest on NODE1.
    srun --overlap --jobid "$JOB" -N2 -w "$NODE0,$NODE1" --ntasks-per-node="$PAIRS" --container-name "$CONTAINER" bash -c \
      "$(rank_cmd '$SLURM_PROCID' "$MASTER" '$SLURM_LOCALID')" ;;
  rack)
    JOB0="$1"; NODE0="$2"; JOB1="$3"; NODE1="$4"; MASTER=$(node_ip "$JOB0" "$NODE0")
    srun --overlap --jobid "$JOB0" -N1 -w "$NODE0" --ntasks="$PAIRS" --container-name "$CONTAINER" bash -c \
      "$(rank_cmd '$SLURM_PROCID' "$MASTER" '$SLURM_LOCALID')" &
    sleep 3
    srun --overlap --jobid "$JOB1" -N1 -w "$NODE1" --ntasks="$PAIRS" --container-name "$CONTAINER1" bash -c \
      "$(rank_cmd "\$((SLURM_PROCID + $PAIRS))" "$MASTER" '$SLURM_LOCALID')"
    wait ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
