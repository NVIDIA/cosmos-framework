#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Run the repository-owned two-node asymmetric NIXL qualification inside a Slurm allocation.
# The container image must already satisfy the package contract checked by nixl_preflight.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VLLM_VENV="${VLLM_VENV:-/opt/venv/cosmos_rl}"
PYTHON_BIN="$VLLM_VENV/bin/python"
# This wrapper runs on the Slurm launcher host, while VLLM_VENV exists inside each persistent
# container. Checking it here rejects a valid image before the all-node in-container preflight can
# run; VENV_ACTIVATE/PREFLIGHT_CMD below validate the path on every participating node instead.

UCX_DEVICES="${UCX_NET_DEVICES_CFG:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,eth0}"
if [ -n "${NIXL_UCX_MODULE_DIR_CFG:-}" ]; then
  NIXL_UCX_SETUP="export UCX_MODULE_DIR=$NIXL_UCX_MODULE_DIR_CFG"
else
  NIXL_UCX_SETUP="NIXL_PURELIB=\$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path(\"purelib\") or \"\")'); \
test -n \"\$NIXL_PURELIB\" || { echo \"sysconfig purelib path is empty\" >&2; exit 1; }; \
export UCX_MODULE_DIR=\$NIXL_PURELIB/nixl_cu13.libs/ucx"
fi
NIXL_ENV="export PYTHONPATH=$REPO_ROOT; \
export UCX_TLS=rc_mlx5,dc_mlx5,cuda_copy,cuda_ipc,sm,self,tcp; \
export UCX_NET_DEVICES=$UCX_DEVICES; \
$NIXL_UCX_SETUP; \
export UCX_MAX_RMA_RAILS=4; \
export UCX_CUDA_IPC_ENABLE_MNNVL=yes"
PREFLIGHT_CMD="$PYTHON_BIN -m cosmos_framework.posttrain.utils.nixl_preflight"
TEST_CMD="$NIXL_ENV; export POSTTRAIN_NIXL_CROSS_NODE=1; \
$PYTHON_BIN -m pytest -q -rP --L1 --GPU \
cosmos_framework/posttrain/tests/test_workers_nixl_e2e.py::test_probe_buffer_can_be_deregistered_and_reallocated \
cosmos_framework/posttrain/tests/test_workers_nixl_e2e.py::test_second_sync_delivers_new_weights_not_stale_ones \
cosmos_framework/posttrain/tests/test_workers_nixl_e2e.py::test_real_actor_to_actor_value_probe \
cosmos_framework/posttrain/tests/test_workers_nixl_e2e.py::test_forced_cross_node_actor_to_actor_probe \
cosmos_framework/posttrain/tests/test_comm_weight_sync_asym.py::test_vllm_sync_nixl_hsdp2x2_policy4_to_rollout2_selected_actors \
cosmos_framework/posttrain/tests/test_comm_weight_sync_asym.py::test_vllm_sync_nixl_policy2_to_rollout4_direct_actor_fanout \
cosmos_framework/posttrain/tests/test_comm_weight_sync_asym.py::test_vllm_sync_nixl_multi_engine_direct_actor_fanout"
RAY_RUNTIME_ENV="$NIXL_ENV; export POSTTRAIN_NIXL_RAYLETS_READY=1"

export REPO_ROOT
export CONTAINER_NAME="${CONTAINER_NAME:-posttrain_mn}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export RAY_PORT="${RAY_PORT:-6380}"
export VENV_ACTIVATE="${VENV_ACTIVATE:-$VLLM_VENV/bin/activate}"
export PREFLIGHT_CMD
export RUN_CMD="$TEST_CMD"
export RAY_RUNTIME_ENV
exec "$REPO_ROOT/cosmos_framework/posttrain/utils/launch_multinode.sh"
