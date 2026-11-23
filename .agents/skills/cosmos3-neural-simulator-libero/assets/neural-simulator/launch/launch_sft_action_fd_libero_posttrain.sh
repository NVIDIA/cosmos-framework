#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_fd_libero_posttrain (LIBERO-10 forward
# dynamics / neural simulator, single GPU by default). See
# docs/action_fd_libero_posttrain.md.
#
# Env vars:
#   LIBERO_ROOT                  local libero_10 LeRobot dir (required)
#   LIBERO_HOLDOUT_PATH          held-out episode id list (required; never trained on)
#   BASE_CHECKPOINT_PATH         Base DCP checkpoint (default examples/checkpoints/Cosmos3-Nano)
#   WAN_VAE_PATH                 Wan2.2 VAE .pth (default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth)
#   NPROC_PER_NODE               torchrun --nproc_per_node (default 1)
#   OUTPUT_ROOT                  output root (default outputs/train)
#   EXTRA_TAIL_OVERRIDES         space-separated Hydra overrides
#
#   NPROC_PER_NODE=N also needs EXTRA_TAIL_OVERRIDES="model.parallelism.data_parallel_shard_degree=N
#   trainer.grad_accum_iter=<16/N>" (the TOML pins shard degree 1). MASTER_PORT defaults to 50012
#   (shared with every launcher; set it when co-launching jobs). Relative LIBERO_ROOT /
#   LIBERO_HOLDOUT_PATH are anchored to the repo root.
#
# Smoke (use a distinct job.name or OUTPUT_ROOT: a same-job checkpoint dir auto-resumes):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 job.name=action_fd_libero_smoke10"
#   bash examples/launch_sft_action_fd_libero_posttrain.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_fd_libero_posttrain.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${NPROC_PER_NODE:=1}"
export NPROC_PER_NODE

export LIBERO_ROOT="${LIBERO_ROOT:-}"
export LIBERO_HOLDOUT_PATH="${LIBERO_HOLDOUT_PATH:-}"

EXTRA_DATASET_CHECK='[[ "$LIBERO_ROOT" = /* ]] || LIBERO_ROOT="$WORKDIR/$LIBERO_ROOT"; export LIBERO_ROOT; [[ "$LIBERO_HOLDOUT_PATH" = /* ]] || LIBERO_HOLDOUT_PATH="$WORKDIR/$LIBERO_HOLDOUT_PATH"; export LIBERO_HOLDOUT_PATH; [[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). See docs/action_fd_libero_posttrain.md" >&2; exit 1; }; [[ -f "$LIBERO_HOLDOUT_PATH" ]] || { echo "ERROR: LIBERO_HOLDOUT_PATH not found: '\''$LIBERO_HOLDOUT_PATH'\''" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
