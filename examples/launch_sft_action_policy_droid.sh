#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for DROID action-policy SFT on Cosmos3-Nano (8B MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_droid_repro.toml (selects the
# registered `action_policy_droid_nano` experiment; res480, joint_pos 8D +
# use_state, trains the generation + action heads). See
# docs/action_policy_droid_posttraining.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH          DROID LeRobot v3.0 success split (…/droid_lerobot/success)
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#
# Single-node smoke (config/data sanity, a few iters):
#   TAIL_OVERRIDES=(trainer.max_iter=10 checkpoint.save_iter=10 \
#                   dataloader_train.max_samples_per_batch=32)
#   bash examples/launch_sft_action_policy_droid.sh
#
# Multi-node: launch on every worker; the trainer reads torchrun's
# --nnodes/--node_rank. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_droid_repro.toml"
: "${DATASET_PATH:=examples/data/lerobot_v30/droid_lerobot/success}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The experiment reads ${oc.env:DROID_ROOT}; bridge the launcher's DATASET_PATH to it.
export DROID_ROOT="${DROID_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$DROID_ROOT/meta/info.json" ]] || { echo "ERROR: missing $DROID_ROOT/meta/info.json (prepare DROID LeRobot v3.0 — see docs/action_policy_droid_posttraining.md)" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
