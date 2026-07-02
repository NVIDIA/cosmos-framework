#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_nano — Cosmos3-Nano LIBERO
# action-policy SFT (HSDP, full SFT). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/action_policy_libero_repro.toml.
#
# Point LIBERO_ROOT at the libero_10 suite ONLY. Use the 20 FPS
# nvidia/LIBERO_LeRobot_v3. The default recipe is HSDP 2x8 (global batch 2048);
# set NNODES/NODE_RANK/MASTER_ADDR per node.
# See docs/action_policy_libero_sft.md.
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10 (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Pre-sync the 20 FPS suite once:
#   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' --local-dir <dir>
#   export LIBERO_ROOT=<dir>/libero_10
#
# Usage (HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node):
#   LIBERO_ROOT=<dir>/libero_10 bash examples/launch_sft_action_policy_libero.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_repro.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} directly (a LOCAL LeRobot dir);
# export it so torchrun (launched in this shell) inherits it.
export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' --local-dir <dir> (then LIBERO_ROOT=<dir>/libero_10). See docs/action_policy_libero_sft.md" >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell. Use it for smoke runs,
# e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=offline".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
