#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_all_nano — Cosmos3-Nano
# LIBERO-all (4-suite) action-policy SFT (HSDP, full SFT). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_libero_all_repro.toml.
#
# Trains on all 4 LIBERO suites (equal mix). Point LIBERO_ROOT at the
# LIBERO_LeRobot_v3 PARENT dir (containing libero_spatial/object/goal/10), NOT a
# single suite. Use the 20 FPS nvidia/LIBERO_LeRobot_v3. Default recipe is
# HSDP 2x8 (global batch 2048); set NNODES/NODE_RANK/MASTER_ADDR per node.
# The 4-suite mix is coverage-limited — it needs ~4500 iters to reach ~95% on
# libero_10 (max_iter defaults to 5000). See docs/action_policy_libero_sft.md.
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO_LeRobot_v3 PARENT dir (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Pre-sync all 4 suites once:
#   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --local-dir <dir>
#   export LIBERO_ROOT=<dir>
#
# Usage (HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node):
#   LIBERO_ROOT=<dir> bash examples/launch_sft_action_policy_libero_all.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_all_repro.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The libero-all experiment reads ${oc.env:LIBERO_ROOT}/<suite> for each of the 4
# suites; export the PARENT dir so torchrun (launched in this shell) inherits it.
export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='for _s in libero_spatial libero_object libero_goal libero_10; do [[ -f "$LIBERO_ROOT/$_s/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be the LIBERO_LeRobot_v3 parent dir containing all 4 suites (missing $_s; got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --local-dir <dir> (then LIBERO_ROOT=<dir>). See docs/action_policy_libero_sft.md" >&2; exit 1; }; done'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell. Use it for smoke runs,
# e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=offline".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
