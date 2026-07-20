#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_sft_edge (VLM dialog SFT on VideoPhy-2 via
# CosmosDataLoader) targeting the Cosmos3-Edge reasoner backbone (public,
# ungated nvidia/Cosmos3-Edge, model_type cosmos3_edge — native HF metadata,
# no remote code; the classes are registered in-framework). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_sft_edge.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/reasoner/config.py as the base config.
#
# Reasoner weights load DIRECTLY from the nvidia/Cosmos3-Edge snapshot resolved
# via model_name: the training loader follows the repo's root safetensors index
# into its weight shards. No converter step and no required weights env var.
#
# Required env:
#   VIDEOPHYSICS_ROOT      dir containing videophy2_train/ and videophy2_val/
#                          (each with meta.json + media/ + text/). Populate via
#                          `python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf`.
#
# Optional env:
#   VLM_SAFETENSORS_PATH   local directory of reasoner safetensors to load
#                          INSTEAD of the nvidia/Cosmos3-Edge snapshot (same
#                          optional override as the nano/super launchers).
#                          When set, plumbed to backbone.safetensors_path via a
#                          tail override; model_name still drives
#                          tokenizer/architecture discovery.
#   HF_TOKEN               NOT needed for nvidia/Cosmos3-Edge (the repo is
#                          ungated); set it only if other downloads in your
#                          environment require authentication.
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4
#                          node — Edge is only 2B and fits a 4-GPU allocation.
#   EXTRA_TAIL_OVERRIDES   extra Hydra-style overrides. On nodes without a
#                          flash-attn wheel fall back to the portable attention
#                          impl:
#                          EXTRA_TAIL_OVERRIDES='model.config.policy.attn_implementation=sdpa'
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_edge.sh
#   # on a 4-GPU node (e.g. GB200x4):
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_edge.sh

TOML_FILE="examples/toml/sft_config/videophy2_sft_edge.toml"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

# Optional: when VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path
# so the framework loads reasoner weights from the local directory instead of the
# nvidia/Cosmos3-Edge snapshot (the public HF model_name still drives
# tokenizer/architecture discovery). When unset, nothing is added and weights come
# directly from the snapshot.
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
