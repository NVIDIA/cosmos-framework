#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_sft_edge (VLM dialog SFT on VideoPhy-2 via
# CosmosDataLoader) targeting the Cosmos3-Edge reasoner backbone
# (public nvidia/Cosmos3-Edge, model_type nemotron_siglip2). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_sft_edge.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/reasoner/config.py as the base config.
#
# Required env:
#   VIDEOPHYSICS_ROOT      dir containing videophy2_train/ and videophy2_val/
#                          (each with meta.json + media/ + text/). Populate via
#                          `python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf`.
#   VLM_SAFETENSORS_PATH   canonical Edge-reasoner weights snapshot, built with
#                          `python -m cosmos_framework.scripts.convert_edge_reasoner_to_vlm_safetensors
#                          --checkpoint-path Cosmos3-Edge -o examples/checkpoints/Cosmos3-Edge-Reasoner-VLM`.
#                          Plumbed to backbone.safetensors_path; model_name
#                          (nvidia/Cosmos3-Edge) still drives arch/tokenizer.
#                          REQUIRED — Cosmos3-Edge's own weights are in
#                          Diffusers-shard layout, so unlike the nano recipe the
#                          public model_name is NOT a valid weight fallback.
#
# Optional env:
#   HF_TOKEN               for the (gated) nvidia/Cosmos3-Edge download used by
#                          the converter + tokenizer/arch discovery.
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4
#                          node — Edge is only 2B and fits a 4-GPU allocation.
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_edge.sh
#   # on a 4-GPU node (e.g. GB200x4):
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_edge.sh

TOML_FILE="examples/toml/sft_config/videophy2_sft_edge.toml"

# VLM_SAFETENSORS_PATH is REQUIRED for edge (unlike nano/super, whose public Qwen
# model_name is a valid weight fallback): Cosmos3-Edge ships its own weights in a
# Diffusers-shard layout, so model_name alone cannot supply reasoner weights. Fail
# fast with a clear message rather than deep inside weight loading.
: "${VLM_SAFETENSORS_PATH:?required for the edge recipe — build it via convert_edge_reasoner_to_vlm_safetensors (see docs/training.md Step 2)}"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

# Plumb the required snapshot to backbone.safetensors_path so the framework loads
# reasoner weights from it while the public HF model_name still drives
# tokenizer/architecture discovery.
TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
