#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_lora_edge (LoRA SFT on VideoPhy-2 via
# CosmosDataLoader) targeting the Cosmos3-Edge reasoner backbone (public,
# ungated nvidia/Cosmos3-Edge, model_type cosmos3_edge — native HF metadata,
# no remote code; the classes are registered in-framework). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_lora_edge.toml.
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
#                          INSTEAD of the nvidia/Cosmos3-Edge snapshot.
#   HF_TOKEN               NOT needed for nvidia/Cosmos3-Edge (the repo is ungated).
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4
#                          node — Edge is only 2B and fits a 4-GPU allocation.
#   WANDB_API_KEY          the TOML sets wandb_mode="online"; export a key or set
#                          EXTRA_TAIL_OVERRIDES='job.wandb_mode=offline'.
#   EXTRA_TAIL_OVERRIDES   extra Hydra-style overrides. On nodes without a
#                          flash-attn wheel fall back to the portable attention
#                          impl:
#                          EXTRA_TAIL_OVERRIDES='model.config.policy.attn_implementation=sdpa'
#
# Usage (from the repo root, inside the training container):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_edge.sh
#   # on a 4-GPU node (e.g. GB200x4):
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_edge.sh

TOML_FILE="examples/toml/sft_config/videophy2_lora_edge.toml"

# The base recipe enables hf_export so eval_videophy2 can read each save as HF
# safetensors. The export is correct for a LoRA run -- HFExportCallback merges
# the adapter into the base weights -- but it gathers the full backbone onto
# rank 0 and writes it out, which is wasted on a convergence smoke run. Drop
# this line when the run's output is actually meant to be evaluated.
#
# This is the ONE knob the structured TOML cannot express (no [checkpoint]
# hf_export field in the schema); everything else lives in the TOML.
TAIL_OVERRIDES=(
    checkpoint.hf_export.enabled=false
    ${EXTRA_TAIL_OVERRIDES:-}
)

# Optional: when VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path
# so the framework loads reasoner weights from the local directory instead of the
# nvidia/Cosmos3-Edge snapshot (the public HF model_name still drives
# tokenizer/architecture discovery).
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
