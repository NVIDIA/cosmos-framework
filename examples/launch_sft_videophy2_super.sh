#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_sft_super (VLM dialog SFT on VideoPhy-2
# via CosmosDataLoader, Cosmos3-Super tier / Qwen3-VL-32B full fine-tune). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_sft_super.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/reasoner/config.py as the base config.
#
# Required env:
#   VIDEOPHYSICS_ROOT  dir containing videophy2_train/ and videophy2_val/
#                      (each with meta.json + media/ + text/). Populate via
#                      `python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf`.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL-32B-Instruct downloads.
#   VLM_SAFETENSORS_PATH   local directory of pre-converted Qwen3-VL safetensors
#                          (e.g. Cosmos3-Super LM merged with the Qwen3-VL-32B visual
#                          tower via `cosmos_framework.scripts.convert_model_to_vlm_safetensors
#                          --checkpoint-path Cosmos3-Super --vlm-model-name Qwen/Qwen3-VL-32B-Instruct`).
#                          When set, plumbed to backbone.safetensors_path via a
#                          tail override. When unset, the framework falls back
#                          to the public Qwen/Qwen3-VL-32B-Instruct HF snapshot.
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4 node.
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_super.sh
#   # on a 4-GPU node (e.g. GB200x4):
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_super.sh

TOML_FILE="examples/toml/sft_config/videophy2_sft_super.toml"

# Super-variant allocator tweak: expandable_segments so the 32B backbone fits
# without OOM during compile/decode. (Unlike launch_sft_vision_super.sh we do NOT
# clear LD_LIBRARY_PATH — this reasoner recipe decodes VideoPhy-2 clips with
# torchcodec, which dlopen()s the CUDA NPP + FFmpeg libs off LD_LIBRARY_PATH; the
# nano videophy2 launcher leaves it untouched for the same reason.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

# When VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path so the
# framework loads weights from the local snapshot (e.g. a Cosmos3-Super LM merged
# with the Qwen3-VL-32B visual tower via
# `cosmos_framework.scripts.convert_model_to_vlm_safetensors`) while keeping the
# public HF model_name for tokenizer/architecture discovery.
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
