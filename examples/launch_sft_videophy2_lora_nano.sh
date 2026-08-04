#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for videophy2_lora_nano (LoRA SFT on VideoPhy-2 via
# CosmosDataLoader, Qwen3-VL-8B-Instruct). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/videophy2_lora_nano.toml.
#
# [job].task = "vlm" — picks cosmos_framework/configs/base/reasoner/config.py as the base config.
#
# Required env:
#   VIDEOPHYSICS_ROOT      dir containing videophy2_train/ and videophy2_val/
#                          (each with meta.json + media/ + text/). Populate via
#                          `python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf`.
#
# Optional env:
#   HF_TOKEN               for gated Qwen3-VL-8B-Instruct downloads.
#   VLM_SAFETENSORS_PATH   local directory of pre-converted Qwen3-VL safetensors
#                          (e.g. Cosmos3-Nano LM merged with Qwen3-VL visual via
#                          `cosmos_framework.scripts.convert_model_to_vlm_safetensors`).
#                          When set, plumbed to backbone.safetensors_path via a
#                          tail override. When unset, the framework falls back
#                          to the public Qwen/Qwen3-VL-8B-Instruct HF snapshot.
#   NPROC_PER_NODE         torchrun GPUs per node; default 8. Set 4 on a GB200x4 node.
#   WANDB_API_KEY          the TOML sets wandb_mode="online"; export a key or set
#                          EXTRA_TAIL_OVERRIDES='job.wandb_mode=offline'.
#
# Usage (from the repo root, inside the training container):
#   VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_nano.sh
#   # on a 4-GPU node:
#   NPROC_PER_NODE=4 VIDEOPHYSICS_ROOT=/path/to/videophysics bash examples/launch_sft_videophy2_lora_nano.sh

TOML_FILE="examples/toml/sft_config/videophy2_lora_nano.toml"

# The base recipe enables hf_export so eval_videophy2 can read each save as HF
# safetensors. The export is correct for a LoRA run -- HFExportCallback merges
# the adapter into the base weights -- but it gathers an 8B backbone (~16 GB) onto
# rank 0 and writes it out, which is wasted on a convergence smoke run. Drop
# this line when the run's output is actually meant to be evaluated.
#
# This is the ONE knob the structured TOML cannot express (no [checkpoint]
# hf_export field in the schema); everything else lives in the TOML.
TAIL_OVERRIDES=(
    checkpoint.hf_export.enabled=false
    ${EXTRA_TAIL_OVERRIDES:-}
)

# When VLM_SAFETENSORS_PATH is set, plumb it to backbone.safetensors_path so the
# framework loads weights from the local snapshot while keeping the public HF
# model_name for tokenizer/architecture discovery.
if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
