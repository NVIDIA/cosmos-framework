#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for transfer_multicontrol_lora_nano: one Cosmos3-Nano LoRA for image->video,
# seg->video and depth->video (three streams in one run, 720p, 8-GPU FSDP). Drives
# cosmos_framework.scripts.train against examples/toml/sft_config/transfer_multicontrol_lora_nano.toml.
# See docs/transfer_multicontrol_lora_nano.md.
#
# Optional env vars (defaults below point under examples/; override to put data or checkpoints on a
# different filesystem):
#   DATASET_PATH          default: examples/data/bonsai-vineyard-sample
#                         (the downloaded dataset as is: {,val_}{vision,seg,depth}.jsonl + clips/)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT           default: outputs/train
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides, e.g. "trainer.max_iter=5" for a smoke run
#
# Transfer loaders hand ~1.35 GB samples to the trainer through /dev/shm: give the container at least
# 200 GB of it (e.g. `docker run --shm-size=200g`), or the loader workers die.
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_transfer_multicontrol_lora_nano.sh

TOML_FILE="examples/toml/sft_config/transfer_multicontrol_lora_nano.toml"
: "${DATASET_PATH:=examples/data/bonsai-vineyard-sample}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='for f in vision seg depth val_vision val_seg val_depth; do [[ -f "$DATASET_PATH/$f.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/$f.jsonl (download the dataset, see docs/transfer_multicontrol_lora_nano.md)" >&2; exit 1; }; done
shm_gb=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)
[[ -z "$shm_gb" || "$shm_gb" -ge 200 ]] || echo "WARNING: /dev/shm has ${shm_gb} GB free; transfer loaders need ~200 GB" >&2'

# Large transfer samples exhaust file descriptors with torch's default fd-based sharing.
export COSMOS_TORCH_SHARING_STRATEGY=file_system

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
