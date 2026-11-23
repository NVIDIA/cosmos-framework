# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Source me from the cosmos-framework checkout: common env for the neural-simulator recipe.
# REPO    = the cosmos-framework checkout (default: the current directory)
# WORKDIR = where data/, runs/, eval/, logs/ live (default: $REPO). Set either before sourcing to override.
export LD_LIBRARY_PATH=
export PYTORCH_ALLOC_CONF=expandable_segments:True
export REPO=${REPO:-$PWD}
export WORKDIR=${WORKDIR:-$REPO}
export RECIPE=$WORKDIR
export LIBERO_ROOT=${LIBERO_ROOT:-$REPO/examples/data/LIBERO_LeRobot_v3/libero_10}
export LIBERO_HOLDOUT_PATH=${LIBERO_HOLDOUT_PATH:-$WORKDIR/data/holdout_19.txt}
export BASE_CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH:-$REPO/examples/checkpoints/Cosmos3-Nano}
export WAN_VAE_PATH=${WAN_VAE_PATH:-$REPO/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}
