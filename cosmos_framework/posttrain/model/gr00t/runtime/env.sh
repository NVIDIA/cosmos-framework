#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Sourced by the shared launcher before Ray and the driver; no OCI ENV is required.
source /opt/venv/posttrain/bin/activate
export PYTHONPATH=/workspace/cosmos-framework
export POSTTRAIN_RESOURCE_ROOT=/resources
export POSTTRAIN_SIM_PYTHON=/opt/venv/libero/bin/python
export LIBERO_CONFIG_PATH=/opt/posttrain/libero/config
export LIBERO_MANIFEST_ROOT=/opt/posttrain/libero/manifests
export HF_HOME=/resources/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GR00T_DIT_SDPA_MODE=math
unset GR00T_NOISE_SEED
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/opt/posttrain/libero/egl-nvidia.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [[ -z "${WANDB_API_KEY:-}" && -f /run/secrets/wandb_api_key ]]; then
    export WANDB_API_KEY="$(< /run/secrets/wandb_api_key)"
fi
export IMAGINAIRE_OUTPUT_ROOT=/path/to/resource
export SLURM_LOG_DIR="/path/to/resource}"
export GR00T_RUNTIME_DIR="/path/to/resource}/$(hostname)"
mkdir -p "$GR00T_RUNTIME_DIR"/{tmp,cache,triton,inductor,numba,ray,plasma,shm,ipc,wandb-cache,wandb-config,matplotlib,cuda,torch-extensions,flashinfer,vllm}
export TMPDIR="$GR00T_RUNTIME_DIR/tmp"
export XDG_CACHE_HOME="$GR00T_RUNTIME_DIR/cache"
export TRITON_CACHE_DIR="$GR00T_RUNTIME_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$GR00T_RUNTIME_DIR/inductor"
export TORCH_EXTENSIONS_DIR="$GR00T_RUNTIME_DIR/torch-extensions"
export CUDA_CACHE_PATH="$GR00T_RUNTIME_DIR/cuda"
export FLASHINFER_WORKSPACE_BASE="$GR00T_RUNTIME_DIR/flashinfer"
export VLLM_CACHE_ROOT="$GR00T_RUNTIME_DIR/vllm"
export NUMBA_CACHE_DIR="$GR00T_RUNTIME_DIR/numba"
export RAY_TMPDIR="$GR00T_RUNTIME_DIR/ray"
export RAY_PLASMA_DIRECTORY="$GR00T_RUNTIME_DIR/plasma"
export B1K_SHM_DIR="$GR00T_RUNTIME_DIR/shm"
export B1K_SIM_IPC_DIR="$GR00T_RUNTIME_DIR/ipc"
export WANDB_CACHE_DIR="$GR00T_RUNTIME_DIR/wandb-cache"
export WANDB_CONFIG_DIR="$GR00T_RUNTIME_DIR/wandb-config"
export MPLCONFIGDIR="$GR00T_RUNTIME_DIR/matplotlib"
