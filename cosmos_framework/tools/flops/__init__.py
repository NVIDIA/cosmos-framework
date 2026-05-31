# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reusable FLOPs estimation utilities for model architectures."""

from cosmos_framework.tools.flops.omni_mot import (
    OmniMoTModelDescriptor,
    compute_omni_mot_flops_per_batch,
    get_omni_mot_model_descriptor,
)
from cosmos_framework.tools.flops.qwen3_vl import (
    compute_qwen3vl_flops,
    compute_qwen3vl_flops_from_config,
)
from cosmos_framework.tools.flops.wan_vae import compute_wan_vae_encoder_flops

__all__ = [
    "OmniMoTModelDescriptor",
    "compute_omni_mot_flops_per_batch",
    "compute_qwen3vl_flops",
    "compute_qwen3vl_flops_from_config",
    "compute_wan_vae_encoder_flops",
    "get_omni_mot_model_descriptor",
]
