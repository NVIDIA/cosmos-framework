# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

cuDNN Backend: metadata
Always safe to import (as long as torch is available.)
"""

import torch

from cosmos_framework.model.attention.utils.safe_ops import log

# cuDNN's fused attention (SDPA) requires every head dim (Q/K and V) to be a multiple of 8. This
# mirrors PyTorch's own cuDNN SDPA eligibility logic (see check_cudnn_head_dim in sdp_utils.cpp).
CUDNN_HEAD_DIM_ALIGNMENT = 8


def get_max_head_dim(arch_tag: int) -> int:
    """
    Returns the maximum head dim cuDNN's fused attention (SDPA) supports for the given arch.

    PyTorch's cuDNN SDPA eligibility logic enforces an architecture-specific head-dim maximum
    (typically 128). We stay conservative: when unsure we prefer the lower bound so the chooser
    falls back to another backend (e.g. NATTEN) rather than accepting a shape that then fails
    inside the raw ``torch.ops.aten._scaled_dot_product_cudnn_attention`` operator.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100,
            120 for workstation/consumer Blackwell (e.g. RTX PRO 6000).

    Returns:
        max_head_dim (int): the maximum supported head dim, or 0 if the arch is unsupported.
    """
    if arch_tag < 80:
        log.debug("cuDNN Attention is not supported because compute capability is below the minimum (8.0).")
        return 0

    # Hopper's cuDNN fused attention supports head dims up to 256 (FP16/BF16). Every other
    # currently-supported arch (Ampere/Ada and the Blackwell family, including workstation/consumer
    # sm_120/121) is held to the conservative 128 maximum until larger dims are verified.
    if arch_tag == 90:
        return 256
    return 128


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for forward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 80:
        log.debug("cuDNN Attention is not supported because compute capability is below the minimum (8.0).")
        return []

    # As of version 91400 FP8 inference via the python frontend does not seem to work.
    log.debug(f"cuDNN Attention only supports FP16 and BF16 for {arch_tag=}.")
    return [torch.float16, torch.bfloat16]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for backward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 80:
        log.debug("cuDNN Attention is not supported because compute capability is below the minimum (8.0).")
        return []

    # cuDNN SDPA backward supports the same FP16/BF16 precisions as the forward pass.
    log.debug(f"cuDNN Attention backward supports FP16 and BF16 for {arch_tag=}.")
    return [torch.float16, torch.bfloat16]
