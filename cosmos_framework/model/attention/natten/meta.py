# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NATTEN Backend: metadata
Always safe to import (as long as torch is available.)
"""

import torch

from cosmos_framework.model.attention.utils.safe_ops import log


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for forward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug("NATTEN is not supported because compute capability is below the minimum (7.5).")
        return []

    if arch_tag in [100, 103]:
        return [torch.float32, torch.float16, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]

    return [torch.float32, torch.float16, torch.bfloat16]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for backward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug("NATTEN is not supported because compute capability is below the minimum (7.5).")
        return []

    return [torch.float32, torch.float16, torch.bfloat16]
