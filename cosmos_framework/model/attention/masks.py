# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Mask utilities
"""

from enum import Enum


class CausalType(Enum):
    """
    Different types of causal masking supported by backends of interest.
    """

    # Top-Left: Simplified: mask if q_idx < kv_idx
    # CUTLASS / NATTEN default
    # Q = 2, KV = 5:
    # O____
    # OO___
    #
    # Q = 5, KV = 2:
    # O_
    # OO
    # OO
    # OO
    # OO
    TopLeft = 0

    # Bottom-right: mask if q_idx + KV - Q < kv_idx
    # Flash Attention default
    # Q = 2, KV = 5:
    # OOOO_
    # OOOOO
    #
    # Q = 5, KV = 2:
    # __
    # __
    # __
    # O_
    # OO
    BottomRight = 1

    # When seqlen_q == seqlen_kv, we don't care about the causal type
    # because top-left and bottom-right are equivalent
    DontCare = 2
