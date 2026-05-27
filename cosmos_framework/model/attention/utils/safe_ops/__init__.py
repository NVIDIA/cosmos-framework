# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Safe operations for torch.compile: operations that should be disabled or modified
when in a torch.compiled regions.
"""

from cosmos_framework.model.attention.utils.safe_ops import functools, log

__all__ = ["log", "functools"]
