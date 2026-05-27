# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Utilities: deterministic mode helpers.
"""

from contextlib import contextmanager

import torch


@contextmanager
def torch_deterministic_mode():
    """Context manager that enables ``torch.use_deterministic_algorithms`` and restores the
    previous state on exit (including the ``warn_only`` flag)."""
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_mode, warn_only=prev_warn_only)
