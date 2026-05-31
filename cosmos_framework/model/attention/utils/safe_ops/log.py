# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

torch.compile-safe log wrappers.
"""

from cosmos_framework.model.attention.utils.environment import is_torch_compiling
from cosmos_framework.utils import log


def trace(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.trace(message=message, rank0_only=rank0_only)


def debug(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.debug(message=message, rank0_only=rank0_only)


def info(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.info(message=message, rank0_only=rank0_only)


def success(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.success(message=message, rank0_only=rank0_only)


def warning(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.warning(message=message, rank0_only=rank0_only)


def error(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def critical(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.critical(message=message, rank0_only=rank0_only)


def exception(message: str, rank0_only: bool = True) -> None:
    if not is_torch_compiling():
        log.exception(message=message, rank0_only=rank0_only)
