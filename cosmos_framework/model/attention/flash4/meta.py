# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v4 (flash4) Backend: metadata
Always safe to import (as long as torch is available.)
"""

import torch

from cosmos_framework.model.attention.utils.safe_ops import log

# FA4 has Hopper (SM90) kernels as well as Blackwell 10.x/11.x kernels
# (`flash_attn/cute/interface.py::_validate_head_dims`).
# Our arch tags are `major * 10 + minor`.
FLASH4_ARCH_MAJORS = (10, 11)

# (head_dim_qk, head_dim_v) pairs FA4 special-cases outside the standard range on 10.x/11.x.
_DEEPSEEK_SHAPE = (192, 128)
_HD256_SHAPE = (256, 256)


def _arch_supported(arch_tag: int) -> bool:
    return arch_tag == 90 or arch_tag // 10 in FLASH4_ARCH_MAJORS


def get_fwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for forward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    NOTE: FA4 also accepts fp8 (e4m3fn / e5m2) inputs on 10.x, but that path returns a
    bfloat16 output rather than an fp8 one and has not been validated through this
    frontend, so it is deliberately not advertised here.
    """

    if not _arch_supported(arch_tag):
        log.debug(
            "Flash Attention v4 (flash4) only supports compute capability 9.0 (Hopper) and "
            f"10.x/11.x (Blackwell), got {arch_tag=}."
        )
        return []

    return [torch.float16, torch.bfloat16]


def get_bwd_dtypes(arch_tag: int) -> list[torch.dtype]:
    """
    Returns data type choices for backward pass according to arch tag (attention.utils.get_arch_tag).

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        data_type_choices (list): a list of PyTorch data types. Empty if device is not supported.

    """

    if not _arch_supported(arch_tag):
        log.debug(
            "Flash Attention v4 (flash4) only supports compute capability 9.0 (Hopper) and "
            f"10.x/11.x (Blackwell), got {arch_tag=}."
        )
        return []

    return [torch.float16, torch.bfloat16]


def backward_head_dims_supported(head_dim_qk: int, head_dim_v: int) -> bool:
    """
    Whether this (head_dim_qk, head_dim_v) pair is qualified for FA4 backward.

    FA4's backward kernels are narrower than its forward ones, and upstream does not validate
    this: `_validate_head_dims` accepts the full forward range regardless of autograd, and the
    mismatch only surfaces as a CuTe-DSL "ICE IR Verification Failed" when the backward kernel is
    compiled. Measured on sm_100 over every forward-supported pair, backward succeeds exactly when
    both head dims are multiples of 32::

        forward  : 8..128 step 8, plus (192, 128) and (256, 256)
        backward : (32,32) (64,64) (96,96) (128,128) (192,128) (256,256)

    i.e. 40, 56, 72, 88, 104, 120 (and 8, 16, 24, 48, 80, 112) forward-compile fine but fail to
    build a backward. Pairing this with `head_dims_supported` keeps a training use case that FA4
    cannot differentiate falling through to another backend instead of crashing mid-step.
    Retain this conservative alignment for Hopper as well; additional Hopper head dims have
    not been qualified through this frontend.

    Parameters:
        head_dim_qk (int): head dim of the query / key tensors.

        head_dim_v (int): head dim of the value tensor.

    Returns:
        supported (bool): whether FA4 has a backward kernel for this head dim pair.
    """
    if head_dim_qk % 32 != 0 or head_dim_v % 32 != 0:
        log.debug(
            "Flash Attention v4 (flash4) has no backward kernel for this head dim pair; backward "
            f"requires both head dims to be multiples of 32, got {head_dim_qk=}, {head_dim_v=}."
        )
        return False

    return True


def head_dims_supported(head_dim_qk: int, head_dim_v: int, dtype: torch.dtype, arch_tag: int) -> bool:
    """
    Whether FA4 has a kernel for this (head_dim_qk, head_dim_v) pair.

    Mirrors `flash_attn/cute/interface.py::_validate_head_dims` for compute capability
    10.x/11.x, so that an unsupported pair makes the dispatcher fall through to the next
    backend instead of tripping FA4's own assertion.

    Hopper also supports this range; retain the same frontend shape limits on both architectures.
    The upstream Blackwell rule is::

        is_standard_range          = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128
        is_deepseek_shape          = (head_dim, head_dim_v) == (192, 128)
        is_dedicate_kernel_shape   = (head_dim, head_dim_v) == (256, 256)
        is_deepseek_mla_absorbed   = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
        assert (any of the above) and head_dim % alignment == 0 and head_dim_v % alignment == 0

    with ``alignment = 16 // element_size`` (8 for fp16/bf16).

    We deliberately exclude ``is_deepseek_mla_absorbed`` (head_dim_v == 512): it passes
    upstream validation but the kernel itself fails at launch on SM100, so accepting it
    here would turn a clean fall-through into a hard error.

    Parameters:
        head_dim_qk (int): head dim of the query / key tensors.

        head_dim_v (int): head dim of the value tensor.

        dtype (torch.dtype): Data type of tensors, which sets the alignment requirement.

        arch_tag (int): Arch tag for the current CUDA device.

    Returns:
        supported (bool): whether FA4 has a kernel for this head dim pair.
    """
    if not _arch_supported(arch_tag):
        return False

    element_size = torch.empty((), dtype=dtype).element_size()
    alignment = 16 // element_size
    if head_dim_qk % alignment != 0 or head_dim_v % alignment != 0:
        log.debug(
            f"Flash Attention v4 (flash4) requires head dims divisible by {alignment} for {dtype}, "
            f"got {head_dim_qk=}, {head_dim_v=}."
        )
        return False

    if (head_dim_qk, head_dim_v) in (_DEEPSEEK_SHAPE, _HD256_SHAPE):
        return True

    if 8 <= head_dim_qk <= 128 and 8 <= head_dim_v <= 128:
        return True

    log.debug(
        "Flash Attention v4 (flash4) has no kernel for this head dim pair on compute capability "
        f"9.0/10.x/11.x: {head_dim_qk=}, {head_dim_v=}. Supported: both between 8 and 128, or "
        f"{_DEEPSEEK_SHAPE} (DeepSeek), or {_HD256_SHAPE} (hd256)."
    )
    return False
