# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v4 (flash4) backend checks
"""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.flash4 import FLASH4_SUPPORTED
from cosmos_framework.model.attention.flash4.meta import (
    backward_head_dims_supported,
    get_bwd_dtypes,
    get_fwd_dtypes,
    head_dims_supported,
)
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import get_arch_tag, log_or_raise_error


def flash4_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType,
    is_varlen: bool,
    deterministic: bool = False,
    return_lse: bool = False,
    is_compiling: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the flash4 backend.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with flash4 backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not FLASH4_SUPPORTED:
        target_fn(
            "Flash Attention v4 (flash4) is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
            exception=RuntimeError,
        )
        return False

    arch_tag = get_arch_tag(device)
    fwd_dtypes = get_fwd_dtypes(arch_tag)
    bwd_dtypes = get_bwd_dtypes(arch_tag)
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=fwd_dtypes,
        supported_dtypes_backward=bwd_dtypes,
        supports_mla=True,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Flash Attention v4 (flash4)",
    ):
        target_fn("Flash Attention v4 (flash4) does not support the given inputs.", exception=RuntimeError)
        return False

    # Head dim constraints, mirroring FA4's own `_validate_head_dims` so that an
    # unsupported pair falls through to the next backend rather than tripping its assert.
    if not head_dims_supported(
        head_dim_qk=query_shape[-1],
        head_dim_v=value_shape[-1],
        dtype=dtype,
        arch_tag=arch_tag,
    ):
        target_fn(
            "Flash Attention v4 (flash4) does not support this head dim combination, got "
            f"head_dim_qk={query_shape[-1]}, head_dim_v={value_shape[-1]}.",
            exception=ValueError,
        )
        return False

    # FA4's backward kernels cover a narrower set of head dims than its forward kernels, and
    # upstream does not check this -- the mismatch only shows up as a CuTe-DSL compile failure
    # partway through the first backward. Verify against the narrower set whenever gradients are
    # required, so a training use case falls through to another backend instead of crashing.
    if requires_grad and not backward_head_dims_supported(
        head_dim_qk=query_shape[-1],
        head_dim_v=value_shape[-1],
    ):
        target_fn(
            "Flash Attention v4 (flash4) has no backward kernel for this head dim combination, got "
            f"head_dim_qk={query_shape[-1]}, head_dim_v={value_shape[-1]}; backward requires both "
            "head dims to be multiples of 32. The forward pass alone would have been supported.",
            exception=ValueError,
        )
        return False

    # Upstream's Hopper tests exclude backward with query/key head dims above 192.
    if arch_tag == 90 and requires_grad and query_shape[-1] > 192:
        target_fn("Flash Attention v4 on Hopper supports backward head dims up to 192.", exception=ValueError)
        return False

    # Hopper's GQA backward kernel requires matching query/key and value head dims.
    if arch_tag == 90 and requires_grad and query_shape[-2] != key_shape[-2] and query_shape[-1] != value_shape[-1]:
        target_fn("Flash Attention v4 on Hopper requires equal head dims for GQA backward.", exception=ValueError)
        return False

    # FA4's dedicated Blackwell hd256 backward kernel does not support deterministic mode.
    if requires_grad and deterministic and query_shape[-1] == 256:
        target_fn(
            "Flash Attention v4 does not support deterministic backward with head dim 256.",
            exception=ValueError,
        )
        return False

    # Verifies causal_type is a CausalType instance when is_causal
    # Verifies DontCare is not used unless seqlen_q == seqlen_kv
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    if is_causal and causal_type not in [CausalType.BottomRight, CausalType.DontCare]:
        target_fn("Flash Attention v4 only supports bottom-right causal masking.", exception=ValueError)
        return False

    return True
