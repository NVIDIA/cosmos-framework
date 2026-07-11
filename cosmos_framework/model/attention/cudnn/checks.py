# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

cudNN backend checks
"""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.cudnn import CUDNN_SUPPORTED
from cosmos_framework.model.attention.cudnn.meta import CUDNN_HEAD_DIM_ALIGNMENT, get_bwd_dtypes, get_fwd_dtypes, get_max_head_dim
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import get_arch_tag, log_or_raise_error


def cudnn_sdpa_eligible(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    arch_tag: int,
    raise_error: bool = False,
) -> bool:
    """
    Mirror the cuDNN SDPA eligibility constraints that the generic ``attention_tensor_checks`` does
    not cover, so unsupported shapes are rejected here (and the chooser can fall back to another
    backend such as NATTEN) instead of failing deep inside the raw
    ``torch.ops.aten._scaled_dot_product_cudnn_attention`` operator.

    PyTorch's cuDNN eligibility logic additionally rejects:
        * a key/value sequence length of 1, and
        * head dims (Q/K and V) that are not a multiple of ``CUDNN_HEAD_DIM_ALIGNMENT`` (8) or that
          exceed the architecture-specific maximum (see ``get_max_head_dim``, typically 128).

    This helper is intentionally tensorless and takes ``arch_tag`` directly (rather than a device)
    so the eligibility rules can be unit-tested for a specific architecture (e.g. SM120) without
    requiring that physical GPU.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        arch_tag (int): Arch tag for the target CUDA device (see ``get_arch_tag``). Example: 90 for
            Hopper, 120 for workstation/consumer Blackwell.

        raise_error (bool): whether to raise an error if any checks fail, instead of just returning
            False. Default is False.

    Returns:
        success (bool): whether the shapes satisfy cuDNN's SDPA eligibility constraints.
    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    # cuDNN's fused attention rejects a KV sequence length of 1. Layout is [batch, seqlen, heads,
    # head_dim], so the KV sequence length is index 1 of the key shape.
    seqlen_kv = key_shape[1]
    if seqlen_kv == 1:
        target_fn(
            f"cuDNN Attention does not support a key/value sequence length of 1, got {seqlen_kv=}.",
            exception=ValueError,
        )
        return False

    max_head_dim = get_max_head_dim(arch_tag)
    # QK and V head dims are the last dimension of each tensor. Q/K head-dim equality and (absent
    # MLA) Q/V head-dim equality are already enforced by attention_tensor_checks, but we validate
    # both explicitly for precise, actionable error messages.
    for name, head_dim in (("QK", query_shape[-1]), ("V", value_shape[-1])):
        if head_dim % CUDNN_HEAD_DIM_ALIGNMENT != 0:
            target_fn(
                f"cuDNN Attention requires the {name} head dim to be a multiple of "
                f"{CUDNN_HEAD_DIM_ALIGNMENT}, got {head_dim=}.",
                exception=ValueError,
            )
            return False
        if head_dim > max_head_dim:
            target_fn(
                f"cuDNN Attention on this architecture ({arch_tag=}) supports a maximum {name} head "
                f"dim of {max_head_dim}, got {head_dim=}.",
                exception=ValueError,
            )
            return False

    return True


def cudnn_attention_check(
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
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the cuDNN backend.
    Runs the common and cuDNN-specific checks. Returns False if any checks fail, otherwise True.

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
        success (bool): whether use case is compatible with cuDNN backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not CUDNN_SUPPORTED:
        target_fn(
            "cuDNN is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
            exception=RuntimeError,
        )
        return False

    if deterministic:
        target_fn("cuDNN Attention does not support deterministic mode.", exception=RuntimeError)
        return False

    # cuDNN Attention supports both the forward (inference) and backward (training) passes: it runs on
    # native, differentiable PyTorch ops (F.scaled_dot_product_attention / the cuDNN SDPA ATen op), so
    # autograd handles the backward pass. attention_tensor_checks validates the backward dtype below
    # (see get_bwd_dtypes) whenever operands require grad.
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
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="cuDNN Attention",
    ):
        target_fn("cuDNN does not support the given inputs.", exception=RuntimeError)
        return False

    # Mirror PyTorch's cuDNN SDPA eligibility constraints that attention_tensor_checks does not
    # cover (KV sequence length 1, head-dim alignment, and the architecture-specific head-dim
    # maximum). cuDNN is first in the Blackwell backend order, so without this the chooser would
    # accept these shapes and then fail inside the raw cuDNN ATen operator instead of falling back
    # to another backend (e.g. NATTEN).
    if not cudnn_sdpa_eligible(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        arch_tag=arch_tag,
        raise_error=raise_error,
    ):
        return False

    if is_varlen:
        target_fn("Varlen for cuDNN Attention is not integrated yet.", exception=RuntimeError)
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

    if is_causal and causal_type not in [CausalType.TopLeft, CausalType.DontCare]:
        target_fn("cuDNN Attention only supports top-left causal masking for now.", exception=RuntimeError)
        return False

    return True
