# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

NATTEN Backend: intermediate API stubs
Always safe to import (as long as torch is available.)
"""

from torch import Tensor

from cosmos_framework.model.attention.masks import CausalType


def natten_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    raise RuntimeError(
        "Tried to run NATTEN attention, but it is not supported / available. "
        "Try running with debug logs enabled to see why."
    )


def natten_multi_dim_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    raise RuntimeError(
        "Tried to run NATTEN's Multi-Dimensional attention, but it is not supported / available. "
        "Try running with debug logs enabled to see why."
    )


def natten_multi_dim_attention_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: dict,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    raise RuntimeError(
        "Tried to run NATTEN's variable-length/size Multi-Dimensional attention, but it is not supported / available. "
        "Try running with debug logs enabled to see why."
    )
