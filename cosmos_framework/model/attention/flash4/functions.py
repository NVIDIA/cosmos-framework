# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v4 (flash4) Backend: intermediate APIs
Only safe to import when FLASH4_SUPPORTED is True.
"""

from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.flash4.checks import flash4_attention_check

# Registers imaginaire_flash4::fmha_fwd / fmha_bwd. Going through a real torch.library op with a
# fake kernel (rather than calling flash_attn.cute directly behind `allow_in_graph`) is what makes
# flash4 survive torch.compile -- see custom_ops.py for why.
from cosmos_framework.model.attention.flash4.custom_ops import fmha_fwd
from cosmos_framework.model.attention.masks import CausalType


def flash4_attention(
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
    """
    Runs Flash Attention v4 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`. flash4 implements
            bottom-right causal masking only.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (flash4) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert flash4_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    # FA4 always returns a 2-tuple; the second element is None unless return_lse is set.
    # We always request it, because a caller that merges partial attentions needs an LSE whose
    # storage is the one the autograd node recorded -- recomputing it later would break
    # `merge_attentions` backward.
    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        out, lse_ = fmha_fwd(
            q,
            k,
            v,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            max_seqlen_Q,
            max_seqlen_KV,
            scale,
            is_causal,
            deterministic,
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        assert lse_.dim() == 2  # [H,total_tokens]

        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]
        lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse = fmha_fwd(  # output: [B,N,H,Dv], lse: [B,H,N]
            query,
            key,
            value,
            None,
            None,
            None,
            None,
            scale,
            is_causal,
            deterministic,
        )

    assert isinstance(output, Tensor)
    assert isinstance(lse, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]

    # NOTE: Do NOT call .contiguous on LSE, otherwise Attention Merging backward pass will be
    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    if return_lse:
        return output, lse

    return output
