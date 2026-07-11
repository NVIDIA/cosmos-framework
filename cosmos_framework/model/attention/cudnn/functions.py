# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

cuDNN Backend: intermediate APIs
Only safe to import when CUDNN_SUPPORTED is True.

This backend runs cuDNN attention through PyTorch's *own* cuDNN SDPA dispatch rather than the
standalone cuDNN Python frontend (``import cudnn`` + ``cudnn.pygraph``). It always uses
``torch.ops.aten._scaled_dot_product_cudnn_attention`` -- the exact ATen op that
``torch.nn.functional.scaled_dot_product_attention`` lowers to for the cuDNN backend -- and simply
discards the logsumexp when ``return_lse=False``. Unlike SDPA's public API, this op also exposes the
logsumexp statistics needed for the ``return_lse=True`` path.

It is a first-class, meta-registered PyTorch op, so it traces cleanly under
``torch.compile(fullgraph=True)``. This deliberately replaces the previous cuDNN-frontend graph
builder, which had to be hidden behind an opaque custom op (Dynamo cannot trace the frontend's
sourceless enum objects) and required a special TorchInductor flag to run on Thor -- a flag that
regressed other GPUs.
"""

import torch
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.cudnn.checks import cudnn_attention_check
from cosmos_framework.model.attention.masks import CausalType


def _cudnn_sdpa_with_lse(q: Tensor, k: Tensor, v: Tensor, is_causal: bool, scale: float) -> tuple[Tensor, Tensor]:
    """cuDNN SDPA returning logsumexp, via torch's native cuDNN ATen op.

    ``torch.ops.aten._scaled_dot_product_cudnn_attention`` is the op ``F.scaled_dot_product_attention``
    dispatches to for the cuDNN backend; unlike the public API it exposes the logsumexp statistics. The
    op is autograd-aware, so backward is provided automatically (no explicit autograd wrapper here).
    The op requires equal Q/KV head counts, so the caller must expand K/V heads to match Q beforehand
    (GQA/MQA). Inputs are heads-first ``[B, H, S, D]``; returns output ``[B, H, S, Dv]`` and logsumexp
    ``[B, H, S]`` (float32). Note some torch versions return logsumexp with a trailing singleton
    dim (``[B, H, S, 1]``); callers must normalize the rank.

    Only positional args ``(query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal,
    return_debug_mask)`` plus the keyword-only ``scale`` are used; these indices/names have been
    stable across torch versions, and only the first two outputs (output, logsumexp) are consumed.
    """
    results = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q,
        k,
        v,
        None,  # attn_bias
        True,  # compute_log_sumexp
        0.0,  # dropout_p
        is_causal,
        False,  # return_debug_mask
        scale=scale,
    )
    output, logsumexp = results[0], results[1]  # [B,H,S,Dv], [B,H,S] or [B,H,S,1]
    return output, logsumexp


def cudnn_attention(
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
    Runs cuDNN Attention on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`), dispatched through PyTorch's built-in cuDNN SDPA.

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.DontCare`. Required when `is_causal = True`. cuDNN SDPA's causal masking is
            top-left aligned.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Not supported by this backend.

        cumulative_seqlen_KV (Tensor | None): (varlen) Not supported by this backend.

        max_seqlen_Q (int | None): (varlen) Not supported by this backend.

        max_seqlen_KV (int | None): (varlen) Not supported by this backend.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing backend-specific arguments. Only
            ``deterministic`` is recognized (and must be False); any other key raises an error.

        deterministic (bool): Deterministic backward pass required. Not supported by this backend.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last layout
            (`[batch, seqlen, heads]`). Only returned when return_lse is True.
            NOTE: not guaranteed to be contiguous (it is a transposed view) and must not be
            made contiguous, so its results stay correct when merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert cudnn_attention_check(
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

    # cudnn_attention_check should prevent this assertion failing (varlen is not integrated).
    assert not is_varlen

    # cuDNN SDPA takes no extra operator arguments; reject anything unrecognized instead of silently
    # ignoring it.
    if backend_kwargs:
        raise ValueError(f"cuDNN Attention backend received unsupported backend_kwargs: {sorted(backend_kwargs)}.")

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    heads = query.shape[-2]
    heads_kv = key.shape[-2]
    assert heads % heads_kv == 0
    group_size = heads // heads_kv

    # The cuDNN ATen op expects the heads-first layout: [B,H,S,D].
    q = query.transpose(1, 2)  # [B,H,S_q,D]
    k = key.transpose(1, 2)  # [B,H_kv,S_kv,D]
    v = value.transpose(1, 2)  # [B,H_kv,S_kv,D_v]

    # Always take the LSE-capable ATen path (the same op SDPA lowers to for cuDNN) and simply drop
    # the logsumexp when it isn't requested. This keeps a single code path for both cases. Tradeoff:
    # the raw ATen op requires equal Q/KV head counts, so GQA/MQA must materialize expanded K/V here,
    # whereas the public SDPA API (enable_gqa=True) can handle grouped heads without materializing.
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1, output_size=heads)  # [B,H,S_kv,D]
        v = v.repeat_interleave(group_size, dim=1, output_size=heads)  # [B,H,S_kv,D_v]

    output, logsumexp = _cudnn_sdpa_with_lse(q, k, v, is_causal, scale)  # [B,H,S_q,D_v], [B,H,S_q(,1)]

    assert output.dim() == 4
    output = output.transpose(1, 2).contiguous()  # [B,S_q,H,D_v]

    if not return_lse:
        return output

    # The cuDNN ATen op returns logsumexp with a trailing singleton dim on some torch
    # versions ([B,H,S_q,1]); drop it so LSE is rank-3, matching every other backend and
    # what `merge_attentions` requires. Guarded on rank so we neither crash on torch builds
    # that already return [B,H,S_q] nor squeeze the heads dim when S_q/H is 1.
    if logsumexp.dim() == 4:
        logsumexp = logsumexp.squeeze(-1)  # [B,H,S_q]

    # NOTE: Do NOT call .contiguous() on LSE. Attention merging's backward pass requires the
    # output and LSE tensors passed into `merge_attentions` to share the same (transposed)
    # data layout; forcing contiguity here makes that backward pass incorrect (see flash2).
    logsumexp = logsumexp.transpose(1, 2)  # [B,S_q,H]
    assert logsumexp.dim() == 3
    return output, logsumexp
