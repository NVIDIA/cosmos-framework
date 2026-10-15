# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v4 (flash4) Backend: torch.library custom ops
Only safe to import when FLASH4_SUPPORTED is True.

WHY THIS FILE EXISTS
--------------------
`flash_attn.cute` exposes its kernels as plain `torch.autograd.Function`s that allocate their
output with `torch.empty` and then launch CuTe-DSL kernels by taking data pointers through DLPack.
It registers no `torch.library` op and no fake/meta kernel, so it is invisible to PyTorch's
tracing machinery: `torch._dynamo.allow_in_graph` stops Dynamo from tracing the *Python* wrapper,
but AOTAutograd / make_fx still trace through it. On a cold FA4 JIT cache that raises
"Cannot access data pointer of Tensor (e.g. FakeTensor)"; on a warm one the launch is simply
dropped from the compiled graph and the uninitialised `torch.empty` buffer is returned -- silent
NaN. Verified still true as of upstream tag `fa4-v4.0.0.beta31`.

Wrapping FA4's *raw* entrypoints (`_flash_attn_fwd` / `_flash_attn_bwd`, which are ordinary
functions rather than autograd.Functions) in real custom ops with registered fake kernels makes
the launch an opaque, shape-inferable node that survives compilation, and lets us attach an
explicit autograd formula.
"""

import torch

# pyrefly: ignore  # missing-import
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch import Tensor

_NS = "imaginaire_flash4"


def _is_empty_varlen_batch(
    cumulative_seqlen_Q: Tensor | None,
    max_seqlen_Q: int | None,
    max_seqlen_KV: int | None,
) -> bool:
    """Whether this is a varlen call in which every sequence is zero-length.

    ``max_seqlen`` is the longest sequence in the pack, so both being 0 means the whole batch is
    empty and there is nothing for the kernels to do.

    This lives inside the custom op on purpose. FA4 has no short circuit of its own -- it launches
    the forward kernel and all three backward kernels for an all-zero pack -- whereas NATTEN skips
    the launch inside its library. Putting the branch here rather than in ``flash4_attention``
    keeps it invisible to Dynamo: the op is opaque, so branching on ``max_seqlen`` costs no guard
    and no recompile. (flash3 has to wrap its equivalent check in ``if not is_torch_compiling()``
    for exactly that reason.)
    """
    return cumulative_seqlen_Q is not None and max_seqlen_Q == 0 and max_seqlen_KV == 0


def _empty_batch_outputs(query: Tensor, value: Tensor, cumulative_seqlen_Q: Tensor | None):
    """Zero output and zero LSE, matching what FA4 and NATTEN both already produce here.

    Measured on sm_100: for an all-zero varlen pack both backends return an all-zero output and an
    all-zero logsumexp (not -inf, not NaN), so short-circuiting is value-identical to launching.
    """
    head_dim_v = value.shape[-1]
    total_q, heads = query.shape[0], query.shape[1]
    out = query.new_zeros((total_q, heads, head_dim_v))
    lse = query.new_zeros((heads, total_q), dtype=torch.float32)
    return out, lse


@torch.library.custom_op(f"{_NS}::fmha_fwd", mutates_args=())
def fmha_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cumulative_seqlen_Q: Tensor | None,
    cumulative_seqlen_KV: Tensor | None,
    max_seqlen_Q: int | None,
    max_seqlen_KV: int | None,
    scale: float,
    is_causal: bool,
    deterministic: bool,
) -> tuple[Tensor, Tensor]:
    """FA4 forward. Dense layout is `[B, N, H, D]`; varlen is `[total_tokens, H, D]`.

    ``deterministic`` is unused by the forward kernel; it is carried here so that
    ``setup_context`` can stash it for the backward, which is where it takes effect.
    """
    if _is_empty_varlen_batch(cumulative_seqlen_Q, max_seqlen_Q, max_seqlen_KV):
        return _empty_batch_outputs(query, value, cumulative_seqlen_Q)

    out, lse, _, _ = _flash_attn_fwd(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cumulative_seqlen_Q,
        cu_seqlens_k=cumulative_seqlen_KV,
        max_seqlen_q=max_seqlen_Q,
        max_seqlen_k=max_seqlen_KV,
        softmax_scale=scale,
        causal=is_causal,
        return_lse=True,
    )
    return out, lse


@fmha_fwd.register_fake
def _fmha_fwd_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cumulative_seqlen_Q: Tensor | None,
    cumulative_seqlen_KV: Tensor | None,
    max_seqlen_Q: int | None,
    max_seqlen_KV: int | None,
    scale: float,
    is_causal: bool,
    deterministic: bool,
) -> tuple[Tensor, Tensor]:
    head_dim_v = value.shape[-1]
    if cumulative_seqlen_Q is not None:
        total_q, heads = query.shape[0], query.shape[1]
        out = query.new_empty((total_q, heads, head_dim_v))
        lse = query.new_empty((heads, total_q), dtype=torch.float32)
    else:
        batch, seqlen_q, heads = query.shape[0], query.shape[1], query.shape[2]
        out = query.new_empty((batch, seqlen_q, heads, head_dim_v))
        lse = query.new_empty((batch, heads, seqlen_q), dtype=torch.float32)
    return out, lse


@torch.library.custom_op(f"{_NS}::fmha_bwd", mutates_args=())
def fmha_bwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    grad_out: Tensor,
    lse: Tensor,
    cumulative_seqlen_Q: Tensor | None,
    cumulative_seqlen_KV: Tensor | None,
    max_seqlen_Q: int | None,
    max_seqlen_KV: int | None,
    scale: float,
    is_causal: bool,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """FA4 backward, returning (dq, dk, dv)."""
    if _is_empty_varlen_batch(cumulative_seqlen_Q, max_seqlen_Q, max_seqlen_KV):
        # Mirrors the forward short circuit: FA4 otherwise launches dq/dk/dv kernels that compute
        # nothing. Gradients of an empty attention are zero.
        return torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

    grad_query, grad_key, grad_value = _flash_attn_bwd(
        query,
        key,
        value,
        out,
        grad_out,
        lse,
        scale,
        is_causal,
        cu_seqlens_q=cumulative_seqlen_Q,
        cu_seqlens_k=cumulative_seqlen_KV,
        max_seqlen_q=max_seqlen_Q,
        max_seqlen_k=max_seqlen_KV,
        deterministic=deterministic,
    )
    return grad_query, grad_key, grad_value


@fmha_bwd.register_fake
def _fmha_bwd_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    grad_out: Tensor,
    lse: Tensor,
    cumulative_seqlen_Q: Tensor | None,
    cumulative_seqlen_KV: Tensor | None,
    max_seqlen_Q: int | None,
    max_seqlen_KV: int | None,
    scale: float,
    is_causal: bool,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _fmha_fwd_setup_context(ctx, inputs, output) -> None:
    (
        query,
        key,
        value,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
        scale,
        is_causal,
        deterministic,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(query, key, value, out, lse, cumulative_seqlen_Q, cumulative_seqlen_KV)
    ctx.max_seqlen_Q = max_seqlen_Q
    ctx.max_seqlen_KV = max_seqlen_KV
    ctx.scale = scale
    ctx.is_causal = is_causal
    ctx.deterministic = deterministic


def _fmha_fwd_backward(ctx, grad_out, grad_lse):
    query, key, value, out, lse, cumulative_seqlen_Q, cumulative_seqlen_KV = ctx.saved_tensors
    # FA4 has no gradient path for the logsumexp output; the cosmos_framework frontend only ever
    # differentiates the attention output, and `merge_attentions` consumes LSE as a constant.
    grad_query, grad_key, grad_value = torch.ops.imaginaire_flash4.fmha_bwd(
        query,
        key,
        value,
        out,
        grad_out.contiguous(),
        lse,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        ctx.max_seqlen_Q,
        ctx.max_seqlen_KV,
        ctx.scale,
        ctx.is_causal,
        ctx.deterministic,
    )
    return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


torch.library.register_autograd(
    f"{_NS}::fmha_fwd",
    _fmha_fwd_backward,
    setup_context=_fmha_fwd_setup_context,
)
