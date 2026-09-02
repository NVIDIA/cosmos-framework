# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal (AR / KV-cache) attention dispatch functions.

These functions were extracted from ``cosmos_framework.model.generator.mot.attention``
so that the base VFM attention module no longer depends on concrete KV-cache
types.  The dispatch is installed on each attention layer by
``OmniMoTCausalModel.install_attention_dispatch()``.
"""

from collections.abc import Callable

import torch

from cosmos_framework.model.attention import (
    attention,
    multi_dimensional_attention,
    multi_dimensional_attention_varlen,
)
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.generator.mot.attention import SplitInfo, two_way_attention
from cosmos_framework.model.generator.mot.attention import dispatch_attention as vfm_dispatch_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    from_und_gen_splits,
    get_causal_seq,
    get_full_only_seq,
    get_gen_seq,
)
from cosmos_framework.model.generator.utils.kv_cache import (
    ARMemoryValue,
    KVTrainMemoryValue,
    TFNoisyMemoryValue,
    TFReplayCleanMemoryValue,
)

BridgeFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class MergeAttentionsBridge(torch.autograd.Function):
    """Autograd bridge that preserves ``merge_attentions``' data-pointer
    contract across an arbitrary invertible shape-changing op.

    ``merge_attentions`` (NATTEN's ``MergeAttentionsAutogradFn``, see
    ``data_local/attn_merge.py``) implements its backward via a hack:
    instead of computing real gradients w.r.t. its inputs, it writes the
    *merged* output and LSE back into each input tensor's storage via
    ``.data.copy_()`` and returns the upstream gradient unchanged.  The
    attention kernel that produced the input then reads the patched
    storage as its saved ``O`` / ``LSE`` during its own backward, and its
    standard backward formula then computes the gradient *as if* the
    kernel had produced the merged output.

    This contract is broken whenever a tensor-allocating op (e.g.
    ``torch.cat`` to insert a zero-padded frame 0) sits between the
    attention kernel and ``merge_attentions``: the op's result has its
    own storage, so ``merge_attentions``' ``.data.copy_()`` patches the
    op's output storage, not the kernel's saved output → the kernel's
    backward then runs against unpatched data and produces gradients
    that don't account for the merge.

    This Function rebridges the contract across any invertible action
    on the inner ``(out, lse)`` pair.  The action is supplied as two
    callables:

    - ``forward_fn(out_inner, lse_inner) -> (out_full, lse_full)``: the
      invertible action applied in the forward pass (e.g. cat-pad a
      frame, permute, scatter, …).  ``out_full`` / ``lse_full`` are the
      tensors that ``merge_attentions`` will receive (and later patch in
      its backward).
    - ``inverse_fn(out_full, lse_full) -> (out_inner, lse_inner)``: the
      exact inverse — undoes ``forward_fn`` so that ``inverse_fn ∘
      forward_fn`` is the identity on the inner tensors.

    For ``forward_fn`` that is linear with constant-fill (cat-pad,
    permutation, scatter with zeros, …), the *gradient* w.r.t. the
    inner input is also ``inverse_fn`` applied to the upstream gradient
    — so the same callable serves both backward roles below.  If your
    forward is not in this class (e.g. it has trainable parameters, or
    is non-linear), do not use this bridge.

    Backward:
      Runs *after* ``merge_attentions``' backward (autograd is
      reverse-order), at which point the outer tensors have already
      been patched.  We then apply ``inverse_fn`` to the patched outer
      data and ``.data.copy_()`` it into the inner kernel's saved
      output / LSE storage.  When the inner kernel's backward runs
      next, it reads the patched data and produces gradients relative
      to the merged attention.  We also return ``inverse_fn`` of the
      upstream gradient as the gradient w.r.t. the inner inputs.
    """

    @staticmethod
    def forward(
        ctx,
        out_inner: torch.Tensor,
        lse_inner: torch.Tensor,
        forward_fn: BridgeFn,
        inverse_fn: BridgeFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_full, lse_full = forward_fn(out_inner, lse_inner)
        out_full = out_full.contiguous()
        lse_full = lse_full.contiguous()
        # Save BOTH the inner kernel outputs (target of the .data.copy_ back)
        # AND the outer tensors (source of the patched data, as patched
        # by merge_attentions.backward before our backward runs).
        ctx.save_for_backward(out_inner, lse_inner, out_full, lse_full)
        ctx.inverse_fn = inverse_fn
        return out_full, lse_full

    @staticmethod
    def backward(
        ctx,
        grad_out_full: torch.Tensor,
        grad_lse_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        out_inner, lse_inner, out_full, lse_full = ctx.saved_tensors
        inverse_fn: BridgeFn = ctx.inverse_fn
        # By now merge_attentions.backward has already run and patched
        # out_full.data / lse_full.data with the merged output / LSE.
        # Apply inverse_fn to recover the data corresponding to the inner
        # attention's range and write it into the inner kernel's saved
        # output / LSE so the kernel's backward (which runs after ours)
        # reads the merged data.
        patched_out_inner, patched_lse_inner = inverse_fn(out_full, lse_full)
        out_inner.data.copy_(patched_out_inner.data)
        lse_inner.data.copy_(patched_lse_inner.data)
        # For linear-with-constant-fill forward_fn (cat-pad, permute,
        # scatter-with-zeros, …), the backward gradient operator equals
        # inverse_fn.  (Constant rows added by forward_fn are not
        # functions of the inner inputs, so their gradient does not flow
        # back; the remaining rows pass through.)
        grad_out_inner, grad_lse_inner = inverse_fn(grad_out_full, grad_lse_full)
        return grad_out_inner, grad_lse_inner, None, None


def _bridge_lse_with_neg_inf_mask(
    out: torch.Tensor,
    lse: torch.Tensor,
    has_real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask *lse* to ``-inf`` when *has_real* is False, preserving the
    ``merge_attentions`` data-pointer contract via :class:`MergeAttentionsBridge`.

    Used to neutralize a merge component whose real K/V is logically empty.
    Production runs the FA varlen kernel with a clamped ``cumulative_seqlen_KV``
    (``[0, max(real, 1)]``) so the kernel never sees a zero-length range
    (which can return NaN in fp32).  This helper then masks the resulting
    spurious LSE to ``-inf`` so the LSE-rescaled merge gives that
    component weight zero.

    The mask is a ``torch.where`` that allocates a new tensor, which
    would normally break ``merge_attentions``' data-pointer contract on
    the LSE side (the kernel's saved LSE would no longer share storage
    with the tensor ``merge_attentions`` patches in its backward).  The
    wrapping :class:`MergeAttentionsBridge` propagates the merge's
    ``.data.copy_()`` patch back into the kernel's saved LSE storage so
    the kernel's backward sees the merged LSE.

    Used by ``three_way_attention_with_kv_cache`` for both:
      - The cached-video CA component, gated by ``has_cached_video``.
      - The video-to-text CA component, gated by ``has_caption``.
    """
    neg_inf = torch.finfo(lse.dtype).min

    def _forward(o: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return o, torch.where(has_real, l, neg_inf)

    def _inverse(o: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape-preserving forward → identity inverse: the merge-patched
        # (O_merged, LSE_merged) flow back unchanged to the kernel's saved
        # O / LSE storage.  (Strict invertibility — recovering the original
        # pre-where LSE — is not needed; we only need backward to push the
        # merged values into the kernel's storage.)
        return o, l

    return MergeAttentionsBridge.apply(out, lse, _forward, _inverse)


def dispatch_varlen_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cumulative_seqlen_Q: torch.Tensor,
    cumulative_seqlen_KV: torch.Tensor,
    max_seqlen_Q: int,
    max_seqlen_KV: int,
    has_real: torch.Tensor,
    clamp_empty_varlen_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Varlen cross-attention plus optional empty-KV LSE neg-inf masking.

    Args:
        q, k, v: ``[B, S, H, D]`` query / key / value tensors.
        cumulative_seqlen_Q, cumulative_seqlen_KV: varlen offsets.  For kernels
            which do not support 0-length ranges, these should be clamped to
            length >= 1, and clamp_empty_varlen_kv should be True.
        max_seqlen_Q, max_seqlen_KV: static padded buffer sizes.
        has_real: scalar bool tensor that is True iff the K/V side has
            any real (non-padding) tokens for this attention component.
        clamp_empty_varlen_kv: gate for the bridge-masked LSE override.
            Set True for kernels that don't support 0-length varlen
            ranges (or platforms with allocator-state-driven NaN bugs
            that the clamp+mask happens to mask out); False to skip the
            mask (slightly faster) when the kernel is known to handle
            empty ranges correctly.

    Returns:
        ``(out, lse)``.  ``out`` is ``[B, S, H, D]``; ``lse`` is
        ``[B, S, H]``.
    """

    # ``cumulative_seqlen_KV = [0, max(real_video_cache_len, 1)]`` restricts
    # the FA kernel to the real (non-padding) portion of the K/V buffer.
    # Without this, when the K/V buffer is only partially filled
    # the padded zero-K/V positions dilute the softmax and degrade quality.
    #
    # This function is called when attending to the text caption (which can
    # be variable-length) or attending the past video history in the KV-cache.
    #
    # ``max_seqlen_KV`` is the static padded buffer size (compile-time
    # const), which keeps the K/V shape stable across frames so Dynamo
    # and CUDA graphs do not recompile / recapture.
    #
    # We deliberately always take this varlen path — even when the cache
    # is fully saturated (``real == max``) — instead of branching on
    # ``Optional[Tensor]`` offsets.  Branching would compile *two*
    # specializations.
    out, lse = attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
        return_lse=True,
        backend="natten",  # Sanity check; this should be the only option.
    )

    # For kernels which do not natively support 0-length KV buffers,
    # cumulative_seqlen_KV must be clamped to length >= 1.
    if clamp_empty_varlen_kv:
        out, lse = _bridge_lse_with_neg_inf_mask(out, lse, has_real)
    return out, lse


def _compact_padded_und_cache_for_varlen(
    und_kv: torch.Tensor,  # [B,S_und_max,H,D]
    nonund_kv: torch.Tensor,  # [B,S_nonund,H,D]
    real_und_len_t: torch.Tensor,  # [1]
) -> torch.Tensor:  # [B,S_und_max+S_nonund,H,D]
    """Move fixed-size und-cache padding to the ignored varlen suffix.

    Post-saturation static compile pads cached text K/V to ``S_und_max`` so
    prompt length changes do not retrace. Dense attention cannot consume that
    right-padded cache directly because the padded tail would sit before the
    generated tokens. This helper compacts the real prefix to
    ``[und_real | nonund_real | ignored_tail]`` with tensor indexing, so the
    varlen ``cu_seqlens_kv`` prefix excludes padding without Python
    value-dependent slicing.
    """
    # Layout transformation:
    #   input  und_kv    = [und_real | und_pad]
    #   input  nonund_kv = [gen_history | current]
    #   source concat    = [und_real | und_pad | gen_history | current]
    #   output compacted = [und_real | gen_history | current | ignored_zero_tail]
    # This makes the real KV entries a contiguous prefix, so
    # ``cumulative_seqlen_KV = [0, real_und_len + nonund_len]`` excludes the
    # padded und tail while the tensor shape remains fixed for torch.compile.
    kv_source = torch.cat([und_kv, nonund_kv], dim=1)  # [B,S_und_max+S_nonund,H,D]
    max_und_len = und_kv.shape[1]
    nonund_len = nonund_kv.shape[1]
    max_total_len = max_und_len + nonund_len
    positions = torch.arange(max_total_len, device=und_kv.device, dtype=real_und_len_t.dtype)  # [S_total_max]
    real_und_len = real_und_len_t[0]  # []
    real_total_len = real_und_len + nonund_len  # []
    in_und = positions < real_und_len  # [S_total_max]
    in_real = positions < real_total_len  # [S_total_max]
    nonund_source_positions = positions - real_und_len + max_und_len  # [S_total_max]
    source_positions = torch.where(in_und, positions, nonund_source_positions)  # [S_total_max]
    source_positions = torch.where(in_real, source_positions, torch.zeros_like(source_positions))  # [S_total_max]
    gather_index = source_positions.to(dtype=torch.long)  # [S_total_max]
    gather_index = gather_index.view(1, max_total_len, 1, 1)  # [1,S_total_max,1,1]
    gather_index = gather_index.expand(
        kv_source.shape[0],
        max_total_len,
        kv_source.shape[2],
        kv_source.shape[3],
    )  # [B,S_total_max,H,D]
    compacted = torch.gather(kv_source, dim=1, index=gather_index)  # [B,S_total_max,H,D]
    zeros = torch.zeros_like(compacted)  # [B,S_total_max,H,D]
    valid_mask = in_real.view(1, max_total_len, 1, 1)  # [1,S_total_max,1,1]
    return torch.where(valid_mask, compacted, zeros)  # [B,S_und_max+S_nonund,H,D]


def three_way_attention_no_memory_ac_safe(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> SequencePack:
    """No-memory three-way attention using the interactive AC-safe merge."""
    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)  # [N_und,H,D]
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)  # [N_und,H_kv,D]
    causal_v, _ = get_causal_seq(packed_value_states)  # [N_und,H_kv,D]
    # For gen→und cross-attention use normed und K when provided, else fall back to raw K.
    if packed_key_states_normalized is not None:
        causal_k_normalized, causal_k_normalized_offsets = get_causal_seq(
            packed_key_states_normalized
        )  # [N_und,H_kv,D]
    else:
        causal_k_normalized, causal_k_normalized_offsets = causal_k, causal_k_offsets
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)  # [N_gen,H,D]
    full_k, full_k_offsets = get_full_only_seq(packed_key_states)  # [N_gen,H_kv,D]
    full_v, _ = get_full_only_seq(packed_value_states)  # [N_gen,H_kv,D]

    if attention_meta is not None and attention_meta.null_action_supertokens:
        full_v = full_v.clone()  # [N_gen,H_kv,D]
        starts = full_q_offsets[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)  # [B*N_null]
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,H,D]
        causal_k.unsqueeze(0),  # [1,N_und,H_kv,D]
        causal_v.unsqueeze(0),  # [1,N_und,H_kv,D]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,H,D]
    assert isinstance(causal_res, torch.Tensor)
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # [N_und,H*D]

    if natten_metadata is None:
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_gen,H,D]
            full_k.unsqueeze(0),  # [1,N_gen,H_kv,D]
            full_v.unsqueeze(0),  # [1,N_gen,H_kv,D]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_k_offsets,
            max_seqlen_Q=packed_query_states["max_full_len"],
            max_seqlen_KV=packed_query_states["max_full_len"],
            return_lse=True,
        )  # full_sa: [1,N_gen,H,D], full_sa_lse: [1,N_gen,H]
    else:
        full_sa, full_sa_lse = multi_dimensional_attention_varlen(
            full_q.unsqueeze(0),  # [1,N_gen,H,D]
            full_k.unsqueeze(0),  # [1,N_gen,H_kv,D]
            full_v.unsqueeze(0),  # [1,N_gen,H_kv,D]
            metadata=natten_metadata,
            return_lse=True,
        )  # full_sa: [1,N_gen,H,D], full_sa_lse: [1,N_gen,H]

    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_gen,H,D]
        causal_k_normalized.unsqueeze(0),  # [1,N_und,H_kv,D]  normed und K for gen→und CA
        causal_v.unsqueeze(0),  # [1,N_und,H_kv,D]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_normalized_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        return_lse=True,
    )  # full_ca: [1,N_gen,H,D], full_ca_lse: [1,N_gen,H]

    assert isinstance(full_sa, torch.Tensor)
    assert isinstance(full_sa_lse, torch.Tensor)
    assert isinstance(full_ca, torch.Tensor)
    assert isinstance(full_ca_lse, torch.Tensor)
    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions_ac_safe(
        outputs=[full_sa, full_ca],
        lse_tensors=[full_sa_lse, full_ca_lse],
    )  # [1,N_gen,H,D]

    full_out = full_res.squeeze(0).flatten(-2, -1)  # [N_gen,H*D]
    return from_mode_splits(causal_out, full_out, packed_query_states)


def dispatch_attention_no_memory_ac_safe(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> SequencePack:
    """Interactive no-memory dispatch with AC-safe three-way merging."""
    if isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        return three_way_attention_no_memory_ac_safe(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
            packed_key_states_normalized=packed_key_states_normalized,
        )
    if isinstance(attention_mask, SplitInfo):
        return two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            packed_key_states_normalized=packed_key_states_normalized,
        )
    output, _ = vfm_dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
        packed_key_states_normalized=packed_key_states_normalized,
    )
    return output


def _frame0_pad_forward(
    out_inner: torch.Tensor,
    lse_inner: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend a frame-0 pad along dim 1: zeros for out, ``-inf`` for lse.

    Shape-agnostic in the leading layout: works for both 5-D
    ``[B, T, S, H, D]`` (multi_dim layout, used by
    :func:`strictly_past_causal_attention`) and 4-D ``[B, S, H, D]``
    (attention layout, used by the unit tests).
    """
    out_pad_shape = list(out_inner.shape)
    out_pad_shape[1] = 1
    lse_pad_shape = list(lse_inner.shape)
    lse_pad_shape[1] = 1
    neg_inf = torch.finfo(lse_inner.dtype).min
    frame0_out = torch.zeros(out_pad_shape, device=out_inner.device, dtype=out_inner.dtype)
    frame0_lse = torch.full(lse_pad_shape, neg_inf, device=lse_inner.device, dtype=lse_inner.dtype)
    return (
        torch.cat([frame0_out, out_inner], dim=1),
        torch.cat([frame0_lse, lse_inner], dim=1),
    )


def _frame0_pad_inverse(
    out_full: torch.Tensor,
    lse_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_frame0_pad_forward`: drop the frame-0 row along dim 1."""
    return out_full[:, 1:], lse_full[:, 1:]


def strictly_past_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    frames_per_chunk: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strictly-past causal cross-attention along the chunk dimension.

    Frames are grouped into contiguous chunks of ``frames_per_chunk`` (``C``)
    frames.  For each chunk ``c`` in ``[0, T // C)``, attends every query token
    in chunk ``c`` over the *past* chunks' K/V — i.e. all tokens in chunks
    ``0..c-1`` (fully visible, no causality within those chunks).  Chunk ``0``
    has no past, so its output is zero and its LSE is ``-inf`` (so an
    LSE-rescaled merge gives this component zero weight at chunk 0).

    ``frames_per_chunk == 1`` (the default) recovers the original framewise
    behavior exactly: frame ``t`` attends to clean frames ``0..t-1``.

    Uses ``multi_dimensional_attention`` with a chunk-dim causal mask to
    batch the ``n-1`` past-only attentions into one kernel call, then
    routes the (cat-padded) output through :class:`MergeAttentionsBridge`
    to preserve ``merge_attentions``' data-pointer contract on the
    backward pass (see :class:`MergeAttentionsBridge`).

    Args:
        q: ``[B, T, S, num_heads, head_dim]`` query tensor.
        k: ``[B, T, S, num_kv_heads, head_dim]`` key tensor.
        v: ``[B, T, S, num_kv_heads, head_dim]`` value tensor.
        frames_per_chunk: Number of consecutive frames per causal chunk.
            ``T`` must be divisible by it.

    Returns:
        ``(out, lse)`` with the same shape conventions as
        ``multi_dimensional_attention``:
          - ``out``: ``[B, T, S, num_heads, head_dim]``
          - ``lse``: ``[B, T, S, num_heads]``
    """
    B, T, S, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]
    C = frames_per_chunk
    if T % C != 0:
        raise ValueError(f"strictly_past_causal_attention requires T % frames_per_chunk == 0; got T={T}, C={C}.")
    n = T // C

    # Collapse each C-frame chunk into a single chunk-row of C*S tokens so the
    # causal mask operates over chunks, not frames.  Reshape is a no-op view
    # when C == 1.  q[:, 1:] etc. below are then strictly-past *chunks*.
    if C > 1:
        sc = C * S
        q = q.reshape(B, n, sc, num_heads, head_dim)
        k = k.reshape(B, n, sc, num_kv_heads, head_dim)
        v = v.reshape(B, n, sc, num_kv_heads, head_dim)
    else:
        sc = S

    if n > 1:
        out_inner, lse_inner = multi_dimensional_attention(
            q[:, 1:, :, :, :],  # [B, n-1, sc, H, D]
            k[:, : n - 1, :, :, :],  # [B, n-1, sc, H_kv, D]
            v[:, : n - 1, :, :, :],
            is_causal=(True, False),
            return_lse=True,
            backend="natten",  # Sanity check; this should be the only option.
        )
        out, lse = MergeAttentionsBridge.apply(
            out_inner,
            lse_inner,
            _frame0_pad_forward,
            _frame0_pad_inverse,
        )
    else:
        # n == 1: no past anywhere.  The whole component is degenerate, so
        # return zeros + -inf LSE.  No attention kernel call, so no merge
        # contract to preserve.  multi_dim's LSE is fp32 by convention,
        # so we use fp32 here too for dtype consistency with the n > 1 path.
        lse_dtype = torch.float32
        out = torch.zeros(B, 1, sc, num_heads, head_dim, device=q.device, dtype=q.dtype)
        lse = torch.full((B, 1, sc, num_heads), torch.finfo(lse_dtype).min, device=q.device, dtype=lse_dtype)

    # Expand chunk-rows back to frame granularity.  This is a contiguous view
    # of the (contiguous) bridge output, so it preserves the data-pointer link
    # the outer merge / bridge relies on.
    if C > 1:
        out = out.reshape(B, T, S, num_heads, head_dim)
        lse = lse.reshape(B, T, S, num_heads)

    return out, lse


def teacher_forcing_gen_attention(
    gen_q_2d: torch.Tensor,  # [1,T,S_super,H,D]
    gen_k_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    gen_v_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    memory_value: TFNoisyMemoryValue,
    frames_per_chunk: int = 1,
    cached_clean_gen_k: torch.Tensor | None = None,  # [1,T*S_super,H_kv,D]
    cached_clean_gen_v: torch.Tensor | None = None,  # [1,T*S_super,H_kv,D]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Teacher-forcing self/cross attention over the gen tokens.

    Replaces the standard temporal-causal gen self-attention with a set of
    LSE-merge components that together cover the teacher-forcing receptive
    field.  Frames are grouped into the chunk partition ``[1, C, C, ...]``
    (``C = frames_per_chunk``): latent frame 0 is always its own singleton chunk
    (mirroring the VAE's special first-frame encoding so the I2V conditioning
    frame stays pure clean causal context).  ``(T - 1) % C == 0`` is required
    when ``C > 1``.

    Receptive field for a query frame ``t`` in chunk ``c``:
      (a) all noisy tokens in chunk ``c`` (bidirectional, intra-chunk), and
      (b) all clean tokens in strictly-earlier chunks ``0..c-1``.

    ``frames_per_chunk == 1`` (framewise, the default) is the uniform-partition
    special case: every chunk is a single frame, so frame 0 needs no special
    handling and the field collapses to **two** components (per-frame noisy
    self-attention + strictly-past clean cross-attention).  ``C > 1`` needs
    **four** components because the ragged first chunk (1 frame vs ``C``) cannot
    be batched with the body by the uniform chunk-reshape kernels, so frame 0 is
    peeled out of both the intra-chunk SA and the strictly-past clean CA.

    Args:
        gen_q_2d: ``[1, T, S_super, num_heads, head_dim]`` query tensor.
        gen_k_2d: ``[1, T, S_super, num_kv_heads, head_dim]`` key tensor.
        gen_v_2d: ``[1, T, S_super, num_kv_heads, head_dim]`` value tensor.
        memory_value: TF memory carrying the cached clean gen K/V buffers.
        frames_per_chunk: Latent frames per causal chunk (``1`` == framewise).
        cached_clean_gen_k: Optional clean K override for a subrange such as the
            target item in transfer teacher forcing.
        cached_clean_gen_v: Optional clean V override matching ``cached_clean_gen_k``.

    Returns:
        A list of ``(out, lse)`` components for the caller's LSE merge, each with
        the shape conventions of ``multi_dimensional_attention``
        (``[1, T, S_super, num_heads, head_dim]`` / ``[1, T, S_super, num_heads]``)
        and ``-inf`` LSE on the frames a component does not cover.  Two
        components when ``C == 1``, four when ``C > 1``.

    Every cat / pad that sits between an attention kernel and the downstream
    merge is wrapped in :class:`MergeAttentionsBridge` so the merge's
    data-pointer backward contract is preserved (see that class).
    """
    clean_k = memory_value.cached_clean_gen_k if cached_clean_gen_k is None else cached_clean_gen_k  # [1,T*S,H_kv,D]
    clean_v = memory_value.cached_clean_gen_v if cached_clean_gen_v is None else cached_clean_gen_v  # [1,T*S,H_kv,D]
    if frames_per_chunk > 1:
        return _tf_gen_attention_chunkwise(gen_q_2d, gen_k_2d, gen_v_2d, clean_k, clean_v, frames_per_chunk)
    return _tf_gen_attention_framewise(gen_q_2d, gen_k_2d, gen_v_2d, clean_k, clean_v)


def _tf_gen_attention_framewise(
    gen_q_2d: torch.Tensor,  # [1,T,S_super,H,D]
    gen_k_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    gen_v_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    cached_clean_gen_k: torch.Tensor,  # [1,T*S_super,H_kv,D]
    cached_clean_gen_v: torch.Tensor,  # [1,T*S_super,H_kv,D]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Framewise (``C == 1``) teacher forcing: two merge components.

    (a) Spatial-only self-attention within each frame (T folded into the batch
        dim) — uses the current (noisy) Q/K/V.
    (b) Strictly-past causal cross-attention from the current (noisy) gen
        queries to the cached *clean* gen K/V from previous frames.

    See :func:`teacher_forcing_gen_attention` for the shared contract.
    """
    _, T, S_super, num_heads, head_dim = gen_q_2d.shape
    num_kv_heads = gen_k_2d.shape[-2]

    # --- (a) Spatial-only self-attention (T folded into batch dim) ---
    # Naming: ``_sa`` = self-attention, ``_ca`` = cross-attention, ``_lse`` = log-sum-exp.
    spatial_sa, spatial_sa_lse = attention(
        gen_q_2d.reshape(T, S_super, num_heads, head_dim),
        gen_k_2d.reshape(T, S_super, num_kv_heads, head_dim),
        gen_v_2d.reshape(T, S_super, num_kv_heads, head_dim),
        is_causal=False,
        return_lse=True,
        backend="natten",  # This is the only place where a non-natten backend might be selected.
    )
    # Use view instead of reshape; data cannot be copied before merge_attentions.
    spatial_sa = spatial_sa.view(1, T, S_super, num_heads, head_dim)
    spatial_sa_lse = spatial_sa_lse.view(1, T, S_super, num_heads)

    # --- (b) Strictly-past causal cross-attention to current-segment clean KV ---
    clean_k_2d = cached_clean_gen_k.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]
    clean_v_2d = cached_clean_gen_v.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]
    clean_ca, clean_ca_lse = strictly_past_causal_attention(  # [1,T,S_super,H,D], [1,T,S_super,H]
        gen_q_2d, clean_k_2d, clean_v_2d
    )

    return [(spatial_sa, spatial_sa_lse), (clean_ca, clean_ca_lse)]


def _tf_gen_attention_chunkwise(
    gen_q_2d: torch.Tensor,  # [1,T,S_super,H,D]
    gen_k_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    gen_v_2d: torch.Tensor,  # [1,T,S_super,H_kv,D]
    cached_clean_gen_k: torch.Tensor,  # [1,T*S_super,H_kv,D]
    cached_clean_gen_v: torch.Tensor,  # [1,T*S_super,H_kv,D]
    frames_per_chunk: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Chunkwise (``C > 1``) teacher forcing: four merge components.

    The chunk partition is ``[1, C, C, ...]`` (frame 0 is its own singleton
    chunk).  The four components together cover, per query frame ``t`` in chunk
    ``c``, all noisy tokens in chunk ``c`` (bidirectional) plus all clean tokens
    in strictly-earlier chunks ``0..c-1``:
      1. frame-0 intra-frame spatial self-attention   (covers frame 0)
      2. body intra-chunk spatio-temporal SA           (covers frames 1..T-1)
      3. body -> clean frame 0 cross-attention         (covers frames 1..T-1)
      4. body -> clean strictly-past body chunks CA     (covers frames 1..T-1)

    See :func:`teacher_forcing_gen_attention` for the shared contract.
    """
    _, T, S_super, num_heads, head_dim = gen_q_2d.shape
    num_kv_heads = gen_k_2d.shape[-2]
    C = frames_per_chunk
    if (T - 1) % C != 0:
        raise ValueError(f"Chunkwise teacher forcing requires (T - 1) % frames_per_chunk == 0; got T={T}, C={C}.")
    n_body = (T - 1) // C
    if n_body < 1:
        # Frame 0 (singleton chunk) plus at least one C-frame body chunk are
        # required, i.e. T >= 1 + C.  Shorter clips would have been truncated to
        # T == 1 upstream; filter them out at the data level (min latent frames
        # >= 1 + frames_per_chunk) before chunkwise teacher forcing.
        raise ValueError(
            f"Chunkwise teacher forcing needs T >= 1 + frames_per_chunk (one singleton frame plus at "
            f"least one full chunk); got T={T}, C={C}."
        )
    body_super = C * S_super
    neg_inf = torch.finfo(torch.float32).min

    clean_k_2d = cached_clean_gen_k.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]
    clean_v_2d = cached_clean_gen_v.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]

    # ---- (1) frame-0 intra-frame spatial self-attention (the singleton chunk) ----
    sa0, sa0_lse = attention(
        gen_q_2d[:, 0:1].reshape(1, S_super, num_heads, head_dim),
        gen_k_2d[:, 0:1].reshape(1, S_super, num_kv_heads, head_dim),
        gen_v_2d[:, 0:1].reshape(1, S_super, num_kv_heads, head_dim),
        is_causal=False,
        return_lse=True,
    )
    # ``attention`` returns LSE as [B,S,H,1]; drop the trailing singleton (a
    # storage-preserving view) so the bridge's backward copy_ into the kernel's
    # saved LSE shape-matches the [B,*,H] tensors the pad/inverse produce.
    sa0_lse = sa0_lse.reshape(1, S_super, num_heads)

    def _only_frame0_forward(o: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Place the single computed frame at index 0; pad frames 1..T-1 with -inf.
        o = o.reshape(1, 1, S_super, num_heads, head_dim)
        lse = lse.reshape(1, 1, S_super, num_heads)
        pad_o = torch.zeros(1, T - 1, S_super, num_heads, head_dim, device=o.device, dtype=o.dtype)
        pad_lse = torch.full((1, T - 1, S_super, num_heads), neg_inf, device=lse.device, dtype=lse.dtype)
        return torch.cat([o, pad_o], dim=1), torch.cat([lse, pad_lse], dim=1)

    def _only_frame0_inverse(o: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return o[:, :1].reshape(1, S_super, num_heads, head_dim), lse[:, :1].reshape(1, S_super, num_heads)

    sa0_full, sa0_full_lse = MergeAttentionsBridge.apply(sa0, sa0_lse, _only_frame0_forward, _only_frame0_inverse)

    # ---- (2) body intra-chunk spatio-temporal self-attention (bidirectional within chunk) ----
    # Frames 1..T-1 are grouped into n_body chunks of C frames; folding the chunk
    # index into the batch dim makes each chunk attend over its own C*S_super tokens.
    sab, sab_lse = attention(
        gen_q_2d[:, 1:].reshape(n_body, body_super, num_heads, head_dim),
        gen_k_2d[:, 1:].reshape(n_body, body_super, num_kv_heads, head_dim),
        gen_v_2d[:, 1:].reshape(n_body, body_super, num_kv_heads, head_dim),
        is_causal=False,
        return_lse=True,
    )
    sab_lse = sab_lse.reshape(n_body, body_super, num_heads)

    def _body_forward(o: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = o.reshape(1, T - 1, S_super, num_heads, head_dim)
        lse = lse.reshape(1, T - 1, S_super, num_heads)
        return _frame0_pad_forward(o, lse)  # prepend frame 0 (-inf) -> [1, T, ...]

    def _body_inverse(o: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o, lse = _frame0_pad_inverse(o, lse)  # drop frame 0 -> [1, T-1, ...]
        return (
            o.reshape(n_body, body_super, num_heads, head_dim),
            lse.reshape(n_body, body_super, num_heads),
        )

    sab_full, sab_full_lse = MergeAttentionsBridge.apply(sab, sab_lse, _body_forward, _body_inverse)

    strictly_past_components = _tf_strictly_past_attention_components(
        gen_q_2d,
        clean_k_2d,
        clean_v_2d,
        frames_per_chunk=C,
    )

    return [(sa0_full, sa0_full_lse), (sab_full, sab_full_lse), *strictly_past_components]


def _tf_strictly_past_attention_components(
    query: torch.Tensor,  # [B,T,S,H,D]
    key: torch.Tensor,  # [B,T,S,H_kv,D]
    value: torch.Tensor,  # [B,T,S,H_kv,D]
    *,
    frames_per_chunk: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Attend to K/V from strictly earlier ``[1, C, C, ...]`` chunks.

    This is the past-only half of chunkwise teacher forcing. It is shared by
    target self-attention and history-aware control attention so neither path
    can consume clean RGB from its current jointly denoised chunk.
    """
    batch_size, T, S_super, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[-2]
    C = frames_per_chunk
    if C < 1:
        raise ValueError(f"frames_per_chunk must be >= 1, got {C}.")
    if C == 1:
        past_out, past_lse = strictly_past_causal_attention(  # [B,T,S,H,D], [B,T,S,H]
            query,
            key,
            value,
        )
        return [(past_out, past_lse)]
    if (T - 1) % C != 0:
        raise ValueError(f"Chunkwise teacher forcing requires (T - 1) % frames_per_chunk == 0; got T={T}, C={C}.")
    n_body = (T - 1) // C
    if n_body < 1:
        raise ValueError(
            f"Chunkwise teacher forcing needs T >= 1 + frames_per_chunk (one singleton frame plus at "
            f"least one full chunk); got T={T}, C={C}."
        )

    # Body queries always see the clean singleton frame 0.
    body_tokens = (T - 1) * S_super
    frame0_ca, frame0_ca_lse = attention(  # [B,(T-1)*S,H,D], [B,(T-1)*S,H,1]
        query[:, 1:].reshape(batch_size, body_tokens, num_heads, head_dim),  # [B,(T-1)*S,H,D]
        key[:, 0:1].reshape(batch_size, S_super, num_kv_heads, head_dim),  # [B,S,H_kv,D]
        value[:, 0:1].reshape(batch_size, S_super, num_kv_heads, head_dim),  # [B,S,H_kv,D]
        is_causal=False,
        return_lse=True,
    )
    frame0_ca_lse = frame0_ca_lse.reshape(batch_size, body_tokens, num_heads)  # [B,(T-1)*S,H]

    def _frame0_context_forward(
        out: torch.Tensor,  # [B,(T-1)*S,H,D]
        lse: torch.Tensor,  # [B,(T-1)*S,H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = out.reshape(batch_size, T - 1, S_super, num_heads, head_dim)  # [B,T-1,S,H,D]
        lse = lse.reshape(batch_size, T - 1, S_super, num_heads)  # [B,T-1,S,H]
        return _frame0_pad_forward(out, lse)  # [B,T,S,H,D], [B,T,S,H]

    def _frame0_context_inverse(
        out: torch.Tensor,  # [B,T,S,H,D]
        lse: torch.Tensor,  # [B,T,S,H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = _frame0_pad_inverse(out, lse)  # [B,T-1,S,H,D], [B,T-1,S,H]
        return (
            out.reshape(batch_size, body_tokens, num_heads, head_dim),  # [B,(T-1)*S,H,D]
            lse.reshape(batch_size, body_tokens, num_heads),  # [B,(T-1)*S,H]
        )

    frame0_full, frame0_full_lse = MergeAttentionsBridge.apply(  # [B,T,S,H,D], [B,T,S,H]
        frame0_ca,
        frame0_ca_lse,
        _frame0_context_forward,
        _frame0_context_inverse,
    )

    # Later body chunks also see all strictly earlier clean body chunks.
    body_past, body_past_lse = strictly_past_causal_attention(  # [B,T-1,S,H,D], [B,T-1,S,H]
        query[:, 1:],
        key[:, 1:],
        value[:, 1:],
        frames_per_chunk=C,
    )
    body_past_full, body_past_full_lse = MergeAttentionsBridge.apply(  # [B,T,S,H,D], [B,T,S,H]
        body_past,
        body_past_lse,
        _frame0_pad_forward,
        _frame0_pad_inverse,
    )
    return [(frame0_full, frame0_full_lse), (body_past_full, body_past_full_lse)]


def _pad_transfer_item_component(
    out: torch.Tensor,  # [1,item_len,H,D] or a reshape-compatible layout
    lse: torch.Tensor,  # [1,item_len,H] or a reshape-compatible layout
    *,
    item_idx: int,
    item_len: int,
    total_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place one item's component in a two-item ``[control, target]`` stream.

    Returns tensors shaped ``[1,total_len,H,D]`` and ``[1,total_len,H]``.
    """
    if item_idx not in (0, 1):
        raise ValueError(f"Transfer item index must be 0 or 1, got {item_idx}.")
    if total_len != 2 * item_len:
        raise ValueError(f"Transfer stream must contain two equal items; got item_len={item_len}, total={total_len}.")
    start = item_idx * item_len
    end = start + item_len

    num_heads = out.shape[-2]
    head_dim = out.shape[-1]
    out = out.reshape(1, item_len, num_heads, head_dim)  # [1,item_len,H,D]
    lse = lse.reshape(1, item_len, num_heads)  # [1,item_len,H]
    neg_inf = torch.finfo(lse.dtype).min

    def _forward(
        item_out: torch.Tensor,  # [1,item_len,H,D]
        item_lse: torch.Tensor,  # [1,item_len,H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad one item to ``[1,total_len,H,D]`` and ``[1,total_len,H]``."""
        prefix_out = torch.zeros(  # [1,start,H,D]
            1, start, num_heads, head_dim, device=item_out.device, dtype=item_out.dtype
        )
        suffix_out = torch.zeros(  # [1,total_len-end,H,D]
            1, total_len - end, num_heads, head_dim, device=item_out.device, dtype=item_out.dtype
        )
        prefix_lse = torch.full(  # [1,start,H]
            (1, start, num_heads), neg_inf, device=item_lse.device, dtype=item_lse.dtype
        )
        suffix_lse = torch.full(  # [1,total_len-end,H]
            (1, total_len - end, num_heads), neg_inf, device=item_lse.device, dtype=item_lse.dtype
        )
        return (
            torch.cat([prefix_out, item_out, suffix_out], dim=1),  # [1,total_len,H,D]
            torch.cat([prefix_lse, item_lse, suffix_lse], dim=1),  # [1,total_len,H]
        )

    def _inverse(
        full_out: torch.Tensor,  # [1,total_len,H,D]
        full_lse: torch.Tensor,  # [1,total_len,H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover ``[1,item_len,H,D]`` and ``[1,item_len,H]``."""
        return full_out[:, start:end], full_lse[:, start:end]

    return MergeAttentionsBridge.apply(out, lse, _forward, _inverse)


def _inclusive_chunk_causal_attention_components(
    query: torch.Tensor,  # [1,T,S,H,D]
    key: torch.Tensor,  # [1,T,S,H_kv,D]
    value: torch.Tensor,  # [1,T,S,H_kv,D]
    frames_per_chunk: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Attend each query chunk to aligned K/V chunks up to and including itself."""
    _, num_frames, spatial_tokens, _num_heads, _head_dim = query.shape
    num_kv_heads = key.shape[-2]
    head_dim = key.shape[-1]
    flat_key = key.reshape(1, num_frames * spatial_tokens, num_kv_heads, head_dim)  # [1,T*S,H_kv,D]
    flat_value = value.reshape(1, num_frames * spatial_tokens, num_kv_heads, head_dim)  # [1,T*S,H_kv,D]
    if frames_per_chunk > 1:
        return _tf_gen_attention_chunkwise(query, key, value, flat_key, flat_value, frames_per_chunk)
    return _tf_gen_attention_framewise(query, key, value, flat_key, flat_value)


def teacher_forcing_transfer_attention(
    control_q: torch.Tensor,  # [T*S,H,D]
    control_k: torch.Tensor,  # [T*S,H_kv,D]
    control_v: torch.Tensor,  # [T*S,H_kv,D]
    target_q: torch.Tensor,  # [T*S,H,D]
    target_k: torch.Tensor,  # [T*S,H_kv,D]
    target_v: torch.Tensor,  # [T*S,H_kv,D]
    memory_value: TFReplayCleanMemoryValue | TFNoisyMemoryValue,
    *,
    target_shape: tuple[int, int, int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build control-conditioned temporal-causal TF components for aligned transfer items.

    Each returned pair is shaped ``[1,2*T*S,H,D]`` and ``[1,2*T*S,H]``.
    ``causal_control_with_rgb_history`` lets control queries additionally read
    clean RGB from strictly earlier teacher-forcing chunks. Target queries see
    target history through ordinary replay teacher forcing and see control only
    through their current ``[1, C, C, ...]`` chunk in either causal mode.
    """
    num_frames, height, width = target_shape
    spatial_tokens = height * width
    item_len = num_frames * spatial_tokens
    total_len = 2 * item_len
    num_heads = target_q.shape[-2]
    num_kv_heads = target_k.shape[-2]
    head_dim = target_q.shape[-1]

    control_q_2d = control_q.reshape(1, num_frames, spatial_tokens, num_heads, head_dim)  # [1,T,S,H,D]
    control_k_2d = control_k.reshape(  # [1,T,S,H_kv,D]
        1, num_frames, spatial_tokens, num_kv_heads, head_dim
    )
    control_v_2d = control_v.reshape(  # [1,T,S,H_kv,D]
        1, num_frames, spatial_tokens, num_kv_heads, head_dim
    )
    control_attention_mode = memory_value.transfer_control_attention_mode
    if control_attention_mode == "global_control":
        control_sa, control_sa_lse = attention(  # [1,T*S,H,D], [1,T*S,H,1]
            control_q.unsqueeze(0),  # [1,T*S,H,D]
            control_k.unsqueeze(0),  # [1,T*S,H_kv,D]
            control_v.unsqueeze(0),  # [1,T*S,H_kv,D]
            is_causal=False,
            return_lse=True,
        )
        control_components = [(control_sa, control_sa_lse)]
    elif control_attention_mode in ("causal_control", "causal_control_with_rgb_history"):
        control_components = _inclusive_chunk_causal_attention_components(
            control_q_2d,
            control_k_2d,
            control_v_2d,
            memory_value.frames_per_chunk,
        )
    else:
        raise ValueError(f"Unknown transfer control attention mode: {control_attention_mode!r}")

    components: list[tuple[torch.Tensor, torch.Tensor]] = []
    for control_out, control_lse in control_components:
        components.append(
            _pad_transfer_item_component(
                control_out,
                control_lse,
                item_idx=0,
                item_len=item_len,
                total_len=total_len,
            )
        )

    target_q_2d = target_q.reshape(1, num_frames, spatial_tokens, num_heads, head_dim)  # [1,T,S,H,D]
    target_k_2d = target_k.reshape(  # [1,T,S,H_kv,D]
        1, num_frames, spatial_tokens, num_kv_heads, head_dim
    )
    target_v_2d = target_v.reshape(  # [1,T,S,H_kv,D]
        1, num_frames, spatial_tokens, num_kv_heads, head_dim
    )
    if isinstance(memory_value, TFNoisyMemoryValue):
        clean_target_k = memory_value.cached_clean_gen_k[:, item_len:total_len]  # [1,T*S,H_kv,D]
        clean_target_v = memory_value.cached_clean_gen_v[:, item_len:total_len]  # [1,T*S,H_kv,D]
    else:
        clean_target_k = target_k.unsqueeze(0)  # [1,T*S,H_kv,D]
        clean_target_v = target_v.unsqueeze(0)  # [1,T*S,H_kv,D]

    if control_attention_mode == "causal_control_with_rgb_history":
        clean_target_k_2d = clean_target_k.reshape(  # [1,T,S,H_kv,D]
            1, num_frames, spatial_tokens, num_kv_heads, head_dim
        )
        clean_target_v_2d = clean_target_v.reshape(  # [1,T,S,H_kv,D]
            1, num_frames, spatial_tokens, num_kv_heads, head_dim
        )
        control_rgb_history_components = _tf_strictly_past_attention_components(
            control_q_2d,
            clean_target_k_2d,
            clean_target_v_2d,
            frames_per_chunk=memory_value.frames_per_chunk,
        )
        for control_rgb_history_out, control_rgb_history_lse in control_rgb_history_components:
            components.append(
                _pad_transfer_item_component(
                    control_rgb_history_out,
                    control_rgb_history_lse,
                    item_idx=0,
                    item_len=item_len,
                    total_len=total_len,
                )
            )

    if isinstance(memory_value, TFNoisyMemoryValue):
        target_components = teacher_forcing_gen_attention(
            target_q_2d,
            target_k_2d,
            target_v_2d,
            memory_value,
            memory_value.frames_per_chunk,
            cached_clean_gen_k=clean_target_k,
            cached_clean_gen_v=clean_target_v,
        )
    else:
        target_sa, target_sa_lse = multi_dimensional_attention(  # [1,T,S,H,D], [1,T,S,H]
            target_q_2d,
            target_k_2d,
            target_v_2d,
            is_causal=(True, False),
            return_lse=True,
            backend="natten",
        )
        target_components = [(target_sa, target_sa_lse)]

    for target_out, target_lse in target_components:
        components.append(
            _pad_transfer_item_component(
                target_out,
                target_lse,
                item_idx=1,
                item_len=item_len,
                total_len=total_len,
            )
        )

    # Use Pass-1 clean control K/V during the noisy pass so target representations
    # cannot acquire a target -> control -> earlier-target leakage path across layers.
    if isinstance(memory_value, TFNoisyMemoryValue):
        target_control_k = memory_value.cached_clean_gen_k[:, :item_len]  # [1,T*S,H_kv,D]
        target_control_v = memory_value.cached_clean_gen_v[:, :item_len]  # [1,T*S,H_kv,D]
    else:
        target_control_k = control_k.unsqueeze(0)  # [1,T*S,H_kv,D]
        target_control_v = control_v.unsqueeze(0)  # [1,T*S,H_kv,D]
    if control_attention_mode == "global_control":
        target_control, target_control_lse = attention(  # [1,T*S,H,D], [1,T*S,H,1]
            target_q.unsqueeze(0),  # [1,T*S,H,D]
            target_control_k,
            target_control_v,
            is_causal=False,
            return_lse=True,
        )
        target_control_components = [(target_control, target_control_lse)]
    elif control_attention_mode in ("causal_control", "causal_control_with_rgb_history"):
        target_control_k_2d = target_control_k.reshape(  # [1,T,S,H_kv,D]
            1, num_frames, spatial_tokens, num_kv_heads, head_dim
        )
        target_control_v_2d = target_control_v.reshape(  # [1,T,S,H_kv,D]
            1, num_frames, spatial_tokens, num_kv_heads, head_dim
        )
        target_control_components = _inclusive_chunk_causal_attention_components(
            target_q_2d,
            target_control_k_2d,
            target_control_v_2d,
            memory_value.frames_per_chunk,
        )
    else:
        raise ValueError(f"Unknown transfer control attention mode: {control_attention_mode!r}")

    for target_control_out, target_control_lse in target_control_components:
        components.append(
            _pad_transfer_item_component(
                target_control_out,
                target_control_lse,
                item_idx=1,
                item_len=item_len,
                total_len=total_len,
            )
        )
    return components


def three_way_attention_with_kv_cache(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: KVTrainMemoryValue,
    attention_meta: SplitInfo | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> SequencePack:
    """Branchless three-way attention for KV cache training.

    Originally introduced for KV-cache training; also used by compile-safe
    AR inference under ``torch.compile`` (which constructs a
    ``KVTrainMemoryValue`` directly).

    Always processes tokens for both text and video.  When no caption is
    present, the text tokens will consist only of padding.

    If a new caption has been provided at the start of the current segment
    (usually when starting a new video), the new caption will be part of the
    input text tokens, otherwise the previous cached caption will be used.
    To prevent recompilation with torch.compile, torch.where will select
    between incoming and cached text segments.

    Args:
        packed_query_states: Factored Q pack (always contains both text and
            video tokens; the upstream "causal"/"full_only" sub-packs map to
            text/video respectively).
        packed_key_states: Factored K pack.
        packed_value_states: Factored V pack.
        memory_value: Read-only tensor container with cached K/V tensors,
            boolean flags, and varlen offsets.
        attention_meta: SplitInfo with vision/action token shape metadata.
    """
    num_action_tokens = memory_value.num_action_tokens_per_supertoken
    has_new_caption = memory_value.has_new_caption
    has_caption = memory_value.has_caption
    text_kv_offsets = memory_value.und_kv_offsets
    video_q_offsets = memory_value.gen_q_offsets
    has_cached_video = memory_value.has_cached_gen

    # Whether K/V buffers must be clamped to length >= 1.
    # Set in KVCacheTrainMemoryState; defaults to True there.
    clamp_empty_varlen_kv = memory_value.clamp_empty_varlen_kv

    # ``get_full_only_seq`` returns the video token slice of the pack.
    # "video" in this case refers to non-text; e.g. video/action/sound.
    video_q, video_pack_q_offsets = get_full_only_seq(packed_query_states)
    video_k, _video_pack_k_offsets = get_full_only_seq(packed_key_states)
    video_v, _ = get_full_only_seq(packed_value_states)

    if attention_meta is not None and attention_meta.null_action_supertokens:
        video_v = video_v.clone()
        starts = video_pack_q_offsets[:-1].long()
        null_positions = (starts.unsqueeze(1) + torch.arange(num_action_tokens, device=starts.device)).reshape(-1)
        video_v[null_positions] = 0

    # --- video self-attention: temporal-causal via multi_dimensional_attention ---
    vision_token_shapes = memory_value.vision_token_shapes
    is_transfer = len(vision_token_shapes) == 2
    if is_transfer:
        if not isinstance(memory_value, (TFReplayCleanMemoryValue, TFNoisyMemoryValue)):
            raise TypeError("Two-item temporal-causal transfer is supported only by replay teacher forcing.")
        if num_action_tokens != 0:
            raise ValueError("Two-item temporal-causal transfer does not support action tokens.")
        if vision_token_shapes[0] != vision_token_shapes[1]:
            raise ValueError(
                f"Temporal-causal transfer requires aligned control and target shapes; got {vision_token_shapes}."
            )
    T, H_p, W_p = vision_token_shapes[0]
    S_super = num_action_tokens + H_p * W_p
    item_len = T * S_super
    video_len = item_len * (2 if is_transfer else 1)
    num_heads = video_q.shape[1]
    num_kv_heads = video_k.shape[1]
    head_dim = video_q.shape[2]

    # Strip padding introduced by sequence_pack_from_packed_sequence().
    # The video pack has been padded up to a rounded-up length for compile
    # stability.  Strip the padding here because multi_dimensional_attention
    # requires an exact (T, S_super) reshape.
    # The padding will be added back after merge_attentions.
    padded_video_len = video_q.shape[0]
    video_q = video_q[:video_len]
    video_k = video_k[:video_len]
    video_v = video_v[:video_len]

    # Naming: ``_sa`` = self-attention, ``_ca`` = cross-attention, ``_lse`` = log-sum-exp.
    attn_outputs: list[torch.Tensor]
    lse_outputs: list[torch.Tensor]
    if is_transfer:
        assert isinstance(memory_value, (TFReplayCleanMemoryValue, TFNoisyMemoryValue))
        transfer_components = teacher_forcing_transfer_attention(
            video_q[:item_len],  # [T*S,H,D]
            video_k[:item_len],  # [T*S,H_kv,D]
            video_v[:item_len],  # [T*S,H_kv,D]
            video_q[item_len:video_len],  # [T*S,H,D]
            video_k[item_len:video_len],  # [T*S,H_kv,D]
            video_v[item_len:video_len],  # [T*S,H_kv,D]
            memory_value,
            target_shape=vision_token_shapes[1],
        )
        attn_outputs = [out for out, _lse in transfer_components]  # each [1,2*T*S,H,D]
        lse_outputs = [lse for _out, lse in transfer_components]  # each [1,2*T*S,H]
    else:
        # Reshape to expose temporal dimension for the causal mask.
        video_q_2d = video_q.reshape(1, T, S_super, num_heads, head_dim)  # [1,T,S_super,H,D]
        video_k_2d = video_k.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]
        video_v_2d = video_v.reshape(1, T, S_super, num_kv_heads, head_dim)  # [1,T,S_super,H_kv,D]

        video_components: list[tuple[torch.Tensor, torch.Tensor]]
        if isinstance(memory_value, TFNoisyMemoryValue):
            # Teacher forcing: two merge components framewise, four chunkwise
            # (frames_per_chunk > 1, chunk partition [1, C, C, ...]).
            video_components = teacher_forcing_gen_attention(
                video_q_2d,
                video_k_2d,
                video_v_2d,
                memory_value,
                memory_value.frames_per_chunk,
            )
        else:
            # --- Standard temporal-causal self-attention ---
            video_sa, video_sa_lse = multi_dimensional_attention(  # [1,T,S_super,H,D], [1,T,S_super,H]
                video_q_2d,
                video_k_2d,
                video_v_2d,
                is_causal=(True, False),
                return_lse=True,
                backend="natten",  # Sanity check; this should be the only option.
            )
            video_components = [(video_sa, video_sa_lse)]

        # Flatten video frames into a sequence again.
        attn_outputs = [
            out.reshape(1, item_len, num_heads, head_dim) for out, _lse in video_components
        ]  # each [1,T*S_super,H,D]
        lse_outputs = [lse.reshape(1, item_len, num_heads) for _out, lse in video_components]  # each [1,T*S_super,H]

    # --- Shared tail: rolling-cache CA, text CA, merge ---
    video_q_flat = video_q.unsqueeze(0)  # [1, S_video, H, D]

    # -- Video cross-attention to KV-cache.  (Compile-stable path). ---
    cached_video_k = memory_value.cached_gen_k
    cached_video_v = memory_value.cached_gen_v

    video_ca_cached, video_ca_cached_lse = dispatch_varlen_cross_attention(
        video_q_flat,
        cached_video_k,
        cached_video_v,
        cumulative_seqlen_Q=video_q_offsets,
        cumulative_seqlen_KV=memory_value.gen_ca_cached_kv_offsets,
        max_seqlen_Q=video_len,
        max_seqlen_KV=memory_value.max_gen_cache_tokens,
        has_real=has_cached_video,
        clamp_empty_varlen_kv=clamp_empty_varlen_kv,
    )
    attn_outputs.append(video_ca_cached)
    lse_outputs.append(video_ca_cached_lse)

    # --- Video cross-attention to text K/V ---
    # Naming: ``live_*`` = freshly projected from the current pack;
    #         ``cached_*`` = stored from a prior segment.
    # For the gen→und (video→text) cross-attention use the normed und K when provided
    # (Nemotron-3 qk_norm config), else fall back to the raw und K.  ``memory_value.cached_und_k``
    # is expected to already hold the normed K (mirrors unified_mot ``k_und_to_store``).
    if packed_key_states_normalized is not None:
        text_k, _ = get_causal_seq(packed_key_states_normalized)
    else:
        text_k, _ = get_causal_seq(packed_key_states)
    text_v, _ = get_causal_seq(packed_value_states)
    live_text_k = text_k.unsqueeze(0)  # [1, MAX_CAUSAL_LEN, H_kv, D]
    live_text_v = text_v.unsqueeze(0)

    # torch.where selects live vs cached text K/V without branching.
    selected_text_k = torch.where(has_new_caption, live_text_k, memory_value.cached_und_k)
    selected_text_v = torch.where(has_new_caption, live_text_v, memory_value.cached_und_v)
    max_text_len = selected_text_k.shape[1]

    video_ca_text, video_ca_text_lse = dispatch_varlen_cross_attention(
        video_q_flat,
        selected_text_k,
        selected_text_v,
        cumulative_seqlen_Q=video_q_offsets,
        cumulative_seqlen_KV=text_kv_offsets,
        max_seqlen_Q=video_len,
        max_seqlen_KV=max_text_len,
        has_real=has_caption,
        clamp_empty_varlen_kv=clamp_empty_varlen_kv,
    )
    attn_outputs.append(video_ca_text)
    lse_outputs.append(video_ca_text_lse)

    # --- Merge all video attention outputs ---
    # AC-safe variant: NATTEN's merge_attentions backward indexes
    # ctx.saved_tensors multiple times, which trips the non-reentrant
    # checkpoint unpack-once guard.  merge_attentions_ac_safe preserves
    # the same storage-patching contract with a single saved_tensors
    # access.  See _ACSafeMergeAttentionsFn for details.
    video_res, _ = merge_attentions_ac_safe(outputs=attn_outputs, lse_tensors=lse_outputs)
    video_out = video_res.squeeze(0).flatten(-2, -1)  # [video_len, C]

    # Repad to the rounded-up length that sequence_pack_from_packed_sequence produced,
    # so from_mode_splits can write video_out back into the padded pack structure.
    pad_len = padded_video_len - video_len
    if pad_len > 0:
        video_out = torch.nn.functional.pad(video_out, (0, 0, 0, pad_len))

    # --- text self-attention ---
    # Varlen self-attention over the live text in the current pack.  The
    # text Q/K/V tensors are padded to a constant length (compile-stable
    # shape); the varlen offsets restrict the kernel to the real
    # (non-padded) range.  Output positions beyond the real range are
    # discarded by ``from_mode_splits`` so they never reach the loss.
    #
    # Reuse ``text_kv_offsets``, which should be the same as text Q offsets, and
    # is already pre-clamped to >=1 by ``KVCacheTrainMemoryState.init()``
    # to avoid kernel failure in case length == 0.
    text_q, _ = get_causal_seq(packed_query_states)
    text_k, _ = get_causal_seq(packed_key_states)
    text_v, _ = get_causal_seq(packed_value_states)
    padded_text_len = text_q.shape[0]

    text_res = attention(
        text_q.unsqueeze(0),
        text_k.unsqueeze(0),
        text_v.unsqueeze(0),
        cumulative_seqlen_Q=text_kv_offsets,
        cumulative_seqlen_KV=text_kv_offsets,
        max_seqlen_Q=padded_text_len,
        max_seqlen_KV=padded_text_len,
        is_causal=True,
        causal_type=CausalType.TopLeft,
        backend="natten",  # Sanity check; this should be the only option.
    )
    assert isinstance(text_res, torch.Tensor)
    text_out = text_res.squeeze(0).flatten(-2, -1)
    return from_mode_splits(text_out, video_out, packed_query_states)


class _ACSafeMergeAttentionsFn(torch.autograd.Function):
    """AC-compatible drop-in for NATTEN's ``MergeAttentionsAutogradFn``.

    NATTEN's backward indexes ``ctx.saved_tensors`` in multiple slices
    (``[:2]``, ``[2 : N+2]``, ``[N+2:]``), and each indexing access fires
    the non-reentrant ``torch.utils.checkpoint`` unpack hook for *every*
    saved tensor. The hook only permits one unpack per saved tensor, so
    activation checkpointing + NATTEN merge_attentions raises
    ``CheckpointError: Unpack is being triggered for a tensor that was
    already unpacked once`` (see the replayed-LSE + AC=full long-video
    TF path).

    This version preserves the same forward math and the same
    storage-patching backward contract (see :class:`MergeAttentionsBridge`
    docstring for the full description), but reads ``ctx.saved_tensors``
    exactly once.  Numerics match ``naive_merge_attentions`` (iterative
    pairwise LSE rescale) — which is what NATTEN's kernel implements up
    to reduction order.
    """

    @staticmethod
    def forward(
        ctx,
        num_components: int,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = tensors[:num_components]
        lses = tensors[num_components:]
        output_dtype = outputs[0].dtype
        normalized_lses = [lse.squeeze(-1) if lse.ndim == 4 else lse for lse in lses]

        merged_lse = normalized_lses[0]
        merged_out = outputs[0]
        for i in range(1, num_components):
            new_lse = torch.logaddexp(merged_lse, normalized_lses[i])
            w_old = torch.exp(merged_lse - new_lse).unsqueeze(-1)
            w_new = torch.exp(normalized_lses[i] - new_lse).unsqueeze(-1)
            merged_out = w_old * merged_out + w_new * outputs[i]
            merged_lse = new_lse
        merged_out = merged_out.to(output_dtype)

        ctx.save_for_backward(merged_out, merged_lse, *outputs, *lses)
        ctx.num_components = num_components
        return merged_out, merged_lse

    @staticmethod
    def backward(
        ctx,
        grad_merged_out: torch.Tensor,
        grad_merged_lse: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        # Single access — avoid retriggering the AC unpack hook for any saved tensor.
        saved = ctx.saved_tensors
        merged_out = saved[0]
        merged_lse = saved[1]
        num = ctx.num_components
        outputs = saved[2 : 2 + num]
        lses = saved[2 + num : 2 + 2 * num]

        # Patch each component's storage with the merged O / LSE.  The
        # upstream attention kernel's backward will read these as its
        # saved O / LSE and compute gradients as if it had produced the
        # merged output.  The original LSE shape is preserved (the
        # forward squeezes a trailing singleton, so we re-broadcast).
        for o in outputs:
            o.data.copy_(merged_out.data)
        for l in lses:
            if l.ndim == merged_lse.ndim + 1 and l.shape[-1] == 1:
                l.data.copy_(merged_lse.data.unsqueeze(-1))
            else:
                l.data.copy_(merged_lse.data)

        # Same upstream-grad contract as NATTEN: dL/dO_i = dL/dO_merged
        # for every component; dL/dLSE_i is forwarded unchanged for
        # parity (i4 attention treats LSE as non-differentiable, so this
        # gradient is silently dropped at the kernel boundary).
        grads = (None,) + (grad_merged_out,) * num + (grad_merged_lse,) * num
        return grads


def merge_attentions_ac_safe(
    outputs: list[torch.Tensor],
    lse_tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """AC-safe drop-in for ``cosmos_framework.model.attention.merge_attentions``.

    Use at call sites that live inside an activation-checkpointed module
    boundary.  Matches NATTEN's storage-patching backward contract so
    upstream i4 attention kernels (whose LSE is not differentiable)
    still receive correct gradients via their own saved O / LSE
    backward formulas.
    """
    assert len(outputs) == len(lse_tensors) >= 2
    return _ACSafeMergeAttentionsFn.apply(len(outputs), *outputs, *lse_tensors)


def naive_merge_attentions(
    outputs: list[torch.Tensor],
    lse_tensors: list[torch.Tensor],
    torch_compile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge multiple attention outputs using LSE rescaling.

    NOTE: this function does NOT work properly with cosmos_framework attention,
    because the LSE outputs of cosmos_framework attention are not differentiable.
    It can be used with a simple pytorch-only implementation of attention, and
    is defined here for use with unit tests.

    The merge computes the combined output as if all K/V sets had been
    concatenated before attention::

        lse_combined  = logsumexp(lse_1, lse_2, …)
        out_combined  = Σ_i  exp(lse_i − lse_combined) · out_i

    Args:
        outputs: Attention output tensors ``[B, S, H, D]``.
        lse_tensors: Corresponding logsumexp tensors.  Accepted shapes:
            ``[B, S, H]`` or ``[B, S, H, 1]`` (trailing dim is squeezed).
        torch_compile: Unused; accepted for interface compatibility.

    Returns:
        (merged_output, merged_lse) with shapes ``[B, S, H, D]`` and
        ``[B, S, H]``.
    """
    assert len(outputs) == len(lse_tensors) >= 2
    output_dtype = outputs[0].dtype

    # Normalise LSE shapes to [B, S, H].
    lses = [lse.squeeze(-1) if lse.ndim == 4 else lse for lse in lse_tensors]

    # Iterative pairwise merge.  At each step:
    #   new_lse = logsumexp(merged_lse, lse_i)
    #   merged  = exp(merged_lse - new_lse) * merged + exp(lse_i - new_lse) * out_i
    # The weights exp(…) sum to 1, so the result is correctly normalised.
    merged_lse = lses[0]
    merged_out = outputs[0]

    for i in range(1, len(outputs)):
        lse_i = lses[i]
        out_i = outputs[i]

        new_lse = torch.logaddexp(merged_lse, lse_i)  # [B, S, H]
        # Unsqueeze to broadcast over the head dimension.
        w_old = torch.exp(merged_lse - new_lse).unsqueeze(-1)
        w_new = torch.exp(lse_i - new_lse).unsqueeze(-1)

        merged_out = w_old * merged_out + w_new * out_i  # [B, S, H, D]
        merged_lse = new_lse

    return merged_out.to(output_dtype), merged_lse


def attention_AR_gen_only(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    """Autoregressive frame 1+ temporal-causal attention.

    This version of attention is used for temporal-causal AR inference,
    after the initial frame 0.

    It attends bidirectionally over the current frame, and also over previous
    frames from the KV cache.  Unlike dispatch_attention, it assumes that the
    packed sequences only contain gen tokens -- the caption should have been
    previously stored in the KV cache.

    Three flavors selected by ``memory_value``:

    - **Dynamic-shape** (``gen_k_hist`` set, ``gen_k_buf_full`` ``None``):
      cat ``[und || gen_hist || curr]`` and call dense ``attention()``
      (no varlen kwargs).  Used by eager and compile-no-CG, where
      ``gen_hist`` grows per frame via ``gen_cache.fetch_kv``.
    - **Post-saturation static compile** (``post_saturation_static_compile``):
      und K/V is padded to ``S_und_max`` while the rolling gen history is
      already fixed-size.  The real und prefix is compacted with gen K/V so
      varlen attention ignores only the padded text suffix.
    - **Static-shape** (``gen_k_buf_full`` set, ``gen_k_hist`` ``None``):
      cat ``[und || curr || gen_buf]`` (real positions contiguous from
      offset 0; padding in the gen-buf tail) and call ``attention()``
      with the varlen kwargs ``cumulative_seqlen_Q`` /
      ``cumulative_seqlen_KV`` pre-built outside the captured region in
      ``ARMemoryState.init`` (see ``ARMemoryValue.cu_seqlens_q_t`` /
      ``cu_seqlens_kv_t``).  ``cumulative_seqlen_KV = [0,
      real_total_kv_len]`` restricts the kernel to the real (non-padding)
      K prefix while the tensor *shape* stays at the static max size,
      so a single CUDA-graph capture replays for every frame; only the
      second slot's *value* updates per frame.

    The call signature matches :func:`dispatch_attention_with_memory` so the
    two are interchangeable as the ``attention_function`` callback in
    :func:`context_parallel_attention`.

    When wrapped by :func:`context_parallel_attention`, this function
    receives **head-sharded** tensors: the packed Q/K/V have shape
    ``[S, H/cp, D]`` (full sequence, sharded heads) and the cached
    tensors in ``memory_value`` (``und_k_cached``, ``gen_k_hist``, etc.)
    are likewise ``[1, S, H/cp, D]``.  The implementation is
    dimension-agnostic — all tensors share the same head count — so it
    works identically for both the full-head and head-sharded cases.
    """
    assert isinstance(memory_value, ARMemoryValue)
    del attention_mask  # unused.
    del natten_metadata  # unused.
    del packed_key_states_normalized  # unused.

    q_gen = get_gen_seq(packed_query_states)  # [S_curr, H, D]
    k_gen = get_gen_seq(packed_key_states)  # [S_curr, H_kv, D]
    v_gen = get_gen_seq(packed_value_states)  # [S_curr, H_kv, D]

    gen_len = memory_value.gen_len
    k_gen_real = k_gen[:gen_len]  # [S_gen_real, H_kv, D]
    v_gen_real = v_gen[:gen_len]  # [S_gen_real, H_kv, D]

    k_curr = k_gen_real.unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    v_curr = v_gen_real.unsqueeze(0)  # [1, S_gen_real, H_kv, D]

    if memory_value.for_cuda_graphs:
        # Static-shape branch.  Real positions live in [0, S_und + gen_len +
        # real_gen_cache_len); the gen-buffer tail is zero-padding to a
        # fixed max size.  Putting the current frame *before* the gen
        # buffer keeps real positions contiguous from offset 0, so a
        # single ``cumulative_seqlen_KV = [0, real_total_kv_len]``
        # restricts the kernel to the real prefix without any padding
        # hole.  RoPE was applied to each K vector at projection time,
        # so order within the seq dim is irrelevant for correctness.
        assert memory_value.und_k_cached is not None and memory_value.und_v_cached is not None, (
            "static-shape branch requires the und cache to be populated"
        )
        assert memory_value.gen_k_buf_full is not None
        assert memory_value.gen_v_buf_full is not None
        assert memory_value.cu_seqlens_q_t is not None
        assert memory_value.cu_seqlens_kv_t is not None

        k_full = torch.cat([memory_value.und_k_cached, k_curr, memory_value.gen_k_buf_full], dim=1)
        v_full = torch.cat([memory_value.und_v_cached, v_curr, memory_value.gen_v_buf_full], dim=1)

        # ``cu_seqlens_q_t`` and ``cu_seqlens_kv_t`` are pre-built outside
        # the captured region (in ``ARMemoryState.init``) as ``[2]`` int32
        # tensors.
        attn_result = attention(
            query=q_gen.unsqueeze(0),  # [1, S_curr, H, D]
            key=k_full,  # [1, KV_LEN_MAX, H_kv, D]
            value=v_full,
            cumulative_seqlen_Q=memory_value.cu_seqlens_q_t,
            cumulative_seqlen_KV=memory_value.cu_seqlens_kv_t,
            max_seqlen_Q=gen_len,
            max_seqlen_KV=memory_value.max_seqlen_KV,
            is_causal=False,
            return_lse=False,
            backend="natten",
        )
        assert isinstance(attn_result, torch.Tensor)
        gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_curr, H*D]
    elif memory_value.post_saturation_static_compile:
        # Static compile without CUDA graphs: the generated-history window is
        # already saturated and fixed-size, but prompt text length can vary by
        # sample. ``ARMemoryState`` pads und K/V to ``S_und_max``; compact the
        # real text prefix together with gen history/current frame so varlen
        # attention can ignore the padded suffix.
        assert memory_value.und_k_cached is not None and memory_value.und_v_cached is not None, (
            "post-saturation static compile requires the und cache to be populated"
        )
        assert memory_value.real_und_cache_len_t is not None
        assert memory_value.cu_seqlens_q_t is not None
        assert memory_value.cu_seqlens_kv_t is not None

        if memory_value.gen_k_hist is not None:
            assert memory_value.gen_v_hist is not None
            nonund_k = torch.cat([memory_value.gen_k_hist, k_curr], dim=1)  # [1,S_nonund,H_kv,D]
            nonund_v = torch.cat([memory_value.gen_v_hist, v_curr], dim=1)  # [1,S_nonund,H_kv,D]
        else:
            nonund_k = k_curr  # [1,S_gen_real,H_kv,D]
            nonund_v = v_curr  # [1,S_gen_real,H_kv,D]

        k_full = _compact_padded_und_cache_for_varlen(
            memory_value.und_k_cached,
            nonund_k,
            memory_value.real_und_cache_len_t,
        )  # [1,KV_LEN_MAX,H_kv,D]
        v_full = _compact_padded_und_cache_for_varlen(
            memory_value.und_v_cached,
            nonund_v,
            memory_value.real_und_cache_len_t,
        )  # [1,KV_LEN_MAX,H_kv,D]

        attn_result = attention(
            query=q_gen.unsqueeze(0),  # [1,S_curr,H,D]
            key=k_full,
            value=v_full,
            cumulative_seqlen_Q=memory_value.cu_seqlens_q_t,
            cumulative_seqlen_KV=memory_value.cu_seqlens_kv_t,
            max_seqlen_Q=gen_len,
            max_seqlen_KV=memory_value.max_seqlen_KV,
            is_causal=False,
            return_lse=False,
            backend="natten",
        )  # [1,S_curr,H,D]
        assert isinstance(attn_result, torch.Tensor)
        gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_curr,H*D]
    else:
        # Dynamic-shape branch (eager / compile-no-CG / frame 0).  Same layout as
        # the original AR-gen-only path: cat([und, gen_hist, curr]).
        kv_parts_k = [k_curr]
        kv_parts_v = [v_curr]
        if memory_value.und_k_cached is not None:
            assert memory_value.und_v_cached is not None
            kv_parts_k.insert(0, memory_value.und_k_cached)
            kv_parts_v.insert(0, memory_value.und_v_cached)
        if memory_value.gen_k_hist is not None:
            assert memory_value.gen_v_hist is not None
            kv_parts_k.insert(-1, memory_value.gen_k_hist)
            kv_parts_v.insert(-1, memory_value.gen_v_hist)

        k_full = torch.cat(kv_parts_k, dim=1)  # [1, S_total, H_kv, D]
        v_full = torch.cat(kv_parts_v, dim=1)  # [1, S_total, H_kv, D]

        attn_result = attention(
            query=q_gen.unsqueeze(0),  # [1, S_curr, H, D]
            key=k_full,
            value=v_full,
            is_causal=False,
            return_lse=False,
            backend="natten",
        )
        assert isinstance(attn_result, torch.Tensor)
        gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_curr, H*D]

    output = from_und_gen_splits(
        gen_out.new_empty(0, gen_out.shape[-1]),
        gen_out,
        packed_query_states,
    )
    return output, None


def dispatch_attention_with_memory(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    """Dispatch that routes memory-augmented attention to the appropriate kernel.

    - ``KVTrainMemoryValue`` → ``three_way_attention_with_kv_cache``
      (rolling-KV *training* path)
    - ``ARMemoryValue`` with ``frame_idx > 0`` → ``attention_AR_gen_only``
      (AR inference frame 1+; covers eager, compile-no-CG, and compile+CG.
      The static-shape branch inside ``attention_AR_gen_only`` is selected
      by ``ARMemoryState(for_cuda_graphs=True)`` populating
      ``memory_value.gen_k_buf_full`` and friends.)
    - ``ARMemoryValue`` with ``frame_idx == 0`` → interactive no-memory dispatch
    - ``None`` → interactive no-memory dispatch
    """
    if isinstance(memory_value, KVTrainMemoryValue):
        attention_meta = attention_mask if isinstance(attention_mask, SplitInfo) else None
        output = three_way_attention_with_kv_cache(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            memory_value=memory_value,
            attention_meta=attention_meta,
            packed_key_states_normalized=packed_key_states_normalized,
        )
        return output, None
    # Keep the post-saturation static-compile predicate first so Dynamo can
    # short-circuit on the stable flag without guarding on the changing
    # Python frame_idx value.
    if isinstance(memory_value, ARMemoryValue) and (
        memory_value.post_saturation_static_compile or memory_value.frame_idx > 0
    ):
        return attention_AR_gen_only(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=packed_key_states_normalized,
        )
    if isinstance(attention_mask, SplitInfo) and attention_mask.is_three_way:
        output = dispatch_attention_no_memory_ac_safe(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            natten_metadata=natten_metadata,
            packed_key_states_normalized=packed_key_states_normalized,
        )
        return output, None
    # Non-three-way paths do not merge NATTEN attention outputs, so keep the shared implementation.
    return vfm_dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
        packed_key_states_normalized=packed_key_states_normalized,
    )
