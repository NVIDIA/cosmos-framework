# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Equivalence tests for ``three_way_attention_with_kv_cache``.

Instructions
------------

This test will fail with many backends; use natten.  Run with:

I4_ATTN_BACKENDS=natten I4_ATTN_BACKENDS_MERGE=natten \
pytest cosmos_framework/model/generator/mot/three_way_attention_test.py -v -rA

It is also worth checking whether clamp_empty_varlen_kv is needed on any
particular platform, run:

CLAMP_EMPTY_VARLEN_KV=false \
pytest cosmos_framework/model/generator/mot/three_way_attention_test.py -v -rA


Purpose
-------

The production implementation decomposes a long sliding-window attention
into four pieces — text self-attention, video-to-text CA, video-to-cache
CA, and video self-attention within the current segment (with optional
teacher forcing) — merged via NATTEN's ``merge_attentions``.  This test
asserts the decomposition is mathematically equivalent to a single dense
attention over the *concatenated* K/V sequence with the appropriate
composite mask.

Teacher-forcing: two-pass mirror of production
----------------------------------------------
In TF mode the tested path mirrors the production two-pass mechanism
exactly:

  * **Pass 1** runs ``three_way_attention_with_kv_cache`` on the *clean*
    Q/K/V using ``KVTrainMemoryValue`` (standard, non-TF flags) — so clean
    queries see text + cache + clean (temporal-causal SA).  The output of
    Pass 1 is *discarded*; only the clean K/V tensors are kept and threaded
    into Pass 2 as ``cached_clean_gen_k/v``.
  * **Pass 2** runs ``three_way_attention_with_kv_cache`` on the *noisy*
    Q/K/V using ``TFNoisyMemoryValue`` populated with the clean K/V from
    Pass 1.

The training loss is applied only to Pass 2's noisy output.  Clean Q is
therefore disconnected from the loss and receives no gradient; clean K/V
receive gradient solely because Pass 2's noisy queries attend to them.
Both passes share the same input leaves (text_q/k/v, clean_q/k/v,
video_q/k/v), and autograd backprops through both passes without any
gradient tricks — exactly as production does when
``teacher_forcing_detach_clean_kv=False``.

Reference path
--------------
Builds one long ``[Q | K | V]`` sequence and evaluates a dense PyTorch
attention reference with a custom boolean mask::

    Q   = [ text_q (real, unpadded) | clean_q (TF only) | video_q ]
    K/V = [ text_k (real) | cache_k (real) | clean_k (TF only) | video_k ]

The mask permits:
- Text queries → text keys, top-left causal.
- (TF) Clean Q (Pass-1 mirror) → text K, cache K, and clean K at frames
  ``[0, t]`` (temporal-causal).  Matches what Pass 1 of
  ``three_way_attention_with_kv_cache`` computes.
- Video queries → all text and all cache (unconditional).
- (TF only)   Video frame t → clean frames ``[0, t)`` (strictly past).
- (TF)        Video frame t → video frame ``t`` only (spatial within frame).
- (non-TF)    Video frame t → video frames ``[0, t]`` (temporal-causal).

The KV-cache K/V are non-differentiable (frozen from previous segments) so
the cache section appears only on the KV axis — no corresponding cache Q.

Gradient comparison
-------------------
Tested and reference forward paths consume the *same* input leaves — no
cloning.  Gradients are pulled out via ``torch.autograd.grad`` (with
``materialize_grads=True``), which fetches grad of a specific output
w.r.t. specific inputs without writing to ``.grad``, so the two paths
do not interfere with each other.  Upstream gradient is applied only to
the video output (matching production where the loss is on Pass 2's
noisy-video prediction); ``materialize_grads=True`` resolves the
``None`` vs. zero-tensor mismatch that would otherwise occur for inputs
disconnected from the video output in one path but reachable through a
zero-grad route in the other.

Variants
--------
Cartesian product of:
- ``has_text``: caption present (real length 7) vs. absent (length 0)
- ``has_cache``: prior segment cached vs. empty cache (``seg_idx == 0``)
- ``mode``: standard temporal-causal vs. teacher-forcing
- ``dtype``: fp32 (tol 1e-5) and bf16 (tol 1e-2)
"""

import math
import os

# Force the NATTEN backend for both regular attention and merge_attentions
# unless the user has explicitly overridden it via the environment.  This
# test was designed around NATTEN's hopper-fmha / hopper-fna kernels;
# other backends (cudnn, flash3, ...) fail variants of it for reasons
# unrelated to what the test is meant to check.
#
# ``setdefault`` preserves an explicit user override, e.g.
#   ``I4_ATTN_BACKENDS=cudnn pytest ...``
# still wins.  Must run BEFORE any import that triggers attention backend
# selection — cosmos_framework.model.attention.backends.choose_backend is
# ``@lru_cache``'d, so once it's been called the env var won't take effect.
#
# Update: backend is now locked to natten in three_way_attention_with_kv_cache.
# To test with other backends, remove the backend="natten" arguments, and
# (optionally) uncomment the following:
#
# os.environ.setdefault("I4_ATTN_BACKENDS", "natten")
# os.environ.setdefault("I4_ATTN_BACKENDS_MERGE", "natten")
#
from collections.abc import Callable

import pytest
import torch

from cosmos_framework.model.attention import attention as imaginaire_attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_gen_seq,
)
from cosmos_framework.model.generator.mot.causal_attention import (
    three_way_attention_with_kv_cache,
)
from cosmos_framework.model.generator.utils.kv_cache import (
    KVTrainMemoryValue,
    TFNoisyMemoryValue,
)

# Shape / config constants.
_T = 8
_H_P, _W_P = 4, 4
_TCF = 0  # no action tokens; user opted to skip this variant
_S_SUPER = _TCF + _H_P * _W_P  # 16
_TOKENS_PER_SEG = _T * _S_SUPER  # 128

_NUM_HEADS = 4
_NUM_KV_HEADS = 4  # GQA disabled (num_heads == num_kv_heads)
_HEAD_DIM = 64

_S_TEXT_REAL = 7  # used when has_text=True
_PADDED_TEXT_LEN = 128  # text padding; must be >= _S_TEXT_REAL

_CACHE_SIZE = 3
_MAX_VIDEO_CACHE_TOKENS = (_CACHE_SIZE - 1) * _TOKENS_PER_SEG  # 256
_S_CACHE_REAL = _TOKENS_PER_SEG  # has_cache=True caches exactly one segment

# Set to True if the underlying attention backend does not support varlen
# attention to zero-length sequences.  Not all kernels are safe for zero-length.
_CLAMP_EMPTY_VARLEN_KV = os.environ.get("CLAMP_EMPTY_VARLEN_KV", "true").lower() in (
    "true",
    "1",
    "yes",
)


# ---------------------------------------------------------------------------
# Input construction (real, unpadded tensors used by both paths)
# ---------------------------------------------------------------------------


def _build_raw_inputs(
    *,
    has_text: bool,
    has_cache: bool,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    requires_grad: bool,
) -> dict[str, torch.Tensor | None]:
    """Generate the un-padded per-variant input tensors.

    ``text_*`` and ``video_*`` are the live Q/K/V for the current pack;
    ``cache_*`` are the prior-segment K/V (no Q); ``clean_*`` are the
    current-segment clean Q/K/V used only in teacher forcing.

    ``requires_grad=True`` enables backward-pass leaves on the
    differentiable inputs: text_q/k/v, video_q/k/v, and (in TF) clean_q/k/v.
    Cache K/V is always no-grad — the rolling cache stores frozen K/V
    projections from finished segments.  In production with
    ``teacher_forcing_detach_clean_kv=True`` the clean K/V would also be
    detached for memory savings; this test exercises the
    ``detach_clean_kv=False`` configuration where clean gradients flow.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    s_text = _S_TEXT_REAL if has_text else 0
    s_cache = _S_CACHE_REAL if has_cache else 0

    def randn(*shape: int, grad: bool = False) -> torch.Tensor:
        t = torch.randn(*shape, device=device, dtype=dtype, generator=g)
        if grad and requires_grad:
            t.requires_grad_(True)
        return t

    text_q = randn(s_text, _NUM_HEADS, _HEAD_DIM, grad=True)
    text_k = randn(s_text, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
    text_v = randn(s_text, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
    cache_k = randn(s_cache, _NUM_KV_HEADS, _HEAD_DIM)
    cache_v = randn(s_cache, _NUM_KV_HEADS, _HEAD_DIM)
    video_q = randn(_TOKENS_PER_SEG, _NUM_HEADS, _HEAD_DIM, grad=True)
    video_k = randn(_TOKENS_PER_SEG, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
    video_v = randn(_TOKENS_PER_SEG, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
    if mode == "teacher_forcing":
        clean_q = randn(_TOKENS_PER_SEG, _NUM_HEADS, _HEAD_DIM, grad=True)
        clean_k = randn(_TOKENS_PER_SEG, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
        clean_v = randn(_TOKENS_PER_SEG, _NUM_KV_HEADS, _HEAD_DIM, grad=True)
    else:
        clean_q = None
        clean_k = None
        clean_v = None

    return {
        "text_q": text_q,
        "text_k": text_k,
        "text_v": text_v,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "video_q": video_q,
        "video_k": video_k,
        "video_v": video_v,
        "clean_q": clean_q,
        "clean_k": clean_k,
        "clean_v": clean_v,
    }


# ---------------------------------------------------------------------------
# Reference path: single dense attention call over the concatenated sequence
# ---------------------------------------------------------------------------


def _make_composite_mask_mod(
    *,
    s_text: int,
    s_cache: int,
    num_frames: int,
    s_super: int,
    is_tf: bool,
):
    """Build the long-concat attention mask rule.

    Layout (Q):   [ text(s_text) | clean(opt) | noisy(num_frames * s_super) ]
    Layout (KV):  [ text(s_text) | cache(s_cache) | clean(opt) | noisy ]

    The clean span on both axes has length ``num_frames * s_super`` when
    ``is_tf=True`` and length 0 in standard (non-TF) mode.

    Per-token attention rules:
      - Text Q → text K, top-left causal (``kv_idx <= q_idx``).
      - (TF) Clean Q at frame ``t`` (Pass-1 mirror) → all text K, all
        cache K, and clean K at frames ``<= t`` (temporal-causal within
        the clean span).  This is the full Pass-1 three-way attention
        pattern: ``three_way_attention_with_kv_cache`` called on clean
        Q/K/V with a ``KVTrainMemoryValue`` (no TF flags).
      - Noisy Q → all text K, all cache K (unconditional).
      - (TF only)   Noisy frame ``t`` → clean frames ``< t`` (strictly past).
      - (TF)        Noisy frame ``t`` → noisy frame ``== t`` (spatial only).
      - (Standard)  Noisy frame ``t`` → noisy frames ``<= t`` (causal).

    Args:
        s_text: Real (un-padded) text caption length on both the Q and KV
            axes.  ``0`` when no caption is present; in that case the text
            spans collapse and no text rules fire.
        s_cache: Real (un-padded) length of the prior-segments rolling KV
            cache on the KV axis.  ``0`` when the cache is empty
            (``seg_idx == 0``); the cache span then has zero width.
        num_frames: Number of latent frames in the current segment along
            the noisy (and, in TF mode, clean) Q + KV spans.  The
            segment's token count per span is ``num_frames * s_super``.
        s_super: Tokens per latent frame (action tokens + spatial
            tokens).  Used to map a Q/KV offset within the current
            segment to its frame index via integer division.
        is_tf: ``True`` selects the teacher-forcing rules and adds a clean
            section to both the Q and KV axes; ``False`` selects standard
            temporal-causal noisy SA and the clean spans are empty.

    Returns:
        A ``mask_mod(b, h, q_idx, kv_idx) -> bool`` callable. ``True`` means
        the ``(q_idx, kv_idx)`` pair attends; ``False`` means it is masked out.
    """
    tokens_per_seg = num_frames * s_super
    # ---------------------------------------------------------------------
    # Section boundaries (closed at start, open at end) in the concat layout.
    #
    # Q axis:  [ 0 ..... clean_q_start ......... noisy_q_start ........ S_q )
    #          |<-- text -->|<- clean (TF only) ->|<-------- noisy -------->|
    #
    # KV axis: [ 0 ... text_k_end ... cache_k_end ... clean_k_end .... S_kv )
    #          |<- text ->|<- cache ->|<- clean (TF only) ->|<--- noisy --->|
    #
    # In the non-TF case the clean spans have length 0 on BOTH axes
    # (start == end), so the Q axis reduces to [ text | noisy ] and the KV
    # axis reduces to [ text | cache | noisy ].  No branch is needed inside
    # mask_mod: all "is_clean_*" comparisons evaluate to False against the
    # collapsed span, and the clean-row rules vanish from the disjunction.
    # ---------------------------------------------------------------------

    # Q axis boundaries.
    clean_q_start = s_text
    clean_q_end = clean_q_start + (tokens_per_seg if is_tf else 0)
    noisy_q_start = clean_q_end

    # KV axis boundaries.
    text_k_end = s_text
    cache_k_start = text_k_end
    cache_k_end = cache_k_start + s_cache
    clean_k_start = cache_k_end
    clean_k_end = clean_k_start + (tokens_per_seg if is_tf else 0)
    noisy_k_start = clean_k_end

    def mask_mod(b, h, q_idx, kv_idx):
        # mask_mod is called with scalar tensor indices (b, h, q_idx, kv_idx)
        # for every (q, kv) pair the reference considers.  It must return a
        # scalar bool tensor: True ⇒ this pair attends, False ⇒ masked out.
        # All operations must be tensor-friendly (no Python `if` on tensor
        # values; use `&`, `|`, `torch.where`).

        # --- Classify the query row -----------------------------------
        # Q layout: [text | (clean if TF) | noisy].  Exactly one is_*_q is
        # True for any in-range q_idx; in non-TF the clean span has zero
        # width so is_clean_q is always False.
        is_text_q = q_idx < clean_q_start
        is_clean_q = (q_idx >= clean_q_start) & (q_idx < clean_q_end)

        # Frame indices for clean Q and noisy Q.  When the corresponding
        # is_*_q is False the value is negative/out-of-range, but it's
        # gated out below by the is_*_q checks before being used.
        clean_q_frame = (q_idx - clean_q_start) // s_super
        noisy_q_frame = (q_idx - noisy_q_start) // s_super

        # --- Classify the key/value column ----------------------------
        # Each KV row falls into exactly one of {text, cache, clean, noisy}
        # depending on which half-open span it lands in.
        is_text_k = kv_idx < text_k_end
        is_cache_k = (kv_idx >= cache_k_start) & (kv_idx < cache_k_end)
        is_clean_k = (kv_idx >= clean_k_start) & (kv_idx < clean_k_end)
        is_noisy_k = kv_idx >= noisy_k_start

        # Frame indices for the current-segment clean / noisy K rows.
        # Same out-of-range caveat as above: gated by the is_*_k masks.
        clean_kv_frame = (kv_idx - clean_k_start) // s_super
        noisy_kv_frame = (kv_idx - noisy_k_start) // s_super

        # --- Text-Q row -------------------------------------------------
        # Text queries only attend within the text block, with top-left
        # causal mask: position i sees positions [0, i].  Outside the text
        # block is_text_k is False so the whole row is False.
        text_allowed = is_text_k & (kv_idx <= q_idx)

        # --- Clean-Q row (TF only) --------------------------------------
        # Pass 1 of teacher forcing runs the full
        # ``three_way_attention_with_kv_cache`` over clean tokens with
        # ``KVTrainMemoryValue`` (standard, non-TF flags), so clean Q at
        # frame f attends to:
        #   - all real text K
        #   - all real cache K
        #   - clean K at frames [0, f] (temporal-causal SA in the frame
        #     dim, full within a frame).
        # Clean Q does not attend to noisy K — Pass 1 has no noisy tokens.
        clean_text = is_text_k
        clean_cache = is_cache_k
        clean_clean = is_clean_k & (clean_kv_frame <= clean_q_frame)
        clean_allowed = clean_text | clean_cache | clean_clean

        # --- Noisy-Q row -----------------------------------------------
        # Noisy attends unconditionally to all real text and all real cache.
        noisy_text = is_text_k
        noisy_cache = is_cache_k
        if is_tf:
            # Teacher forcing decomposes "noisy Q attends to
            # concat(clean[:t], noisy[t])" into two disjoint pieces:
            #   - strictly-past clean: noisy frame f's query attends to
            #     clean K at frames < f (i.e. frames [0, f));
            #   - spatial within own frame: noisy frame f's query attends
            #     to noisy K at frame == f (any spatial position).
            # These mirror teacher_forcing_gen_attention's (clean_ca,
            # spatial_sa) split — see [attention.py:191-240].
            noisy_clean = is_clean_k & (clean_kv_frame < noisy_q_frame)
            noisy_noisy = is_noisy_k & (noisy_kv_frame == noisy_q_frame)
        else:
            # Standard mode: noisy Q at frame f sees noisy K at frames
            # [0, f] (temporal-causal in the frame dim, full spatial
            # within each frame).  No clean block exists.
            noisy_clean = is_clean_k & False
            noisy_noisy = is_noisy_k & (noisy_kv_frame <= noisy_q_frame)
        noisy_allowed = noisy_text | noisy_cache | noisy_clean | noisy_noisy

        # --- Select the right row by query type -----------------------
        # Nested torch.where: text queries → text_allowed; clean queries
        # → clean_allowed; everything else (noisy) → noisy_allowed.
        return torch.where(
            is_text_q,
            text_allowed,
            torch.where(is_clean_q, clean_allowed, noisy_allowed),
        )

    return mask_mod


def _pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Pure-pytorch scaled dot-product attention with a dense bool mask.

    No torch.compile or SDPA fused kernels — just
    ``softmax(Q @ K^T / sqrt(d)) @ V`` written out as einsums and a
    ``masked_fill(~mask, -inf)``.  This is the most kernel-independent
    reference we can write; the caller uses it to decouple the algebraic
    intent of the test from any hardware-specific attention kernel
    behavior.

    Test setup uses ``num_heads == num_kv_heads`` (no GQA), so we can
    einsum across the head dim directly without broadcasting.  If GQA
    is ever needed, repeat-K/V along the head axis or replace the
    einsums with explicit broadcasting.

    Args:
        q: Query tensor of shape ``[B, S_q, H, D]``.
        k: Key tensor of shape ``[B, S_kv, H_kv, D]``.
        v: Value tensor of shape ``[B, S_kv, H_kv, D]``.
        attn_mask: Bool mask of shape ``[S_q, S_kv]``.  ``True`` means
            the (Q, KV) pair may attend; ``False`` means masked out
            (score forced to ``-inf`` before softmax).

    Returns:
        Output tensor of shape ``[S_q, H, D]`` (batch squeezed; the
        sole caller is single-batch).
    """
    head_dim = q.shape[-1]
    s_q = q.shape[1]
    s_kv = k.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    q_bhsd = q.transpose(1, 2)  # [B, H, S_q, D]
    k_bhsd = k.transpose(1, 2)  # [B, H_kv, S_kv, D]
    v_bhsd = v.transpose(1, 2)  # [B, H_kv, S_kv, D]
    scores = torch.einsum("bhsd,bhkd->bhsk", q_bhsd, k_bhsd) * scale
    # This will cause a natten compile bug:
    # scores = scores.masked_fill(~attn_mask.view(1, 1, s_q, s_kv), float("-inf"))
    neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
    scores = torch.where(attn_mask.view(1, 1, s_q, s_kv), scores, neg_inf)
    weights = torch.softmax(scores, dim=-1)
    out_bhsd = torch.einsum("bhsk,bhkd->bhsd", weights, v_bhsd)
    return out_bhsd.transpose(1, 2).squeeze(0)  # [S_q, H, D]


def _reference_forward(
    inputs: dict[str, torch.Tensor | None],
    *,
    mode: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Run the long-concat reference and return ``(text_out, clean_out, video_out)``.

    Tensors come from :func:`_build_raw_inputs` at their *real* (un-padded)
    lengths.  ``text_out`` is ``None`` when ``s_text == 0``; ``clean_out`` is
    ``None`` outside of TF mode.

    The reference expands :func:`_make_composite_mask_mod` into a dense
    ``[S_q, S_kv]`` boolean mask and runs ``softmax(QK^T / sqrt(d)) V``
    manually in pure PyTorch.
    """
    text_q = inputs["text_q"]
    text_k = inputs["text_k"]
    text_v = inputs["text_v"]
    cache_k = inputs["cache_k"]
    cache_v = inputs["cache_v"]
    video_q = inputs["video_q"]
    video_k = inputs["video_k"]
    video_v = inputs["video_v"]
    assert text_q is not None and text_k is not None and text_v is not None
    assert cache_k is not None and cache_v is not None
    assert video_q is not None and video_k is not None and video_v is not None
    is_tf = mode == "teacher_forcing"

    s_text = text_q.shape[0]
    s_cache = cache_k.shape[0]

    # Build the Q axis: [ text | clean (TF only) | video ].
    q_parts = [text_q]
    if is_tf:
        clean_q = inputs["clean_q"]
        assert clean_q is not None
        q_parts.append(clean_q)
    q_parts.append(video_q)
    q_long = torch.cat(q_parts, dim=0).unsqueeze(0)  # [1, S_q, H, D]

    # Build the KV axis: [ text | cache | clean (TF only) | video ].
    k_parts = [text_k, cache_k]
    v_parts = [text_v, cache_v]
    if is_tf:
        clean_k = inputs["clean_k"]
        clean_v = inputs["clean_v"]
        assert clean_k is not None and clean_v is not None
        k_parts.append(clean_k)
        v_parts.append(clean_v)
    k_parts.append(video_k)
    v_parts.append(video_v)
    k_long = torch.cat(k_parts, dim=0).unsqueeze(0)  # [1, S_kv, H_kv, D]
    v_long = torch.cat(v_parts, dim=0).unsqueeze(0)

    S_q_total = q_long.shape[1]
    S_kv_total = k_long.shape[1]

    mask_mod = _make_composite_mask_mod(
        s_text=s_text,
        s_cache=s_cache,
        num_frames=_T,
        s_super=_S_SUPER,
        is_tf=is_tf,
    )

    # Materialize mask_mod into a dense [S_q_total, S_kv_total] bool.
    # mask_mod is written to be tensor-friendly, so calling it on a
    # meshgrid evaluates the entire mask in one shot.  b and h are
    # unused by our mask_mod; pass scalars to keep the signature happy.
    device = q_long.device
    q_idx = torch.arange(S_q_total, device=device)
    kv_idx = torch.arange(S_kv_total, device=device)
    q_grid, kv_grid = torch.meshgrid(q_idx, kv_idx, indexing="ij")
    b0 = torch.tensor(0, device=device)
    h0 = torch.tensor(0, device=device)
    attn_mask = mask_mod(b0, h0, q_grid, kv_grid)  # [S_q, S_kv] bool
    out = _pytorch_attention(q_long, k_long, v_long, attn_mask)

    # Split out the three Q-axis regions in concat order.
    text_out = out[:s_text] if s_text > 0 else None
    if is_tf:
        clean_out = out[s_text : s_text + _TOKENS_PER_SEG]
        video_out = out[s_text + _TOKENS_PER_SEG : s_text + 2 * _TOKENS_PER_SEG]
    else:
        clean_out = None
        video_out = out[s_text : s_text + _TOKENS_PER_SEG]
    return text_out, clean_out, video_out


# ---------------------------------------------------------------------------
# Tested path: three_way_attention_with_kv_cache via SequencePack
# ---------------------------------------------------------------------------


def _build_packed_state(
    *,
    text_seq: torch.Tensor,
    video_seq: torch.Tensor,
    device: torch.device,
) -> SequencePack:
    """Build a SequencePack with text padded to ``_PADDED_TEXT_LEN``.

    The pack mirrors the layout that ``sequence_pack_from_packed_sequence`` would
    produce: ``causal_seq`` is padded to a constant size, ``max_causal_len``
    reflects the *real* (unpadded) text length, and ``_causal_seq_offsets``
    encodes the varlen window the attention kernels will respect.
    """
    s_text_real = text_seq.shape[0]
    pad_rows = _PADDED_TEXT_LEN - s_text_real
    assert pad_rows >= 0
    pad = torch.zeros(pad_rows, *text_seq.shape[1:], dtype=text_seq.dtype, device=device)
    causal_seq = torch.cat([text_seq, pad], dim=0) if s_text_real > 0 else pad
    S_v = video_seq.shape[0]
    return {
        # Upstream schema keys (``full_only_seq`` / ``_full_only_seq_offsets``
        # / ``_num_full_tokens``) hold the *video* tokens; the "full" name is
        # upstream schema terminology.
        "causal_seq": causal_seq,
        "full_only_seq": video_seq,
        "is_sharded": False,
        "sample_offsets": torch.tensor([0, _PADDED_TEXT_LEN + S_v], device=device, dtype=torch.int32),
        "max_sample_len": _PADDED_TEXT_LEN + S_v,
        "max_causal_len": s_text_real,
        "max_full_len": S_v,
        "_causal_indices": torch.arange(_PADDED_TEXT_LEN, device=device),
        "_full_indices": torch.arange(_PADDED_TEXT_LEN, _PADDED_TEXT_LEN + S_v, device=device),
        "_causal_seq_offsets": torch.tensor([0, s_text_real], device=device, dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, S_v], device=device, dtype=torch.int32),
        "_num_causal_tokens": s_text_real,
        "_num_full_tokens": S_v,
    }


def _build_memory_value(
    inputs: dict[str, torch.Tensor | None],
    *,
    has_text: bool,
    has_cache: bool,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> KVTrainMemoryValue | TFNoisyMemoryValue:
    """Hand-build the memory value carrying cache K/V, offsets, and flags.

    The structure mirrors what ``KVCacheTrainMemoryState.init()`` produces:
    cache tensors are padded to constant size, and the ``has_*`` flags + the
    varlen offsets work together to mask out padding contributions
    (e.g. ``has_cached_gen=False`` forces the cache-CA LSE to ``-inf`` at
    seg_idx==0, neutralizing the merge weight regardless of cache contents).

    The upstream ``KVTrainMemoryValue`` dataclass uses historical ``*_gen_*`` /
    ``*_und_*`` field names; we keep those as kwarg names but use ``video``
    / ``text`` for local variables.
    """
    s_text_real = _S_TEXT_REAL if has_text else 0
    s_cache_real = _S_CACHE_REAL if has_cache else 0

    clamp_empty_varlen_kv = _CLAMP_EMPTY_VARLEN_KV
    clamp_min = 1 if clamp_empty_varlen_kv else 0

    cache_k = inputs["cache_k"]
    cache_v = inputs["cache_v"]
    assert cache_k is not None and cache_v is not None

    # Cached text K/V: zero-filled (the live caption path is exercised when
    # has_new_caption=True; when False, no real cached caption exists either).
    cached_text_k = torch.zeros(1, _PADDED_TEXT_LEN, _NUM_KV_HEADS, _HEAD_DIM, dtype=dtype, device=device)
    cached_text_v = torch.zeros(1, _PADDED_TEXT_LEN, _NUM_KV_HEADS, _HEAD_DIM, dtype=dtype, device=device)

    # Cached video K/V: real prefix + zero padding up to _MAX_VIDEO_CACHE_TOKENS.
    if has_cache:
        cached_video_body_k = cache_k.unsqueeze(0)
        cached_video_body_v = cache_v.unsqueeze(0)
    else:
        cached_video_body_k = torch.zeros(1, 0, _NUM_KV_HEADS, _HEAD_DIM, dtype=dtype, device=device)
        cached_video_body_v = torch.zeros(1, 0, _NUM_KV_HEADS, _HEAD_DIM, dtype=dtype, device=device)
    pad_tokens = _MAX_VIDEO_CACHE_TOKENS - s_cache_real
    pad_shape = (1, pad_tokens, _NUM_KV_HEADS, _HEAD_DIM)
    cached_video_k = torch.cat([cached_video_body_k, torch.zeros(pad_shape, dtype=dtype, device=device)], dim=1)
    cached_video_v = torch.cat([cached_video_body_v, torch.zeros(pad_shape, dtype=dtype, device=device)], dim=1)

    kwargs: dict = dict(
        vision_token_shapes=[(_T, _H_P, _W_P)],
        num_action_tokens_per_supertoken=_TCF,
        # has_text=True ⇒ caption lives in current pack ⇒ has_new_caption=True
        # has_text=False ⇒ no caption anywhere ⇒ has_new_caption=False
        # (cached_text_* is zeros).  has_caption mirrors the cached-video
        # `has_cached_gen` flag: True when any real text exists.
        has_new_caption=torch.tensor(has_text, device=device),
        has_caption=torch.tensor(has_text, device=device),
        has_cached_gen=torch.tensor(has_cache, device=device),
        # When clamp_empty_varlen_kv is True, clamp varlen offsets
        # to >=1 so the FA kernel never sees a zero-length range.
        und_kv_offsets=torch.tensor([0, max(s_text_real, clamp_min)], device=device, dtype=torch.int32),
        gen_q_offsets=torch.tensor([0, _TOKENS_PER_SEG], device=device, dtype=torch.int32),
        gen_ca_cached_kv_offsets=torch.tensor(
            [0, max(s_cache_real, clamp_min)],
            device=device,
            dtype=torch.int32,
        ),
        cached_und_k=cached_text_k,
        cached_und_v=cached_text_v,
        cached_gen_k=cached_video_k,
        cached_gen_v=cached_video_v,
        max_gen_cache_tokens=_MAX_VIDEO_CACHE_TOKENS,
        clamp_empty_varlen_kv=clamp_empty_varlen_kv,
    )

    if mode == "teacher_forcing":
        clean_k = inputs["clean_k"]
        clean_v = inputs["clean_v"]
        assert clean_k is not None and clean_v is not None
        return TFNoisyMemoryValue(
            **kwargs,
            cached_clean_gen_k=clean_k.unsqueeze(0),
            cached_clean_gen_v=clean_v.unsqueeze(0),
        )
    return KVTrainMemoryValue(**kwargs)


def _run_three_way(
    *,
    text_q: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    video_q: torch.Tensor,
    video_k: torch.Tensor,
    video_v: torch.Tensor,
    memory_value: KVTrainMemoryValue | TFNoisyMemoryValue,
    has_text: bool,
    device: torch.device,
    attention_fn: Callable | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Single ``three_way_attention_with_kv_cache`` invocation.

    Returns ``(text_out, video_out)`` with the padded text region sliced
    down to its real length.

    ``attention_fn`` lets the caller inject a wrapped variant of
    ``three_way_attention_with_kv_cache`` (e.g. ``torch.compile``'d) so
    the same helper can drive both the eager and compiled test variants.
    """
    if attention_fn is None:
        attention_fn = three_way_attention_with_kv_cache
    q_pack = _build_packed_state(text_seq=text_q, video_seq=video_q, device=device)
    k_pack = _build_packed_state(text_seq=text_k, video_seq=video_k, device=device)
    v_pack = _build_packed_state(text_seq=text_v, video_seq=video_v, device=device)
    mask = SplitInfo(
        split_lens=[_PADDED_TEXT_LEN, _TOKENS_PER_SEG],
        attn_modes=["causal", "full"],
        sample_lens=[_PADDED_TEXT_LEN + _TOKENS_PER_SEG],
        actual_len=_PADDED_TEXT_LEN + _TOKENS_PER_SEG,
    )
    out_pack = attention_fn(
        q_pack,
        k_pack,
        v_pack,
        memory_value=memory_value,
        attention_meta=mask,
    )
    # ``get_gen_seq`` is the upstream accessor for the video-token portion
    # of the pack; the "gen" name comes from the shared schema.
    video_out = get_gen_seq(out_pack).unflatten(-1, (_NUM_HEADS, _HEAD_DIM))
    text_out_padded, _ = get_causal_seq(out_pack)
    text_out_padded_4d = text_out_padded.unflatten(-1, (_NUM_HEADS, _HEAD_DIM))
    s_text_real = _S_TEXT_REAL if has_text else 0
    text_out = text_out_padded_4d[:s_text_real] if s_text_real > 0 else None
    return text_out, video_out


def _tested_forward(
    inputs: dict[str, torch.Tensor | None],
    *,
    has_text: bool,
    has_cache: bool,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    use_compile: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Run the tested decomposition and return ``(text_out, clean_out, video_out)``.

    Standard mode: a single ``three_way_attention_with_kv_cache`` call.
    ``clean_out`` is ``None``.

    Teacher-forcing mode: two passes, mirroring production exactly.

      * Pass 1 calls ``three_way_attention_with_kv_cache`` on the *clean*
        Q/K/V with a ``KVTrainMemoryValue`` (standard flags, no TF).
        The returned text+video output is the Pass-1 attention output;
        the video part is ``clean_out``.  Production discards Pass 1's
        per-layer outputs and keeps only the clean K/V projections to
        feed into Pass 2; this test keeps the video output too so the
        forward check can compare it to the reference's clean-Q rows.
      * Pass 2 calls ``three_way_attention_with_kv_cache`` on the *noisy*
        video Q/K/V with a ``TFNoisyMemoryValue`` whose
        ``cached_clean_gen_k/v`` point at the SAME clean K/V leaf tensors
        used in Pass 1.  Pass 2's text+video output supplies ``text_out``
        (returned) and ``video_out`` (the noisy-video attention output
        the training loss is applied to).

    Both passes share the same input leaves (text, clean, video) and the
    autograd graph naturally records both, so backprop through Pass 2's
    output accumulates gradients into shared leaves without any data-
    pointer or detach trick — matching production with
    ``teacher_forcing_detach_clean_kv=False``.

    When ``use_compile`` is True, ``three_way_attention_with_kv_cache``
    is wrapped in ``torch.compile(fullgraph=True)`` so both Pass 1 and
    Pass 2 run through the compiled graph.  The Dynamo cache is reset
    first so each test starts from a clean compile.
    """
    text_q = inputs["text_q"]
    text_k = inputs["text_k"]
    text_v = inputs["text_v"]
    video_q = inputs["video_q"]
    video_k = inputs["video_k"]
    video_v = inputs["video_v"]
    assert text_q is not None and text_k is not None and text_v is not None
    assert video_q is not None and video_k is not None and video_v is not None

    if use_compile:
        torch._dynamo.reset()
        attention_fn: Callable = torch.compile(three_way_attention_with_kv_cache, fullgraph=True)
    else:
        attention_fn = three_way_attention_with_kv_cache

    if mode == "teacher_forcing":
        clean_q = inputs["clean_q"]
        clean_k = inputs["clean_k"]
        clean_v = inputs["clean_v"]
        assert clean_q is not None and clean_k is not None and clean_v is not None

        # Pass 1: clean tokens through the standard (non-TF) three_way.
        # The standard-mode KVTrainMemoryValue makes the video self-attention
        # branch take the temporal-causal SA path (no TFNoisyMemoryValue
        # split), exactly matching the clean-Q rules in mask_mod.
        mv_pass1 = _build_memory_value(
            inputs,
            has_text=has_text,
            has_cache=has_cache,
            mode="standard",
            device=device,
            dtype=dtype,
        )
        _text_out_p1, clean_out = _run_three_way(
            text_q=text_q,
            text_k=text_k,
            text_v=text_v,
            video_q=clean_q,
            video_k=clean_k,
            video_v=clean_v,
            memory_value=mv_pass1,
            has_text=has_text,
            device=device,
            attention_fn=attention_fn,
        )

        # Pass 2: noisy-video tokens with the captured clean K/V wired in
        # via TFNoisyMemoryValue.cached_clean_gen_k/v.  Same clean_k /
        # clean_v leaves as Pass 1 — backprop from Pass 2's video_out
        # flows through cached_clean_gen_k/v back into clean_k / clean_v.
        mv_pass2 = _build_memory_value(
            inputs,
            has_text=has_text,
            has_cache=has_cache,
            mode="teacher_forcing",
            device=device,
            dtype=dtype,
        )
        text_out, video_out = _run_three_way(
            text_q=text_q,
            text_k=text_k,
            text_v=text_v,
            video_q=video_q,
            video_k=video_k,
            video_v=video_v,
            memory_value=mv_pass2,
            has_text=has_text,
            device=device,
            attention_fn=attention_fn,
        )
        return text_out, clean_out, video_out

    # Standard mode: single pass, no clean output.
    mv = _build_memory_value(
        inputs,
        has_text=has_text,
        has_cache=has_cache,
        mode=mode,
        device=device,
        dtype=dtype,
    )
    text_out, video_out = _run_three_way(
        text_q=text_q,
        text_k=text_k,
        text_v=text_v,
        video_q=video_q,
        video_k=video_k,
        video_v=video_v,
        memory_value=mv,
        has_text=has_text,
        device=device,
        attention_fn=attention_fn,
    )
    return text_out, None, video_out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


_DTYPE_TOL = [
    pytest.param(torch.float32, 1e-3, id="fp32"),
    # bf16 ULP is ~1/64 = 0.015625 for values around 1.  The gradient path
    # chains many bf16 ops (Pass 1 + Pass 2, four attention kernels each
    # in TF + the merge) and the largest observed mismatches land at
    # exactly one ULP; 2e-2 leaves ~1 ULP of headroom above that floor.
    pytest.param(torch.bfloat16, 2e-2, id="bf16"),
]


def _seed(has_text: bool, has_cache: bool, mode: str) -> int:
    """Stable per-variant seed (avoids cross-variant correlation)."""
    return 1000 + int(has_text) * 100 + int(has_cache) * 10 + (0 if mode == "standard" else 1)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("has_text", [True, False], ids=["with_text", "no_text"])
@pytest.mark.parametrize("has_cache", [True, False], ids=["with_cache", "no_cache"])
@pytest.mark.parametrize("mode", ["standard", "teacher_forcing"])
@pytest.mark.parametrize("dtype, tol", _DTYPE_TOL)
def test_three_way_attention_forward_matches_concat_reference(
    has_text: bool,
    has_cache: bool,
    mode: str,
    dtype: torch.dtype,
    tol: float,
):
    """Forward output of ``three_way_attention_with_kv_cache`` must match a
    single dense attention call over the concatenated K/V sequence with
    the variant's composite mask.
    """
    device = torch.device("cuda")
    seed = _seed(has_text, has_cache, mode)

    # Tested and reference paths share the same input leaves.  Forward only
    # — no backward — so there is no .grad accumulator interference to worry
    # about.  The two forwards build independent autograd subgraphs that
    # both originate from these leaves.
    inputs = _build_raw_inputs(
        has_text=has_text,
        has_cache=has_cache,
        mode=mode,
        dtype=dtype,
        device=device,
        seed=seed,
        requires_grad=False,
    )

    text_ref, clean_ref, video_ref = _reference_forward(inputs, mode=mode)
    text_out, clean_out, video_out = _tested_forward(
        inputs,
        has_text=has_text,
        has_cache=has_cache,
        mode=mode,
        device=device,
        dtype=dtype,
    )

    torch.testing.assert_close(video_out, video_ref, rtol=tol, atol=tol)
    if has_text:
        assert text_out is not None and text_ref is not None
        torch.testing.assert_close(text_out, text_ref, rtol=tol, atol=tol)
    else:
        assert text_out is None and text_ref is None
    if mode == "teacher_forcing":
        assert clean_out is not None and clean_ref is not None
        torch.testing.assert_close(clean_out, clean_ref, rtol=tol, atol=tol)
    else:
        assert clean_out is None and clean_ref is None


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("has_text", [True, False], ids=["with_text", "no_text"])
@pytest.mark.parametrize("has_cache", [True, False], ids=["with_cache", "no_cache"])
@pytest.mark.parametrize("mode", ["standard", "teacher_forcing"])
@pytest.mark.parametrize("dtype, tol", _DTYPE_TOL)
@pytest.mark.parametrize("use_compile", [False, True], ids=["eager", "compile"])
def test_three_way_attention_gradients_match_concat_reference(
    request: pytest.FixtureRequest,
    has_text: bool,
    has_cache: bool,
    mode: str,
    dtype: torch.dtype,
    tol: float,
    use_compile: bool,
):
    """Backward gradients through the full tested path (Pass 1 + Pass 2 in
    TF, single pass in standard) must match the concat-reference
    dense reference path when both consume the same random upstream grad
    on the noisy-video output.

    Scope of the comparison
    -----------------------
    We check gradients on every differentiable input leaf:
      * ``text_q/k/v``: the live text Q/K/V for the current pack.
      * ``video_q/k/v``: the current-segment noisy-video Q/K/V.
      * ``clean_q/k/v`` (TF only): the current-segment clean-video Q/K/V.

    Excluded:
      * ``cache_k/v``: not differentiable — production stores frozen K/V
        projections in the rolling cache.  ``_build_raw_inputs`` builds
        these without ``requires_grad`` so no leaf grad ever exists.

    Two ``torch.where`` calls inside ``three_way_attention_with_kv_cache``
    could in principle break the ``merge_attentions`` data-pointer
    contract:
      - The cached-video LSE mask (``has_cached_gen`` → -inf) sits between
        the kernel and ``merge_attentions``; it would break the contract
        without ``MergeAttentionsBridge``, but the bridge is applied
        whenever ``clamp_empty_varlen_kv`` is True (and the bf16 path
        skips both the clamp and the where).
      - The text-K/V selection (``has_new_caption`` → live vs. cached
        text K/V) sits *before* the text-CA kernel, on its K/V inputs,
        not on its O/LSE outputs — so the merge contract is preserved
        on the output side, and standard autograd through ``torch.where``
        flows the kernel's ``d(selected_text_k)`` back to ``text_k``.

    Cache K/V and (in TF) clean K/V flow into Pass 2's video attention
    via ``cached_clean_gen_k/v`` and ``cached_gen_k/v`` respectively.
    Clean K/V is the same leaf in Pass 1 and Pass 2; backward through
    Pass 2's strictly-past-clean attention writes the only gradient on
    clean K/V (Pass 1's output is not in the loss).

    Tested and reference share the same input leaves;
    ``torch.autograd.grad`` with ``materialize_grads=True`` fetches
    per-output gradients without touching ``.grad``, so neither path
    interferes with the other and ``None`` vs. zero-tensor mismatches
    are resolved to zero on both sides.
    """
    # Two specific bf16 + compile variants are known to hit a NATTEN
    # compiled-path bug on H100 (sm90) — see natten_bug_3way.py for the
    # isolated A/B reproducer.  The failing test IDs are:
    #   [compile-bf16-standard-with_cache-with_text]
    #   [compile-bf16-teacher_forcing-no_cache-with_text]
    is_known_natten_bf16_compile_fail = (
        use_compile
        and dtype == torch.bfloat16
        and has_text
        and ((mode == "standard" and has_cache) or (mode == "teacher_forcing"))
    )
    if is_known_natten_bf16_compile_fail:
        request.applymarker(
            pytest.mark.xfail(
                reason="Known to fail on H100 (sm90) with NATTEN + torch.compile + bf16.",
                strict=False,
            )
        )

    device = torch.device("cuda")
    seed = _seed(has_text, has_cache, mode)

    inputs = _build_raw_inputs(
        has_text=has_text,
        has_cache=has_cache,
        mode=mode,
        dtype=dtype,
        device=device,
        seed=seed,
        requires_grad=True,
    )

    _text_ref, _clean_ref, video_ref = _reference_forward(inputs, mode=mode)
    _text_out, _clean_out, video_out = _tested_forward(
        inputs,
        has_text=has_text,
        has_cache=has_cache,
        mode=mode,
        device=device,
        dtype=dtype,
        use_compile=use_compile,
    )

    # Upstream gradient is applied only to the noisy-video output,
    # matching production (the training loss is on Pass 2's noisy-video
    # prediction).  Seed the grad so the test is deterministic —
    # otherwise cases near the tolerance threshold can flip pass/fail
    # between runs.
    grad_video_generator = torch.Generator(device=device).manual_seed(seed + 1)
    grad_video = torch.randn(
        video_out.shape,
        device=device,
        dtype=video_out.dtype,
        generator=grad_video_generator,
    )

    # Collect the leaves whose gradients we want to compare — every
    # differentiable input.  See the docstring for why cache K/V are
    # excluded.
    grad_keys = ["text_q", "text_k", "text_v", "video_q", "video_k", "video_v"]
    if mode == "teacher_forcing":
        grad_keys += ["clean_q", "clean_k", "clean_v"]
    leaves: list[torch.Tensor] = []
    for k in grad_keys:
        t = inputs[k]
        assert t is not None
        leaves.append(t)

    # ``materialize_grads=True`` returns a zero tensor for inputs the
    # output does not depend on (instead of ``None``).  ``text_q`` and
    # ``clean_q`` are read by the loss only through routes whose net
    # contribution to ``video_out`` is zero (text_q feeds only the text
    # SA; clean_q feeds only Pass 1), so the autograd graphs reach them
    # in one path but not the other; materializing both to zero makes
    # them directly comparable.
    grads_tested = torch.autograd.grad(
        outputs=video_out,
        inputs=leaves,
        grad_outputs=grad_video,
        materialize_grads=True,
    )
    grads_ref = torch.autograd.grad(
        outputs=video_ref,
        inputs=leaves,
        grad_outputs=grad_video,
        materialize_grads=True,
    )

    for key, g_tested, g_ref in zip(grad_keys, grads_tested, grads_ref):
        torch.testing.assert_close(
            g_tested,
            g_ref,
            rtol=tol,
            atol=tol,
            msg=lambda m, k=key: f"{k}.grad mismatch: {m}",
        )


# ---------------------------------------------------------------------------
# Kernel-behavior regression: empty-KV varlen attention
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.bfloat16, id="bf16"),
    ],
)
def test_attention_empty_varlen_kv_returns_zero_neg_inf(dtype: torch.dtype):
    """Pins the FA varlen kernel's behavior on which our
    ``clamp_empty_varlen_kv`` decision rests: when
    ``cumulative_seqlen_KV=[0, 0]`` (KV padded out to a fixed length but
    the varlen offset says zero real positions), the kernel should return
    ``out=0`` and ``lse=-inf`` so the merge weight is zero by
    construction and no clamp/mask workaround is needed.

      - bf16: kernel returns ``0`` / ``-inf`` cleanly — we skip the
        clamp+mask in ``three_way_attention_with_kv_cache`` for bf16.
      - fp32: kernel returns NaN — we apply the clamp+mask workaround
        (``cumulative_seqlen_KV = [0, max(real, 1)]`` plus a
        ``MergeAttentionsBridge``-wrapped ``torch.where(has_*, lse, -inf)``).

    This test fixes the current kernel behavior; if it XPASSes for fp32,
    the workaround can be retired (see
    ``OmniMoTCausalModel._resolve_clamp_empty_varlen_kv``).
    """

    if _CLAMP_EMPTY_VARLEN_KV:
        return

    device = torch.device("cuda")
    S_q = 4  # non-zero query length
    S_kv_padded = 4  # K/V tensors padded to this length
    H = 2
    D = 8

    q = torch.randn(1, S_q, H, D, device=device, dtype=dtype)
    k = torch.randn(1, S_kv_padded, H, D, device=device, dtype=dtype)
    v = torch.randn(1, S_kv_padded, H, D, device=device, dtype=dtype)

    cum_q = torch.tensor([0, S_q], device=device, dtype=torch.int32)
    cum_kv_empty = torch.tensor([0, 0], device=device, dtype=torch.int32)

    out, lse = imaginaire_attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=cum_q,
        cumulative_seqlen_KV=cum_kv_empty,
        max_seqlen_Q=S_q,
        max_seqlen_KV=S_kv_padded,
        return_lse=True,
    )

    assert (out == 0).all().item(), (
        f"empty-KV varlen output should be exactly zero; got "
        f"max |out| = {out.abs().max().item():.6g} "
        f"(any NaN: {torch.isnan(out).any().item()})"
    )
    assert torch.isneginf(lse).all().item(), (
        f"empty-KV varlen LSE should be -inf everywhere; got "
        f"min lse = {lse.min().item():.6g}, max lse = {lse.max().item():.6g} "
        f"(any NaN: {torch.isnan(lse).any().item()})"
    )


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_attention_empty_varlen_self_attention_returns_zero(dtype: torch.dtype):
    """Pure varlen self-attention with Q, K, V all zero-length (offset
    ``[0, 0]``) but padded to a fixed buffer size should return a zero
    output, not NaN.

    Mirrors the text self-attention call inside
    ``three_way_attention_with_kv_cache`` when no caption is present:
    Q / K / V come from the pack's ``causal_seq`` slot (padded to a
    constant length for compile stability), and ``text_kv_offsets`` is
    ``[0, 0]`` (or, if ``clamp_empty_varlen_kv`` is True, ``[0, 1]`` —
    this test pins the unclamped case).  The output is passed straight
    to ``from_mode_splits``; positions beyond the real range are
    discarded, but they must be *zero*, not NaN, to avoid contaminating
    downstream tensors.  The LSE is not requested here — the text SA
    output does not feed ``merge_attentions``, so its LSE value is
    irrelevant.
    """

    device = torch.device("cuda")
    S_padded = 8
    H = 2
    D = 8

    q = torch.randn(1, S_padded, H, D, device=device, dtype=dtype)
    k = torch.randn(1, S_padded, H, D, device=device, dtype=dtype)
    v = torch.randn(1, S_padded, H, D, device=device, dtype=dtype)

    cum_empty = torch.tensor([0, 0], device=device, dtype=torch.int32)

    out = imaginaire_attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=cum_empty,
        cumulative_seqlen_KV=cum_empty,
        max_seqlen_Q=S_padded,
        max_seqlen_KV=S_padded,
        is_causal=True,
        causal_type=CausalType.TopLeft,
    )
    assert isinstance(out, torch.Tensor)

    assert (out == 0).all().item(), (
        f"empty-varlen self-attention output should be zero everywhere; "
        f"got max |out| = {out.abs().max().item():.6g} "
        f"(any NaN: {torch.isnan(out).any().item()})"
    )
