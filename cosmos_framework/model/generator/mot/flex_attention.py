# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FlexAttention implementation of the generation-tower self-attention.

``three_way_attention`` (see ``attention.py``) computes the generator's
self-attention term (``full_sa``) over the packed GEN tokens and later merges it
by log-sum-exp with the gen->und cross-attention (``full_ca``). In the dense
case each GEN token attends to every GEN token *within its own sample*
(block-diagonal, bidirectional).

This reproduces that dense term with a single FlexAttention call in two explicit
phases:

1. :func:`build_flex_metadata` assembles the per-token :class:`FlexMetadata`
   from caller-supplied fields (packed ``sample_id``, plus the multiview
   supertoken fields ``frame_id`` / ``view_id`` / ``is_noisy`` /
   ``cond_type_id``). The caller builds this and passes it into
   :func:`flex_attention_varlen`.
2. :func:`flex_attention_varlen` derives the ``BlockMask`` from that metadata
   (via :func:`build_block_mask`) and runs the attention.

When the multiview fields are populated, :func:`build_block_mask` enforces the
supertoken rules: conditioning tokens attend only to conditioning tokens of the
same type in the same (frame, view) and never to noisy tokens; noisy tokens
attend to all noisy tokens within the sample and, additionally, to every
conditioning token in the same (frame, view). Richer patterns can be added later
by populating extra metadata fields and extending the ``mask_mod`` instead of
hand-rolling new varlen bookkeeping.

LSE convention
--------------
``flex_attention(..., return_lse=True)`` returns the log-sum-exp of the scaled
scores in **natural log** with default scale ``1/sqrt(head_dim)`` and layout
``[B, H, S]`` -- identical to ``cosmos_framework.model.attention(..., return_lse=True)`` once
transposed to the heads-last ``[B, S, H]`` layout. This makes the output
directly mergeable with ``full_ca`` via ``merge_attentions``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# FlexAttention works at block granularity; the GEN sequence length is expected
# to be pre-padded to a multiple of this by the caller so the compiled kernel
# sees stable, block-aligned shapes.
_FLEX_BLOCK_SIZE = 128

# ``dynamic=False`` specialises one kernel per (block-aligned) shape and reuses
# it across steps. Only the BlockMask *data* changes per step, which does not
# trigger recompilation.
_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)

# A FlexAttention mask predicate: (b, h, q_idx, kv_idx) -> bool tensor.
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FlexMetadata:
    """Per-GEN-token metadata that drives the flex block mask.

    Each field is a 1-D ``int64`` tensor of length ``seq_len`` (the block-padded
    GEN sequence length), aligned with the packed token order. Padding positions
    use the sentinel ``-1`` so real queries never attend to padding and padded
    queries attend only to padding (no empty-softmax NaN).

    All fields are required. They describe the multiview supertoken layout and
    drive the ``mask_mod`` in :func:`build_block_mask`:

    * ``sample_id``: packed sample index per token; yields the block-diagonal
      same-sample constraint every rule requires.
    * ``frame_id`` / ``view_id``: per-token frame and view indices.
    * ``is_noisy``: ``bool`` tensor, ``True`` for noisy (visual) tokens and
      ``False`` for conditioning tokens.
    * ``cond_type_id``: conditioning-stream type per token (e.g. 0 = type A,
      1 = type B); ``-1`` for noisy tokens.

    The mask enforces (all four Q/K quadrants):

    * conditioning Q -> conditioning K: same ``(frame, view, cond_type)``;
    * conditioning Q -> noisy K: never;
    * noisy Q -> noisy K: full (bidirectional) within the sample;
    * noisy Q -> conditioning K: same ``(frame, view)`` (any cond type).
    """

    seq_len: int
    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_noisy: torch.Tensor
    cond_type_id: torch.Tensor


def _to_flex_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[1, S, H, D]`` to the FlexAttention layout ``[1, H, S, D]``.

    ``S`` must already be a multiple of ``_FLEX_BLOCK_SIZE`` (the caller pre-pads).
    """
    return x.transpose(1, 2).contiguous()  # [1,H,S,D]


def _from_flex_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert a FlexAttention output back to the heads-last layout.

    Inverse of :func:`_to_flex_layout`: swaps the heads and sequence axes, so
    ``[1, H, S, D] -> [1, S, H, D]`` (attention output) and ``[1, H, S] ->
    [1, S, H]`` (LSE) both work.
    """
    return x.transpose(1, 2).contiguous()  # [1,S,H,...]


def _build_gen_sample_ids(
    full_q_offsets: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-GEN-token sample id in packed order (``-1`` for padding positions).

    ``full_q_offsets`` is the cumulative per-sample offset array for the packed
    GEN segment (shape ``[num_samples + 1]``), so ``searchsorted`` maps every
    position to its sample. Positions at or beyond the last real token
    (``full_q_offsets[-1]``) are marked ``-1`` so that (a) real queries never
    attend to padding and (b) padded queries attend only to other padding,
    avoiding an empty-softmax NaN. Uses only tensor ops (no host sync) so it
    stays inside a compiled graph.
    """
    real_count = full_q_offsets[-1]  # 0-dim tensor; no .item() / host sync.
    positions = torch.arange(seq_len, device=device)
    sample_id = torch.searchsorted(full_q_offsets[1:].contiguous(), positions, right=True)
    return torch.where(positions < real_count, sample_id, -1).to(torch.long)


def _multiview_mask_mod_factory(metadata: FlexMetadata) -> MaskMod:
    """Return the multiview supertoken ``mask_mod``.

    All rules are gated on ``same_sample`` (block-diagonal packing). On top of
    that, using the per-token frame/view/type metadata (all four Q/K quadrants):

    * conditioning Q attends only to conditioning K of the **same conditioning
      type** in the **same (frame, view)**;
    * conditioning Q -> noisy K: never (no rule fires for this quadrant);
    * noisy Q attends to **all** noisy K (bidirectional, within the sample) and,
      in addition, to every conditioning K in the **same (frame, view)**
      regardless of conditioning type.

    Padding positions carry ``sample_id == -1`` (and ``-1`` in the other fields),
    so real queries never see them and padded queries attend only to padding.
    """
    sample_id = metadata.sample_id
    frame_id = metadata.frame_id
    view_id = metadata.view_id
    is_noisy = metadata.is_noisy
    cond_type_id = metadata.cond_type_id

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        same_sample = sample_id[q_idx] == sample_id[kv_idx]
        same_fv = (frame_id[q_idx] == frame_id[kv_idx]) & (view_id[q_idx] == view_id[kv_idx])
        same_cond_type = cond_type_id[q_idx] == cond_type_id[kv_idx]
        q_noisy = is_noisy[q_idx]
        k_noisy = is_noisy[kv_idx]

        # conditioning Q -> conditioning K: same (frame, view, cond type).
        cond_to_cond = (~q_noisy) & (~k_noisy) & same_fv & same_cond_type
        # noisy Q -> noisy K: full within the sample.
        noisy_to_noisy = q_noisy & k_noisy
        # noisy Q -> conditioning K: same (frame, view), any cond type.
        noisy_to_cond = q_noisy & (~k_noisy) & same_fv

        return same_sample & (cond_to_cond | noisy_to_noisy | noisy_to_cond)

    return mask_mod


def build_flex_metadata(
    seq_len: int,
    *,
    sample_id: torch.Tensor,
    frame_id: torch.Tensor,
    view_id: torch.Tensor,
    is_noisy: torch.Tensor,
    cond_type_id: torch.Tensor,
) -> FlexMetadata:
    """Assemble per-GEN-token flex metadata from precomputed per-token fields.

    The caller supplies ``sample_id`` (packed sample per token; e.g. via
    :func:`_build_gen_sample_ids`) together with the multiview supertoken fields
    ``frame_id`` / ``view_id`` / ``is_noisy`` / ``cond_type_id`` (each
    ``[seq_len]``, ``-1`` for padding), which drive the multiview ``mask_mod``
    in :func:`build_block_mask`.
    """
    return FlexMetadata(
        seq_len=seq_len,
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_noisy=is_noisy,
        cond_type_id=cond_type_id,
    )


def build_block_mask(
    metadata: FlexMetadata,
    device: torch.device,
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` from precomputed flex metadata.

    Uses the multiview supertoken ``mask_mod``; the multiview fields
    (``frame_id`` / ``view_id`` / ``is_noisy`` / ``cond_type_id``) must be
    populated on ``metadata``.

    The mask *data* depends on the per-step packing (``metadata.sample_id`` etc.)
    and is rebuilt every call; because ``metadata.seq_len`` is block-aligned, the
    compiled ``create_block_mask`` / attention kernels are still reused.
    """
    mask_mod = _multiview_mask_mod_factory(metadata)
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=metadata.seq_len,
        KV_LEN=metadata.seq_len,
        device=device,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
        _compile=True,
    )


def flex_attention_varlen(
    full_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    metadata: FlexMetadata,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dense (per-sample, bidirectional) GEN-tower self-attention via FlexAttention.

    Drop-in replacement for the dense ``full_sa`` branch of
    ``three_way_attention``: each GEN token attends to every GEN token within its
    own packed sample.

    The GEN sequence length ``N_full`` must already be a multiple of
    ``_FLEX_BLOCK_SIZE`` (the caller pre-pads); this is asserted rather than
    padded here.

    Args:
        full_q: GEN queries, ``[1, N_full, heads, head_dim]`` (may include
            trailing pack padding beyond the last real token).
        full_k: GEN keys, ``[1, N_full, kv_heads, head_dim]``.
        full_v: GEN values, ``[1, N_full, kv_heads, head_dim]``.
        metadata: precomputed per-token :class:`FlexMetadata` (see
            :func:`build_flex_metadata`); its ``seq_len`` must match ``N_full``.
        return_lse: when ``True`` also return the log-sum-exp, needed to merge
            this term with ``full_ca`` via ``merge_attentions``.

    Returns:
        ``full_sa`` of shape ``[1, N_full, heads, head_dim]`` -- the heads-last
        layout expected by ``merge_attentions``, with the sequence length
        matching ``full_q`` (pack padding preserved) so the result lines up with
        ``full_ca``. When ``return_lse`` is ``True``, returns the tuple
        ``(full_sa, full_sa_lse)`` where ``full_sa_lse`` has shape
        ``[1, N_full, heads]``.
    """
    seq_len = full_q.shape[1]
    num_q_heads = full_q.shape[2]
    num_kv_heads = full_k.shape[2]
    device = full_q.device

    # Padding to the flex block size is assumed to have been done upstream.
    assert seq_len % _FLEX_BLOCK_SIZE == 0, (
        f"flex_attention_varlen expects the GEN sequence length to be pre-padded to a multiple "
        f"of _FLEX_BLOCK_SIZE={_FLEX_BLOCK_SIZE}, got {seq_len}."
    )
    assert metadata.seq_len == seq_len, (
        f"FlexMetadata.seq_len ({metadata.seq_len}) must match the GEN sequence length ({seq_len})."
    )

    q = _to_flex_layout(full_q)  # [1,num_q_heads,N_full,head_dim]
    k = _to_flex_layout(full_k)  # [1,num_kv_heads,N_full,head_dim]
    v = _to_flex_layout(full_v)  # [1,num_kv_heads,N_full,head_dim]

    # Build the block mask from the precomputed per-token flex metadata.
    block_mask = build_block_mask(metadata, device)

    if return_lse:
        attn_out, lse = _COMPILED_FLEX_ATTENTION(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=num_q_heads != num_kv_heads,
            return_lse=True,
        )  # attn_out: [1,num_q_heads,N_full,head_dim], lse: [1,num_q_heads,N_full]
        # Convert to the heads-last layout ([1,S,H,D] / [1,S,H]) that
        # merge_attentions and from_mode_splits expect. Sequence length is unchanged.
        return _from_flex_layout(attn_out), _from_flex_layout(lse)  # [1,N_full,heads,head_dim], [1,N_full,heads]

    attn_out = _COMPILED_FLEX_ATTENTION(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=num_q_heads != num_kv_heads,
        return_lse=False,
    )  # attn_out: [1,num_q_heads,N_full,head_dim]
    # Convert to the heads-last layout ([1,S,H,D]) that from_mode_splits expects.
    return _from_flex_layout(attn_out)  # [1,N_full,heads,head_dim]
