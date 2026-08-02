# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FlexAttention implementation of the generation-tower self-attention.

``three_way_attention`` (see ``attention.py``) computes the generator's
self-attention term (``full_sa``) over the packed GEN tokens and later merges it
by log-sum-exp with the gen->und cross-attention (``full_ca``). In the dense
case each GEN token attends to every GEN token *within its own sample*
(block-diagonal, bidirectional).

This reproduces that dense term with a single FlexAttention call in three explicit
phases:

1. :func:`build_flex_metadata` assembles the per-token :class:`FlexMetadata`
   from caller-supplied fields (packed ``sample_id``, plus the multiview
   supertoken fields ``frame_id`` / ``view_id`` / ``is_noisy`` /
   ``cond_type_id``).
2. :func:`build_block_mask` derives the ``BlockMask`` from that metadata, once per
   forward and outside the decoder layers.
3. :func:`flex_attention_varlen` runs the attention with that mask.

The ``BlockMask`` is the only artifact the attention path carries. The per-token
fields exist solely to define the ``mask_mod``, so a caller that just wants the mask
for a packed multiview batch should use :func:`build_multiview_block_mask`, which
runs phases 1 and 2 and returns the mask alone.

When the multiview fields are populated, :func:`build_block_mask` enforces the
supertoken rules: conditioning tokens attend only to conditioning tokens of the
same type in the same (frame, view) and never to noisy tokens; noisy tokens
attend to all noisy tokens within the sample and, additionally, to every
conditioning token in the same (frame, view). Richer patterns can be added later
by populating extra metadata fields and extending the ``mask_mod`` instead of
hand-rolling new varlen bookkeeping.

LSE convention
--------------
``flex_attention(..., return_aux=AuxRequest(lse=True))`` returns the log-sum-exp of the scaled
scores in **natural log** with default scale ``1/sqrt(head_dim)`` and layout
``[B, H, S]`` -- identical to ``cosmos_framework.model.attention(..., return_lse=True)`` once
transposed to the heads-last ``[B, S, H]`` layout. This makes the output
directly mergeable with ``full_ca`` via ``merge_attentions``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# FlexAttention works at block granularity; inputs and metadata must already be a
# multiple of this, which callers arrange when packing the GEN stream.
FLEX_BLOCK_SIZE = 128

# ``dynamic=False`` specialises one kernel per (block-aligned) shape and reuses
# it across steps. Only the BlockMask *data* changes per step, which does not
# trigger recompilation.
_COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)

# Compiling ``create_block_mask`` here replaces its deprecated ``_compile=True``
# flag, which emits a ``DeprecationWarning`` that Dynamo cannot trace when the
# mask is built inside a compiled (and activation-checkpointed) decoder layer.
# The compiled form is also what keeps the build fused: the eager path
# materialises a dense ``[Q_LEN, KV_LEN]`` mask tensor.
_COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=False)

# A FlexAttention mask predicate: (b, h, q_idx, kv_idx) -> bool tensor.
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FlexMetadata:
    """Per-GEN-token metadata that drives the flex block mask.

    Each field is a 1-D ``[seq_len]`` tensor (``seq_len`` being the block-padded GEN
    sequence length), aligned with the packed token order. All are ``int64`` except
    ``is_noisy``, which is ``bool``. Padding positions use the sentinel ``-1``
    (``False`` for ``is_noisy``) so real queries never attend to padding and padded
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

    This type is an intermediate: :func:`build_block_mask` folds it into a
    :class:`BlockMask` and only that mask travels on to
    :func:`flex_attention_varlen`. The tensors themselves stay reachable through the
    mask's ``mask_mod`` closure, which the kernel evaluates on partially-masked
    blocks, so nothing here can be freed early.
    """

    seq_len: int
    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_noisy: torch.Tensor
    cond_type_id: torch.Tensor


def _to_flex_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[1, S, H, D]`` to the FlexAttention layout ``[1, H, S, D]``.

    ``S`` must already be a multiple of ``FLEX_BLOCK_SIZE`` (the caller pre-pads).
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

    Returns a ``[seq_len]`` ``int64`` tensor.
    """
    real_count = full_q_offsets[-1]  # 0-dim tensor; no .item() / host sync.
    positions = torch.arange(seq_len, device=device)  # [seq_len]
    sample_id = torch.searchsorted(full_q_offsets[1:].contiguous(), positions, right=True)  # [seq_len]
    return torch.where(positions < real_count, sample_id, -1).to(torch.long)  # [seq_len]


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


def build_multiview_flex_metadata(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    token_shapes: Sequence[tuple[int, ...]],
    condition_masks: Sequence[torch.Tensor],
    num_vision_items_per_sample: list[int],
    num_views_per_vision_item: list[int],
    device: torch.device,
) -> FlexMetadata:
    """Build GEN-token metadata for camera-major multiview transfer items.

    Each vision item contributes ``latent_t * patch_h * patch_w`` tokens laid out
    view-outer, frame-inner, spatial-innermost, matching the dataset, which
    concatenates the per-camera clips along the latent time axis.
    ``condition_masks[i]`` indexes that same camera-major latent axis, with a
    non-zero entry marking a conditioning (clean) frame.

    Two invariants are checked because violating either corrupts the mask
    silently rather than failing: the item token counts have to add up to the
    packed GEN token count (``full_q_offsets[-1]``), and all items of one sample
    have to share a ``(num_views, frames_per_view)`` grid, since conditioning
    tokens reach noisy tokens by matching ``(frame, view)``.

    Shape notation used here and in the inline annotations below:

    * ``num_samples = len(num_vision_items_per_sample)``;
    * ``num_items = sum(num_vision_items_per_sample)``, the flattened item count;
    * for item ``i``, ``token_shapes[i] = (latent_t, patch_h, patch_w)`` with
      ``latent_t = num_views * frames_per_view`` and
      ``spatial_tokens = patch_h * patch_w``, so the item owns
      ``item_tokens = latent_t * spatial_tokens`` GEN tokens;
    * ``real_token_count = sum(item_tokens)``, which must equal
      ``full_q_offsets[-1]`` and cannot exceed ``seq_len``.

    Args:
        seq_len: block-padded GEN sequence length. Every returned field is ``[seq_len]``.
        full_q_offsets: cumulative per-sample GEN offsets, ``[num_samples + 1]``;
            ``full_q_offsets[-1]`` is the real (unpadded) GEN token count.
        token_shapes: one ``(latent_t, patch_h, patch_w)`` triple per item,
            ``num_items`` entries.
        condition_masks: one mask per item, each ``[latent_t]`` over the camera-major
            latent axis; a non-zero entry marks a conditioning (clean) frame.
        num_vision_items_per_sample: items owned by each sample, ``num_samples`` entries.
        num_views_per_vision_item: cameras per item, ``num_items`` entries.
        device: device for the returned tensors.

    Returns:
        :class:`FlexMetadata` whose five per-token fields are each ``[seq_len]``: the
        ``real_token_count`` real tokens in packed order, followed by ``-1`` sentinels
        (``False`` for ``is_noisy``) across the trailing pad.

    Raises:
        ValueError: if the per-item sequences disagree on ``num_items``, if a
            ``latent_t`` is not divisible by its ``num_views``, if one sample's items
            disagree on the ``(num_views, frames_per_view)`` grid, if a condition mask
            length does not match its ``latent_t``, or if ``real_token_count`` does not
            match ``full_q_offsets[-1]`` or exceeds ``seq_len``.
    """
    num_items = sum(num_vision_items_per_sample)
    if not (len(token_shapes) == len(condition_masks) == len(num_views_per_vision_item) == num_items):
        raise ValueError(
            "Multiview FlexAttention metadata must align with flattened vision items: "
            f"got {len(token_shapes)} token shapes, {len(condition_masks)} condition masks, "
            f"{len(num_views_per_vision_item)} view counts, and {num_items} expected items."
        )

    frame_ids: list[torch.Tensor] = []
    view_ids: list[torch.Tensor] = []
    noisy_flags: list[torch.Tensor] = []
    cond_type_ids: list[torch.Tensor] = []
    item_idx = 0
    for sample_num_items in num_vision_items_per_sample:
        sample_grid: tuple[int, int] | None = None
        for cond_type in range(sample_num_items):
            latent_t, patch_h, patch_w = token_shapes[item_idx]
            num_views = num_views_per_vision_item[item_idx]
            if num_views < 1 or latent_t % num_views != 0:
                raise ValueError(
                    f"Vision item {item_idx} has latent_t={latent_t}, which is not divisible by num_views={num_views}."
                )

            frames_per_view = latent_t // num_views
            # Noisy tokens see conditioning tokens by matching (frame, view), so an item
            # on a different grid would pair up unrelated frames and leave the extra
            # cameras unconditioned without any complaint from the mask.
            if sample_grid is None:
                sample_grid = (num_views, frames_per_view)
            elif sample_grid != (num_views, frames_per_view):
                raise ValueError(
                    "All vision items of a sample must share the same (num_views, frames_per_view) grid: "
                    f"item {item_idx} has {(num_views, frames_per_view)}, expected {sample_grid}."
                )
            spatial_tokens = patch_h * patch_w
            # Frame index cycles within each view; view index is constant across a view's
            # whole frame run. Both expand to one entry per token: [item_tokens].
            frame_ids.append(
                torch.arange(frames_per_view, device=device).repeat(num_views).repeat_interleave(spatial_tokens)
            )  # [item_tokens]
            view_ids.append(
                torch.arange(num_views, device=device).repeat_interleave(frames_per_view * spatial_tokens)
            )  # [item_tokens]

            condition_mask = condition_masks[item_idx].to(device=device, dtype=torch.bool)  # [latent_t]
            if condition_mask.numel() != latent_t:
                raise ValueError(f"Condition mask {item_idx} has {condition_mask.numel()} frames, expected {latent_t}.")
            # Per-frame flag -> per-token flag, every token of a frame sharing the frame's state.
            is_conditioning = condition_mask.reshape(latent_t).repeat_interleave(spatial_tokens)  # [item_tokens], bool
            noisy_flags.append(~is_conditioning)  # [item_tokens], bool
            cond_type_ids.append(
                torch.where(
                    is_conditioning,
                    torch.full(is_conditioning.shape, cond_type, device=device, dtype=torch.long),
                    torch.full(is_conditioning.shape, -1, device=device, dtype=torch.long),
                )
            )  # [item_tokens]
            item_idx += 1

    frame_id = torch.cat(frame_ids)  # [real_token_count]
    view_id = torch.cat(view_ids)  # [real_token_count]
    is_noisy = torch.cat(noisy_flags)  # [real_token_count], bool
    cond_type_id = torch.cat(cond_type_ids)  # [real_token_count]
    real_token_count = frame_id.shape[0]
    # These counts come from token_shapes while the offsets come from the packer's full
    # splits. If they disagree, _build_gen_sample_ids draws the padding boundary somewhere
    # else than this metadata does, which mislabels real tokens as padding or padding as
    # conditioning tokens. Reading the last offset costs one device sync per forward; this
    # runs outside the compiled decoder layers.
    packed_token_count = int(full_q_offsets[-1])
    if real_token_count != packed_token_count:
        raise ValueError(
            f"Multiview metadata covers {real_token_count} GEN tokens but the pack holds "
            f"{packed_token_count}; token_shapes and the packed full-attention splits disagree."
        )
    if real_token_count > seq_len:
        raise ValueError(f"Multiview metadata has {real_token_count} tokens, exceeding GEN sequence length {seq_len}.")

    pad = seq_len - real_token_count
    if pad:
        sentinel = torch.full((pad,), -1, device=device, dtype=torch.long)  # [pad]
        frame_id = torch.cat((frame_id, sentinel))  # [seq_len]
        view_id = torch.cat((view_id, sentinel))  # [seq_len]
        cond_type_id = torch.cat((cond_type_id, sentinel))  # [seq_len]
        is_noisy = torch.cat((is_noisy, torch.zeros(pad, device=device, dtype=torch.bool)))  # [seq_len], bool

    return build_flex_metadata(
        seq_len,
        sample_id=_build_gen_sample_ids(full_q_offsets, seq_len, device),  # [seq_len]
        frame_id=frame_id,  # [seq_len]
        view_id=view_id,  # [seq_len]
        is_noisy=is_noisy,  # [seq_len], bool
        cond_type_id=cond_type_id,  # [seq_len]
    )


def _check_block_aligned(seq_len: int) -> None:
    """Raise if ``seq_len`` is not a multiple of :data:`FLEX_BLOCK_SIZE`."""
    if seq_len % FLEX_BLOCK_SIZE != 0:
        raise ValueError(
            f"FlexAttention needs a block-aligned GEN sequence length, got {seq_len}, which is not "
            f"a multiple of {FLEX_BLOCK_SIZE}. Callers pad the GEN stream at packing time via "
            "full_seq_alignment in build_packed_sequence."
        )


def build_block_mask(
    metadata: FlexMetadata,
    device: torch.device,
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` from precomputed flex metadata.

    Uses the multiview supertoken ``mask_mod``; the multiview fields
    (``frame_id`` / ``view_id`` / ``is_noisy`` / ``cond_type_id``) must be
    populated on ``metadata``.

    Call this from outside the decoder layers and hand the returned mask to the
    attention path. ``create_block_mask`` enters a ``TorchFunctionMode``, and
    Dynamo rejects that global side effect inside the ``torch.utils.checkpoint`` HOP, so
    the mask cannot be built lazily from within a compiled, activation-checkpointed
    layer. Building it once per forward also avoids rebuilding the identical mask in
    every layer.

    The mask *data* depends on the per-step packing (``metadata.sample_id`` etc.)
    and is rebuilt every call; because ``metadata.seq_len`` is block-aligned, the
    compiled ``create_block_mask`` / attention kernels are still reused.

    Raises:
        ValueError: if ``metadata.seq_len`` is not a multiple of ``FLEX_BLOCK_SIZE``.
    """
    _check_block_aligned(metadata.seq_len)
    mask_mod = _multiview_mask_mod_factory(metadata)
    return _COMPILED_CREATE_BLOCK_MASK(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=metadata.seq_len,
        KV_LEN=metadata.seq_len,
        device=device,
        BLOCK_SIZE=FLEX_BLOCK_SIZE,
    )


def build_multiview_block_mask(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    token_shapes: Sequence[tuple[int, ...]],
    condition_masks: Sequence[torch.Tensor],
    num_vision_items_per_sample: list[int],
    num_views_per_vision_item: list[int],
    device: torch.device,
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` for camera-major multiview items.

    The entry point for the packed-batch path: it runs
    :func:`build_multiview_flex_metadata` and :func:`build_block_mask` back to back so
    the caller only handles the mask. See those two for the token layout the metadata
    encodes and for why the mask has to be built outside the decoder layers.

    The two stages stay separately callable because the metadata layout is checked on
    CPU in the unit tests, while building the mask goes through a compiled
    ``create_block_mask``.
    """
    metadata = build_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        token_shapes=token_shapes,
        condition_masks=condition_masks,
        num_vision_items_per_sample=num_vision_items_per_sample,
        num_views_per_vision_item=num_views_per_vision_item,
        device=device,
    )
    return build_block_mask(metadata, device)


def flex_attention_varlen(
    full_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    block_mask: BlockMask,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dense (per-sample, bidirectional) GEN-tower self-attention via FlexAttention.

    Drop-in replacement for the dense ``full_sa`` branch of
    ``three_way_attention``: each GEN token attends to every GEN token within its
    own packed sample.

    ``N_full`` must be a multiple of ``FLEX_BLOCK_SIZE``: the GEN stream is padded
    to that multiple at packing time (see ``full_seq_alignment`` in
    ``build_packed_sequence``), so this function never has to re-pad q/k/v and the
    metadata in every layer.

    Args:
        full_q: GEN queries, ``[1, N_full, heads, head_dim]`` (may include
            trailing pack padding beyond the last real token).
        full_k: GEN keys, ``[1, N_full, kv_heads, head_dim]``.
        full_v: GEN values, ``[1, N_full, kv_heads, head_dim]``.
        block_mask: the GEN-tower mask, precomputed outside the decoder layers via
            :func:`build_block_mask` (or :func:`build_multiview_block_mask`); it must
            have been built for the same ``N_full``.
        return_lse: when ``True`` also return the log-sum-exp, needed to merge
            this term with ``full_ca`` via ``merge_attentions``.

    Returns:
        ``full_sa`` of shape ``[1, N_full, heads, head_dim]`` -- the heads-last
        layout expected by ``merge_attentions``, with the sequence length
        matching ``full_q`` (pack padding preserved) so the result lines up with
        ``full_ca``. When ``return_lse`` is ``True``, returns the tuple
        ``(full_sa, full_sa_lse)`` where ``full_sa_lse`` has shape
        ``[1, N_full, heads]``.

    Raises:
        ValueError: if ``N_full`` is not a multiple of ``FLEX_BLOCK_SIZE``, or if
            ``block_mask`` was built for a different sequence length.
    """
    from torch.nn.attention.flex_attention import AuxRequest

    seq_len = full_q.shape[1]
    num_q_heads = full_q.shape[2]
    num_kv_heads = full_k.shape[2]

    _check_block_aligned(seq_len)
    # BlockMask.shape is (*batch_dims, Q_LEN, KV_LEN), holding the lengths the mask was
    # built for. A mask carried over from a differently-shaped pack would mask the wrong
    # tokens rather than fail, so the lengths are compared before the kernel sees either.
    mask_q_len, mask_kv_len = block_mask.shape[-2:]
    if (mask_q_len, mask_kv_len) != (seq_len, seq_len):
        raise ValueError(
            f"block_mask covers Q_LEN={mask_q_len}, KV_LEN={mask_kv_len}, but the GEN sequence is "
            f"{seq_len} tokens; the mask must be built from the same pack that produced q/k/v."
        )

    q = _to_flex_layout(full_q)  # [1,num_q_heads,N_full,head_dim]
    k = _to_flex_layout(full_k)  # [1,num_kv_heads,N_full,head_dim]
    v = _to_flex_layout(full_v)  # [1,num_kv_heads,N_full,head_dim]

    if return_lse:
        # return_aux rather than the deprecated return_lse: the latter records a
        # FutureWarning in a module-level set, which Dynamo rejects as an unsafe
        # side effect inside the activation-checkpointing HOP.
        attn_out, aux = _COMPILED_FLEX_ATTENTION(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=num_q_heads != num_kv_heads,
            return_aux=AuxRequest(lse=True),
        )  # attn_out: [1,num_q_heads,N_full,head_dim], aux.lse: [1,num_q_heads,N_full]
        # Convert to the heads-last layout ([1,S,H,D] / [1,S,H]) that
        # merge_attentions and from_mode_splits expect.
        return _from_flex_layout(attn_out), _from_flex_layout(aux.lse)  # [1,N_full,heads,head_dim], [1,N_full,heads]

    attn_out = _COMPILED_FLEX_ATTENTION(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=num_q_heads != num_kv_heads,
    )  # attn_out: [1,num_q_heads,N_full,head_dim]
    # Convert to the heads-last layout ([1,S,H,D]) that from_mode_splits expects.
    return _from_flex_layout(attn_out)  # [1,N_full,heads,head_dim]
