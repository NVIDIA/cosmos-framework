# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FlexAttention implementation of the generation tower's attention.

``two_way_attention`` (see ``attention.py``) computes the generator's attention over
every token of its own sample -- the UND (understanding / caption) tokens and the GEN
tokens together. Its dense path does that in one kernel because "attend to the whole
sample" is plain block-diagonal, which flash can express. The multiview supertoken
rules restrict the GEN->GEN quadrant only, and the resulting combined mask is no
longer block-diagonal, so this module expresses it as a single FlexAttention call over
the concatenated ``[UND | GEN]`` key/value stream.

One fused call rather than two merged ones is deliberate. Splitting the quadrants into
separate kernels and recombining them with ``merge_attentions`` gives an identical
forward, but NATTEN's merge repairs the backward by writing the merged ``(out, lse)``
back into the buffers the attention kernel saved, and the heads-last conversion below
copies those buffers, so the repair silently misses and the GEN branch backpropagates
against its local softmax normalization. Fusing needs no LSE, no merge, and no such
contract; ``flex_attention_test`` checks the fused gradients against a dense reference.

This runs in three explicit phases:

1. :class:`FlexMetadata` holds the per-token fields (packed ``sample_id``, plus the
   multiview supertoken fields ``frame_id`` / ``view_id`` / ``is_noisy`` /
   ``cond_type_id``), covering the ``[UND | GEN]`` key stream;
   :func:`build_multiview_flex_metadata` derives them from a packed batch.
2. :func:`build_block_mask` derives the ``BlockMask`` from that metadata, once per
   forward and outside the decoder layers. The mask is rectangular: GEN tokens are
   the queries, ``[UND | GEN]`` the keys.
3. :func:`flex_attention` runs the attention with that mask.

The ``BlockMask`` is the only artifact the attention path carries. The per-token
fields exist solely to define the ``mask_mod``, so a caller that just wants the mask
for a packed multiview batch should use :func:`build_multiview_block_mask`, which
runs phases 1 and 2 and returns the mask alone.

Which kernels this lowers onto is :func:`resolve_flex_backend`'s answer, taken once per
forward: FlashAttention-4 where it is installed and supported, FlexAttention's Triton
kernels otherwise (``flex_attention.backend`` in the model config pins either). The FA4
mask has to be built at :func:`flash_backend_block_size` and its streams padded to that
coarser query block, so the returned :class:`FlexBackend` carries the block size and the
padding multiple alongside the kernel options rather than leaving the three to be matched
up by hand. ``flex_attention_bench`` times both backends and documents the install.

When the multiview fields are populated, :func:`build_block_mask` enforces the
supertoken rules: conditioning tokens attend only to conditioning tokens of the
same type in the same (frame, view) and never to noisy tokens; noisy tokens
attend to the noisy tokens of their sample that the ``noisy_attention_scope``
admits -- by default all of them -- and, additionally, to every conditioning
token in the same (frame, view). Every GEN token, conditioning or
noisy, attends to every UND token of its own sample, which is what the dense
gen->und pass does. Richer patterns can be added later by populating extra
metadata fields and extending the ``mask_mod`` instead of hand-rolling new varlen
bookkeeping.

LSE convention
--------------
torch's ``flex_attention(..., return_aux=AuxRequest(lse=True))`` returns the log-sum-exp of the scaled
scores in **natural log** with default scale ``1/sqrt(head_dim)`` and layout
``[B, H, S]`` -- identical to ``cosmos_framework.model.attention(..., return_lse=True)`` once
transposed to the heads-last ``[B, S, H]`` layout, so a caller that does want to merge
this output with another attention term can. The fused attention path does not: it is a
complete attention, and asking for the LSE would put a gradient for it in the graph,
which is what the FlashAttention-4 backward refuses to lower.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from cosmos_framework.configs.base.defaults.flex_attention import (
    FLEX_BACKEND_PREFERENCES,
    NOISY_ATTENTION_SCOPES,
    NoisyAttentionScope,
)

# ``dynamic=False`` specialises one kernel per (block-aligned) shape and reuses
# it across steps. Only the BlockMask *data* changes per step, which does not
# trigger recompilation. torch's entry point is imported under an alias because this
# module's own, :func:`flex_attention`, takes that name.
_COMPILED_FLEX_ATTENTION = torch.compile(torch_flex_attention, dynamic=False)

# A FlexAttention mask predicate: (b, h, q_idx, kv_idx) -> bool tensor.
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FlexMetadata:
    """Per-key-token metadata that drives the flex block mask.

    Each field is a 1-D ``[seq_len]`` tensor covering the key/value stream the mask
    spans, in packed token order. All are ``int64`` except ``is_noisy``, which is
    ``bool``. Padding positions use the sentinel ``-1`` (``False`` for ``is_noisy``) so
    real queries never attend to padding and padded queries attend only to padding (no
    empty-softmax NaN).

    The key stream is the concatenation ``[UND | GEN]``: ``num_und`` block-padded
    understanding tokens followed by the block-padded GEN tokens. The GEN tokens are
    also the queries, so the mask this drives is rectangular -- ``q_len`` rows,
    ``seq_len`` columns -- and query index ``i`` is key index ``i + num_und``. Leaving
    ``num_und`` at 0 describes a GEN-only stream and yields the square GEN self-attention
    mask, which is what the metadata unit tests exercise.

    All tensor fields are required. They describe the multiview supertoken layout and
    drive the ``mask_mod`` in :func:`build_block_mask`:

    * ``sample_id``: packed sample index per token; yields the block-diagonal
      same-sample constraint every rule requires. UND tokens carry the sample they
      belong to, which is the only field their rule reads.
    * ``frame_id`` / ``view_id``: per-token frame and view indices; ``-1`` on UND
      tokens, which have no place in the camera grid.
    * ``is_noisy``: ``bool`` tensor, ``True`` for noisy (visual) tokens and
      ``False`` for conditioning tokens (and for UND tokens).
    * ``cond_type_id``: conditioning-stream type per token (e.g. 0 = type A,
      1 = type B); ``-1`` for noisy and UND tokens.

    The mask enforces, within a sample:

    * GEN Q -> UND K: always (the gen->und pass, unrestricted);
    * conditioning Q -> conditioning K: same ``(frame, view, cond_type)``;
    * conditioning Q -> noisy K: never;
    * noisy Q -> noisy K: bidirectional over whatever ``NoisyAttentionScope`` admits --
      every noisy token of the sample, the query's own view, or its own view or frame;
    * noisy Q -> conditioning K: same ``(frame, view)`` (any cond type).

    ``noisy_attention_scope`` is the one field that is a choice rather than a description of
    the batch. It rides here so that the two places the predicate is built from -- the
    ``mask_mod`` the kernel calls and the block-level collapsing in
    :func:`build_block_mask` -- cannot be given different answers: a block the collapsing
    calls fully unmasked is one the kernel never calls ``mask_mod`` on at all, so the two
    disagreeing would not raise, it would silently attend across views.

    This type is an intermediate: :func:`build_block_mask` folds it into a
    :class:`BlockMask` and only that mask travels on to
    :func:`flex_attention`. The tensors themselves stay reachable through the
    mask's ``mask_mod`` closure, which the kernel evaluates on partially-masked
    blocks, so nothing here can be freed early.
    """

    seq_len: int
    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_noisy: torch.Tensor
    cond_type_id: torch.Tensor
    num_und: int
    noisy_attention_scope: NoisyAttentionScope

    def __post_init__(self) -> None:
        # The annotation is a Literal, which nothing enforces at runtime, and an unrecognised
        # scope would not fail: the predicate reads it by equality, so a typo leaves both of
        # its gates off and masks as "same_view". A config reaching here has already been
        # validated by attrs, but this type is also built directly, and a silently narrower
        # mask is a worse outcome than a rejected one.
        if self.noisy_attention_scope not in NOISY_ATTENTION_SCOPES:
            raise ValueError(
                f"Unknown noisy_attention_scope {self.noisy_attention_scope!r}; "
                f"expected one of {NOISY_ATTENTION_SCOPES}."
            )

    @property
    def q_len(self) -> int:
        """Query count: the GEN tail of the key stream."""
        return self.seq_len - self.num_und


def _to_flex_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[1, S, H, D]`` to the FlexAttention layout ``[1, H, S, D]``.

    ``S`` must already be a multiple of the mask's block size (the caller pre-pads).
    """
    return x.transpose(1, 2).contiguous()  # [1,H,S,D]


def _from_flex_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert a FlexAttention output back to the heads-last layout.

    Inverse of :func:`_to_flex_layout`: swaps the heads and sequence axes, so
    ``[1, H, S, D] -> [1, S, H, D]`` (attention output) and ``[1, H, S] ->
    [1, S, H]`` (LSE) both work.
    """
    return x.transpose(1, 2).contiguous()  # [1,S,H,...]


def _build_stream_sample_ids(
    offsets: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-token sample id for one packed stream in packed order (``-1`` for padding).

    ``offsets`` is the cumulative per-sample offset array for the stream (shape
    ``[num_samples + 1]``), so ``searchsorted`` maps every position to its sample.
    Positions at or beyond the last real token (``offsets[-1]``) are marked ``-1`` so
    that (a) real queries never attend to padding and (b) padded queries attend only
    to other padding, avoiding an empty-softmax NaN. Uses only tensor ops (no host
    sync) so it stays inside a compiled graph.

    Both streams of the fused key layout go through this: the GEN offsets give the
    query-side ids, the UND (causal) offsets the ids of the key prefix.

    Returns a ``[seq_len]`` ``int64`` tensor.
    """
    real_count = offsets[-1]  # 0-dim tensor; no .item() / host sync.
    positions = torch.arange(seq_len, device=device)  # [seq_len]
    sample_id = torch.searchsorted(offsets[1:].contiguous(), positions, right=True)  # [seq_len]
    return torch.where(positions < real_count, sample_id, -1).to(torch.long)  # [seq_len]


def _und_flags(metadata: FlexMetadata) -> torch.Tensor:
    """``[seq_len]`` bool marking the UND prefix of the key stream.

    Derived from ``num_und`` rather than stored, so it cannot drift out of step with
    the concatenation the metadata describes. Both the predicate and the run splitting
    in :func:`_metadata_groups` read it: a run that straddled the UND/GEN boundary
    would hold tokens the predicate treats differently, which is exactly what the run
    collapsing assumes cannot happen.
    """
    positions = torch.arange(metadata.seq_len, device=metadata.sample_id.device)  # [seq_len]
    return positions < metadata.num_und  # [seq_len], bool


@dataclass(frozen=True)
class _StreamFields:
    """The metadata one side of the predicate reads, indexed in that side's own coordinates.

    Bundled per side because the two sides of this mask are different streams: the keys
    span ``[UND | GEN]`` and the queries are its GEN tail. Giving the query side its own
    tensors is what lets the predicate index both sides directly, with no offset between
    the two coordinate systems -- see :func:`_multiview_mask_mod` for why that matters to
    the FlashAttention-4 backend.
    """

    sample_id: torch.Tensor
    frame_id: torch.Tensor
    view_id: torch.Tensor
    is_noisy: torch.Tensor
    cond_type_id: torch.Tensor
    is_und: torch.Tensor

    def tail(self, start: int) -> _StreamFields:
        """The same fields covering the tokens from ``start`` on, re-based to offset zero.

        Copies, not views: a view of ``field[start:]`` carries ``start`` as its storage
        offset, and under dynamic shapes that offset reaches the graph as a symbolic scalar,
        which the FlashAttention-4 backend rejects for the same reason it rejects a captured
        Python int -- see :func:`_multiview_mask_mod`. Cloning leaves the fields' own lengths
        as the only symbols the mask carries, and those the backend does accept. The cost is
        six int64/bool rows of the GEN stream, copied once per step alongside the mask itself.
        """
        return _StreamFields(**{f.name: getattr(self, f.name)[start:].clone() for f in fields(self)})


def _key_stream_fields(metadata: FlexMetadata) -> _StreamFields:
    """The fields as stored: one entry per token of the ``[UND | GEN]`` key stream."""
    return _StreamFields(
        sample_id=metadata.sample_id,
        frame_id=metadata.frame_id,
        view_id=metadata.view_id,
        is_noisy=metadata.is_noisy,
        cond_type_id=metadata.cond_type_id,
        is_und=_und_flags(metadata),
    )


def _query_stream_fields(metadata: FlexMetadata) -> _StreamFields:
    """The same fields for the queries, i.e. the GEN tail of the key stream.

    Each is the ``[num_und:]`` tail of a key-stream field, copied to its own storage
    (:meth:`_StreamFields.tail`), so query ``i`` reads exactly what key ``i + num_und``
    reads. ``is_und`` comes along as an all-``False`` tail rather than being special-cased,
    which keeps one predicate serving both this and the key-key form.
    """
    return _key_stream_fields(metadata).tail(metadata.num_und)


def _multiview_pair_predicate(
    q_fields: _StreamFields,
    kv_fields: _StreamFields,
    noisy_attention_scope: NoisyAttentionScope,
) -> MaskMod:
    """Return the multiview supertoken predicate, reading each side's own fields.

    Passing the key-stream fields on both sides gives the predicate over **key-stream
    coordinates**, which is what the block-level collapsing in :func:`build_block_mask`
    needs: it evaluates the predicate on group representatives, and those are key-stream
    positions on both sides. Passing the query fields on the left instead gives the
    ``mask_mod`` FlexAttention itself calls -- see :func:`_multiview_mask_mod`.

    All rules are gated on ``same_sample`` (block-diagonal packing). On top of
    that, using the per-token frame/view/type metadata:

    * any GEN Q attends to every UND K of its sample -- the gen->und pass, which
      carries no further restriction;
    * conditioning Q attends only to conditioning K of the **same conditioning
      type** in the **same (frame, view)**;
    * conditioning Q -> noisy K: never (no rule fires for this quadrant);
    * noisy Q attends to the noisy K that ``noisy_attention_scope`` admits -- every one in
      the sample, those of its own view, or those of its own view or its own frame -- and,
      in addition, to every conditioning K in the **same (frame, view)** regardless of
      conditioning type.

    The narrow scopes restrict on the view or the frame alone and not on ``(frame, view)``:
    a view keeps its own frames, which is the temporal attention that makes it a clip rather
    than a set of stills, and a frame keeps its own views, which is what registers the
    cameras against each other. Both are free at the block level, because
    :func:`_metadata_groups` already splits its runs per ``(item, frame, view)`` cell, so the
    coarse predicate can tell views and frames apart without a finer grouping.

    The GEN rules all read ``frame_id`` / ``view_id`` / ``cond_type_id``, which UND
    tokens carry as ``-1``, so none of them fires on a UND key and the UND rule is
    the only one that admits that quadrant.

    Padding positions carry ``sample_id == -1`` (and ``-1`` in the other fields),
    so real queries never see them and padded queries attend only to padding. The
    packer keeps at least one padded row on each stream once either is padded, so a
    padded query always has a padded key to attend to.
    """
    # The scope enters as tensors rather than as the string it arrives as, because the closure
    # below is what Inductor traces and a captured scalar is what the FlashAttention-4 backend
    # refuses to lower -- see :func:`_multiview_mask_mod`, and the structural test that pins it.
    # Folding them into the expression rather than branching around it also leaves a single path
    # through the predicate, so the element-level and block-level forms cannot come apart. Both
    # gates off leaves the query its own view, which every scope admits.
    device = q_fields.view_id.device
    reaches_every_view = torch.tensor(noisy_attention_scope == "all_views", device=device)
    reaches_own_frame = torch.tensor(noisy_attention_scope == "same_view_or_frame", device=device)

    def pair_allowed(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        same_sample = q_fields.sample_id[q_idx] == kv_fields.sample_id[kv_idx]
        same_frame = q_fields.frame_id[q_idx] == kv_fields.frame_id[kv_idx]
        same_view = q_fields.view_id[q_idx] == kv_fields.view_id[kv_idx]
        same_fv = same_frame & same_view
        same_cond_type = q_fields.cond_type_id[q_idx] == kv_fields.cond_type_id[kv_idx]
        q_noisy = q_fields.is_noisy[q_idx]
        k_noisy = kv_fields.is_noisy[kv_idx]

        # GEN Q -> UND K: the whole caption of the sample, conditioning and noisy alike.
        gen_to_und = kv_fields.is_und[kv_idx]
        # conditioning Q -> conditioning K: same (frame, view, cond type).
        cond_to_cond = (~q_noisy) & (~k_noisy) & same_fv & same_cond_type
        # noisy Q -> noisy K: the whole sample, the query's own view, or its view or frame.
        in_scope = reaches_every_view | same_view | (reaches_own_frame & same_frame)
        noisy_to_noisy = q_noisy & k_noisy & in_scope
        # noisy Q -> conditioning K: same (frame, view), any cond type.
        noisy_to_cond = q_noisy & (~k_noisy) & same_fv

        return same_sample & (gen_to_und | cond_to_cond | noisy_to_noisy | noisy_to_cond)

    return pair_allowed


def _multiview_mask_mod(metadata: FlexMetadata) -> MaskMod:
    """The predicate in FlexAttention's own index convention: ``q_idx`` numbers GEN from zero.

    This is the closure that ends up inside the :class:`BlockMask` and therefore the one
    Inductor traces, so what it captures is a hard constraint rather than a detail. It
    captures tensors, and tensors that start at offset zero. The obvious alternative -- keep
    the key-stream predicate and shift the row index, ``pair_allowed(q_idx + num_und,
    kv_idx)`` -- captures ``num_und`` as a Python int, and the FlashAttention-4 backend
    rejects the whole graph for it: a captured scalar goes dynamic as soon as the enclosing
    region is compiled with dynamic shapes, and CuteDSL cannot inline a symbolic value into
    its template. Slicing the fields per side rather than shifting the index does not get rid
    of that scalar on its own -- it moves it into the slice's storage offset, which goes
    dynamic just the same, hence the copy in :meth:`_StreamFields.tail`.

    Symbolically shaped fields are fine, and a training run demonstrates it: one graph over a
    hundred padded geometries, on the flash backend, because the lengths the mask needs are
    ones the surrounding graph already has symbols for and nothing has to be lifted into the
    subgraph. A caller that hands the mask over without that relation available can still trip
    the check, which is what ``attention_test`` exercises.
    """
    return _multiview_pair_predicate(
        _query_stream_fields(metadata),
        _key_stream_fields(metadata),
        metadata.noisy_attention_scope,
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
    num_und: int = 0,
    causal_offsets: torch.Tensor | None = None,
    noisy_attention_scope: NoisyAttentionScope = "all_views",
) -> FlexMetadata:
    """Build key-stream metadata for camera-major multiview transfer items.

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
        seq_len: block-padded GEN sequence length, i.e. the query count. The returned
            fields are ``[num_und + seq_len]``, covering the fused ``[UND | GEN]`` key
            stream.
        full_q_offsets: cumulative per-sample GEN offsets, ``[num_samples + 1]``;
            ``full_q_offsets[-1]`` is the real (unpadded) GEN token count.
        token_shapes: one ``(latent_t, patch_h, patch_w)`` triple per item,
            ``num_items`` entries.
        condition_masks: one mask per item, each ``[latent_t]`` over the camera-major
            latent axis; a non-zero entry marks a conditioning (clean) frame.
        num_vision_items_per_sample: items owned by each sample, ``num_samples`` entries.
        num_views_per_vision_item: cameras per item, ``num_items`` entries.
        device: device for the returned tensors.
        num_und: block-padded UND (causal) stream length, which prefixes the key
            stream. 0 leaves the metadata GEN-only, for a square self-attention mask.
        causal_offsets: cumulative per-sample UND offsets, ``[num_samples + 1]``;
            required when ``num_und`` is non-zero, since the UND rule is "same sample"
            and nothing else. ``causal_offsets[-1]`` is the real UND token count, so
            everything past it is padding.
        noisy_attention_scope: which noisy tokens of its sample a noisy token reaches --
            those of every view, of its own view, or of its own view or frame. See
            ``NoisyAttentionScope``; :class:`FlexMetadata` carries it on to both forms
            of the predicate.

    Returns:
        :class:`FlexMetadata` whose five per-token fields are each
        ``[num_und + seq_len]``: the UND tokens (sample ids only, ``-1`` in every
        multiview field), then the ``real_token_count`` real GEN tokens in packed
        order, each stream followed by ``-1`` sentinels (``False`` for ``is_noisy``)
        across its trailing pad.

    Raises:
        ValueError: if the per-item sequences disagree on ``num_items``, if a
            ``latent_t`` is not divisible by its ``num_views``, if one sample's items
            disagree on the ``(num_views, frames_per_view)`` grid, if a condition mask
            length does not match its ``latent_t``, if ``real_token_count`` does not
            match ``full_q_offsets[-1]`` or exceeds ``seq_len``, or if ``num_und`` is
            non-zero without ``causal_offsets``.
    """
    if num_und and causal_offsets is None:
        raise ValueError(
            f"A fused key stream with {num_und} UND tokens needs causal_offsets to label them by "
            "sample; without it every UND key would look like padding to the mask."
        )
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
    # splits. If they disagree, _build_stream_sample_ids draws the padding boundary somewhere
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

    sample_id = _build_stream_sample_ids(full_q_offsets, seq_len, device)  # [seq_len]

    if num_und:
        assert causal_offsets is not None  # guarded above; narrows the type for the checker.
        # UND keys join the front of the stream carrying nothing but their sample: the
        # multiview fields are what the GEN rules match on, and holding them at -1 is
        # what keeps those rules from firing on this quadrant.
        und_sentinel = torch.full((num_und,), -1, device=device, dtype=torch.long)  # [num_und]
        sample_id = torch.cat((_build_stream_sample_ids(causal_offsets, num_und, device), sample_id))
        frame_id = torch.cat((und_sentinel, frame_id))  # [num_und+seq_len]
        view_id = torch.cat((und_sentinel, view_id))  # [num_und+seq_len]
        cond_type_id = torch.cat((und_sentinel, cond_type_id))  # [num_und+seq_len]
        is_noisy = torch.cat((torch.zeros(num_und, device=device, dtype=torch.bool), is_noisy))  # bool

    return FlexMetadata(
        seq_len=num_und + seq_len,
        sample_id=sample_id,  # [num_und+seq_len]
        frame_id=frame_id,  # [num_und+seq_len]
        view_id=view_id,  # [num_und+seq_len]
        is_noisy=is_noisy,  # [num_und+seq_len], bool
        cond_type_id=cond_type_id,  # [num_und+seq_len]
        num_und=num_und,
        noisy_attention_scope=noisy_attention_scope,
    )


def _check_block_aligned(seq_len: int, stream: str, alignment: int) -> None:
    """Raise if ``seq_len`` is not a multiple of ``alignment``.

    Both streams of the fused key layout have to be aligned, not just their total: the
    UND prefix ends on a block boundary only if its own length is a multiple of the
    block size, and a block straddling the boundary would be a partial one for every
    query, costing the fully-unmasked fast path on the whole gen->und quadrant.

    ``alignment`` is the block that tiles the stream, which is the backend's rather than
    a fixed number: :func:`triton_backend_block_size` on Triton, or FlashAttention-4's
    coarser query tile on Blackwell (:func:`flash_backend_block_size`). The latter is
    always a multiple of the former, so the backend's own block is the only multiple a
    length has to satisfy, and it is the one worth naming: raising about the 128 floor
    instead would understate the requirement.
    """
    if seq_len % alignment != 0:
        knob = "full_seq_alignment" if stream == "GEN" else "causal_seq_alignment"
        raise ValueError(
            f"FlexAttention needs a block-aligned {stream} sequence length, got {seq_len}, which is "
            f"not a multiple of {alignment}. Callers pad the {stream} stream at packing time "
            f"via {knob} in build_packed_sequence."
        )


def triton_backend_block_size() -> tuple[int, int]:
    """``(Q, KV)`` block size the Triton FlexAttention backend iterates on.

    FlexAttention's own default, and the finest granularity anything here is built at:
    every mask this module produces is tiled at this or at a multiple of it, and so is
    every padded stream length. Unlike :func:`flash_backend_block_size` the answer does
    not depend on the device -- the Triton kernels step 128x128 wherever torch runs them.
    """
    return (128, 128)


def flash_backend_block_size(device: torch.device) -> tuple[int, int]:
    """``(Q, KV)`` block size the FlashAttention-4 backend iterates on ``device``.

    FA4 drives its tile scheduler from the block mask, so a mask handed to
    ``BACKEND="FLASH"`` has to be built at the block size the kernel steps in. On
    Blackwell each CTA processes two M-tiles to keep the async pipeline full
    (``q_stage=2``), which makes 256 rows the smallest unit the scheduler can skip;
    Hopper stays at the 128x128 the Triton backend also uses. A finer mask is not
    expressible on the FA4 path, which is why the GEN stream needs padding to this
    Q block size (a multiple of :func:`triton_backend_block_size`'s in both cases)
    when the backend is in play.

    Raises:
        ValueError: on pre-Hopper hardware, where the backend does not exist, or for a
            non-CUDA device.
    """
    if device.type != "cuda":
        raise ValueError(f"The FlashAttention-4 backend needs a CUDA device, got {device}.")
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        return (256, 128)
    if major == 9:
        return (128, 128)
    raise ValueError(
        f"The FlashAttention-4 backend supports Hopper (sm90) and Blackwell (sm100+) only, "
        f"got sm{major}{minor} ({torch.cuda.get_device_name(device)}); use the Triton backend there."
    )


@dataclass(frozen=True)
class FlexBackend:
    """A choice of FlexAttention backend, and the mask geometry that choice forces.

    Picking the kernels is not a decision anyone gets to make on its own. FA4 drives its
    tile scheduler from the block mask, so the mask has to be built at the block size the
    kernel steps in, and the streams have to be padded far enough for that (coarser) mask to
    tile them. Mismatches are not reliably caught -- an over-fine mask handed to FA4 attends
    to the wrong tokens rather than raising -- so callers take the whole set from one object
    that :func:`resolve_flex_backend` returns instead of assembling it by hand.

    Attributes:
        name: ``"triton"`` or ``"flash"``, for logs and error messages.
        block_size: ``(Q, KV)`` block size to build the mask at.
        kernel_options: which kernels :func:`flex_attention` requests of torch, ``None``
            leaving the choice to FlexAttention's own heuristics.
    """

    name: str
    block_size: tuple[int, int]
    kernel_options: dict[str, int | bool | str] | None

    @property
    def full_seq_alignment(self) -> int:
        """``build_packed_sequence``'s GEN padding multiple for this backend.

        The GEN stream supplies the mask's rows, so it has to be padded to the query block
        -- 256 on Blackwell FA4, which is coarser than the granularity the metadata itself
        is built at. That block is always a multiple of :func:`triton_backend_block_size`'s
        (see :func:`flash_backend_block_size`), so padding to it satisfies both at once.
        """
        return self.block_size[0]

    @property
    def causal_seq_alignment(self) -> int:
        """``build_packed_sequence``'s UND padding multiple for this backend.

        The UND stream is keys only, so what it answers to is the key block, and that is 128
        on both backends -- FA4 buys its coarser tiles in the query dimension alone. Padding
        it to the query block instead would also be correct, just more padding for nothing:
        all the boundary between the two streams has to fall on is a key-block boundary.
        """
        return self.block_size[1]


def _get_triton_flex_backend() -> FlexBackend:
    """FlexAttention's own default, and the fallback whenever FlashAttention-4 is unavailable."""
    return FlexBackend(name="triton", block_size=triton_backend_block_size(), kernel_options=None)


def _get_flash_flex_backend(device: torch.device) -> FlexBackend:
    """FlashAttention-4 on ``device``, at the block size its tile scheduler steps in.

    Only ask for this once :func:`flash_backend_unavailable_reason` has returned ``None`` for
    ``device``: the block size is the one the kernels dictate, so requesting it on hardware
    they have no tiles for raises rather than returning a geometry that could be used.
    """
    # ``BACKEND="FLASH"`` is what selects the FlashAttention-4 kernels (CuTeDSL) over
    # FlexAttention's default Triton ones, and requires the ``flash-attn-4`` package; see
    # flex_attention_bench for the install and the current limitations. Paired here with the
    # block size the mask then has to be built at, which is the whole reason the two travel
    # together rather than being picked separately.
    kernel_options: dict[str, int | bool | str] = {"BACKEND": "FLASH"}
    return FlexBackend(name="flash", block_size=flash_backend_block_size(device), kernel_options=kernel_options)


def flash_backend_unavailable_reason(device: torch.device) -> str | None:
    """Why FlexAttention cannot lower onto FlashAttention-4 here, or ``None`` if it can.

    Three independent things have to hold, and each fails in its own place: the kernels
    (``flash_attn.cute``, from the ``flash-attn-4`` package) have to be installed, this
    torch has to carry the Inductor lowering that emits CuTeDSL for them, and the GPU has
    to be one the kernels exist for. Torch's own ``ensure_flash_available`` covers the
    first, its importability the second, and :func:`flash_backend_block_size` the third.

    Torch never selects this backend on its own -- ``BACKEND="FLASH"`` is opt-in and
    *raises* when it cannot be honoured -- so choosing it automatically means answering
    this question before the call, not recovering after it.

    This does not attempt to predict the checks Inductor makes on the traced graph itself
    (dtypes, head widths, scalars captured by the ``mask_mod``). The multiview mask and
    bf16 q/k/v satisfy them; if a future caller does not, the lowering raises with torch's
    own explanation of what to do, and ``flex_attention_backend="triton"`` pins the run
    back to Triton in the meantime.
    """
    try:
        flash_backend_block_size(device)
    except ValueError as e:
        return str(e)
    try:
        from torch._inductor.kernel.flex.flex_flash_attention import ensure_flash_available
    except ImportError as e:
        return (
            f"this torch build carries no FlexAttention FlashAttention-4 lowering to reach the kernels with ({e}); "
            "it needs one new enough to have the Inductor CuTeDSL codegen"
        )
    # lru_cached in torch, and its own docstring asks for ensure_flash_available.cache_clear()
    # if the package is installed into a running interpreter.
    if not ensure_flash_available():
        return (
            'the flash_attn.cute kernels are not installed; `pip install --pre "flash-attn-4[cu13]"` '
            "adds them (drop the extra on CUDA 12.x), as flex_attention_bench documents"
        )
    return None


def resolve_flex_backend(device: torch.device, preference: str = "auto") -> FlexBackend:
    """The backend the multiview FlexAttention path should run on, and its mask geometry.

    ``preference`` is a run's policy, not its outcome:

    * ``"auto"`` takes FlashAttention-4 wherever it is available and Triton elsewhere.
      This is the default, so a host that has the kernels installed uses them; the flip
      side is that the same config on a host without them runs different kernels, at a
      different padded length, with different rounding. A run that has to stay
      bit-comparable with another should pin the backend rather than rely on the
      environments matching.
    * ``"triton"`` pins FlexAttention's Triton kernels, ignoring what is installed.
    * ``"flash"`` demands FA4 and raises if it cannot be used, for a benchmark or a test
      that is meaningless on the other backend.

    Raises:
        ValueError: for an unknown ``preference``, or for ``"flash"`` when the backend is
            unavailable -- with the reason from :func:`flash_backend_unavailable_reason`.
    """
    if preference not in FLEX_BACKEND_PREFERENCES:
        raise ValueError(f"Unknown flex_attention_backend {preference!r}; expected one of {FLEX_BACKEND_PREFERENCES}.")
    if preference == "triton":
        return _get_triton_flex_backend()
    reason = flash_backend_unavailable_reason(device)
    if reason is None:
        return _get_flash_flex_backend(device)
    if preference == "flash":
        raise ValueError(
            f"flex_attention_backend='flash' requires FlexAttention's FlashAttention-4 backend, but {reason}. "
            "Use 'auto' to fall back to Triton where it is unavailable."
        )
    return _get_triton_flex_backend()


def _block_presence(
    group_id: torch.Tensor,
    num_blocks: int,
    block_size: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    """One-hot ``[num_blocks, num_groups]`` marking which metadata runs each block touches."""
    presence = torch.zeros(num_blocks, num_groups, dtype=torch.float32, device=device)
    block_of_token = torch.arange(group_id.numel(), device=device) // block_size  # [seq_len]
    presence[block_of_token, group_id] = 1.0
    return presence


def _metadata_groups(metadata: FlexMetadata, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the key stream into runs of tokens that agree on every metadata field.

    The predicate reads nothing but those fields, so two tokens carrying the same
    tuple are interchangeable on either side of it. Evaluating it once per run
    therefore reproduces every token pair exactly. Runs rather than unique tuples
    keep this to a single comparison pass; a tuple that reappears in a later run just
    gets a second id, which changes no result.

    The UND flag is one of the fields, so no run spans the UND/GEN boundary. Two
    tokens either side of it can otherwise carry identical tuples -- the trailing pad
    of the UND stream and the trailing pad of the GEN stream both read as all ``-1`` --
    and the predicate does distinguish them.

    Returns ``(group_id [seq_len], representatives [num_groups])``, where
    ``representatives`` holds the first token index of each run.
    """
    fields = torch.stack(
        (
            metadata.sample_id,
            metadata.frame_id,
            metadata.view_id,
            metadata.is_noisy.to(torch.long),
            metadata.cond_type_id,
            _und_flags(metadata).to(torch.long),
        ),
        dim=1,
    )  # [seq_len, 6]
    starts_run = torch.ones(metadata.seq_len, dtype=torch.bool, device=device)  # [seq_len]
    starts_run[1:] = (fields[1:] != fields[:-1]).any(dim=1)
    group_id = torch.cumsum(starts_run, dim=0) - 1  # [seq_len]
    representatives = torch.nonzero(starts_run, as_tuple=False).squeeze(1)  # [num_groups]
    return group_id, representatives


def _ordered_blocks(dense_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a ``[1, 1, nQ, nKV]`` bool block matrix to FlexAttention's block form.

    Mirrors ``torch.nn.attention.flex_attention._dense_to_ordered``: per query-block
    row, how many kv blocks are selected and their indices with the selected ones
    first. The sort is stable and descending, so the selected indices stay ascending.
    """
    dense = dense_blocks.to(torch.int32)  # [1,1,nQ,nKV]
    counts = dense.sum(dim=-1).to(torch.int32)  # [1,1,nQ]
    indices = torch.argsort(dense, dim=-1, descending=True, stable=True).to(torch.int32)  # [1,1,nQ,nKV]
    return counts.contiguous(), indices.contiguous()


def build_block_mask(
    metadata: FlexMetadata,
    device: torch.device,
    block_size: tuple[int, int],
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` from precomputed flex metadata.

    Uses the multiview supertoken ``mask_mod``; the multiview fields
    (``frame_id`` / ``view_id`` / ``is_noisy`` / ``cond_type_id``) must be
    populated on ``metadata``.

    ``create_block_mask`` is deliberately not used here. It evaluates the predicate
    into a dense ``[Q_LEN, KV_LEN]`` bool tensor before reducing it to blocks, and
    Inductor does not fuse that intermediate away for this ``mask_mod``, so the build
    costs one byte per token pair: 24 GiB at 162k tokens and 98 GiB at 324k, which is
    where 11 cameras at 720p land.

    Instead the predicate is collapsed onto the ``G`` runs of equal metadata
    (:func:`_metadata_groups`), which for a multiview pack is one run per
    ``(item, frame, view)`` cell -- a few hundred, against hundreds of thousands of
    tokens. ``allowed [G, G]`` is the predicate at that granularity, ``presence
    [num_blocks, G]`` marks which runs each block of tokens touches, and the two
    matmuls below count, for every block pair, the allowed and the blocked run pairs
    it spans. A block pair is fully unmasked when it spans no blocked pair and
    partially masked when it spans at least one of each. Both matmuls accumulate
    non-negative terms and are only ever compared against zero, so the counts do not
    have to be exactly representable for the comparisons to be exact.

    That makes the build ``O(num_q_blocks * num_kv_blocks * G)`` in time and
    ``O(num_q_blocks * num_kv_blocks)`` in memory, versus ``O(seq_len**2)`` for both
    before.

    Call this from outside the decoder layers and hand the returned mask to the
    attention path: the group count is data-dependent, so materializing it syncs with
    the host, which Dynamo cannot trace inside a compiled, activation-checkpointed
    layer. Building it once per forward also avoids rebuilding the identical mask in
    every layer.

    Args:
        metadata: per-key-token fields driving the multiview ``mask_mod``, spanning the
            ``[UND | GEN]`` key stream. Its GEN tail is the query side, so the mask is
            ``metadata.q_len`` by ``metadata.seq_len``.
        device: device for the mask tensors.
        block_size: ``(Q, KV)`` granularity of the mask, which the backend dictates:
            :func:`triton_backend_block_size` for Triton, or
            :func:`flash_backend_block_size` for FlashAttention-4, which iterates a
            coarser Q tile on Blackwell. A coarser mask is always valid, just less
            sparse: the kernel visits whole blocks, and ``mask_mod`` still masks
            individual pairs inside a partial one.

    Raises:
        ValueError: if either stream's length is not a multiple of the block that tiles it.
    """
    q_block_size, kv_block_size = block_size
    # One check per stream, each against the block that tiles it: the GEN stream supplies the
    # mask's rows and answers to the Q block, the UND prefix is keys only and answers to the KV
    # block. Both blocks are multiples of the Triton one, so these subsume the granularity the
    # metadata itself is built at, and the fused key length follows from the two of them.
    _check_block_aligned(metadata.q_len, "GEN", q_block_size)
    _check_block_aligned(metadata.num_und, "UND", kv_block_size)
    key_fields = _key_stream_fields(metadata)
    pair_allowed = _multiview_pair_predicate(key_fields, key_fields, metadata.noisy_attention_scope)
    mask_mod = _multiview_mask_mod(metadata)
    num_q_blocks = metadata.q_len // q_block_size
    num_kv_blocks = metadata.seq_len // kv_block_size

    group_id, representatives = _metadata_groups(metadata, device)
    batch_index = torch.zeros((), dtype=torch.long, device=device)  # the predicate ignores b/h
    # The predicate rather than the mask_mod: representatives index the key stream on both
    # sides, while the mask_mod expects a query-relative row index.
    allowed = pair_allowed(
        batch_index,
        batch_index,
        representatives.unsqueeze(1),  # [num_groups,1]
        representatives.unsqueeze(0),  # [1,num_groups]
    ).to(torch.float32)  # [num_groups,num_groups]

    num_groups = representatives.numel()
    # Queries are the GEN tail of the key stream, so their presence comes from that slice.
    presence_q = _block_presence(
        group_id[metadata.num_und :], num_q_blocks, q_block_size, num_groups, device
    )  # [nQ,num_groups]
    presence_kv = (
        presence_q
        if metadata.num_und == 0 and kv_block_size == q_block_size
        else _block_presence(group_id, num_kv_blocks, kv_block_size, num_groups, device)
    )  # [nKV,num_groups]

    allowed_hits = (presence_q @ allowed) @ presence_kv.t()  # [nQ,nKV]
    blocked_hits = (presence_q @ (1.0 - allowed)) @ presence_kv.t()  # [nQ,nKV]
    full_blocks = blocked_hits == 0.0  # [nQ,nKV]
    partial_blocks = (allowed_hits > 0.0) & ~full_blocks  # [nQ,nKV]

    kv_num_blocks, kv_indices = _ordered_blocks(partial_blocks.view(1, 1, num_q_blocks, num_kv_blocks))
    full_kv_num_blocks, full_kv_indices = _ordered_blocks(full_blocks.view(1, 1, num_q_blocks, num_kv_blocks))
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(metadata.q_len, metadata.seq_len),
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
    block_size: tuple[int, int],
    num_und: int = 0,
    causal_offsets: torch.Tensor | None = None,
    noisy_attention_scope: NoisyAttentionScope = "all_views",
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` for camera-major multiview items.

    The entry point for the packed-batch path: it runs
    :func:`build_multiview_flex_metadata` and :func:`build_block_mask` back to back so
    the caller only handles the mask. See those two for the token layout the metadata
    encodes, for why the mask has to be built outside the decoder layers, for what
    ``block_size`` selects, for what ``num_und`` / ``causal_offsets`` add, and for what
    ``noisy_attention_scope`` lets the noisy tokens reach.

    The two stages stay separately callable because the metadata layout is checked on
    CPU in the unit tests, independently of the block-mask construction.
    """
    metadata = build_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        token_shapes=token_shapes,
        condition_masks=condition_masks,
        num_vision_items_per_sample=num_vision_items_per_sample,
        num_views_per_vision_item=num_views_per_vision_item,
        device=device,
        num_und=num_und,
        causal_offsets=causal_offsets,
        noisy_attention_scope=noisy_attention_scope,
    )
    return build_block_mask(metadata, device, block_size)


def flex_attention(
    full_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    block_mask: BlockMask,
    backend: FlexBackend,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The generator's attention over its own sample, via a single FlexAttention call.

    Drop-in replacement for the dense full branch of ``two_way_attention``: GEN tokens
    query the fused ``[UND | GEN]`` key stream under the multiview mask, so the
    gen->und and gen->gen quadrants come out of one kernel with one softmax. The
    key length may therefore exceed the query length; passing a square GEN-only mask
    and GEN-only k/v gives plain self-attention instead.

    Both stream lengths must be multiples of ``block_mask``'s own block size, arranged at
    packing time (``full_seq_alignment`` and ``causal_seq_alignment`` in
    ``build_packed_sequence``), so this function never has to re-pad q/k/v and the
    metadata in every layer.

    Args:
        full_q: GEN queries, ``[1, N_full, heads, head_dim]`` (may include
            trailing pack padding beyond the last real token).
        full_k: fused keys, ``[1, N_und + N_full, kv_heads, head_dim]``, the padded UND
            stream followed by the padded GEN stream -- the order the mask was built for.
        full_v: fused values, ``[1, N_und + N_full, kv_heads, head_dim]``.
        block_mask: the GEN-tower mask, precomputed outside the decoder layers via
            :func:`build_block_mask` (or :func:`build_multiview_block_mask`); it must
            have been built for the same two lengths.
        backend: the :class:`FlexBackend` from the :func:`resolve_flex_backend` call that
            decided ``block_mask``'s block size; its ``kernel_options`` are what select the
            kernels. The whole object rather than those options alone, because the two only
            mean anything together -- the FlashAttention-4 options are correct for a mask
            built at that backend's block size and silently wrong for any other -- so
            passing it lets that agreement be checked here. Required, since every mask was
            built at some backend's block size and running it on another's is the bug this
            checks for; :func:`resolve_flex_backend` is where one comes from. Dynamo guards
            on the ``kernel_options`` dict, so each distinct value compiles its own kernel.
        return_lse: when ``True`` also return the log-sum-exp, for a caller that merges
            this output with another attention term. The fused path does not, and asking
            for it costs the FlashAttention-4 backward, which cannot differentiate it.

    Returns:
        The attention output, ``[1, N_full, heads, head_dim]`` -- the heads-last layout
        ``from_mode_splits`` expects, with the sequence length matching ``full_q`` (pack
        padding preserved). When ``return_lse`` is ``True``, returns the tuple
        ``(out, lse)`` where ``lse`` has shape ``[1, N_full, heads]``.

    Raises:
        ValueError: if either length is not a multiple of the corresponding block size of
            the mask, if k and v disagree on length, if ``block_mask`` was built for
            different lengths, or if it was built at a block size other than ``backend``'s.
    """
    from torch.nn.attention.flex_attention import AuxRequest

    q_seq_len = full_q.shape[1]
    kv_seq_len = full_k.shape[1]
    num_q_heads = full_q.shape[2]
    num_kv_heads = full_k.shape[2]

    if full_v.shape[1] != kv_seq_len:
        raise ValueError(f"Keys cover {kv_seq_len} tokens but values cover {full_v.shape[1]}.")
    if kv_seq_len < q_seq_len:
        raise ValueError(
            f"The fused key stream holds {kv_seq_len} tokens, fewer than the {q_seq_len} GEN tokens it "
            "has to contain alongside the UND prefix."
        )
    # BlockMask.shape is (*batch_dims, Q_LEN, KV_LEN), holding the lengths the mask was
    # built for. A mask carried over from a differently-shaped pack would mask the wrong
    # tokens rather than fail, so the lengths are compared before the kernel sees either.
    mask_q_len, mask_kv_len = block_mask.shape[-2:]
    if (mask_q_len, mask_kv_len) != (q_seq_len, kv_seq_len):
        raise ValueError(
            f"block_mask covers Q_LEN={mask_q_len}, KV_LEN={mask_kv_len}, but the GEN sequence is "
            f"{q_seq_len} tokens against {kv_seq_len} keys; the mask must be built from the same pack "
            "that produced q/k/v."
        )
    # The mask carries the backend's geometry: it was built at the block size the kernel steps
    # in, 256 query rows on Blackwell FA4 against 128 on Triton, so it is what the streams have
    # to be measured against. One check per stream, each against the block that tiles it, as
    # build_block_mask makes them -- a partial trailing block would silently drop the rows the
    # mask has no entry for.
    mask_q_block, mask_kv_block = block_mask.BLOCK_SIZE
    if (mask_q_block, mask_kv_block) != backend.block_size:
        # The kernels below step backend.block_size whatever the mask says, and an over-fine
        # mask handed to FA4 attends to the wrong tokens rather than raising, so this is the
        # one mismatch in this function that nothing downstream would report.
        raise ValueError(
            f"block_mask was built at {(mask_q_block, mask_kv_block)} blocks, but the {backend.name} backend "
            f"steps {backend.block_size}; build the mask from the same FlexBackend that runs it."
        )
    _check_block_aligned(q_seq_len, "GEN", mask_q_block)
    _check_block_aligned(kv_seq_len - q_seq_len, "UND", mask_kv_block)

    q = _to_flex_layout(full_q)  # [1,num_q_heads,N_full,head_dim]
    k = _to_flex_layout(full_k)  # [1,num_kv_heads,N_und+N_full,head_dim]
    v = _to_flex_layout(full_v)  # [1,num_kv_heads,N_und+N_full,head_dim]

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
            kernel_options=backend.kernel_options,
        )  # attn_out: [1,num_q_heads,N_full,head_dim], aux.lse: [1,num_q_heads,N_full]
        # Convert to the heads-last layout ([1,S,H,D] / [1,S,H]) that from_mode_splits
        # expects, and that a caller merging this output would need.
        return _from_flex_layout(attn_out), _from_flex_layout(aux.lse)  # [1,N_full,heads,head_dim], [1,N_full,heads]

    attn_out = _COMPILED_FLEX_ATTENTION(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=num_q_heads != num_kv_heads,
        kernel_options=backend.kernel_options,
    )  # attn_out: [1,num_q_heads,N_full,head_dim]
    # Convert to the heads-last layout ([1,S,H,D]) that from_mode_splits expects.
    return _from_flex_layout(attn_out)  # [1,N_full,heads,head_dim]
