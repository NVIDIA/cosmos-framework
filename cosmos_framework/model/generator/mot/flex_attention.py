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
   ``is_control``), covering the ``[UND | GEN]`` key stream;
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

The supertoken rules :func:`build_block_mask` enforces, all within a sample: every RGB
token reaches every RGB token within whatever view footprint ``attention_scope`` admits
(every view by default). Control tokens -- a WSM (World Scenario Map) control video, say --
are a separate modality read off ``is_control``: RGB tokens reach them at their own view only,
control tokens reach each other at their own view only, and no control token reaches an RGB
token at all. Every GEN token, conditioning or noisy, attends to every UND token of its own
sample, as the dense gen->und pass does. See :func:`_multiview_pair_predicate` for the rules
themselves; richer patterns mean more metadata fields and a longer ``mask_mod``, not new
varlen bookkeeping.

LSE convention
--------------
``AuxRequest(lse=True)`` returns the log-sum-exp of the scaled scores in **natural log**, at
the default ``1/sqrt(head_dim)`` scale, in the ``[B, H, S]`` layout -- identical to
``cosmos_framework.model.attention(..., return_lse=True)`` once transposed to heads-last ``[B, S, H]``,
so a caller that does want to merge this output with another attention term can. The fused
path does not ask for it: it is a complete attention, and the gradient the request puts in
the graph is what the FlashAttention-4 backward refuses to lower.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from cosmos_framework.configs.base.defaults.flex_attention import (
    ATTENTION_SCOPES,
    FLEX_BACKEND_PREFERENCES,
    AttentionScope,
)
from cosmos_framework.model.generator.mot.flex_attention_utils import (
    build_block_mask_from_metadata_runs,
    metadata_run_groups,
)

# ``dynamic=True`` lets one compiled kernel handle varying (block-aligned) shapes
# across steps instead of specialising and recompiling per shape. Only the BlockMask
# *data* changes per step, which does not trigger recompilation either way. torch's
# entry point is imported under an alias because this module's own,
# :func:`flex_attention`, takes that name.
_COMPILED_FLEX_ATTENTION = torch.compile(torch_flex_attention, dynamic=True)

# A FlexAttention mask predicate: (b, h, q_idx, kv_idx) -> bool tensor.
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FlexMetadata:
    """Per-key-token metadata that drives the flex block mask.

    Each field is a 1-D ``[seq_len]`` tensor covering the key/value stream the mask
    spans, in packed token order. All are ``int64`` except ``is_noisy`` and ``is_control``,
    which are ``bool``. Padding positions carry the sentinel ``-1`` (``False`` for the two
    bool fields) so real queries never attend to padding and padded queries attend only to
    padding (no empty-softmax NaN).

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
    * ``is_control``: ``bool`` tensor, ``True`` for tokens of a control item (e.g. a WSM --
      World Scenario Map -- control video), ``False`` for every other token (RGB and UND
      alike). It carries no per-item identity, so control tokens sharing a view read as one
      signal; since the control rules only ever pair within a view, that costs nothing as
      long as a view holds a single control stream, which is the caller's to arrange (see
      ``MaskItem.is_control``). The field is also the only switch the control rules have: the
      ordinary T2V/I2V/V2V batch marks no control item, carries it all-``False``, and the
      control terms drop out of the predicate on their own -- one fewer place for a config flag
      and the data it describes to disagree.
    * ``timestamp``: ``float`` tensor, the wall-clock instant (in seconds, relative to
      whatever origin the caller's items share) a token's ``(view, frame)`` cell was
      captured at; ``-1.0`` on UND tokens and on padding, matching the other fields'
      sentinel. Read only by the ``"decomposed"`` scope's temporal half, and only when
      ``decomposed_temporal_window_seconds`` is set -- see :func:`_multiview_pair_predicate`.
      A token's timestamp is a function of its ``(view_id, frame_id)`` alone (one view's
      frames tick at one rate), which is what lets :func:`_metadata_groups` keep grouping on
      the integer fields without this one: two tokens sharing a run already share their
      timestamp.

    The mask enforces, within a sample:

    * GEN Q -> UND K: always (the gen->und pass, unrestricted);
    * sensor Q -> sensor K: within whatever ``AttentionScope`` admits (every view, its own
      view, or -- ``"decomposed"`` -- its own view or its own frame, or -- with
      ``decomposed_temporal_window_seconds`` set -- its own view or a key within that many
      seconds of it, at or before it), regardless of whether either token is conditioning;
    * sensor Q (conditioning or noisy) -> control K: same view, any frame;
    * control Q -> control K: same view, any frame;
    * control Q -> sensor K: never.

    :func:`_multiview_pair_predicate` is where those rules are actually expressed, so it is
    the one to trust if this list and it ever drift apart.

    ``attention_scope`` is a choice rather than a description of the batch; it rides here
    so that the two places the predicate is built from -- the ``mask_mod`` the kernel
    calls and the block-level collapsing in :func:`build_block_mask` -- cannot be given
    different answers: a block the collapsing calls fully unmasked is one the kernel never
    calls ``mask_mod`` on at all, so the two disagreeing would not raise, it would silently
    attend across views.

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
    is_control: torch.Tensor
    timestamp: torch.Tensor
    num_und: int
    attention_scope: AttentionScope
    decomposed_temporal_window_seconds: float | None = None

    def __post_init__(self) -> None:
        # Nothing enforces the Literal at runtime, and an unrecognised scope would not fail:
        # the predicate reads it by equality, so a typo leaves both gates off and masks as
        # "same_view". attrs already validated a scope that arrived via config, but this type
        # is also built directly, and a silently narrower mask is worse than a rejected one.
        if self.attention_scope not in ATTENTION_SCOPES:
            raise ValueError(f"Unknown attention_scope {self.attention_scope!r}; expected one of {ATTENTION_SCOPES}.")
        if self.decomposed_temporal_window_seconds is not None and self.decomposed_temporal_window_seconds < 0:
            raise ValueError(
                "decomposed_temporal_window_seconds must be non-negative, got "
                f"{self.decomposed_temporal_window_seconds}."
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
    is_control: torch.Tensor
    is_und: torch.Tensor
    timestamp: torch.Tensor

    def tail(self, start: int) -> _StreamFields:
        """The same fields covering the tokens from ``start`` on, re-based to offset zero.

        Copies, not views: a view of ``field[start:]`` carries ``start`` as its storage offset,
        which under dynamic shapes reaches the graph as a symbolic scalar -- rejected by the
        FlashAttention-4 backend for the same reason a captured Python int is, see
        :func:`_multiview_mask_mod`. Cloning leaves the fields' own lengths as the only symbols
        the mask carries, and those the backend accepts. The cost is six int64/bool rows of the
        GEN stream, copied once per step alongside the mask itself.
        """
        return _StreamFields(**{f.name: getattr(self, f.name)[start:].clone() for f in fields(self)})


def _key_stream_fields(metadata: FlexMetadata) -> _StreamFields:
    """The fields as stored: one entry per token of the ``[UND | GEN]`` key stream."""
    return _StreamFields(
        sample_id=metadata.sample_id,
        frame_id=metadata.frame_id,
        view_id=metadata.view_id,
        is_noisy=metadata.is_noisy,
        is_control=metadata.is_control,
        is_und=_und_flags(metadata),
        timestamp=metadata.timestamp,
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
    attention_scope: AttentionScope,
    decomposed_temporal_window_seconds: float | None = None,
) -> MaskMod:
    """Return the multiview supertoken predicate, reading each side's own fields.

    Passing the key-stream fields on both sides gives the predicate over **key-stream
    coordinates**, which is what the block-level collapsing in :func:`build_block_mask`
    needs: it evaluates the predicate on group representatives, and those are key-stream
    positions on both sides. Passing the query fields on the left instead gives the
    ``mask_mod`` FlexAttention itself calls -- see :func:`_multiview_mask_mod`.

    All rules are gated on ``same_sample`` (block-diagonal packing). On top of
    that, using the per-token frame/view/modality metadata:

    * any GEN Q attends to every UND K of its sample -- the gen->und pass, which
      carries no further restriction;
    * every sensor Q attends every sensor K within whatever ``attention_scope`` admits
      (every view, its own view, or -- ``"decomposed"`` -- its own view or its own frame /
      temporal window), regardless of whether either token is conditioning;
    * sensor Q (conditioning or noisy alike) attends every control K of its **own view**, any
      frame;
    * control Q attends every control K of its **own view**, any frame, and no sensor K at all.

    Every rule here is a view rule, not a ``(frame, view)`` one. Every scope gives a query
    its own view at *any* frame, which is what makes ``"same_view"`` a clip's own attention
    rather than a set of stills, and a token's frame matters only through
    ``"decomposed"``, which adds every other view at the query's own instant and so
    registers the cameras against each other. Frames are free at the block level anyway,
    because :func:`_metadata_groups` already splits its runs per ``(item, frame, view)``
    cell, so the coarse predicate tells views and frames apart without a finer grouping.

    ``"decomposed"``'s temporal half is, by default, "the query's own frame index" --
    ``same_frame``, which only means the same instant when every sensor shares one clock.
    ``decomposed_temporal_window_seconds`` replaces that with a real-time comparison instead:
    a key is in the query's temporal reach when ``0 <= q_timestamp - k_timestamp <=
    decomposed_temporal_window_seconds`` (both bounds widened by a small float32 tolerance, since
    ``timestamp`` is computed from ``frame_id * seconds_per_frame`` and a pair meant to land
    exactly on a boundary can round a hair past it), i.e. the key was captured at or before the
    query, within that many seconds -- what lets a 7.5 Hz camera and a 10 Hz LiDAR sweep register
    against each other by real capture time rather than by a frame index neither shares. Left
    ``None``, the scope keeps its original same-frame-index meaning.

    A batch with no control item carries ``is_control`` all-``False``, which drops both
    control terms on its own -- see :class:`FlexMetadata`.

    Neither control rule can fire on a UND key: both require ``k_control``, which UND tokens
    carry as ``False``. The sensor rules are less tidy. Under ``"all_views"``,
    ``reaches_every_view`` alone satisfies ``sensor_pair_in_scope`` for a UND key, so a GEN
    query admits its own sample's UND keys through that term too; nothing observable changes,
    since ``gen_to_und`` already admits them unconditionally, but the UND quadrant is not
    exclusively ``gen_to_und``'s doing under that scope. Under the other two scopes the sensor
    rules compare ``view_id`` / ``frame_id`` (or ``timestamp``), where UND's ``-1`` sentinel
    matches no real GEN token (a UND key's ``timestamp`` sentinel of ``-1.0`` could satisfy
    ``0 <= q_ts - (-1.0) <= window`` for a query at ``q_ts <= window - 1``, but that path is
    unreachable: ``gen_to_und`` already admits every UND key unconditionally, and no rule ever
    needs the temporal term to do so).

    Padding carries ``-1`` in every id field, so real queries never see it and padded queries
    attend only to padding. The packer keeps at least one padded row on each stream once
    either is padded, so a padded query always has a padded key to attend to.
    """
    # The scope enters as tensors rather than as the string/float it arrives as: the closure
    # below is what Inductor traces, and a captured scalar is what the FlashAttention-4 backend
    # refuses to lower -- see :func:`_multiview_mask_mod` and the structural test that pins
    # it. Folding it into the expression rather than branching around it also leaves one path
    # through the predicate, so the element-level and block-level forms cannot come apart.
    # Both gates off leaves the query its own view, which every scope admits.
    device = q_fields.view_id.device
    reaches_every_view = torch.tensor(attention_scope == "all_views", device=device)
    is_decomposed = torch.tensor(attention_scope == "decomposed", device=device)
    has_temporal_window = torch.tensor(decomposed_temporal_window_seconds is not None, device=device)
    # The value is meaningless while has_temporal_window is False (the same_frame branch runs
    # instead), so 0.0 stands in rather than a sentinel that would need its own guard.
    temporal_window = torch.tensor(
        0.0 if decomposed_temporal_window_seconds is None else decomposed_temporal_window_seconds,
        dtype=torch.float32,
        device=device,
    )
    # timestamp is frame_id * seconds_per_frame in float32 (build_multiview_flex_metadata), so a
    # pair that is mathematically exactly on the window boundary -- the motivating case, aligning
    # sensors at rates like 7.5 Hz and 10 Hz -- can round a hair to either side of it. This
    # widens both edges of the ``[0, temporal_window]`` bound by a tolerance well above float32
    # rounding noise at realistic frame counts, and well below any window worth configuring in
    # seconds, so it only ever pulls a boundary pair back in, never admits an unrelated one.
    temporal_window_eps = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def pair_allowed(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        same_sample = q_fields.sample_id[q_idx] == kv_fields.sample_id[kv_idx]
        same_frame = q_fields.frame_id[q_idx] == kv_fields.frame_id[kv_idx]
        same_view = q_fields.view_id[q_idx] == kv_fields.view_id[kv_idx]
        q_control = q_fields.is_control[q_idx]
        k_control = kv_fields.is_control[kv_idx]

        # GEN Q -> UND K: the whole caption of the sample, conditioning and noisy alike.
        gen_to_und = kv_fields.is_und[kv_idx]
        # The "own instant" half of decomposed: same frame index by default, or a
        # non-negative, bounded gap in real capture time once a window is configured.
        timestamp_gap = q_fields.timestamp[q_idx] - kv_fields.timestamp[kv_idx]
        within_temporal_window = (timestamp_gap >= -temporal_window_eps) & (
            timestamp_gap <= temporal_window + temporal_window_eps
        )
        reaches_own_instant = is_decomposed & torch.where(has_temporal_window, within_temporal_window, same_frame)
        # Sensor attention uses the same view/instant footprint for conditioning and noisy
        # tokens, matching the dense base I2V/V2V attention pattern within that scope.
        in_scope = reaches_every_view | same_view | reaches_own_instant
        sensor_to_sensor = (~q_control) & (~k_control) & in_scope
        # sensor Q (any) -> control K: own view, any frame; control Q -> control K: own view,
        # any frame; control Q -> sensor K: never (no rule admits it).
        sensor_to_control = (~q_control) & k_control & same_view
        control_to_control = q_control & k_control & same_view

        return same_sample & (gen_to_und | sensor_to_sensor | sensor_to_control | control_to_control)

    return pair_allowed


def _multiview_mask_mod(metadata: FlexMetadata) -> MaskMod:
    """The predicate in FlexAttention's own index convention: ``q_idx`` numbers GEN from zero.

    This closure ends up inside the :class:`BlockMask` and is therefore the one Inductor
    traces, so what it captures is a hard constraint: tensors only, and tensors that start at
    offset zero. The obvious alternative -- keep the key-stream predicate and shift the row
    index, ``pair_allowed(q_idx + num_und, kv_idx)`` -- captures ``num_und`` as a Python int,
    and the FlashAttention-4 backend rejects the whole graph for it, because a captured scalar
    goes dynamic as soon as the enclosing region is compiled with dynamic shapes and CuteDSL
    cannot inline a symbolic value into its template. Slicing the fields per side does not get
    rid of that scalar either -- it moves it into the slice's storage offset, which goes
    dynamic just the same, hence the copy in :meth:`_StreamFields.tail`.

    Symbolically *shaped* fields are fine: a training run holds one graph over a hundred
    padded geometries on the flash backend, because the lengths the mask needs are ones the
    surrounding graph already has symbols for, so nothing is lifted into the subgraph. A
    caller without that relation available can still trip the check, as ``attention_test``
    exercises.
    """
    return _multiview_pair_predicate(
        _query_stream_fields(metadata),
        _key_stream_fields(metadata),
        metadata.attention_scope,
        metadata.decomposed_temporal_window_seconds,
    )


@dataclass(frozen=True)
class MaskItem:
    """One latent clip of one generation stream, as the multiview mask sees it.

    An item is a camera clip, a LiDAR range clip, or a control stream conditioning one of
    those. It contributes ``latent_t * patch_h * patch_w`` tokens laid out view-outer,
    frame-inner, spatial-innermost -- the camera-major order the multiview dataset
    concatenates its per-camera clips in.

    The per-item invariants are checked here, at construction, rather than by the builder
    that consumes a batch of these: an item whose latent axis does not divide into its
    views, or whose condition mask does not cover that axis, is malformed on its own terms,
    and catching it here names the one item rather than an index into a flattened list.

    Attributes:
        token_shape: ``(latent_t, patch_h, patch_w)``, where ``latent_t = num_views *
            frames_per_view`` counts the camera-major latent axis.
        condition_mask: ``[latent_t]`` over that same axis, a non-zero entry marking a
            conditioning (clean) frame -- what ``SequencePacker.pack_vision_tokens``
            appends to ``vision.condition_mask``. Any shape whose element count is
            ``latent_t`` will do; the builder reshapes it.
        num_views: cameras this item covers.
        view_offset: where its views start on its sample's view axis. Items on disjoint
            offsets are paired by no rule at all, which is what keeps a second sensor's
            clips off the camera rig's views -- see :func:`build_multiview_flex_metadata`.
        is_control: whether this is a control stream rather than a target conditioning on
            one. The mask confines a control item to its own view and keeps it out of the
            sensor rules entirely -- see :func:`_multiview_pair_predicate`.
        seconds_per_frame: real-world time between two consecutive latent frames of one of
            this item's cameras, i.e. the inverse of that sensor's latent frame rate. Only
            read to build :attr:`FlexMetadata.timestamp`, which only the ``"decomposed"``
            scope's ``decomposed_temporal_window_seconds`` form consults -- every other rule
            and the default ``"decomposed"`` form still compare ``frame_id`` directly. The
            default, ``1.0``, makes a token's timestamp equal its frame index, which is
            harmless as long as nothing reads it.
    """

    token_shape: tuple[int, ...]
    condition_mask: torch.Tensor
    num_views: int = 1
    view_offset: int = 0
    is_control: bool = False
    seconds_per_frame: float = 1.0

    def __post_init__(self) -> None:
        if self.num_views < 1 or self.latent_t % self.num_views != 0:
            raise ValueError(
                f"MaskItem has latent_t={self.latent_t}, which is not divisible by num_views={self.num_views}."
            )
        if self.condition_mask.numel() != self.latent_t:
            raise ValueError(
                f"MaskItem condition mask covers {self.condition_mask.numel()} frames, expected {self.latent_t}."
            )
        if self.seconds_per_frame <= 0:
            raise ValueError(f"MaskItem.seconds_per_frame must be positive, got {self.seconds_per_frame}.")

    @property
    def latent_t(self) -> int:
        """Frames on the camera-major latent axis, i.e. ``num_views * frames_per_view``."""
        return self.token_shape[0]

    @property
    def frames_per_view(self) -> int:
        """Latent frames one of this item's cameras contributes."""
        return self.latent_t // self.num_views

    @property
    def spatial_tokens(self) -> int:
        """Tokens one latent frame of one camera contributes."""
        return self.token_shape[1] * self.token_shape[2]

    @property
    def num_tokens(self) -> int:
        """GEN tokens this item owns."""
        return self.latent_t * self.spatial_tokens

    @property
    def view_grid(self) -> tuple[int, int]:
        """``(num_views, frames_per_view)``, the grid its tokens are laid out on."""
        return (self.num_views, self.frames_per_view)


def _check_view_grids_agree(items_per_sample: Sequence[Sequence[MaskItem]]) -> None:
    """Reject a sample whose items disagree on a view offset they share.

    An offset's views and frames are the coordinate system the mask's comparisons live in,
    so two items sharing an offset but dividing it differently would pair up unrelated
    cameras and leave the wider item's extra views outside the grid the others cover. The
    same goes for ``seconds_per_frame``: a shared offset means the items describe the same
    cameras, and :attr:`FlexMetadata.timestamp` is a function of ``(view_id, frame_id)``
    alone (see its docstring), so two items at that offset ticking at different rates would
    give one ``(view_id, frame_id)`` cell two different timestamps depending on which item's
    token landed there.

    Items on *disjoint* offsets are compared by no rule, so they are free to differ: that is
    what lets a joint camera + LiDAR sample carry a camera grid on one offset beside a range
    grid of another shape (and its own frame rate) on the next.
    """
    for sample_idx, sample_items in enumerate(items_per_sample):
        grid_by_view_offset: dict[int, tuple[int, int]] = {}
        seconds_per_frame_by_view_offset: dict[int, float] = {}
        for item_idx, item in enumerate(sample_items):
            # setdefault records the offset's first grid and returns it thereafter, so every
            # later item at that offset is compared against the one that established it.
            expected_grid = grid_by_view_offset.setdefault(item.view_offset, item.view_grid)
            if item.view_grid != expected_grid:
                raise ValueError(
                    "All items of a sample sharing a view offset must share the same "
                    f"(num_views, frames_per_view) grid: item {item_idx} of sample {sample_idx} at view "
                    f"offset {item.view_offset} has {item.view_grid}, expected {expected_grid}."
                )
            expected_seconds_per_frame = seconds_per_frame_by_view_offset.setdefault(
                item.view_offset, item.seconds_per_frame
            )
            # isclose rather than ==: seconds_per_frame is typically 1/fps, and two items
            # computing the "same" rate from different but equal fractions (e.g. 1/7.5 vs
            # 4/30.0) can otherwise land a ULP apart and trip this check spuriously.
            if not math.isclose(item.seconds_per_frame, expected_seconds_per_frame):
                raise ValueError(
                    "All items of a sample sharing a view offset must share the same "
                    f"seconds_per_frame: item {item_idx} of sample {sample_idx} at view offset "
                    f"{item.view_offset} has {item.seconds_per_frame}, expected {expected_seconds_per_frame}."
                )


def build_multiview_flex_metadata(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    items_per_sample: Sequence[Sequence[MaskItem]],
    device: torch.device,
    num_und: int = 0,
    causal_offsets: torch.Tensor | None = None,
    attention_scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
) -> FlexMetadata:
    """Build key-stream metadata for camera-major multiview transfer items.

    ``items_per_sample`` is one list of :class:`MaskItem` per packed sample, in packed
    order, describing every generation stream the sample carries: its camera clips, its
    LiDAR range clips, and whichever of those are control streams. The nesting is the
    grouping, so nothing here has to check that a flat item list and a per-sample count
    agree -- and :class:`MaskItem` has already checked each item on its own terms.

    What is left to check is the one invariant that spans items: those of a sample
    **sharing a view offset** have to agree on their ``(num_views, frames_per_view)``
    grid, since that is the coordinate system the mask compares views and frames in. Items
    on disjoint offsets are paired by no rule and are free to differ -- a joint camera +
    LiDAR sample carries a 71-latent camera pair on view 0 beside a 94-sweep range pair on
    view 1. See :func:`_check_view_grids_agree`.

    Args:
        seq_len: block-padded GEN sequence length, i.e. the query count. The returned
            fields are ``[num_und + seq_len]``, covering the fused ``[UND | GEN]`` key
            stream.
        full_q_offsets: cumulative per-sample GEN offsets, ``[len(items_per_sample) + 1]``;
            ``full_q_offsets[-1]`` is the real (unpadded) GEN token count, which the items'
            own token counts have to add up to.
        items_per_sample: the items each sample owns, in packed order. See
            :class:`MaskItem` for what one describes and which of its fields the mask
            rules read.
        device: device for the returned tensors.
        num_und: block-padded UND (causal) stream length, which prefixes the key
            stream. 0 leaves the metadata GEN-only, for a square self-attention mask.
        causal_offsets: cumulative per-sample UND offsets, ``[num_samples + 1]``;
            required when ``num_und`` is non-zero, since the UND rule is "same sample"
            and nothing else. ``causal_offsets[-1]`` is the real UND token count, so
            everything past it is padding.
        attention_scope: which same-kind (sensor) tokens of its sample a token reaches --
            those of every view, of its own view, or of its own view or instant. See
            ``AttentionScope``; :class:`FlexMetadata` carries it on to both forms of the
            predicate. Never widens a control token's reach, which is always its own view.
            ``"decomposed"`` is rejected once more than one view offset is in play and
            ``decomposed_temporal_window_seconds`` is ``None``, because that combination pairs
            across views by frame index and a second sensor does not share the camera's frame
            index -- 7.5 Hz camera latents against 10 Hz sweeps. Passing a window lifts that
            restriction: the temporal half then compares real capture time instead, which is
            defined across sensors.
        decomposed_temporal_window_seconds: with ``attention_scope="decomposed"``, replaces
            "the query's own frame index" with "any key within this many seconds at or before
            the query's real capture time" for the scope's temporal half -- see
            :func:`_multiview_pair_predicate`. ``None`` (the default) keeps the frame-index
            form, which is also the only form :class:`MaskItem`'s default
            ``seconds_per_frame=1.0`` produces the same answer for. Ignored outside
            ``"decomposed"``.

    Returns:
        :class:`FlexMetadata` whose per-token fields are each ``[num_und + seq_len]``: the
        UND tokens (sample ids only, ``-1`` / ``False`` in every multiview field), then the
        real GEN tokens in packed order, each stream followed by ``-1`` / ``-1.0`` sentinels
        (``False`` for ``is_noisy`` and ``is_control``) across its trailing pad.

    Raises:
        ValueError: if the items of one sample sharing a view offset disagree on the
            ``(num_views, frames_per_view)`` grid or on ``seconds_per_frame``, if the items'
            token counts do not add up to ``full_q_offsets[-1]`` or exceed ``seq_len``, if
            ``num_und`` is non-zero without ``causal_offsets``, or if ``attention_scope`` is
            ``"decomposed"`` with more than one view offset present and
            ``decomposed_temporal_window_seconds`` is ``None``.
    """
    if num_und and causal_offsets is None:
        raise ValueError(
            f"A fused key stream with {num_und} UND tokens needs causal_offsets to label them by "
            "sample; without it every UND key would look like padding to the mask."
        )
    # The two streams label their tokens with sample ids drawn from their own offsets, and the
    # predicate compares those ids across the streams (``same_sample``). That only means anything
    # if both number their samples the same way, which holds because the packer emits exactly one
    # causal and one full split per sample (``pack_text_tokens`` / ``finish_sample``). Nothing here
    # can see that, so check it: a pack that broke the pairing -- an AR no-text pack reaching this
    # path, say -- would pair GEN and UND tokens of unrelated samples with no other symptom.
    if causal_offsets is not None and causal_offsets.shape[0] != full_q_offsets.shape[0]:
        raise ValueError(
            f"The UND and GEN streams disagree on their sample count: causal_offsets describes "
            f"{causal_offsets.shape[0] - 1} samples and full_q_offsets {full_q_offsets.shape[0] - 1}. "
            "The multiview mask labels both streams from their own offsets and compares those "
            "labels, so the two have to number the same samples."
        )
    _check_view_grids_agree(items_per_sample)
    items = [item for sample_items in items_per_sample for item in sample_items]

    # decomposed is spatial + temporal attention decomposed into two terms: a noisy token
    # reaches its own view at every frame, and every view at its own frame index (or, with a
    # temporal window configured, at its own real capture time). A second sensor on its own
    # view offset does
    # not share the camera's frame index -- 7.5 Hz camera latents vs 10 Hz sweeps -- so the
    # frame-index form of the temporal half would pair unrelated moments; the timestamp form
    # does not have that problem, since it compares real time rather than an index each sensor
    # numbers its own way, which is exactly what a window opts into.
    #
    # The test is how many view ranges are in play, not whether any of them is non-zero: a
    # single range renumbered off zero shifts every view id by a constant, which same_view
    # (an equality) and same_frame (blind to the view) both ignore. Rejecting that would
    # forbid a layout the scope handles correctly.
    view_ranges = {item.view_offset for item in items}
    if attention_scope == "decomposed" and len(view_ranges) > 1 and decomposed_temporal_window_seconds is None:
        raise ValueError(
            "attention_scope='decomposed' is not allowed on a joint camera + "
            "LiDAR pack without decomposed_temporal_window_seconds: camera frames and LiDAR "
            "sweeps do not correspond by index. Use 'all_views', 'same_view', or set "
            "decomposed_temporal_window_seconds to compare by real capture time instead."
        )

    frame_ids: list[torch.Tensor] = []
    view_ids: list[torch.Tensor] = []
    timestamps: list[torch.Tensor] = []
    noisy_flags: list[torch.Tensor] = []
    control_flags: list[torch.Tensor] = []
    # Purely constructive: every field below is per item, and every invariant an item could
    # violate has been checked already, so this needs no per-sample scope of its own.
    for item in items:
        num_views, frames_per_view = item.view_grid
        spatial_tokens = item.spatial_tokens
        # Frame index cycles within each view; view index is constant across a view's
        # whole frame run. Both expand to one entry per token: [item_tokens].
        item_frame_ids = torch.arange(frames_per_view, device=device).repeat(num_views)  # [num_views*frames_per_view]
        frame_ids.append(item_frame_ids.repeat_interleave(spatial_tokens))  # [item_tokens]
        view_ids.append(
            (item.view_offset + torch.arange(num_views, device=device)).repeat_interleave(
                frames_per_view * spatial_tokens
            )
        )  # [item_tokens]
        # Real capture time of a frame index, at this item's own rate -- see
        # MaskItem.seconds_per_frame and FlexMetadata.timestamp.
        timestamps.append(
            (item_frame_ids.to(torch.float32) * item.seconds_per_frame).repeat_interleave(spatial_tokens)
        )  # [item_tokens]

        condition_mask = item.condition_mask.to(device=device, dtype=torch.bool)  # [latent_t]
        # Per-frame flag -> per-token flag, every token of a frame sharing the frame's state.
        is_conditioning = condition_mask.reshape(item.latent_t).repeat_interleave(spatial_tokens)  # [item_tokens], bool
        noisy_flags.append(~is_conditioning)  # [item_tokens], bool
        control_flags.append(
            torch.full(is_conditioning.shape, item.is_control, device=device, dtype=torch.bool)
        )  # [item_tokens]

    frame_id = torch.cat(frame_ids)  # [real_token_count]
    view_id = torch.cat(view_ids)  # [real_token_count]
    timestamp = torch.cat(timestamps)  # [real_token_count], float
    is_noisy = torch.cat(noisy_flags)  # [real_token_count], bool
    is_control = torch.cat(control_flags)  # [real_token_count], bool
    real_token_count = frame_id.shape[0]
    # These counts come from token_shapes while the offsets come from the packer's full splits.
    # If they disagree, _build_stream_sample_ids draws the padding boundary somewhere else than
    # this metadata does, so real tokens read as padding or padding reads as a real conditioning
    # token. Reading the last offset costs one device sync per forward, which is affordable
    # because this runs outside the compiled decoder layers.
    packed_token_count = int(full_q_offsets[-1])
    if real_token_count != packed_token_count:
        raise ValueError(
            f"Multiview metadata covers {real_token_count} GEN tokens but the pack holds "
            f"{packed_token_count}; the items and the packed full-attention splits disagree."
        )
    if real_token_count > seq_len:
        raise ValueError(f"Multiview metadata has {real_token_count} tokens, exceeding GEN sequence length {seq_len}.")

    pad = seq_len - real_token_count
    if pad:
        sentinel = torch.full((pad,), -1, device=device, dtype=torch.long)  # [pad]
        timestamp_sentinel = torch.full((pad,), -1.0, device=device, dtype=torch.float32)  # [pad]
        frame_id = torch.cat((frame_id, sentinel))  # [seq_len]
        view_id = torch.cat((view_id, sentinel))  # [seq_len]
        timestamp = torch.cat((timestamp, timestamp_sentinel))  # [seq_len]
        is_noisy = torch.cat((is_noisy, torch.zeros(pad, device=device, dtype=torch.bool)))  # [seq_len], bool
        is_control = torch.cat((is_control, torch.zeros(pad, device=device, dtype=torch.bool)))  # [seq_len], bool

    sample_id = _build_stream_sample_ids(full_q_offsets, seq_len, device)  # [seq_len]

    if num_und:
        assert causal_offsets is not None  # guarded above; narrows the type for the checker.
        # UND keys join the front of the stream carrying nothing but their sample: the
        # multiview fields are what the GEN rules match on, and holding them at -1 / False is
        # what keeps those rules from firing on this quadrant.
        und_sentinel = torch.full((num_und,), -1, device=device, dtype=torch.long)  # [num_und]
        und_timestamp_sentinel = torch.full((num_und,), -1.0, device=device, dtype=torch.float32)  # [num_und]
        sample_id = torch.cat((_build_stream_sample_ids(causal_offsets, num_und, device), sample_id))
        frame_id = torch.cat((und_sentinel, frame_id))  # [num_und+seq_len]
        view_id = torch.cat((und_sentinel, view_id))  # [num_und+seq_len]
        timestamp = torch.cat((und_timestamp_sentinel, timestamp))  # [num_und+seq_len]
        is_noisy = torch.cat((torch.zeros(num_und, device=device, dtype=torch.bool), is_noisy))  # bool
        is_control = torch.cat((torch.zeros(num_und, device=device, dtype=torch.bool), is_control))  # bool

    return FlexMetadata(
        seq_len=num_und + seq_len,
        sample_id=sample_id,  # [num_und+seq_len]
        frame_id=frame_id,  # [num_und+seq_len]
        view_id=view_id,  # [num_und+seq_len]
        is_noisy=is_noisy,  # [num_und+seq_len], bool
        is_control=is_control,  # [num_und+seq_len], bool
        timestamp=timestamp,  # [num_und+seq_len], float
        num_und=num_und,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
    )


def _check_block_aligned(seq_len: int, stream: str, alignment: int) -> None:
    """Raise if ``seq_len`` is not a multiple of ``alignment``.

    Both streams of the fused key layout have to be aligned, not just their total: the
    UND prefix ends on a block boundary only if its own length is a multiple of the
    block size, and a block straddling the boundary would be a partial one for every
    query, costing the fully-unmasked fast path on the whole gen->und quadrant.

    ``alignment`` is the backend's block rather than a fixed number:
    :func:`triton_backend_block_size` on Triton, or FlashAttention-4's coarser query tile on
    Blackwell (:func:`flash_backend_block_size`). The latter is always a multiple of the
    former, so naming the backend's own block is both sufficient and the honest requirement;
    raising about the 128 floor instead would understate it.
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

        The GEN stream supplies the mask's rows, so it pads to the query block -- 256 on
        Blackwell FA4, coarser than the granularity the metadata itself is built at. That
        block is always a multiple of :func:`triton_backend_block_size`'s, so padding to it
        satisfies both.
        """
        return self.block_size[0]

    @property
    def causal_seq_alignment(self) -> int:
        """``build_packed_sequence``'s UND padding multiple for this backend.

        The UND stream is keys only, so it answers to the key block, 128 on both backends --
        FA4 buys its coarser tiles in the query dimension alone. Padding it to the query block
        would also be correct, just more padding for nothing: all the boundary between the two
        streams has to land on is a key-block boundary.
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
    return metadata_run_groups(
        (
            metadata.sample_id,
            metadata.frame_id,
            metadata.view_id,
            metadata.is_noisy,
            metadata.is_control,
            _und_flags(metadata),
        ),
        device=device,
    )


def build_block_mask(
    metadata: FlexMetadata,
    device: torch.device,
    block_size: tuple[int, int],
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` from precomputed flex metadata.

    Uses the multiview supertoken ``mask_mod``; the multiview fields
    (``frame_id`` / ``view_id`` / ``is_noisy`` / ``is_control``) must be populated on
    ``metadata``.

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
    pair_allowed = _multiview_pair_predicate(
        key_fields, key_fields, metadata.attention_scope, metadata.decomposed_temporal_window_seconds
    )
    mask_mod = _multiview_mask_mod(metadata)
    group_id, representatives = _metadata_groups(metadata, device)
    # Queries are the GEN tail of the key stream, so their presence comes from that slice.
    query_group_id = group_id if metadata.num_und == 0 else group_id[metadata.num_und :]
    return build_block_mask_from_metadata_runs(
        q_group_id=query_group_id,
        kv_group_id=group_id,
        q_representatives=representatives,
        kv_representatives=representatives,
        pair_allowed=pair_allowed,
        mask_mod=mask_mod,
        q_len=metadata.q_len,
        kv_len=metadata.seq_len,
        device=device,
        block_size=block_size,
    )


def build_multiview_block_mask(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    items_per_sample: Sequence[Sequence[MaskItem]],
    device: torch.device,
    block_size: tuple[int, int],
    num_und: int = 0,
    causal_offsets: torch.Tensor | None = None,
    attention_scope: AttentionScope = "all_views",
    decomposed_temporal_window_seconds: float | None = None,
) -> BlockMask:
    """Build the GEN-tower :class:`BlockMask` for camera-major multiview items.

    The entry point for the packed-batch path: it runs
    :func:`build_multiview_flex_metadata` and :func:`build_block_mask` back to back so
    the caller only handles the mask. See :class:`MaskItem` for what one item describes,
    and those two functions for the token layout the metadata encodes, for why the mask has
    to be built outside the decoder layers, for what ``block_size`` selects, for what
    ``num_und`` / ``causal_offsets`` add, for what ``attention_scope`` lets same-kind
    (sensor) tokens reach, and for what ``decomposed_temporal_window_seconds`` changes about
    the ``"decomposed"`` scope's temporal half.

    The two stages stay separately callable because the metadata layout is checked on
    CPU in the unit tests, independently of the block-mask construction.
    """
    metadata = build_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        items_per_sample=items_per_sample,
        device=device,
        num_und=num_und,
        causal_offsets=causal_offsets,
        attention_scope=attention_scope,
        decomposed_temporal_window_seconds=decomposed_temporal_window_seconds,
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
            decided ``block_mask``'s block size; its ``kernel_options`` select the kernels.
            The whole object rather than those options alone, because the two only mean
            anything together -- FlashAttention-4's options are correct for a mask built at
            that backend's block size and silently wrong for any other -- so passing it lets
            the check below compare the two. Dynamo guards on the ``kernel_options`` dict, so
            each distinct value compiles its own kernel.
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
    # The mask was built at the block size the kernel steps in -- 256 query rows on Blackwell
    # FA4 against 128 on Triton -- so it carries the geometry the streams are measured against.
    mask_q_block, mask_kv_block = block_mask.BLOCK_SIZE
    if (mask_q_block, mask_kv_block) != backend.block_size:
        # The kernels below step backend.block_size whatever the mask says, and an over-fine
        # mask handed to FA4 attends to the wrong tokens rather than raising, so this is the
        # one mismatch in this function that nothing downstream would report.
        raise ValueError(
            f"block_mask was built at {(mask_q_block, mask_kv_block)} blocks, but the {backend.name} backend "
            f"steps {backend.block_size}; build the mask from the same FlexBackend that runs it."
        )
    # One check per stream, each against the block that tiles it, as build_block_mask makes
    # them: a partial trailing block would silently drop the rows the mask has no entry for.
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
