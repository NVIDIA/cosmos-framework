# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The ``"decomposed"`` multiview attention scope, as dense kernels instead of a mask.

Split out of ``attention.py`` because it is a self-contained alternative to that module's
generation attention rather than another branch of it: it builds no ``BlockMask``, needs no
FlexAttention backend, and the geometry it folds by is described by its own plan. ``attention``
imports it for :func:`~...attention.dispatch_attention` to route to, and nothing here imports
``attention`` back -- which is why :class:`~...merge_bridge.MergeAttentionsBridge` lives in its
own module rather than in either.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cosmos_framework.model.attention import attention, merge_attentions
from cosmos_framework.configs.base.defaults.multiview_attention import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_NONE,
    AttentionScope,
    CaptionAccess,
    MultiviewAttentionConfig,
    resolve_caption_scope,
)
from cosmos_framework.model.generator.mot.activation_marks import mark_next_activation
from cosmos_framework.model.generator.mot.merge_bridge import BridgeFn, MergeAttentionsBridge
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    get_num_real_samples,
)

# The scopes these folds express as a partition of the GEN stream, and so can run without a
# mask. ``"same_view"`` is one partition and is exact against its mask. ``"decomposed"`` is two
# overlapping ones, which is why it is deliberately *not* the same attention as its mask -- see
# ``multiview_maskless_attention``. ``"all_views"`` is absent because it is not a partition at all:
# it is one unmasked pass over each whole sample, which these folds do not build.
MASKLESS_ATTENTION_SCOPES: tuple[AttentionScope, ...] = ("decomposed", "same_view")


def maskless_unavailable_reason(config: MultiviewAttentionConfig) -> str | None:
    """Why a config rules these folds out, or ``None`` when it does not.

    Lives beside the folds rather than on the config because every condition is a fact about
    what they express -- which scopes are a partition -- and a config dataclass that answered it
    would be asserting implementation knowledge it does not own.

    ``control_attends_sensor`` is deliberately *not* a condition here, though it was one. A
    control item shares its target's view group, so with the flag off a control query wants a
    narrower key set than a sensor query on the same view -- and one unmasked pass over that
    group as a single varlen segment cannot give two query roles different keys. Two segments
    can, which is what the folds now cut it into: the group's sensor tokens keyed against the
    whole group, its control tokens against the group's control tokens alone. Both values of the
    flag are therefore expressed exactly, and which one a plan was built for is recorded on it.
    See :func:`build_multiview_maskless_plan`.

    Config-level only, and deliberately so. Whether a *batch* can be folded is decided per
    forward by ``_multiview_maskless_geometry``, which raises rather than substituting a mask.
    So "auto" reads this to fix one attention for the whole run, and a batch that then turns out
    not to fit is an error rather than a quiet change of what is being trained.

    Phrased as a reason rather than a bool so ``"maskless"`` can fail with the cause, the way
    ``flash_backend_unavailable_reason`` does for FA4.
    """
    if config.mask.decomposed_temporal_window_seconds is not None:
        return (
            f"decomposed_temporal_window_seconds={config.mask.decomposed_temporal_window_seconds} "
            "asks for the sliding-window form of the scope, which is directed and overlapping. "
            "Only an equivalence relation can be an unmasked pass, so no partition of the "
            "stream reproduces it"
        )
    if config.mask.attention_scope not in MASKLESS_ATTENTION_SCOPES:
        return (
            f"attention_scope={config.mask.attention_scope!r} is not a scope the maskless folds "
            f"express; they cover {MASKLESS_ATTENTION_SCOPES}. 'all_views' is one unmasked pass "
            "over each whole sample rather than a partition of one, so it is a mask rule here "
            "and not a fold"
        )
    return None


@dataclass(frozen=True)
class MultiviewMasklessPlan:
    """How :func:`multiview_maskless_attention` folds one batch's GEN stream.

    The per-sample geometry, plus the index tensors the ragged path needs. Built once per
    forward by :func:`build_multiview_maskless_plan`, outside the compiled and
    activation-checkpointed decoder layers and for the same reasons the multiview block mask
    is: the indices are data-dependent, every layer shares the one answer, and rebuilding
    them per layer would be that index math over again for no new information.

    Every batch is addressed the same way, by varlen ranges over the packed stream. A uniform
    batch could ride the batch axis instead, its groups being all of one length, but that form
    is gone: it made the trim of the pack's padding structural -- a ``view()`` into
    ``[V, F*S, ...]`` only factors on the real token count -- and kept two shapes of every pass
    alive for one case.

    Attributes:
        attention_scope: the scope this plan was folded for. ``"decomposed"`` keeps both sensor
            partitions; ``"same_view"`` builds no cross-instant partition at all, so the pass is
            skipped and a query reaches only its own view. Recorded rather than inferred so a
            plan says which attention it describes.
        control_attends_sensor: the control rule this plan was folded for, recorded for the same
            reason ``attention_scope`` is. ``True`` leaves a same-view group as one varlen
            segment attending itself; ``False`` cuts it into a sensor segment and a control one,
            which is the only thing the flag changes here. What the caller stated, not what the
            batch needed: a batch marking no control item has no control query to narrow, so it
            is folded the same way under either value and still records the one its run means.
            ``True`` where the caller stated nothing, which only a batch of that kind may do.
        num_views: cameras each sample's item covers, which its ``latent_t`` divides into.
        token_shapes: each sample's ``(latent_t, patch_h, patch_w)``, ``latent_t`` counting
            the camera-major latent axis (``num_views * frames_per_view``).
        num_gen_tokens: real GEN tokens the batch contributes.
        padded_gen_tokens: the GEN stream's padded length, which the pack must agree with. The
            partitions below cover all of it: the padding is a group of its own, so a padded
            query only ever meets a padded key and every row the kernels are handed is written.
            Addressing the padded stream rather than trimming to the real one is what keeps this
            plan in the pack's own coordinates, so the pack's offsets and maximum lengths can be
            used as they stand instead of being converted at each use.
        same_view_offsets: cumulative ``(sample, view)`` group boundaries over the GEN stream,
            in packed order.
        same_view_max_len: longest ``(sample, view)`` group, for varlen kernel sizing.
        same_view_q_gather: the same-view pass's query side under
            ``control_attends_sensor=False``: every group's sensor tokens in group order, then
            every group's control tokens in group order, so a group is two varlen segments rather
            than one. A permutation of the padded stream, not a subset -- see
            :func:`_control_split_fields`. ``None`` -- every other batch -- leaves the pass keyed
            by ``same_view_gather`` on both sides, which is a group attending itself. Set only
            where the split is in play: the flag off *and* some item marked control. The three
            ``same_view_*`` fields above stay the whole partition either way, because the
            gen->und pass borrows them for *its* queries and a control token reads its captions
            whatever this flag says.
        same_view_q_offsets: cumulative per-segment boundaries into ``same_view_q_gather``.
        same_view_q_max_len: longest query segment.
        same_view_kv_gather: the key side, segment for segment with the above: a sensor segment
            is keyed against its whole group, a control segment against the group's control
            tokens alone. Longer than the stream, since a group's control tokens appear in both
            of its segments' keys.
        same_view_kv_offsets: cumulative per-segment boundaries into ``same_view_kv_gather``.
        same_view_kv_max_len: longest key segment.
        cross_view_offsets: cumulative ``(sample, frame)`` group boundaries, in *gathered*
            (frame-major) order.
        cross_view_max_len: longest ``(sample, frame)`` group.
        cross_view_gather: the sensor tokens' packed indices, in instant-major order. ``None``
            when the cross-instant partition is empty.
        cross_view_empty: whether no sample contributes to the cross-instant partition, so
            that pass is skipped outright and the merge runs on two branches. True exactly when
            every sample owns a single same-view group -- see :func:`build_multiview_maskless_plan`.
        caption_gather: the caption tokens' indices into the causal stream, one contiguous run
            per same-view group, in that partition's order. ``None`` unless the batch carries
            per-view captions, in which case the gen->und pass keys each sample's GEN tokens
            against that sample's whole causal run and needs no per-group keys.
        caption_offsets: cumulative per-group boundaries into ``caption_gather``.
        caption_max_len: longest run of captions one group reads.
        gen_to_und_gather: the GEN tokens that read their sample's caption, as packed indices
            in packed order, for the sample-level caption layout only. ``None`` -- the usual
            case -- leaves the gen->und pass keying the whole GEN stream, which is what every
            token reading the one caption means. Set only under
            ``lidar_attends_captions=False``, where the LiDAR tokens read no caption and so
            cannot be part of a pass keyed per *sample*: the pass then runs over this subset and
            is scattered back, leaving the LiDAR rows at a log-sum-exp the merge gives no
            weight. The per-view layout needs none of this -- there a LiDAR group simply takes
            an empty run of captions, see :func:`_caption_partition`.
        gen_to_und_max_len: longest run of caption-reading GEN tokens one group owns, the pack's
            trailing pad segment included. The *offsets* are not stored beside it: they have to
            agree in length with the causal stream's, whose segment count is the pack's rather
            than this plan's, so the pass derives them from the pack's own GEN offsets by
            searchsorted -- see :func:`multiview_maskless_attention`.
        cross_view_inverse: its inverse permutation, frame-major back to packed.
    """

    attention_scope: str
    control_attends_sensor: bool
    num_views: tuple[int, ...]
    token_shapes: tuple[tuple[int, int, int], ...]
    seconds_per_frame: tuple[float, ...]
    items_per_sample: tuple[int, ...]
    is_control: tuple[bool, ...]
    view_axis: tuple[int, ...]
    num_gen_tokens: int
    padded_gen_tokens: int = 0
    same_view_offsets: torch.Tensor | None = None
    same_view_max_len: int = 0
    same_view_gather: torch.Tensor | None = None
    same_view_q_gather: torch.Tensor | None = None
    same_view_q_offsets: torch.Tensor | None = None
    same_view_q_max_len: int = 0
    same_view_kv_gather: torch.Tensor | None = None
    same_view_kv_offsets: torch.Tensor | None = None
    same_view_kv_max_len: int = 0
    cross_view_offsets: torch.Tensor | None = None
    cross_view_max_len: int = 0
    cross_view_gather: torch.Tensor | None = None
    cross_view_empty: bool = False
    caption_gather: torch.Tensor | None = None
    caption_offsets: torch.Tensor | None = None
    caption_max_len: int = 0
    caption_q_gather: torch.Tensor | None = None
    caption_q_offsets: torch.Tensor | None = None
    caption_q_max_len: int = 0
    gen_to_und_gather: torch.Tensor | None = None
    gen_to_und_max_len: int = 0


def _cumulative_offsets(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """``[len(lengths)+1]`` int32 cumulative offsets, the layout the varlen kernels take."""
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return offsets


def _partition(group_ids: torch.Tensor) -> tuple[torch.Tensor | None, list[int]]:
    """A partition from per-token group ids: ``(gather or None, group lengths)``.

    ``None`` means the groups already tile the stream in packed order, so the pass reads the
    stream as it stands and its output needs no reordering -- the case a batch of one
    single-item sample is in for both partitions, and every non-transfer batch is in for the
    same-view one.
    """
    gather = torch.argsort(group_ids, stable=True)  # [N]
    lengths = torch.bincount(torch.unique_consecutive(group_ids[gather], return_inverse=True)[1]).tolist()
    identity = torch.equal(gather, torch.arange(group_ids.shape[0], device=group_ids.device))
    return (None if identity else gather), lengths


def _control_split_fields(
    sensor_runs: dict[int, list[torch.Tensor]],
    control_runs: dict[int, list[torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    """The same-view pass re-cut by what a token is, for ``control_attends_sensor=False``.

    That flag leaves a same-view group with two query roles wanting different keys. A sensor
    query reaches the whole group -- its own view's sensor tokens by the attention scope, its
    view's control tokens by the mask's sensor->control rule, neither of which the flag touches.
    A control query reaches that view's control tokens and nothing else, the withheld direction
    being exactly what the flag names. One rectangle cannot say that, but one *varlen call* can:
    a group simply becomes two varlen segments, a sensor one keyed against the whole group and a
    control one keyed against the group's control tokens. So this returns a query partition and a
    key partition, segment for segment -- still one pass, one kernel and one merge branch, which
    is why the fold's shape is unchanged and only its offsets are.

    The query side is every group's sensor run in group order, then every group's control run in
    group order. That is a **permutation** of the padded GEN stream rather than a subset of it:
    every token is a sensor token or a control token, so every row of the pass is written and
    none is written twice. The key side duplicates a group's control tokens -- once inside the
    whole-group run the sensor segment reads, once as the control segment's own keys -- which a
    gather may do freely: the forward reads each segment independently and the backward
    accumulates into the duplicated rows, the same thing :func:`_caption_partition` relies on.

    Each argument maps a same-view group id to that group's token runs in packed order, one per
    item that covers the group. A group absent from one of them owns no token of that kind: a
    view carrying no control stream is missing from ``control_runs``, which is every group of an
    ordinary batch, and a view whose only items are control is missing from ``sensor_runs`` --
    what a control item covering more views than the target it conditions produces. Such a group
    contributes the one segment it has, which is exact rather than approximate: with no sensor
    token there is no sensor query of that view to key against anything.

    Key order *within* a segment is free, attention summing over a key set, so a group's keys are
    its sensor runs followed by its control runs rather than being re-sorted back into packed
    order. Query order within a segment is not free in the same way, but neither does it have to
    be packed order: the scatter back is by the query gather itself, which inverts whatever order
    it is in.

    The caller reaches here only for a batch that marks a control item, so the control segments
    are never empty; and a sample whose items are *all* control is refused before this, so
    neither are the sensor ones.
    """
    groups = sorted(set(sensor_runs) | set(control_runs))
    # Sensor segments first, then control ones. Any order would do -- the two sides are read
    # segment for segment -- but blocking them keeps a query gather whose two halves are each in
    # group order, which is the order every other partition here is in.
    segments = [(sensor_runs[g], sensor_runs[g] + control_runs.get(g, [])) for g in groups if g in sensor_runs]
    segments += [(control_runs[g], control_runs[g]) for g in groups if g in control_runs]

    queries = [torch.cat(q) for q, _ in segments]
    keys = [torch.cat(kv) for _, kv in segments]
    q_lens = [int(run.shape[0]) for run in queries]
    kv_lens = [int(run.shape[0]) for run in keys]
    return {
        "same_view_q_gather": torch.cat(queries),  # [N_gen], a permutation
        "same_view_q_offsets": _cumulative_offsets(q_lens, device),
        "same_view_q_max_len": max(q_lens),
        "same_view_kv_gather": torch.cat(keys),  # [N_gen + control tokens], duplicating those
        "same_view_kv_offsets": _cumulative_offsets(kv_lens, device),
        "same_view_kv_max_len": max(kv_lens),
    }


def _caption_partition(
    captions: Sequence[Sequence[tuple[int, int]]] | None,
    view_group: dict[tuple[int, int, int], int],
    group_sample: dict[int, int],
    group_access: dict[int, CaptionAccess],
    device: torch.device,
    pad_group: bool = False,
) -> tuple[torch.Tensor | None, list[int] | None]:
    """Which captions each same-view group reads, as a gather into the causal stream.

    ``(None, None, 0)`` unless some sample packs more than one caption. A batch whose samples
    each pack one caption needs nothing here: every GEN token of a sample reads that caption,
    which the per-sample gen->und pass already gives it without replicating anything.

    The rule is the mask's, and is *taken* from the same place rather than worked out again:
    ``group_access`` carries each same-view group's :data:`CaptionAccess` -- what the caller's
    items say they are -- and :func:`resolve_caption_scope` turns it into the same
    ``CAPTION_SCOPE_*`` the mask puts on every token. A camera reads the caption written for its
    view and a sweep every caption of its sample or none of them. A sample-level caption
    (``view_id`` ``-1``) is read by every group of its sample, which is what makes the
    single-caption case a special case of this one rather than a different rule -- though no
    *pack* brings one here, since the network refuses a pack captioned both ways before it
    builds anything from the captions (``multiview_attention.reject_mixed_caption_layouts``,
    which sits above this module rather than being reachable from it).

    Deriving it here instead -- from the view axis, as this did -- is what let these folds ignore
    ``lidar_attends_captions`` entirely while the mask honoured it: one rule, expressed twice,
    with nothing to keep the two in step.

    ``CAPTION_SCOPE_NONE`` comes out as an empty run, and is the *only* group that may: a view
    meant to read a caption that finds none is a pack whose caption view ids do not cover its
    camera views, which is refused here rather than dropped from the pass. Only the per-view
    layout reaches here; the sample-level one is a subset of the GEN stream instead, since its
    pass is keyed per sample rather than per group -- see
    ``MultiviewMasklessPlan.gen_to_und_gather``.

    The runs come out in same-view group order, so the pass can key its gathered queries
    against them with no further reordering.
    """
    if not captions or all(len(sample_captions) <= 1 for sample_captions in captions):
        return None, None

    # Where each sample's captions start in the causal stream, which the packer lays down
    # sample by sample in the order the caption lists record.
    spans: list[list[tuple[int, int, int]]] = []
    position = 0
    for sample_captions in captions:
        sample_spans: list[tuple[int, int, int]] = []
        for view_id, num_tokens in sample_captions:
            sample_spans.append((view_id, position, num_tokens))
            position += num_tokens
        spans.append(sample_spans)

    runs: list[torch.Tensor] = []
    lengths: list[int] = []
    for group in sorted(group_sample):
        sample = group_sample[group]
        view = next(v for (s, a, v), gid in view_group.items() if gid == group)
        # Per-view captions by construction: this function returns early otherwise.
        scope = resolve_caption_scope(group_access[group], per_view_captions=True)
        chosen = (
            []
            if scope == CAPTION_SCOPE_NONE
            else [
                (start, num_tokens)
                for view_id, start, num_tokens in spans[sample]
                if scope == CAPTION_SCOPE_ALL or view_id in (-1, view)
            ]
        )
        if not chosen and scope != CAPTION_SCOPE_NONE:
            # A group that is *meant* to read a caption and found none. Only a
            # ``CAPTION_SCOPE_NONE`` group -- a sweep under ``lidar_attends_captions=False`` --
            # legitimately reads nothing; a camera reads the caption written for its view or the
            # sample-level one, and a sweep reading all of them reads its sample's. Reaching here
            # means the pack's caption view ids do not cover the sample's camera views, which
            # would silently leave this group out of the gen->und pass (see
            # ``_restrict_to_caption_readers``) and train it with no text conditioning. The mask
            # refuses the same layout in ``_build_und_view_ids``; this is the folds' half of that
            # one rule.
            raise ValueError(
                f"Sample {sample}'s view {view} reads no caption at all: the sample packs captions "
                f"for views {[view_id for view_id, _, _ in spans[sample]]}, none of which is view "
                f"{view} or a sample-level caption (view id -1). Per-view captions must cover each "
                "camera view of their sample."
            )
        indices = [torch.arange(start, start + num_tokens, device=device) for start, num_tokens in chosen]
        run = torch.cat(indices) if indices else torch.zeros(0, dtype=torch.long, device=device)
        runs.append(run)
        lengths.append(int(run.shape[0]))
    if pad_group:
        # The same-view partition's pad group, keyed against nothing: padding reads no caption,
        # which is the empty run a ``CAPTION_SCOPE_NONE`` sweep already takes above.
        runs.append(torch.zeros(0, dtype=torch.long, device=device))
        lengths.append(0)
    # The per-group lengths travel with the gather: a group that reads nothing has to leave the
    # pass altogether, on the query side as well as this one, and only the caller has the
    # query-side partition to drop it from. See ``_restrict_to_caption_readers``.
    return torch.cat(runs), lengths


def _restrict_to_caption_readers(
    caption_lens: list[int],
    same_view_gather: torch.Tensor | None,
    same_view_lens: list[int],
    num_gen_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
    """Drop the groups that read no caption from *both* sides of the gen->und pass.

    The pass borrows the same-view partition for its queries and keys each group against its own
    run of captions. A group that reads nothing -- a range clip under
    ``lidar_attends_captions=False``, and the pack's trailing pad group always -- would otherwise
    be handed to the kernel as a varlen group with zero keys. That is a degenerate input the pass
    has no reason to create: those rows can simply leave it, the way the cross-instant pass drops
    control tokens, and take the scatter's ``finfo.min`` log-sum-exp that the merge gives no
    weight -- which is the same "reads no caption" the empty group expressed.

    Returns the query gather and offsets over the kept groups, their longest run, and the key
    offsets and longest run to match. The *key* gather needs no filtering: it is the runs
    concatenated, and an empty run contributes nothing to it already.
    """
    keep = [index for index, length in enumerate(caption_lens) if length > 0]
    # Segment boundaries of the same-view partition, which is in group order, as is caption_lens.
    starts, offset = [], 0
    for length in same_view_lens:
        starts.append(offset)
        offset += length
    packed_order = (
        torch.arange(num_gen_tokens, device=device) if same_view_gather is None else same_view_gather
    )  # [N_gen]
    q_gather = torch.cat([packed_order[starts[i] : starts[i] + same_view_lens[i]] for i in keep])
    q_lens = [same_view_lens[i] for i in keep]
    kv_lens = [caption_lens[i] for i in keep]
    return (
        q_gather,
        _cumulative_offsets(q_lens, device),
        max(q_lens, default=0),
        _cumulative_offsets(kv_lens, device),
        max(kv_lens, default=0),
    )


def build_multiview_maskless_plan(
    num_views: Sequence[int],
    token_shapes: Sequence[tuple[int, int, int]],
    *,
    device: torch.device,
    seconds_per_frame: Sequence[float] | None = None,
    items_per_sample: Sequence[int] | None = None,
    is_control: Sequence[bool] | None = None,
    control_attends_sensor: bool | None = None,
    view_axis: Sequence[int] | None = None,
    captions: Sequence[Sequence[tuple[int, int]]] | None = None,
    padded_gen_tokens: int | None = None,
    attention_scope: str = "decomposed",
    caption_access: Sequence[CaptionAccess] | None = None,
) -> MultiviewMasklessPlan:
    """Describe a batch's GEN stream to :func:`multiview_maskless_attention`.

    Both sensor passes are the same operation over different partitions of that stream --
    attend within a group -- and differ only in which partition:

    * **same view**, groups ``(sample, view_axis, view)``. Every item on one view axis takes
      part, so a transfer sample's control item shares its target's groups: a view's tokens are
      its control and its target together, which is what the mask's view rules come to once
      ``control_attends_sensor`` fills in the one direction that would otherwise be one-way.
      One item per axis leaves the groups tiling the stream in packed order, so the usual batch
      needs no gather here; a control item costs one, since a view's two runs sit apart.

      With ``control_attends_sensor=False`` that one direction is *not* filled in, and the group
      stops being one key set: a sensor query still takes it whole, while a control query takes
      its view's control tokens alone. Two query roles with different keys is two varlen
      segments rather than one -- still one pass, since varlen pairs each query segment with its
      own key segment and requires nothing of their lengths. See :func:`_control_split_fields`.
      Nothing else about the fold changes: the cross-instant partition never held a control token
      to begin with, and a control token reads its captions under either value of the flag.
    * **cross instant**, groups ``(sample, instant)`` over the **sensor tokens alone**. Every
      control rule in the mask is a view rule, never an instant one, so a control token is
      absent from this partition as a key and as a query alike. It is sliced out rather than
      masked, so nothing is computed for it and nothing reaches it.

    A sample owning a *single* same-view group is sliced out of the cross-instant partition on
    the same terms, and for a sharper reason: its instant groups are then subsets of its one
    view group, so they admit no key that group does not already admit. Merging them would only
    weight the query's own instant twice over -- the deliberate ``(view, frame)`` double count,
    but with nothing bought for it, since there is no second view to reach. Left in, such a
    sample diverges from the mask, which counts each key once and at one view degenerates to
    plain attention over the item. That is every LiDAR-only sample, since a range clip is always
    one view, and every single-camera sample. It is emphatically *not* a joint camera + LiDAR
    sample: the range item is one view there too, but its instant group also holds camera
    tokens, which its view group does not.

    The instant of a token is its frame's **midpoint**, quantised by the anchor item's frame
    period::

        instant = floor((frame_id + 0.5) * seconds_per_frame / anchor_seconds_per_frame)

    A latent frame is a span rather than a point -- temporal compression folds several pixel
    frames into one -- so the midpoint is what says which anchor frame a frame mostly happened
    *during*. When the two spans differ in length that is provably the anchor frame it overlaps
    most, so this is maximum-overlap assignment computed with one addition rather than a search.
    Taking the frame's start instead would assign a frame to whichever anchor frame it happened
    to begin in, which can be the one it overlaps less: at 10Hz sweeps against a 7.5Hz camera,
    a sweep straddling a camera boundary shares 67ms with the later frame and 33ms with the
    earlier, and starts in the earlier.

    The anchor is the sample's first item, which the caller orders so that a camera item comes
    first: anchoring on the camera keeps a joint sample's camera tokens on exactly the frame
    indices a camera-only sample gives them, so the joint case adds LiDAR connectivity without
    disturbing the camera path. An item anchored on itself lands every token on its own frame
    index, which is what makes a single-sensor batch bit-identical to the frame-index form.

    Samples may differ in views, frames, resolution and rate; nothing here requires them to
    agree.

    Args:
        num_views: cameras per item, flattened over samples. A LiDAR item takes 1.
        token_shapes: ``(latent_t, patch_h, patch_w)`` per item, parallel to ``num_views``.
        device: where the index tensors are built, i.e. where the batch will attend.
        seconds_per_frame: real time between two latent frames of each item. ``None`` gives
            every item 1.0, which makes an instant the frame index -- correct whenever the
            batch carries one rate.
        items_per_sample: how many items each sample owns. ``None`` means one each.
        is_control: whether each item conditions the target that follows it. ``None`` means
            none of them do.
        control_attends_sensor: what a control query may reach, matching the mask flag of that
            name. ``True`` gives it its whole view group, its target's sensor tokens included,
            which is one unmasked pass over the group. ``False`` gives it the group's control
            tokens alone, which takes the two passes described above. ``None`` states nothing,
            and is accepted only for a batch that marks no control item -- where the two are the
            same attention, since there is no control query to narrow. A batch that carries one
            is refused rather than given a default: a default here is a silently wider or
            narrower key set for exactly the tokens the flag is about, and which value the run
            means is the config's answer, not this builder's.
        view_axis: which sensor's view numbering each item is on, so a camera view 0 and a
            range item's only view are told apart. ``None`` puts every item on axis 0, which
            is right whenever the batch carries one sensor.
        captions: per sample, its captions as ``(view_id, num_tokens)`` in packed order, with
            ``view_id`` ``-1`` for a sample-level caption. ``None``, or a batch whose samples
            each pack one caption, leaves the gen->und pass keying the whole causal run per
            sample -- the cheaper form, since no caption is then replicated per view.
        attention_scope: which scope to fold for. ``"decomposed"`` (the default) builds both
            sensor partitions. ``"same_view"`` builds only the first: no instant ids are
            recorded, so the cross-instant pass is skipped and each query reaches its own view
            across all frames and nothing else. That is the whole difference between the two
            here, and it is why ``"same_view"`` is *exact* against its mask where
            ``"decomposed"`` is not -- one partition cannot overlap itself, so no key is
            double-weighted and there is nothing for inclusion-exclusion to subtract back out.
            ``"all_views"`` is not accepted: it is one unmasked pass per sample rather than a
            partition of one, which this builder does not express.

    Returns:
        The plan, with its partitions as varlen offsets and the gathers that reach them.

    Raises:
        ValueError: for mismatched lengths, an empty batch, a non-positive rate, a sample whose
            items are all control, a batch that marks a control item without stating
            ``control_attends_sensor``, a ``latent_t`` its item's view count does not divide, or
            a batch no token of which reads a caption -- every item cut off from the captions by
            ``lidar_attends_captions=False``, which is generation without text conditioning
            rather than an attention this builds.
    """
    if attention_scope not in ("decomposed", "same_view"):
        raise ValueError(
            f"attention_scope={attention_scope!r} is not one this fold expresses; expected "
            "'decomposed' or 'same_view'. 'all_views' is one unmasked pass per sample rather "
            "than a partition of one, and is not built here."
        )
    num_items = len(num_views)
    if len(token_shapes) != num_items:
        raise ValueError(f"num_views describes {num_items} items but token_shapes describes {len(token_shapes)}.")
    if not num_items:
        raise ValueError("build_multiview_maskless_plan needs at least one item.")
    rates = [1.0] * num_items if seconds_per_frame is None else list(seconds_per_frame)
    control = [False] * num_items if is_control is None else list(is_control)
    axes = [0] * num_items if view_axis is None else list(view_axis)
    # ``None`` reproduces what the axis used to decide on its own: the cameras' axis is the one
    # captions are written for, and every other axis is a sensor that reads all of them. Callers
    # that have items -- the network does, the same ones it hands the mask -- pass those instead,
    # so the two backends describe one batch once. See ``CaptionAccess``.
    accesses: list[CaptionAccess] = (
        ["camera" if axis == 0 else "all_captions" for axis in axes] if caption_access is None else list(caption_access)
    )
    for name, values in (
        ("seconds_per_frame", rates),
        ("is_control", control),
        ("view_axis", axes),
        ("caption_access", accesses),
    ):
        if len(values) != num_items:
            raise ValueError(f"num_views describes {num_items} items but {name} describes {len(values)}.")
    if any(rate <= 0 for rate in rates):
        raise ValueError(f"seconds_per_frame must be positive, got {rates}.")
    counts = [1] * num_items if items_per_sample is None else list(items_per_sample)
    if sum(counts) != num_items:
        raise ValueError(f"items_per_sample sums to {sum(counts)} but the batch holds {num_items} items.")
    if control_attends_sensor is None and any(control):
        raise ValueError(
            "This batch marks a control item, so control_attends_sensor decides what its control "
            "queries reach -- their view's sensor tokens as well as its control ones, or only the "
            "latter -- and no default is right for both. Pass the value the run's mask config "
            "states (MultiviewAttentionMaskConfig.control_attends_sensor)."
        )
    # Unstated is only reachable for a batch with no control query to narrow, where the two
    # values are the same attention, so the permissive one is what the plan records.
    control_reaches_sensor = True if control_attends_sensor is None else control_attends_sensor
    # The split is the flag's only effect, and only where there is a control token to withhold.
    needs_control_split = not control_reaches_sensor and any(control)

    frames_per_view: list[int] = []
    spatial_tokens: list[int] = []
    item_lens: list[int] = []
    for views, (latent_t, patch_h, patch_w) in zip(num_views, token_shapes):
        if views < 1 or latent_t % views != 0:
            raise ValueError(f"latent_t={latent_t} is not divisible by num_views={views}.")
        frames_per_view.append(latent_t // views)
        spatial_tokens.append(patch_h * patch_w)
        item_lens.append(latent_t * patch_h * patch_w)

    plan = MultiviewMasklessPlan(
        attention_scope=attention_scope,
        control_attends_sensor=control_reaches_sensor,
        num_views=tuple(num_views),
        token_shapes=tuple(token_shapes),
        seconds_per_frame=tuple(rates),
        items_per_sample=tuple(counts),
        is_control=tuple(control),
        view_axis=tuple(axes),
        num_gen_tokens=sum(item_lens),
        padded_gen_tokens=sum(item_lens) if padded_gen_tokens is None else padded_gen_tokens,
    )
    if plan.padded_gen_tokens < plan.num_gen_tokens:
        raise ValueError(
            f"The GEN stream is padded to {plan.padded_gen_tokens} tokens but the batch's items "
            f"cover {plan.num_gen_tokens}."
        )
    pad_tokens = plan.padded_gen_tokens - plan.num_gen_tokens
    # Per-token ids for both partitions, laid down in packed order: sample by sample, its items
    # in order, each item view-outer / frame-inner / spatial-innermost.
    # A sample owns one same-view group when all its items sit on one view axis and each
    # covers a single view -- the case whose instant groups add nothing its view group lacks.
    single_group_sample: list[bool] = []
    cursor = 0
    for count in counts:
        span = range(cursor, cursor + count)
        single_group_sample.append(len({axes[i] for i in span}) == 1 and all(num_views[i] == 1 for i in span))
        cursor += count

    # Which GEN tokens read their sample's caption, per sample, for the sample-level layout
    # under ``lidar_attends_captions=False``: there the pass is keyed per sample, so a LiDAR
    # token that reads nothing has to leave the pass rather than take an empty key run the way
    # a per-view group does. Collected unconditionally -- they are aranges over spans the loop
    # already walks -- and materialized only when that case is in play.
    per_view_captions = bool(captions) and not all(len(sample_captions) <= 1 for sample_captions in captions)
    subset_gen_to_und = (not per_view_captions) and any(access == "no_captions" for access in accesses)

    view_group: dict[tuple[int, int, int], int] = {}
    group_sample: dict[int, int] = {}
    group_access: dict[int, CaptionAccess] = {}
    view_ids: list[torch.Tensor] = []
    instant_ids: list[torch.Tensor] = []
    sensor_positions: list[torch.Tensor] = []
    caption_reader_runs: list[torch.Tensor] = []
    caption_reader_lens: list[int] = []
    # Each same-view group's tokens kept apart by what they are, for the split
    # ``control_attends_sensor=False`` takes. Collected only then: an ordinary batch pays nothing
    # for a flag whose one effect it does not see.
    group_sensor_runs: dict[int, list[torch.Tensor]] = {}
    group_control_runs: dict[int, list[torch.Tensor]] = {}
    item = position = 0
    for sample, count in enumerate(counts):
        if all(control[item + offset] for offset in range(count)):
            raise ValueError(f"Sample {sample} carries only control items and so generates nothing.")
        # The anchor is the sample's first item; the caller puts a camera item there when the
        # sample has one, so a joint sample's camera tokens keep their own frame indices.
        anchor_rate = rates[item]
        sample_reader_tokens = 0
        for _ in range(count):
            views, frames, spatial = num_views[item], frames_per_view[item], spatial_tokens[item]
            # A dict rather than arithmetic packing, so nothing rests on a bound for the view
            # count or the axis count. Two items on one axis of one sample share a view's id,
            # which is what puts a control item into its target's groups.
            ids = torch.tensor(
                [view_group.setdefault((sample, axes[item], view), len(view_group)) for view in range(views)],
                device=device,
            )  # [V]
            group_sample.update({view_group[(sample, axes[item], view)]: sample for view in range(views)})
            # A control item shares its target's groups, and the two agree on what they are to
            # the captions -- both cameras, or both the same sweep -- so recording per item is
            # the same answer either way.
            group_access.update({view_group[(sample, axes[item], view)]: accesses[item] for view in range(views)})
            view_ids.append(ids.repeat_interleave(frames * spatial))  # [V*F*S]
            if needs_control_split:
                # One run per (item, view), which is what makes a group's two halves separable at
                # all: an item is view-outer, so a view's share of it is one contiguous run.
                cell = frames * spatial
                runs = group_control_runs if control[item] else group_sensor_runs
                for view in range(views):
                    start = position + view * cell
                    runs.setdefault(view_group[(sample, axes[item], view)], []).append(
                        torch.arange(start, start + cell, device=device)  # [F*S]
                    )
            if not control[item] and not single_group_sample[sample] and attention_scope != "same_view":
                frame_ids = torch.arange(frames, device=device, dtype=torch.float64)  # [F]
                # The epsilon nudges a frame whose midpoint lands exactly on an anchor boundary
                # into the later group rather than leaving it to float rounding. Exact landings
                # need commensurate spans; at 10Hz against 7.5Hz the margin is an eighth of a
                # frame.
                instants = torch.floor((frame_ids + 0.5) * (rates[item] / anchor_rate) + 1e-6).long()  # [F]
                instant_ids.append(instants.repeat_interleave(spatial).repeat(views) + (sample << 32))
                sensor_positions.append(torch.arange(position, position + item_lens[item], device=device))
            # The cameras' axis is the one captions are written for; every other axis is a
            # sensor no caption describes, which is what the flag decides the fate of.
            if subset_gen_to_und and accesses[item] != "no_captions":
                caption_reader_runs.append(torch.arange(position, position + item_lens[item], device=device))
                sample_reader_tokens += item_lens[item]
            position += item_lens[item]
            item += 1
        caption_reader_lens.append(sample_reader_tokens)

    def _gen_to_und_subset() -> tuple[torch.Tensor | None, int]:
        """The sample-level gen->und pass's query subset, or the whole-stream form as ``None``.

        The pack's trailing padding joins the subset rather than being dropped from it, so those
        rows keep pairing with the causal stream's own pad segment exactly as they do without a
        subset. The refusal below therefore counts the *real* readers: a batch of nothing but
        padding and text-free items has a non-empty gather and still no pass to run.
        """
        if not subset_gen_to_und:
            return None, 0
        runs = list(caption_reader_runs)
        if pad_tokens:
            runs.append(torch.arange(plan.num_gen_tokens, plan.padded_gen_tokens, device=device))  # [pad_tokens]
        return torch.cat(runs), max([*caption_reader_lens, pad_tokens], default=0)

    if pad_tokens:
        # One group for the pack's padding, carrying an id past every real group's so it sorts
        # to the tail where it already sits. The pass then covers the whole stream: a padded
        # query meets only padded keys, and no row is left for a varlen kernel to skip.
        view_ids.append(torch.full((pad_tokens,), len(view_group), device=device))
        if needs_control_split:
            # Padding is nobody's control stream, so it is a sensor group here: its queries take
            # the whole (single-run) group as keys, which is the padding attending itself, the
            # same thing the unsplit pass gives it.
            group_sensor_runs[len(view_group)] = [
                torch.arange(plan.num_gen_tokens, plan.padded_gen_tokens, device=device)  # [pad_tokens]
            ]
    split_fields = _control_split_fields(group_sensor_runs, group_control_runs, device) if needs_control_split else {}
    same_view_gather, same_view_lens = _partition(torch.cat(view_ids))
    if not instant_ids:
        # Every sample owned a single view group, so nothing is left to attend by instant.
        caption_gather, caption_lens = _caption_partition(
            captions, view_group, group_sample, group_access, device, bool(pad_tokens)
        )
        caption_q_gather, caption_q_offsets, caption_q_max_len, caption_offsets, caption_max_len = (
            (None, None, 0, None, 0)
            if caption_lens is None
            else _restrict_to_caption_readers(
                caption_lens, same_view_gather, same_view_lens, plan.padded_gen_tokens, device
            )
        )
        gen_to_und_gather, gen_to_und_max_len = _gen_to_und_subset()
        return dataclasses.replace(
            plan,
            cross_view_empty=True,
            caption_gather=caption_gather,
            caption_offsets=caption_offsets,
            caption_max_len=caption_max_len,
            caption_q_gather=caption_q_gather,
            caption_q_offsets=caption_q_offsets,
            caption_q_max_len=caption_q_max_len,
            gen_to_und_gather=gen_to_und_gather,
            gen_to_und_max_len=gen_to_und_max_len,
            same_view_offsets=_cumulative_offsets(same_view_lens, device),
            same_view_max_len=max(same_view_lens),
            same_view_gather=same_view_gather,
            **split_fields,
        )
    # Sliced to the sensor tokens, so the gather indexes the packed stream but is shorter than
    # it: the pass runs over that subset and its output is scattered back, leaving the control
    # rows at a log-sum-exp the merge gives no weight.
    sensor_index = torch.cat(sensor_positions)  # [N_sensor]
    order, cross_view_lens = _partition(torch.cat(instant_ids))
    cross_view_gather = sensor_index if order is None else sensor_index[order]

    caption_gather, caption_lens = _caption_partition(
        captions, view_group, group_sample, group_access, device, bool(pad_tokens)
    )
    caption_q_gather, caption_q_offsets, caption_q_max_len, caption_offsets, caption_max_len = (
        (None, None, 0, None, 0)
        if caption_lens is None
        else _restrict_to_caption_readers(
            caption_lens, same_view_gather, same_view_lens, plan.padded_gen_tokens, device
        )
    )
    gen_to_und_gather, gen_to_und_max_len = _gen_to_und_subset()

    return dataclasses.replace(
        plan,
        caption_gather=caption_gather,
        caption_offsets=caption_offsets,
        caption_max_len=caption_max_len,
        caption_q_gather=caption_q_gather,
        caption_q_offsets=caption_q_offsets,
        caption_q_max_len=caption_q_max_len,
        gen_to_und_gather=gen_to_und_gather,
        gen_to_und_max_len=gen_to_und_max_len,
        same_view_offsets=_cumulative_offsets(same_view_lens, device),
        same_view_max_len=max(same_view_lens),
        same_view_gather=same_view_gather,
        **split_fields,
        cross_view_offsets=_cumulative_offsets(cross_view_lens, device),
        cross_view_max_len=max(cross_view_lens),
        cross_view_gather=cross_view_gather,
    )


def _scatter_to_packed(gather: torch.Tensor, num_gen_tokens: int) -> BridgeFn:
    """Group-major order back to packed order, filling the rows the pass did not cover.

    ``gather`` indexes the packed stream, so scattering by it is the exact inverse of gathering
    by it. When the pass ran over a subset -- the cross-instant one skips control tokens, whose
    every rule in the mask is a view rule -- the rows it never saw are filled with an output of
    zero and a log-sum-exp of the dtype's minimum, which is the weight ``merge_attentions``
    gives a branch that contributes nothing. ``finfo.min`` rather than ``-inf`` so a row with no
    real branch at all cannot produce ``inf - inf``.

    That weight is only nil in a branch that is **not the first one merged**. ``merge_attentions``
    accumulates as ``lse0 - logsigmoid(lse0 - lse1)``, and at ``lse0 = finfo.min`` the difference
    rounds back to ``lse0`` in float32, so ``logsigmoid`` returns it unchanged and the
    subtraction cancels to ``0.0`` -- destroying the real branch rather than out-weighing it.
    Measured on this stack: merging ``[-5, finfo.min, -3]`` gives the right answer (1.8808) and
    ``[finfo.min, -5, -3]`` gives 1.0474, with a log-sum-exp of ``+0.05`` that is then written
    back into every branch's saved LSE and rescales their gradients.

    So every caller of this has to be a branch the first one does not need to carry, and the
    first branch has to cover every row. :func:`multiview_maskless_gen_attention` keeps that
    invariant by construction -- see the comment where it assembles the merge -- and
    :func:`_control_split_fields` is written the way it is in order to preserve it.
    """

    def _forward(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_full = out.new_zeros(1, num_gen_tokens, out.shape[-2], out.shape[-1])
        lse_full = lse.new_full((1, num_gen_tokens, lse.shape[-1]), torch.finfo(lse.dtype).min)
        out_full[0, gather] = out[0]
        lse_full[0, gather] = lse[0]
        return out_full, lse_full

    return _forward


def _gather_from_packed(gather: torch.Tensor) -> BridgeFn:
    """The exact inverse of :func:`_scatter_to_packed`, back into the kernel's own layout."""

    def _inverse(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return out.squeeze(0)[gather].unsqueeze(0), lse.squeeze(0)[gather].unsqueeze(0)

    return _inverse


def _check_plan_matches_pack(plan: MultiviewMasklessPlan, packed_query_states: SequencePack) -> None:
    """Refuse a pack whose shape the plan does not describe, before any of the folds run.

    Raises:
        ValueError: when the pack's sample count or padded GEN token count disagrees with it.
    """
    # Real samples only: a pack with a pad segment describes it as one more entry in
    # ``sample_offsets``, and that pseudo-sample is not one the plan has a fold for.
    num_samples = get_num_real_samples(packed_query_states)
    if num_samples != len(plan.items_per_sample):
        raise ValueError(f"The plan describes {len(plan.items_per_sample)} samples but the pack holds {num_samples}.")

    # A shape, not a value read off a tensor: the latter is an unbacked symbol under
    # torch.compile, and comparing one to a Python int is a data-dependent guard Dynamo refuses.
    full_q, _ = get_full_only_seq(packed_query_states)
    packed_gen_tokens = full_q.shape[0]
    if packed_gen_tokens != plan.padded_gen_tokens:
        raise ValueError(
            f"The plan describes a GEN stream padded to {plan.padded_gen_tokens} tokens but the pack "
            f"holds {packed_gen_tokens}."
        )


def multiview_maskless_gen_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    *,
    plan: MultiviewMasklessPlan,
    packed_key_states_normalized: SequencePack | None = None,
) -> torch.Tensor:
    """The ``"decomposed"`` attention scope without FlexAttention, as three dense kernels.

    FlexAttention expresses that scope as one masked kernel over the fused ``[UND | GEN]``
    stream: a GEN token reaches every GEN token of its own view (at any frame) or of its own
    frame (in any view), plus its sample's captions. This computes the same three quadrants as
    three unmasked attention calls, merged by log-sum-exp, which needs no ``BlockMask`` and no
    Triton/CuTeDSL lowering:

    * **same view** -- each camera attends within itself across all its frames. The view axis is
      folded into the batch, so one kernel does every view at once: attention is independent per
      batch entry, which is exactly the independence the views want, and the camera-major layout
      makes the fold a free reshape rather than a copy (see the comment on the call).
    * **cross view** -- at each frame, every view's spatial tokens attend to every other view's.
      The frame axis becomes the batch, so the frames are independent by construction.
    * **gen->und** -- the sample's captions, the same pass ``three_way_attention`` calls
      ``full_ca``.

    Still three passes under ``control_attends_sensor=False``, which withholds from a control
    query the sensor tokens of its own view. That is not a fourth kernel but a finer cut of the
    first: a same-view group becomes two varlen segments, its sensor tokens keyed against the
    whole group and its control tokens against the group's control tokens alone. The segments
    still partition the stream, so every row is written exactly once, and everything below about
    the sensor overlap and the merge holds unchanged -- a control query is in neither the
    overlapping pass nor the cross-instant one, so its keys are counted once and the fold is
    *exact* against the mask for those rows. With the flag on, or with no control item, the
    group is one segment again and nothing here differs.

    The two sensor passes overlap on the query's own ``(view, frame)`` cell, which belongs to
    both "my view, all frames" and "my frame, all views". ``merge_attentions`` merges as if the
    key sets had been concatenated, so those ``spatial_tokens`` keys carry twice the softmax
    weight they would under FlexAttention's single OR-mask, which counts each key once.

    That is deliberate, and it is not a rounding-level difference: on a 3-view, 4-frame,
    16-spatial-token sample the outputs sit ~22% (mean absolute, against the output's own rms)
    away from ``attention_scope="decomposed"``. The own cell is a large share of both key sets,
    so double-weighting it moves the distribution. ``multiview_maskless_attention`` is
    therefore its own attention pattern rather than a drop-in for that scope -- a checkpoint
    trained under the flex mask is not one this can serve unchanged. Making the two agree takes
    inclusion-exclusion: a third pass over the own cell alone, subtracted from the merge, which
    ``merge_attentions`` cannot express (it has no negative weights) and which nothing here
    does.

    Trains as well as it infers. Both fold-backs run through :class:`MergeAttentionsBridge`,
    which is what makes the backward correct: ``merge_attentions`` repairs each branch by
    writing the merged output and LSE into the storage the kernel saved, and any autograd node
    between the kernel and the merge -- a copying permute *or* a storage-sharing reshape --
    leaves the kernel's own saved pair unpatched. Measured against a float64 dense reference,
    the unbridged form puts the two sensor branches' gradients ~100% out; bridged, every
    gradient lands within bf16 rounding. This is the same hazard that keeps FlexAttention off
    ``three_way_attention``'s merged path, met with the bridge rather than avoided.

    Any number of samples is folded the same way, one group per ``(sample, view)`` and one per
    ``(sample, frame)``, with the samples free to differ in views, frames and resolution. Both
    address their groups with varlen offsets, and the cross-view pass reaches its own through
    the gather :func:`build_multiview_maskless_plan` prepared. Note that the varlen kernels
    decline cuDNN, so this is a different backend as well as a different launch shape from the
    dense gen->und form a single sample still takes.

    It still takes the narrow case and refuses the rest rather than masking it silently:

    * one camera item per sample, no LiDAR and no control stream: the item's ``(latent_t,
      patch_h, patch_w)`` and view count are the whole geometry, and the scope's own rules
      make the conditioning/noisy split irrelevant.
    * one caption per sample: the gen->und pass keys each sample's GEN tokens against that
      sample's whole causal run, which per-view captions exist to narrow.

    Args:
        packed_query_states: the pack's queries.
        packed_key_states: the pack's keys; its causal stream is the reasoner's own.
        packed_value_states: the pack's values.
        plan: the batch's geometry and, for a ragged batch, its index tensors.
        packed_key_states_normalized: optional alternative K pack for the gen->und pass, as in
            ``two_way_attention``. ``None`` uses ``packed_key_states`` for both.

    Returns:
        The GEN stream's output, ``[N_full, heads * head_dim]``, in packed order. The UND
        half and the assembly into a pack belong to the caller, which is what lets the
        masked path and this one share them.

    Raises:
        ValueError: when the pack's sample count or GEN token count disagrees with the plan.
    """
    # ── GEN streams, trimmed to their real tokens ─────────────────────────────
    _check_plan_matches_pack(plan, packed_query_states)

    full_q, full_q_offsets = get_full_only_seq(packed_query_states)  # [N_full,heads,head_dim]
    full_k, _ = get_full_only_seq(packed_key_states)  # [N_full,kv_heads,head_dim]
    full_v, _ = get_full_only_seq(packed_value_states)  # [N_full,kv_heads,head_dim]

    num_gen_tokens = plan.padded_gen_tokens
    # A shape, not a value read off a tensor: the latter is an unbacked symbol under
    # torch.compile, and comparing one to a Python int is a data-dependent guard Dynamo refuses.
    # The pack's padding is a group of the plan's own, so the passes address the stream whole:
    # a padded key only ever meets a padded query, which is what trimming used to buy, and the
    # pack's offsets and maximum lengths describe exactly the tensors handed to the kernels.
    q, k, v = full_q, full_k, full_v  # [N_full,*,head_dim]

    # ── Pass 1: same view, every frame ────────────────────────────────────────
    # Tokens are camera-major (view-outer, frame-inner, spatial-innermost), so one item per view
    # axis leaves a view's tokens already contiguous: the groups tile the stream in packed order
    # and the kernel's own output is that order -- no gather, no bridge. A control item puts a
    # view's tokens in two runs instead, which costs the gather and the bridge back.
    #
    # Under ``control_attends_sensor=False`` a group is no longer one segment: its sensor queries
    # take it whole while its control queries take its control tokens alone, so the plan cuts it
    # into two varlen segments with their own keys. Still one pass and one kernel -- varlen pairs
    # segment ``i`` of the queries with segment ``i`` of the keys, and nothing requires the two
    # to be the same length or even the same tokens. Both sides are then a gather, the identity
    # form belonging to the unsplit case alone.
    if plan.same_view_q_gather is None:
        view_gather = kv_gather = plan.same_view_gather
        q_offsets, kv_offsets = plan.same_view_offsets, plan.same_view_offsets
        q_max_len, kv_max_len = plan.same_view_max_len, plan.same_view_max_len
    else:
        view_gather, kv_gather = plan.same_view_q_gather, plan.same_view_kv_gather
        q_offsets, kv_offsets = plan.same_view_q_offsets, plan.same_view_kv_offsets
        q_max_len, kv_max_len = plan.same_view_q_max_len, plan.same_view_kv_max_len
    # Keep this fold's output under selective AC rather than recomputing it: it is
    # ~96% of forward attention time and ~94% of backward, against three other
    # calls running the same kernel that a name-matching policy cannot tell apart.
    # The mark goes on K because it is the smallest operand the call takes -- 32 query
    # heads against 8 KV heads, 2 against 1 per rank under CP16 -- and marking copies
    # what it marks. The split leaves this a single call, so the mark still covers all of it.
    same_view_k = mark_next_activation((k if kv_gather is None else k[kv_gather]).unsqueeze(0))
    same_view_out, same_view_lse = attention(
        (q if view_gather is None else q[view_gather]).unsqueeze(0),  # [1,N_q,heads,head_dim]
        same_view_k,  # [1,N_kv,kv_heads,head_dim]
        (v if kv_gather is None else v[kv_gather]).unsqueeze(0),  # [1,N_kv,kv_heads,head_dim]
        cumulative_seqlen_Q=q_offsets,
        cumulative_seqlen_KV=kv_offsets,
        max_seqlen_Q=q_max_len,
        max_seqlen_KV=kv_max_len,
        return_lse=True,
    )  # out: [1,N_q,heads,head_dim], lse: [1,N_q,heads]
    if view_gather is not None:
        # By the query gather, which is what indexes this pass's *output* rows; the key gather is
        # internal to the kernel's own sum and has nothing on the far side to be put back into.
        same_view_out, same_view_lse = MergeAttentionsBridge.apply(
            same_view_out,
            same_view_lse,
            _scatter_to_packed(view_gather, num_gen_tokens),
            _gather_from_packed(view_gather),
        )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]

    # A batch whose every sample owns one view group has no cross-instant work to do: its
    # instant groups sit inside its view groups, so the pass would only double-weight each
    # query's own instant. Skipped outright rather than merged at zero weight, which saves
    # the kernel as well as the distortion.
    cross_view_out = cross_view_lse = None
    if not plan.cross_view_empty:
        # ── Pass 2: same frame, every view ────────────────────────────────────────
        # A frame's views are strided through the packed order, so this pass reaches them through
        # a gather. The way back is a copy too, and a copy is what
        # ``merge_attentions``'s backward cannot see through -- it repairs each branch by writing
        # the merged output and LSE into the storage the kernel saved, found by data pointer, and a
        # copy leaves the kernel's own storage unpatched. The bridge re-establishes that link.
        #
        # The scatter back is linear with constant fill -- the control rows the pass skips take an
        # output of zero and a log-sum-exp the merge gives no weight -- which is the class the
        # bridge documents itself as valid for, and it is why the same callable serves as the
        # gradient operator and as the inverse.
        gather = plan.cross_view_gather
        assert gather is not None, "A plan with a cross-instant partition carries its gather."
        cross_view_out, cross_view_lse = attention(
            q[gather].unsqueeze(0),  # [1,N_gen,heads,head_dim]  frame-major
            k[gather].unsqueeze(0),  # [1,N_gen,kv_heads,head_dim]
            v[gather].unsqueeze(0),  # [1,N_gen,kv_heads,head_dim]
            cumulative_seqlen_Q=plan.cross_view_offsets,
            cumulative_seqlen_KV=plan.cross_view_offsets,
            max_seqlen_Q=plan.cross_view_max_len,
            max_seqlen_KV=plan.cross_view_max_len,
            return_lse=True,
        )  # out: [1,N_gen,heads,head_dim], lse: [1,N_gen,heads]

        cross_view_out, cross_view_lse = MergeAttentionsBridge.apply(
            cross_view_out,
            cross_view_lse,
            _scatter_to_packed(gather, num_gen_tokens),
            _gather_from_packed(gather),
        )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]

    # ── Pass 3: gen->und ──────────────────────────────────────────────────────
    # Every sample's GEN tokens against its own captions and nothing else, which the two offset
    # tensors do -- the pack's own, since the stream handed over is the pack's own. The padding
    # pairs with the causal stream's pad segment, so its rows are written like any other.
    packed_key_normalized = (
        packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
    )
    causal_k_normalized, causal_k_normalized_offsets = get_causal_seq(packed_key_normalized)
    causal_v_unpadded, _ = get_causal_seq(packed_value_states)  # [N_und,kv_heads,head_dim]

    # The single-sample branch below is load-bearing, not tidiness. With ranges this pass keys
    # the whole GEN stream as one of them, so its ``max_seqlen_Q`` is that stream's
    # length -- and the varlen kernels index a sequence with int32, which overflows once
    # ``max_seqlen * heads * head_dim`` reaches 2**31: 524288 queries at 32 heads of 128. A
    # transfer sample doubles its own GEN stream, which is what first crosses that line. The
    # dense form has no such limit, and one sample never needs the ranges in the first place.
    if plan.caption_gather is not None:
        # Per-view captions: a camera view reads the caption written for it, a range clip reads
        # all of its sample's. That is a key set per *view*, not per sample, so this pass
        # borrows the same-view partition for its queries -- the same gather, so the same
        # scatter back -- and keys each group against its own run of captions.
        assert plan.caption_offsets is not None and plan.caption_q_gather is not None
        # The same-view partition restricted to the groups that read a caption: a group reading
        # none leaves the pass rather than being keyed against nothing (see
        # _restrict_to_caption_readers), so no group the kernel sees has zero keys.
        view_gather = plan.caption_q_gather
        gen_to_und_out, gen_to_und_lse = attention(
            q[view_gather].unsqueeze(0),  # [1,N_readers,heads,head_dim]
            causal_k_normalized[plan.caption_gather].unsqueeze(0),  # [1,N_caption_keys,kv_heads,head_dim]
            causal_v_unpadded[plan.caption_gather].unsqueeze(0),  # [1,N_caption_keys,kv_heads,head_dim]
            cumulative_seqlen_Q=plan.caption_q_offsets,
            cumulative_seqlen_KV=plan.caption_offsets,
            max_seqlen_Q=plan.caption_q_max_len,
            max_seqlen_KV=plan.caption_max_len,
            return_lse=True,
        )  # out: [1,N_readers,heads,head_dim], lse: [1,N_readers,heads]
        gen_to_und_out, gen_to_und_lse = MergeAttentionsBridge.apply(
            gen_to_und_out,
            gen_to_und_lse,
            _scatter_to_packed(view_gather, num_gen_tokens),
            _gather_from_packed(view_gather),
        )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]

    else:
        # ``gen_to_und_gather`` is the sample-level layout under
        # ``lidar_attends_captions=False``: the tokens that read their sample's one caption,
        # which is every token unless a LiDAR stream has been cut off from the text. Keyed per
        # sample either way -- the subset just renumbers the query side, so its own offsets come
        # along -- and scattered back through the bridge like pass 2, leaving the tokens it
        # skipped at a log-sum-exp the merge gives no weight, i.e. reading no caption.
        reader_gather = plan.gen_to_und_gather
        gen_to_und_out, gen_to_und_lse = attention(
            (q if reader_gather is None else q[reader_gather]).unsqueeze(0),  # [1,N_gen,heads,head_dim]
            causal_k_normalized.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
            causal_v_unpadded.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
            cumulative_seqlen_Q=(
                full_q_offsets
                if reader_gather is None
                # Where each of the pack's own GEN boundaries lands in the gathered stream, which
                # is the count of gathered tokens before it. Derived rather than stored so it
                # cannot disagree in length with the causal stream's offsets below: the pack
                # decides how many segments there are (its samples, plus its pad segment), not
                # the plan. The gather is built in packed order, so it is sorted, which is what
                # searchsorted needs.
                else torch.searchsorted(reader_gather, full_q_offsets.long()).to(torch.int32)
            ),
            cumulative_seqlen_KV=causal_k_normalized_offsets,
            max_seqlen_Q=(
                int(packed_query_states["max_full_len"]) if reader_gather is None else plan.gen_to_und_max_len
            ),
            max_seqlen_KV=int(packed_key_normalized["max_causal_len"]),
            return_lse=True,
        )  # out: [1,N_gen,heads,head_dim], lse: [1,N_gen,heads]
        if reader_gather is not None:
            gen_to_und_out, gen_to_und_lse = MergeAttentionsBridge.apply(
                gen_to_und_out,
                gen_to_und_lse,
                _scatter_to_packed(reader_gather, num_gen_tokens),
                _gather_from_packed(reader_gather),
            )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]

    # The same-view pass goes first, and that is an invariant rather than an ordering. It is the
    # only branch that covers every row of the stream -- the cross-instant one skips control
    # tokens, the gen->und one skips whatever reads no caption -- and a first branch that does
    # not cover a row silently destroys the real branches for it, see ``_scatter_to_packed``.
    # Under ``control_attends_sensor=False`` that is exactly why the split is two varlen segments
    # of this pass rather than two passes: either pass alone would leave the other's rows
    # uncovered, and no ordering of the two would fix it.
    outputs = [same_view_out]
    lse_tensors = [same_view_lse]
    if cross_view_out is not None:
        assert cross_view_lse is not None
        outputs.append(cross_view_out)
        lse_tensors.append(cross_view_lse)
    # The gen->und pass always ran: a plan whose every item is cut off from the captions is
    # refused where it is built, so there is always something for it to read.
    outputs.append(gen_to_und_out)
    lse_tensors.append(gen_to_und_lse)
    full_res, _ = merge_attentions(
        outputs=outputs, lse_tensors=lse_tensors, torch_compile=True
    )  # [1,N_gen,heads,head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # [N_full,heads*head_dim]

    # Already the pack's own stream length -- the passes covered the padding rather than
    # dropping it, so there is nothing to re-pad -- but the padding's own rows are zeroed. They
    # hold whatever a group of padding attending itself comes to, which is meaningless either
    # way, and zero is what every consumer of this pack has been given until now. Masked rather
    # than assigned in place: ``merge_attentions`` reaches the tensors it merged by data pointer
    # on the way back, and writing through this one would be writing through one of those.
    if plan.padded_gen_tokens > plan.num_gen_tokens:
        rows = torch.arange(full_out.shape[0], device=full_out.device).unsqueeze(-1)  # [N_full,1]
        full_out = torch.where(rows < plan.num_gen_tokens, full_out, full_out.new_zeros(()))

    return full_out
