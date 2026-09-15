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
    AttentionScope,
    MultiviewAttentionConfig,
)
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
# ``multiview_dense_attention``. ``"all_views"`` is absent because it is not a partition at all:
# it is one unmasked pass over each whole sample, which these folds do not build.
DENSE_ATTENTION_SCOPES: tuple[AttentionScope, ...] = ("decomposed", "same_view")


def dense_unavailable_reason(config: MultiviewAttentionConfig) -> str | None:
    """Why a config rules these folds out, or ``None`` when it does not.

    Lives beside the folds rather than on the config because every condition is a fact about
    what they express -- which scopes are a partition, that a control item needs the symmetric
    rule -- and a config dataclass that answered it would be asserting implementation knowledge
    it does not own.

    Config-level only, and deliberately so. Whether a *batch* can be folded is decided per
    forward by ``_multiview_dense_geometry``, which raises rather than substituting a mask.
    So "auto" reads this to fix one attention for the whole run, and a batch that then turns out
    not to fit is an error rather than a quiet change of what is being trained.

    Phrased as a reason rather than a bool so ``"dense"`` can fail with the cause, the way
    ``flash_backend_unavailable_reason`` does for FA4.
    """
    if config.mask.decomposed_temporal_window_seconds is not None:
        return (
            f"decomposed_temporal_window_seconds={config.mask.decomposed_temporal_window_seconds} "
            "asks for the sliding-window form of the scope, which is directed and overlapping. "
            "Only an equivalence relation can be an unmasked pass, so no partition of the "
            "stream reproduces it"
        )
    if config.mask.attention_scope not in DENSE_ATTENTION_SCOPES:
        return (
            f"attention_scope={config.mask.attention_scope!r} is not a scope the maskless folds "
            f"express; they cover {DENSE_ATTENTION_SCOPES}. 'all_views' is one unmasked pass "
            "over each whole sample rather than a partition of one, so it is a mask rule here "
            "and not a fold"
        )
    if not config.mask.control_attends_sensor:
        return (
            "control_attends_sensor is off. A control item shares its target's view group, so "
            "with the flag off a control query would need a narrower key set than a sensor "
            "query on the same view, and one unmasked pass cannot give two query roles "
            "different keys. Required unconditionally rather than only for the batches that "
            "carry a control item: which batches those are is the dataloader's business and "
            "not this config's, and the flag costs a batch without one nothing, since it only "
            "ever widens a control query's reach and such a batch has no control queries"
        )
    return None


@dataclass(frozen=True)
class MultiviewDensePlan:
    """How :func:`multiview_dense_attention` folds one batch's GEN stream.

    The per-sample geometry, plus the index tensors the ragged path needs. Built once per
    forward by :func:`build_multiview_dense_plan`, outside the compiled and
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
        cross_view_offsets: cumulative ``(sample, frame)`` group boundaries, in *gathered*
            (frame-major) order.
        cross_view_max_len: longest ``(sample, frame)`` group.
        cross_view_gather: the sensor tokens' packed indices, in instant-major order. ``None``
            when the cross-instant partition is empty.
        cross_view_empty: whether no sample contributes to the cross-instant partition, so
            that pass is skipped outright and the merge runs on two branches. True exactly when
            every sample owns a single same-view group -- see :func:`build_multiview_dense_plan`.
        caption_gather: the caption tokens' indices into the causal stream, one contiguous run
            per same-view group, in that partition's order. ``None`` unless the batch carries
            per-view captions, in which case the gen->und pass keys each sample's GEN tokens
            against that sample's whole causal run and needs no per-group keys.
        caption_offsets: cumulative per-group boundaries into ``caption_gather``.
        caption_max_len: longest run of captions one group reads.
        cross_view_inverse: its inverse permutation, frame-major back to packed.
    """

    attention_scope: str
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
    cross_view_offsets: torch.Tensor | None = None
    cross_view_max_len: int = 0
    cross_view_gather: torch.Tensor | None = None
    cross_view_empty: bool = False
    caption_gather: torch.Tensor | None = None
    caption_offsets: torch.Tensor | None = None
    caption_max_len: int = 0


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


def _caption_partition(
    captions: Sequence[Sequence[tuple[int, int]]] | None,
    view_group: dict[tuple[int, int, int], int],
    group_owner: dict[int, tuple[int, int]],
    device: torch.device,
    pad_group: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Which captions each same-view group reads, as a gather into the causal stream.

    ``(None, None, 0)`` unless some sample packs more than one caption. A batch whose samples
    each pack one caption needs nothing here: every GEN token of a sample reads that caption,
    which the per-sample gen->und pass already gives it without replicating anything.

    With per-view captions the rule is the mask's: a camera view reads the caption written for
    it, and an item on any other view axis -- a range clip, which fuses the whole rig rather
    than covering one of its cameras -- reads every caption of its sample. That is
    ``SensorMaskItem.reads_every_caption``, derived here from the axis rather than passed,
    since it is the axis that decides it. A sample-level caption (``view_id`` ``-1``) is read
    by every group of its sample, which is what makes the single-caption case a special case of
    this one rather than a different rule.

    The runs come out in same-view group order, so the pass can key its gathered queries
    against them with no further reordering.
    """
    if not captions or all(len(sample_captions) <= 1 for sample_captions in captions):
        return None, None, 0

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
    for group in sorted(group_owner):
        sample, axis = group_owner[group]
        view = next(v for (s, a, v), gid in view_group.items() if gid == group)
        reads_every = axis != 0
        chosen = [
            (start, num_tokens) for view_id, start, num_tokens in spans[sample] if reads_every or view_id in (-1, view)
        ]
        # A view with neither a caption of its own nor a sample-level one reads nothing; the
        # merge then gives this branch no weight for those rows.
        indices = [torch.arange(start, start + num_tokens, device=device) for start, num_tokens in chosen]
        run = torch.cat(indices) if indices else torch.zeros(0, dtype=torch.long, device=device)
        runs.append(run)
        lengths.append(int(run.shape[0]))
    if pad_group:
        # The same-view partition's pad group, keyed against nothing: padding reads no caption,
        # which is the empty run a view with neither its own caption nor a sample-level one
        # already takes above.
        runs.append(torch.zeros(0, dtype=torch.long, device=device))
        lengths.append(0)
    return torch.cat(runs), _cumulative_offsets(lengths, device), max(lengths)


def build_multiview_dense_plan(
    num_views: Sequence[int],
    token_shapes: Sequence[tuple[int, int, int]],
    *,
    device: torch.device,
    seconds_per_frame: Sequence[float] | None = None,
    items_per_sample: Sequence[int] | None = None,
    is_control: Sequence[bool] | None = None,
    view_axis: Sequence[int] | None = None,
    captions: Sequence[Sequence[tuple[int, int]]] | None = None,
    padded_gen_tokens: int | None = None,
    attention_scope: str = "decomposed",
) -> MultiviewDensePlan:
    """Describe a batch's GEN stream to :func:`multiview_dense_attention`.

    Both sensor passes are the same operation over different partitions of that stream --
    attend within a group -- and differ only in which partition:

    * **same view**, groups ``(sample, view_axis, view)``. Every item on one view axis takes
      part, so a transfer sample's control item shares its target's groups: a view's tokens are
      its control and its target together, which is what the mask's view rules come to once
      ``control_attends_sensor`` fills in the one direction that would otherwise be one-way.
      One item per axis leaves the groups tiling the stream in packed order, so the usual batch
      needs no gather here; a control item costs one, since a view's two runs sit apart.
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
            items are all control, or a ``latent_t`` its item's view count does not divide.
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
        raise ValueError("build_multiview_dense_plan needs at least one item.")
    rates = [1.0] * num_items if seconds_per_frame is None else list(seconds_per_frame)
    control = [False] * num_items if is_control is None else list(is_control)
    axes = [0] * num_items if view_axis is None else list(view_axis)
    for name, values in (("seconds_per_frame", rates), ("is_control", control), ("view_axis", axes)):
        if len(values) != num_items:
            raise ValueError(f"num_views describes {num_items} items but {name} describes {len(values)}.")
    if any(rate <= 0 for rate in rates):
        raise ValueError(f"seconds_per_frame must be positive, got {rates}.")
    counts = [1] * num_items if items_per_sample is None else list(items_per_sample)
    if sum(counts) != num_items:
        raise ValueError(f"items_per_sample sums to {sum(counts)} but the batch holds {num_items} items.")

    frames_per_view: list[int] = []
    spatial_tokens: list[int] = []
    item_lens: list[int] = []
    for views, (latent_t, patch_h, patch_w) in zip(num_views, token_shapes):
        if views < 1 or latent_t % views != 0:
            raise ValueError(f"latent_t={latent_t} is not divisible by num_views={views}.")
        frames_per_view.append(latent_t // views)
        spatial_tokens.append(patch_h * patch_w)
        item_lens.append(latent_t * patch_h * patch_w)

    plan = MultiviewDensePlan(
        attention_scope=attention_scope,
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

    view_group: dict[tuple[int, int, int], int] = {}
    group_owner: dict[int, tuple[int, int]] = {}
    view_ids: list[torch.Tensor] = []
    instant_ids: list[torch.Tensor] = []
    sensor_positions: list[torch.Tensor] = []
    item = position = 0
    for sample, count in enumerate(counts):
        if all(control[item + offset] for offset in range(count)):
            raise ValueError(f"Sample {sample} carries only control items and so generates nothing.")
        # The anchor is the sample's first item; the caller puts a camera item there when the
        # sample has one, so a joint sample's camera tokens keep their own frame indices.
        anchor_rate = rates[item]
        for _ in range(count):
            views, frames, spatial = num_views[item], frames_per_view[item], spatial_tokens[item]
            # A dict rather than arithmetic packing, so nothing rests on a bound for the view
            # count or the axis count. Two items on one axis of one sample share a view's id,
            # which is what puts a control item into its target's groups.
            ids = torch.tensor(
                [view_group.setdefault((sample, axes[item], view), len(view_group)) for view in range(views)],
                device=device,
            )  # [V]
            group_owner.update({view_group[(sample, axes[item], view)]: (sample, axes[item]) for view in range(views)})
            view_ids.append(ids.repeat_interleave(frames * spatial))  # [V*F*S]
            if not control[item] and not single_group_sample[sample] and attention_scope != "same_view":
                frame_ids = torch.arange(frames, device=device, dtype=torch.float64)  # [F]
                # The epsilon nudges a frame whose midpoint lands exactly on an anchor boundary
                # into the later group rather than leaving it to float rounding. Exact landings
                # need commensurate spans; at 10Hz against 7.5Hz the margin is an eighth of a
                # frame.
                instants = torch.floor((frame_ids + 0.5) * (rates[item] / anchor_rate) + 1e-6).long()  # [F]
                instant_ids.append(instants.repeat_interleave(spatial).repeat(views) + (sample << 32))
                sensor_positions.append(torch.arange(position, position + item_lens[item], device=device))
            position += item_lens[item]
            item += 1

    if pad_tokens:
        # One group for the pack's padding, carrying an id past every real group's so it sorts
        # to the tail where it already sits. The pass then covers the whole stream: a padded
        # query meets only padded keys, and no row is left for a varlen kernel to skip.
        view_ids.append(torch.full((pad_tokens,), len(view_group), device=device))
    same_view_gather, same_view_lens = _partition(torch.cat(view_ids))
    if not instant_ids:
        # Every sample owned a single view group, so nothing is left to attend by instant.
        caption_gather, caption_offsets, caption_max_len = _caption_partition(
            captions, view_group, group_owner, device, bool(pad_tokens)
        )
        return dataclasses.replace(
            plan,
            cross_view_empty=True,
            caption_gather=caption_gather,
            caption_offsets=caption_offsets,
            caption_max_len=caption_max_len,
            same_view_offsets=_cumulative_offsets(same_view_lens, device),
            same_view_max_len=max(same_view_lens),
            same_view_gather=same_view_gather,
        )
    # Sliced to the sensor tokens, so the gather indexes the packed stream but is shorter than
    # it: the pass runs over that subset and its output is scattered back, leaving the control
    # rows at a log-sum-exp the merge gives no weight.
    sensor_index = torch.cat(sensor_positions)  # [N_sensor]
    order, cross_view_lens = _partition(torch.cat(instant_ids))
    cross_view_gather = sensor_index if order is None else sensor_index[order]

    caption_gather, caption_offsets, caption_max_len = _caption_partition(
        captions, view_group, group_owner, device, bool(pad_tokens)
    )

    return dataclasses.replace(
        plan,
        caption_gather=caption_gather,
        caption_offsets=caption_offsets,
        caption_max_len=caption_max_len,
        same_view_offsets=_cumulative_offsets(same_view_lens, device),
        same_view_max_len=max(same_view_lens),
        same_view_gather=same_view_gather,
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


def _check_plan_matches_pack(plan: MultiviewDensePlan, packed_query_states: SequencePack) -> None:
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


def multiview_dense_gen_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    *,
    plan: MultiviewDensePlan,
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

    The two sensor passes overlap on the query's own ``(view, frame)`` cell, which belongs to
    both "my view, all frames" and "my frame, all views". ``merge_attentions`` merges as if the
    key sets had been concatenated, so those ``spatial_tokens`` keys carry twice the softmax
    weight they would under FlexAttention's single OR-mask, which counts each key once.

    That is deliberate, and it is not a rounding-level difference: on a 3-view, 4-frame,
    16-spatial-token sample the outputs sit ~22% (mean absolute, against the output's own rms)
    away from ``attention_scope="decomposed"``. The own cell is a large share of both key sets,
    so double-weighting it moves the distribution. ``multiview_dense_attention`` is
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
    the gather :func:`build_multiview_dense_plan` prepared. Note that the varlen kernels
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
    view_gather = plan.same_view_gather
    same_view_out, same_view_lse = attention(
        (q if view_gather is None else q[view_gather]).unsqueeze(0),  # [1,N_gen,heads,head_dim]
        (k if view_gather is None else k[view_gather]).unsqueeze(0),  # [1,N_gen,kv_heads,head_dim]
        (v if view_gather is None else v[view_gather]).unsqueeze(0),  # [1,N_gen,kv_heads,head_dim]
        cumulative_seqlen_Q=plan.same_view_offsets,
        cumulative_seqlen_KV=plan.same_view_offsets,
        max_seqlen_Q=plan.same_view_max_len,
        max_seqlen_KV=plan.same_view_max_len,
        return_lse=True,
    )  # out: [1,N_gen,heads,head_dim], lse: [1,N_gen,heads]
    if view_gather is not None:
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
        assert plan.caption_offsets is not None
        view_gather = plan.same_view_gather
        gen_to_und_out, gen_to_und_lse = attention(
            (q if view_gather is None else q[view_gather]).unsqueeze(0),  # [1,N_gen,heads,head_dim]
            causal_k_normalized[plan.caption_gather].unsqueeze(0),  # [1,N_caption_keys,kv_heads,head_dim]
            causal_v_unpadded[plan.caption_gather].unsqueeze(0),  # [1,N_caption_keys,kv_heads,head_dim]
            cumulative_seqlen_Q=plan.same_view_offsets,
            cumulative_seqlen_KV=plan.caption_offsets,
            max_seqlen_Q=plan.same_view_max_len,
            max_seqlen_KV=plan.caption_max_len,
            return_lse=True,
        )  # out: [1,N_gen,heads,head_dim], lse: [1,N_gen,heads]
        if view_gather is not None:
            gen_to_und_out, gen_to_und_lse = MergeAttentionsBridge.apply(
                gen_to_und_out,
                gen_to_und_lse,
                _scatter_to_packed(view_gather, num_gen_tokens),
                _gather_from_packed(view_gather),
            )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]

    else:
        gen_to_und_out, gen_to_und_lse = attention(
            q.unsqueeze(0),  # [1,N_gen,heads,head_dim]
            causal_k_normalized.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
            causal_v_unpadded.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=causal_k_normalized_offsets,
            max_seqlen_Q=int(packed_query_states["max_full_len"]),
            max_seqlen_KV=int(packed_key_normalized["max_causal_len"]),
            return_lse=True,
        )  # out: [1,N_gen,heads,head_dim], lse: [1,N_gen,heads]

    outputs = [same_view_out, gen_to_und_out]
    lse_tensors = [same_view_lse, gen_to_und_lse]
    if cross_view_out is not None:
        assert cross_view_lse is not None
        outputs.insert(1, cross_view_out)
        lse_tensors.insert(1, cross_view_lse)
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
