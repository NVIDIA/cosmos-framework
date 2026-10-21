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

import copy
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cosmos_framework.model.attention import attention, multi_dimensional_attention
from cosmos_framework.model.attention import merge_attentions as _native_merge_attentions
from cosmos_framework.configs.base.defaults.multiview_attention import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_NONE,
    AttentionScope,
    CaptionAccess,
    CrossViewTemporalWindowConfig,
    MultiviewAttentionConfig,
    SameViewTemporalWindowConfig,
    TemporalWindow,
    resolve_caption_scope,
    temporal_window_bounds,
)
from cosmos_framework.model.generator.mot.activation_marks import mark_next_activation
from cosmos_framework.model.generator.mot.camera_relative_pose import (
    CameraPosePartition,
    CameraRelativePosePass,
    CameraRelativePosePlan,
    apply_camera_pose,
    apply_prope_pairs,
    prope_pair_channels,
)
from cosmos_framework.model.generator.mot.merge_bridge import (
    BridgeFn,
    DisjointQueriesBridge,
    MergeAttentionsBridge,
)
from cosmos_framework.model.generator.mot.merge_attention import merge_attentions_ac_safe
from cosmos_framework.model.generator.mot.rigrope import apply_mrope_rotary, apply_rig_rotary
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
    get_num_real_samples,
)

# The scopes these folds express as disjoint attention rectangles, and so can run without a
# mask. ``"same_view"`` is one partition; ``"decomposed"`` also partitions cross-view edges,
# with disjoint query/key rectangles counting every permitted edge once. ``"all_views"`` is
# absent because it is not a partition at all:
# it is one unmasked pass over each whole sample, which these folds do not build.
MASKLESS_ATTENTION_SCOPES: tuple[AttentionScope, ...] = ("decomposed", "same_view")

# Kernel geometry uses frame offsets; public configuration expresses bounds in seconds.
TemporalFrameWindow = tuple[int, int]


def merge_attentions(
    outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor], torch_compile: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # As in causal_attention, the checkpointed training path must unpack saved
    # tensors only once while preserving fused kernels' storage-patching contract.
    # Keep the existing optimized inference path unchanged.
    if torch.is_grad_enabled():
        return merge_attentions_ac_safe(outputs=outputs, lse_tensors=lse_tensors)
    return _native_merge_attentions(outputs=outputs, lse_tensors=lse_tensors, torch_compile=torch_compile)


def _is_supported_neighborhood_window(window: TemporalWindow) -> bool:
    """Whether NATTEN's size and causal flag can express these temporal bounds."""
    lower, upper = window
    return lower <= 0 and (upper == -lower or upper == 0)


def _neighborhood_frame_window(bounds: TemporalWindow | None, rates: Sequence[float]) -> TemporalFrameWindow | None:
    """Resolve validated seconds bounds to one inclusive frame window shared by the batched kernel."""
    if bounds is None:
        return None
    if not _is_supported_neighborhood_window(bounds):
        raise ValueError(f"Same-view temporal bounds must be centered -N:N or causal -N:0; got {bounds!r}.")
    lower, upper = temporal_window_bounds(bounds)
    windows = {(math.ceil(lower / rate), math.floor(upper / rate)) for rate in rates}
    if len(windows) != 1:
        raise ValueError(
            "Same-view temporal bounds must resolve to one common frame window for batched neighborhood attention; "
            f"bounds={bounds!r}, seconds_per_frame={list(rates)!r}, frame windows={sorted(windows)!r}."
        )
    return next(iter(windows))


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
    Config construction has already validated the temporal bounds themselves.
    """
    cross_view = config.mask.cross_view_temporal_window
    if cross_view.includes_first_frame and (config.mask.attention_scope != "decomposed" or cross_view.bounds is None):
        return (
            "cross_view_temporal_window.includes_first_frame widens a capture-time window, so it "
            "needs attention_scope='decomposed' and cross_view_temporal_window.bounds; this "
            f"config has attention_scope={config.mask.attention_scope!r} and window "
            f"{cross_view.bounds!r}"
        )
    unsupported_windows: dict[str, TemporalWindow] = {}
    for name in (
        "sensor_to_sensor_bounds",
        "sensor_to_control_bounds",
        "control_to_control_bounds",
        "control_to_sensor_bounds",
    ):
        window = getattr(config.mask.same_view_temporal_window, name)
        if window is not None and not _is_supported_neighborhood_window(window):
            unsupported_windows[name] = window
    if unsupported_windows:
        return (
            f"same-view temporal windows {unsupported_windows!r} are not centered -N:N or causal -N:0 windows "
            "supported by fused neighborhood attention"
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
class CrossViewPass:
    """One varlen cross-view pass: query runs gathered from the GEN stream and their key runs."""

    gather: torch.Tensor  # [N_query]
    offsets: torch.Tensor  # [R+1]
    max_len: int
    kv_gather: torch.Tensor  # [N_keys]
    kv_offsets: torch.Tensor  # [R+1]
    kv_max_len: int


@dataclass(frozen=True)
class CrossViewPartition:
    """Capture-time rectangles batched as varlen, with unique queries per pass.

    Each hierarchy level keys queries against the opposite half of its camera
    group, excluding their own view. Q/K offsets can describe different lengths.
    """

    query_indices: torch.Tensor  # [Nq]
    key_indices: torch.Tensor  # [Nk]
    query_offsets: torch.Tensor  # [segments+1]
    key_offsets: torch.Tensor  # [segments+1]
    max_query_len: int
    max_key_len: int


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
            :func:`_control_split_fields`. ``None`` leaves an unwindowed pass keyed by
            ``same_view_gather`` on both sides, which is a group attending itself. Windowed
            passes use ``sensor_gather`` and ``control_gather`` instead. Set only where the
            varlen split is in play: no neighborhood windows, the flag off, and some item
            marked control. The three
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
        cross_view_temporal_window: cross-view bounds and first-frame policy. Bounds
            are (start, end) key-time offsets from the query in seconds: (-N, 0)
            is past-only, and None retains instant matching. includes_first_frame
            also admits each sample's earliest capture time for every query time.
        cross_view_partitions: one varlen pass per binary-tree depth. Each directed
            cross-view edge occurs once across all levels; a view's own keys are
            never repeated by any level. Empty when no sample contributes cross-view
            attention: queries whose instant/window contains no other view add no work.
            A configured window may also admit no keys; see :func:`build_multiview_maskless_plan`.
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
    cross_view_temporal_window: CrossViewTemporalWindowConfig = dataclasses.field(
        default_factory=CrossViewTemporalWindowConfig
    )
    cross_view_partitions: tuple[CrossViewPartition, ...] = ()
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
    # Geometry rotary is opt-in and prepared outside the compiled layers. Incoming GEN Q/K
    # stay raw only for the direct cross-view RigRoPE and PRoPE modes; these mRoPE tables restore the
    # pretrained rotation for same-view and caption attention.
    camera_relative_pose: CameraRelativePosePlan | None = None
    rigrope_cos: torch.Tensor | None = None  # [N_gen,D]
    rigrope_sin: torch.Tensor | None = None  # [N_gen,D]
    # Which GEN tokens' samples carry RigRoPE geometry; the rest keep mRoPE in the cross-view
    # pass. ``None`` means every token does.
    rigrope_valid: torch.Tensor | None = None  # [N_gen] bool
    # Which of this rank's query and KV heads use RigRoPE; the rest keep mRoPE. ``None`` means
    # all. Local heads, so under context parallelism these are the rank's own head shard.
    rigrope_q_heads: torch.Tensor | None = None  # [H_local] bool
    rigrope_k_heads: torch.Tensor | None = None  # [H_kv_local] bool
    # Which channels of a RigRoPE head take RigRoPE; the rest keep mRoPE. ``None`` means all.
    rigrope_pairs: torch.Tensor | None = None  # [D] bool
    # ``"prope_cross_view"``: each GEN token's reference-to-camera transform and its inverse,
    # identity for tokens without geometry. The geometry heads are ``rigrope_q_heads`` and
    # ``rigrope_k_heads``.
    prope_reference_to_camera: torch.Tensor | None = None  # [N_gen,4,4]
    prope_camera_to_reference: torch.Tensor | None = None  # [N_gen,4,4]
    mrope_cos: torch.Tensor | None = None  # [N_gen,D]
    mrope_sin: torch.Tensor | None = None  # [N_gen,D]
    caption_gather: torch.Tensor | None = None
    caption_offsets: torch.Tensor | None = None
    caption_max_len: int = 0
    caption_q_gather: torch.Tensor | None = None
    caption_q_offsets: torch.Tensor | None = None
    caption_q_max_len: int = 0
    gen_to_und_gather: torch.Tensor | None = None
    gen_to_und_max_len: int = 0
    sensor_gather: torch.Tensor | None = None
    control_gather: torch.Tensor | None = None
    neighborhood_layout: tuple[int, int, int] | None = None
    sensor_to_sensor_window: TemporalFrameWindow | None = None
    sensor_to_control_window: TemporalFrameWindow | None = None
    control_to_control_window: TemporalFrameWindow | None = None
    control_to_sensor_window: TemporalFrameWindow | None = None

    @property
    def cross_view_empty(self) -> bool:
        """Whether no sample contributes any other-view edges."""
        return len(self.cross_view_partitions) == 0


def cross_view_passes(plan: MultiviewMasklessPlan) -> list[CrossViewPass]:
    """The exact cross-view rectangles, one pass per level of the view tree."""
    return [
        CrossViewPass(
            partition.query_indices,
            partition.query_offsets,
            partition.max_query_len,
            partition.key_indices,
            partition.key_offsets,
            partition.max_key_len,
        )
        for partition in plan.cross_view_partitions
    ]


def _cumulative_offsets(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """``[len(lengths)+1]`` int32 cumulative offsets, the layout the varlen kernels take."""
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return offsets


def _span_indices(spans: Sequence[tuple[int, int]], device: torch.device) -> torch.Tensor:  # [N]
    # Construct indices once per forward on the host, then transfer once per gather.
    return torch.cat([torch.arange(start, end, dtype=torch.int64, device="cpu") for start, end in spans]).to(
        device
    )  # [N]


def _partition(
    group_spans: Sequence[Sequence[tuple[int, int]]], device: torch.device
) -> tuple[torch.Tensor | None, list[int]]:  # gather: [N] or None
    """A partition from each group's packed spans: ``(gather or None, group lengths)``.

    ``None`` means the groups already tile the stream in packed order, so the pass reads the
    stream as it stands and its output needs no reordering -- the case a batch of one
    single-item sample is in for both partitions, and every non-transfer batch is in for the
    same-view one.
    """
    lengths = [length for group in group_spans if (length := sum(end - start for start, end in group))]
    spans = [span for group in group_spans for span in group if span[0] != span[1]]
    position = 0
    for start, end in spans:
        if start != position:
            return _span_indices(spans, device), lengths
        position = end
    return None, lengths


def _cross_view_partitions(
    instant_cells: dict[tuple[int, float], dict[int, list[tuple[int, int]]]],
    device: torch.device,
    cross_view_temporal_window: CrossViewTemporalWindowConfig,
    *,
    include_frame_zero: bool = False,
) -> tuple[CrossViewPartition, ...]:
    """Batch capture-time rectangles that exclude same-view keys.

    Config construction validates bounds; the plan builder checks first-frame requirements.

    Split views into halves L/R, attend L->R and R->L, and recurse within each
    half. A directed pair of distinct views occurs exactly at their lowest common
    ancestor; same-view pairs never occur. All rectangles at one depth share one
    varlen call. Thus a query appears at most once per depth, with ceil(log2(V))
    levels rather than V-1 copies of its KV per query time. Queries with no other
    view in their instant/window add no work.

    Cells are host-side packed-token spans derived from layout metadata, not GPU
    boolean selections. One view can have several spans in an instant (e.g. two
    LiDAR sweeps), all of which must stay together to avoid same-view duplicates.

    With a window, cell times are float32 frame-start timestamps, not quantised
    instants. Each query reads keys with capture-time offsets inside (start, end).
    Bounds (-N, 0) give a past-only window. KV can repeat across
    distinct query times, never for the same query. The frame-level time table
    is built on CPU once per forward; no token mask or GPU selection is needed.

    ``cross_view_temporal_window.includes_first_frame`` admits each sample's earliest key time
    for every query time on top of the window. It edits that same boolean table rather than adding a rectangle, so a
    first frame already inside the window is admitted once, and the view split still keeps
    own-view keys out.
    """
    # Each view has separate query/key spans: identical for an instant, different
    # for a directed window. Missing views on either side have empty spans.
    groups: list[list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]] = []
    bounds = temporal_window_bounds(cross_view_temporal_window.bounds)
    if bounds is None:
        groups = [[(spans, spans) for spans in views.values()] for views in instant_cells.values() if len(views) > 1]
    else:
        samples: dict[int, dict[float, dict[int, list[tuple[int, int]]]]] = {}
        for (sample, timestamp), views in instant_cells.items():
            samples.setdefault(sample, {})[timestamp] = views
        start, end = torch.tensor(bounds, dtype=torch.float32, device="cpu").unbind()  # each []
        for cells in samples.values():
            timestamps = sorted(cells)
            times = torch.tensor(timestamps, dtype=torch.float32, device="cpu")  # [F_union]
            offsets = times[None, :] - times[:, None]  # [F_union,F_union], key time minus query time
            allowed = (offsets >= start) & (offsets <= end)  # [F_union,F_union]
            if cross_view_temporal_window.includes_first_frame:
                # Column 0 is the sample's earliest capture time across its streams;
                # without frame offsets, every stream starts at capture time 0.
                allowed[:, 0] = True  # [F_union]
            if include_frame_zero and 0 in cells:
                # Experimental bands refer to instant zero, even for offset clips.
                allowed[:, timestamps.index(0)] = True  # [F_union]
            for row, timestamp in enumerate(timestamps):
                queries = cells[timestamp]
                keys: dict[int, list[tuple[int, int]]] = {}
                for column in allowed[row].nonzero(as_tuple=True)[0].tolist():
                    for view, spans in cells[timestamps[column]].items():
                        keys.setdefault(view, []).extend(spans)
                views = sorted(queries.keys() | keys.keys())
                if len(views) > 1:
                    groups.append([(queries.get(view, []), keys.get(view, [])) for view in views])
    partitions: list[CrossViewPartition] = []

    while groups:
        query_spans: list[tuple[int, int]] = []
        key_spans: list[tuple[int, int]] = []
        query_lengths: list[int] = []
        key_lengths: list[int] = []
        next_groups: list[list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]] = []
        for views in groups:
            middle = len(views) // 2
            left, right = views[:middle], views[middle:]
            for query_half, key_half in ((left, right), (right, left)):
                q_spans = [span for query, _ in query_half for span in query]
                k_spans = [span for _, key in key_half for span in key]
                if q_spans and k_spans:
                    query_spans.extend(q_spans)
                    key_spans.extend(k_spans)
                    query_lengths.append(sum(end - start for start, end in q_spans))
                    key_lengths.append(sum(end - start for start, end in k_spans))
            next_groups.extend(half for half in (left, right) if len(half) > 1)
        if query_lengths:
            partitions.append(
                CrossViewPartition(
                    query_indices=_span_indices(query_spans, device),  # [Nq]
                    key_indices=_span_indices(key_spans, device),  # [Nk]
                    query_offsets=_cumulative_offsets(query_lengths, device),  # [segments+1]
                    key_offsets=_cumulative_offsets(key_lengths, device),  # [segments+1]
                    max_query_len=max(query_lengths),
                    max_key_len=max(key_lengths),
                )
            )
        groups = next_groups
    return tuple(partitions)


def _control_split_fields(
    sensor_runs: dict[int, list[tuple[int, int]]],
    control_runs: dict[int, list[tuple[int, int]]],
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

    q_lens = [sum(end - start for start, end in q) for q, _ in segments]
    kv_lens = [sum(end - start for start, end in kv) for _, kv in segments]
    return {
        "same_view_q_gather": _span_indices(
            [span for q, _ in segments for span in q], device
        ),  # [N_gen], a permutation
        "same_view_q_offsets": _cumulative_offsets(q_lens, device),
        "same_view_q_max_len": max(q_lens),
        "same_view_kv_gather": _span_indices(
            [span for _, kv in segments for span in kv], device
        ),  # [N_gen + control tokens], duplicating those
        "same_view_kv_offsets": _cumulative_offsets(kv_lens, device),
        "same_view_kv_max_len": max(kv_lens),
    }


def _caption_partition(
    captions: Sequence[Sequence[tuple[int, int]]] | None,
    view_group: dict[tuple[int, int, int], int],
    group_access: dict[int, CaptionAccess],
    device: torch.device,
    pad_group: bool = False,
) -> tuple[torch.Tensor | None, list[int] | None]:
    """Which captions each same-view group reads, as a gather into the causal stream.

    ``(None, None)`` unless some sample packs more than one caption. A batch whose samples
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
    # Group IDs are assigned in insertion order, matching the same-view partition.
    for (sample, _, view), group in view_group.items():
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
    frame_offsets: Sequence[int] | None = None,
    items_per_sample: Sequence[int] | None = None,
    is_control: Sequence[bool] | None = None,
    control_attends_sensor: bool | None = None,
    view_axis: Sequence[int] | None = None,
    captions: Sequence[Sequence[tuple[int, int]]] | None = None,
    padded_gen_tokens: int | None = None,
    attention_scope: str = "decomposed",
    caption_access: Sequence[CaptionAccess] | None = None,
    cross_view_temporal_window: CrossViewTemporalWindowConfig | None = None,
    same_view_temporal_window: SameViewTemporalWindowConfig | None = None,
    cross_view_band_radius: int = 0,
    include_frame_zero: bool = False,
    cross_view_band_keys: str = "other_views",
) -> MultiviewMasklessPlan:
    """Describe a batch's GEN stream to :func:`multiview_maskless_attention`.

    Dense attention runs over same-view groups and disjoint cross-view rectangles:

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
    * **cross instant**, groups ``(sample, instant)`` over the **sensor tokens alone**,
      partitioned into rectangles between distinct views. Every
      control rule in the mask is a view rule, never an instant one, so a control token is
      absent from this partition as a key and as a query alike. It is sliced out rather than
      masked, so nothing is computed for it and nothing reaches it.

    A sample owning a *single* same-view group is sliced out of the cross-instant partition on
    the same terms, and for a sharper reason: its instant groups are then subsets of its one
    view group, so they admit no key that group does not already admit. There is no second
    view to reach and no cross-view rectangle to build: attention at one view degenerates to
    plain attention over the item. That is every LiDAR-only sample, since a range clip is always
    one view, and every single-camera sample. It is emphatically *not* a joint camera + LiDAR
    sample: the range item is one view there too, but its instant group also holds camera
    tokens, which its view group does not.

    Without a temporal window, the instant of a token is its frame's **midpoint**, quantised by the anchor item's frame
    period::

        instant = floor((frame_id + frame_offset + 0.5) * seconds_per_frame / anchor_seconds_per_frame)

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
        frame_offsets: first frame of each item on its own clock; ``None`` keeps every
            item starting at zero. Frame-rate robot actions grouped by the VAE factor
            start at latent frame 1, after the initial observation, without null padding.
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
            here. Both count each permitted key once: the cross-instant rectangles
            exclude own-view keys already covered by the first pass.
            ``"all_views"`` is not accepted: it is one unmasked pass per sample rather than a
            partition of one, which this builder does not express.
        cross_view_band_radius: instants either side of each query instant its cross-view key
            run also covers. Zero keeps the aligned partition.
        include_frame_zero: whether every cross-view key run also covers the sample's first
            instant.
        cross_view_band_keys: expanded bands must use ``"other_views"`` so own-view
            keys already covered by the same-view pass are not counted twice.
            With no expansion, the default is accepted and all cross-view rectangles
            still exclude own-view keys.


    Returns:
        The plan, with its partitions as varlen offsets and the gathers that reach them.

    The instant passes use disjoint cross-view rectangles: same-view edges remain
    solely in the first pass. A non-None
    ``cross_view_temporal_window.bounds`` additionally replaces instant matching
    with Flex's ``start <= key_time - query_time <= end`` rule, using float32
    frame-start times and the same tolerance. Bounds (-N, 0) are past-only.
    Without a window, mixed-rate midpoint grouping can differ from Flex's frame-index
    matching even though every admitted key is counted once.
    ``cross_view_temporal_window.includes_first_frame`` widens the window to each sample's
    first frame as well, and needs a window to widen.

    ``same_view_temporal_window`` gives the four sensor/control edges' inclusive bounds in
    seconds. Explicit ``seconds_per_frame`` converts them into frame offsets. All items must
    resolve each configured edge to the same frame window for the batched NATTEN call.
    Windowed sensor/control edges also require matching clocks within each view group:
    equal frame offsets describe equal time offsets only when the two periods agree.
    Centered windows retain NATTEN's inward shift at sequence boundaries; causal windows
    clip at the beginning. An unset edge retains attention over its entire same-view stream.

    Raises:
        ValueError: for mismatched lengths, an empty batch, a non-positive rate, a sample whose
            items are all control, a batch that marks a control item without stating
            ``control_attends_sensor``, a ``latent_t`` its item's view count does not divide, or
            a batch no token of which reads a caption -- every item cut off from the captions by
            ``lidar_attends_captions=False``, which is generation without text conditioning
            rather than an attention this builds.
    """
    if cross_view_band_radius < 0:
        raise ValueError(f"cross_view_band_radius must be non-negative, got {cross_view_band_radius}.")
    if cross_view_band_keys not in ("all_views", "own_view", "other_views"):
        raise ValueError(f"Unknown cross_view_band_keys={cross_view_band_keys!r}.")
    expands_band = cross_view_band_radius > 0 or include_frame_zero
    if expands_band and attention_scope != "decomposed":
        raise ValueError("The experimental cross-view expansion requires attention_scope='decomposed'.")
    if expands_band and cross_view_band_keys != "other_views":
        raise ValueError(
            "Overlapping cross-view bands are not supported; set cross_view_band_keys='other_views' "
            "so every allowed key is counted once."
        )
    if attention_scope not in MASKLESS_ATTENTION_SCOPES:
        raise ValueError(
            f"attention_scope={attention_scope!r} is not one this fold expresses; expected "
            "'decomposed' or 'same_view'. 'all_views' is one unmasked pass per sample rather "
            "than a partition of one, and is not built here."
        )
    # Snapshot the mutable config so the plan retains the settings used to build its partitions.
    cross_view = (
        copy.copy(cross_view_temporal_window)
        if cross_view_temporal_window is not None
        else CrossViewTemporalWindowConfig()
    )
    if expands_band and cross_view.bounds is not None:
        raise ValueError(
            "Experimental cross-view bands and cross_view_temporal_window.bounds both select the cross-view "
            "keys; set one or the other."
        )
    if cross_view.bounds is not None:
        if seconds_per_frame is None:
            raise ValueError("cross_view_temporal_window.bounds requires explicit seconds_per_frame")
    elif cross_view.includes_first_frame:
        raise ValueError("cross_view_temporal_window.includes_first_frame needs cross_view_temporal_window.bounds.")
    num_items = len(num_views)
    if len(token_shapes) != num_items:
        raise ValueError(f"num_views describes {num_items} items but token_shapes describes {len(token_shapes)}.")
    if not num_items:
        raise ValueError("build_multiview_maskless_plan needs at least one item.")
    rates = [1.0] * num_items if seconds_per_frame is None else list(seconds_per_frame)
    offsets = [0] * num_items if frame_offsets is None else list(frame_offsets)
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
        ("frame_offsets", offsets),
        ("is_control", control),
        ("view_axis", axes),
        ("caption_access", accesses),
    ):
        if len(values) != num_items:
            raise ValueError(f"num_views describes {num_items} items but {name} describes {len(values)}.")
    if any(rate <= 0 or not math.isfinite(rate) for rate in rates):
        raise ValueError(f"seconds_per_frame must be positive and finite, got {rates}.")
    same_view = same_view_temporal_window or SameViewTemporalWindowConfig()
    bounds = (
        same_view.sensor_to_sensor_bounds,
        same_view.sensor_to_control_bounds,
        same_view.control_to_control_bounds,
        same_view.control_to_sensor_bounds,
    )
    uses_frame_windows = any(window is not None for window in bounds)
    if uses_frame_windows and seconds_per_frame is None:
        raise ValueError("same_view_temporal_window requires explicit seconds_per_frame")
    sensor_to_sensor_window, sensor_to_control_window, control_to_control_window, control_to_sensor_window = (
        _neighborhood_frame_window(window, rates) for window in bounds
    )
    if any(not isinstance(offset, int) or offset < 0 for offset in offsets):
        raise ValueError(f"frame_offsets must be nonnegative integers, got {offsets}.")
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
    uses_cross_role_window = sensor_to_control_window is not None or (
        control_reaches_sensor and control_to_sensor_window is not None
    )
    # The varlen split is needed only where there is a control token to withhold.
    # Windowed attention uses its sensor/control neighborhood gathers instead.
    needs_control_split = not uses_frame_windows and not control_reaches_sensor and any(control)

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
        cross_view_temporal_window=cross_view,
        padded_gen_tokens=sum(item_lens) if padded_gen_tokens is None else padded_gen_tokens,
        sensor_to_sensor_window=sensor_to_sensor_window,
        sensor_to_control_window=sensor_to_control_window,
        control_to_control_window=control_to_control_window,
        control_to_sensor_window=control_to_sensor_window,
    )
    if plan.padded_gen_tokens < plan.num_gen_tokens:
        raise ValueError(
            f"The GEN stream is padded to {plan.padded_gen_tokens} tokens but the batch's items "
            f"cover {plan.num_gen_tokens}."
        )
    pad_tokens = plan.padded_gen_tokens - plan.num_gen_tokens
    # Packed spans for both partitions: sample by sample, its items in order, each item
    # view-outer / frame-inner / spatial-innermost.
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
    if uses_frame_windows and any(offsets):
        raise ValueError("Nonzero item frame offsets are not supported by same-view neighborhood windows.")

    view_group: dict[tuple[int, int, int], int] = {}
    group_access: dict[int, CaptionAccess] = {}
    view_spans: list[list[tuple[int, int]]] = []
    instant_cells: dict[tuple[int, float], dict[int, list[tuple[int, int]]]] = {}
    caption_reader_runs: list[torch.Tensor] = []
    caption_reader_lens: list[int] = []
    # Each same-view group's tokens kept apart by what they are, for the split
    # ``control_attends_sensor=False`` takes. Collected only then: an ordinary batch pays nothing
    # for a flag whose one effect it does not see. Windowed attention uses the separate
    # sensor_runs/control_runs below and needs no varlen split.
    group_sensor_runs: dict[int, list[tuple[int, int]]] = {}
    group_control_runs: dict[int, list[tuple[int, int]]] = {}
    sensor_runs: dict[tuple[int, int, int], tuple[int, int]] = {}
    control_runs: dict[tuple[int, int, int], tuple[int, int]] = {}
    neighborhood_layouts: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    neighborhood_periods: dict[tuple[int, int, int], float] = {}
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
            for view in range(views):
                key = (sample, axes[item], view)
                group = view_group.setdefault(key, len(view_group))
                if group == len(view_spans):
                    view_spans.append([])
                # A control item shares its target's groups, and the two agree on what they are to
                # the captions -- both cameras, or both the same sweep -- so recording per item is
                # the same answer either way.
                group_access[group] = accesses[item]
                # One run per (item, view), which is what makes a group's two halves separable at
                # all: an item is view-outer, so a view's share of it is one contiguous run.
                cell = frames * spatial
                start = position + view * cell
                span = (start, start + cell)
                view_spans[group].append(span)
                if needs_control_split:
                    runs = group_control_runs if control[item] else group_sensor_runs
                    runs.setdefault(group, []).append(span)

                if uses_frame_windows:
                    # Record where each view's tokens live in the packed GEN stream. A control item
                    # writes its per-view ranges to control_runs; its sensor target later writes the
                    # corresponding ranges to sensor_runs under the same (sample, axis, view) keys:
                    #
                    #   control_runs[key] = packed span of this view's control tokens
                    #   sensor_runs[key]  = packed span of this view's sensor tokens
                    #
                    # Below, the runs are concatenated in the same key order to form control_gather and
                    # sensor_gather. The attention calls then select Q and K from those two gathers to run
                    # Sensor->Sensor, Sensor->Control, Control->Sensor, and Control->Control.
                    destination = control_runs if control[item] else sensor_runs
                    if key in destination:
                        raise ValueError(f"Maskless neighborhood attention found multiple items for view group {key}.")
                    destination[key] = span
                    layout = (frames, token_shapes[item][1], token_shapes[item][2])
                    if key in neighborhood_layouts and neighborhood_layouts[key] != layout:
                        raise ValueError(
                            f"Maskless neighborhood attention needs matching control/sensor layouts for {key}; "
                            f"got {neighborhood_layouts[key]} and {layout}."
                        )
                    neighborhood_layouts[key] = layout
                    if uses_cross_role_window:
                        # NATTEN aligns sensor/control frame indices, so a shared integer window
                        # preserves seconds bounds only when the paired streams share a clock.
                        period = neighborhood_periods.setdefault(key, rates[item])
                        if period != rates[item]:
                            raise ValueError(
                                "Same-view sensor/control temporal windows require matching seconds_per_frame "
                                f"within each view group; group={key}, periods={period!r} and {rates[item]!r}."
                            )

            if not control[item] and not single_group_sample[sample] and attention_scope != "same_view":
                # The epsilon nudges a frame whose midpoint lands exactly on an anchor boundary
                # into the later group rather than leaving it to float rounding. Exact landings
                # need commensurate spans; at 10Hz against 7.5Hz the margin is an eighth of a
                # frame.
                # Match Flex's float32 frame-start clock exactly for windows;
                # leave the original midpoint quantisation intact otherwise.
                instants = (
                    (
                        (torch.arange(frames, dtype=torch.float32, device="cpu") + offsets[item]) * rates[item]
                    ).tolist()  # [F] -> list
                    if cross_view.bounds is not None
                    else [
                        math.floor((frame + offsets[item] + 0.5) * (rates[item] / anchor_rate) + 1e-6)
                        for frame in range(frames)
                    ]
                )
                for view in range(views):
                    group = view_group[(sample, axes[item], view)]
                    for frame, instant in enumerate(instants):
                        start = position + (view * frames + frame) * spatial
                        instant_cells.setdefault((sample, instant), {}).setdefault(group, []).append(
                            (start, start + spatial)
                        )
            # The cameras' axis is the one captions are written for; every other axis is a
            # sensor no caption describes, which is what the flag decides the fate of.
            if subset_gen_to_und and accesses[item] != "no_captions":
                caption_reader_runs.append(torch.arange(position, position + item_lens[item], device=device))
                sample_reader_tokens += item_lens[item]
            position += item_lens[item]
            item += 1
        caption_reader_lens.append(sample_reader_tokens)

    if uses_frame_windows:
        if not control_runs and any(
            window is not None
            for window in (sensor_to_control_window, control_to_control_window, control_to_sensor_window)
        ):
            raise ValueError(
                "A batch without control items supports only same_view_temporal_window.sensor_to_sensor_bounds; "
                "the configured control edges have no control queries or keys."
            )
        if control_runs and set(sensor_runs) != set(control_runs):
            raise ValueError(
                "Neighborhood windows require either sensor-only view groups or exactly one control and one sensor "
                f"item for every view; sensor groups={sorted(sensor_runs)}, control groups={sorted(control_runs)}."
            )
        layouts = set(neighborhood_layouts.values())
        if len(layouts) != 1:
            raise ValueError(
                "Maskless neighborhood windows currently require one common (frames, height, width) layout; "
                f"got {sorted(layouts)}."
            )
        ordered_groups = sorted(sensor_runs)
        # Update the immutable base plan for windowed attention with the packed sensor/control
        # token gathers and the common per-view NATTEN layout.
        plan = dataclasses.replace(
            plan,
            sensor_gather=_span_indices([sensor_runs[group] for group in ordered_groups], device),  # [N_sensor]
            control_gather=(
                _span_indices([control_runs[group] for group in ordered_groups], device) if control_runs else None
            ),  # [N_control] or None
            neighborhood_layout=next(iter(layouts)),
        )

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
        # One group for the pack's padding, after every real group's spans so it stays
        # at the tail where it already sits. The pass then covers the whole stream: a padded
        # query meets only padded keys, and no row is left for a varlen kernel to skip.
        pad_span = (plan.num_gen_tokens, plan.padded_gen_tokens)
        view_spans.append([pad_span])
        if needs_control_split:
            # Padding is nobody's control stream, so it is a sensor group here: its queries take
            # the whole (single-run) group as keys, which is the padding attending itself, the
            # same thing the unsplit pass gives it.
            group_sensor_runs[len(view_group)] = [pad_span]
    split_fields = _control_split_fields(group_sensor_runs, group_control_runs, device) if needs_control_split else {}
    same_view_gather, same_view_lens = _partition(view_spans, device)  # [N_gen] or None, group lengths
    # Every rectangle excludes own-view keys; a batch with no cross-view edges has
    # no partitions and needs no temporal attention kernels.
    # Experimental bands use quantised instants rather than capture-time seconds,
    # but the same rectangle partitioner keeps their edges disjoint.
    partition_window = (
        CrossViewTemporalWindowConfig(bounds=(-float(cross_view_band_radius), float(cross_view_band_radius)))
        if expands_band
        else cross_view
    )
    partitions = _cross_view_partitions(instant_cells, device, partition_window, include_frame_zero=include_frame_zero)
    caption_gather, caption_lens = _caption_partition(captions, view_group, group_access, device, bool(pad_tokens))
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
        cross_view_partitions=partitions,
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


def _rigrope_or_mrope(
    rigrope: torch.Tensor,  # [N,H,D]
    mrope: torch.Tensor,  # [N,H,D]
    token_has_geometry: torch.Tensor | None,  # [N] bool
    rigrope_heads: torch.Tensor | None,  # [H] bool
) -> torch.Tensor:  # returns [N,H,D]
    """Take RigRoPE on the RigRoPE heads of tokens with geometry, mRoPE everywhere else."""
    use = torch.ones(1, rigrope.shape[1], dtype=torch.bool, device=rigrope.device)  # [1,H]
    if rigrope_heads is not None:
        if rigrope_heads.shape != (rigrope.shape[1],):
            raise ValueError(
                f"RigRoPE head mask {tuple(rigrope_heads.shape)} does not match {rigrope.shape[1]} local heads"
            )
        use = rigrope_heads[None, :]  # [1,H]
    if token_has_geometry is not None:
        use = use & token_has_geometry[:, None]  # [N,H]
    return torch.where(use[..., None], rigrope, mrope)  # [N,H,D]


def rigrope_local_head_masks(
    num_heads: int,
    num_kv_heads: int,
    rigrope_kv_heads: int,
    *,
    cp_rank: int = 0,
    cp_size: int = 1,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [H_local], [H_kv_local]
    """Mark this rank's query and KV heads that belong to the leading ``rigrope_kv_heads`` groups.

    Global query head ``h`` reads KV head ``h // (num_heads // num_kv_heads)``. Under context
    parallelism the heads arrive as ``context_parallel_attention`` shards them: each KV head is
    first repeated ``max(cp_size // num_kv_heads, 1)`` times, then both head axes are split into
    contiguous per-rank chunks. The masks follow that layout so a query head and the KV head it
    reads always agree.
    """
    group = num_heads // num_kv_heads
    repeats = max(cp_size // num_kv_heads, 1)
    q_per_rank = num_heads // cp_size
    kv_per_rank = num_kv_heads * repeats // cp_size
    q_heads = torch.arange(cp_rank * q_per_rank, (cp_rank + 1) * q_per_rank, device=device)  # [H_local]
    kv_slots = torch.arange(cp_rank * kv_per_rank, (cp_rank + 1) * kv_per_rank, device=device)  # [H_kv_local]
    return q_heads // group < rigrope_kv_heads, kv_slots // repeats < rigrope_kv_heads


def _check_plan_matches_pack(plan: MultiviewMasklessPlan, packed_query_states: SequencePack) -> None:
    """Refuse a pack whose shape the plan does not describe, before any of the folds run.

    Raises:
        ValueError: when the pack's sample count or padded GEN token count disagrees with it.
    """
    # Eager only: the sample count is a dynamic size of ``sample_offsets``, and comparing it to a
    # Python int specializes the compiled graph on it, recompiling for every count packing draws
    # (see ``attention._use_varlen``). Eager runs of the same plans still exercise the check.
    if not torch.compiler.is_compiling():
        # Real samples only: a pack with a pad segment describes it as one more entry in
        # ``sample_offsets``, and that pseudo-sample is not one the plan has a fold for.
        num_samples = get_num_real_samples(packed_query_states)
        if num_samples != len(plan.items_per_sample):
            raise ValueError(
                f"The plan describes {len(plan.items_per_sample)} samples but the pack holds {num_samples}."
            )

    # A shape, not a value read off a tensor: the latter is an unbacked symbol under
    # torch.compile, and comparing one to a Python int is a data-dependent guard Dynamo refuses.
    full_q, _ = get_full_only_seq(packed_query_states)
    packed_gen_tokens = full_q.shape[0]
    if packed_gen_tokens != plan.padded_gen_tokens:
        raise ValueError(
            f"The plan describes a GEN stream padded to {plan.padded_gen_tokens} tokens but the pack "
            f"holds {packed_gen_tokens}."
        )


def _camera_pose_attention(
    q: torch.Tensor,  # [N,H,D]
    k: torch.Tensor,  # [N,Hkv,D]
    v: torch.Tensor,  # [N,Hkv,D]
    partition: CameraPosePartition,
    num_gen_tokens: int,
    pose_plan: CameraRelativePosePass | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N,H,D], [1,N,H]
    """Run one PRoPE partition and bridge its transformed output/gradients."""
    query = q[partition.query_indices]  # [Nq,H,D]
    key = k[partition.key_indices]  # [Nk,Hkv,D]
    value = v[partition.key_indices]  # [Nk,Hkv,D]
    if pose_plan is not None:
        query = apply_camera_pose(query, pose_plan.query_reference_to_camera.transpose(-1, -2))  # [Nq,H,D]
        key = apply_camera_pose(key, pose_plan.key_camera_to_reference)  # [Nk,Hkv,D]
        value = apply_camera_pose(value, pose_plan.key_camera_to_reference)  # [Nk,Hkv,D]
    out, lse = attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        cumulative_seqlen_Q=partition.query_offsets,
        cumulative_seqlen_KV=partition.key_offsets,
        max_seqlen_Q=partition.max_query_len,
        max_seqlen_KV=partition.max_key_len,
        return_lse=True,
    )  # [1,Nq,H,D], [1,Nq,H]
    scatter = _scatter_to_packed(partition.query_indices, num_gen_tokens)
    gather = _gather_from_packed(partition.query_indices)
    if pose_plan is None:
        return MergeAttentionsBridge.apply(out, lse, scatter, gather)

    def output_to_packed(inner: torch.Tensor, inner_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = apply_camera_pose(inner[0], pose_plan.query_reference_to_camera).unsqueeze(0)  # [1,Nq,H,D]
        return scatter(transformed, inner_lse)

    def output_from_packed(outer: torch.Tensor, outer_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inner, inner_lse = gather(outer, outer_lse)
        restored = apply_camera_pose(inner[0], pose_plan.query_camera_to_reference).unsqueeze(0)  # [1,Nq,H,D]
        return restored, inner_lse

    def gradient_from_packed(grad: torch.Tensor, grad_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inner, inner_lse = gather(grad, grad_lse)
        transformed = apply_camera_pose(inner[0], pose_plan.query_reference_to_camera.transpose(-1, -2)).unsqueeze(
            0
        )  # [1,Nq,H,D]
        return transformed, inner_lse

    return MergeAttentionsBridge.apply(out, lse, output_to_packed, output_from_packed, gradient_from_packed)


def _on_heads(
    transformed: torch.Tensor,  # [N,H,D]
    original: torch.Tensor,  # [N,H,D]
    heads: torch.Tensor | None,  # [H] bool
) -> torch.Tensor:  # returns [N,H,D]
    """Take ``transformed`` on the marked heads and ``original`` on the rest; all heads when unmarked."""
    if heads is None:
        return transformed
    if heads.shape != (transformed.shape[1],):
        raise ValueError(f"PRoPE head mask {tuple(heads.shape)} does not match {transformed.shape[1]} local heads")
    return torch.where(heads[None, :, None], transformed, original)  # [N,H,D]


def _prope_transforms(plan: MultiviewMasklessPlan) -> tuple[torch.Tensor, torch.Tensor]:  # each [N_gen,4,4]
    reference_to_camera = plan.prope_reference_to_camera
    camera_to_reference = plan.prope_camera_to_reference
    if reference_to_camera is None or camera_to_reference is None:
        raise ValueError("A PRoPE cross-view plan must carry both per-token transforms")
    return reference_to_camera, camera_to_reference


def _prope_cross_view_inputs(
    q: torch.Tensor,  # [N_gen,heads,head_dim], mRoPE applied
    k: torch.Tensor,  # [N_gen,kv_heads,head_dim], mRoPE applied
    v: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    raw_q: torch.Tensor,  # [N_gen,heads,head_dim]
    raw_k: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    plan: MultiviewMasklessPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [N_gen,heads,D], [N_gen,kv_heads,D] x2
    """The PRoPE queries, keys and values every cross-view pass gathers from.

    On a geometry head the even split-half pairs carry PRoPE on the raw projections and the odd
    pairs keep mRoPE. Both act pair-wise on disjoint channels, so each stays exactly relative;
    the odd pairs keep every mRoPE axis, which is what lets the band's time offsets still reach
    the scores.

    The transforms are per token, so they run once over the stream rather than on each pass's
    gather: a band copies each key into every query instant that reaches it, and an exact band
    once more per view-tree level, so transforming after the gather repeats the work per copy.
    Tokens no pass gathers carry the identity transform and go unread.
    """
    reference_to_camera, camera_to_reference = _prope_transforms(plan)
    pairs = prope_pair_channels(q.shape[-1], q.device)  # [D]
    cross_q = apply_prope_pairs(torch.where(pairs, raw_q, q), reference_to_camera.transpose(-1, -2))
    cross_k = apply_prope_pairs(torch.where(pairs, raw_k, k), camera_to_reference)
    cross_v = apply_prope_pairs(v, camera_to_reference)
    return (
        _on_heads(cross_q, q, plan.rigrope_q_heads),
        _on_heads(cross_k, k, plan.rigrope_k_heads),
        _on_heads(cross_v, v, plan.rigrope_k_heads),
    )


def _prope_cross_view_attention(
    cross_q: torch.Tensor,  # [N_gen,heads,head_dim], from _prope_cross_view_inputs
    cross_k: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    cross_v: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    plan: MultiviewMasklessPlan,
    cross_pass: CrossViewPass,
    num_gen_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [1,N_gen,heads,head_dim], [1,N_gen,heads]
    """Run a cross-instant pass with PRoPE on the geometry heads and mRoPE on the rest.

    The output's PRoPE channels go back to the query camera through the bridge, whose gradient
    is the transform's transpose rather than its inverse.
    """
    reference_to_camera, camera_to_reference = _prope_transforms(plan)
    gather, kv_gather = cross_pass.gather, cross_pass.kv_gather

    out, lse = attention(
        cross_q[gather].unsqueeze(0),  # [1,N_query,heads,head_dim], one run per query instant
        cross_k[kv_gather].unsqueeze(0),  # [1,N_keys,kv_heads,head_dim], optionally expanded
        cross_v[kv_gather].unsqueeze(0),  # [1,N_keys,kv_heads,head_dim]
        cumulative_seqlen_Q=cross_pass.offsets,
        cumulative_seqlen_KV=cross_pass.kv_offsets,
        max_seqlen_Q=cross_pass.max_len,
        max_seqlen_KV=cross_pass.kv_max_len,
        return_lse=True,
    )  # out: [1,N_query,heads,head_dim], lse: [1,N_query,heads]

    scatter = _scatter_to_packed(gather, num_gen_tokens)
    unscatter = _gather_from_packed(gather)
    q_pose = reference_to_camera[gather]  # [N_query,4,4]
    q_pose_inverse = camera_to_reference[gather]  # [N_query,4,4]
    heads = plan.rigrope_q_heads

    def transform_geometry_heads(features: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        transformed = apply_prope_pairs(features, matrices)
        return transformed if heads is None else _on_heads(transformed, features, heads)

    def output_to_packed(inner: torch.Tensor, inner_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = transform_geometry_heads(inner[0], q_pose)
        return scatter(transformed.unsqueeze(0), inner_lse)

    def output_from_packed(outer: torch.Tensor, outer_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inner, inner_lse = unscatter(outer, outer_lse)
        restored = transform_geometry_heads(inner[0], q_pose_inverse)
        return restored.unsqueeze(0), inner_lse

    def gradient_from_packed(grad: torch.Tensor, grad_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inner, inner_lse = unscatter(grad, grad_lse)
        transformed = transform_geometry_heads(inner[0], q_pose.transpose(-1, -2))
        return transformed.unsqueeze(0), inner_lse

    return MergeAttentionsBridge.apply(out, lse, output_to_packed, output_from_packed, gradient_from_packed)


def _aligned_cross_view_inputs(
    q: torch.Tensor,  # [N_gen,heads,head_dim], mRoPE applied when the plan carries it
    k: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    raw_q: torch.Tensor,  # [N_gen,heads,head_dim]
    raw_k: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    plan: MultiviewMasklessPlan,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [N_gen,heads,head_dim], [N_gen,kv_heads,head_dim]
    """The queries and keys every cross-view pass gathers from: mRoPE, or RigRoPE where the plan carries it.

    RigRoPE is a per-token rotation, so it runs once over the stream rather than on each pass's
    gather: a band copies each key into every query instant that reaches it, and an exact band
    once more per view-tree level, so rotating after the gather repeats the work per copy.
    Tokens no pass gathers carry zero angles and go unread.
    """
    if plan.rigrope_cos is None:
        return q, k
    if plan.rigrope_sin is None:
        raise ValueError("A decomposed RigRoPE plan with RigRoPE cosine must carry its sine")
    cross_q = apply_rig_rotary(raw_q, plan.rigrope_cos, plan.rigrope_sin)  # [N_gen,heads,D]
    cross_k = apply_rig_rotary(raw_k, plan.rigrope_cos, plan.rigrope_sin)  # [N_gen,kv_heads,D]
    if plan.rigrope_pairs is not None:
        if plan.mrope_cos is None:
            raise ValueError("A RigRoPE plan that keeps mRoPE on some channels must carry mRoPE")
        cross_q = torch.where(plan.rigrope_pairs, cross_q, q)  # [N_gen,heads,D]
        cross_k = torch.where(plan.rigrope_pairs, cross_k, k)  # [N_gen,kv_heads,D]
    if plan.rigrope_valid is not None or plan.rigrope_q_heads is not None:
        if plan.mrope_cos is None:
            raise ValueError("A RigRoPE plan that keeps mRoPE on some tokens or heads must carry mRoPE")
        cross_q = _rigrope_or_mrope(cross_q, q, plan.rigrope_valid, plan.rigrope_q_heads)
        cross_k = _rigrope_or_mrope(cross_k, k, plan.rigrope_valid, plan.rigrope_k_heads)
    return cross_q, cross_k


def _aligned_cross_view_attention(
    cross_q: torch.Tensor,  # [N_gen,heads,head_dim], from _aligned_cross_view_inputs
    cross_k: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    v: torch.Tensor,  # [N_gen,kv_heads,head_dim]
    cross_pass: CrossViewPass,
    num_gen_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [1,N_gen,heads,head_dim], [1,N_gen,heads]
    """Run a cross-instant pass under mRoPE, or RigRoPE where the plan carries it."""
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
    gather, kv_gather = cross_pass.gather, cross_pass.kv_gather
    # Queries and keys are gathered separately because a band keys each query instant
    # against a run of instants; RigRoPE is a per-token rotation, so each side takes the
    # rotation of the tokens it gathered.
    out, lse = attention(
        cross_q[gather].unsqueeze(0),  # [1,N_query,heads,head_dim], one run per query instant
        cross_k[kv_gather].unsqueeze(0),  # [1,N_keys,kv_heads,head_dim], optionally expanded
        v[kv_gather].unsqueeze(0),  # [1,N_keys,kv_heads,head_dim]
        cumulative_seqlen_Q=cross_pass.offsets,
        cumulative_seqlen_KV=cross_pass.kv_offsets,
        max_seqlen_Q=cross_pass.max_len,
        max_seqlen_KV=cross_pass.kv_max_len,
        return_lse=True,
    )  # out: [1,N_query,heads,head_dim], lse: [1,N_query,heads]

    return MergeAttentionsBridge.apply(
        out,
        lse,
        _scatter_to_packed(gather, num_gen_tokens),
        _gather_from_packed(gather),
    )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]


def _natten_window(
    window: TemporalFrameWindow | None,
    layout: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Translate an inclusive frame window into NATTEN kernel geometry.

    Returns ``(window_size, is_causal)``. Both tuples follow NATTEN's
    ``(time, height, width)`` dimension order. Only the temporal dimension can
    be causal, so the height and width flags are always ``False``. NATTEN keeps
    centered windows at a fixed width by shifting them inward at the sequence
    boundaries; causal windows instead clip at the beginning.
    """
    frames, height, width = layout
    if window is None:
        return layout, (False, False, False)
    if not _is_supported_neighborhood_window(window):
        raise ValueError(
            f"Maskless neighborhood attention needs a centered -N:N or causal -N:0 window; got {window!r}."
        )
    lower, upper = window
    is_causal = upper == 0
    temporal = min(frames, -lower + 1 if is_causal else upper - lower + 1)
    return (temporal, height, width), (is_causal, False, False)


def _reshape_neighborhood_for_merge(
    batch_size: int,
    layout: tuple[int, int, int],
) -> tuple[BridgeFn, BridgeFn]:
    """Bridge a NATTEN multidimensional result to merge-attention's packed layout."""

    def _forward(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_flat = out.reshape(1, -1, out.shape[-2], out.shape[-1])  # [1,N,H,D]
        lse_flat = lse.reshape(1, -1, lse.shape[-1])  # [1,N,H]
        return out_flat, lse_flat

    def _inverse(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_shaped = out.reshape(batch_size, *layout, out.shape[-2], out.shape[-1])  # [B,F,Y,X,H,D]
        lse_shaped = lse.reshape(batch_size, *layout, lse.shape[-1])  # [B,F,Y,X,H]
        return out_shaped, lse_shaped

    return _forward, _inverse


def _same_view_neighborhood_attention(
    q: torch.Tensor,  # [B,F,Y,X,H,D]
    k: torch.Tensor,  # [B,F,Y,X,Hkv,D]
    v: torch.Tensor,  # [B,F,Y,X,Hkv,D]
    *,
    window: TemporalFrameWindow | None,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N,H,D], [1,N,H]
    """Run same-view neighborhood attention for one query-role and key/value-role pair."""
    batch_size = q.shape[0]
    layout = (q.shape[1], q.shape[2], q.shape[3])
    window_size, is_causal = _natten_window(window, layout)
    edge_out, edge_lse = multi_dimensional_attention(
        q,
        k,
        v,
        window_size=window_size,
        is_causal=is_causal,
        backend="natten",
        return_lse=True,
    )  # out: [B,F,Y,X,H,D], lse: [B,F,Y,X,H]
    forward_fn, inverse_fn = _reshape_neighborhood_for_merge(batch_size, layout)
    return MergeAttentionsBridge.apply(edge_out, edge_lse, forward_fn, inverse_fn)


def multiview_maskless_gen_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    *,
    plan: MultiviewMasklessPlan,
    packed_key_states_normalized: SequencePack | None = None,
) -> torch.Tensor:
    """Multiview attention through disjoint dense kernels and a log-sum-exp merge.

    FlexAttention expresses the decomposed scope as one masked kernel over the fused ``[UND | GEN]``
    stream: a GEN token reaches every GEN token of its own view (at any frame) or of its own
    frame (in any view), plus its permitted captions. This separates the three edge types
    into unmasked attention calls, merged by log-sum-exp, which needs no ``BlockMask`` and no
    Triton/CuTeDSL mask lowering:

    * **same view** -- each camera attends within itself across all its frames. The view axis is
      partitioned into varlen segments, so one kernel does every view at once: attention is
      independent per segment. Camera-major layouts without separate controls already tile
      the stream; separate controls require a gather (see the comment on the call).
    * **cross view** -- sensor queries attend other views at their instant or inside the
      configured capture-time window. A binary hierarchy partitions directed view pairs into
      disjoint rectangles, with one varlen call per hierarchy depth. Own-view keys never enter
      these rectangles, and each permitted cross-view key occurs once for each query.
    * **gen->und** -- each query's permitted captions, per sample or per view.

    ``control_attends_sensor=False`` withholds from a control query the sensor tokens of
    its own view. That requires no extra kernel, only a finer cut of the
    first: a same-view group becomes two varlen segments, its sensor tokens keyed against the
    whole group and its control tokens against the group's control tokens alone. The segments
    still partition the stream, so every row is written exactly once. A control query never
    enters the cross-view passes, and every key is counted once after the merge.
    With the flag on, or with no control item, the group is one segment again.

    ``merge_attentions`` combines the disjoint key sets as if they were concatenated,
    so every permitted key contributes once, with no subtraction. With shared clocks,
    this matches the decomposed mask; explicit temporal windows also match its float32
    timestamp rule. Without a window, mixed-rate sensors retain the plan's midpoint-based
    instant grouping, which can differ from the mask's frame-index grouping.

    Trains as well as it infers. The fold-backs run through :class:`MergeAttentionsBridge`,
    which is what makes the backward correct: ``merge_attentions`` repairs each branch by
    writing the merged output and LSE into the storage the kernel saved, and any autograd node
    between the kernel and the merge -- a copying permute *or* a storage-sharing reshape --
    leaves the kernel's own saved pair unpatched. Measured against a float64 dense reference,
    the unbridged form puts the two sensor branches' gradients ~100% out; bridged, every
    gradient lands within bf16 rounding. This is the same hazard that keeps FlexAttention off
    ``three_way_attention``'s merged path, met with the bridge rather than avoided.

    Any number of samples is folded the same way, with same-view groups and cross-view
    rectangles, and samples free to differ in views, frames and resolution. The calls
    address their groups with varlen offsets and the gathers
    :func:`build_multiview_maskless_plan` prepared. Note that the varlen kernels
    decline cuDNN, so this is a different backend as well as a different launch shape from the
    dense gen->und form a single sample still takes.

    The plan carries camera and additional sensor items, optional control streams, and
    sample-level or per-view captions. Unsupported geometry is rejected by the builder
    rather than silently replaced with a masked pass. Same-view frame windows use the
    plan's neighborhood attention calls in place of the full same-view rectangles.

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
    raw_q, raw_k = q, k
    if plan.mrope_cos is not None:
        if plan.mrope_sin is None:
            raise ValueError("A decomposed RigRoPE plan with mRoPE cosine must carry its sine")
        q = apply_mrope_rotary(q, plan.mrope_cos, plan.mrope_sin)  # [N_full,heads,head_dim]
        k = apply_mrope_rotary(k, plan.mrope_cos, plan.mrope_sin)  # [N_full,kv_heads,head_dim]

    # ── Pass 1: same view, every frame or configured neighborhood ─────────────
    # Tokens are camera-major (view-outer, frame-inner, spatial-innermost), so one item per view
    # axis leaves a view's tokens already contiguous: the groups tile the stream in packed order
    # and the kernel's own output is that order -- no gather, no bridge. A control item puts a
    # view's tokens in two runs instead, which costs the gather and the bridge back.
    if plan.neighborhood_layout is not None:
        if plan.sensor_gather is None:
            raise ValueError("A windowed multiview attention plan is missing its sensor neighborhood geometry.")
        layout = plan.neighborhood_layout
        # Each role participates in multiple edges. Share its gathered inputs while keeping
        # every kernel's output/LSE bridge independent for the attention merges' backward.
        sensor_q = q[plan.sensor_gather].reshape(-1, *layout, q.shape[-2], q.shape[-1])  # [B,F,Y,X,H,D]
        sensor_k = k[plan.sensor_gather].reshape(-1, *layout, k.shape[-2], k.shape[-1])  # [B,F,Y,X,Hkv,D]
        sensor_v = v[plan.sensor_gather].reshape(-1, *layout, v.shape[-2], v.shape[-1])  # [B,F,Y,X,Hkv,D]
        sensor_to_sensor_out, sensor_to_sensor_lse = _same_view_neighborhood_attention(
            sensor_q,
            sensor_k,
            sensor_v,
            window=plan.sensor_to_sensor_window,
        )  # [1,N_sensor,H,D], [1,N_sensor,H]
        if plan.control_gather is None:
            # Sensor is the only query population. The bridge still matters when the pack has
            # padding: it fills those rows with zero merge weight and preserves the attention
            # kernel's saved output/LSE storage during the outer log-sum-exp merge's backward.
            same_view_out, same_view_lse = MergeAttentionsBridge.apply(
                sensor_to_sensor_out,
                sensor_to_sensor_lse,
                _scatter_to_packed(plan.sensor_gather, num_gen_tokens),
                _gather_from_packed(plan.sensor_gather),
            )  # [1,N_gen,H,D], [1,N_gen,H]
        else:
            control_q = q[plan.control_gather].reshape(-1, *layout, q.shape[-2], q.shape[-1])  # [B,F,Y,X,H,D]
            control_k = k[plan.control_gather].reshape(-1, *layout, k.shape[-2], k.shape[-1])  # [B,F,Y,X,Hkv,D]
            control_v = v[plan.control_gather].reshape(-1, *layout, v.shape[-2], v.shape[-1])  # [B,F,Y,X,Hkv,D]
            sensor_to_control_out, sensor_to_control_lse = _same_view_neighborhood_attention(
                sensor_q,
                control_k,
                control_v,
                window=plan.sensor_to_control_window,
            )  # [1,N_sensor,H,D], [1,N_sensor,H]
            sensor_out, sensor_lse = merge_attentions(
                outputs=[sensor_to_sensor_out, sensor_to_control_out],
                lse_tensors=[sensor_to_sensor_lse, sensor_to_control_lse],
                torch_compile=True,
            )  # [1,N_sensor,H,D], [1,N_sensor,H]
            control_to_control_out, control_to_control_lse = _same_view_neighborhood_attention(
                control_q,
                control_k,
                control_v,
                window=plan.control_to_control_window,
            )  # [1,N_control,H,D], [1,N_control,H]
            # Windowing narrows enabled edges; it must not restore the sensor keys
            # withheld from control queries by control_attends_sensor=False.
            control_out, control_lse = (
                control_to_control_out,
                control_to_control_lse,
            )  # [1,N_control,H,D], [1,N_control,H]
            if plan.control_attends_sensor:
                control_to_sensor_out, control_to_sensor_lse = _same_view_neighborhood_attention(
                    control_q,
                    sensor_k,
                    sensor_v,
                    window=plan.control_to_sensor_window,
                )  # [1,N_control,H,D], [1,N_control,H]
                control_out, control_lse = merge_attentions(
                    outputs=[control_to_sensor_out, control_to_control_out],
                    lse_tensors=[control_to_sensor_lse, control_to_control_lse],
                    torch_compile=True,
                )  # [1,N_control,H,D], [1,N_control,H]
            same_view_out, same_view_lse = DisjointQueriesBridge.apply(
                sensor_out,
                sensor_lse,
                control_out,
                control_lse,
                plan.sensor_gather,
                plan.control_gather,
                num_gen_tokens,
            )  # [1,N_gen,H,D], [1,N_gen,H]
    else:
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
    # One branch per cross-view pass; ``"other_views"`` band keys run one per view-tree level.
    cross_view_outs: list[torch.Tensor] = []
    cross_view_lses: list[torch.Tensor] = []
    if plan.camera_relative_pose is not None:
        for pose_pass in plan.camera_relative_pose.camera_passes:
            cross_view_out, cross_view_lse = _camera_pose_attention(
                q, k, v, pose_pass.cameras, num_gen_tokens, pose_pass
            )  # [1,N_gen,H,D], [1,N_gen,H]
            cross_view_outs.append(cross_view_out)
            cross_view_lses.append(cross_view_lse)
        for other_pass in plan.camera_relative_pose.other_passes:
            cross_view_out, cross_view_lse = _camera_pose_attention(
                q, k, v, other_pass, num_gen_tokens
            )  # [1,N_gen,H,D], [1,N_gen,H]
            cross_view_outs.append(cross_view_out)
            cross_view_lses.append(cross_view_lse)
    elif plan.prope_reference_to_camera is not None and not plan.cross_view_empty:
        if plan.mrope_cos is None:
            raise ValueError("A PRoPE cross-view plan keeps mRoPE on its other channels and heads, so must carry it")
        cross_q, cross_k, cross_v = _prope_cross_view_inputs(q, k, v, raw_q, raw_k, plan)
        for cross_pass in cross_view_passes(plan):
            cross_view_out, cross_view_lse = _prope_cross_view_attention(
                cross_q, cross_k, cross_v, plan, cross_pass, num_gen_tokens
            )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]
            cross_view_outs.append(cross_view_out)
            cross_view_lses.append(cross_view_lse)
    elif not plan.cross_view_empty:
        cross_q, cross_k = _aligned_cross_view_inputs(q, k, raw_q, raw_k, plan)
        for cross_pass in cross_view_passes(plan):
            cross_view_out, cross_view_lse = _aligned_cross_view_attention(
                cross_q, cross_k, v, cross_pass, num_gen_tokens
            )  # [1,N_gen,heads,head_dim], [1,N_gen,heads]
            cross_view_outs.append(cross_view_out)
            cross_view_lses.append(cross_view_lse)

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
        attend_k = causal_k_normalized[plan.caption_gather].unsqueeze(0)  # [1,N_caption_keys,kv_heads,head_dim]
        attend_v = causal_v_unpadded[plan.caption_gather].unsqueeze(0)  # [1,N_caption_keys,kv_heads,head_dim]
        reader_gather = plan.caption_q_gather
        q_offsets, kv_offsets = plan.caption_q_offsets, plan.caption_offsets
        q_max_len, kv_max_len = plan.caption_q_max_len, plan.caption_max_len
    else:
        # ``gen_to_und_gather`` is the sample-level layout under
        # ``lidar_attends_captions=False``: the tokens that read their sample's one caption,
        # which is every token unless a LiDAR stream has been cut off from the text. Keyed per
        # sample either way -- the subset just renumbers the query side, so its own offsets come
        # along -- and scattered back through the bridge like pass 2, leaving the tokens it
        # skipped at a log-sum-exp the merge gives no weight, i.e. reading no caption.
        attend_k = causal_k_normalized.unsqueeze(0)  # [1,N_caption_keys,kv_heads,head_dim]
        attend_v = causal_v_unpadded.unsqueeze(0)  # [1,N_caption_keys,kv_heads,head_dim]
        reader_gather = plan.gen_to_und_gather
        q_offsets = (
            full_q_offsets
            if reader_gather is None
            # Where each of the pack's own GEN boundaries lands in the gathered stream, which
            # is the count of gathered tokens before it. Derived rather than stored so it
            # cannot disagree in length with the causal stream's offsets below: the pack
            # decides how many segments there are (its samples, plus its pad segment), not
            # the plan. The gather is built in packed order, so it is sorted, which is what
            # searchsorted needs.
            else torch.searchsorted(reader_gather, full_q_offsets.long()).to(torch.int32)
        )  # [segments+1]
        kv_offsets = causal_k_normalized_offsets  # [segments+1]
        q_max_len = int(packed_query_states["max_full_len"]) if reader_gather is None else plan.gen_to_und_max_len
        kv_max_len = int(packed_key_normalized["max_causal_len"])
    gen_to_und_out, gen_to_und_lse = attention(
        (q if reader_gather is None else q[reader_gather]).unsqueeze(0),  # [1,N_readers,heads,head_dim]
        attend_k,  # [1,N_caption_keys,kv_heads,head_dim]
        attend_v,  # [1,N_caption_keys,kv_heads,head_dim]
        cumulative_seqlen_Q=q_offsets,
        cumulative_seqlen_KV=kv_offsets,
        max_seqlen_Q=q_max_len,
        max_seqlen_KV=kv_max_len,
        return_lse=True,
    )  # out: [1,N_readers,heads,head_dim], lse: [1,N_readers,heads]
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
    outputs = [same_view_out, *cross_view_outs]
    lse_tensors = [same_view_lse, *cross_view_lses]
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
