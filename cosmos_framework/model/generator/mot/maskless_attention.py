# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal replay using the upstream maskless sensor/caption passes and LSE merge.

Replay roles determine each query group's keys. Grouping happens once before the
decoder, over metadata runs rather than a quadratic token mask. Queries sharing
the same keys share one varlen group, including across frames in a replay chunk.
The two sensor passes deliberately overlap, as in bidirectional maskless attention.

The same-view partition re-reads each view's causal history once per chunk. Gathering it
would copy those keys per group, so it always runs gapped, reading them in place; only the small
partitions are gathered. Its segments come from the same grouping: each group's maximal
query-row intervals read its maximal key-row intervals
(a control run never joins the generated run after it), which is exact for any visibility.

NVIDIA-only copy-free same-view replay attention on FlashAttention-4 gapped varlen.

The gathered replay plan gives every query group a private copy of its keys. In the same-view
partition each causal chunk re-gathers its view's whole history, so the copies grow with
(chunks x stream) and exceed memory at production pack sizes. Here that partition runs as
FlashAttention-4 varlen launches whose segments are start offsets (``cu_seqlens``) plus lengths
(``seqused``): each segment reads its query and key rows in place from the global Q/K/V streams,
and unrelated tokens between segments stay in the gaps.

The planner turns the run-level visibility into segments. Segments whose query rows are pairwise
disjoint share one output/LSE buffer (a component). Inside a component, segments whose key rows are
pairwise disjoint share one launch, so backward dK/dV writes never collide.

The cross-instant/window and caption partitions stay gathered: whole query groups are batched
into bounded gathers, with one launch per batch and one contiguous segment per query group.
Every part is merged by fp32 log-sum-exp, and
the backward runs each kernel's native backward on its rows of the merged O / LSE / dO, the
``merge_attentions`` contract. dK/dV are accumulated in fp32. Each launch batches several KV heads,
with each head's query group folded into rows, so FA4 sees equal query and KV head counts.

FlashAttention-4 is not a torch.library operator: traced by torch.compile it returns garbage.
The replay therefore runs inside two opaque custom operators whose only plan input is a CPU
handle, so a new pack is a new handle value and never a recompile.
Execution requires NVIDIA CUDA, FlashAttention-4 and Triton; there is no alternate backend.

The kernels read gap rows (partial tiles, backward preprocessing) but never write them. Gap
rows are real tokens of the same pack and must be finite, as every token must be.
"""

import bisect
import heapq
import itertools
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, fields
from importlib import import_module
from itertools import accumulate, pairwise
from types import ModuleType
from typing import Any, NamedTuple

import attrs
import numpy as np
import torch
import triton

from cosmos_framework.model.attention.utils.environment import is_torch_compiling
from cosmos_framework.configs.base.defaults.multiview_attention import DECOMPOSED_TEMPORAL_WINDOW_EPS
from cosmos_framework.model.generator.mot.flex_attention import SensorMaskItem
from cosmos_framework.model.generator.mot import maskless_kernels as kernels
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    _ROLE_PADDING,
    _ROLE_UND,
    TeacherForcingFlexMetadata,
    _key_stream_fields,
    _query_stream_fields,
    _stream_metadata_groups,
    _StreamFields,
    _teacher_forcing_pair_predicate,
)

_ACCUMULATE_BLOCK_ROWS = 32

# Gathered partitions (cross-instant/window and caption) batch whole query groups into varlen
# calls of at most 2M key tokens. A single group above this bound is rejected at plan build time.
_MAX_REPLAY_KV_GATHER_TOKENS = 1 << 21


class GappedSegment(NamedTuple):
    """Query rows ``[q_row, q_row + q_len)`` attend key rows ``[k_row, k_row + k_len)`` in place."""

    q_row: int
    q_len: int
    k_row: int
    k_len: int


@dataclass(frozen=True)
class GappedLaunch:
    """One varlen launch: segments in query order whose key rows are pairwise disjoint."""

    segments: tuple[GappedSegment, ...]


@dataclass(frozen=True)
class GappedComponent:
    """Segments with pairwise-disjoint query rows, sharing one output/LSE buffer over ``[q_lo, q_hi)``."""

    q_lo: int
    q_hi: int
    launches: tuple[GappedLaunch, ...]


@dataclass(frozen=True)
class GappedReplayPlan:
    """Host-side packing plus the CPU handle through which the custom operators reach device data."""

    handle: torch.Tensor  # [1] int64 on CPU
    components: tuple[GappedComponent, ...]

    @property
    def launches(self) -> int:
        return sum(len(component.launches) for component in self.components)

    @property
    def segments(self) -> int:
        return sum(len(launch.segments) for component in self.components for launch in component.launches)


@dataclass(frozen=True)
class ReplayMasklessPass:
    """Nonempty varlen groups for one upstream maskless partition."""

    name: str
    q_gather: torch.Tensor  # [N_Q_pass]
    kv_gather: torch.Tensor  # [N_KV_pass]
    q_offsets: tuple[int, ...]  # [G+1] host offsets; the runtime uploads the combined launch tables
    kv_offsets: tuple[int, ...]  # [G+1]


@dataclass(frozen=True)
class ReplayMasklessPlan:
    """Layer-independent plan in global (post-Ulysses) Q/KV coordinates."""

    passes: tuple[ReplayMasklessPass, ...]  # gathered partitions: cross-instant, caption
    q_len: int
    kv_len: int
    gapped: GappedReplayPlan  # in-place same-view launches, run and merged together with ``passes``


class _ConcatReplayKV(torch.autograd.Function):
    """Keep a queued clean-cache gradient from retaining the full KV gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        *parts: torch.Tensor,  # each [S_part,H,D], last part is clean memory
    ) -> torch.Tensor:  # [S_total,H,D]
        ctx.lengths = tuple(part.shape[0] for part in parts)
        return torch.cat(parts)  # [S_total,H,D]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,  # [S_total,H,D]
    ) -> tuple[torch.Tensor, ...]:  # each [S_part,H,D]
        parts = gradient.split(ctx.lengths)  # tuple[[S_part,H,D]]
        # Clean replay backpropagates after the noisy pass. Ordinary CatBackward
        # returns views, keeping caption/current-token storage alive at every
        # layer until then. Only the clean-memory slice needs independent storage.
        return (*parts[:-1], parts[-1].clone())  # tuple[[S_part,H,D]]


def cat_replay_kv(parts: list[torch.Tensor]) -> torch.Tensor:  # each [S_part,H,D]; returns [S_total,H,D]
    """Concatenate replay KV with a compact gradient for the final clean-memory part."""
    return _ConcatReplayKV.apply(*parts)  # [S_total,H,D]


def _run_fields(fields_: _StreamFields) -> tuple[_StreamFields, np.ndarray]:  # returns fields, [R+1]
    """Copy one representative per metadata run to the CPU for grouping, with the run start rows."""
    _, representatives = _stream_metadata_groups(fields_, fields_.sample_id.device)  # [S], [R]
    compact = _StreamFields(
        **{
            field.name: value[representatives].cpu()
            for field in fields(fields_)
            if (value := getattr(fields_, field.name)) is not None
        }  # each [R]; absent caption bounds keep their None defaults
    )
    return compact, np.asarray(representatives.tolist() + [fields_.sample_id.numel()])


def _instant_ids(
    fields_: _StreamFields, sensor_items: Sequence[Sequence[SensorMaskItem]]
) -> tuple[torch.Tensor, torch.Tensor]:  # returns [R], [R]
    """Use upstream camera-anchored midpoint buckets, including cached frame ids."""
    instants = torch.full_like(fields_.frame_id, -1)  # [R]
    has_cross_view = torch.zeros_like(fields_.is_control)  # [R]
    for sample, items in enumerate(sensor_items):
        camera = next((item for item in items if item.caption_access == "camera"), None)
        if camera is None:
            raise ValueError("Maskless replay requires a camera anchor in every sample.")
        view_count = len(
            {view for item in items for view in range(item.view_offset, item.view_offset + item.num_views)}
        )
        has_cross_view |= (fields_.sample_id == sample) & (view_count > 1)  # [R]
        for item in items:
            belongs = (
                (fields_.sample_id == sample)
                & (fields_.view_id >= item.view_offset)
                & (fields_.view_id < item.view_offset + item.num_views)
            )  # [R]
            bucket = torch.floor(
                (fields_.frame_id.double() + 0.5) * item.seconds_per_frame / camera.seconds_per_frame
            ).long()  # [R]
            instants = torch.where(belongs, bucket, instants)  # [R]
    return instants, has_cross_view


def _gather_ranges(ranges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:  # returns [N]
    """Expand CPU token intervals with one device allocation per index tensor."""
    lengths = [stop - start for start, stop in ranges]
    offsets = list(accumulate(lengths, initial=0))
    starts = torch.tensor([start - offset for (start, _), offset in zip(ranges, offsets)], device=device)  # [R]
    repeats = torch.tensor(lengths, device=device)  # [R]
    return torch.repeat_interleave(starts, repeats, output_size=offsets[-1]) + torch.arange(
        offsets[-1], device=device
    )  # [N]


def _row_intervals(
    runs: np.ndarray,  # [N] ascending
    starts: np.ndarray,  # [R+1]
    apart: np.ndarray | None = None,  # [R] True where a run must not join the run before it
) -> list[tuple[int, int]]:
    """Maximal row intervals ``(start, end)`` covered by metadata runs; adjacent runs are adjacent rows."""
    split = np.diff(runs) > 1  # [N-1]
    if apart is not None:
        split |= apart[runs[1:]]
    breaks = np.flatnonzero(split) + 1  # [N_intervals-1]
    first = runs[np.concatenate(([0], breaks))]  # [N_intervals]
    last = runs[np.concatenate((breaks - 1, [runs.size - 1]))]  # [N_intervals]
    return list(zip(starts[first].tolist(), starts[last + 1].tolist()))


def _run_groups(
    visibility: torch.Tensor,  # [R_Q,R_KV] on CPU
    q_starts: np.ndarray,  # [R_Q+1]
    kv_starts: np.ndarray,  # [R_KV+1]
    kv_apart: np.ndarray | None = None,  # [R_KV] key runs that start a new key interval
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """Query runs with equal nonempty key-run sets, as (query row intervals, key row intervals), in query order.

    Queries without keys are omitted instead of becoming empty kernels.
    """
    edges = visibility.numpy()  # [R_Q,R_KV]
    key_sets = np.packbits(edges, axis=1)  # [R_Q,ceil(R_KV/8)], one hashable row per query run
    groups: dict[bytes, list[int]] = {}
    for query in np.flatnonzero(edges.any(1)).tolist():
        groups.setdefault(key_sets[query].tobytes(), []).append(query)
    return [
        (
            _row_intervals(np.asarray(queries), q_starts),
            _row_intervals(np.flatnonzero(edges[queries[0]]), kv_starts, kv_apart),
        )
        for queries in groups.values()
    ]


def _build_passes(
    name: str,
    visibility: torch.Tensor,  # [R_Q,R_KV] on CPU
    q_starts: np.ndarray,  # [R_Q+1]
    kv_starts: np.ndarray,  # [R_KV+1]
    device: torch.device,
) -> tuple[ReplayMasklessPass, ...]:
    """Batch whole query groups with equal keys without changing any query's visibility."""
    groups = _run_groups(visibility, q_starts, kv_starts)
    if not groups:
        return ()
    q_lengths = [sum(end - start for start, end in q_intervals) for q_intervals, _ in groups]
    kv_lengths = [sum(end - start for start, end in kv_intervals) for _, kv_intervals in groups]
    boundaries = [0]
    gathered_tokens = 0
    for index, length in enumerate(kv_lengths):
        # Each gathered pass copies all of its keys in one go, so the gather size bounds memory. Keys
        # shared by several groups are gathered once per group, so this counts gathered tokens, which
        # can exceed the KV stream length. Check before allocating any gather indices.
        if length > _MAX_REPLAY_KV_GATHER_TOKENS:
            raise ValueError(
                f"Maskless replay group in {name!r} gathers {length:,} key tokens, "
                f"above the supported {_MAX_REPLAY_KV_GATHER_TOKENS:,}. Reduce the size of this attention group."
            )
        if gathered_tokens + length > _MAX_REPLAY_KV_GATHER_TOKENS:
            boundaries.append(index)
            gathered_tokens = 0
        gathered_tokens += length
    boundaries.append(len(groups))
    return tuple(
        ReplayMasklessPass(
            name=name,
            q_gather=_gather_ranges(
                [interval for q_intervals, _ in groups[start:stop] for interval in q_intervals], device
            ),  # [N_Q_pass]
            kv_gather=_gather_ranges(
                [interval for _, kv_intervals in groups[start:stop] for interval in kv_intervals], device
            ),  # [N_KV_pass]
            q_offsets=tuple(accumulate(q_lengths[start:stop], initial=0)),
            kv_offsets=tuple(accumulate(kv_lengths[start:stop], initial=0)),
        )
        for start, stop in pairwise(boundaries)
    )


def _segments(
    visibility: torch.Tensor,  # [R_Q,R_KV] on CPU
    q_starts: np.ndarray,  # [R_Q+1]
    kv_starts: np.ndarray,  # [R_KV+1]
    kv_apart: np.ndarray | None = None,  # [R_KV] key runs that start a new key interval
) -> list[GappedSegment]:
    """In-place segments for any visibility: every query interval of a group reads every key interval of it."""
    return [
        GappedSegment(q_row, q_end - q_row, k_row, k_end - k_row)
        for q_intervals, kv_intervals in _run_groups(visibility, q_starts, kv_starts, kv_apart)
        for q_row, q_end in q_intervals
        for k_row, k_end in kv_intervals
    ]


def require_gapped_replay(device: torch.device) -> None:
    """Raise unless gapped same-view replay can run on ``device``; maskless replay has no other same-view path."""
    reason = gapped_replay_unavailable_reason(device)
    if reason is not None:
        raise RuntimeError(f"Maskless replay runs its same-view partition with gapped replay, but {reason}.")


def build_replay_maskless_plan(
    metadata: TeacherForcingFlexMetadata,
    sensor_items: Sequence[Sequence[SensorMaskItem]],
) -> ReplayMasklessPlan:
    """Intersect replay roles/causality with the upstream maskless partitions.

    The same-view partition runs as in-place gapped segments; cross-instant and caption are gathered.
    """
    policy = metadata.teacher_forcing_replay_policy
    if policy.multiview_attention_scope not in ("same_view", "decomposed"):
        raise ValueError("Maskless replay supports same_view or decomposed scope.")
    device = metadata.sample_id.device
    q, q_starts = _run_fields(_query_stream_fields(metadata))
    kv, kv_starts = _run_fields(_key_stream_fields(metadata))
    # Spatial scope is applied by the partitions below. Keep every existing role,
    # sample, condition, caption and causal rule in the shared replay predicate.
    role_policy = attrs.evolve(policy, multiview_attention_scope="all_views")
    q_ids = torch.arange(q.sample_id.numel())[:, None]  # [R_Q,1]
    kv_ids = torch.arange(kv.sample_id.numel())[None, :]  # [1,R_KV]
    unused = torch.tensor(0)  # []
    allowed = _teacher_forcing_pair_predicate(q, kv, role_policy)(unused, unused, q_ids, kv_ids)  # [R_Q,R_KV]
    caption = kv.token_role_id[None, :] == _ROLE_UND  # [1,R_KV]
    view_match = q.view_id[:, None] == kv.view_id[None, :]  # [R_Q,R_KV]
    # Padding queries reach only padding keys. Nothing computes them, so their outputs stay zero.
    real = q.token_role_id != _ROLE_PADDING  # [R_Q]
    same_view = allowed & ~caption & view_match & real[:, None]  # [R_Q,R_KV]
    partitions: list[tuple[str, torch.Tensor]] = []  # each [R_Q,R_KV]
    if policy.multiview_attention_scope == "decomposed":
        q_instants, cross_enabled = _instant_ids(q, sensor_items)  # [R_Q], [R_Q]
        kv_instants, _ = _instant_ids(kv, sensor_items)  # [R_KV], [R_KV]
        window = policy.decomposed_temporal_window_seconds
        if window is None:
            # Match the bidirectional maskless model's camera-anchored instants.
            # All spatial tokens at that instant remain visible, subject to replay roles.
            cross_time = q_instants[:, None] == kv_instants[None, :]  # [R_Q,R_KV]
        else:
            # Physical-time windows may cross replay chunks. The shared role
            # predicate still enforces clean/noisy ownership and hides future chunks.
            gap = q.timestamp[:, None] - kv.timestamp[None, :]  # [R_Q,R_KV]
            cross_time = (gap >= -DECOMPOSED_TEMPORAL_WINDOW_EPS) & (
                gap <= window + DECOMPOSED_TEMPORAL_WINDOW_EPS
            )  # [R_Q,R_KV]
        cross = (
            allowed
            & ~caption
            & real[:, None]
            & ~q.is_control[:, None]
            & ~kv.is_control[None, :]
            & cross_enabled[:, None]
            & cross_time
        )  # [R_Q,R_KV]
        # Preserve production maskless weighting: a same-view key present in
        # both partitions contributes twice to the merged softmax denominator.
        partitions.append(("cross_instant" if window is None else "cross_window", cross))
    partitions.append(("caption", allowed & caption))  # [R_Q,R_KV]
    passes = tuple(
        plan for name, visibility in partitions for plan in _build_passes(name, visibility, q_starts, kv_starts, device)
    )
    # A single-view item's controls sit right before its generated tokens. Their histories stay separate
    # segments, so no query tile walks both serially: FA4 splits segments across launches, not keys.
    control = kv.is_control.numpy()  # [R_KV]
    segments = _segments(same_view, q_starts, kv_starts, np.concatenate(([False], control[1:] != control[:-1])))
    return ReplayMasklessPlan(
        passes=passes,
        q_len=metadata.q_len,
        kv_len=metadata.seq_len,
        gapped=build_gapped_replay_plan(segments, passes, device),
    )


def replay_maskless_attention(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,D]
    plan: ReplayMasklessPlan,
) -> torch.Tensor:  # [1,Q,H,D]
    """Run the gapped same-view launches and gathered passes, merged by log-sum-exp, as one opaque operator."""
    if is_torch_compiling():
        # State the plan invariants without branching on unbacked sequence lengths.
        torch._check(q.shape[1] == plan.q_len)
        torch._check(k.shape[1] == plan.kv_len)
        torch._check(v.shape[1] == plan.kv_len)
    elif q.shape[1] != plan.q_len or k.shape[1] != plan.kv_len or v.shape[1] != plan.kv_len:
        raise ValueError("Maskless replay plan does not match the global Q/KV streams.")
    # No part reaches padding queries, so their outputs stay zero.
    out, _ = _gapped_replay_forward(q, k, v, plan.gapped.handle)  # [1,Q,H,D], [1,Q,H]
    return out


def _flash_attn_interface() -> ModuleType:
    """Load the optional FA4 backend only when checking or executing gapped replay."""
    return import_module("flash_attn.cute.interface")


def gapped_replay_unavailable_reason(device: torch.device) -> str | None:
    """Why gapped replay cannot run on ``device``, or ``None`` if it can."""
    if device.type != "cuda":
        return f"it needs a CUDA device, got {device}"
    if torch.version.hip is not None:
        return "it needs NVIDIA CUDA, not ROCm"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (9, 0) and major != 10:
        # Hopper and Blackwell use different FA4 kernels; the replay GPU tests cover
        # the segment isolation and untouched gap rows required by both paths.
        return f"it needs FlashAttention-4 on Hopper (sm90) or Blackwell (sm100/sm103), got sm{major}{minor}"
    try:
        _flash_attn_interface()
    except ImportError as error:
        return f"it needs FlashAttention-4 (flash_attn.cute.interface), which could not be imported: {error}"
    return None


class _Intervals:
    """Disjoint half-open row intervals, sorted by start."""

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []

    def is_free(self, start: int, end: int) -> bool:
        index = bisect.bisect_right(self.starts, start)
        if index and self.ends[index - 1] > start:
            return False
        return index == len(self.starts) or self.starts[index] >= end

    def add(self, start: int, end: int) -> None:
        index = bisect.bisect_right(self.starts, start)
        self.starts.insert(index, start)
        self.ends.insert(index, end)


def _coalesce(segments: Sequence[GappedSegment]) -> list[GappedSegment]:
    """Join segments that read the same key rows from adjacent query rows."""
    merged: list[GappedSegment] = []
    for segment in sorted(segments, key=lambda s: (s.k_row, s.k_len, s.q_row)):
        last = merged[-1] if merged else None
        if last is not None and (last.k_row, last.k_len, last.q_row + last.q_len) == (
            segment.k_row,
            segment.k_len,
            segment.q_row,
        ):
            merged[-1] = last._replace(q_len=last.q_len + segment.q_len)
        else:
            merged.append(segment)
    return merged


def _chains(segments: list[GappedSegment]) -> list[list[GappedSegment]]:
    """Consecutive query rows reading growing key ranges from one key row on, like a causal history.

    Every segment of a chain covers its first key row, so a chain costs one launch per segment
    wherever it goes; two chains share launches only when their query and key rows are disjoint.
    """
    chains: list[list[GappedSegment]] = []
    for segment in sorted(segments, key=lambda s: (s.k_row, s.q_row)):
        last = chains[-1][-1] if chains else None
        if (
            last is not None
            and (last.k_row, last.q_row + last.q_len) == (segment.k_row, segment.q_row)
            and last.k_len <= segment.k_len
        ):
            chains[-1].append(segment)
        else:
            chains.append([segment])
    return chains


def pack_segments(segments: Sequence[GappedSegment]) -> tuple[GappedComponent, ...]:
    """Pack segments into query-disjoint components, each split into key-disjoint launches.

    A component needs as many launches as its deepest key row. Longest chains first, each chain
    joins the query-disjoint component it deepens least. Every packing needs one component per
    segment over the busiest query row, so below that count a chain that would deepen every
    candidate opens a new component instead, where later chains over other rows can run beside
    it. Within a component, interval colouring by key start then reaches its depth.
    """
    segments = _coalesce(segments)
    if not segments:
        return ()
    table = np.asarray(segments, dtype=np.int64)  # [N,4]
    bounds = np.unique(np.concatenate([table[:, 2], table[:, 2] + table[:, 3]]))  # [B]
    q_starts, q_ends = np.sort(table[:, 0]), np.sort(table[:, 0] + table[:, 1])  # [N], [N]
    needed = int((np.arange(1, len(segments) + 1) - np.searchsorted(q_ends, q_starts, side="right")).max())
    occupied: list[_Intervals] = []
    members: list[list[GappedSegment]] = []
    depths: list[np.ndarray] = []  # per component: [B] segments over each key interval
    deepest: list[int] = []
    for chain in sorted(_chains(segments), key=len, reverse=True):
        q_lo, q_hi = chain[0].q_row, chain[-1].q_row + chain[-1].q_len
        keys = np.asarray([(s.k_row, s.k_row + s.k_len) for s in chain], dtype=np.int64)  # [C,2]
        first, end = np.searchsorted(bounds, keys[:, 0]), np.searchsorted(bounds, keys[:, 1])  # [C], [C]
        profile = np.zeros(bounds.size, dtype=np.int32)  # [B]
        np.add.at(profile, first, 1)
        np.add.at(profile, end, -1)
        profile = profile.cumsum(dtype=np.int32)  # [B]
        lo, hi = int(first.min()), int(end.max())
        chosen, cost = None, 0
        for index, rows in enumerate(occupied):
            if rows.is_free(q_lo, q_hi):
                increase = max(int((depths[index][lo:hi] + profile[lo:hi]).max()) - deepest[index], 0)
                if chosen is None or increase < cost:
                    chosen, cost = index, increase
        if chosen is None or (cost > 0 and len(occupied) < needed):
            chosen = len(occupied)
            occupied.append(_Intervals())
            members.append([])
            depths.append(np.zeros(bounds.size, dtype=np.int32))
            deepest.append(0)
        occupied[chosen].add(q_lo, q_hi)
        members[chosen].extend(chain)
        depths[chosen] += profile
        deepest[chosen] = max(deepest[chosen], int(depths[chosen][lo:hi].max()))
    components: list[GappedComponent] = []
    for rows, component_segments in zip(occupied, members):
        launches: list[list[GappedSegment]] = []
        ends: list[tuple[int, int]] = []  # min-heap of (key end of the launch's last segment, launch)
        for segment in sorted(component_segments, key=lambda s: (s.k_row, s.q_row)):
            if ends and ends[0][0] <= segment.k_row:
                _, index = heapq.heappop(ends)
            else:
                index = len(launches)
                launches.append([])
            launches[index].append(segment)
            heapq.heappush(ends, (segment.k_row + segment.k_len, index))
        components.append(
            GappedComponent(
                q_lo=rows.starts[0],
                q_hi=max(rows.ends),
                launches=tuple(GappedLaunch(tuple(sorted(launch))) for launch in launches),
            )
        )
    return tuple(components)


class _Launch(NamedTuple):
    """Host geometry of one launch; the segment tensors are views of the plan's offset tables."""

    k_lo: int  # the launch reads rows [k_lo, k_hi) of its part's K/V
    k_hi: int
    max_q: int
    max_k: int
    offset: int  # first entry of this launch in both offset tables
    count: int  # segments


class _Part(NamedTuple):
    """Launches sharing one output/LSE and dQ buffer: a component in place, or a gathered pass."""

    q_rows: slice | torch.Tensor  # the buffer's global query rows; gathered rows are unique
    kv_rows: torch.Tensor | None  # [N_KV] rows to gather K/V from, or None to read them in place
    rows: int
    launches: tuple[_Launch, ...]


_SegmentTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _Runtime:
    """Device data one plan's operators read. It lives exactly as long as the plan's handle."""

    def __init__(
        self,
        components: tuple[GappedComponent, ...],
        passes: tuple["ReplayMasklessPass", ...],
        device: torch.device,
    ) -> None:
        q_values: list[int] = []
        k_values: list[int] = []

        def launch(q_segments: list[tuple[int, int]], q_total: int, k_segments: list[tuple[int, int]]) -> _Launch:
            # FA4 reads segment i at start cu_seqlens[i] with length seqused[i]; cu_seqlens[-1] is the span.
            k_lo = min(start for start, _ in k_segments)
            k_hi = max(start + length for start, length in k_segments)
            offset = len(q_values)
            q_values.extend([start for start, _ in q_segments] + [q_total] + [length for _, length in q_segments])
            k_values.extend(
                [start - k_lo for start, _ in k_segments] + [k_hi - k_lo] + [length for _, length in k_segments]
            )
            return _Launch(
                k_lo,
                k_hi,
                max(length for _, length in q_segments),
                max(length for _, length in k_segments),
                offset,
                len(q_segments),
            )

        parts = [
            _Part(
                slice(component.q_lo, component.q_hi),
                None,
                component.q_hi - component.q_lo,
                tuple(
                    launch(
                        [(s.q_row - component.q_lo, s.q_len) for s in gapped.segments],
                        component.q_hi - component.q_lo,
                        [(s.k_row, s.k_len) for s in gapped.segments],
                    )
                    for gapped in component.launches
                ),
            )
            for component in components
        ]
        for partition in passes:
            # Each query group reads its own contiguous key range of the gathered buffers.
            q_bounds = partition.q_offsets
            k_bounds = partition.kv_offsets
            segments = launch(
                [(start, end - start) for start, end in pairwise(q_bounds)],
                q_bounds[-1],
                [(start, end - start) for start, end in pairwise(k_bounds)],
            )
            parts.append(_Part(partition.q_gather, partition.kv_gather, q_bounds[-1], (segments,)))
        self.parts = tuple(parts)
        self.q_offsets = torch.tensor(q_values, dtype=torch.int32, device=device)  # [N_offsets]
        self.k_offsets = torch.tensor(k_values, dtype=torch.int32, device=device)  # [N_offsets]
        self._segments: dict[int, dict[int, _SegmentTensors]] = {}

    def segment_tensors(self, launch: _Launch, fold: int) -> _SegmentTensors:  # [B+1], [B], [B+1], [B]
        """``cu_seqlens_q``, ``seqused_q``, ``cu_seqlens_k``, ``seqused_k`` views for one launch.

        Query offsets count rows of the ``[rows*fold, H_KV, D]`` layout. Prepare all views once per
        fold; the plan-owned cache keeps their storage alive without recreating slices per step.
        """
        if fold not in self._segments:
            q_offsets = self.q_offsets if fold == 1 else self.q_offsets * fold  # [N_offsets]
            prepared: dict[int, _SegmentTensors] = {}
            for part in self.parts:
                for entry in part.launches:
                    start, count = entry.offset, entry.count
                    prepared[start] = (
                        q_offsets[start : start + count + 1],  # [B+1]
                        q_offsets[start + count + 1 : start + 2 * count + 1],  # [B]
                        self.k_offsets[start : start + count + 1],  # [B+1]
                        self.k_offsets[start + count + 1 : start + 2 * count + 1],  # [B]
                    )
            self._segments[fold] = prepared
        return self._segments[fold][launch.offset]


_RUNTIMES: dict[int, _Runtime] = {}
_HANDLE_IDS = itertools.count(1)


def build_gapped_replay_plan(
    segments: Sequence[GappedSegment],
    passes: Sequence["ReplayMasklessPass"],
    device: torch.device,
) -> GappedReplayPlan:
    """Pack the in-place segments and register the device data the operators read through the handle.

    ``passes`` are the gathered varlen passes merged with the segments. Autograd saves the handle
    for backward, so the registered data stays alive until every forward and backward that can
    read it has released the handle.
    """
    components = pack_segments(segments)
    if not components and not passes:
        raise ValueError("Maskless replay has no nonempty attention groups.")
    key = next(_HANDLE_IDS)
    handle = torch.tensor([key], dtype=torch.int64)  # [1] on CPU
    _RUNTIMES[key] = _Runtime(components, tuple(passes), device)
    weakref.finalize(handle, _RUNTIMES.pop, key, None)
    return GappedReplayPlan(handle=handle, components=components)


def _runtime(handle: torch.Tensor) -> _Runtime:
    if handle.device.type != "cpu" or handle.dtype != torch.int64 or handle.numel() != 1:
        raise ValueError("A gapped replay handle is the plan's CPU int64 [1] tensor.")
    runtime = _RUNTIMES.get(int(handle.item()))
    if runtime is None:
        raise RuntimeError("This gapped replay handle is not registered; pass the plan's own handle, not a copy.")
    return runtime


def _accumulate_key_segments(
    dk: torch.Tensor,  # [K_span,H_KV,D], valid on segment rows only
    dv: torch.Tensor,  # [K_span,H_KV,Dv], valid on segment rows only
    dk_total: torch.Tensor,  # [KV,H_KV,D] FP32
    dv_total: torch.Tensor,  # [KV,H_KV,Dv] FP32
    k_lo: int,
    starts: torch.Tensor,  # [B] int32, relative to k_lo
    lengths: torch.Tensor,  # [B] int32
    max_length: int,
) -> None:
    """Add the segment rows of one launch's dK/dV into the FP32 totals (segments are key-disjoint)."""
    k_row, v_row = dk[0].numel(), dv[0].numel()
    # Keep each program's tile bounded as more head columns share the launch.
    block_rows = max(1, _ACCUMULATE_BLOCK_ROWS // triton.next_power_of_2(dk.shape[1]))
    # Long key segments can need more than CUDA's 65,535 blocks on grid axis 1.
    grid = (triton.cdiv(max_length, block_rows), lengths.numel())
    kernels.accumulate[grid](
        dk,
        dv,
        dk_total,
        dv_total,
        starts,
        lengths,
        k_lo,
        K_ROW=k_row,
        V_ROW=v_row,
        BLOCK_ROWS=block_rows,
        BLOCK_K=triton.next_power_of_2(k_row),
        BLOCK_V=triton.next_power_of_2(v_row),
    )


def _launch_rows(
    kernel: Any, rows: slice | torch.Tensor, length: int, *tensors: torch.Tensor, **options: int | bool
) -> None:
    """Launch a row kernel with either a contiguous slice or an index map (all tensors are contiguous)."""
    indexed = isinstance(rows, torch.Tensor)
    indices = rows if indexed else tensors[0]  # [N] or unused tensor
    start = (rows.start or 0) if isinstance(rows, slice) else 0
    kernel[((length + 1023) // 1024,)](*tensors, indices, length, start, INDEXED=indexed, BLOCK=1024, **options)


def _query_layout(
    tensor: torch.Tensor, rows: slice | torch.Tensor, part: torch.Tensor, *, add: bool = False
) -> None:  # tensor: [Q,H,D], part: [N*G,H_KV,D]
    """Copy query rows into FA4 layout, or unfold and add gradients without a transposed temporary."""
    _launch_rows(
        kernels.query_layout,
        rows,
        part.numel(),
        tensor,
        part,
        KV_HEADS=part.shape[1],
        FOLD=tensor.shape[1] // part.shape[1],
        DIM=part.shape[2],
        ADD=add,
    )


def _query_part(
    tensor: torch.Tensor, rows: slice | torch.Tensor, kv_heads: int
) -> torch.Tensor:  # [Q,H,D] -> [N*G,H_KV,D]
    """Gather and fold the G query heads per KV head together, keeping KV heads in columns."""
    if kv_heads == 1:
        # Nano at CP8/CP16 needs no head reorder; only indexed rows allocate a gather.
        return _rows(tensor, rows).view(-1, 1, tensor.shape[-1])  # [N*H,1,D]
    count = rows.stop - (rows.start or 0) if isinstance(rows, slice) else rows.numel()
    part = tensor.new_empty((count * (tensor.shape[1] // kv_heads), kv_heads, tensor.shape[2]))  # [N*G,H_KV,D]
    _query_layout(tensor, rows, part)
    return part


def _rows(tensor: torch.Tensor, rows: slice | torch.Tensor | None) -> torch.Tensor:  # [N,...] -> [N_part,...]
    """A part's rows: a view when in place, a copy when gathered."""
    return tensor if rows is None else tensor[rows] if isinstance(rows, slice) else tensor.index_select(0, rows)


def _add_rows(total: torch.Tensor, rows: torch.Tensor, value: torch.Tensor) -> None:
    """Deterministically add gathered ``value`` ([N_part,...]) at ``rows`` ([N_part]) into ``total`` ([N,...])."""
    # Gathered keys can repeat. CUDA index_put_(accumulate=True) uses the same sorted reduction
    # as deterministic index_add_, without changing the global determinism mode used by FA4.
    total.index_put_((rows,), value.to(total.dtype), accumulate=True)  # [N,...]


def _merge_parts(
    q: torch.Tensor,  # [1,Q,H,D]
    parts: list[tuple[slice | torch.Tensor, torch.Tensor, torch.Tensor]],  # rows, [N*G,H_KV,Dv], [H_KV,N*G]
    dim_v: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,Dv], [1,Q,H]
    """Merge each part with fused row mapping, normalization and FP32 updates."""
    q_len, heads = q.shape[1:3]
    # Two-phase merge: the FP32 log-sum-exp of every part first, then each part's weighted output.
    top = q.new_full((q_len, heads), -torch.inf, dtype=torch.float32)  # [Q,H]
    total = torch.zeros_like(top)  # [Q,H]
    for summed in (False, True):
        for rows, _, lse in parts:
            _launch_rows(
                kernels.merge_lse_update,
                rows,
                lse.numel(),
                lse,
                top,
                total,
                HEADS=heads,
                KV_HEADS=kv_heads,
                SUM=summed,
            )
    merged_lse = torch.empty_like(top)  # [Q,H]; -inf where nothing reaches the row
    kernels.merge_lse_finish[(triton.cdiv(top.numel(), 1024),)](top, total, merged_lse, top.numel(), BLOCK=1024)
    del top, total
    # Rows that no part reaches keep weight exp(-inf - 0) = 0 instead of NaN.
    merged = q.new_zeros((q_len, heads, dim_v), dtype=torch.float32)  # [Q,H,Dv]
    while parts:
        rows, out, lse = parts.pop(0)
        _launch_rows(
            kernels.merge_output_update,
            rows,
            out.numel(),
            out,
            lse,
            merged_lse,
            merged,
            HEADS=heads,
            KV_HEADS=kv_heads,
            DIM=dim_v,
        )
        del out, lse
    return merged.to(q.dtype).unsqueeze(0), merged_lse.unsqueeze(0)  # [1,Q,H,Dv], [1,Q,H]


@torch.library.custom_op("cosmos3::gapped_replay_forward", mutates_args=(), device_types="cuda")
def _gapped_replay_forward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,Dv]
    handle: torch.Tensor,  # [1] int64 on CPU
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,Dv], [1,Q,H] FP32
    """Run each part's native forward and merge by FP32 log-sum-exp. Rows nothing reaches are zero."""
    flash_attn = _flash_attn_interface()
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"Query heads ({q.shape[2]}) must be a multiple of KV heads ({k.shape[2]}).")
    runtime = _runtime(handle)
    q, k, v = (tensor.contiguous() for tensor in (q, k, v))  # [1,Q,H,D], [1,KV,H_KV,D], [1,KV,H_KV,Dv]
    kv_heads = k.shape[2]
    heads, dim = q.shape[2:]
    fold, dim_v = heads // kv_heads, v.shape[-1]
    q_rows, k_rows, v_rows = (tensor.squeeze(0) for tensor in (q, k, v))  # [Q,H,D], [KV,H_KV,D], [KV,H_KV,Dv]
    parts: list[tuple[slice | torch.Tensor, torch.Tensor, torch.Tensor]] = []  # rows, [N*G,H_KV,Dv], [H_KV,N*G]
    for part in runtime.parts:
        # The query group is folded into rows: FA4 sees H_KV heads over rows*G rows.
        q_part = _query_part(q_rows, part.q_rows, kv_heads)  # [N*G,H_KV,D]
        k_part = _rows(k_rows, part.kv_rows)  # [KV|N_KV,H_KV,D]
        v_part = _rows(v_rows, part.kv_rows)  # [KV|N_KV,H_KV,Dv]
        # The launches write only their segment rows; the others keep zero / -inf.
        out = q.new_zeros(part.rows * fold, kv_heads, dim_v)  # [N*G,H_KV,Dv]
        lse = q.new_full((kv_heads, part.rows * fold), -torch.inf, dtype=torch.float32)  # [H_KV,N*G]
        for launch in part.launches:
            cu_q, used_q, cu_k, used_k = runtime.segment_tensors(launch, fold)
            k_span = k_part[launch.k_lo : launch.k_hi]  # [K_span,H_KV,D]
            v_span = v_part[launch.k_lo : launch.k_hi]  # [K_span,H_KV,Dv]
            flash_attn._flash_attn_fwd(
                q_part,
                k_span,
                v_span,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                seqused_q=used_q,
                seqused_k=used_k,
                max_seqlen_q=launch.max_q * fold,
                max_seqlen_k=launch.max_k,
                softmax_scale=dim**-0.5,
                out=out,
                lse=lse,
                return_lse=True,
            )
            del k_span, v_span
        parts.append((part.q_rows, out, lse))
        del q_part, k_part, v_part, out, lse
    return _merge_parts(q, parts, dim_v, kv_heads)  # [1,Q,H,Dv], [1,Q,H]


@_gapped_replay_forward.register_fake
def _gapped_replay_forward_fake(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,Dv]
    handle: torch.Tensor,  # [1] int64 on CPU
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,Q,H,Dv], [1,Q,H] FP32
    return q.new_empty((*q.shape[:-1], v.shape[-1])), q.new_empty(q.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("cosmos3::gapped_replay_backward", mutates_args=(), device_types="cuda")
def _gapped_replay_backward(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,Dv]
    out: torch.Tensor,  # [1,Q,H,Dv]
    lse: torch.Tensor,  # [1,Q,H] FP32
    grad_out: torch.Tensor,  # [1,Q,H,Dv]
    handle: torch.Tensor,  # [1] int64 on CPU
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [1,Q,H,D], [1,KV,H_KV,D], [1,KV,H_KV,Dv]
    """Run native backward against merged O/LSE/dO and accumulate dQ/dK/dV in FP32.

    Every launch uses the same folded query layout and segment arguments as forward.
    Rows nothing reaches are zero.
    """
    flash_attn = _flash_attn_interface()
    runtime = _runtime(handle)
    q, k, v = (tensor.contiguous() for tensor in (q, k, v))  # [1,Q,H,D], [1,KV,H_KV,D], [1,KV,H_KV,Dv]
    out, lse, grad_out = (tensor.contiguous() for tensor in (out, lse, grad_out))  # [1,Q,H,Dv], [1,Q,H], [1,Q,H,Dv]
    kv_heads = k.shape[2]
    heads, dim = q.shape[2:]
    fold = heads // kv_heads
    q_rows, k_rows, v_rows = (tensor.squeeze(0) for tensor in (q, k, v))  # [Q,H,D], [KV,H_KV,D], [KV,H_KV,Dv]
    o_rows, lse_rows, g_rows = (tensor.squeeze(0) for tensor in (out, lse, grad_out))  # [Q,H,Dv], [Q,H], [Q,H,Dv]
    dq, dk, dv = (
        torch.zeros(x.shape, dtype=torch.float32, device=q.device) for x in (q_rows, k_rows, v_rows)
    )  # [Q,H,D], [KV,H_KV,D], [KV,H_KV,Dv]
    for part in runtime.parts:
        q_part = _query_part(q_rows, part.q_rows, kv_heads)  # [N*G,H_KV,D]
        k_part = _rows(k_rows, part.kv_rows)  # [KV|N_KV,H_KV,D]
        v_part = _rows(v_rows, part.kv_rows)  # [KV|N_KV,H_KV,Dv]
        # Q, merged O, dO and LSE rows with the query heads folded in, and a dQ buffer that the
        # part's launches write on disjoint rows.
        o_part = _query_part(o_rows, part.q_rows, kv_heads)  # [N*G,H_KV,Dv]
        g_part = _query_part(g_rows, part.q_rows, kv_heads)  # [N*G,H_KV,Dv]
        lse_part = _rows(lse_rows, part.q_rows).view(part.rows, kv_heads, fold)  # [N,H_KV,G]
        lse_part = lse_part.transpose(0, 1).reshape(kv_heads, -1).contiguous()  # [H_KV,N*G]
        dq_part = torch.zeros_like(q_part)  # [N*G,H_KV,D]
        for launch in part.launches:
            cu_q, used_q, cu_k, used_k = runtime.segment_tensors(launch, fold)
            k_span = k_part[launch.k_lo : launch.k_hi]  # [K_span,H_KV,D]
            v_span = v_part[launch.k_lo : launch.k_hi]  # [K_span,H_KV,Dv]
            dk_span = torch.empty_like(k_span)  # written on segment rows only, and only those are read
            dv_span = torch.empty_like(v_span)  # [K_span,H_KV,Dv]
            flash_attn._flash_attn_bwd(
                q_part,
                k_span,
                v_span,
                o_part,
                g_part,
                lse_part,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                seqused_q=used_q,
                seqused_k=used_k,
                max_seqlen_q=launch.max_q * fold,
                max_seqlen_k=launch.max_k,
                softmax_scale=dim**-0.5,
                dq=dq_part,
                dk=dk_span,
                dv=dv_span,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            del k_span, v_span
            if part.kv_rows is None:
                _accumulate_key_segments(dk_span, dv_span, dk, dv, launch.k_lo, cu_k[:-1], used_k, launch.max_k)
            else:
                # The groups tile the gathered keys; a key gathered for several groups adds once per group.
                _add_rows(dk, part.kv_rows[launch.k_lo : launch.k_hi], dk_span)
                _add_rows(dv, part.kv_rows[launch.k_lo : launch.k_hi], dv_span)
            del dk_span, dv_span
        # dQ accumulation is still needed with one local KV head, including at CP16.
        _query_layout(dq, part.q_rows, dq_part, add=True)  # [Q,H,D]
        del q_part, k_part, v_part, o_part, g_part, lse_part, dq_part
    # Retire each FP32 total as its gradient is cast, so two full copies never coexist.
    grad_q = dq.to(q.dtype).view(q.shape)  # [1,Q,H,D]
    del dq
    grad_k = dk.to(k.dtype).view(k.shape)  # [1,KV,H_KV,D]
    del dk
    return grad_q, grad_k, dv.to(v.dtype).view(v.shape)  # [1,Q,H,D], [1,KV,H_KV,D], [1,KV,H_KV,Dv]


@_gapped_replay_backward.register_fake
def _gapped_replay_backward_fake(
    q: torch.Tensor,  # [1,Q,H,D]
    k: torch.Tensor,  # [1,KV,H_KV,D]
    v: torch.Tensor,  # [1,KV,H_KV,Dv]
    out: torch.Tensor,  # [1,Q,H,Dv]
    lse: torch.Tensor,  # [1,Q,H] FP32
    grad_out: torch.Tensor,  # [1,Q,H,Dv]
    handle: torch.Tensor,  # [1] int64 on CPU
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [1,Q,H,D], [1,KV,H_KV,D], [1,KV,H_KV,Dv]
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
    q, k, v, handle = inputs
    out, lse = output
    # The forward is a pure function of these, so activation checkpointing may recompute it.
    ctx.save_for_backward(q, k, v, out, lse, handle)


def _backward(
    ctx: Any, grad_out: torch.Tensor | None, grad_lse: torch.Tensor | None
) -> tuple[torch.Tensor | None, ...]:
    del grad_lse  # The merged LSE only carries the forward state to the native backward.
    if grad_out is None:
        return None, None, None, None
    q, k, v, out, lse, handle = ctx.saved_tensors
    dq, dk, dv = _gapped_replay_backward(q, k, v, out, lse, grad_out, handle)
    return dq, dk, dv, None


torch.library.register_autograd("cosmos3::gapped_replay_forward", _backward, setup_context=_setup_context)
