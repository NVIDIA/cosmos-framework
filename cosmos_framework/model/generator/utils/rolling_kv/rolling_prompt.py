# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded segmented prompt visibility and immutable initial text-prefix K/V."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate

import torch

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack


@dataclass(frozen=True)
class RollingPromptSegment:
    """A half-open source-frame interval and its unframed tokenizer output."""

    start_frame: int
    end_frame: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class RollingPromptDocument:
    """One independently encoded caption in the current UND stream."""

    view_id: int
    start_frame: int
    end_frame: int
    token_ids: tuple[int, ...]
    is_sink: bool = False
    # Caption identity keeps its original interval. Only direct media visibility
    # expands when neighboring caption context is requested.
    visible_start_frame: int | None = None
    visible_end_frame: int | None = None


@dataclass(frozen=True)
class RollingPromptLayout:
    """Global post-Ulysses text coordinates, independent of local head shards."""

    documents: tuple[RollingPromptDocument, ...]
    fps: float
    sink_tokens: int
    num_views: int
    # These are global UND token indices; Ulysses shards heads at attention time.
    sink_capture_indexes: torch.Tensor  # [V*K]
    sink_read_indexes: torch.Tensor  # [V*K]
    sink_cache_indexes: torch.Tensor  # [V*K]

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(doc.token_ids) for doc in self.documents)

    def install_document_offsets(self, input_pack: SequencePack, num_und: int) -> None:
        """Keep time segments and cameras isolated, including CP's pad document."""
        offsets = list(accumulate(self.lengths, initial=0))
        if offsets[-1] > num_und:
            raise ValueError("Segmented prompt document lengths exceed the global UND stream.")
        if offsets[-1] < num_und:
            offsets.append(num_und)
        input_pack["_caption_seq_offsets"] = torch.tensor(
            offsets, dtype=torch.int32, device=self.sink_read_indexes.device
        )  # [N_docs+1 or N_docs+2]
        input_pack["max_caption_len"] = max(b - a for a, b in zip(offsets, offsets[1:]))

    def time_bounds(self, kv_length: int) -> tuple[torch.Tensor, torch.Tensor]:  # returns 2*[KV]
        starts = torch.full((kv_length,), -torch.inf, device=self.sink_read_indexes.device)  # [KV]
        ends = torch.full((kv_length,), torch.inf, device=starts.device)  # [KV]
        offset = 0
        for doc in self.documents:
            stop = offset + len(doc.token_ids)
            if not doc.is_sink:
                first = doc.start_frame if doc.visible_start_frame is None else doc.visible_start_frame
                last = doc.end_frame if doc.visible_end_frame is None else doc.visible_end_frame
                starts[offset:stop] = first / self.fps  # [N_doc]
                ends[offset:stop] = last / self.fps  # [N_doc]
            offset = stop
        return starts, ends  # 2*[KV]


@dataclass(frozen=True)
class RollingPromptPack:
    """Current documents, grouped by camera for the existing base mask builder."""

    documents: tuple[RollingPromptDocument, ...]
    text_tokens: list[list[int]]
    fps: float
    sink_tokens: int
    num_views: int
    fixed_text_extent: int

    def prepare(
        self,
        packed: PackedSequence,
        *,
        position_mode: str = "fixed",
        temporal_units_per_second: float = 1.0,
        chunk_start_seconds: float = 0.0,
    ) -> RollingPromptLayout:
        """Position independent documents without changing their visibility or sinks."""
        if position_mode not in ("fixed", "segment_start", "chunk_start"):
            raise ValueError("Rolling text position mode must be fixed, segment_start or chunk_start.")
        if (
            not math.isfinite(temporal_units_per_second)
            or temporal_units_per_second <= 0
            or not math.isfinite(chunk_start_seconds)
            or chunk_start_seconds < 0
        ):
            raise ValueError("Rolling text positioning requires a positive temporal scale and nonnegative chunk time.")
        lengths = [sum(len(d.token_ids) for d in self.documents if d.view_id == view) for view in range(self.num_views)]
        if packed.text_caption_lens != [lengths]:
            raise ValueError("Prompt framing disagrees with native caption packing.")
        if position_mode != "fixed" and not packed.position_ids.is_floating_point():
            packed.position_ids = packed.position_ids.float()  # [3,S]
        offset = 0
        for doc in self.documents:
            length = len(doc.token_ids)
            # The native per-view pack rewinds each view to zero. Rewind each
            # independent time segment as well. Optional temporal translation
            # follows the absolute rollout clock; sink K/V stays at its origin.
            positions = torch.arange(
                length, dtype=packed.position_ids.dtype, device=packed.position_ids.device
            )  # [N_doc]
            packed.position_ids[:, offset : offset + length] = positions[None, :]  # [3,N_doc]
            if not doc.is_sink and position_mode != "fixed":
                seconds = doc.start_frame / self.fps if position_mode == "segment_start" else chunk_start_seconds
                packed.position_ids[0, offset : offset + length] += seconds * temporal_units_per_second  # [N_doc]
            offset += length
        # Length changes must never move a media key relative to previously
        # committed keys. The native text cursor advances past the longest view.
        delta = self.fixed_text_extent - max(lengths)
        for modality in (packed.vision, packed.lidar):
            if modality is not None:
                packed.position_ids[0, modality.sequence_indexes] += delta  # [S_modality]
        return self.layout(packed.position_ids.device)

    def layout(self, device: torch.device) -> RollingPromptLayout:
        """Build global attention indices once per current pack, not once per layer."""
        first_by_view: dict[int, int] = {}
        read_indexes: list[int] = []
        cache_indexes: list[int] = []
        offset = 0
        for doc in self.documents:
            if doc.is_sink:
                if doc.view_id in first_by_view or len(doc.token_ids) != self.sink_tokens:
                    raise ValueError("Segmented prompting requires exactly one fixed-size sink document per camera.")
                first_by_view[doc.view_id] = offset
                read_indexes.extend(range(offset, offset + self.sink_tokens))
                cache_indexes.extend(range(doc.view_id * self.sink_tokens, (doc.view_id + 1) * self.sink_tokens))
            offset += len(doc.token_ids)
        if self.sink_tokens and set(first_by_view) != set(range(self.num_views)):
            raise ValueError("Segmented prompting requires a retained sink for every camera.")
        capture_indexes = [
            index
            for view in range(self.num_views)
            for index in range(first_by_view.get(view, 0), first_by_view.get(view, 0) + self.sink_tokens)
        ]
        return RollingPromptLayout(
            documents=self.documents,
            fps=self.fps,
            sink_tokens=self.sink_tokens,
            num_views=self.num_views,
            sink_capture_indexes=torch.tensor(capture_indexes, device=device, dtype=torch.long),  # [V*K]
            sink_read_indexes=torch.tensor(read_indexes, device=device, dtype=torch.long),  # [V*K]
            sink_cache_indexes=torch.tensor(cache_indexes, device=device, dtype=torch.long),  # [V*K]
        )


class RollingPromptSchedule:
    """Read captions on the original finite video clock, without repeating them."""

    fps: float
    num_frames: int
    views: tuple[tuple[RollingPromptSegment, ...], ...]
    sink_tokens: int
    special_tokens: dict[str, int]
    prefixes: tuple[tuple[int, ...], ...]
    fixed_text_extent: int
    context_before: int
    context_after: int

    def __init__(
        self,
        *,
        fps: float,
        num_frames: int,
        views: tuple[tuple[RollingPromptSegment, ...], ...],
        sink_tokens: int,
        special_tokens: dict[str, int],
        context_before: int = 0,
        context_after: int = 0,
        media_text_extent: int | None = None,
    ) -> None:
        if (
            not math.isfinite(fps)
            or fps <= 0
            or isinstance(num_frames, bool)
            or not isinstance(num_frames, int)
            or num_frames < 1
            or not views
        ):
            raise ValueError("Segmented prompting requires a positive FPS, frame count and camera count.")
        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
            raise ValueError("Rolling text sink size must be a nonnegative integer.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (context_before, context_after)
        ):
            raise ValueError("Neighboring caption counts must be nonnegative integers.")
        self.fps = fps
        self.num_frames = num_frames
        self.views = views
        self.sink_tokens = sink_tokens
        self.special_tokens = special_tokens
        self.context_before = context_before
        self.context_after = context_after
        for segments in views:
            cursor = 0
            for segment in segments:
                if segment.start_frame != cursor or not cursor < segment.end_frame <= num_frames:
                    raise ValueError("Rolling captions must cover the video without gaps or overlaps.")
                cursor = segment.end_frame
            if cursor != num_frames:
                raise ValueError("Rolling captions do not cover the video.")
        # Empty captions are valid for unconditional sampling. A short prompt
        # retains only the available prefix; the full current document remains.
        self.sink_tokens = min(sink_tokens, min(len(self._frame(segments[0].token_ids)) for segments in views))
        self.prefixes = tuple(self._frame(segments[0].token_ids)[: self.sink_tokens] for segments in views)
        longest = max(len(self._frame(segment.token_ids)) for segments in views for segment in segments)
        if media_text_extent is not None and (
            isinstance(media_text_extent, bool) or not isinstance(media_text_extent, int) or media_text_extent < longest
        ):
            raise ValueError("Media text extent must cover the longest independently framed caption.")
        self.fixed_text_extent = longest if media_text_extent is None else media_text_extent

    def _frame(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        prefix = (self.special_tokens["bos_token_id"],) if "bos_token_id" in self.special_tokens else ()
        return prefix + tokens + (self.special_tokens["eos_token_id"], self.special_tokens["start_of_generation"])

    def for_times(self, timestamps: tuple[float, ...]) -> RollingPromptPack:
        if not timestamps or any(not math.isfinite(t) or t < 0 for t in timestamps):
            raise ValueError("Segmented prompting requires finite nonnegative query timestamps.")
        frames = sorted({math.floor(t * self.fps + 1e-5) for t in timestamps})
        if frames[-1] >= self.num_frames:
            raise ValueError("Rolling query exceeds the video horizon; prompts are never repeated.")
        documents: list[RollingPromptDocument] = []
        packed_tokens: list[list[int]] = []
        for view, segments in enumerate(self.views):
            selected: set[int] = set()
            for frame in frames:
                for index, segment in enumerate(segments):
                    if segment.start_frame <= frame < segment.end_frame:
                        selected.update(
                            range(
                                max(0, index - self.context_before), min(len(segments), index + self.context_after + 1)
                            )
                        )
                        break
            view_tokens: list[int] = []
            if self.sink_tokens:
                # Keep the original prefix in its own causal text document.
                # Current captions retain every token, including a different
                # opening word after a switch. Media can always read this sink.
                documents.append(RollingPromptDocument(view, 0, self.num_frames, self.prefixes[view], is_sink=True))
                view_tokens.extend(self.prefixes[view])
            for index in sorted(selected):
                segment = segments[index]
                tokens = self._frame(segment.token_ids)
                # A document d is visible to query-segment q exactly when
                # q-before <= d <= q+after. The inverse bounds are therefore
                # [d-after, d+before], clipped to this video's actual captions.
                first_index = max(0, index - self.context_after)
                last_index = min(len(segments) - 1, index + self.context_before)
                documents.append(
                    RollingPromptDocument(
                        view_id=view,
                        start_frame=segment.start_frame,
                        end_frame=segment.end_frame,
                        token_ids=tokens,
                        visible_start_frame=segments[first_index].start_frame,
                        visible_end_frame=segments[last_index].end_frame,
                    )
                )
                view_tokens.extend(tokens)
            # Native packing adds precisely these outside tokens. Internal
            # document delimiters stay explicit; the replay hook isolates them.
            prefix_count = int("bos_token_id" in self.special_tokens)
            packed_tokens.append(view_tokens[prefix_count:-2])
        return RollingPromptPack(
            documents=tuple(documents),
            text_tokens=packed_tokens,
            fps=self.fps,
            sink_tokens=self.sink_tokens,
            num_views=len(self.views),
            fixed_text_extent=self.fixed_text_extent,
        )


class RollingTextSinkLayer:
    """One layer's original RoPE K, normalized K and V; never overwritten."""

    cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None  # each [V*K,H_KV_local,D]
    reads: int

    def __init__(self) -> None:
        self.cached = None
        self.reads = 0

    def apply(
        self,
        k_rope: torch.Tensor,  # [UND,H_KV_local,D]
        k_normalized: torch.Tensor,  # [UND,H_KV_local,D]
        value: torch.Tensor,  # [UND,H_KV_local,D]
        layout: RollingPromptLayout,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # each [UND,H_KV_local,D]
        if layout.sink_tokens == 0:
            return k_rope, k_normalized, value
        if torch.is_grad_enabled():
            raise ValueError("Retained text K/V is inference-only.")
        if self.cached is None:
            initial_views = {doc.view_id for doc in layout.documents if not doc.is_sink and doc.start_frame == 0}
            if initial_views != set(range(layout.num_views)):
                raise ValueError("Segmented prompting must capture its text sink at the start of the rollout.")
            self.cached = tuple(
                tensor.index_select(0, layout.sink_capture_indexes).detach().clone()
                for tensor in (k_rope, k_normalized, value)
            )  # 3*[V*K,H_KV_local,D]
        result: list[torch.Tensor] = []  # each [UND,H_KV_local,D]
        for tensor, saved in zip((k_rope, k_normalized, value), self.cached, strict=True):
            if saved.shape != (layout.num_views * layout.sink_tokens, *tensor.shape[1:]):
                raise ValueError("Rolling text sink camera/head geometry changed.")
            restored = tensor.clone()  # [UND,H_KV_local,D]
            restored.index_copy_(
                0, layout.sink_read_indexes, saved.index_select(0, layout.sink_cache_indexes)
            )  # [UND,H_KV_local,D]
            result.append(restored)
        self.reads += 1
        return result[0], result[1], result[2]
