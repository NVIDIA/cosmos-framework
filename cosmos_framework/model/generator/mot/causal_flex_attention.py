# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Interactive-only multiview FlexAttention masks for replayed teacher forcing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Literal, get_args

import torch
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.model.generator.mot.flex_attention import (
    FlexMetadata as BaseFlexMetadata,
)
from cosmos_framework.model.generator.mot.flex_attention import (
    MaskItem,
    build_multiview_flex_metadata,
)
from cosmos_framework.model.generator.mot.flex_attention_utils import (
    build_block_mask_from_metadata_runs,
    metadata_run_groups,
)
from cosmos_framework.configs.base.defaults.replay_attention import (
    TeacherForcingReplayPolicyConfig,
)

MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MultiviewTransferARCurrentRole = Literal["control", "target_condition", "current_target", "clean_target"]
MULTIVIEW_TRANSFER_AR_CURRENT_ROLES = get_args(MultiviewTransferARCurrentRole)

_ROLE_PADDING = -1
_ROLE_UND = 0
_ROLE_CONTROL = 1
_ROLE_TARGET_CONDITION = 2
_ROLE_CURRENT_TARGET = 3
_ROLE_CLEAN_TARGET = 4


@dataclass(frozen=True)
class FlexQueryMetadata:
    """Explicit metadata for the GEN query side of a rectangular mask."""

    sample_id: torch.Tensor  # [Q]
    frame_id: torch.Tensor  # [Q]
    view_id: torch.Tensor  # [Q]
    timestamp: torch.Tensor  # [Q]
    is_noisy: torch.Tensor  # [Q]
    is_control: torch.Tensor  # [Q]
    token_role_id: torch.Tensor  # [Q]
    causal_step_id: torch.Tensor  # [Q]


@dataclass(frozen=True)
class TeacherForcingFlexMetadata:
    """Per-token fields for ``[UND | current GEN | clean memory]`` attention."""

    sample_id: torch.Tensor  # [KV]
    frame_id: torch.Tensor  # [KV]
    view_id: torch.Tensor  # [KV]
    timestamp: torch.Tensor  # [KV]
    is_noisy: torch.Tensor  # [KV]
    is_control: torch.Tensor  # [KV]
    token_role_id: torch.Tensor  # [KV]
    causal_step_id: torch.Tensor  # [KV]
    num_und: int
    query: FlexQueryMetadata
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig

    def __post_init__(self) -> None:
        if not isinstance(self.teacher_forcing_replay_policy, TeacherForcingReplayPolicyConfig):
            raise TypeError("teacher_forcing_replay_policy must be a TeacherForcingReplayPolicyConfig.")
        kv_len = self.sample_id.numel()
        for field_name in (
            "frame_id",
            "view_id",
            "timestamp",
            "is_noisy",
            "is_control",
            "token_role_id",
            "causal_step_id",
        ):
            field_value = getattr(self, field_name)
            if field_value.numel() != kv_len:
                raise ValueError(f"{field_name} has {field_value.numel()} entries, expected {kv_len}.")

    @property
    def seq_len(self) -> int:
        """Return the key/value stream length."""
        return self.sample_id.numel()

    @property
    def q_len(self) -> int:
        """Return the query stream length."""
        return self.query.sample_id.numel()


@dataclass(frozen=True)
class MultiviewTransferARMemoryLayout:
    """Fixed-capacity metadata and prefill indexes for multiview transfer AR memory."""

    sample_id: torch.Tensor  # [M]
    frame_id: torch.Tensor  # [M]
    view_id: torch.Tensor  # [M]
    is_noisy: torch.Tensor  # [M]
    is_control: torch.Tensor  # [M]
    token_role_id: torch.Tensor  # [M]
    causal_step_id: torch.Tensor  # [M]
    prefill_source_token_indexes: torch.Tensor  # [M_prefill]
    prefill_cache_token_indexes: torch.Tensor  # [M_prefill]
    num_views: int
    frames_per_view: int
    spatial_tokens_per_frame: int

    def __post_init__(self) -> None:
        if self.num_views < 1 or self.frames_per_view < 1 or self.spatial_tokens_per_frame < 1:
            raise ValueError(
                "Multiview transfer AR memory geometry must be positive; got "
                f"V={self.num_views}, T={self.frames_per_view}, S={self.spatial_tokens_per_frame}."
            )
        memory_len = self.sample_id.numel()
        for field_name in (
            "frame_id",
            "view_id",
            "is_noisy",
            "is_control",
            "token_role_id",
            "causal_step_id",
        ):
            field_value = getattr(self, field_name)
            if field_value.numel() != memory_len:
                raise ValueError(f"{field_name} has {field_value.numel()} entries, expected {memory_len}.")
        if self.prefill_source_token_indexes.numel() != self.prefill_cache_token_indexes.numel():
            raise ValueError(
                "prefill source and cache indexes must have the same length; got "
                f"{self.prefill_source_token_indexes.numel()} and {self.prefill_cache_token_indexes.numel()}."
            )

    @property
    def seq_len(self) -> int:
        """Return the fixed key/value suffix capacity."""
        return self.sample_id.numel()

    @property
    def item_slot_count(self) -> int:
        """Return the fixed slots reserved for one complete multiview item."""
        return self.num_views * self.frames_per_view * self.spatial_tokens_per_frame

    def _item_cache_token_indexes(self, frame_range: tuple[int, int], *, item_offset: int) -> torch.Tensor:
        """Return fixed cache destinations for one camera-major item frame range."""  # returns [V*C*S]
        frame_start, frame_end = frame_range
        if frame_start < 0 or frame_end <= frame_start or frame_end > self.frames_per_view:
            raise ValueError(
                f"frame_range must be non-empty and within [0, {self.frames_per_view}], got {frame_range}."
            )
        view_offsets = (
            torch.arange(self.num_views, device=self.sample_id.device, dtype=torch.long) * self.frames_per_view
        )[:, None]  # [V,1]
        frame_offsets = torch.arange(
            frame_start,
            frame_end,
            device=self.sample_id.device,
            dtype=torch.long,
        )[None, :]  # [1,C]
        frame_indexes = (view_offsets + frame_offsets).reshape(-1)  # [V*C]
        spatial_offsets = torch.arange(
            self.spatial_tokens_per_frame,
            device=self.sample_id.device,
            dtype=torch.long,
        )  # [S]
        return (
            item_offset + frame_indexes[:, None] * self.spatial_tokens_per_frame + spatial_offsets[None, :]
        ).reshape(-1)  # [V*C*S]

    def control_cache_token_indexes(self, frame_range: tuple[int, int]) -> torch.Tensor:
        """Return destinations for a materialized control frame range."""  # returns [V*C*S]
        return self._item_cache_token_indexes(frame_range, item_offset=0)  # [V*C*S]

    def target_cache_token_indexes(self, frame_range: tuple[int, int]) -> torch.Tensor:
        """Return destinations for a materialized target frame range."""  # returns [V*C*S]
        return self._item_cache_token_indexes(frame_range, item_offset=self.item_slot_count)  # [V*C*S]


@dataclass(frozen=True)
class _StreamFields:
    """One side of the teacher-forcing predicate in its own coordinates."""

    sample_id: torch.Tensor  # [S]
    frame_id: torch.Tensor  # [S]
    view_id: torch.Tensor  # [S]
    timestamp: torch.Tensor  # [S]
    is_noisy: torch.Tensor  # [S]
    is_control: torch.Tensor  # [S]
    token_role_id: torch.Tensor  # [S]
    causal_step_id: torch.Tensor  # [S]


def _key_stream_fields(metadata: TeacherForcingFlexMetadata) -> _StreamFields:
    """Return fields for the full key/value stream."""
    return _StreamFields(
        sample_id=metadata.sample_id,
        frame_id=metadata.frame_id,
        view_id=metadata.view_id,
        timestamp=metadata.timestamp,
        is_noisy=metadata.is_noisy,
        is_control=metadata.is_control,
        token_role_id=metadata.token_role_id,
        causal_step_id=metadata.causal_step_id,
    )


def _query_stream_fields(metadata: TeacherForcingFlexMetadata) -> _StreamFields:
    """Return fields for the explicit current-GEN query stream."""
    query = metadata.query
    return _StreamFields(
        sample_id=query.sample_id,
        frame_id=query.frame_id,
        view_id=query.view_id,
        timestamp=query.timestamp,
        is_noisy=query.is_noisy,
        is_control=query.is_control,
        token_role_id=query.token_role_id,
        causal_step_id=query.causal_step_id,
    )


def _causal_steps_from_frames(frame_id: torch.Tensor, frames_per_chunk: int) -> torch.Tensor:
    """Map frame ids to the ``[1, C, C, ...]`` causal partition."""  # frame_id: [S], returns [S]
    if frames_per_chunk < 1:
        raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}.")
    nonnegative_frame = frame_id.clamp_min(0)  # [S]
    body_step = 1 + torch.div(nonnegative_frame - 1, frames_per_chunk, rounding_mode="floor")  # [S]
    causal_step = torch.where(nonnegative_frame == 0, torch.zeros_like(body_step), body_step)  # [S]
    return torch.where(frame_id >= 0, causal_step, torch.full_like(causal_step, -1))  # [S]


def build_teacher_forcing_clean_target_token_indexes(
    *,
    items_per_sample: Sequence[Sequence[MaskItem]],
    device: torch.device,
) -> torch.Tensor:
    """Return GEN-relative indexes of target tokens cached by the clean pass."""  # returns [N_clean]
    selected_indexes: list[torch.Tensor] = []
    token_offset = 0
    for sample_idx, sample_items in enumerate(items_per_sample):
        target_items = [item for item in sample_items if not item.is_control]
        if len(target_items) != 1:
            raise ValueError(
                f"Teacher forcing requires exactly one target item per sample; sample {sample_idx} has "
                f"{len(target_items)}."
            )
        for item_idx, item in enumerate(sample_items):
            condition_mask = item.condition_mask.to(device=device, dtype=torch.bool).reshape(-1)  # [T]
            generated_tokens = (~condition_mask).repeat_interleave(item.spatial_tokens)  # [N_item]
            if item.is_control and generated_tokens.any():
                raise ValueError(f"Control item {item_idx} of sample {sample_idx} contains unconditioned tokens.")
            if not item.is_control:
                item_positions = torch.arange(item.num_tokens, device=device, dtype=torch.long)  # [N_item]
                selected_indexes.append(item_positions[generated_tokens] + token_offset)  # [N_target]
            token_offset += item.num_tokens
    if not selected_indexes:
        return torch.empty(0, device=device, dtype=torch.long)  # [0]
    return torch.cat(selected_indexes)  # [N_clean]


def _stream_membership(base: BaseFlexMetadata) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positions plus real-UND and real-GEN flags from base metadata."""
    stream_len = base.seq_len
    positions = torch.arange(stream_len, device=base.sample_id.device)  # [S]
    is_real = base.sample_id >= 0  # [S]
    is_und = (positions < base.num_und) & is_real  # [S]
    is_gen = (positions >= base.num_und) & is_real  # [S]
    return positions, is_und, is_gen


def _base_token_roles(
    base: BaseFlexMetadata,
    *,
    pass_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build training role ids plus real-UND and real-GEN flags from base metadata."""
    positions, is_und, is_gen = _stream_membership(base)
    stream_len = base.seq_len
    is_target_condition = is_gen & (~base.is_noisy) & (~base.is_control)  # [S]
    is_target_generated = is_gen & base.is_noisy & (~base.is_control)  # [S]
    token_role_id = torch.full((stream_len,), _ROLE_PADDING, device=positions.device, dtype=torch.long)  # [S]
    token_role_id = torch.where(is_und, torch.full_like(token_role_id, _ROLE_UND), token_role_id)  # [S]
    token_role_id = torch.where(
        is_gen & base.is_control,
        torch.full_like(token_role_id, _ROLE_CONTROL),
        token_role_id,
    )  # [S]
    token_role_id = torch.where(
        is_target_condition,
        torch.full_like(token_role_id, _ROLE_TARGET_CONDITION),
        token_role_id,
    )  # [S]
    target_role = _ROLE_CLEAN_TARGET if pass_kind == "clean" else _ROLE_CURRENT_TARGET
    token_role_id = torch.where(
        is_target_generated,
        torch.full_like(token_role_id, target_role),
        token_role_id,
    )  # [S]
    return token_role_id, is_und, is_gen


def _teacher_forcing_materialized_stream_mask(
    *,
    items_per_sample: Sequence[Sequence[MaskItem]],
    target_frame_ranges: Sequence[tuple[int, int]],
    num_und: int,
    gen_seq_len: int,
    device: torch.device,
) -> torch.Tensor:  # returns [UND+GEN]
    """Mark controls, target conditions, and materialized target ranges as real tokens."""
    target_items = [item for sample_items in items_per_sample for item in sample_items if not item.is_control]
    if len(items_per_sample) != 1 or len(target_items) != 1:
        raise ValueError("Materialized target ranges require one teacher-forcing sample with exactly one target item.")
    materialized_item_tokens: list[torch.Tensor] = []
    for item in items_per_sample[0]:
        if item.is_control:
            materialized_frames = torch.ones(item.latent_t, device=device, dtype=torch.bool)  # [V*T]
        else:
            local_materialized_frames = _materialized_frame_mask(
                target_frame_ranges,
                frames_per_view=item.frames_per_view,
                device=device,
                range_name="materialized_target_frame_ranges",
            )  # [T]
            ranged_frames = local_materialized_frames.repeat(item.num_views)  # [V*T]
            condition_frames = item.condition_mask.to(device=device, dtype=torch.bool).reshape(-1)  # [V*T]
            materialized_frames = ranged_frames | condition_frames  # [V*T]
        materialized_item_tokens.append(materialized_frames.repeat_interleave(item.spatial_tokens))  # [N_item]
    materialized_real_gen = torch.cat(materialized_item_tokens)  # [N_real]
    if materialized_real_gen.numel() > gen_seq_len:
        raise ValueError(
            f"Materialized teacher-forcing metadata has {materialized_real_gen.numel()} real GEN positions, "
            f"exceeding GEN sequence length {gen_seq_len}."
        )
    gen_padding = torch.zeros(gen_seq_len - materialized_real_gen.numel(), device=device, dtype=torch.bool)  # [P]
    materialized_gen = torch.cat((materialized_real_gen, gen_padding))  # [GEN]
    materialized_und = torch.ones(num_und, device=device, dtype=torch.bool)  # [UND]
    return torch.cat((materialized_und, materialized_gen))  # [UND+GEN]


def build_teacher_forcing_multiview_flex_metadata(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    items_per_sample: Sequence[Sequence[MaskItem]],
    device: torch.device,
    num_und: int,
    causal_offsets: torch.Tensor,
    frames_per_chunk: int,
    pass_kind: str,
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig,
    materialized_target_frame_ranges: Sequence[tuple[int, int]] | None = None,
    clean_memory_seq_len: int = 0,
) -> TeacherForcingFlexMetadata:
    """Build metadata for clean replay or noisy replay with a clean K/V suffix.

    ``materialized_target_frame_ranges`` is an AR clean-prefill override. Target
    conditions are always materialized; the ranges add generated clean history,
    while every other target frame becomes padding even if its tensor is present.
    """
    if pass_kind not in ("clean", "noisy"):
        raise ValueError(f"Unknown teacher-forcing pass_kind {pass_kind!r}; expected 'clean' or 'noisy'.")
    if pass_kind == "clean" and clean_memory_seq_len:
        raise ValueError("The clean pass does not consume cached clean target K/V.")
    if materialized_target_frame_ranges is not None and pass_kind != "clean":
        raise ValueError("Materialized target ranges are only supported by the AR clean-prefill pass.")
    base = build_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        items_per_sample=items_per_sample,
        device=device,
        num_und=num_und,
        causal_offsets=causal_offsets,
        attention_scope=teacher_forcing_replay_policy.multiview_attention_scope,
        decomposed_temporal_window_seconds=teacher_forcing_replay_policy.decomposed_temporal_window_seconds,
    )
    token_role_id, _is_und, _is_gen = _base_token_roles(base, pass_kind=pass_kind)
    sample_id = base.sample_id  # [S]
    frame_id = base.frame_id  # [S]
    view_id = base.view_id  # [S]
    timestamp = base.timestamp  # [S]
    is_noisy = base.is_noisy  # [S]
    is_control = base.is_control  # [S]
    if materialized_target_frame_ranges is not None:
        materialized = _teacher_forcing_materialized_stream_mask(
            items_per_sample=items_per_sample,
            target_frame_ranges=materialized_target_frame_ranges,
            num_und=num_und,
            gen_seq_len=seq_len,
            device=device,
        )  # [S]
        sentinel = torch.full_like(sample_id, -1)  # [S]
        timestamp_sentinel = torch.full_like(timestamp, -1.0)  # [S]
        sample_id = torch.where(materialized, sample_id, sentinel)  # [S]
        frame_id = torch.where(materialized, frame_id, sentinel)  # [S]
        view_id = torch.where(materialized, view_id, sentinel)  # [S]
        timestamp = torch.where(materialized, timestamp, timestamp_sentinel)  # [S]
        is_noisy = is_noisy & materialized  # [S]
        is_control = is_control & materialized  # [S]
        token_role_id = torch.where(
            materialized,
            token_role_id,
            torch.full_like(token_role_id, _ROLE_PADDING),
        )  # [S]
    causal_step_id = _causal_steps_from_frames(frame_id, frames_per_chunk)  # [S]
    query = FlexQueryMetadata(
        sample_id=sample_id[num_und:].clone(),  # [Q]
        frame_id=frame_id[num_und:].clone(),  # [Q]
        view_id=view_id[num_und:].clone(),  # [Q]
        timestamp=timestamp[num_und:].clone(),  # [Q]
        is_noisy=is_noisy[num_und:].clone(),  # [Q]
        is_control=is_control[num_und:].clone(),  # [Q]
        token_role_id=token_role_id[num_und:].clone(),  # [Q]
        causal_step_id=causal_step_id[num_und:].clone(),  # [Q]
    )
    if pass_kind == "noisy":
        clean_indexes = build_teacher_forcing_clean_target_token_indexes(
            items_per_sample=items_per_sample,
            device=device,
        )  # [N_clean]
        if clean_indexes.numel() > clean_memory_seq_len:
            raise ValueError(
                f"Clean target metadata needs {clean_indexes.numel()} keys but clean_memory_seq_len is "
                f"{clean_memory_seq_len}."
            )
        clean_source_indexes = clean_indexes + num_und  # [N_clean]
        clean_sample_id = base.sample_id[clean_source_indexes]  # [N_clean]
        clean_frame_id = base.frame_id[clean_source_indexes]  # [N_clean]
        clean_view_id = base.view_id[clean_source_indexes]  # [N_clean]
        clean_timestamp = base.timestamp[clean_source_indexes]  # [N_clean]
        clean_step_id = causal_step_id[clean_source_indexes]  # [N_clean]
        clean_pad = clean_memory_seq_len - clean_indexes.numel()
        if clean_pad:
            sentinel = torch.full((clean_pad,), -1, device=device, dtype=torch.long)  # [P]
            clean_sample_id = torch.cat((clean_sample_id, sentinel))  # [M]
            clean_frame_id = torch.cat((clean_frame_id, sentinel))  # [M]
            clean_view_id = torch.cat((clean_view_id, sentinel))  # [M]
            clean_timestamp = torch.cat(
                (clean_timestamp, torch.full((clean_pad,), -1.0, device=device, dtype=torch.float32))
            )  # [M]
            clean_step_id = torch.cat((clean_step_id, sentinel))  # [M]
        sample_id = torch.cat((sample_id, clean_sample_id))  # [KV]
        frame_id = torch.cat((frame_id, clean_frame_id))  # [KV]
        view_id = torch.cat((view_id, clean_view_id))  # [KV]
        timestamp = torch.cat((timestamp, clean_timestamp))  # [KV]
        causal_step_id = torch.cat((causal_step_id, clean_step_id))  # [KV]
        is_noisy = torch.cat((is_noisy, torch.zeros(clean_memory_seq_len, device=device, dtype=torch.bool)))  # [KV]
        is_control = torch.cat((is_control, torch.zeros(clean_memory_seq_len, device=device, dtype=torch.bool)))  # [KV]
        clean_roles = torch.where(
            clean_sample_id >= 0,
            torch.full_like(clean_sample_id, _ROLE_CLEAN_TARGET),
            torch.full_like(clean_sample_id, _ROLE_PADDING),
        )  # [M]
        token_role_id = torch.cat((token_role_id, clean_roles))  # [KV]

    return TeacherForcingFlexMetadata(
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        timestamp=timestamp,
        is_noisy=is_noisy,
        is_control=is_control,
        token_role_id=token_role_id,
        causal_step_id=causal_step_id,
        num_und=num_und,
        query=query,
        teacher_forcing_replay_policy=teacher_forcing_replay_policy,
    )


def _materialized_frame_mask(
    frame_ranges: Sequence[tuple[int, int]],
    *,
    frames_per_view: int,
    device: torch.device,
    range_name: str,
) -> torch.Tensor:  # returns [T]
    """Turn validated, non-overlapping half-open frame ranges into a mask."""
    materialized = torch.zeros(frames_per_view, device=device, dtype=torch.bool)  # [T]
    for frame_start, frame_end in frame_ranges:
        if frame_start < 0 or frame_end <= frame_start or frame_end > frames_per_view:
            raise ValueError(
                f"{range_name} entries must be non-empty and within [0, {frames_per_view}], "
                f"got {(frame_start, frame_end)}."
            )
        if materialized[frame_start:frame_end].any():
            raise ValueError(
                f"{range_name} entries must not overlap; got overlapping range {(frame_start, frame_end)}."
            )
        materialized[frame_start:frame_end] = True  # [T]
    return materialized  # [T]


def build_multiview_transfer_ar_memory_layout(
    *,
    token_shapes: Sequence[tuple[int, ...]],
    target_condition_mask: torch.Tensor,
    num_views: int,
    frames_per_chunk: int,
    control_frame_ranges: Sequence[tuple[int, int]],
    target_condition_frame_ranges: Sequence[tuple[int, int]],
    history_frame_ranges: Sequence[tuple[int, int]],
    memory_seq_len: int,
    device: torch.device,
) -> MultiviewTransferARMemoryLayout:
    """Describe materialized K/V in fixed full-control and full-target slots."""
    if len(token_shapes) != 2:
        raise ValueError(f"Multiview transfer AR requires exactly two vision items, got {len(token_shapes)}.")
    control_shape, target_shape = token_shapes
    if len(control_shape) != 3 or control_shape != target_shape:
        raise ValueError(f"Control and target token shapes must match, got {token_shapes}.")
    latent_t, patch_h, patch_w = target_shape
    if num_views < 1 or latent_t % num_views != 0:
        raise ValueError(f"latent_t={latent_t} must be divisible by num_views={num_views}.")
    frames_per_view = latent_t // num_views
    spatial_tokens = patch_h * patch_w
    condition_mask = target_condition_mask.to(device=device, dtype=torch.bool).reshape(-1)  # [V*T]
    if condition_mask.numel() != latent_t:
        raise ValueError(f"Target condition mask has {condition_mask.numel()} frames, expected {latent_t}.")
    condition_grid = condition_mask.reshape(num_views, frames_per_view)  # [V,T]
    if not torch.equal(condition_grid, condition_grid[0:1].expand_as(condition_grid)):
        raise ValueError("Multiview transfer AR requires synchronized target conditioning indexes across every view.")
    local_condition_mask = condition_grid[0]  # [T]
    materialized_control_frames = _materialized_frame_mask(
        control_frame_ranges,
        frames_per_view=frames_per_view,
        device=device,
        range_name="control_frame_ranges",
    )  # [T]
    materialized_condition_range = _materialized_frame_mask(
        target_condition_frame_ranges,
        frames_per_view=frames_per_view,
        device=device,
        range_name="target_condition_frame_ranges",
    )  # [T]
    materialized_history_frames = _materialized_frame_mask(
        history_frame_ranges,
        frames_per_view=frames_per_view,
        device=device,
        range_name="history_frame_ranges",
    )  # [T]
    if (local_condition_mask & materialized_history_frames).any():
        raise ValueError("Multiview transfer AR history_frame_ranges must not overlap target condition frames.")
    materialized_condition_frames = local_condition_mask & materialized_condition_range  # [T]

    item_frame_id = (
        torch.arange(frames_per_view, device=device).repeat(num_views).repeat_interleave(spatial_tokens)
    )  # [N_item]
    item_view_id = torch.arange(num_views, device=device).repeat_interleave(
        frames_per_view * spatial_tokens
    )  # [N_item]
    materialized_control_tokens = materialized_control_frames.repeat(num_views).repeat_interleave(
        spatial_tokens
    )  # [N_item]
    materialized_condition_tokens = materialized_condition_frames.repeat(num_views).repeat_interleave(
        spatial_tokens
    )  # [N_item]
    materialized_history_tokens = materialized_history_frames.repeat(num_views).repeat_interleave(
        spatial_tokens
    )  # [N_item]
    materialized_target_tokens = materialized_condition_tokens | materialized_history_tokens  # [N_item]
    materialized_slots = torch.cat((materialized_control_tokens, materialized_target_tokens))  # [N_fixed]
    fixed_frame_id = torch.cat((item_frame_id, item_frame_id))  # [N_fixed]
    fixed_view_id = torch.cat((item_view_id, item_view_id))  # [N_fixed]
    fixed_slot_count = fixed_frame_id.numel()
    if fixed_slot_count > memory_seq_len:
        raise ValueError(
            f"Multiview transfer AR fixed slots need {fixed_slot_count} tokens but capacity is {memory_seq_len}."
        )
    sentinel_slots = torch.full((fixed_slot_count,), -1, device=device, dtype=torch.long)  # [N_fixed]
    sample_id = torch.where(materialized_slots, torch.zeros_like(sentinel_slots), sentinel_slots)  # [N_fixed]
    frame_id = torch.where(materialized_slots, fixed_frame_id, sentinel_slots)  # [N_fixed]
    view_id = torch.where(materialized_slots, fixed_view_id, sentinel_slots)  # [N_fixed]
    control_roles = torch.where(
        materialized_control_tokens,
        torch.full_like(item_frame_id, _ROLE_CONTROL),
        torch.full_like(item_frame_id, _ROLE_PADDING),
    )  # [N_item]
    target_roles = torch.full_like(item_frame_id, _ROLE_PADDING)  # [N_item]
    target_roles = torch.where(
        materialized_condition_tokens,
        torch.full_like(target_roles, _ROLE_TARGET_CONDITION),
        target_roles,
    )  # [N_item]
    target_roles = torch.where(
        materialized_history_tokens,
        torch.full_like(target_roles, _ROLE_CLEAN_TARGET),
        target_roles,
    )  # [N_item]
    token_role_id = torch.cat((control_roles, target_roles))  # [N_fixed]

    memory_pad = memory_seq_len - fixed_slot_count
    if memory_pad:
        memory_sentinel = torch.full((memory_pad,), -1, device=device, dtype=torch.long)  # [P]
        sample_id = torch.cat((sample_id, memory_sentinel))  # [M]
        frame_id = torch.cat((frame_id, memory_sentinel))  # [M]
        view_id = torch.cat((view_id, memory_sentinel))  # [M]
        token_role_id = torch.cat(
            (token_role_id, torch.full((memory_pad,), _ROLE_PADDING, device=device, dtype=torch.long))
        )  # [M]
    causal_step_id = _causal_steps_from_frames(frame_id, frames_per_chunk)  # [M]
    is_noisy = torch.zeros(memory_seq_len, device=device, dtype=torch.bool)  # [M]
    is_control = token_role_id == _ROLE_CONTROL  # [M]
    control_source_token_indexes = torch.nonzero(materialized_control_tokens, as_tuple=False).squeeze(1)  # [N_control]
    condition_source_token_indexes = (
        torch.nonzero(materialized_condition_tokens, as_tuple=False).squeeze(1) + item_frame_id.numel()
    )  # [N_condition]
    prefill_source_token_indexes = torch.cat(
        (control_source_token_indexes, condition_source_token_indexes)
    )  # [M_prefill]
    prefill_cache_token_indexes = prefill_source_token_indexes.clone()  # [M_prefill]
    return MultiviewTransferARMemoryLayout(
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_noisy=is_noisy,
        is_control=is_control,
        token_role_id=token_role_id,
        causal_step_id=causal_step_id,
        prefill_source_token_indexes=prefill_source_token_indexes,
        prefill_cache_token_indexes=prefill_cache_token_indexes,
        num_views=num_views,
        frames_per_view=frames_per_view,
        spatial_tokens_per_frame=spatial_tokens,
    )


def build_multiview_transfer_ar_flex_metadata(
    *,
    seq_len: int,
    full_q_offsets: torch.Tensor,
    items_per_sample: Sequence[Sequence[MaskItem]],
    device: torch.device,
    num_und: int,
    causal_offsets: torch.Tensor,
    current_frame_start: int,
    frames_per_view: int,
    frames_per_chunk: int,
    current_role: MultiviewTransferARCurrentRole,
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig,
    memory_layout: MultiviewTransferARMemoryLayout,
) -> TeacherForcingFlexMetadata:
    """Build replay metadata for one explicitly typed multiview AR chunk."""
    if current_role not in MULTIVIEW_TRANSFER_AR_CURRENT_ROLES:
        raise ValueError(
            f"Unknown multiview transfer AR current_role {current_role!r}; "
            f"expected one of {MULTIVIEW_TRANSFER_AR_CURRENT_ROLES}."
        )
    if len(items_per_sample) != 1 or len(items_per_sample[0]) != 1:
        raise ValueError("Multiview transfer AR current packs require one sample with one vision item.")
    current_item = items_per_sample[0][0]
    if current_item.num_views != memory_layout.num_views:
        raise ValueError(
            f"Current multiview transfer AR pack has {current_item.num_views} views, "
            f"but its memory layout has {memory_layout.num_views}."
        )
    if frames_per_view != memory_layout.frames_per_view:
        raise ValueError(
            f"Current multiview transfer AR pack declares {frames_per_view} frames per view, "
            f"but its memory layout has {memory_layout.frames_per_view}."
        )
    if current_item.spatial_tokens != memory_layout.spatial_tokens_per_frame:
        raise ValueError(
            f"Current multiview transfer AR pack has {current_item.spatial_tokens} spatial tokens per frame, "
            f"but its memory layout has {memory_layout.spatial_tokens_per_frame}."
        )
    current_chunk_len = current_item.frames_per_view
    if current_frame_start < 0 or current_frame_start + current_chunk_len > frames_per_view:
        raise ValueError(
            f"Current multiview transfer AR chunk "
            f"[{current_frame_start}, {current_frame_start + current_chunk_len}) "
            f"is outside {frames_per_view} frames per view."
        )
    base = build_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        items_per_sample=items_per_sample,
        device=device,
        num_und=num_und,
        causal_offsets=causal_offsets,
        attention_scope=teacher_forcing_replay_policy.multiview_attention_scope,
        decomposed_temporal_window_seconds=teacher_forcing_replay_policy.decomposed_temporal_window_seconds,
    )
    positions, is_und, is_gen = _stream_membership(base)
    current_role_id = {
        "control": _ROLE_CONTROL,
        "target_condition": _ROLE_TARGET_CONDITION,
        "current_target": _ROLE_CURRENT_TARGET,
        "clean_target": _ROLE_CLEAN_TARGET,
    }[current_role]
    token_role_id = torch.full((base.seq_len,), _ROLE_PADDING, device=device, dtype=torch.long)  # [S]
    token_role_id = torch.where(is_und, torch.full_like(token_role_id, _ROLE_UND), token_role_id)  # [S]
    token_role_id = torch.where(
        is_gen,
        torch.full_like(token_role_id, current_role_id),
        token_role_id,
    )  # [S]
    current_is_noisy = torch.full_like(base.is_noisy, current_role == "current_target") & is_gen  # [S]
    current_is_control = torch.full_like(base.is_control, current_role == "control") & is_gen  # [S]
    global_frame_id = torch.where(is_gen, base.frame_id + current_frame_start, base.frame_id)  # [S]
    global_timestamp = torch.where(
        is_gen,
        base.timestamp + current_frame_start * current_item.seconds_per_frame,
        base.timestamp,
    )  # [S]
    memory_timestamp = torch.where(
        memory_layout.frame_id >= 0,
        memory_layout.frame_id.to(dtype=base.timestamp.dtype) * current_item.seconds_per_frame,
        torch.full_like(memory_layout.frame_id, -1.0, dtype=base.timestamp.dtype),
    )  # [M]
    causal_step_id = _causal_steps_from_frames(global_frame_id, frames_per_chunk)  # [S]
    query = FlexQueryMetadata(
        sample_id=base.sample_id[num_und:].clone(),  # [Q]
        frame_id=global_frame_id[num_und:].clone(),  # [Q]
        view_id=base.view_id[num_und:].clone(),  # [Q]
        timestamp=global_timestamp[num_und:].clone(),  # [Q]
        is_noisy=current_is_noisy[num_und:].clone(),  # [Q]
        is_control=current_is_control[num_und:].clone(),  # [Q]
        token_role_id=token_role_id[num_und:].clone(),  # [Q]
        causal_step_id=causal_step_id[num_und:].clone(),  # [Q]
    )
    del positions, is_und
    return TeacherForcingFlexMetadata(
        sample_id=torch.cat((base.sample_id, memory_layout.sample_id)),  # [KV]
        frame_id=torch.cat((global_frame_id, memory_layout.frame_id)),  # [KV]
        view_id=torch.cat((base.view_id, memory_layout.view_id)),  # [KV]
        timestamp=torch.cat((global_timestamp, memory_timestamp)),  # [KV]
        is_noisy=torch.cat((current_is_noisy, memory_layout.is_noisy)),  # [KV]
        is_control=torch.cat((current_is_control, memory_layout.is_control)),  # [KV]
        token_role_id=torch.cat((token_role_id, memory_layout.token_role_id)),  # [KV]
        causal_step_id=torch.cat((causal_step_id, memory_layout.causal_step_id)),  # [KV]
        num_und=num_und,
        query=query,
        teacher_forcing_replay_policy=teacher_forcing_replay_policy,
    )


def _teacher_forcing_pair_predicate(
    q_fields: _StreamFields,
    kv_fields: _StreamFields,
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig,
) -> MaskMod:
    """Return replayed teacher-forcing visibility in query/key coordinates."""
    device = q_fields.view_id.device
    control_is_global = torch.tensor(teacher_forcing_replay_policy.control_visibility == "global", device=device)  # []
    control_is_causal = torch.tensor(teacher_forcing_replay_policy.control_visibility == "causal", device=device)  # []
    control_is_current = torch.tensor(
        teacher_forcing_replay_policy.control_visibility == "current", device=device
    )  # []
    controls_read_clean_history = torch.tensor(
        teacher_forcing_replay_policy.controls_read_strict_past_clean_rgb,
        device=device,
    )  # []
    clean_pass_is_frame_causal = torch.tensor(
        teacher_forcing_replay_policy.clean_pass_causality == "frame",
        device=device,
    )  # []
    attention_scope = teacher_forcing_replay_policy.multiview_attention_scope
    decomposed_temporal_window_seconds = teacher_forcing_replay_policy.decomposed_temporal_window_seconds
    reaches_every_view = torch.tensor(attention_scope == "all_views", device=device)  # []
    is_decomposed = torch.tensor(attention_scope == "decomposed", device=device)  # []
    has_temporal_window = torch.tensor(decomposed_temporal_window_seconds is not None, device=device)  # []
    temporal_window = torch.tensor(
        0.0 if decomposed_temporal_window_seconds is None else decomposed_temporal_window_seconds,
        dtype=torch.float32,
        device=device,
    )  # []
    temporal_window_eps = torch.tensor(1e-4, dtype=torch.float32, device=device)  # []

    def pair_allowed(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:  # q_idx: [Q,1], kv_idx: [1,KV], returns [Q,KV]
        del b, h
        same_sample = q_fields.sample_id[q_idx] == kv_fields.sample_id[kv_idx]  # [Q,KV]
        q_frame = q_fields.frame_id[q_idx]  # [Q,1]
        kv_frame = kv_fields.frame_id[kv_idx]  # [1,KV]
        same_frame = q_frame == kv_frame  # [Q,KV]
        same_view = q_fields.view_id[q_idx] == kv_fields.view_id[kv_idx]  # [Q,KV]
        timestamp_gap = q_fields.timestamp[q_idx] - kv_fields.timestamp[kv_idx]  # [Q,KV]
        within_temporal_window = (timestamp_gap >= -temporal_window_eps) & (
            timestamp_gap <= temporal_window + temporal_window_eps
        )  # [Q,KV]
        reaches_own_instant = is_decomposed & torch.where(
            has_temporal_window, within_temporal_window, same_frame
        )  # [Q,KV]
        in_scope = reaches_every_view | same_view | reaches_own_instant  # [Q,KV]
        q_step = q_fields.causal_step_id[q_idx]  # [Q,1]
        kv_step = kv_fields.causal_step_id[kv_idx]  # [1,KV]
        q_role = q_fields.token_role_id[q_idx]  # [Q,1]
        kv_role = kv_fields.token_role_id[kv_idx]  # [1,KV]
        q_is_target = (q_role == _ROLE_CURRENT_TARGET) | (q_role == _ROLE_CLEAN_TARGET)  # [Q,1]
        q_is_current = q_role == _ROLE_CURRENT_TARGET  # [Q,1]
        q_is_control = q_role == _ROLE_CONTROL  # [Q,1]
        q_is_target_condition = q_role == _ROLE_TARGET_CONDITION  # [Q,1]
        q_is_condition = q_is_control | q_is_target_condition  # [Q,1]
        kv_is_noisy_rgb = kv_fields.is_noisy[kv_idx] & (~kv_fields.is_control[kv_idx])  # [1,KV]
        kv_is_clean_rgb = (kv_role == _ROLE_TARGET_CONDITION) | (kv_role == _ROLE_CLEAN_TARGET)  # [1,KV]
        control_step_allowed = (
            control_is_global | (control_is_causal & (kv_step <= q_step)) | (control_is_current & (kv_step == q_step))
        )  # [Q,KV]
        target_to_und = q_is_target & (kv_role == _ROLE_UND)  # [Q,KV]
        condition_to_und = q_is_condition & (kv_role == _ROLE_UND)  # [Q,KV]
        control_to_control = q_is_control & (kv_role == _ROLE_CONTROL) & same_view & control_step_allowed  # [Q,KV]
        control_to_clean_rgb_history = (
            q_is_control & controls_read_clean_history & kv_is_clean_rgb & (kv_step < q_step) & in_scope
        )  # [Q,KV]
        target_condition_to_itself = (
            q_is_target_condition & (kv_role == _ROLE_TARGET_CONDITION) & same_frame & same_view
        )  # [Q,KV]
        target_condition_to_control = (
            q_is_target_condition & (kv_role == _ROLE_CONTROL) & same_view & control_step_allowed
        )  # [Q,KV]
        # Target-condition K/V is reused during AR. It must not encode a noisy target
        # from the same or a future replay chunk because neither is available when
        # the condition state is prefetched at inference.
        condition_to_noisy_rgb = q_is_target_condition & kv_is_noisy_rgb & (kv_step < q_step) & in_scope  # [Q,KV]
        target_to_current = q_is_current & (kv_role == _ROLE_CURRENT_TARGET) & (kv_step == q_step) & in_scope  # [Q,KV]
        clean_pass_causal = torch.where(
            clean_pass_is_frame_causal,
            kv_frame <= q_frame,
            kv_step <= q_step,
        )  # [Q,KV]
        clean_step_allowed = (q_is_current & (kv_step < q_step)) | ((~q_is_current) & clean_pass_causal)  # [Q,KV]
        target_to_clean = q_is_target & (kv_role == _ROLE_CLEAN_TARGET) & clean_step_allowed & in_scope  # [Q,KV]
        target_to_control = q_is_target & (kv_role == _ROLE_CONTROL) & same_view & control_step_allowed  # [Q,KV]
        target_to_condition = (
            q_is_target & (kv_role == _ROLE_TARGET_CONDITION) & same_view & (kv_step <= q_step)
        )  # [Q,KV]
        padding_to_padding = (q_role == _ROLE_PADDING) & (kv_role == _ROLE_PADDING)  # [Q,KV]
        allowed = (
            target_to_und
            | condition_to_und
            | control_to_control
            | control_to_clean_rgb_history
            | target_condition_to_itself
            | target_condition_to_control
            | condition_to_noisy_rgb
            | target_to_current
            | target_to_clean
            | target_to_control
            | target_to_condition
        )  # [Q,KV]
        return (same_sample & allowed) | padding_to_padding  # [Q,KV]

    return pair_allowed


def _stream_metadata_groups(fields_: _StreamFields, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a stream into runs that agree on every predicate field."""
    return metadata_run_groups(
        tuple(getattr(fields_, field_.name) for field_ in fields(fields_) if field_.name != "timestamp"),
        device=device,
    )


def build_teacher_forcing_block_mask(
    metadata: TeacherForcingFlexMetadata,
    device: torch.device,
    block_size: tuple[int, int],
) -> BlockMask:
    """Build a memory-augmented rectangular block mask without a dense token mask."""
    q_block_size, kv_block_size = block_size
    if metadata.q_len % q_block_size != 0:
        raise ValueError(f"Query length {metadata.q_len} is not aligned to block size {q_block_size}.")
    if metadata.num_und % kv_block_size != 0 or metadata.seq_len % kv_block_size != 0:
        raise ValueError(f"UND/key lengths {(metadata.num_und, metadata.seq_len)} are not aligned to {kv_block_size}.")
    query_fields = _query_stream_fields(metadata)
    key_fields = _key_stream_fields(metadata)
    pair_allowed = _teacher_forcing_pair_predicate(
        query_fields,
        key_fields,
        metadata.teacher_forcing_replay_policy,
    )
    query_group_id, query_representatives = _stream_metadata_groups(query_fields, device)  # [Q], [GQ]
    key_group_id, key_representatives = _stream_metadata_groups(key_fields, device)  # [KV], [GKV]
    mask_mod = _teacher_forcing_pair_predicate(
        query_fields,
        key_fields,
        metadata.teacher_forcing_replay_policy,
    )
    return build_block_mask_from_metadata_runs(
        q_group_id=query_group_id,
        kv_group_id=key_group_id,
        q_representatives=query_representatives,
        kv_representatives=key_representatives,
        pair_allowed=pair_allowed,
        mask_mod=mask_mod,
        q_len=metadata.q_len,
        kv_len=metadata.seq_len,
        device=device,
        block_size=block_size,
    )
