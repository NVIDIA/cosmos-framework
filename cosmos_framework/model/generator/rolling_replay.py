# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded clean replay K/V with pinned sink chunks and absolute sensor clocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Literal

import torch

from cosmos_framework.model.generator.mot.flex_attention import CaptionMaskItem, SensorMaskItem
from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    _ROLE_CLEAN_TARGET,
    _ROLE_CURRENT_TARGET,
    FlexQueryMetadata,
    TeacherForcingFlexMetadata,
    build_teacher_forcing_multiview_flex_metadata,
)
from cosmos_framework.model.generator.rolling_prompt import RollingPromptLayout, RollingTextSinkLayer
from cosmos_framework.model.generator.rolling_sink_rope import MediaSinkKeyRotation, MediaSinkMemoryState
from cosmos_framework.model.generator.utils.kv_cache import KVToStore, MultiviewARMemoryState


@dataclass
class RollingReplayRequest:
    """One current chunk plus the branch's previously committed clean memory.

    Original condition masks survive a clean refresh, whose embedding inputs
    mark all tokens as clean. The network records the current query metadata so
    the cache can commit exactly the same roles and absolute positions as K/V.
    """

    step: int
    frame_starts: tuple[int, ...]
    frames_per_chunk: int
    pass_kind: Literal["clean", "noisy"]
    condition_masks: list[torch.Tensor]  # each [V*T,1,1]
    history: FlexQueryMetadata | None = None
    current: FlexQueryMetadata | None = None
    prompt_layout: RollingPromptLayout | None = None


def build_rolling_replay_metadata(
    *,
    request: RollingReplayRequest,
    seq_len: int,
    full_q_offsets: torch.Tensor,  # [B+1 or B+2]
    sensor_mask_items: Sequence[Sequence[SensorMaskItem]],
    caption_mask_items: Sequence[Sequence[CaptionMaskItem]] | None,
    device: torch.device,
    num_und: int,
    causal_offsets: torch.Tensor,  # [B+1 or B+2]
    policy: TeacherForcingReplayPolicyConfig,
) -> TeacherForcingFlexMetadata:
    """Retain replay visibility while moving current and cached frames globally."""
    if len(sensor_mask_items) != 1 or len(sensor_mask_items[0]) != len(request.frame_starts):
        raise ValueError("Rolling replay requires one sample and one absolute start per sensor item.")
    if request.step < 0 or request.frames_per_chunk < 1 or any(start < 0 for start in request.frame_starts):
        raise ValueError("Rolling replay requires nonnegative positions and a positive chunk size.")
    if request.pass_kind not in ("clean", "noisy"):
        raise ValueError("Rolling replay pass_kind must be clean or noisy.")
    # Build only the current pack. The legacy noisy builder expects an entire
    # prefix when constructing its suffix; rolling memory already owns that
    # metadata, so assign the current noisy role after this clean-only build.
    current = build_teacher_forcing_multiview_flex_metadata(
        seq_len=seq_len,
        full_q_offsets=full_q_offsets,
        sensor_mask_items=sensor_mask_items,
        caption_mask_items=caption_mask_items,
        device=device,
        num_und=num_und,
        causal_offsets=causal_offsets,
        frames_per_chunk=request.frames_per_chunk,
        pass_kind="clean",
        teacher_forcing_replay_policy=policy,
    )
    frame_id = current.frame_id.clone()  # [UND+GEN]
    timestamp = current.timestamp.clone()  # [UND+GEN]
    causal_step_id = current.causal_step_id.clone()  # [UND+GEN]
    offset = num_und
    for item, start in zip(sensor_mask_items[0], request.frame_starts, strict=True):
        count = item.latent_t * item.spatial_tokens
        frame_id[offset : offset + count] += start  # [S_item]
        timestamp[offset : offset + count] = (
            frame_id[offset : offset + count].to(torch.float32) * item.seconds_per_frame
        )  # [S_item]
        # Do not divide large float32 timestamps to recover a chunk index.
        # The shared clock assigned this complete chunk before token packing.
        causal_step_id[offset : offset + count] = request.step  # [S_item]
        offset += count
    roles = current.token_role_id  # [UND+GEN]
    if request.pass_kind == "noisy":
        roles = torch.where(roles == _ROLE_CLEAN_TARGET, _ROLE_CURRENT_TARGET, roles)  # [UND+GEN]
    current = replace(
        current, frame_id=frame_id, timestamp=timestamp, causal_step_id=causal_step_id, token_role_id=roles
    )
    query = FlexQueryMetadata(
        **{field.name: getattr(current, field.name)[num_und:] for field in fields(FlexQueryMetadata)}  # each [GEN]
    )
    request.current = query
    if request.history is None:
        result = replace(current, query=query)
    else:
        combined = {
            field.name: torch.cat((getattr(current, field.name), getattr(request.history, field.name)))
            for field in fields(FlexQueryMetadata)
        }  # each [UND+GEN+M]
        result = TeacherForcingFlexMetadata(
            **combined, num_und=num_und, query=query, teacher_forcing_replay_policy=policy
        )
    if request.prompt_layout is not None:
        result = replace(result, caption_time_bounds=request.prompt_layout.time_bounds(result.seq_len))
    return result


class _RollingReplayWriter(MediaSinkMemoryState):
    """Read committed history while staging one clean chunk outside the live ring."""

    owner: RollingReplayCache
    step: int
    token_count: int
    staged: MultiviewARMemoryState
    written_layers: set[int]

    def __init__(self, owner: RollingReplayCache, *, step: int, token_count: int, device: torch.device) -> None:
        indexes = torch.arange(token_count, device=device, dtype=torch.long)  # [S_current]
        super().__init__(
            num_layers=len(owner.cache),
            memory_seq_len=owner.capacity,
            cache=owner.cache,
            write_indexes=indexes,
            text_sink_layers=owner.text_sink_layers,
            prompt_layout=owner.prompt_layout,
        )
        self.owner, self.step, self.token_count = owner, step, token_count
        self.sink_key_rotation = owner.sink_key_rotation
        self.staged = MultiviewARMemoryState(
            num_layers=len(owner.cache),
            memory_seq_len=owner.tokens_per_chunk,
            cache=owner._staging_cache,
            write_indexes=indexes,
        )
        self.written_layers = set()

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self.owner._pending is not self:
            raise ValueError("Only the active rolling refresh can stage clean K/V.")
        self.staged.write_for_layer(layer_idx, kv_to_store)
        self.written_layers.add(layer_idx)


class RollingReplayCache:
    """One CFG branch's fixed-capacity ring of complete clean sensor chunks.

    ``cache_chunks`` counts completed history chunks, including ``sink_chunks``;
    the current query pack is separate. Chunk zero is the training singleton.
    Every slot reserves the maximum synchronized chunk token count, so short
    chunks need only mark their unused slots as padding. Cache tensors remain
    allocated and their addresses remain stable after the first clean refresh.
    Refreshes use one reusable staging chunk per layer; publication temporarily
    backs up the overwritten region so a failed commit can restore the ring.
    """

    cache_chunks: int
    sink_chunks: int
    tokens_per_chunk: int
    capacity: int
    cache: list[tuple[torch.Tensor, torch.Tensor] | None]  # per layer, each [1,M,H_kv,D]
    metadata: FlexQueryMetadata | None
    slot_steps: list[int | None]
    next_step: int
    _staging_cache: list[tuple[torch.Tensor, torch.Tensor] | None]  # per layer, each [1,S_slot,H_kv,D]
    _pending: _RollingReplayWriter | None
    text_sink_layers: list[RollingTextSinkLayer] | None
    prompt_layout: RollingPromptLayout | None
    initial_sink_token_count: int
    sink_key_rotation: MediaSinkKeyRotation | None

    def __init__(self, *, num_layers: int, cache_chunks: int, sink_chunks: int, tokens_per_chunk: int) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (num_layers, cache_chunks, sink_chunks, tokens_per_chunk)
        ):
            raise ValueError("Rolling replay cache sizes must be integers.")
        if num_layers < 1 or tokens_per_chunk < 1 or not 0 <= sink_chunks < cache_chunks:
            raise ValueError("Rolling replay requires positive capacity and 0 <= sink_chunks < cache_chunks.")
        self.cache_chunks = cache_chunks
        self.sink_chunks = sink_chunks
        self.tokens_per_chunk = tokens_per_chunk
        self.capacity = cache_chunks * tokens_per_chunk
        self.cache = [None] * num_layers
        self.metadata = None
        self.slot_steps = [None] * cache_chunks
        self.next_step = 0
        self._staging_cache = [None] * num_layers
        self._pending = None
        self.text_sink_layers = None
        self.prompt_layout = None
        self.initial_sink_token_count = 0
        self.sink_key_rotation = None

    def _slot(self, step: int) -> int:
        if step != self.next_step:
            raise ValueError(f"Expected the next rolling chunk {self.next_step}, got {step}.")
        if step < self.sink_chunks:
            return step
        return self.sink_chunks + (step - self.sink_chunks) % (self.cache_chunks - self.sink_chunks)

    def read(self) -> MultiviewARMemoryState:
        """Denoising cannot write K/V, even if the network returns intermediate K/V."""
        state_type = MultiviewARMemoryState if self.sink_key_rotation is None else MediaSinkMemoryState
        state = state_type(
            num_layers=len(self.cache),
            memory_seq_len=self.capacity,
            cache=self.cache,
            text_sink_layers=self.text_sink_layers,
            prompt_layout=self.prompt_layout,
        )
        if isinstance(state, MediaSinkMemoryState):
            state.sink_key_rotation = self.sink_key_rotation
        return state

    def writer(self, *, step: int, token_count: int, device: torch.device) -> MultiviewARMemoryState:
        """Stage real current GEN tokens after each layer has read committed history."""
        self._slot(step)
        if not 0 < token_count <= self.tokens_per_chunk:
            raise ValueError(f"Chunk has {token_count} tokens, exceeding slot capacity {self.tokens_per_chunk}.")
        self._pending = _RollingReplayWriter(self, step=step, token_count=token_count, device=device)
        return self._pending

    def commit(self, *, step: int, token_count: int, request: RollingReplayRequest) -> None:
        """Publish clean K/V and metadata only after every layer finished this refresh."""
        slot = self._slot(step)
        if request.pass_kind != "clean" or request.step != step or request.current is None:
            raise ValueError("Only a completed clean replay can be committed to rolling memory.")
        if not 0 < token_count <= self.tokens_per_chunk or token_count > request.current.sample_id.numel():
            raise ValueError("Rolling replay commit has invalid current token geometry.")
        pending = self._pending
        if pending is None or pending.step != step or pending.token_count != token_count:
            raise ValueError("Rolling replay commit must match the active refresh.")
        if pending.written_layers != set(range(len(self.cache))):
            raise ValueError("Every transformer layer must write clean K/V before committing a chunk.")
        if self.metadata is None:
            metadata = FlexQueryMetadata(
                **{
                    field.name: getattr(request.current, field.name).new_full(
                        (self.capacity,), False if getattr(request.current, field.name).dtype == torch.bool else -1
                    )
                    for field in fields(FlexQueryMetadata)
                }  # each [M]
            )
        else:
            metadata = FlexQueryMetadata(
                **{field.name: getattr(self.metadata, field.name).clone() for field in fields(FlexQueryMetadata)}
            )  # each [M]
        start = slot * self.tokens_per_chunk
        for field in fields(FlexQueryMetadata):
            destination = getattr(metadata, field.name)  # [M]
            destination[start : start + self.tokens_per_chunk].fill_(
                False if destination.dtype == torch.bool else -1
            )  # [S_slot]
            source = getattr(request.current, field.name)[:token_count]  # [S_current]
            destination[start : start + token_count].copy_(source)  # [S_current]
        # Clean history is never a second current-noise stream in a later pass.
        metadata.is_noisy[start : start + token_count] = False  # [S_current]

        # Finish allocations and validation before touching any committed K/V.
        targets: list[tuple[torch.Tensor, torch.Tensor]] = []  # per layer, each [1,M,H_kv,D]
        backups: list[tuple[torch.Tensor, torch.Tensor] | None] = []  # each [1,S_current,H_kv,D]
        for cached, staged in zip(self.cache, pending.staged.cache, strict=True):
            assert staged is not None
            staged_k, staged_v = staged  # each [1,S_slot,H_kv,D]
            if cached is None:
                shape = (staged_k.shape[0], self.capacity, *staged_k.shape[2:])
                targets.append((staged_k.new_zeros(shape), staged_v.new_zeros(shape)))  # each [1,M,H_kv,D]
                backups.append(None)
            else:
                for target, source in zip(cached, staged, strict=True):
                    if (
                        target.shape[:1] + target.shape[2:] != source.shape[:1] + source.shape[2:]
                        or target.dtype != source.dtype
                        or target.device != source.device
                    ):
                        raise ValueError("Rolling clean K/V geometry, dtype and device must remain unchanged.")
                targets.append(cached)
                backups.append(
                    (
                        cached[0][:, start : start + token_count].clone(),
                        cached[1][:, start : start + token_count].clone(),
                    )
                )  # each [1,S_current,H_kv,D]
        try:
            for target, staged in zip(targets, pending.staged.cache, strict=True):
                assert staged is not None
                target[0][:, start : start + token_count].copy_(staged[0][:, :token_count])  # [1,S_current,H_kv,D]
                target[1][:, start : start + token_count].copy_(staged[1][:, :token_count])  # [1,S_current,H_kv,D]
        except Exception:
            for target, backup in zip(targets, backups, strict=True):
                if backup is not None:
                    target[0][:, start : start + token_count].copy_(backup[0])  # [1,S_current,H_kv,D]
                    target[1][:, start : start + token_count].copy_(backup[1])  # [1,S_current,H_kv,D]
            raise
        self.cache[:] = targets
        self.metadata = metadata
        self.slot_steps[slot] = step
        if step == 0 and self.sink_chunks:
            self.initial_sink_token_count = token_count
        self.next_step += 1
        self._pending = None

    @property
    def retained_steps(self) -> tuple[int, ...]:
        """Return retained absolute chunk IDs without recording the entire rollout."""
        return tuple(sorted(step for step in self.slot_steps if step is not None))
