# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal

import torch

from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    _ROLE_CLEAN_TARGET,
    _ROLE_CURRENT_TARGET,
    FlexQueryMetadata,
    TeacherForcingFlexMetadata,
)
from cosmos_framework.model.generator.rolling_prompt import RollingPromptSchedule
from cosmos_framework.model.generator.teacher_forcing import mark_modality_as_clean_condition
from cosmos_framework.model.generator.utils.kv_cache import JointChunkMemory, TeacherForcingMemoryState

if TYPE_CHECKING:
    from cosmos_framework.model.generator.mot.flex_attention import SensorMaskItem
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel


class JointARCache:
    """Handle persistent-K/V layout and attention metadata for a joint AR pipeline.

    This coordinator maps current RGB/LiDAR chunks to cache slots, restores their
    original position IDs, and builds token labels for maskless attention.
    JointChunkMemory owns the actual per-layer K/V buffers.

    Each layer's K and V buffers have shape [1,S,H_kv/CP,D], where S is the allocated
    RGB+LiDAR token capacity, including unwritten slots.
    Storage along S and its matching metadata are ordered as follows::

        K/V slots:      [ RGB control | RGB target | LiDAR control | LiDAR target ]
        Metadata slots: [ RGB control | RGB target | LiDAR control | LiDAR target ]
                              same slot index identifies the same token

        Within each RGB item:   [view 0: frames_0,frame_1..][view 1: ...]...
        Within each LiDAR item: [frames 0,1...]

    Each frame contains contiguous spatial tokens.
    Unwritten slots have padding roles and are excluded from attention groups.
    """

    reference: PackedSequence
    memory: JointChunkMemory
    metadata: dict[str, torch.Tensor] | None
    current_metadata: dict[str, torch.Tensor] | None
    item_layouts: list[tuple[int, int, int, int]]
    rgb_frames: tuple[int, ...]
    lidar_frames: tuple[int, ...]
    spans: list[tuple[int, int, int]]

    def __init__(self, reference: PackedSequence, num_layers: int) -> None:
        self.reference = reference
        assert reference.vision is not None and reference.lidar is not None
        views = reference.num_views_per_vision_item
        assert views is not None
        if len(reference.vision.token_shapes) != 2 or len(reference.lidar.token_shapes) != 2 or len(views) != 2:
            raise ValueError("Joint AR cache requires one control and one target item for each of RGB and LiDAR.")
        rgb_control_shape, rgb_target_shape = reference.vision.token_shapes
        lidar_control_shape, lidar_target_shape = reference.lidar.token_shapes
        # Cache order is RGB control, RGB target, LiDAR control, LiDAR target.
        # RGB items have per-item camera counts; LiDAR has no camera-view axis.
        items = (
            (rgb_control_shape, views[0]),
            (rgb_target_shape, views[1]),
            (lidar_control_shape, 1),
            (lidar_target_shape, 1),
        )
        self.item_layouts = []
        capacity = 0
        for (total_frames, height, width), num_views in items:
            slots = total_frames // num_views
            self.item_layouts.append((capacity, num_views, slots, height * width))
            capacity += num_views * slots * height * width
        self.memory = JointChunkMemory(num_layers, capacity)
        self.metadata = None
        self.current_metadata = None
        self.rgb_frames = ()
        self.lidar_frames = ()
        self.spans = []

    def prepare(self, pack: PackedSequence, rgb_frames: tuple[int, ...], lidar_frames: tuple[int, ...]) -> None:
        """Prepare current-chunk cache writes, positional encoding, and attention visibility.

        1. Compute spans mapping current-chunk K/V into persistent history for
           the branch's clean forward.
        2. Restore current tokens' original position IDs from the full reference
           pack before RoPE runs. Packing only a chunk can give its tokens local
           positions. Restoring the original IDs keeps current Q/K in the same
           positional coordinate system as the cached historical keys.
        3. Record original RGB/LiDAR frame IDs and attach this coordinator to the
           pack. The network hook uses those IDs and stored history labels to
           build attention visibility for current tokens and cached history in
           both noisy denoising and clean forwards.

        Position IDs preserve positional encoding; metadata preserves attention
        visibility. This method does not write K/V or enable cache writes.
        """
        self.rgb_frames, self.lidar_frames = rgb_frames, lidar_frames
        self.spans = []
        source = 0
        for modality_name, frame_ids in (("vision", rgb_frames), ("lidar", lidar_frames)):
            modality = getattr(pack, modality_name)
            original = getattr(self.reference, modality_name)
            assert modality is not None and original is not None
            current_offset = original_offset = 0
            for local_item, (total_frames, height, width) in enumerate(original.token_shapes):
                item = local_item + (0 if modality_name == "vision" else 2)
                base, views, slots, spatial = self.item_layouts[item]
                original_frames = total_frames // views
                selected = [view * original_frames + frame for view in range(views) for frame in frame_ids]
                frame_tensor = torch.tensor(
                    selected, dtype=torch.long, device=original.sequence_indexes.device
                )  # [V*C]
                original_indexes = original.sequence_indexes[
                    original_offset : original_offset + total_frames * spatial
                ]  # [V*T*S]
                selected_indexes = original_indexes.reshape(total_frames, spatial)[frame_tensor].flatten()  # [V*C*S]
                length = len(selected) * spatial
                current_indexes = modality.sequence_indexes[current_offset : current_offset + length]  # [V*C*S]
                pack.position_ids[:, current_indexes] = self.reference.position_ids[:, selected_indexes].to(
                    pack.position_ids.device
                )  # [3,V*C*S]
                for view in range(views):
                    for frame in frame_ids:
                        destination = base + (view * slots + frame) * spatial
                        if (
                            self.spans
                            and self.spans[-1][0] + self.spans[-1][2] == source
                            and self.spans[-1][1] + self.spans[-1][2] == destination
                        ):
                            previous = self.spans[-1]
                            self.spans[-1] = (previous[0], previous[1], previous[2] + spatial)
                        else:
                            self.spans.append((source, destination, spatial))
                        source += spatial
                current_offset += length
                original_offset += total_frames * spatial
        setattr(pack, "joint_ar_cache", self)

    def augment_metadata(
        self,
        metadata: TeacherForcingFlexMetadata,
        sensor_items: list[list[SensorMaskItem]],
        chunk_size: int,
        *,
        noisy: bool,
    ) -> TeacherForcingFlexMetadata:
        """Build attention-visibility metadata for current tokens and persistent history.

        The returned metadata describes current sensor queries and keys ordered as follows::

            Query labels: [current sensor chunk]
            Key labels:   [text | current sensor chunk | persistent history]

        The network uses this metadata to build the maskless attention plan.
        """
        names = [field.name for field in fields(FlexQueryMetadata)]
        current = {name: getattr(metadata.query, name).clone() for name in names}  # each [Q]
        if noisy:
            current["token_role_id"] = torch.where(
                current["token_role_id"] == _ROLE_CLEAN_TARGET,
                _ROLE_CURRENT_TARGET,
                current["token_role_id"],
            )  # [Q]
        offset = 0
        for item_index, item in enumerate(sensor_items[0]):
            frame_ids = self.rgb_frames if item_index < 2 else self.lidar_frames
            spatial = item.token_shape[1] * item.token_shape[2]
            logical = (
                torch.tensor(frame_ids, dtype=torch.long, device=current["frame_id"].device)
                .repeat(item.num_views)
                .repeat_interleave(spatial)
            )  # [V*C*S]
            length = logical.numel()
            current["frame_id"][offset : offset + length] = logical  # [V*C*S]
            current["timestamp"][offset : offset + length] = logical * item.seconds_per_frame  # [V*C*S]
            current["causal_step_id"][offset : offset + length] = torch.ceil(
                logical.double() * item.seconds_per_frame / (chunk_size * sensor_items[0][0].seconds_per_frame) - 1e-5
            ).long()  # [V*C*S]
            offset += length
        self.current_metadata = current
        values = {}
        for name in names:
            parts = [getattr(metadata, name)[: metadata.num_und], current[name]]  # [UND], [Q]
            if self.metadata is not None:
                parts.append(self.metadata[name])  # [M]
            values[name] = torch.cat(parts)  # [UND+Q+M]
        return replace(metadata, **values, query=FlexQueryMetadata(**current))

    def commit_metadata(self) -> None:
        """Publish occupied slots only after every layer has written its clean K/V."""
        assert self.current_metadata is not None
        # Stored targets are clean memory, never noisy keys in later chunks.
        self.current_metadata["is_noisy"] = torch.zeros_like(self.current_metadata["is_noisy"])  # [Q]
        if self.metadata is None:
            self.metadata = {
                name: torch.full(
                    (self.memory.memory_seq_len,),
                    False if value.dtype == torch.bool else -1,
                    device=value.device,
                    dtype=value.dtype,
                )
                for name, value in self.current_metadata.items()
            }  # each [M]
        for name, value in self.current_metadata.items():  # value: [Q]
            for source, destination, length in self.spans:
                self.metadata[name][destination : destination + length] = value[source : source + length]  # [L]
        self.memory.spans = None


@dataclass(frozen=True)
class JointARChunk:
    """A complete training-time causal chunk on the shared RGB/LiDAR clock."""

    step: int
    vision_frames: tuple[int, ...]
    lidar_frames: tuple[int, ...]
    vision_prefix_end: int
    lidar_prefix_end: int


def joint_ar_chunks(
    *,
    vision_frames: int,
    lidar_frames: int,
    vision_seconds_per_frame: float,
    lidar_seconds_per_frame: float,
    frames_per_chunk: int,
) -> list[JointARChunk]:
    """Match the teacher-forcing singleton-zero and shared-time chunk boundaries."""
    if min(vision_frames, lidar_frames, frames_per_chunk) < 1:
        raise ValueError("Joint AR requires positive frame counts and chunk size")
    if not all(math.isfinite(value) and value > 0 for value in (vision_seconds_per_frame, lidar_seconds_per_frame)):
        raise ValueError("Joint AR requires finite positive sensor periods")
    chunk_seconds = frames_per_chunk * vision_seconds_per_frame
    vision_steps = [math.ceil(frame / frames_per_chunk - 1e-5) for frame in range(vision_frames)]
    lidar_steps = [math.ceil(frame * lidar_seconds_per_frame / chunk_seconds - 1e-5) for frame in range(lidar_frames)]
    chunks: list[JointARChunk] = []
    for step in sorted(set(vision_steps) | set(lidar_steps)):
        chunks.append(
            JointARChunk(
                step=step,
                vision_frames=tuple(frame for frame, value in enumerate(vision_steps) if value == step),
                lidar_frames=tuple(frame for frame, value in enumerate(lidar_steps) if value == step),
                vision_prefix_end=sum(value <= step for value in vision_steps),
                lidar_prefix_end=sum(value <= step for value in lidar_steps),
            )
        )
    return chunks


def _camera_prefix(
    latent: torch.Tensor,  # [1,C,V*T,H,W]
    num_views: int,
    end: int,
) -> torch.Tensor:  # [1,C,V*end,H,W]
    batch, channels, total_frames, height, width = latent.shape
    grid = latent.reshape(batch, channels, num_views, total_frames // num_views, height, width)  # [1,C,V,T,H,W]
    return grid[:, :, :, :end].reshape(batch, channels, num_views * end, height, width)  # [1,C,V*end,H,W]


def _conditioned_target(
    latent: torch.Tensor,  # [1,C,T,H,W]
    mask: torch.Tensor,  # [T,1,1]
) -> torch.Tensor:  # [1,C,T,H,W]
    """Discard every unconditioned target value before any model forward."""
    condition = mask.to(device=latent.device, dtype=torch.bool).reshape(1, 1, -1, 1, 1)  # [1,1,T,1,1]
    return torch.where(condition, latent, torch.zeros_like(latent))  # [1,C,T,H,W]


def _prefix_condition_count(mask: torch.Tensor, num_views: int) -> int:  # mask: [V*T,1,1]
    """Reject unsynchronized or non-prefix conditioning in this joint sampler."""
    grid = mask.to(dtype=torch.bool).reshape(num_views, -1)  # [V,T]
    if not torch.equal(grid, grid[:1].expand_as(grid)):
        raise ValueError("Joint AR currently requires the same conditioned prefix for every RGB view")
    count = int(grid[0].sum().item())
    expected = torch.arange(grid.shape[1], device=grid.device) < count  # [T]
    if not torch.equal(grid[0], expected):
        raise ValueError("Joint AR requires a contiguous conditioned target prefix")
    return count


def _validate_persistent_chunks(chunks: list[JointARChunk], vision_condition: int, lidar_condition: int) -> None:
    """Check whether persistent K/V can replace replay for the complete chunk schedule.

    Persistent K/V reuses stored clean history instead of rebuilding it through replay,
    but cannot currently handle:

    1. A causal chunk missing RGB or LiDAR frames; both sensors must be present.
    2. User-provided RGB/LiDAR target conditioning extending beyond the first
       causal chunk. Later conditioned queries may read earlier noisy target
       representations in replay; substituting stored clean K/V is not equivalent.

    For either case, use the replay K/V mechanism by setting joint_ar_use_persistent_kv=False.
    This check raises rather than silently falling back to replay.
    """
    if any(not chunk.vision_frames or not chunk.lidar_frames for chunk in chunks):
        raise ValueError(
            "Persistent joint K/V currently requires both sensors (RGB and LiDAR) in every causal chunk. "
            "Use the replay K/V mechanism: set joint_ar_use_persistent_kv=False."
        )
    if vision_condition > chunks[0].vision_prefix_end or lidar_condition > chunks[0].lidar_prefix_end:
        raise ValueError(
            "Persistent joint K/V currently supports target conditioning only in the first causal chunk; "
            "later conditioning can read noisy replay history. Use the replay K/V mechanism: "
            "set joint_ar_use_persistent_kv=False."
        )


def _prefix_data(
    data: GenerationDataClean,
    *,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    """Keep complete sensor chunks, with no future target or control positions."""
    assert data.x0_tokens_vision is not None and data.x0_tokens_lidar is not None
    if data.temporal_positions_vision is not None:
        raise ValueError("Joint AR currently requires the training latent-index temporal positions")
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            _camera_prefix(data.x0_tokens_vision[0], num_views, chunk.vision_prefix_end),  # [1,Cv,V*Tv_prefix,Hv,Wv]
            _camera_prefix(vision_target, num_views, chunk.vision_prefix_end),  # [1,Cv,V*Tv_prefix,Hv,Wv]
        ],
        x0_tokens_lidar=[
            data.x0_tokens_lidar[0][:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
            lidar_target[:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
        ],
    )


def _build_replay_branch(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    text_tokens: list[list[int]],
) -> tuple[PackedSequence, TeacherForcingMemoryState]:
    """Use the actual joint training clean/noisy passes, including caption isolation."""
    pack = host._pack_input_sequence(plans, text_tokens, data, torch.zeros(1, dtype=torch.float32))  # timestep: [1]
    pack.to_cuda()
    host._cast_generated_tokens_to_precision(pack)
    host._validate_teacher_forcing_pack(pack)
    # Replay recomputes clean K/V for the complete causal prefix.
    memory = host._build_clean_tf_cache(
        net=host.net,
        packed_sequence=pack,
        gen_data_clean=data,
        memory_info={},
        detach_clean_kv=True,
    )
    return pack, memory


def _current_chunk_data(
    data: GenerationDataClean,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    """Materialize only this shared-clock chunk, keeping control/target pairing."""
    assert data.x0_tokens_vision is not None and data.x0_tokens_lidar is not None
    frames = vision_target.shape[2] // num_views
    if data.temporal_positions_vision is not None:
        raise ValueError("Joint AR currently requires the training latent-index temporal positions")
    indexes = [view * frames + frame for view in range(num_views) for frame in chunk.vision_frames]
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            data.x0_tokens_vision[0][:, :, indexes],
            vision_target[:, :, indexes],
        ],  # each [1,Cv,V*C_rgb,Hv,Wv]
        x0_tokens_lidar=[
            data.x0_tokens_lidar[0][:, :, list(chunk.lidar_frames)],
            lidar_target[:, :, list(chunk.lidar_frames)],
        ],  # each [1,Cl,C_lidar,Hl,Wl]
    )


def _build_persistent_branch(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    text: list[list[int]],
    cache: JointARCache,
    chunk: JointARChunk,
    *,
    clean: bool = False,
) -> tuple[PackedSequence, JointChunkMemory]:
    """Prepare current-chunk inputs for attention against persistent history K/V."""
    local_plans = [
        replace(
            plans[0],
            condition_frame_indexes_vision=[
                i for i, frame in enumerate(chunk.vision_frames) if frame in plans[0].condition_frame_indexes_vision
            ],
            condition_frame_indexes_lidar=[
                i for i, frame in enumerate(chunk.lidar_frames) if frame in plans[0].condition_frame_indexes_lidar
            ],
        )
    ]
    pack = host._pack_input_sequence(local_plans, text, data, torch.zeros(1))  # timestep: [1]
    host._validate_teacher_forcing_pack(pack)
    assert pack.vision is not None and pack.lidar is not None
    original_masks = [
        mask.clone() for modality in (pack.vision, pack.lidar) for mask in modality.condition_mask
    ]  # list of [V*C,1,1]
    cache.prepare(pack, chunk.vision_frames, chunk.lidar_frames)
    if clean:
        for modality in (pack.vision, pack.lidar):
            mark_modality_as_clean_condition(modality)
    pack.teacher_forcing_pass = "clean" if clean else "noisy"
    pack.teacher_forcing_original_condition_masks_sensors = original_masks
    pack.to_cuda()
    host._cast_generated_tokens_to_precision(pack)
    return pack, cache.memory


def _commit_persistent_joint_chunk(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    branches: list[tuple[list[list[int]], JointARCache]],
    chunk: JointARChunk,
) -> None:
    """Store a finalized chunk's clean K/V and attention metadata in persistent history.

    Prior persistent history metadata stays unchanged.
    """
    for text, cache in branches:
        pack, memory = _build_persistent_branch(host, plans, data, text, cache, chunk, clean=True)
        cache.memory.spans = cache.spans
        try:
            host.denoise(data_batch_packed=pack, memory=memory)
            cache.commit_metadata()
        finally:
            cache.memory.spans = None


def sample_joint_transfer_ar(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    sampler_mode: Literal["rf", "distilled"] | None = None,
    distilled_num_steps: int | None = None,
    prompt_schedule: RollingPromptSchedule | None = None,
    joint_ar_use_persistent_kv: bool = False,
) -> dict[str, list[torch.Tensor]]:  # returns vision: [1,Cv,V*Tv,Hv,Wv], lidar: [1,Cl,Tl,Hl,Wl]
    """Joint AR sampling with reference replay or opt-in persistent clean history.

    Both targets are denoised together. Each clean replay contains only conditions
    and previously generated values; ungenerated targets are zero placeholders.
    The existing noisy-pass mask reads only strictly earlier clean chunks, and its
    current-target edges stay inside the current chunk. A complete prefix is used
    so controls can recompute their training-time history-dependent K/V. This is
    deliberately a separate inference path: it does not alter training or the
    optimized RGB-only AR cache.

    ``joint_ar_use_persistent_kv=True`` selects current-chunk-only maskless
    execution. Each finalized chunk is committed once; later denoising reads
    those immutable keys/values instead of recomputing historical queries.
    This temporary opt-in keeps replay available during numerical qualification.
    """
    fixed_sampler = getattr(host.config, "fixed_step_sampler_config", None)
    if sampler_mode is None:
        sampler_mode = "distilled" if fixed_sampler is not None else "rf"
    if sampler_mode not in ("rf", "distilled"):
        raise ValueError("Joint AR sampler_mode must be 'rf' or 'distilled'.")
    if sampler_mode == "rf" and fixed_sampler is not None:
        raise ValueError("A joint student checkpoint requires sampler_mode='distilled'.")
    if sampler_mode == "distilled":
        if fixed_sampler is None:
            raise ValueError("Joint distilled sampling requires a fixed-step student configuration.")
    if len(plans) != 1 or not plans[0].has_lidar or not plans[0].has_vision:
        raise ValueError("Joint AR requires exactly one RGB+LiDAR sequence")
    if plans[0].has_action or plans[0].has_sound:
        raise ValueError("Joint AR supports only RGB and LiDAR targets")
    if not host._uses_multiview_replay_kv() or host.config.compile.enabled:
        raise ValueError("Joint AR requires eager multiview teacher forcing")
    if host.parallel_dims is not None and host.parallel_dims.cfgp_enabled:
        raise ValueError("Joint AR currently uses serial CFG branches; set cfg_parallel_shard_degree=1")
    policy = host._get_teacher_forcing_replay_policy()
    if policy.control_visibility != "causal":
        raise ValueError("Joint AR prefix replay requires causal controls")
    if data.num_vision_items_per_sample != [2] or data.num_lidar_items_per_sample != [2]:
        raise ValueError("Joint AR requires control and target items for each sensor modality")
    if data.x0_tokens_vision is None or data.x0_tokens_lidar is None:
        raise ValueError("Joint AR requires encoded camera and LiDAR streams")
    views = data.num_views_per_vision_item
    if views is None or len(views) != 2 or views[0] < 1 or views[0] != views[1]:
        raise ValueError("Joint AR requires matching per-camera VAE metadata")
    num_views = views[0]
    if len(data.x0_tokens_vision) != 2 or len(data.x0_tokens_lidar) != 2:
        raise ValueError("Joint AR requires exactly two encoded items per modality")
    for items in (data.x0_tokens_vision, data.x0_tokens_lidar):
        if items[0].ndim != 5 or items[0].shape[0] != 1 or items[0].shape != items[1].shape:
            raise ValueError("Joint AR requires aligned [1,C,T,H,W] control and target latents")
    if data.x0_tokens_vision[1].shape[2] % num_views:
        raise ValueError("Joint AR camera-major latent count must be divisible by its view count")
    if guidance != 1.0 and unconditional_text is None:
        raise ValueError("Joint AR guidance requires unconditional captions")
    if seed < 0 or num_steps < 1 or not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Joint AR requires a nonnegative seed and finite valid sampling settings")

    # Rank-local VAE encoding can differ even with identical pixels and seeds.
    # CP shards must read one canonical set of controls and conditioned targets
    # before the initial pack and every subsequent clean/noisy replay.
    # Keep the existing Flex inference input behavior unchanged.
    if host._get_teacher_forcing_kv_implementation() == "multiview_maskless_kv":
        context_parallel_broadcast_tensor_list(data.x0_tokens_vision, host.parallel_dims)  # each [1,Cv,V*Tv,Hv,Wv]
        context_parallel_broadcast_tensor_list(data.x0_tokens_lidar, host.parallel_dims)  # each [1,Cl,Tl,Hl,Wl]

    if getattr(host.config, "rolling_kv_cache_chunks", None) is not None:
        if joint_ar_use_persistent_kv:
            raise ValueError(
                "rolling_kv_cache_chunks uses a separate cache path; disable --joint-ar-use-persistent-kv."
            )
        from cosmos_framework.model.generator.rolling_transfer_ar import sample_rolling_joint_transfer_ar

        return sample_rolling_joint_transfer_ar(
            host,
            plans=plans,
            data=data,
            conditional_text=conditional_text,
            unconditional_text=unconditional_text,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            prompt_schedule=prompt_schedule,
        )

    if prompt_schedule is not None:
        raise ValueError("Segmented prompts require rolling KV inference.")

    initial_pack = host._pack_input_sequence(plans, conditional_text, data, torch.zeros(1))  # timestep: [1]
    if initial_pack.vision is None or initial_pack.lidar is None:
        raise ValueError("The joint inference pack omitted a required modality")
    # Keep clock and prefix helpers independent of full transformer construction.
    from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items

    sensor_items = build_interactive_multiview_mask_items(initial_pack)
    if len(sensor_items) != 1 or len(sensor_items[0]) != 4:
        raise ValueError("Joint AR requires the training [RGB control, RGB target, LiDAR control, LiDAR target] layout")
    rgb_control, rgb_item, lidar_control, lidar_item = sensor_items[0]
    if not (rgb_control.is_control and lidar_control.is_control) or rgb_item.is_control or lidar_item.is_control:
        raise ValueError("Joint AR sensor control/target roles differ from the training layout")
    vision_condition = _prefix_condition_count(initial_pack.vision.condition_mask[1], num_views)
    lidar_condition = _prefix_condition_count(initial_pack.lidar.condition_mask[1], 1)
    vision_target = _conditioned_target(data.x0_tokens_vision[1], initial_pack.vision.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target = _conditioned_target(data.x0_tokens_lidar[1], initial_pack.lidar.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cl,Tl,Hl,Wl]
    frames_per_view = vision_target.shape[2] // num_views
    chunks = joint_ar_chunks(
        vision_frames=frames_per_view,
        lidar_frames=lidar_target.shape[2],
        vision_seconds_per_frame=rgb_item.seconds_per_frame,
        lidar_seconds_per_frame=lidar_item.seconds_per_frame,
        frames_per_chunk=int(host.config.teacher_forcing_frames_per_chunk),
    )
    persistent_branches = None
    if joint_ar_use_persistent_kv:
        if host._get_teacher_forcing_kv_implementation() != "multiview_maskless_kv":
            raise ValueError("Persistent joint K/V requires maskless attention")
        if host.config.kv_cache_inference_size is not None or host.config.attention_sink_size != 0:
            raise ValueError("Persistent joint K/V is unbounded in this phase; leave the window unset and sink at zero")
        _validate_persistent_chunks(chunks, vision_condition, lidar_condition)
        persistent_branches = [(conditional_text, JointARCache(initial_pack, host.net.num_hidden_layers))]
        if guidance != 1.0 and unconditional_text is not None:
            reference = host._pack_input_sequence(plans, unconditional_text, data, torch.zeros(1))  # timestep: [1]
            persistent_branches.append((unconditional_text, JointARCache(reference, host.net.num_hidden_layers)))
    del initial_pack
    for chunk in chunks:
        _sample_joint_chunk(
            host,
            plans=plans,
            data=data,
            conditional_text=conditional_text,
            unconditional_text=unconditional_text,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            vision_target=vision_target,
            lidar_target=lidar_target,
            num_views=num_views,
            frames_per_view=frames_per_view,
            vision_condition=vision_condition,
            lidar_condition=lidar_condition,
            chunk=chunk,
            sampler_mode=sampler_mode,
            distilled_num_steps=distilled_num_steps,
            persistent_branches=persistent_branches,
            # The last chunk need not be retained: no later chunk reads its K/V.
            retain_history=persistent_branches is not None and chunk is not chunks[-1],
        )
    return {"vision": [vision_target], "lidar": [lidar_target]}


def _sample_joint_chunk(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    frames_per_view: int,
    vision_condition: int,
    lidar_condition: int,
    chunk: JointARChunk,
    sampler_mode: Literal["rf", "distilled"] = "rf",
    distilled_num_steps: int | None = None,
    persistent_branches: list[tuple[list[list[int]], JointARCache]] | None = None,
    retain_history: bool = True,
) -> None:
    """Denoise the current RGB/LiDAR chunk and cache its clean K/V if persistent history is needed."""
    vision_frames = [frame for frame in chunk.vision_frames if frame >= vision_condition]
    lidar_frames = [frame for frame in chunk.lidar_frames if frame >= lidar_condition]
    if not vision_frames and not lidar_frames:
        if persistent_branches is not None and retain_history:
            current = _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
            _commit_persistent_joint_chunk(host, plans, current, persistent_branches, chunk)
        return
    forward_data = (
        _prefix_data(data, vision_target=vision_target, lidar_target=lidar_target, num_views=num_views, chunk=chunk)
        if persistent_branches is None
        else _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
    )
    assert forward_data.x0_tokens_vision is not None and forward_data.x0_tokens_lidar is not None
    vision_indexes = [view * chunk.vision_prefix_end + frame for view in range(num_views) for frame in vision_frames]
    final_vision_indexes = [view * frames_per_view + frame for view in range(num_views) for frame in vision_frames]
    local_lidar_frames = lidar_frames
    if persistent_branches is not None:
        vision_indexes = [
            view * len(chunk.vision_frames) + i
            for view in range(num_views)
            for i, frame in enumerate(chunk.vision_frames)
            if frame >= vision_condition
        ]
        local_lidar_frames = [i for i, frame in enumerate(chunk.lidar_frames) if frame >= lidar_condition]
    vision_shape = (1, vision_target.shape[1], len(vision_indexes), *vision_target.shape[-2:])
    lidar_shape = (1, lidar_target.shape[1], len(lidar_frames), *lidar_target.shape[-2:])
    vision_size = math.prod(vision_shape)
    lidar_size = math.prod(lidar_shape)
    generator = torch.Generator(device=vision_target.device).manual_seed(seed + chunk.step)
    initial_noise = torch.randn(
        (1, vision_size + lidar_size), device=vision_target.device, dtype=torch.float32, generator=generator
    )  # [1,D_vision+D_lidar]
    if persistent_branches is None:
        cond_pack, cond_memory = _build_replay_branch(host, plans, forward_data, conditional_text)
        uncond_branch = (
            _build_replay_branch(host, plans, forward_data, unconditional_text)
            if guidance != 1.0 and unconditional_text is not None
            else None
        )
    else:
        branches = [
            _build_persistent_branch(host, plans, forward_data, text, cache, chunk)
            for text, cache in persistent_branches
        ]
        cond_pack, cond_memory = branches[0]
        uncond_branch = branches[1] if len(branches) > 1 else None

    def branch_velocity(
        pack: PackedSequence,
        memory: TeacherForcingMemoryState | JointChunkMemory,
        noise: torch.Tensor,  # [1,D_vision+D_lidar]
        timestep: torch.Tensor,  # [1,1]
    ) -> torch.Tensor:  # [1,D_vision+D_lidar]
        assert pack.vision is not None and pack.lidar is not None
        noisy_vision = forward_data.x0_tokens_vision[1].clone()  # [1,Cv,V*Tv_forward,Hv,Wv]
        noisy_lidar = forward_data.x0_tokens_lidar[1].clone()  # [1,Cl,Tl_forward,Hl,Wl]
        noisy_vision[:, :, vision_indexes] = noise[:, :vision_size].reshape(vision_shape)  # [1,Cv,V*Tv_chunk,Hv,Wv]
        noisy_lidar[:, :, local_lidar_frames] = noise[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
        host._update_inference_pack_template(
            packed_sequence=pack,
            noise_x_vision=[forward_data.x0_tokens_vision[0], noisy_vision],  # list of [1,Cv,V*Tv_forward,Hv,Wv]
            noise_x_lidar=[forward_data.x0_tokens_lidar[0], noisy_lidar],  # list of [1,Cl,Tl_forward,Hl,Wl]
            noise_x_radar=None,
            noise_x_action=None,
            noise_x_sound=None,
            timestep=timestep,  # [1,1]
        )
        prediction = host.denoise(data_batch_packed=pack, memory=memory)
        vision_velocity = prediction["preds_vision"][1]  # [1,Cv,V*Tv_forward,Hv,Wv]
        lidar_velocity = prediction["preds_lidar"][1]  # [1,Cl,Tl_forward,Hl,Wl]
        if vision_velocity.shape != noisy_vision.shape or lidar_velocity.shape != noisy_lidar.shape:
            raise ValueError("Joint model predictions must retain each input item's [1,C,T,H,W] shape")
        # Grid-stream decoding retains its singleton batch dimension. Select time,
        # preserving every channel before flattening for the shared diffusion solver.
        return torch.cat(
            [
                vision_velocity[:, :, vision_indexes].reshape(1, -1),
                lidar_velocity[:, :, local_lidar_frames].reshape(1, -1),
            ],
            dim=1,
        )  # [1,D_vision+D_lidar]

    def velocity(noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # [1,D], [1,1] -> [1,D]
        conditional = branch_velocity(cond_pack, cond_memory, noise, timestep)  # [1,D]
        if uncond_branch is None:
            return conditional  # [1,D]
        unconditional = branch_velocity(*uncond_branch, noise, timestep)  # [1,D]
        return unconditional + guidance * (conditional - unconditional)  # [1,D]

    # Match the joint SF-DMD RGB clock, including a final LiDAR-only chunk.
    # Keep chunk-based noise seeds independent of the schedule's frame index.
    schedule_frame_idx = chunk.vision_frames[0] if chunk.vision_frames else frames_per_view - 1
    denoised = host._run_ar_sampler(
        velocity,
        initial_noise,
        sampler_mode=sampler_mode,
        num_steps=num_steps,
        shift=shift,
        seed=seed,
        sample_idx=chunk.step,
        num_frames=frames_per_view,
        distilled_num_steps=distilled_num_steps,
        schedule_frame_idx=schedule_frame_idx,
    )  # [1,D_vision+D_lidar]
    if not torch.isfinite(denoised).all():
        raise FloatingPointError(f"Joint AR produced nonfinite latents in causal chunk {chunk.step}")
    vision_target[:, :, final_vision_indexes] = denoised[:, :vision_size].reshape(
        vision_shape
    )  # [1,Cv,V*Tv_chunk,Hv,Wv]
    lidar_target[:, :, lidar_frames] = denoised[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
    if persistent_branches is not None and retain_history:
        current = _current_chunk_data(data, vision_target, lidar_target, num_views, chunk)
        # No later query reads the last chunk; do not run an unused clean forward.
        _commit_persistent_joint_chunk(host, plans, current, persistent_branches, chunk)
