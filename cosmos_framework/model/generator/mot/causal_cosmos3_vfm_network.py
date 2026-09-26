# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Interactive network adapter for multiview replay attention."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import attrs
import torch
from torch.utils.hooks import RemovableHandle

from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    _multiview_caption_mask_items,
    _multiview_sensor_mask_items,
)
from cosmos_framework.model.generator.mot.flex_attention import SensorMaskItem
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
)
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.model.generator.mot.causal_flex_attention import (
    MULTIVIEW_TRANSFER_AR_CURRENT_ROLES,
    MultiviewTransferARMemoryLayout,
    build_multiview_transfer_ar_flex_metadata,
    build_teacher_forcing_block_mask,
    build_teacher_forcing_multiview_flex_metadata,
)
from cosmos_framework.model.generator.mot.maskless_attention import build_replay_maskless_plan


def build_interactive_multiview_mask_items(
    packed_seq: PackedSequence,
    *,
    lidar_attends_captions: bool = True,
    condition_masks: Sequence[torch.Tensor] | None = None,
) -> list[list[SensorMaskItem]]:
    """Build base sensor mask items, optionally restoring pre-replay condition masks."""
    sensor_mask_items = _multiview_sensor_mask_items(packed_seq, lidar_attends_captions=lidar_attends_captions)
    if condition_masks is None:
        return sensor_mask_items
    flat_items = [item for sample_items in sensor_mask_items for item in sample_items]
    if len(flat_items) != len(condition_masks):
        raise ValueError(
            f"Teacher-forcing condition masks contain {len(condition_masks)} items, but the pack contains "
            f"{len(flat_items)}."
        )
    mask_iter = iter(condition_masks)
    return [
        [replace(item, condition_mask=next(mask_iter)) for item in sample_items] for sample_items in sensor_mask_items
    ]


def _global_flex_stream_lengths(
    input_pack: SequencePack,
    *,
    local_gen_seq_len: int,
    local_und_seq_len: int,
    parallel_dims: ParallelDims | None,
) -> tuple[int, int]:
    """Recover the padded global stream lengths seen by Ulysses attention."""
    if not input_pack["is_sharded"]:
        return local_gen_seq_len, local_und_seq_len
    if parallel_dims is None or not parallel_dims.cp_enabled:
        raise ValueError("A sharded SequencePack requires enabled context-parallel dimensions.")
    return local_gen_seq_len * parallel_dims.cp_size, local_und_seq_len * parallel_dims.cp_size


class InteractiveCosmos3VFMNetwork(Cosmos3VFMNetwork):
    """Replace the base multiview mask immediately before decoder execution.

    The base network owns packing, projections, and decoder invocation. Its language
    model pre-hook is the narrow extension point available without modifying core
    Cosmos3: the hook receives the finished ``SplitInfo`` and packed Q/KV geometry,
    after the base network has built them but before any decoder layer consumes them.
    """

    _active_packed_seq: PackedSequence | None
    _teacher_forcing_mask_hook: RemovableHandle
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig | None
    teacher_forcing_frames_per_chunk: int
    teacher_forcing_maskless: bool = False

    def __init__(self, language_model: torch.nn.Module, config: Any) -> None:
        super().__init__(language_model=language_model, config=config)
        self._active_packed_seq = None
        self.teacher_forcing_replay_policy = None
        self.teacher_forcing_frames_per_chunk = 1
        self.teacher_forcing_maskless = False
        self._teacher_forcing_mask_hook = self.language_model.register_forward_pre_hook(
            self._replace_teacher_forcing_mask,
            with_kwargs=True,
        )

    def _replace_teacher_forcing_mask(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Replace only replayed-TF or multiview-transfer-AR masks; preserve every base path."""
        del module
        packed_seq = self._active_packed_seq
        if packed_seq is None:
            return None
        ar_metadata = getattr(packed_seq, "multiview_transfer_ar_metadata", None)
        teacher_forcing_pass = getattr(packed_seq, "teacher_forcing_pass", None)
        if ar_metadata is None and teacher_forcing_pass is None:
            return None
        teacher_forcing_replay_policy = self.teacher_forcing_replay_policy
        if not isinstance(teacher_forcing_replay_policy, TeacherForcingReplayPolicyConfig):
            raise TypeError("Interactive teacher forcing requires a TeacherForcingReplayPolicyConfig.")
        if self.flex_backend is None and not self.teacher_forcing_maskless:
            raise ValueError("Interactive teacher forcing requires a resolved FlexAttention backend.")
        # Metadata is independent of spatial scope. Use the same-view metadata
        # builder for maskless joint inputs; the Flex-only cross-sensor frame-index
        # restriction does not apply to the maskless midpoint buckets.
        metadata_policy = (
            attrs.evolve(teacher_forcing_replay_policy, multiview_attention_scope="same_view")
            if self.teacher_forcing_maskless
            else teacher_forcing_replay_policy
        )
        attention_meta = kwargs.get("attention_mask")
        if not isinstance(attention_meta, SplitInfo) or attention_meta.is_three_way:
            raise ValueError("Interactive multiview teacher forcing requires two-way SplitInfo metadata.")
        if not args or not isinstance(args[0], dict):
            raise TypeError("The interactive network expected a SequencePack as the language-model input.")
        input_pack: SequencePack = args[0]
        full_only_seq, full_q_offsets = get_full_only_seq(input_pack)  # [N_gen,H,D], [B+1 or B+2]
        causal_seq, causal_offsets = get_causal_seq(input_pack)  # [N_und,H], [B+1 or B+2]
        global_gen_seq_len, global_und_seq_len = _global_flex_stream_lengths(
            input_pack,
            local_gen_seq_len=full_only_seq.shape[0],
            local_und_seq_len=causal_seq.shape[0],
            parallel_dims=self.parallel_dims,
        )
        if ar_metadata is not None:
            if not isinstance(ar_metadata, dict):
                raise TypeError("Multiview transfer AR metadata must be a dictionary.")
            memory_layout = ar_metadata.get("memory_layout")
            if not isinstance(memory_layout, MultiviewTransferARMemoryLayout):
                raise TypeError("Multiview transfer AR metadata requires a MultiviewTransferARMemoryLayout.")
            current_frame_start = ar_metadata.get("current_frame_start")
            frames_per_view = ar_metadata.get("frames_per_view")
            frames_per_chunk = ar_metadata.get("frames_per_chunk")
            current_role = ar_metadata.get("current_role")
            if not isinstance(current_frame_start, int):
                raise TypeError("Multiview transfer AR current_frame_start must be an int.")
            if not isinstance(frames_per_view, int):
                raise TypeError("Multiview transfer AR frames_per_view must be an int.")
            if not isinstance(frames_per_chunk, int):
                raise TypeError("Multiview transfer AR frames_per_chunk must be an int.")
            if current_role not in MULTIVIEW_TRANSFER_AR_CURRENT_ROLES:
                raise ValueError(
                    f"Multiview transfer AR current_role must be one of {MULTIVIEW_TRANSFER_AR_CURRENT_ROLES}, "
                    f"got {current_role!r}."
                )
            sensor_items = build_interactive_multiview_mask_items(
                packed_seq,
                lidar_attends_captions=self.config.multiview_attention_config.mask.lidar_attends_captions,
            )
            flex_metadata = build_multiview_transfer_ar_flex_metadata(
                seq_len=global_gen_seq_len,
                full_q_offsets=full_q_offsets,
                sensor_mask_items=sensor_items,
                caption_mask_items=_multiview_caption_mask_items(packed_seq),
                device=full_only_seq.device,
                num_und=global_und_seq_len,
                causal_offsets=causal_offsets,
                current_frame_start=current_frame_start,
                frames_per_view=frames_per_view,
                frames_per_chunk=frames_per_chunk,
                current_role=current_role,
                teacher_forcing_replay_policy=metadata_policy,
                memory_layout=memory_layout,
            )
        elif teacher_forcing_pass is not None:
            if teacher_forcing_pass not in ("clean", "noisy"):
                raise ValueError(f"Unknown teacher-forcing pass {teacher_forcing_pass!r}.")
            materialized_target_frame_ranges = getattr(
                packed_seq,
                "teacher_forcing_materialized_target_frame_ranges",
                None,
            )
            if materialized_target_frame_ranges is not None and (
                not isinstance(materialized_target_frame_ranges, Sequence)
                or isinstance(materialized_target_frame_ranges, (str, bytes))
            ):
                raise TypeError("teacher_forcing_materialized_target_frame_ranges must be a sequence of ranges.")
            original_masks = getattr(packed_seq, "teacher_forcing_original_condition_masks_sensors", None)
            if original_masks is None:
                # RGB-only AR callers created before the joint replay path use this name.
                original_masks = getattr(packed_seq, "teacher_forcing_original_condition_masks_vision", None)
            if original_masks is None:
                raise ValueError("Flex teacher forcing requires the original sensor condition masks.")
            sensor_items = build_interactive_multiview_mask_items(
                packed_seq,
                lidar_attends_captions=self.config.multiview_attention_config.mask.lidar_attends_captions,
                condition_masks=original_masks,
            )
            flex_metadata = build_teacher_forcing_multiview_flex_metadata(
                seq_len=global_gen_seq_len,
                full_q_offsets=full_q_offsets,
                sensor_mask_items=sensor_items,
                caption_mask_items=_multiview_caption_mask_items(packed_seq),
                device=full_only_seq.device,
                num_und=global_und_seq_len,
                causal_offsets=causal_offsets,
                frames_per_chunk=self.teacher_forcing_frames_per_chunk,
                pass_kind=teacher_forcing_pass,
                teacher_forcing_replay_policy=metadata_policy,
                materialized_target_frame_ranges=materialized_target_frame_ranges,
                clean_memory_seq_len=(
                    int(getattr(packed_seq, "teacher_forcing_selected_clean_target_padded_capacity", 0))
                    if teacher_forcing_pass == "noisy"
                    else 0
                ),
            )
        else:
            return None

        if self.teacher_forcing_maskless:
            flex_metadata = replace(flex_metadata, teacher_forcing_replay_policy=teacher_forcing_replay_policy)
            setattr(attention_meta, "replay_maskless_plan", build_replay_maskless_plan(flex_metadata, sensor_items))
            # The marked replay dispatch must consume this plan, never the base
            # bidirectional plan that was constructed before the pre-hook.
            attention_meta.multiview_maskless = None
            return args, kwargs
        assert self.flex_backend is not None
        attention_meta.flex_block_mask = build_teacher_forcing_block_mask(
            flex_metadata,
            full_only_seq.device,
            self.flex_backend.block_size,
        )
        attention_meta.flex_backend = self.flex_backend
        return args, kwargs

    def forward(
        self,
        packed_seq: PackedSequence,
        memory: Any | None = None,
        video_temporal_causal: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Expose the active pack to the instance-local language-model hook."""
        previous_packed_seq = self._active_packed_seq
        self._active_packed_seq = packed_seq
        try:
            if (
                self.teacher_forcing_maskless
                and video_temporal_causal is not False
                and (
                    getattr(packed_seq, "teacher_forcing_pass", None) is not None
                    or getattr(packed_seq, "multiview_transfer_ar_metadata", None) is not None
                )
            ):
                kwargs["bounded_cp_output_gather"] = True
            return super().forward(
                packed_seq=packed_seq,
                memory=memory,
                video_temporal_causal=video_temporal_causal,
                **kwargs,
            )
        finally:
            self._active_packed_seq = previous_packed_seq
