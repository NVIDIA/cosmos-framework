# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sequence builder/output and plan helpers for VFM sequence packing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalityDataBuilder, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.mrope import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)

if TYPE_CHECKING:
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def _empty_long_tensor() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)  # [0]


@dataclass
class PackedSequenceBuilder:
    """Mutable construction state for sequence packing.

    Attributes:
        sample_lens: Length of each sample in the packed sequence.
        split_lens: Length of each attention split. Each sample contributes a causal
            text split and a full generation split.
        attn_modes: Attention mode for each split, such as ``"causal"`` or ``"full"``.
        is_image_batch: Whether this batch contains images rather than videos.
        sequence_length: Total packed sequence length. Filled during finalization.
        current_seq_index: Next global packed-sequence index to append.
        text_ids: Text token IDs accumulated during packing, including special tokens.
        text_indexes: Global packed-sequence indexes for text tokens.
        position_ids: mRoPE position ID blocks accumulated as ``[3, N]`` tensors.
        label_ids: Label IDs for text cross-entropy loss.
        ce_loss_indexes: Global packed-sequence indexes for text cross-entropy loss.
        ce_loss_weights: Per-token weights for text cross-entropy loss.
        _mrope_temporal_offset: Running temporal cursor for per-sample mRoPE generation.
        _mrope_reset_spatial: Whether spatial mRoPE IDs reset for each segment.
        null_action_supertokens: Whether temporal-causal supertoken 0 contains null
            action tokens.
        num_action_tokens_per_supertoken: Number of action tokens prefixing each
            temporal-causal vision supertoken.
        vision: Vision modality construction state, or ``None`` if no vision was appended.
        action: Action modality construction state, or ``None`` if no action was appended.
        sound: Sound modality construction state, or ``None`` if no sound was appended.
        vision_item_split_lens: Per-sample per-vision-item token counts for multi-control
            transfer.
        control_weights: Per-sample per-control weights for multi-control weighted V-scaling.
    """

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0

    # Build-time tracking (used during packing, not after finalize)
    current_seq_index: int = 0

    # Text modality append state
    text_ids: list[int] = field(default_factory=list)
    text_indexes: list[int] = field(default_factory=list)
    position_ids: list[torch.Tensor] = field(default_factory=list)

    # Loss computation - Cross Entropy (text)
    label_ids: list[int] = field(default_factory=list)
    ce_loss_indexes: list[int] = field(default_factory=list)
    ce_loss_weights: list[float] = field(default_factory=list)

    # Build-time mRoPE tracking (used during packing, not after finalize).
    # position_ids accumulates (3, N) tensors and finalize() produces a
    # (3, total_seq_len) tensor.
    # Running temporal index for mRoPE position ID generation within a single sample.
    # Reset to 0 at the start of each sample, then advanced by text and vision helpers
    # as segments are packed. Action reuses the pre-vision snapshot (parallel temporal
    # range) without advancing it. Float when FPS modulation is enabled.
    # E.g. offset=0 -> text(4 tokens) -> offset=4 -> vision(3 frames) -> offset=7.
    _mrope_temporal_offset: int | float = 0
    _mrope_reset_spatial: bool = True

    # Temporal causal: whether supertoken 0's action slot contains null tokens.
    # True for all training calls and AR frame 0; False for AR frame N>0 (real actions).
    # Used by three_way_attention to zero out V for null action tokens (inline when attention_meta.null_action_supertokens=True).
    null_action_supertokens: bool = False

    # Temporal causal: number of action tokens prefixing each vision supertoken.
    # Equals temporal_compression_factor when actions are packed inline; 0 when
    # action_gen=False or for non-temporal-causal layouts. Single source of truth
    # for downstream attention/KV-cache code (per-supertoken layout is
    # num_action_tokens_per_supertoken + H_p * W_p).
    num_action_tokens_per_supertoken: int = 0

    # Generation modality construction state
    vision: ModalityDataBuilder | None = None
    action: ModalityDataBuilder | None = None
    sound: ModalityDataBuilder | None = None

    def ensure_vision(self) -> ModalityDataBuilder:
        """Return the vision builder, creating it on first use.

        Returns:
            Vision ``ModalityDataBuilder`` for subsequent append operations.
        """
        if self.vision is None:
            self.vision = ModalityDataBuilder()
        return self.vision

    def ensure_action(self) -> ModalityDataBuilder:
        """Return the action builder, creating it on first use.

        Returns:
            Action ``ModalityDataBuilder`` for subsequent append operations.
        """
        if self.action is None:
            self.action = ModalityDataBuilder()
        return self.action

    def ensure_sound(self) -> ModalityDataBuilder:
        """Return the sound builder, creating it on first use.

        Returns:
            Sound ``ModalityDataBuilder`` for subsequent append operations.
        """
        if self.sound is None:
            self.sound = ModalityDataBuilder()
        return self.sound

    # Multi-control transfer: per-sample list of per-vision-item token counts.
    # For a multi-control transfer sample with N controls + 1 noisy target,
    # vision_item_split_lens[i] = [L_ctrl0, L_ctrl1, ..., L_ctrlN-1, L_noisy].
    # Used by cosmos3_vfm_network.py to derive gen-relative control/noisy ranges
    # for multi_control_two_way_attention.
    vision_item_split_lens: list[list[int]] = field(default_factory=list)

    # Per-sample per-control weights for multi-control weighted V-scaling.
    # Parallel to vision_item_split_lens[i][:-1] (excludes noisy item).
    # None for non-transfer or standard single-control samples.
    control_weights: list[list[float]] | None = None

    @property
    def mrope_temporal_offset(self) -> int | float:
        """Current per-sample temporal cursor used for mRoPE position generation."""
        return self._mrope_temporal_offset

    def set_mrope_temporal_offset(self, temporal_offset: int | float) -> None:
        """Set the per-sample temporal cursor used for mRoPE position generation.

        Args:
            temporal_offset: New mRoPE temporal cursor value.
        """
        self._mrope_temporal_offset = temporal_offset

    def advance_mrope_temporal_offset(self, delta: int | float) -> None:
        """Advance the per-sample temporal cursor by ``delta``.

        Args:
            delta: Amount to add to the current mRoPE temporal cursor.
        """
        self._mrope_temporal_offset += delta

    def begin_sample(self, initial_mrope_temporal_offset: int | float) -> None:
        """Reset per-sample position cursors before appending sample tokens.

        Args:
            initial_mrope_temporal_offset: Initial mRoPE temporal cursor for this sample.
        """
        self._mrope_temporal_offset = initial_mrope_temporal_offset

    def pack_text_tokens(
        self,
        text_ids: list[int],
        special_tokens: dict[str, int],
        has_generation: bool,
        use_float_positions: bool = False,
    ) -> int:
        """Pack text tokens into the sequence.

        Args:
            text_ids: List of text token IDs (integers).
            special_tokens: Dictionary of special token IDs.
            has_generation: Whether there's media/action after text.
            use_float_positions: If True, generate float position IDs for 3D mRoPE
                (for consistency with FPS-modulated vision tokens).

        Returns:
            Text sample length.
        """
        # Prepend BOS token if available
        if "bos_token_id" in special_tokens:
            shifted_text_ids = [special_tokens["bos_token_id"]] + text_ids
        else:
            shifted_text_ids = text_ids

        span_token_ids = shifted_text_ids + [special_tokens["eos_token_id"]]
        # Add start-of-generation token, but only if there's media/action present.
        if has_generation:
            span_token_ids.append(special_tokens["start_of_generation"])

        split_len = len(span_token_ids)
        expected_split_len = len(text_ids)
        if "bos_token_id" in special_tokens:
            expected_split_len += 1
        expected_split_len += 1  # EOS
        if has_generation:
            expected_split_len += 1  # start-of-generation / BOV
        assert split_len == expected_split_len

        position_ids, self._mrope_temporal_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=split_len,
            temporal_offset=self._mrope_temporal_offset,
            use_float_positions=use_float_positions,
        )  # position_ids: [3,split_len]

        span_start, _ = self.append_text_span(span_token_ids, position_ids)

        # Configure loss computation for text tokens
        self.ce_loss_indexes.extend(range(span_start, span_start + len(shifted_text_ids)))
        self.ce_loss_weights.extend([1.0] * len(shifted_text_ids))
        self.label_ids.extend(text_ids[1:] + [special_tokens["eos_token_id"]])

        self.attn_modes.append("causal")
        self.split_lens.append(split_len)

        return split_len

    def pack_vision_tokens(
        self,
        input_vision_tokens: torch.Tensor,
        condition_frame_indexes_vision: list[int],
        input_timestep: float | torch.Tensor,
        latent_patch_size: int = 1,
        vision_fps: float | None = None,
        enable_fps_modulation: bool = False,
        base_fps: float = 24.0,
        temporal_compression_factor: int = 4,
        vision_temporal_positions: torch.Tensor | None = None,
    ) -> int:
        """Pack vision tokens into the sequence.

        Args:
            input_vision_tokens: Vision latent tokens (C, T, H, W).
            condition_frame_indexes_vision: Indexes of conditioning frames.
            input_timestep: Diffusion timestep. Either a float (teacher_forcing/none — all frames
                share the same sigma) or a Tensor(T_max,) (diffusion_forcing — per-frame sigma;
                indexed as input_timestep[frame_idx] for each noisy frame).
            latent_patch_size: Patch size for latent patchification.
            vision_fps: Frames per second of the video. Used when enable_fps_modulation=True.
            enable_fps_modulation: If True, scale temporal position IDs based on video FPS.
            base_fps: Base FPS for normalization (default 24.0).
            temporal_compression_factor: VAE temporal compression factor (default 4).
            vision_temporal_positions: Optional explicit temporal coordinate per latent
                frame, shape ``(T,)``. Used by UniAE to account for kept boundary latents.

        Returns:
            Vision split length.
        """
        vision = self.ensure_vision()

        # Compute position IDs for image patches
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        if latent_patch_size < 1:
            raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}")
        # Use ceil to support latent dims not divisible by patch size (padding handled in network)
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        vision.token_shapes.append((latent_t, patch_h, patch_w))
        vision.tokens.append(input_vision_tokens)
        vision_payload_index = len(vision.tokens) - 1

        # Supervise vision tokens based on conditioning frames
        condition_set = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}

        vision_condition_mask = torch.zeros(
            (latent_t, 1, 1), device=input_vision_tokens.device, dtype=input_vision_tokens.dtype
        )  # [T,1,1]
        for frame_idx in condition_set:
            vision_condition_mask[frame_idx, 0, 0] = 1.0
        vision.condition_mask.append(vision_condition_mask)

        vision_noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in condition_set],
            device=input_vision_tokens.device,
            dtype=torch.long,
        )  # [N_noisy_frames]
        vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

        frame_token_stride = patch_h * patch_w
        effective_fps = vision_fps if enable_fps_modulation else None
        if vision_temporal_positions is not None:
            vision_temporal_positions = vision_temporal_positions.to(device="cpu", dtype=torch.float32)  # [T]

        vision_mrope_ids, self._mrope_temporal_offset = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=self._mrope_temporal_offset,
            reset_spatial_indices=self._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
            temporal_positions=vision_temporal_positions,
            actual_temporal_compression_factor=temporal_compression_factor,
        )  # vision_mrope_ids: [3,N_vision_tokens]
        vision_mrope_ids = vision_mrope_ids.reshape(3, latent_t, frame_token_stride)  # [3,T,H*W]

        vision_split_len = 0
        for frame_idx in range(latent_t):
            position_ids = vision_mrope_ids[:, frame_idx, :]  # [3,H*W]
            frame_indexes = self.append_vision_span(
                frame_token_stride,
                position_ids,
                payload_index=vision_payload_index,
                payload_start=frame_idx * frame_token_stride,
                payload_shape=(1, patch_h, patch_w),
            )
            vision_split_len += frame_token_stride

            if frame_idx in condition_set:
                continue
            vision.mse_loss_indexes.extend(frame_indexes)
            if isinstance(input_timestep, torch.Tensor):
                frame_ts = input_timestep[frame_idx].item()
            else:
                frame_ts = input_timestep
            vision.timesteps.extend([frame_ts] * frame_token_stride)

        return vision_split_len

    def pack_action_tokens(
        self,
        input_action_tokens: torch.Tensor,
        condition_frame_indexes_action: list[int],
        input_timestep: float,
        action_temporal_offset: int | float = 0,
        enable_fps_modulation: bool = False,
        base_fps: float = 24.0,
        action_fps: float | None = None,
        base_temporal_compression_factor: int | None = None,
        action_start_frame_offset: int = 1,
    ) -> int:
        """Pack action tokens into the sequence.

        Args:
            input_action_tokens: Action latent tokens (T, D).
            condition_frame_indexes_action: Indexes of conditioning action steps.
            input_timestep: Diffusion timestep.
            action_temporal_offset: Temporal offset for action mRoPE IDs (typically
                the vision start offset so action aligns temporally with vision).
            enable_fps_modulation: If True, scale temporal position IDs based on FPS.
            base_fps: Base FPS for normalization (default 24.0).
            action_fps: Frames per second of the action data. Used when enable_fps_modulation=True.
            base_temporal_compression_factor: Base temporal compression factor for FPS scaling.
                Should be set to the vision temporal compression factor (e.g. 4) so that action
                tokens advance at frame rate (4x finer) relative to vision latent frames.
                Only affects behavior when FPS modulation is enabled.
            action_start_frame_offset: Frame offset for aligning action[0] with the
                corresponding vision frame. Default 1 aligns action[0] with vision frame 1.

        Returns:
            Number of action tokens added.
        """
        action_split_len = input_action_tokens.shape[0]

        action = self.ensure_action()

        # Add token indexes and loss information
        action.token_shapes.append((action_split_len,))
        action.tokens.append(input_action_tokens)
        action_payload_index = len(action.tokens) - 1

        condition_set = {idx for idx in condition_frame_indexes_action if 0 <= idx < action_split_len}

        action_condition_mask = torch.zeros(
            (action_split_len, 1), device=input_action_tokens.device, dtype=input_action_tokens.dtype
        )  # [T_action,1]
        for frame_idx in condition_set:
            action_condition_mask[frame_idx, 0] = 1.0
        action.condition_mask.append(action_condition_mask)

        action_noisy_frame_indexes = torch.tensor(
            [idx for idx in range(action_split_len) if idx not in condition_set],
            device=input_action_tokens.device,
            dtype=torch.long,
        )  # [N_noisy_action_frames]
        action.noisy_frame_indexes.append(action_noisy_frame_indexes)

        # Action tokens use a 1x1 spatial grid with start_frame_offset=1 by default,
        # so action[0] (null token) aligns with vision frame 1, not frame 0.
        effective_fps = action_fps if enable_fps_modulation else None

        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=action_temporal_offset,
            reset_spatial_indices=self._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Action is at frame rate (no temporal compression)
            base_temporal_compression_factor=base_temporal_compression_factor,
            start_frame_offset=action_start_frame_offset,  # Align action[0] with vision frame action_start_frame_offset
        )  # action_mrope_ids: [3,N_action_tokens]
        # Note: we don't update _mrope_temporal_offset here because action tokens
        # share the temporal space with vision tokens (they run in parallel).

        for frame_idx in range(action_split_len):
            position_ids = action_mrope_ids[:, frame_idx : frame_idx + 1]  # [3,1]
            frame_indexes = self.append_action_span(
                1,
                position_ids,
                payload_index=action_payload_index,
                payload_start=frame_idx,
                payload_shape=(1,),
            )

            if frame_idx in condition_set:
                continue
            action.mse_loss_indexes.extend(frame_indexes)
            action.timesteps.extend([input_timestep])

        return action_split_len

    def pack_sound_tokens(
        self,
        input_sound_tokens: torch.Tensor,
        condition_frame_indexes_sound: list[int],
        input_timestep: float,
        sound_temporal_offset: int | float = 0,
        enable_fps_modulation: bool = False,
        base_fps: float = 24.0,
        sound_fps: float | None = None,
        sound_base_temporal_compression_factor: int | None = None,
    ) -> int:
        """Pack sound/audio tokens into the sequence.

        Sound latents have shape [C, T] where C is channels and T is temporal frames.
        Sound tokens are added to the unified generation split to maintain SequencePack's
        2-split invariant (causal + full).

        Args:
            input_sound_tokens: Sound latent tokens (C, T).
            condition_frame_indexes_sound: Indexes of conditioning frames.
                [] means all frames are noised/supervised.
                All frames specified means all frames are clean (no MSE supervision).
            input_timestep: Diffusion timestep.
            sound_temporal_offset: Temporal offset for m-RoPE position IDs (aligned with vision start).
            enable_fps_modulation: If True, scale temporal positions by FPS ratio.
            base_fps: Base FPS for normalization (default 24.0).
            sound_fps: Sound latent FPS (e.g., 25.0). Used for FPS-aware m-RoPE positions.
            sound_base_temporal_compression_factor: Base temporal compression factor for sound FPS scaling.
                ``None`` preserves the current behavior where sound advances at ``base_fps`` positions/sec.

        Returns:
            Number of sound tokens added.
        """
        # Sound latent shape: [C, T] → T tokens
        _, sound_split_len = input_sound_tokens.shape

        sound = self.ensure_sound()

        # Add token indexes - sound uses (T, 1, 1) shape for compatibility with 3D RoPE
        sound.token_shapes.append((sound_split_len, 1, 1))
        sound.tokens.append(input_sound_tokens)
        sound_payload_index = len(sound.tokens) - 1

        # Supervise sound tokens based on conditioning frames
        condition_set = {idx for idx in condition_frame_indexes_sound if 0 <= idx < sound_split_len}

        # Condition mask: shape (T, 1) — 1 = clean/conditioning, 0 = noised/supervised
        sound_condition_mask = torch.zeros(
            (sound_split_len, 1), device=input_sound_tokens.device, dtype=input_sound_tokens.dtype
        )  # [T_sound,1]
        for frame_idx in condition_set:
            sound_condition_mask[frame_idx, 0] = 1.0
        sound.condition_mask.append(sound_condition_mask)

        sound_noisy_frame_indexes = torch.tensor(
            [idx for idx in range(sound_split_len) if idx not in condition_set],
            device=input_sound_tokens.device,
            dtype=torch.long,
        )  # [N_noisy_sound_frames]
        sound.noisy_frame_indexes.append(sound_noisy_frame_indexes)

        # Sound tokens use a 1x1 spatial grid, aligned with vision temporal positions.
        # sound[0] aligns with vision frame 0 (start_frame_offset=0, unlike action which offsets by 1).
        effective_fps = sound_fps if enable_fps_modulation else None

        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_split_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=sound_temporal_offset,
            reset_spatial_indices=self._mrope_reset_spatial,
            fps=effective_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,  # Sound latent is already at sound_latent_fps (no further compression)
            base_temporal_compression_factor=sound_base_temporal_compression_factor,
            start_frame_offset=0,  # Sound[0] aligns with vision frame 0
        )  # sound_mrope_ids: [3,N_sound_tokens]
        # Note: we don't update _mrope_temporal_offset here because sound tokens
        # share the temporal space with vision tokens (they run in parallel).

        # Add to MSE loss indexes and timesteps for non-conditioning frames
        for frame_idx in range(sound_split_len):
            position_ids = sound_mrope_ids[:, frame_idx : frame_idx + 1]  # [3,1]
            frame_indexes = self.append_sound_span(
                1,
                position_ids,
                payload_index=sound_payload_index,
                payload_start=frame_idx,
                payload_shape=(1, 1, 1),
            )

            if frame_idx in condition_set:
                continue
            sound.mse_loss_indexes.extend(frame_indexes)
            sound.timesteps.extend([input_timestep])

        return sound_split_len

    def _append_position_ids(self, position_ids: torch.Tensor, span_len: int) -> None:
        """Append one block of position IDs.

        Args:
            position_ids: Position ID tensor with shape ``[3, span_len]``.
            span_len: Number of token positions described by ``position_ids``.
        """
        assert position_ids.shape[-1] == span_len, (
            f"position_ids last dimension must match span_len={span_len}; got {tuple(position_ids.shape)}"
        )
        self.position_ids.append(position_ids)

    def _append_modality_span(
        self,
        modality: ModalityDataBuilder,
        span_len: int,
        position_ids: torch.Tensor,
        *,
        payload_index: int,
        payload_start: int,
        payload_shape: tuple[int, ...],
    ) -> list[int]:
        """Append one modality span and return its global sequence indexes.

        Args:
            modality: Modality builder receiving sequence indexes and span metadata.
            span_len: Number of contiguous tokens to append.
            position_ids: Position ID tensor with shape ``[3, span_len]``.
            payload_index: Index into ``modality.tokens`` for the backing payload tensor.
            payload_start: First flattened token offset within the backing payload tensor.
            payload_shape: Logical payload slice shape represented by this span.

        Returns:
            Global packed-sequence indexes covered by the appended span.
        """
        assert payload_index >= 0, f"payload_index must be non-negative, got {payload_index}"
        assert payload_start >= 0, f"payload_start must be non-negative, got {payload_start}"
        payload_len = 1
        for dim in payload_shape:
            assert dim > 0, f"payload_shape dimensions must be positive, got {payload_shape}"
            payload_len *= dim
        assert payload_len == span_len, (
            f"payload_shape={payload_shape} describes {payload_len} tokens, but span_len={span_len}"
        )

        span_start = self.current_seq_index
        span_indexes = list(range(span_start, span_start + span_len))
        modality.spans.append(
            ModalitySpan(
                sequence_start=span_start,
                sequence_len=span_len,
                payload_index=payload_index,
                payload_start=payload_start,
                payload_len=payload_len,
                payload_shape=payload_shape,
            )
        )
        modality.sequence_indexes.extend(span_indexes)
        self._append_position_ids(position_ids, span_len)
        self.current_seq_index += span_len
        return span_indexes

    def append_text_span(
        self,
        token_ids: list[int],
        position_ids: torch.Tensor,
    ) -> tuple[int, int]:
        """Append one contiguous text span.

        Args:
            token_ids: Text token IDs to append.
            position_ids: Position ID tensor with shape ``[3, len(token_ids)]``.

        Returns:
            Tuple of ``(start_index, span_len)`` for the appended span.
        """
        span_start = self.current_seq_index
        span_len = len(token_ids)
        self.text_ids.extend(token_ids)
        self.text_indexes.extend(range(span_start, span_start + span_len))
        self._append_position_ids(position_ids, span_len)
        self.current_seq_index += span_len
        return span_start, span_len

    def append_vision_span(
        self,
        span_len: int,
        position_ids: torch.Tensor,
        *,
        payload_index: int,
        payload_start: int,
        payload_shape: tuple[int, int, int],
    ) -> list[int]:
        """Append one contiguous vision span.

        Args:
            span_len: Number of contiguous vision tokens to append.
            position_ids: Position ID tensor with shape ``[3, span_len]``.
            payload_index: Index into ``vision.tokens`` for the backing payload tensor.
            payload_start: First flattened token offset within the backing vision payload.
            payload_shape: Logical vision slice shape, usually ``(1, patch_h, patch_w)``.

        Returns:
            Global packed-sequence indexes covered by the appended vision span.
        """
        return self._append_modality_span(
            self.ensure_vision(),
            span_len,
            position_ids,
            payload_index=payload_index,
            payload_start=payload_start,
            payload_shape=payload_shape,
        )

    def append_action_span(
        self,
        span_len: int,
        position_ids: torch.Tensor,
        *,
        payload_index: int,
        payload_start: int,
        payload_shape: tuple[int],
    ) -> list[int]:
        """Append one contiguous action span.

        Args:
            span_len: Number of contiguous action tokens to append.
            position_ids: Position ID tensor with shape ``[3, span_len]``.
            payload_index: Index into ``action.tokens`` for the backing payload tensor.
            payload_start: First flattened token offset within the backing action payload.
            payload_shape: Logical action slice shape, usually ``(span_len,)``.

        Returns:
            Global packed-sequence indexes covered by the appended action span.
        """
        return self._append_modality_span(
            self.ensure_action(),
            span_len,
            position_ids,
            payload_index=payload_index,
            payload_start=payload_start,
            payload_shape=payload_shape,
        )

    def append_sound_span(
        self,
        span_len: int,
        position_ids: torch.Tensor,
        *,
        payload_index: int,
        payload_start: int,
        payload_shape: tuple[int, int, int],
    ) -> list[int]:
        """Append one contiguous sound span.

        Args:
            span_len: Number of contiguous sound tokens to append.
            position_ids: Position ID tensor with shape ``[3, span_len]``.
            payload_index: Index into ``sound.tokens`` for the backing payload tensor.
            payload_start: First flattened token offset within the backing sound payload.
            payload_shape: Logical sound slice shape, usually ``(1, 1, 1)``.

        Returns:
            Global packed-sequence indexes covered by the appended sound span.
        """
        return self._append_modality_span(
            self.ensure_sound(),
            span_len,
            position_ids,
            payload_index=payload_index,
            payload_start=payload_start,
            payload_shape=payload_shape,
        )

    def append_end_of_generation_token(
        self,
        token_id: int,
        use_float_mrope_positions: bool,
    ) -> int:
        """Append an end-of-generation text token.

        Args:
            token_id: Token ID for the end-of-generation marker.
            use_float_mrope_positions: Whether to write float position IDs for this token.

        Returns:
            Span length for the appended end-of-generation token.
        """
        eov_dtype = torch.float32 if use_float_mrope_positions else torch.long
        position_ids = torch.full((3, 1), self._mrope_temporal_offset, dtype=eov_dtype)  # [3,1]
        self._mrope_temporal_offset += 1

        _, span_len = self.append_text_span([token_id], position_ids)
        return span_len

    def finish_sample(self, generation_split_len: int, sample_len: int) -> None:
        """Record the non-causal generation split and total sample length.

        Args:
            generation_split_len: Length of the full-attention generation split.
            sample_len: Total number of tokens appended for this sample.
        """
        self.attn_modes.append("full")
        self.split_lens.append(generation_split_len)
        self.sample_lens.append(sample_len)

    def _finalize_modality(
        self,
        modality: ModalityDataBuilder | None,
        *,
        domain_id: list[torch.Tensor] | None = None,
        raw_action_dim: list[torch.Tensor | None] | None = None,
        include_raw_action_dim: bool = False,
    ) -> ModalityData | None:
        """Finalize one modality builder into model-facing modality data.

        Args:
            modality: Modality builder to finalize, or ``None`` if no tokens were appended.
            domain_id: Optional action domain IDs to attach to finalized action data.
            raw_action_dim: Optional raw action dimensions for action-channel masking.
            include_raw_action_dim: Whether to include ``raw_action_dim`` in the finalized data.

        Returns:
            Finalized ``ModalityData`` or ``None`` when the modality has no sequence indexes.
        """
        if modality is None or len(modality.sequence_indexes) == 0:
            return None

        kwargs = {
            "sequence_indexes": torch.tensor(modality.sequence_indexes, dtype=torch.long),  # [N_modality_tokens]
            "timesteps": torch.tensor(modality.timesteps, dtype=torch.float32),  # [N_modality_noisy_tokens]
            "mse_loss_indexes": torch.tensor(modality.mse_loss_indexes, dtype=torch.long),  # [N_modality_noisy_tokens]
            "spans": list(modality.spans),
            "token_shapes": list(modality.token_shapes),
            "tokens": modality.tokens,
            "condition_mask": list(modality.condition_mask),
            "noisy_frame_indexes": list(modality.noisy_frame_indexes),
        }
        if domain_id is not None:
            kwargs["domain_id"] = domain_id
        if include_raw_action_dim:
            kwargs["raw_action_dim"] = raw_action_dim
        return ModalityData(**kwargs)

    def finalize(
        self,
        gen_data_clean: GenerationDataClean,
    ) -> "PackedSequence":
        """Convert all lists to tensors and compute derived values.

        Args:
            gen_data_clean: GenerationDataClean for metadata (e.g., action domain IDs).

        Returns:
            New PackedSequence instance with tensors instead of lists.
        """
        # Compute sequence length
        sequence_length = sum(self.sample_lens)
        sample_lens = self.sample_lens.copy()
        split_lens = self.split_lens.copy()
        attn_modes = self.attn_modes.copy()

        # Prepare loss-related tensors (cross-entropy)
        label_ids: torch.Tensor | None = None
        ce_loss_indexes: torch.Tensor | None = None
        ce_loss_weights: torch.Tensor | None = None
        if self.label_ids and len(self.label_ids) > 0:
            label_ids = torch.tensor(self.label_ids)  # [N_ce_tokens]
            ce_loss_indexes = torch.tensor(self.ce_loss_indexes)  # [N_ce_tokens]
            ce_loss_weights = torch.tensor(self.ce_loss_weights)  # [N_ce_tokens]

        # The condition_mask and noisy_frame_indexes are kept as lists to support variable shapes.

        vision = self._finalize_modality(self.vision)
        action_domain_id = None
        if self.action is not None:
            if gen_data_clean.action_domain_id is not None:
                action_domain_id = gen_data_clean.action_domain_id
            else:
                default_action_domain_id = torch.zeros(1, dtype=torch.long)  # [1]
                action_domain_id = [default_action_domain_id] * len(self.action.token_shapes)
        action = self._finalize_modality(
            self.action,
            domain_id=action_domain_id,
            raw_action_dim=gen_data_clean.raw_action_dim,
            include_raw_action_dim=True,
        )
        sound = self._finalize_modality(self.sound)

        # Finalize position IDs.
        assert isinstance(self.position_ids, list)
        assert all(isinstance(position_id, torch.Tensor) for position_id in self.position_ids)
        if len(self.position_ids) > 0:
            position_ids = torch.cat(self.position_ids, dim=1)  # [3,actual_seq_len]
        else:
            position_ids = torch.empty((3, 0), dtype=torch.long)  # [3,0]

        return PackedSequence(
            # Sequence structure
            sequence_length=sequence_length,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            is_image_batch=gen_data_clean.is_image_batch,
            # Text modality (converted to tensors)
            text_ids=torch.tensor(self.text_ids, dtype=torch.long),  # [N_text_tokens]
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),  # [N_text_tokens]
            position_ids=position_ids,  # [3,seq_len]
            # Loss computation - Cross Entropy
            label_ids=label_ids,
            ce_loss_indexes=ce_loss_indexes,
            ce_loss_weights=ce_loss_weights,
            # Generation modalities
            vision=vision,
            action=action,
            sound=sound,
            # Temporal causal
            null_action_supertokens=self.null_action_supertokens,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
            # Multi-control transfer
            vision_item_split_lens=list(self.vision_item_split_lens),
            control_weights=gen_data_clean.control_weights,
        )


@dataclass
class PackedSequence:
    """Finalized model-facing packed sequence.

    The object remains mutable for model-side payload replacement, clean replay,
    and in-place device transfer. Build-only cursor and mRoPE state live on
    ``PackedSequenceBuilder``.

    Attributes:
        sample_lens: Length of each sample in the packed sequence.
        split_lens: Length of each attention split. Each sample contributes a causal
            text split and a full generation split.
        attn_modes: Attention mode for each split, such as ``"causal"`` or ``"full"``.
        is_image_batch: Whether this batch contains images rather than videos.
        sequence_length: Total length of the packed sequence.
        text_ids: Tensor of all text token IDs, including special tokens.
        text_indexes: Tensor of global packed-sequence indexes for text tokens.
        position_ids: Tensor of mRoPE position IDs for all packed tokens with shape
            ``[3, sequence_length]``.
        label_ids: Optional tensor of label IDs for text cross-entropy loss.
        ce_loss_indexes: Optional tensor of global packed-sequence indexes for text
            cross-entropy loss.
        ce_loss_weights: Optional tensor of per-token weights for text cross-entropy loss.
        null_action_supertokens: Whether temporal-causal supertoken 0 contains null
            action tokens.
        num_action_tokens_per_supertoken: Number of action tokens prefixing each
            temporal-causal vision supertoken.
        vision: Finalized vision modality data, or ``None`` if no vision is present.
        action: Finalized action modality data, or ``None`` if no action is present.
        sound: Finalized sound modality data, or ``None`` if no sound is present.
        vision_item_split_lens: Per-sample per-vision-item token counts for multi-control
            transfer.
        control_weights: Per-sample per-control weights for multi-control weighted V-scaling.
    """

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0

    # Text modality
    text_ids: torch.Tensor = field(default_factory=_empty_long_tensor)
    text_indexes: torch.Tensor = field(default_factory=_empty_long_tensor)
    position_ids: torch.Tensor = field(default_factory=_empty_long_tensor)

    # Loss computation - Cross Entropy (text)
    label_ids: torch.Tensor | None = None
    ce_loss_indexes: torch.Tensor | None = None
    ce_loss_weights: torch.Tensor | None = None

    # Temporal causal: whether supertoken 0's action slot contains null tokens.
    # True for all training calls and AR frame 0; False for AR frame N>0 (real actions).
    # Used by three_way_attention to zero out V for null action tokens (inline when attention_meta.null_action_supertokens=True).
    null_action_supertokens: bool = False

    # Temporal causal: number of action tokens prefixing each vision supertoken.
    # Equals temporal_compression_factor when actions are packed inline; 0 when
    # action_gen=False or for non-temporal-causal layouts. Single source of truth
    # for downstream attention/KV-cache code (per-supertoken layout is
    # num_action_tokens_per_supertoken + H_p * W_p).
    num_action_tokens_per_supertoken: int = 0

    # Generation modalities - NAMED FIELDS for type safety
    vision: ModalityData | None = None
    action: ModalityData | None = None
    sound: ModalityData | None = None

    # Multi-control transfer: per-sample list of per-vision-item token counts.
    # For a multi-control transfer sample with N controls + 1 noisy target,
    # vision_item_split_lens[i] = [L_ctrl0, L_ctrl1, ..., L_ctrlN-1, L_noisy].
    # Used by cosmos3_vfm_network.py to derive gen-relative control/noisy ranges
    # for multi_control_two_way_attention.
    vision_item_split_lens: list[list[int]] = field(default_factory=list)

    # Per-sample per-control weights for multi-control weighted V-scaling.
    # Parallel to vision_item_split_lens[i][:-1] (excludes noisy item).
    # None for non-transfer or standard single-control samples.
    control_weights: list[list[float]] | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.text_ids, torch.Tensor), "PackedSequence.text_ids must be finalized"
        assert isinstance(self.text_indexes, torch.Tensor), "PackedSequence.text_indexes must be finalized"
        assert isinstance(self.position_ids, torch.Tensor), "PackedSequence.position_ids must be finalized"
        if self.label_ids is not None:
            assert isinstance(self.label_ids, torch.Tensor), "PackedSequence.label_ids must be finalized"
        if self.ce_loss_indexes is not None:
            assert isinstance(self.ce_loss_indexes, torch.Tensor), "PackedSequence.ce_loss_indexes must be finalized"
        if self.ce_loss_weights is not None:
            assert isinstance(self.ce_loss_weights, torch.Tensor), "PackedSequence.ce_loss_weights must be finalized"
        for modality in [self.vision, self.action, self.sound]:
            assert modality is None or isinstance(modality, ModalityData), (
                "PackedSequence modality fields must be finalized ModalityData"
            )

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        self.text_ids = self.text_ids.cuda()
        self.text_indexes = self.text_indexes.cuda()
        self.position_ids = self.position_ids.cuda()
        if isinstance(self.label_ids, torch.Tensor):
            self.label_ids = self.label_ids.cuda()
        if isinstance(self.ce_loss_indexes, torch.Tensor):
            self.ce_loss_indexes = self.ce_loss_indexes.cuda()
        if isinstance(self.ce_loss_weights, torch.Tensor):
            self.ce_loss_weights = self.ce_loss_weights.cuda()
        if self.vision is not None:
            self.vision.to_cuda()
        if self.action is not None:
            self.action.to_cuda()
        if self.sound is not None:
            self.sound.to_cuda()


@dataclass
class SequencePlan:
    """Plan describing which modalities are present in a sample.

    This dataclass tracks the presence of different modalities (text, vision, action)
    and their conditioning configurations for a dataset sample. Unlike SequencePlan
    which holds the actual tensor data, this class provides a lightweight summary
    of what modalities exist and how they should be conditioned.

    Attributes:
        has_text: Whether text/caption tokens are present for this sample.
            Used for text-conditioned generation (e.g., text-to-image/video).
        has_vision: Whether vision input (image or video latents) is present.
            Defaults to False.
        condition_frame_indexes_vision: Indexes of latent vision frames that are clean/conditioning.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
            For multi-item samples (e.g. image editing where each sample has multiple
            separately-encoded images), this applies to each vision item individually.
            The number of items per sample is tracked by
            ``GenerationDataClean.num_vision_items_per_sample``.
        share_vision_temporal_positions: Whether all vision items in this sample share
            the same temporal mRoPE grid.
        has_action: Whether action input is present for robotics/embodied AI tasks.
            Defaults to False.
        condition_frame_indexes_action: Indexes of action steps that are clean/conditioning.
            [] means all steps are noised/supervised.
            All steps specified means all steps are clean (no MSE supervision).
        action_start_frame_offset: Frame offset for aligning action[0] to vision frames.
        has_sound: Whether sound/audio input is present.
        condition_frame_indexes_sound: Indexes of sound frames that are clean/conditioning.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
    """

    # -- understanding (text conditioning) --
    has_text: bool

    # -- vision modality --
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    # If True, all vision items in this sample share the same temporal mRoPE grid
    # (controlnet-style transfer: target frame i is spatio-temporally aligned with
    # control frame i). Each item gets the same temporal_offset; spatial reset
    # behavior is unchanged. Requires num_vision_items_per_sample > 1, equal latent_t,
    # and equal fps across items. Default False preserves single-clip and
    # image-editing semantics where items represent distinct time states.
    share_vision_temporal_positions: bool = False

    # -- action modality --
    has_action: bool = False
    condition_frame_indexes_action: list[int] = field(default_factory=list)
    action_start_frame_offset: int = 1

    # -- sound modality --
    has_sound: bool = False
    condition_frame_indexes_sound: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_action": self.has_action,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
            "condition_frame_indexes_action": self.condition_frame_indexes_action,
            "condition_frame_indexes_sound": self.condition_frame_indexes_sound,
            "share_vision_temporal_positions": self.share_vision_temporal_positions,
        }


def build_sequence_plans_from_data_batch(
    data_batch: dict,
    input_video_key,
    input_image_key: str,
) -> list[SequencePlan]:
    """Build or retrieve sequence plans from a data batch dictionary.

    This function extracts sequence plans from the data batch if they exist,
    otherwise creates default SequencePlan objects for each sample
    in the batch.

    Args:
        data_batch: Dictionary containing the data batch from the dataloader.
            Expected keys include 'video' or other tensors to determine batch size.
            If 'sequence_plan' key exists, those plans are returned directly.
        input_video_key: Data-batch key used to find video tensors when inferring batch size.
        input_image_key: Data-batch key used to find image tensors when inferring batch size.

    Returns:
        List of SequencePlan objects, one per sample in the batch.
    """
    # For new modalities, please generate the sequence_plan in the dataset class!!!!

    # If sequence_plan already exists in data_batch, return it
    if "sequence_plan" in data_batch:
        return data_batch["sequence_plan"]

    assert "action" not in data_batch or data_batch["action"] is None, "Action data SHOULD have sequence_plans!"
    assert "sound" not in data_batch or data_batch["sound"] is None, "Sound data SHOULD have sequence_plans!"

    # Determine batch size from available tensors
    batch_size = 0
    for key in [input_video_key, input_image_key]:
        if key in data_batch:
            val = data_batch[key]
            if isinstance(val, torch.Tensor):
                batch_size = val.shape[0]
                break
            elif isinstance(val, list):
                batch_size = len(val)
                break

    if batch_size == 0:
        raise ValueError(
            f"Cannot determine batch size from data_batch. Expected {input_video_key}, {input_image_key}, or similar key."
        )

    # Build default SequencePlan objects
    return [
        SequencePlan(
            has_text=True,  # Has text prompt!
            has_vision=True,
            condition_frame_indexes_vision=[],  # No conditioning frames!
        )
        for _ in range(batch_size)
    ]
