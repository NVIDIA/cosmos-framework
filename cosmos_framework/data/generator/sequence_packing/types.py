# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataclasses and plan helpers for VFM sequence packing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@dataclass
class ModalityData:
    """Unified container for a single generation modality's data.

    This dataclass serves dual purposes:
    1. During packing: Acts as a builder, accumulating data in lists
    2. After finalize(): Holds finalized tensors ready for model consumption

    Attributes:
        sequence_indexes: Indices in the packed sequence where this modality's tokens appear.
            List during building, Tensor after finalize().
        timesteps: Diffusion timesteps for each noised token.
            List during building, Tensor after finalize().
        mse_loss_indexes: Indices where MSE loss should be computed (noised tokens only).
            List during building, Tensor after finalize().
        token_shapes: Shape metadata for each sample's tokens.
            For vision: list of (T, H, W) tuples.
            For action: list of (T,) tuples.
        tokens: The actual latent tokens. List during build, Tensor after finalize().
        condition_mask: Mask indicating clean frames (1=clean, 0=noised). Only after finalize().
        noisy_frame_indexes: Indices of noised frames. Constructed from condition_mask during
            sequence packing to reduce GPU->CPU synchronization later. Only after finalize().
        domain_id: Domain ID for multi-domain training. Only after finalize(). NOTE: only used for action modality.
        raw_action_dim: Raw action dimension. Only after finalize(). NOTE: only used for action modality.
    """

    # Core tracking (list during build, tensor after finalize)
    sequence_indexes: list[int] | torch.Tensor = field(default_factory=list)
    timesteps: list[float] | torch.Tensor = field(default_factory=list)
    mse_loss_indexes: list[int] | torch.Tensor = field(default_factory=list)
    # list[tuple[int,int,int]] for vision, list[tuple[int]] for action, list[tuple[int,int,int]] for sound
    token_shapes: list = field(default_factory=list)

    # Populated during finalization (from GenerationDataClean / noise path)
    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    domain_id: list[torch.Tensor] = field(default_factory=list)
    raw_action_dim: list[torch.Tensor | None] | None = field(default_factory=list)

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.sequence_indexes, torch.Tensor):
            self.sequence_indexes = self.sequence_indexes.cuda()
        if isinstance(self.timesteps, torch.Tensor):
            self.timesteps = self.timesteps.cuda()
        if isinstance(self.mse_loss_indexes, torch.Tensor):
            self.mse_loss_indexes = self.mse_loss_indexes.cuda()
        self.tokens = [token.cuda() for token in self.tokens]
        self.condition_mask = [cm.cuda() for cm in self.condition_mask]
        self.noisy_frame_indexes = [ni.cuda() for ni in self.noisy_frame_indexes]
        self.domain_id = [d.cuda() for d in self.domain_id]
        # raw_action_dim is optional (e.g., when action-channel masking is disabled).
        if self.raw_action_dim is not None:
            self.raw_action_dim = [d.cuda() if d is not None else None for d in self.raw_action_dim]


@dataclass
class PackedSequence:
    """Unified sequence container - works as builder during packing and final output.

    This dataclass replaces the old SequenceStatus + PackedSequence pattern:
    - Build phase: Accumulate data using lists, modalities use ModalityData builders
    - After finalize(): Ready for model consumption with tensors

    Attributes:
        # Sequence structure
        sample_lens: Length of each sample in the packed sequence.
        split_lens: Length of each split (text/vision/action sections).
        attn_modes: Attention mode for each split ('causal', 'full').
        is_image_batch: Whether this batch contains images (vs videos).
        sequence_length: Total length of packed sequence. Computed during finalize().

        # Build-time tracking (not used after finalize)
        curr: Current position in the packed sequence during building.

        # Text modality (list during build, tensor after finalize)
        text_ids: All text token IDs (including special tokens).
        text_indexes: Indices where text tokens appear in sequence.
        position_ids: RoPE position IDs for all tokens.

        # Loss computation - Cross Entropy (text)
        label_ids: Label IDs for cross-entropy loss.
        ce_loss_indexes: Indices for computing cross-entropy loss.
        ce_loss_weights: Weights for cross-entropy loss.

        # Generation modalities - named fields for type safety
        vision: Vision modality data (images/videos). None if no vision in batch.
        action: Action modality data (robotics). None if no actions in batch.
        sound: Sound modality data (audio). None if no sound in batch.
    """

    # Sequence structure
    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0

    # Build-time tracking (used during packing, not after finalize)
    curr: int = 0

    # Text modality (list during build, tensor after finalize)
    text_ids: list[int] | torch.Tensor = field(default_factory=list)
    text_indexes: list[int] | torch.Tensor = field(default_factory=list)
    position_ids: list[torch.Tensor] | torch.Tensor = field(default_factory=list)

    # Loss computation - Cross Entropy (text)
    label_ids: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_indexes: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_weights: list[float] | torch.Tensor | None = field(default_factory=list)

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

        # Finalize vision modality
        vision: ModalityData | None = None
        if self.vision is not None and len(self.vision.sequence_indexes) > 0:
            vision = ModalityData(
                sequence_indexes=torch.tensor(self.vision.sequence_indexes, dtype=torch.long),  # [N_vision_tokens]
                timesteps=torch.tensor(self.vision.timesteps, dtype=torch.float32),  # [N_vision_noisy_tokens]
                mse_loss_indexes=torch.tensor(
                    self.vision.mse_loss_indexes, dtype=torch.long
                ),  # [N_vision_noisy_tokens]
                token_shapes=list(self.vision.token_shapes),
                tokens=self.vision.tokens,
                condition_mask=list(self.vision.condition_mask),
                noisy_frame_indexes=list(self.vision.noisy_frame_indexes),
            )

        # Finalize action modality
        action: ModalityData | None = None
        if self.action is not None and len(self.action.sequence_indexes) > 0:
            action = ModalityData(
                sequence_indexes=torch.tensor(self.action.sequence_indexes, dtype=torch.long),  # [N_action_tokens]
                timesteps=torch.tensor(self.action.timesteps, dtype=torch.float32),  # [N_action_noisy_tokens]
                mse_loss_indexes=torch.tensor(
                    self.action.mse_loss_indexes, dtype=torch.long
                ),  # [N_action_noisy_tokens]
                token_shapes=list(self.action.token_shapes),
                tokens=self.action.tokens,
                condition_mask=list(self.action.condition_mask),  # Keep as list to support variable shapes
                noisy_frame_indexes=list(self.action.noisy_frame_indexes),
                domain_id=(
                    gen_data_clean.action_domain_id
                    if gen_data_clean.action_domain_id is not None
                    else [torch.zeros(1, dtype=torch.long)] * len(self.action.token_shapes)
                ),
                raw_action_dim=gen_data_clean.raw_action_dim,
            )

        # Finalize sound modality (placeholder for future)
        sound: ModalityData | None = None
        if self.sound is not None and len(self.sound.sequence_indexes) > 0:
            sound = ModalityData(
                sequence_indexes=torch.tensor(self.sound.sequence_indexes, dtype=torch.long),  # [N_sound_tokens]
                timesteps=torch.tensor(self.sound.timesteps, dtype=torch.float32),  # [N_sound_noisy_tokens]
                mse_loss_indexes=torch.tensor(self.sound.mse_loss_indexes, dtype=torch.long),  # [N_sound_noisy_tokens]
                token_shapes=list(self.sound.token_shapes),
                tokens=self.sound.tokens,
                condition_mask=list(self.sound.condition_mask),
                noisy_frame_indexes=list(self.sound.noisy_frame_indexes),
            )

        # Finalize position IDs.
        assert isinstance(self.position_ids, list)
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

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.text_ids, torch.Tensor):
            self.text_ids = self.text_ids.cuda()
        if isinstance(self.text_indexes, torch.Tensor):
            self.text_indexes = self.text_indexes.cuda()
        if isinstance(self.position_ids, torch.Tensor):
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
        has_action: Whether action input is present for robotics/embodied AI tasks.
            Defaults to False.
        condition_frame_indexes_action: Indexes of action steps that are clean/conditioning.
            [] means all steps are noised/supervised.
            All steps specified means all steps are clean (no MSE supervision).
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
