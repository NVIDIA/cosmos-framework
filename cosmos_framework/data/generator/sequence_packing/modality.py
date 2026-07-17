# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Modality utility helpers for VFM sequence packing."""

from dataclasses import dataclass, field

import torch


def _empty_long_tensor() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)  # [0]


def _empty_float_tensor() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)  # [0]


@dataclass
class ModalitySpan:
    """One contiguous packed span paired with its logical modality payload slice.

    Attributes:
        sequence_start: First global packed-sequence index owned by this span.
        sequence_len: Number of contiguous tokens in the packed sequence.
        payload_index: Index into ``ModalityData.tokens`` for the backing payload tensor.
        payload_start: First flattened token offset within the backing payload tensor.
        payload_len: Number of flattened payload tokens covered by this span.
        payload_shape: Logical payload slice shape. For example, vision frame spans use
            ``(1, patch_h, patch_w)``, action spans use ``(tcf,)``, and sound spans use
            ``(1, 1, 1)``.
    """

    sequence_start: int
    sequence_len: int
    payload_index: int
    payload_start: int
    payload_len: int
    payload_shape: tuple[int, ...]


@dataclass
class ModalityDataBuilder:
    """Append-only construction state for a single generation modality.

    Attributes:
        spans: Contiguous packed spans pointing back into grouped payload tensors.
        sequence_indexes: Global packed-sequence indexes for all tokens of this modality.
        timesteps: Diffusion timesteps for noised tokens only.
        mse_loss_indexes: Global packed-sequence indexes where MSE loss should be computed.
        token_shapes: Shape metadata for each payload tensor. Vision and sound use
            ``(T, H, W)``-style tuples; action uses ``(T,)``.
        tokens: Original modality payload tensors, kept grouped by sample/item.
        condition_mask: Per-payload masks where 1 indicates clean/conditioning tokens
            and 0 indicates noised/supervised tokens.
        noisy_frame_indexes: Per-payload indexes of noised frames. These are constructed
            during packing to avoid GPU-to-CPU synchronization later.
    """

    spans: list[ModalitySpan] = field(default_factory=list)
    sequence_indexes: list[int] = field(default_factory=list)
    timesteps: list[float] = field(default_factory=list)
    mse_loss_indexes: list[int] = field(default_factory=list)
    # list[tuple[int,int,int]] for vision, list[tuple[int]] for action, list[tuple[int,int,int]] for sound
    token_shapes: list[tuple[int, ...]] = field(default_factory=list)
    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)


@dataclass
class ModalityData:
    """Finalized model-facing data for a single generation modality.

    Index-like fields are tensors after packing. Payload fields remain lists
    because samples may have variable token shapes and because downstream code
    mutates token payloads after noise injection and clean replay.

    Attributes:
        sequence_indexes: Tensor of global packed-sequence indexes for all tokens of
            this modality.
        timesteps: Tensor of diffusion timesteps for noised tokens only.
        mse_loss_indexes: Tensor of global packed-sequence indexes where MSE loss
            should be computed.
        spans: Contiguous packed spans pointing back into grouped payload tensors.
        token_shapes: Shape metadata for each payload tensor. Vision and sound use
            ``(T, H, W)``-style tuples; action uses ``(T,)``.
        tokens: Original modality payload tensors, kept grouped by sample/item.
        condition_mask: Per-payload masks where 1 indicates clean/conditioning tokens
            and 0 indicates noised/supervised tokens.
        noisy_frame_indexes: Per-payload indexes of noised frames.
        domain_id: Domain IDs for multi-domain training. Only used for action.
        raw_action_dim: Raw action dimensions. Only used for action-channel masking.
    """

    sequence_indexes: torch.Tensor = field(default_factory=_empty_long_tensor)
    timesteps: torch.Tensor = field(default_factory=_empty_float_tensor)
    mse_loss_indexes: torch.Tensor = field(default_factory=_empty_long_tensor)
    spans: list[ModalitySpan] = field(default_factory=list)
    # list[tuple[int,int,int]] for vision, list[tuple[int]] for action, list[tuple[int,int,int]] for sound
    token_shapes: list[tuple[int, ...]] = field(default_factory=list)
    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    domain_id: list[torch.Tensor] = field(default_factory=list)
    raw_action_dim: list[torch.Tensor | None] | None = field(default_factory=list)

    def __post_init__(self) -> None:
        assert isinstance(self.sequence_indexes, torch.Tensor), "ModalityData.sequence_indexes must be finalized"
        assert isinstance(self.timesteps, torch.Tensor), "ModalityData.timesteps must be finalized"
        assert isinstance(self.mse_loss_indexes, torch.Tensor), "ModalityData.mse_loss_indexes must be finalized"

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        self.sequence_indexes = self.sequence_indexes.cuda()
        self.timesteps = self.timesteps.cuda()
        self.mse_loss_indexes = self.mse_loss_indexes.cuda()
        self.tokens = [token.cuda() for token in self.tokens]
        self.condition_mask = [cm.cuda() for cm in self.condition_mask]
        self.noisy_frame_indexes = [ni.cuda() for ni in self.noisy_frame_indexes]
        self.domain_id = [d.cuda() for d in self.domain_id]
        # raw_action_dim is optional (e.g., when action-channel masking is disabled).
        if self.raw_action_dim is not None:
            self.raw_action_dim = [d.cuda() if d is not None else None for d in self.raw_action_dim]


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """Prepare dense attention mask for a single sample with multiple splits.

    Args:
        split_lens: List of integers indicating length of each split within the sample
        attn_modes: List of attention modes for each split ('causal', 'full', or 'noise')
        device: Device to place the attention mask tensor on

    Returns:
        Attention mask tensor of shape (sample_len, sample_len) with -inf for masked positions
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)  # [sample_len,sample_len]

    # First pass: Set up basic attention patterns for each split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"], f"Invalid attention mode: {attn_mode}"

        split_start = current_pos
        split_end = current_pos + split_len

        if attn_mode == "causal":
            # Causal: lower triangular within split + full attention to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            ).tril()  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1
        else:  # "full" or "noise"
            # Full attention within split and to previous splits
            attention_mask[split_start:split_end, split_start:split_end] = torch.ones(
                (split_len, split_len), device=device
            )  # [split_len,split_len]
            attention_mask[split_start:split_end, :split_start] = 1

        current_pos += split_len

    # Second pass: Handle noise mode - mask out noise columns except within same split
    current_pos = 0
    for split_len, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            split_start = current_pos
            split_end = current_pos + split_len

            # Zero out the entire column for noise tokens
            attention_mask[:, split_start:split_end] = 0
            # But allow self-attention within the noise split
            attention_mask[split_start:split_end, split_start:split_end] = 1

        current_pos += split_len

    # Convert boolean mask to float with -inf for masked positions
    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )  # [sample_len,sample_len]

    return attention_mask


# ============================================================================
# Tokenizer utilities
# ============================================================================


def add_special_tokens(tokenizer):
    """Add image-related special tokens to tokenizer if not already present.

    Args:
        tokenizer: Tokenizer to add special tokens to

    Returns:
        Tuple of (modified tokenizer, dict of new token IDs)
    """
    # Collect existing special tokens
    existing_special_tokens = []
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            existing_special_tokens.append(value)
        elif isinstance(value, list):
            existing_special_tokens.extend(value)

    # Define image boundary tokens to add if missing
    tokens_to_add = []
    if "<|vision_start|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_start|>")
    if "<|vision_end|>" not in existing_special_tokens:
        tokens_to_add.append("<|vision_end|>")

    # Add new tokens to tokenizer vocabulary
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)

    # Get token IDs for image boundary tokens
    new_token_ids = {
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }

    return tokenizer, new_token_ids


def compute_text_split_length(
    num_caption_tokens: int,
    special_tokens: dict[str, int],
    has_generation: bool = True,
) -> int:
    """Compute the total text split length without mutating any state.

    This is the number of token positions occupied by the text split in a
    packed sequence: caption tokens + optional BOS + EOS + optional BOV.

    Args:
        num_caption_tokens: Number of raw caption token IDs (before special tokens).
        special_tokens: Dictionary of special token IDs (checked for ``"bos_token_id"``).
        has_generation: Whether a start-of-generation (BOV) token follows text.

    Returns:
        Total text split length (positions consumed in the packed sequence).
    """
    n = num_caption_tokens
    if "bos_token_id" in special_tokens:
        n += 1
    n += 1  # EOS
    if has_generation:
        n += 1  # start-of-generation / BOV
    return n
