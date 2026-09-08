# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge-compatible multimodal RoPE helpers for the tokenizer text decoder."""

from __future__ import annotations

import torch

MROPE_AXIS_COUNT = 3
NEMOTRON_2B_MROPE_SECTION = (24, 20, 20)


def _validate_integer_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Require a non-boolean integer tensor."""
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}.")


def _assert_tensor_condition(condition: torch.Tensor, *, message: str) -> None:  # condition: []
    """Validate one device scalar without synchronizing the CUDA host thread."""
    if condition.numel() != 1:
        raise ValueError(f"Tensor condition must contain one element, got shape {tuple(condition.shape)}.")
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise ValueError(message)


def build_multimodal_rope_position_ids(
    *,
    input_ids: torch.Tensor,
    image_patch_indices: torch.Tensor,
    image_coords: torch.Tensor,
    segment_ids: torch.Tensor | None,
    pad_token_id: int | None,
    vision_token_id: int | tuple[int, ...],
) -> torch.Tensor:
    """Build Edge-compatible ``(T,H,W)`` position IDs from realized visual blocks.

    Text tokens advance monotonically on all three axes. Each contiguous run of
    vision placeholders receives its local visual coordinates, offset by the
    preceding text cursor. The cursor after a visual block advances by the
    largest visual-axis extent, matching ``Cosmos3EdgeModel.get_rope_index``.

    Args:
        input_ids: Token IDs with shape ``[B,S]``.
        image_patch_indices: Flat indices into ``input_ids`` with shape ``[N]``.
        image_coords: Pooled feature coordinates with shape ``[N,4]`` as
            ``(T,H,W,Z)`` or ``[N,5]`` as ``(segment,T,H,W,Z)``.
        segment_ids: Optional packed segment IDs with shape ``[B,S]``. Values
            below zero mark padding. When omitted, each row is one right-padded
            logical segment.
        pad_token_id: Padding token used when ``segment_ids`` is omitted.
        vision_token_id: One token ID or a tuple of allowed distinct-media
            token IDs required at every visual patch index.

    Returns:
        Position IDs with shape ``[3,B,S]``. Padding positions contain one,
        matching the canonical Edge/Qwen convention.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B,S], got {tuple(input_ids.shape)}.")
    _validate_integer_tensor(input_ids, name="input_ids")
    if image_patch_indices.ndim != 1:
        raise ValueError(f"image_patch_indices must have shape [N], got {tuple(image_patch_indices.shape)}.")
    _validate_integer_tensor(image_patch_indices, name="image_patch_indices")
    if image_coords.ndim != 2 or image_coords.shape[1] not in (4, 5):
        raise ValueError(f"image_coords must have shape [N,4] or [N,5], got {tuple(image_coords.shape)}.")
    _validate_integer_tensor(image_coords, name="image_coords")
    if image_patch_indices.numel() != image_coords.shape[0]:
        raise ValueError(
            f"Image patch index/coordinate count mismatch: {image_patch_indices.numel()} != {image_coords.shape[0]}."
        )

    batch_size, sequence_length = input_ids.shape
    device = input_ids.device
    if segment_ids is None:
        valid_mask = (
            torch.ones_like(input_ids, dtype=torch.bool) if pad_token_id is None else input_ids != pad_token_id
        )  # [B,S]
        logical_segment_ids = torch.where(
            valid_mask,
            torch.zeros_like(input_ids, dtype=torch.long),
            torch.full_like(input_ids, -1, dtype=torch.long),
        )  # [B,S]
    else:
        if segment_ids.shape != input_ids.shape:
            raise ValueError(
                f"segment_ids must match input_ids shape {tuple(input_ids.shape)}, got {tuple(segment_ids.shape)}."
            )
        _validate_integer_tensor(segment_ids, name="segment_ids")
        logical_segment_ids = segment_ids.to(device=device, dtype=torch.long)  # [B,S]
        valid_mask = logical_segment_ids >= 0  # [B,S]

    valid_token_counts = valid_mask.sum(dim=1)  # [B]
    expected_valid_mask = torch.arange(sequence_length, device=device).unsqueeze(0) < valid_token_counts.unsqueeze(
        1
    )  # [B,S]
    _assert_tensor_condition(
        torch.all(valid_mask == expected_valid_mask),  # []
        message="Valid text tokens must form one contiguous prefix per batch row.",
    )

    flat_patch_indices = image_patch_indices.to(device=device, dtype=torch.long)  # [N]
    if flat_patch_indices.numel() > 0:
        _assert_tensor_condition(
            torch.all((flat_patch_indices >= 0) & (flat_patch_indices < input_ids.numel())),  # []
            message=f"image_patch_indices must be in range [0, {input_ids.numel()}).",
        )
        if torch.unique(flat_patch_indices).numel() != flat_patch_indices.numel():
            raise ValueError("image_patch_indices must not contain duplicates.")
        flat_input_ids = input_ids.reshape(-1)  # [B*S]
        indexed_token_ids = flat_input_ids.index_select(0, flat_patch_indices)  # [N]
        vision_token_ids = (vision_token_id,) if isinstance(vision_token_id, int) else vision_token_id
        if not vision_token_ids:
            raise ValueError("At least one vision placeholder token ID is required.")
        indexed_vision_mask = torch.zeros_like(indexed_token_ids, dtype=torch.bool)  # [N]
        for token_id in vision_token_ids:
            indexed_vision_mask |= indexed_token_ids == token_id  # [N]
        _assert_tensor_condition(
            torch.all(indexed_vision_mask),  # []
            message="Every image patch index must target a vision placeholder token.",
        )

    position_ids = torch.ones(
        MROPE_AXIS_COUNT,
        batch_size,
        sequence_length,
        device=device,
        dtype=torch.long,
    )  # [3,B,S]
    flat_valid_mask = valid_mask.reshape(-1)  # [B*S]
    valid_flat_indices = torch.nonzero(flat_valid_mask, as_tuple=True)[0]  # [V]
    if valid_flat_indices.numel() == 0:
        if flat_patch_indices.numel() > 0:
            raise ValueError("Image patch indices cannot target an all-padding batch.")
        return position_ids

    flat_segment_ids = logical_segment_ids.reshape(-1)  # [B*S]
    valid_segment_ids = flat_segment_ids.index_select(0, valid_flat_indices)  # [V]
    valid_batch_indices = torch.div(valid_flat_indices, sequence_length, rounding_mode="floor")  # [V]
    group_start_mask = torch.ones(valid_flat_indices.shape[0], device=device, dtype=torch.bool)  # [V]
    group_start_mask[1:] = (
        (valid_batch_indices[1:] != valid_batch_indices[:-1])
        | (valid_segment_ids[1:] != valid_segment_ids[:-1])
        | (valid_flat_indices[1:] != valid_flat_indices[:-1] + 1)
    )  # [V-1]
    group_starts = torch.nonzero(group_start_mask, as_tuple=True)[0]  # [R]
    group_ends = torch.cat(
        [
            group_starts[1:],
            torch.tensor([valid_flat_indices.shape[0]], device=device, dtype=torch.long),
        ],
        dim=0,
    )  # [R]
    group_lengths = group_ends - group_starts  # [R]
    group_ids = torch.cumsum(group_start_mask.to(dtype=torch.long), dim=0) - 1  # [V]
    group_pairs = torch.stack(
        [
            valid_batch_indices.index_select(0, group_starts),
            valid_segment_ids.index_select(0, group_starts),
        ],
        dim=1,
    )  # [R,2]
    if torch.unique(group_pairs, dim=0).shape[0] != group_pairs.shape[0]:
        raise ValueError("Packed segment IDs must form exactly one contiguous run per logical segment.")

    valid_offsets = torch.arange(valid_flat_indices.shape[0], device=device, dtype=torch.long)  # [V]
    local_sequence_positions = valid_offsets - torch.repeat_interleave(group_starts, group_lengths)  # [V]
    scalar_position_ids = local_sequence_positions.clone()  # [V]

    if flat_patch_indices.numel() == 0:
        valid_position_ids = scalar_position_ids.unsqueeze(0).expand(MROPE_AXIS_COUNT, -1)  # [3,V]
        flat_position_ids = position_ids.reshape(MROPE_AXIS_COUNT, -1)  # [3,B*S]
        flat_position_ids[:, valid_flat_indices] = valid_position_ids  # [3,V]
        return position_ids

    flat_to_valid_offset = torch.full(
        (input_ids.numel(),),
        -1,
        device=device,
        dtype=torch.long,
    )  # [B*S]
    flat_to_valid_offset[valid_flat_indices] = valid_offsets  # [V]
    visual_valid_offsets = flat_to_valid_offset.index_select(0, flat_patch_indices)  # [N]
    _assert_tensor_condition(
        torch.all(visual_valid_offsets >= 0),  # []
        message="Image patch indices must target valid, non-padding segments.",
    )

    spatial_start_column = 1 if image_coords.shape[1] == 5 else 0
    visual_coords = image_coords[:, spatial_start_column : spatial_start_column + 3].to(
        device=device,
        dtype=torch.long,
    )  # [N,3]
    visual_sort_order = torch.argsort(visual_valid_offsets)  # [N]
    visual_valid_offsets = visual_valid_offsets.index_select(0, visual_sort_order)  # [N]
    visual_coords = visual_coords.index_select(0, visual_sort_order)  # [N,3]
    visual_group_ids = group_ids.index_select(0, visual_valid_offsets)  # [N]
    visual_local_positions = local_sequence_positions.index_select(0, visual_valid_offsets)  # [N]

    block_start_mask = torch.ones(visual_valid_offsets.shape[0], device=device, dtype=torch.bool)  # [N]
    block_start_mask[1:] = (visual_group_ids[1:] != visual_group_ids[:-1]) | (
        visual_local_positions[1:] != visual_local_positions[:-1] + 1
    )  # [N-1]
    block_starts = torch.nonzero(block_start_mask, as_tuple=True)[0]  # [K]
    block_ends = torch.cat(
        [
            block_starts[1:],
            torch.tensor([visual_valid_offsets.shape[0]], device=device, dtype=torch.long),
        ],
        dim=0,
    )  # [K]
    block_lengths = block_ends - block_starts  # [K]
    block_ids = torch.cumsum(block_start_mask.to(dtype=torch.long), dim=0) - 1  # [N]
    block_coord_indices = block_ids.unsqueeze(1).expand(-1, MROPE_AXIS_COUNT)  # [N,3]
    minimum_coords = torch.full(
        (block_starts.shape[0], MROPE_AXIS_COUNT),
        torch.iinfo(torch.long).max,
        device=device,
        dtype=torch.long,
    )  # [K,3]
    minimum_coords.scatter_reduce_(
        0,
        block_coord_indices,
        visual_coords,
        reduce="amin",
        include_self=True,
    )  # [K,3]
    maximum_coords = torch.full(
        (block_starts.shape[0], MROPE_AXIS_COUNT),
        torch.iinfo(torch.long).min,
        device=device,
        dtype=torch.long,
    )  # [K,3]
    maximum_coords.scatter_reduce_(
        0,
        block_coord_indices,
        visual_coords,
        reduce="amax",
        include_self=True,
    )  # [K,3]
    local_visual_coords = visual_coords - minimum_coords.index_select(0, block_ids)  # [N,3]
    axis_extents = maximum_coords - minimum_coords + 1  # [K,3]
    # Avoid the CUDA int64 reduction kernel: torch.prod over multiple rows can
    # fail on the pinned GB200 runtime, while three elementwise factors are exact.
    dense_grid_token_counts = axis_extents[:, 0] * axis_extents[:, 1] * axis_extents[:, 2]  # [K]
    _assert_tensor_condition(
        torch.all(dense_grid_token_counts == block_lengths),  # []
        message="Each visual placeholder block must describe one dense rectangular (T,H,W) grid.",
    )
    unique_block_coords = torch.unique(
        torch.cat([block_ids.unsqueeze(1), local_visual_coords], dim=1),
        dim=0,
    )  # [N_unique,4]
    if unique_block_coords.shape[0] != visual_coords.shape[0]:
        raise ValueError("Visual coordinates must be unique within each placeholder block.")

    block_position_extents = axis_extents.amax(dim=1)  # [K]
    block_compressions = block_lengths - block_position_extents  # [K]
    compression_events = torch.zeros(
        valid_flat_indices.shape[0] + 1,
        device=device,
        dtype=torch.long,
    )  # [V+1]
    block_last_valid_offsets = visual_valid_offsets.index_select(0, block_ends - 1)  # [K]
    compression_events.scatter_add_(0, block_last_valid_offsets + 1, block_compressions)  # [V+1]
    global_compression = torch.cumsum(compression_events, dim=0)[:-1]  # [V]
    group_initial_compression = global_compression.index_select(0, group_starts)  # [R]
    group_local_compression = global_compression - group_initial_compression.index_select(0, group_ids)  # [V]
    scalar_position_ids = scalar_position_ids - group_local_compression  # [V]

    block_first_valid_offsets = visual_valid_offsets.index_select(0, block_starts)  # [K]
    block_bases = visual_local_positions.index_select(0, block_starts) - group_local_compression.index_select(
        0,
        block_first_valid_offsets,
    )  # [K]
    visual_position_ids = local_visual_coords.transpose(0, 1) + block_bases.index_select(
        0,
        block_ids,
    ).unsqueeze(0)  # [3,N]
    valid_position_ids = scalar_position_ids.unsqueeze(0).expand(MROPE_AXIS_COUNT, -1).clone()  # [3,V]
    valid_position_ids[:, visual_valid_offsets] = visual_position_ids  # [3,N]
    flat_position_ids = position_ids.reshape(MROPE_AXIS_COUNT, -1)  # [3,B*S]
    flat_position_ids[:, valid_flat_indices] = valid_position_ids  # [3,V]
    return position_ids


def build_nemotron_mrope_rotary_embeddings(
    *,
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    reference: torch.Tensor,
    mrope_section: tuple[int, int, int] = NEMOTRON_2B_MROPE_SECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build interleaved Nemotron mRoPE cosine and sine tensors.

    This matches Cosmos3 Edge's ``MultiModalRotaryEmbedding`` channel
    assignment. Frequency indices are interleaved T/H/W in groups of three,
    with the remaining high-frequency slots assigned to T.

    Args:
        inv_freq: Rotary inverse frequencies with shape ``[D/2]``.
        position_ids: Multimodal positions with shape ``[3,B,T]``.
        reference: Tensor supplying output device and dtype, ending in ``D``.
        mrope_section: Per-axis frequency counts. The counts must sum to ``D/2``.

    Returns:
        Cosine and sine tensors, each with shape ``[B,1,T,D]``.
    """
    if inv_freq.ndim != 1:
        raise ValueError(f"inv_freq must have shape [D/2], got {tuple(inv_freq.shape)}.")
    if position_ids.ndim != 3 or position_ids.shape[0] != MROPE_AXIS_COUNT:
        raise ValueError(f"position_ids must have shape [3,B,T], got {tuple(position_ids.shape)}.")
    if len(mrope_section) != MROPE_AXIS_COUNT:
        raise ValueError(f"mrope_section must contain three axis widths, got {mrope_section!r}.")
    if any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in mrope_section):
        raise ValueError(f"mrope_section widths must be positive integers, got {mrope_section!r}.")

    frequency_count = int(inv_freq.numel())
    if sum(mrope_section) != frequency_count:
        raise ValueError(
            f"mrope_section must sum to the rotary frequency count {frequency_count}, got {mrope_section!r}."
        )
    if reference.shape[-1] != frequency_count * 2:
        raise ValueError(f"reference last dimension must be {frequency_count * 2}, got {reference.shape[-1]}.")

    inv_freq_expanded = inv_freq.to(device=reference.device, dtype=torch.float32)[None, None, :, None].expand(
        MROPE_AXIS_COUNT,
        position_ids.shape[1],
        -1,
        1,
    )  # [3,B,D/2,1]
    position_ids_expanded = position_ids.to(device=reference.device, dtype=torch.float32)[:, :, None, :]  # [3,B,1,T]
    device_type = reference.device.type if reference.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        axis_frequencies = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(2, 3)  # [3,B,T,D/2]
        interleaved_frequencies = axis_frequencies[0].clone()  # [B,T,D/2]
        for axis_index, channel_offset in enumerate((1, 2), start=1):
            section_end = mrope_section[axis_index] * MROPE_AXIS_COUNT
            channel_slice = slice(channel_offset, section_end, MROPE_AXIS_COUNT)
            interleaved_frequencies[..., channel_slice] = axis_frequencies[
                axis_index,
                ...,
                channel_slice,
            ]  # [B,T,N_axis]

        doubled_frequencies = torch.cat(
            [interleaved_frequencies, interleaved_frequencies],
            dim=-1,
        )  # [B,T,D]
        doubled_frequencies = doubled_frequencies.unsqueeze(1)  # [B,1,T,D]
        cosine = doubled_frequencies.cos().to(dtype=reference.dtype)  # [B,1,T,D]
        sine = doubled_frequencies.sin().to(dtype=reference.dtype)  # [B,1,T,D]
    return cosine, sine
