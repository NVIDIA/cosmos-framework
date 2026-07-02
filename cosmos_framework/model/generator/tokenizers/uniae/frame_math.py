# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared UniAE noncausal temporal chunking math."""

from collections.abc import Iterable, Mapping

from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

DEFAULT_RESOLUTION_KEYS = ("256", "480")


def normalize_resolution_int_mapping(
    value: int | Mapping[str, int],
    *,
    name: str,
    default_keys: Iterable[str] = DEFAULT_RESOLUTION_KEYS,
    required_keys: Iterable[str] | None = None,
) -> dict[str, int]:
    """Normalize a scalar or resolution-keyed integer config."""
    if isinstance(value, int):
        normalized = {str(resolution): int(value) for resolution in default_keys}
    elif isinstance(value, Mapping):
        normalized = {str(resolution): int(config_value) for resolution, config_value in value.items()}
    else:
        raise TypeError(f"{name} must be an int or a resolution-keyed mapping, got {type(value).__name__}.")

    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if required_keys is not None:
        missing = set(required_keys) - set(normalized)
        if missing:
            raise ValueError(f"{name} is missing resolution keys {sorted(missing)}.")
    return normalized


def normalize_uniae_chunk_frames(
    uniae_chunk_frames: int | Mapping[str, int] | None,
    *,
    pad_frames: int | None,
    temporal_compression_factor: int,
    missing_chunk_message: str = "uniae_chunk_frames must be provided when uniae_pad_frames is set",
    missing_pad_message: str = "uniae_pad_frames must be provided when uniae_chunk_frames is set",
    temporal_divisibility_name: str = "temporal_compression_factor",
) -> int | dict[str, int] | None:
    """Normalize and validate UniAE full chunk sizes."""
    if uniae_chunk_frames is None:
        if pad_frames is not None:
            raise ValueError(missing_chunk_message)
        return None

    if pad_frames is None:
        raise ValueError(missing_pad_message)
    if pad_frames <= 0:
        raise ValueError(f"uniae_pad_frames must be positive, got {pad_frames}.")

    if isinstance(uniae_chunk_frames, Mapping):
        normalized = {str(resolution): int(chunk_frames) for resolution, chunk_frames in uniae_chunk_frames.items()}
        if not normalized:
            raise ValueError("uniae_chunk_frames mapping must not be empty")
    else:
        normalized = int(uniae_chunk_frames)

    values = normalized.values() if isinstance(normalized, dict) else [normalized]
    for chunk_frames in values:
        if chunk_frames <= 2 * pad_frames:
            raise ValueError(
                f"uniae_chunk_frames must be greater than 2 * uniae_pad_frames, got {chunk_frames=} and {pad_frames=}."
            )
        if chunk_frames % temporal_compression_factor != 0:
            raise ValueError(
                f"uniae_chunk_frames must be divisible by {temporal_divisibility_name}, "
                f"got {chunk_frames=} and {temporal_compression_factor=}."
            )
    return normalized


def get_uniae_chunk_frames(
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
) -> int:
    """Select a scalar UniAE full chunk size from a scalar or resolution-keyed config."""
    if isinstance(uniae_chunk_frames, int):
        return uniae_chunk_frames

    if target_resolution_key is not None:
        resolved_resolution = target_resolution_key
    elif resolution is not None:
        resolved_resolution = resolution
    elif spatial_shape is not None:
        resolved_resolution = get_vision_data_resolution(spatial_shape)
    else:
        chunk_values = {int(chunk_frames) for chunk_frames in uniae_chunk_frames.values()}
        if len(chunk_values) == 1:
            return next(iter(chunk_values))
        raise ValueError(missing_resolution_message)

    if resolved_resolution not in uniae_chunk_frames:
        raise ValueError(
            f"Resolution {resolved_resolution!r} not found in uniae_chunk_frames. "
            f"Available resolutions: {list(uniae_chunk_frames.keys())}"
        )
    return int(uniae_chunk_frames[resolved_resolution])


def get_uniae_latent_num_frames(
    num_pixel_frames: int,
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    pad_frames: int,
    temporal_compression_factor: int,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
    invalid_frame_message_prefix: str = "Video frame count is not valid for UniAE non-causal chunking",
) -> int:
    """Return UniAE latent frame count for first-frame-alone plus padded-tail chunking."""
    if num_pixel_frames < 1:
        raise ValueError(f"num_pixel_frames must be positive, got {num_pixel_frames}.")
    if num_pixel_frames == 1:
        return 1

    full_chunk = get_uniae_chunk_frames(
        uniae_chunk_frames,
        resolution=resolution,
        spatial_shape=spatial_shape,
        target_resolution_key=target_resolution_key,
        missing_resolution_message=missing_resolution_message,
    )
    _validate_full_chunk(full_chunk, pad_frames=pad_frames, temporal_compression_factor=temporal_compression_factor)

    effective_chunk = full_chunk - 2 * pad_frames
    latents_per_full_chunk = full_chunk // temporal_compression_factor
    remaining_frames = num_pixel_frames - 1
    num_full_chunks = remaining_frames // effective_chunk
    tail_frames = remaining_frames % effective_chunk
    num_latent_frames = 1 + num_full_chunks * latents_per_full_chunk
    if tail_frames == 0:
        return num_latent_frames

    padded_tail_frames = tail_frames + 2 * pad_frames
    if padded_tail_frames % temporal_compression_factor != 0:
        raise ValueError(
            f"{invalid_frame_message_prefix}: "
            f"got {num_pixel_frames=}, {full_chunk=}, {pad_frames=}, {temporal_compression_factor=}."
        )
    return num_latent_frames + padded_tail_frames // temporal_compression_factor


def get_uniae_pixel_num_frames(
    num_latent_frames: int,
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    pad_frames: int,
    temporal_compression_factor: int,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
) -> int:
    """Return pixel frame count represented by a valid UniAE latent frame count."""
    if num_latent_frames < 1:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")
    if num_latent_frames == 1:
        return 1

    full_chunk = get_uniae_chunk_frames(
        uniae_chunk_frames,
        resolution=resolution,
        spatial_shape=spatial_shape,
        target_resolution_key=target_resolution_key,
        missing_resolution_message=missing_resolution_message,
    )
    _validate_full_chunk(full_chunk, pad_frames=pad_frames, temporal_compression_factor=temporal_compression_factor)

    effective_chunk = full_chunk - 2 * pad_frames
    latents_per_full_chunk = full_chunk // temporal_compression_factor
    remaining_latents = num_latent_frames - 1
    num_full_chunks = remaining_latents // latents_per_full_chunk
    tail_latents = remaining_latents % latents_per_full_chunk
    num_pixel_frames = 1 + num_full_chunks * effective_chunk
    if tail_latents == 0:
        return num_pixel_frames

    tail_frames = tail_latents * temporal_compression_factor - 2 * pad_frames
    if tail_frames <= 0:
        raise ValueError(
            "UniAE latent count does not map to a positive noncausal tail: "
            f"got {num_latent_frames=}, {full_chunk=}, {pad_frames=}, {temporal_compression_factor=}."
        )
    return num_pixel_frames + tail_frames


def get_uniae_latent_temporal_positions(
    num_pixel_frames: int,
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    pad_frames: int,
    temporal_compression_factor: int,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
    num_latent_frames: int | None = None,
) -> list[float]:
    """Return UniAE latent temporal coordinates in source-frame / tcf units."""
    if num_pixel_frames < 1:
        raise ValueError(f"num_pixel_frames must be positive, got {num_pixel_frames}.")
    if num_pixel_frames == 1:
        temporal_positions = [0.0]
    else:
        full_chunk = get_uniae_chunk_frames(
            uniae_chunk_frames,
            resolution=resolution,
            spatial_shape=spatial_shape,
            target_resolution_key=target_resolution_key,
            missing_resolution_message=missing_resolution_message,
        )
        _validate_full_chunk(full_chunk, pad_frames=pad_frames, temporal_compression_factor=temporal_compression_factor)

        effective_chunk = full_chunk - 2 * pad_frames
        temporal_positions = [0.0]
        source_start = 1
        while source_start < num_pixel_frames:
            source_end = min(source_start + effective_chunk, num_pixel_frames)
            chunk_source_frames = (
                [source_start] * pad_frames + list(range(source_start, source_end)) + [source_end - 1] * pad_frames
            )
            if len(chunk_source_frames) % temporal_compression_factor != 0:
                raise ValueError(
                    "UniAE frame count is not valid for noncausal chunking: "
                    f"got {num_pixel_frames=}, {full_chunk=}, {pad_frames=}, {temporal_compression_factor=}."
                )
            temporal_positions.extend(
                chunk_source_frames[i + temporal_compression_factor - 1] / temporal_compression_factor
                for i in range(0, len(chunk_source_frames), temporal_compression_factor)
            )
            source_start = source_end

    expected_latent_frames = get_uniae_latent_num_frames(
        num_pixel_frames,
        uniae_chunk_frames,
        pad_frames=pad_frames,
        temporal_compression_factor=temporal_compression_factor,
        resolution=resolution,
        spatial_shape=spatial_shape,
        target_resolution_key=target_resolution_key,
        missing_resolution_message=missing_resolution_message,
        invalid_frame_message_prefix="UniAE frame count is not valid for noncausal chunking",
    )
    if num_latent_frames is not None and num_latent_frames != expected_latent_frames:
        raise ValueError(
            "UniAE latent temporal position count does not match encoded latent frames: "
            f"got {num_latent_frames=}, expected {expected_latent_frames} for {num_pixel_frames=}."
        )
    if len(temporal_positions) != expected_latent_frames:
        raise ValueError(
            "UniAE latent temporal position helper produced an inconsistent count: "
            f"got {len(temporal_positions)}, expected {expected_latent_frames}."
        )
    return temporal_positions


def align_uniae_num_video_frames(
    num_video_frames: int,
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    pad_frames: int,
    temporal_compression_factor: int,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
) -> int:
    """Trim a video frame count down to the nearest valid UniAE noncausal count."""
    if num_video_frames < 1:
        return 0

    full_chunk = get_uniae_chunk_frames(
        uniae_chunk_frames,
        resolution=resolution,
        spatial_shape=spatial_shape,
        target_resolution_key=target_resolution_key,
        missing_resolution_message=missing_resolution_message,
    )
    _validate_full_chunk(full_chunk, pad_frames=pad_frames, temporal_compression_factor=temporal_compression_factor)

    effective_chunk = full_chunk - 2 * pad_frames
    target_r = (-2 * pad_frames) % temporal_compression_factor
    remainder = (num_video_frames - 1) % effective_chunk
    if remainder != 0 and remainder % temporal_compression_factor != target_r:
        delta = (remainder - target_r) % temporal_compression_factor
        if remainder - delta < 0:
            delta = remainder
        num_video_frames -= delta
    return num_video_frames


def ceil_uniae_num_video_frames(
    num_video_frames: int,
    uniae_chunk_frames: int | Mapping[str, int],
    *,
    pad_frames: int,
    temporal_compression_factor: int,
    resolution: str | None = None,
    spatial_shape: tuple[int, int] | None = None,
    target_resolution_key: str | None = None,
    missing_resolution_message: str = (
        "spatial_shape or target resolution must be provided for resolution-keyed UniAE chunks"
    ),
) -> int:
    """Round up to the nearest valid UniAE noncausal count, preserving valid partial tails."""
    if num_video_frames < 1:
        return 0

    for candidate in range(num_video_frames, num_video_frames + temporal_compression_factor + 1):
        aligned_candidate = align_uniae_num_video_frames(
            candidate,
            uniae_chunk_frames,
            pad_frames=pad_frames,
            temporal_compression_factor=temporal_compression_factor,
            resolution=resolution,
            spatial_shape=spatial_shape,
            target_resolution_key=target_resolution_key,
            missing_resolution_message=missing_resolution_message,
        )
        if aligned_candidate == candidate:
            return candidate

    raise RuntimeError(
        "Failed to find a valid UniAE frame count within one temporal-compression window: "
        f"{num_video_frames=}, {uniae_chunk_frames=}, {pad_frames=}, {temporal_compression_factor=}, "
        f"{resolution=}, {spatial_shape=}, {target_resolution_key=}."
    )


def _validate_full_chunk(
    full_chunk: int,
    *,
    pad_frames: int,
    temporal_compression_factor: int,
) -> None:
    if full_chunk % temporal_compression_factor != 0:
        raise ValueError(
            "full_chunk must be divisible by temporal compression factor, "
            f"got {full_chunk=} and {temporal_compression_factor=}."
        )
    if full_chunk <= 2 * pad_frames:
        raise ValueError(f"full_chunk must be greater than 2 * pad_frames, got {full_chunk=} and {pad_frames=}.")
