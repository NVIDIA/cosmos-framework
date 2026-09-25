# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR range-map loading and tokenizer preprocessing."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import IO, Any, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.tokenizers.lidar.paths import is_remote_uri, s3_backend_args
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import (
    V0_TRANSFER_RANGE_PROJECTION,
    LidarRangeProjectionConfig,
)

# Native 3-channel tokenizer layout (range + intensity + mask @ 128x3600), i.e.
# no azimuth downsampling. Spatial compression there is asymmetric: H/8, W/16
# (patch 2x4 plus two 2x2 merges).
DEFAULT_RANGE_PROJECTION = LidarRangeProjectionConfig()
RANGE_HEIGHT = DEFAULT_RANGE_PROJECTION.native_height
RANGE_RAW_WIDTH = DEFAULT_RANGE_PROJECTION.native_width
RANGE_DOWNSAMPLED_WIDTH = V0_TRANSFER_RANGE_PROJECTION.semantic_width
TOKENIZER_WIDTH = V0_TRANSFER_RANGE_PROJECTION.model_width
# Shared metric span for V0 and V1: Drive-Dreams [5, 100] m.
MIN_RANGE_METERS = DEFAULT_RANGE_PROJECTION.min_range_m
MAX_RANGE_METERS = DEFAULT_RANGE_PROJECTION.max_range_m
# Network-space fill for a dropped ray after the [-1, 1] affine.
INVALID_NORMALIZED_VALUE = -1.0
# The tokenizer trained on 8 sampled sweeps plus the duplicated frame-0 prepend.
TOKENIZER_SAMPLE_FRAMES = 8

TarSource: TypeAlias = str | Path | bytes | bytearray | memoryview | IO[bytes]
ExpectedClipKey: TypeAlias = str | tuple[str, ...]


def _backend_args_or_default(backend_args: dict[str, Any] | None) -> dict[str, Any] | None:
    return backend_args if backend_args is not None else s3_backend_args()


def _open_tar_archive(
    tar_source: TarSource,
    *,
    backend_args: dict[str, Any] | None = None,
) -> tarfile.TarFile:
    """Open a local, remote, or in-memory LidarGEN clip tar."""
    if isinstance(tar_source, (bytes, bytearray, memoryview)):
        return tarfile.open(fileobj=io.BytesIO(tar_source), mode="r")
    if hasattr(tar_source, "read"):
        return tarfile.open(fileobj=tar_source, mode="r")
    path = str(tar_source)
    if is_remote_uri(path):
        from cosmos_framework.utils.easy_io import easy_io

        payload = easy_io.get(path, backend_args=_backend_args_or_default(backend_args))
        return tarfile.open(fileobj=io.BytesIO(payload), mode="r")
    return tarfile.open(path, "r")


def normalize_range_map(
    range_map: np.ndarray | torch.Tensor,
    *,
    min_range: float = MIN_RANGE_METERS,
    max_range: float = MAX_RANGE_METERS,
) -> np.ndarray | torch.Tensor:
    """Normalize metric ranges to ``[-1, 1]`` with invalid rays at ``-1``."""
    if max_range <= min_range:
        raise ValueError(f"max_range must exceed min_range, got {min_range=} and {max_range=}")
    if isinstance(range_map, torch.Tensor):
        clipped = range_map.clamp(min_range, max_range)  # [T,H,W]
    else:
        clipped = np.clip(range_map, min_range, max_range)
    unit = (clipped - min_range) / (max_range - min_range)  # [T,H,W]
    return unit * 2.0 - 1.0  # [T,H,W]


def normalize_intensity_values(intensities: np.ndarray) -> np.ndarray:
    """Normalize LiDAR return intensities to finite float32 values in ``[0, 1]``."""
    values = np.asarray(intensities)
    if not np.isfinite(values).all():
        raise ValueError("LiDAR intensity values must be finite")
    if values.size == 0:
        return values.astype(np.float32)
    if np.issubdtype(values.dtype, np.integer):
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < 0 or maximum > 255:
            raise ValueError(f"LiDAR integer intensity values must lie in [0,255], got [{minimum},{maximum}]")
        return values.astype(np.float32) / 255.0

    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    tolerance = 1e-3
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"LiDAR floating-point intensity values must lie in [0,1] within tolerance {tolerance}, "
            f"got [{minimum},{maximum}]"
        )
    return np.clip(values, 0.0, 1.0)


def network_range_to_metric(
    range_map: np.ndarray | torch.Tensor,
    *,
    min_range: float = MIN_RANGE_METERS,
    max_range: float = MAX_RANGE_METERS,
    clamp: bool = True,
) -> np.ndarray | torch.Tensor:
    """Invert :func:`normalize_range_map`: ``[-1, 1]`` back to meters, without masking.

    This is the only affine that maps a normalized range to meters; the callers
    that also drop rays layer their own mask on top, because they disagree about
    what a dropped ray becomes.

    ``clamp`` bounds the input to ``[-1, 1]`` first, which matters for a raw
    network prediction: without it an overshoot leaves the sensor's range
    envelope and reads as a return that the tokenizer never could have encoded.
    """
    if clamp:
        if isinstance(range_map, torch.Tensor):
            range_map = range_map.clamp(-1.0, 1.0)
        else:
            range_map = np.clip(range_map, -1.0, 1.0)
    return (range_map + 1.0) * 0.5 * (max_range - min_range) + min_range


def unit_intensity_to_network(intensity: torch.Tensor) -> torch.Tensor:
    """Map unit intensity ``[0, 1]`` to the network's ``[-1, 1]``."""
    return intensity.clamp(0.0, 1.0) * 2.0 - 1.0


def network_intensity_to_unit(normalized: torch.Tensor) -> torch.Tensor:
    """Invert :func:`unit_intensity_to_network`: ``[-1, 1]`` back to ``[0, 1]``."""
    return (normalized.clamp(-1.0, 1.0) + 1.0) * 0.5


def unnormalize_range_map(
    range_map: np.ndarray | torch.Tensor,
    *,
    min_range: float = MIN_RANGE_METERS,
    max_range: float = MAX_RANGE_METERS,
    valid_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Convert normalized ranges to meters and zero invalid rays.

    The input is not clamped: a caller that passes a raw prediction and no
    ``valid_mask`` relies on the derived mask below to drop an overshoot, rather
    than on the affine folding it back to the range envelope.
    """
    metric = network_range_to_metric(range_map, min_range=min_range, max_range=max_range, clamp=False)  # [T,H,W]
    if valid_mask is None:
        valid_mask = (metric > min_range) & (metric < max_range)  # [T,H,W]
    if isinstance(metric, torch.Tensor):
        metric = torch.where(valid_mask, metric, torch.zeros_like(metric))  # [T,H,W]
    else:
        metric = np.where(valid_mask, metric, np.zeros_like(metric))
    return metric, valid_mask


class RangeMapDownsampler:
    """Downsample a dense range map by retaining the nearest nonzero return."""

    def __init__(self, row_factor: int = 1, col_factor: int = 3) -> None:
        if row_factor < 1 or col_factor < 1:
            raise ValueError("Downsampling factors must be positive")
        self.row_factor = row_factor
        self.col_factor = col_factor

    @staticmethod
    def _minimum_nonzero(groups: np.ndarray, axis: int) -> np.ndarray:
        sentinel = np.finfo(groups.dtype).max
        nonzero = np.where(groups == 0, sentinel, groups)
        minimum = nonzero.min(axis=axis)
        return np.where(minimum == sentinel, 0, minimum)

    @staticmethod
    def _nearest_nonzero_range_and_value(
        range_groups: np.ndarray,
        value_groups: np.ndarray,
        *,
        axis: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sentinel = np.finfo(range_groups.dtype).max
        nonzero_range = np.where(range_groups == 0, sentinel, range_groups)
        nearest_index = nonzero_range.argmin(axis=axis, keepdims=True)
        nearest_range = np.take_along_axis(nonzero_range, nearest_index, axis=axis).squeeze(axis)
        nearest_value = np.take_along_axis(value_groups, nearest_index, axis=axis).squeeze(axis)
        valid = nearest_range != sentinel
        return np.where(valid, nearest_range, 0), np.where(valid, nearest_value, 0)

    def __call__(self, range_map: np.ndarray) -> np.ndarray:
        """Downsample ``[T,H,W]`` range maps."""
        if range_map.ndim != 3:
            raise ValueError(f"Expected [T,H,W], got shape {range_map.shape}")
        result = np.asarray(range_map, dtype=np.float32)
        frames, height, width = result.shape
        if height % self.row_factor:
            raise ValueError(f"Height {height} is not divisible by {self.row_factor}")
        if self.row_factor != 1:
            grouped_rows = result.reshape(
                frames,
                height // self.row_factor,
                self.row_factor,
                width,
            )
            result = self._minimum_nonzero(grouped_rows, axis=2)

        frames, height, width = result.shape
        if width % self.col_factor:
            raise ValueError(f"Width {width} is not divisible by {self.col_factor}")
        if self.col_factor != 1:
            grouped_cols = result.reshape(
                frames,
                height,
                width // self.col_factor,
                self.col_factor,
            )
            result = self._minimum_nonzero(grouped_cols, axis=3)
        return result

    def downsample_with_values(
        self,
        range_map: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample ranges and retain values from each nearest return."""
        if range_map.shape != values.shape or range_map.ndim != 3:
            raise ValueError(f"Expected matching [T,H,W] arrays, got {range_map.shape} and {values.shape}")
        ranges = np.asarray(range_map, dtype=np.float32)
        result_values = np.asarray(values, dtype=np.float32)
        frames, height, width = ranges.shape
        if height % self.row_factor or width % self.col_factor:
            raise ValueError(
                f"Shape {(height, width)} is not divisible by factors {(self.row_factor, self.col_factor)}"
            )
        if self.row_factor != 1:
            ranges, result_values = self._nearest_nonzero_range_and_value(
                ranges.reshape(frames, height // self.row_factor, self.row_factor, width),
                result_values.reshape(frames, height // self.row_factor, self.row_factor, width),
                axis=2,
            )
        frames, height, width = ranges.shape
        if self.col_factor != 1:
            ranges, result_values = self._nearest_nonzero_range_and_value(
                ranges.reshape(frames, height, width // self.col_factor, self.col_factor),
                result_values.reshape(frames, height, width // self.col_factor, self.col_factor),
                axis=3,
            )
        return ranges, result_values


def validate_lidar_layout(*, semantic_width: int, model_width: int) -> tuple[int, int]:
    """Validate semantic/model azimuth widths and return their integer values."""
    semantic_width = int(semantic_width)
    model_width = int(model_width)
    if semantic_width < 1 or RANGE_RAW_WIDTH % semantic_width:
        raise ValueError(
            f"semantic_width must be a positive divisor of native width {RANGE_RAW_WIDTH}, got {semantic_width}"
        )
    if model_width < semantic_width:
        raise ValueError(f"model_width must be at least semantic_width, got {model_width} < {semantic_width}")
    if (model_width - semantic_width) % 2:
        raise ValueError(
            "model_width - semantic_width must be even for symmetric circular padding, "
            f"got {model_width} - {semantic_width}"
        )
    return semantic_width, model_width


def range_video_layout_for_network_width(network_width: int) -> tuple[int, int | None]:
    """``(prepare_width, resize_width)`` for a network reading ``network_width`` columns.

    Prefer :data:`V0_TRANSFER_RANGE_PROJECTION` / :data:`V1_TRANSFER_RANGE_PROJECTION`
    at call sites. This helper only reconstructs that V0 pool-then-resize layout from
    a bare network width, for callers that have not yet loaded a projection object.
    """
    width = int(network_width)
    if width < 1:
        raise ValueError(f"network_width must be positive, got {network_width}")
    if width == V0_TRANSFER_RANGE_PROJECTION.model_width:
        return V0_TRANSFER_RANGE_PROJECTION.semantic_width, width
    if RANGE_RAW_WIDTH % width == 0:
        return width, None
    raise ValueError(
        f"No pooling layout for network width {width}: it does not divide native "
        f"{RANGE_RAW_WIDTH} and is not the V0 transfer width "
        f"{V0_TRANSFER_RANGE_PROJECTION.model_width}"
    )


def circular_pad_range_video_3ch(video: torch.Tensor, *, model_width: int) -> torch.Tensor:
    """Symmetrically circular-pad ``[..., W]`` from semantic to model width."""
    if video.ndim < 1:
        raise ValueError("Expected a tensor with a width dimension")
    semantic_width, model_width = validate_lidar_layout(
        semantic_width=int(video.shape[-1]),
        model_width=model_width,
    )
    pad_each = (model_width - semantic_width) // 2
    if pad_each == 0:
        return video
    if pad_each > semantic_width:
        raise ValueError(f"Circular padding per side cannot exceed semantic width, got {pad_each} > {semantic_width}")
    return torch.cat((video[..., -pad_each:], video, video[..., :pad_each]), dim=-1)


def crop_range_video_width(video: torch.Tensor, *, semantic_width: int) -> torch.Tensor:
    """Centrally crop a circular-padded ``[..., W_model]`` tensor."""
    if video.ndim < 1:
        raise ValueError("Expected a tensor with a width dimension")
    semantic_width, model_width = validate_lidar_layout(
        semantic_width=semantic_width,
        model_width=int(video.shape[-1]),
    )
    crop_each = (model_width - semantic_width) // 2
    if crop_each == 0:
        return video
    return video[..., crop_each : crop_each + semantic_width]


def resize_range_video_width(video: torch.Tensor, *, width: int) -> torch.Tensor:
    """Nearest-resize every channel of a ``[C,T,H,W]`` clip along azimuth."""
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(video.shape)}")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    if video.shape[-1] == width:
        return video
    frames = video.permute(1, 0, 2, 3)  # [T,C,H,W]
    frames = F.interpolate(frames, size=(video.shape[-2], width), mode="nearest-exact")  # [T,C,H,W_out]
    return frames.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W_out]


def prepare_lidar_video(
    metric_range_maps: np.ndarray,
    *,
    prepend_first_frame: bool = True,
    output_width: int = TOKENIZER_WIDTH,
) -> torch.Tensor:
    """Prepare raw range maps as tokenizer input ``[1,T,128,1024]``."""
    if metric_range_maps.ndim != 3 or metric_range_maps.shape[1:] != (RANGE_HEIGHT, RANGE_RAW_WIDTH):
        raise ValueError(f"Expected [T,{RANGE_HEIGHT},{RANGE_RAW_WIDTH}], got {metric_range_maps.shape}")
    downsampled = RangeMapDownsampler()(metric_range_maps)
    normalized = normalize_range_map(downsampled)
    video = torch.from_numpy(np.asarray(normalized, dtype=np.float32))[:, None]  # [T,1,H,W]
    if prepend_first_frame:
        video = torch.cat((video[:1], video), dim=0)  # [T+1,1,H,W]
    if video.shape[-1] != output_width:
        video = F.interpolate(video, size=(RANGE_HEIGHT, output_width), mode="nearest-exact")  # [T,1,H,W_out]
    return video.permute(1, 0, 2, 3).contiguous()  # [1,T,H,W]


def prepare_lidar_video_3ch(
    metric_range_maps: np.ndarray,
    intensity_maps: np.ndarray | None,
    *,
    prepend_first_frame: bool = True,
    range_projection: LidarRangeProjectionConfig = DEFAULT_RANGE_PROJECTION,
) -> torch.Tensor:
    """Prepare metric range, unit intensity, and mask as ``[3,T,128,W_model]``.

    ``metric_range_maps`` and ``intensity_maps`` are dense ``[T,128,3600]`` arrays
    (intensity already in ``[0,1]`` on returns, zeros elsewhere). Native rays are
    nearest-return pooled to ``semantic_width`` with matched intensity, then
    packed to ``model_width`` by circular padding or resize. Range remains in metres,
    intensity remains in ``[0,1]``, and mask is ``{0,1}``. Configured range
    filtering and network normalization belong to the tokenizer model.

    ``intensity_maps=None`` marks a modality that carries no return intensity, such
    as an HD-map rangemap. Its occupied rays take unit intensity rather than zero:
    the shared normalization maps zero onto the same value it fills invalid rays
    with, which would leave the control's populated rays indistinguishable from
    empty sky on the intensity channel.
    """
    semantic_width = range_projection.semantic_width
    expected_shape = (range_projection.native_height, range_projection.native_width)
    if metric_range_maps.ndim != 3 or metric_range_maps.shape[1:] != expected_shape:
        raise ValueError(f"Expected range [T,{expected_shape[0]},{expected_shape[1]}], got {metric_range_maps.shape}")
    if intensity_maps is not None and intensity_maps.shape != metric_range_maps.shape:
        raise ValueError(f"Intensity maps must match range shape {metric_range_maps.shape}, got {intensity_maps.shape}")
    col_factor = range_projection.column_pool_factor
    if col_factor == 1:
        down_range = np.asarray(metric_range_maps, dtype=np.float32)
        down_intensity = None if intensity_maps is None else np.asarray(intensity_maps, dtype=np.float32)
    elif intensity_maps is None:
        # A constant intensity holds no nearest-return information for the pooler to
        # carry, so pool range alone and fill the surviving rays below.
        down_range = RangeMapDownsampler(col_factor=col_factor)(metric_range_maps)
        down_intensity = None
    else:
        downsampler = RangeMapDownsampler(col_factor=col_factor)
        down_range, down_intensity = downsampler.downsample_with_values(metric_range_maps, intensity_maps)
    if down_range.shape[-1] != semantic_width:
        raise ValueError(f"Expected semantic width {semantic_width}, got {down_range.shape[-1]}")

    valid = down_range > 0.0  # [T,H,W]
    metric_range = np.where(valid, down_range, 0.0).astype(np.float32)
    if down_intensity is None:
        intensity_unit = valid.astype(np.float32)
    else:
        intensity_unit = np.clip(down_intensity.astype(np.float32), 0.0, 1.0)
        intensity_unit = np.where(valid, intensity_unit, 0.0).astype(np.float32)
    mask = valid.astype(np.float32)

    stacked = np.stack([metric_range, intensity_unit, mask], axis=1)  # [T,3,H,W]
    video = torch.from_numpy(stacked)
    if prepend_first_frame:
        video = torch.cat((video[:1], video), dim=0)  # [T+1,3,H,W]
    video = video.permute(1, 0, 2, 3).contiguous()  # [3,T,H,W_semantic]
    return apply_model_width_transform(video, range_projection=range_projection).contiguous()


def apply_model_width_transform(
    video: torch.Tensor,  # [C,T,H,W_semantic]
    *,
    range_projection: LidarRangeProjectionConfig,
) -> torch.Tensor:
    """Pack a semantic-width clip onto the tokenizer's model-width canvas."""
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(video.shape)}")
    if video.shape[-1] != range_projection.semantic_width:
        raise ValueError(f"Expected semantic width {range_projection.semantic_width}, got {video.shape[-1]}")
    if range_projection.model_width_transform == "circular_pad":
        return circular_pad_range_video_3ch(video, model_width=range_projection.model_width)
    if range_projection.model_width_transform == "resize":
        return resize_range_video_width(video, width=range_projection.model_width)
    raise ValueError(f"Unsupported model width transform {range_projection.model_width_transform!r}")


def undo_model_width_transform(
    video: torch.Tensor,  # [C,T,H,W_model]
    *,
    range_projection: LidarRangeProjectionConfig,
) -> torch.Tensor:  # [C,T,H,W_semantic]
    """Undo model-width packing so eval artifacts sit on the physical grid.

    ``circular_pad`` is a center crop back to ``semantic_width`` (a no-op when the
    two widths already match, as in transfer V1 at 1800). ``resize`` is a nearest
    resize from ``model_width`` back to ``semantic_width`` (V0: 1024 to 1200).
    LiDARBench and Hyperion score on that physical grid, not the VAE canvas.
    """
    if video.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W], got {tuple(video.shape)}")
    if range_projection.model_width_transform == "circular_pad":
        return crop_range_video_width(video, semantic_width=range_projection.semantic_width)
    if range_projection.model_width_transform == "resize":
        if video.shape[-1] != range_projection.model_width:
            raise ValueError(
                f"Expected model width {range_projection.model_width} for resize artifacts, got {video.shape[-1]}"
            )
        return resize_range_video_width(video, width=range_projection.semantic_width)
    raise ValueError(f"Unsupported model width transform {range_projection.model_width_transform!r}")


def rasterize_sparse_range_frame(
    rows: np.ndarray,
    cols: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """Rasterize sparse LiDAR returns into a dense ``[128,3600]`` range map."""
    if rows.shape != cols.shape or rows.shape != ranges.shape:
        raise ValueError(
            f"Sparse range-map arrays must share one length: rows={rows.shape}, cols={cols.shape}, ranges={ranges.shape}"
        )
    row_index = rows.astype(np.int64)
    col_index = cols.astype(np.int64)
    if row_index.size and (row_index.min() < 0 or row_index.max() >= RANGE_HEIGHT):
        raise ValueError(f"Range-map rows must lie in [0, {RANGE_HEIGHT}), got max {row_index.max()}")
    if col_index.size and (col_index.min() < 0 or col_index.max() >= RANGE_RAW_WIDTH):
        raise ValueError(f"Range-map columns must lie in [0, {RANGE_RAW_WIDTH}), got max {col_index.max()}")
    dense = np.zeros((RANGE_HEIGHT, RANGE_RAW_WIDTH), dtype=np.float32)  # [H,W]
    dense[row_index, col_index] = ranges.astype(np.float32)  # [H,W]
    return dense


def _load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "arr_0" not in payload:
            raise KeyError(f"Expected key 'arr_0' in {path}")
        return np.asarray(payload["arr_0"])


def _load_tar_npz_array(archive: tarfile.TarFile, member_name: str) -> np.ndarray:
    member = archive.extractfile(member_name)
    if member is None:
        raise ValueError(f"Could not read tar member {member_name!r}")
    with np.load(io.BytesIO(member.read()), allow_pickle=False) as payload:
        if "arr_0" not in payload:
            raise KeyError(f"Expected key 'arr_0' in tar member {member_name!r}")
        return np.asarray(payload["arr_0"])


def _sparse_frame_members(archive: tarfile.TarFile) -> dict[int, tuple[str, dict[str, str]]]:
    """Index sparse rangemap members by original source-frame ID."""
    suffixes = {
        "row": ".lidar_row.npz",
        "col": ".lidar_col.npz",
        "range": ".lidar_range.npz",
        "intensity": ".lidar_intensity.npz",
    }
    frames: dict[int, tuple[str, dict[str, str]]] = {}
    for name in archive.getnames():
        kind = next((key for key, suffix in suffixes.items() if name.endswith(suffix)), None)
        if kind is None:
            continue
        prefix = name.removesuffix(suffixes[kind])
        try:
            clip_key, frame_token = prefix.rsplit(".", 1)
            source_frame_id = int(frame_token)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Malformed LiDAR rangemap member name {name!r}") from error
        existing_clip_key, members = frames.setdefault(source_frame_id, (clip_key, {}))
        if existing_clip_key != clip_key:
            raise ValueError(
                f"LiDAR source frame {source_frame_id} has inconsistent clip keys: "
                f"{existing_clip_key!r} and {clip_key!r}"
            )
        if kind in members:
            raise ValueError(f"Duplicate LiDAR {kind} member for source frame {source_frame_id}")
        members[kind] = name
    if not frames:
        raise ValueError("No lidar_row/lidar_col/lidar_range arrays found in LiDAR rangemap tar")
    return frames


def _expected_clip_key_set(expected_clip_key: ExpectedClipKey | None) -> set[str] | None:
    """Return the accepted rangemap clip keys, or None when validation is disabled."""
    if expected_clip_key is None:
        return None
    if isinstance(expected_clip_key, str):
        return {expected_clip_key}
    return set(expected_clip_key)


def _rasterize_tar_frame(archive: tarfile.TarFile, source_frame_id: int, members: dict[str, str]) -> np.ndarray:
    missing = {"row", "col", "range"} - members.keys()
    if missing:
        raise ValueError(f"Incomplete sparse LiDAR frame {source_frame_id}: missing {sorted(missing)}")
    rows = _load_tar_npz_array(archive, members["row"])
    cols = _load_tar_npz_array(archive, members["col"])
    ranges = _load_tar_npz_array(archive, members["range"])
    return rasterize_sparse_range_frame(rows, cols, ranges)


def _rasterize_tar_frame_with_intensity(
    archive: tarfile.TarFile,
    source_frame_id: int,
    members: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    missing = {"row", "col", "range", "intensity"} - members.keys()
    if missing:
        raise ValueError(f"Incomplete sparse LiDAR frame {source_frame_id}: missing {sorted(missing)}")
    rows = _load_tar_npz_array(archive, members["row"])
    cols = _load_tar_npz_array(archive, members["col"])
    ranges = _load_tar_npz_array(archive, members["range"])
    intensities = normalize_intensity_values(_load_tar_npz_array(archive, members["intensity"]))
    if intensities.shape != ranges.shape:
        raise ValueError(
            f"Sparse LiDAR intensity and range arrays must share one length: "
            f"intensity={intensities.shape}, range={ranges.shape}"
        )
    return (
        rasterize_sparse_range_frame(rows, cols, ranges),
        rasterize_sparse_range_frame(rows, cols, intensities),
    )


def load_lidar_frames(
    tar_path: TarSource,
    *,
    frame_indices: list[int] | None = None,
    backend_args: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load selected positional sweeps from one LidarGEN tar as ``[T,128,3600]``."""
    with _open_tar_archive(tar_path, backend_args=backend_args) as archive:
        frame_members = _sparse_frame_members(archive)
        source_frame_ids = sorted(frame_members)
        selected = frame_indices if frame_indices is not None else list(range(len(source_frame_ids)))
        frames: list[np.ndarray] = []
        for frame_index in selected:
            if frame_index < 0 or frame_index >= len(source_frame_ids):
                raise IndexError(f"Frame {frame_index} is outside [0, {len(source_frame_ids)})")
            source_frame_id = source_frame_ids[frame_index]
            _, members = frame_members[source_frame_id]
            frames.append(_rasterize_tar_frame(archive, source_frame_id, members))
    return np.stack(frames, axis=0)


def load_prepared_range_video(
    tar_source: TarSource,
    *,
    min_source_frame_id: int,
    max_source_frame_id: int,
    expected_clip_key: ExpectedClipKey | None = None,
    backend_args: dict[str, Any] | None = None,
    max_frames_per_chunk: int = 8,
    with_intensity: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
    """Load aligned range maps as ``[1,T,128,1024]`` plus optional intensity and source-frame IDs.

    Works for both LiDAR rangemap tars (row/col/range[/intensity]) and HD-map rangemap
    tars that reuse the same sparse member naming without intensity.

    Members are keyed by source-frame ID, so a 10Hz sweep stream inside a 30FPS
    clip is stored at IDs 0, 3, 6, ... Every sweep in the closed interval is
    returned; the bounds are IDs on that clock, not sweep ordinals.
    """
    if min_source_frame_id > max_source_frame_id:
        raise ValueError(
            f"min_source_frame_id must not exceed max_source_frame_id, got "
            f"{min_source_frame_id} > {max_source_frame_id}"
        )
    if max_frames_per_chunk < 1:
        raise ValueError(f"max_frames_per_chunk must be positive, got {max_frames_per_chunk}")
    with _open_tar_archive(tar_source, backend_args=backend_args) as archive:
        frame_members = _sparse_frame_members(archive)
        expected_clip_keys = _expected_clip_key_set(expected_clip_key)
        if expected_clip_keys is not None:
            mismatched_clip_keys = sorted(
                {clip_key for clip_key, _ in frame_members.values() if clip_key not in expected_clip_keys}
            )
            if mismatched_clip_keys:
                raise ValueError(
                    f"Rangemap clip keys {mismatched_clip_keys!r} do not match WebDataset key(s) "
                    f"{tuple(sorted(expected_clip_keys))!r}"
                )
        selected_ids = sorted(
            source_frame_id
            for source_frame_id in frame_members
            if min_source_frame_id <= source_frame_id <= max_source_frame_id
        )
        if not selected_ids:
            raise ValueError(
                f"No rangemap sweeps overlap source-frame interval [{min_source_frame_id}, {max_source_frame_id}]"
            )
        prepared_range_chunks: list[torch.Tensor] = []
        prepared_intensity_chunks: list[torch.Tensor] = []
        for chunk_start in range(0, len(selected_ids), max_frames_per_chunk):
            chunk_ids = selected_ids[chunk_start : chunk_start + max_frames_per_chunk]
            metric_frames: list[np.ndarray] = []
            intensity_frames: list[np.ndarray] = []
            for source_frame_id in chunk_ids:
                _, members = frame_members[source_frame_id]
                if with_intensity:
                    metric, intensity = _rasterize_tar_frame_with_intensity(archive, source_frame_id, members)
                    intensity_frames.append(intensity)
                else:
                    metric = _rasterize_tar_frame(archive, source_frame_id, members)
                metric_frames.append(metric)
            prepared_range_chunks.append(prepare_lidar_video(np.stack(metric_frames), prepend_first_frame=False))
            if with_intensity:
                downsampler = RangeMapDownsampler()
                _, selected_intensity = downsampler.downsample_with_values(
                    np.stack(metric_frames),
                    np.stack(intensity_frames),
                )
                intensity = torch.from_numpy(selected_intensity)[:, None]  # [T,1,H,W]
                intensity = F.interpolate(
                    intensity,
                    size=(RANGE_HEIGHT, TOKENIZER_WIDTH),
                    mode="nearest-exact",
                )
                prepared_intensity_chunks.append(intensity.permute(1, 0, 2, 3).contiguous())  # [1,T,H,W]
    prepared_range = torch.cat(prepared_range_chunks, dim=1)
    prepared_intensity = torch.cat(prepared_intensity_chunks, dim=1) if with_intensity else None
    return prepared_range, prepared_intensity, selected_ids


def enforce_tokenizer_clip_length(
    video: torch.Tensor,
    frame_ids: list[int],
    *,
    num_source_frames: int,
) -> tuple[torch.Tensor, list[int]]:
    """Pad or truncate a prepared clip to ``num_source_frames + 1`` timesteps.

    Tokenizer clips prepend a duplicate of the first source sweep, so the
    expected temporal length is ``num_source_frames + 1``. Short clips are
    right-padded by repeating the last frame; long clips are truncated.
    ``frame_ids`` tracks source sweeps only (length ``num_source_frames``).
    """
    if num_source_frames < 1:
        raise ValueError(f"num_source_frames must be positive, got {num_source_frames}")
    if video.ndim != 4:
        raise ValueError(f"Expected video [C,T,H,W], got {tuple(video.shape)}")
    want_t = int(num_source_frames) + 1
    if video.shape[1] > want_t:
        video = video[:, :want_t]
        frame_ids = list(frame_ids)[:num_source_frames]
    elif video.shape[1] < want_t:
        pad_t = want_t - video.shape[1]
        video = torch.cat([video, video[:, -1:].expand(-1, pad_t, -1, -1)], dim=1)
        last = frame_ids[-1] if frame_ids else 0
        frame_ids = list(frame_ids) + [last] * pad_t
    else:
        frame_ids = list(frame_ids)
    return video, frame_ids


def align_prepared_range_videos(
    control: torch.Tensor,
    control_frame_ids: list[int],
    target: torch.Tensor,
    target_frame_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Keep only the source-frame IDs present in both prepared range videos."""
    if control.ndim != 4 or target.ndim != 4:
        raise ValueError(f"Expected [C,T,H,W] tensors, got {tuple(control.shape)} and {tuple(target.shape)}")
    if control.shape[1] != len(control_frame_ids) or target.shape[1] != len(target_frame_ids):
        raise ValueError(
            "Prepared range temporal length must match frame-id lists: "
            f"control T={control.shape[1]} ids={len(control_frame_ids)}, "
            f"target T={target.shape[1]} ids={len(target_frame_ids)}"
        )
    control_index = {frame_id: idx for idx, frame_id in enumerate(control_frame_ids)}
    target_index = {frame_id: idx for idx, frame_id in enumerate(target_frame_ids)}
    shared_ids = sorted(set(control_index) & set(target_index))
    if not shared_ids:
        raise ValueError("No overlapping source-frame IDs between control and target rangemaps")
    control_positions = torch.tensor([control_index[frame_id] for frame_id in shared_ids], dtype=torch.long)
    target_positions = torch.tensor([target_index[frame_id] for frame_id in shared_ids], dtype=torch.long)
    return control.index_select(1, control_positions), target.index_select(1, target_positions), shared_ids


def list_rangemap_sweep_ids(
    tar_source: TarSource,
    *,
    backend_args: dict[str, Any] | None = None,
) -> list[int]:
    """Return sorted source-frame IDs present in a rangemap tar (10 Hz grid)."""
    with _open_tar_archive(tar_source, backend_args=backend_args) as archive:
        return sorted(_sparse_frame_members(archive))


def load_prepared_lidar_video_3ch(
    tar_source: TarSource,
    *,
    min_source_frame_id: int,
    max_source_frame_id: int,
    expected_clip_key: ExpectedClipKey | None = None,
    backend_args: dict[str, Any] | None = None,
    max_frames_per_chunk: int = 8,
    prepend_first_frame: bool = False,
    range_projection: LidarRangeProjectionConfig = DEFAULT_RANGE_PROJECTION,
    with_intensity: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Load metric range, unit intensity and validity plus source-frame IDs.

    HD-map rangemaps do not carry return intensity. For those controls,
    ``with_intensity=False`` gives every occupied ray unit intensity while
    preserving range-derived validity.
    """
    if min_source_frame_id > max_source_frame_id:
        raise ValueError(
            f"min_source_frame_id must not exceed max_source_frame_id, got "
            f"{min_source_frame_id} > {max_source_frame_id}"
        )
    if max_frames_per_chunk < 1:
        raise ValueError(f"max_frames_per_chunk must be positive, got {max_frames_per_chunk}")
    with _open_tar_archive(tar_source, backend_args=backend_args) as archive:
        frame_members = _sparse_frame_members(archive)
        expected_clip_keys = _expected_clip_key_set(expected_clip_key)
        if expected_clip_keys is not None:
            mismatched_clip_keys = sorted(
                {clip_key for clip_key, _ in frame_members.values() if clip_key not in expected_clip_keys}
            )
            if mismatched_clip_keys:
                raise ValueError(
                    f"LiDAR rangemap clip keys {mismatched_clip_keys!r} do not match "
                    f"WebDataset key(s) {tuple(sorted(expected_clip_keys))!r}"
                )
        selected_ids = sorted(
            source_frame_id
            for source_frame_id in frame_members
            if min_source_frame_id <= source_frame_id <= max_source_frame_id
        )
        if not selected_ids:
            raise ValueError(
                f"No LiDAR rangemap sweeps overlap source-frame interval [{min_source_frame_id}, {max_source_frame_id}]"
            )
        prepared_chunks: list[torch.Tensor] = []
        for chunk_start in range(0, len(selected_ids), max_frames_per_chunk):
            chunk_ids = selected_ids[chunk_start : chunk_start + max_frames_per_chunk]
            metric_frames: list[np.ndarray] = []
            intensity_frames: list[np.ndarray] = []
            for source_frame_id in chunk_ids:
                _, members = frame_members[source_frame_id]
                if with_intensity:
                    metric, intensity = _rasterize_tar_frame_with_intensity(archive, source_frame_id, members)
                    intensity_frames.append(intensity)
                else:
                    metric = _rasterize_tar_frame(archive, source_frame_id, members)
                metric_frames.append(metric)
            # Only prepend on the first chunk when requested, to avoid duplicating
            # frame 0 at every chunk boundary.
            chunk_prepend = prepend_first_frame and chunk_start == 0
            prepared_chunks.append(
                prepare_lidar_video_3ch(
                    np.stack(metric_frames),
                    np.stack(intensity_frames) if with_intensity else None,
                    prepend_first_frame=chunk_prepend,
                    range_projection=range_projection,
                )
            )
    return torch.cat(prepared_chunks, dim=1), selected_ids


def load_lidar_rangeview_directory(
    rangeview_dir: str | Path,
    *,
    frame_prefixes: list[str] | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    """Load loose ``*.lidar_{row,col,range}.npz`` frames as ``[T,128,3600]``."""
    root = Path(rangeview_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Rangeview directory not found: {root}")
    row_files = sorted(root.glob("*.lidar_row.npz"))
    if not row_files:
        raise ValueError(f"No lidar_row arrays found in {root}")
    prefixes = [path.name.removesuffix(".lidar_row.npz") for path in row_files]
    if frame_prefixes is not None:
        missing = [prefix for prefix in frame_prefixes if prefix not in prefixes]
        if missing:
            raise FileNotFoundError(f"Missing rangeview prefixes under {root}: {missing}")
        selected_prefixes = frame_prefixes
    else:
        selected_prefixes = prefixes if max_frames is None else prefixes[:max_frames]
    frames: list[np.ndarray] = []
    for prefix in selected_prefixes:
        rows = _load_npz_array(root / f"{prefix}.lidar_row.npz")
        cols = _load_npz_array(root / f"{prefix}.lidar_col.npz")
        ranges = _load_npz_array(root / f"{prefix}.lidar_range.npz")
        frames.append(rasterize_sparse_range_frame(rows, cols, ranges))
    return np.stack(frames, axis=0)


def load_lidar_sample(
    sample_path: str | Path,
    *,
    num_frames: int | None = TOKENIZER_SAMPLE_FRAMES,
    backend_args: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load selected sweeps of one sample as ``[T,128,3600]``.

    Accepts a local or remote LidarGEN clip tar, or a directory of loose sparse
    frames. ``num_frames=None`` loads every available sweep.
    """
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    path_str = str(sample_path)
    if not is_remote_uri(path_str) and Path(path_str).is_dir():
        frames = load_lidar_rangeview_directory(path_str, max_frames=num_frames)  # [T,128,3600]
    else:
        frame_indices = None if num_frames is None else list(range(num_frames))
        frames = load_lidar_frames(path_str, frame_indices=frame_indices, backend_args=backend_args)  # [T,128,3600]
    if num_frames is not None and frames.shape[0] != num_frames:
        raise ValueError(f"Requested {num_frames} frames but {path_str} yielded {frames.shape[0]}")
    return frames


def _find_local_lidar_samples(data_root: Path) -> list[str]:
    for tar_dir in (data_root / "lidar", data_root):
        tars = sorted(tar_dir.glob("*.tar"))
        if tars:
            return [str(path) for path in tars]
    for frame_dir in (data_root / "rangeview", data_root):
        if any(frame_dir.glob("*.lidar_row.npz")):
            return [str(frame_dir)]
    return []


def _find_remote_lidar_samples(
    data_root: str,
    *,
    backend_args: dict[str, Any] | None,
) -> list[str]:
    from cosmos_framework.utils.easy_io import easy_io

    root = data_root.rstrip("/")
    candidates = (f"{root}/lidar", root)
    args = _backend_args_or_default(backend_args)
    samples: list[str] = []
    last_error: Exception | None = None
    for prefix in candidates:
        try:
            names = list(
                easy_io.list_dir_or_file(
                    prefix if prefix.endswith("/") else f"{prefix}/",
                    list_dir=False,
                    list_file=True,
                    suffix=".tar",
                    recursive=False,
                    backend_args=args,
                )
            )
        except Exception as error:
            last_error = error
            continue
        for name in sorted(names):
            samples.append(name if name.startswith("s3://") else f"{prefix.rstrip('/')}/{name.lstrip('/')}")
        if samples:
            return samples
    if last_error is not None:
        raise RuntimeError(f"Could not list remote LiDAR samples under {data_root}") from last_error
    return []


def find_lidar_samples(
    data_root: str | Path,
    *,
    backend_args: dict[str, Any] | None = None,
) -> list[str]:
    """List clip tars or loose-frame directories under a LiDAR data root.

    Handles the LidarGEN layout (``<root>/lidar/*.tar`` beside ``metadata/``),
    a flat directory of tars, a directory of loose ``*.lidar_row.npz``, and the
    matching S3 prefixes under ``s3://bucket0/lidar/data``.
    """
    root = str(data_root)
    if is_remote_uri(root):
        return _find_remote_lidar_samples(root, backend_args=backend_args)
    path = Path(root)
    if not path.is_dir():
        return []
    return _find_local_lidar_samples(path)
