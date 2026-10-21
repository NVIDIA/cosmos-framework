# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Full-frame preprocessing shared by directly resized Transfer teachers and causal students."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils.generator.host_threads import host_threads_per_local_rank, torch_host_threads

TransferPixelModality = Literal["rgb", "depth", "seg", "edge", "blur"]
_TRAINED_ASPECT_RATIOS: tuple[str, ...] = ("1,1", "4,3", "3,4", "16,9", "9,16")
_MAX_DISTORTION: float = 0.03
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


@dataclass(frozen=True)
class TransferGeometry:
    """One source field of view and the model bucket shared by every modality."""

    original_hw: tuple[int, int]
    target_hw: tuple[int, int]
    aspect_ratio: str
    distortion: float


@dataclass(frozen=True)
class TransferMedia:
    """Resized pixels and decoded timing used to check RGB/control correspondence."""

    frames: torch.Tensor  # [3,T,H,W], [0,255]; depth is float32, other modalities uint8
    fps: float
    original_hw: tuple[int, int]
    timestamps: tuple[float, ...]
    is_video: bool


def _scale_ratio(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> float:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError(f"Media dimensions must be positive: source={source_hw}, target={target_hw}")
    width_product = target_w * source_h
    height_product = target_h * source_w
    return max(width_product, height_product) / min(width_product, height_product)


def select_transfer_geometry(
    source_hw: tuple[int, int], *, resolution: str, aspect_ratio: str | None = None
) -> TransferGeometry:
    """Choose the closest trained bucket, rejecting aspect distortion above 3%.

    Compare the actual bucket dimensions rather than their nominal aspect labels.
    The inclusive scale-ratio comparison matches ``ResizeToSize`` in training,
    including inputs exactly on the limit without subtraction rounding errors.
    """
    if resolution not in {"480", "720"}:
        raise ValueError("Direct Transfer resizing supports the teacher's trained 480 and 720 tiers only")
    if aspect_ratio is not None and aspect_ratio not in _TRAINED_ASPECT_RATIOS:
        raise ValueError(f"Unsupported direct Transfer aspect-ratio bucket: {aspect_ratio}")
    candidates = (aspect_ratio,) if aspect_ratio is not None else _TRAINED_ASPECT_RATIOS
    sizes = VIDEO_RES_SIZE_INFO[resolution]
    selected = min(candidates, key=lambda key: _scale_ratio(source_hw, (sizes[key][1], sizes[key][0])))
    target_w, target_h = sizes[selected]
    target_hw = (target_h, target_w)
    ratio = _scale_ratio(source_hw, target_hw)
    if ratio > 1.0 + _MAX_DISTORTION:
        raise ValueError(
            f"Unsupported Transfer sample: {source_hw[1]}x{source_hw[0]} -> "
            f"{target_hw[1]}x{target_hw[0]} causes {ratio - 1:.2%} aspect distortion, exceeding 3%. "
            "Choose a compatible source aspect ratio or skip this sample."
        )
    return TransferGeometry(source_hw, target_hw, selected, ratio - 1.0)


def resize_transfer_frames(
    frames: torch.Tensor,  # [T,C,H,W], pixels in [0,255]
    target_hw: tuple[int, int],
    *,
    modality: TransferPixelModality,
) -> torch.Tensor:  # [T,C,target_h,target_w], pixels in [0,255]
    """Apply training's interpolation and dtype, without spatial padding or cropping."""
    size = list(target_hw)
    if modality in {"seg", "edge"}:
        # Preserve label palettes and already-computed thin/binary edges.
        return TF.resize(frames, size, interpolation=TF.InterpolationMode.NEAREST, antialias=False)  # [T,C,H,W]
    if modality == "depth":
        # Training parses depth as float32 before its bilinear resize. Quantizing
        # the resized pixels to uint8 would silently discard fractional depths.
        return TF.resize(  # [T,C,H,W]
            frames.float(), size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
    if modality not in {"rgb", "blur"}:
        raise ValueError(f"Unknown Transfer pixel modality: {modality}")
    # RGB training resizes uint8 pixels before normalization. Edge/blur controls
    # are normally computed from that resized RGB; precomputed controls retain
    # their complete field of view with the same continuous-pixel interpolation.
    return TF.resize(frames, size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)  # [T,C,H,W]


def read_direct_transfer_media(
    path: Path,
    *,
    geometry: TransferGeometry,
    modality: TransferPixelModality,
    max_frames: int | None,
) -> TransferMedia:
    """Decode a bounded frame window, resizing batches before retaining them.

    At most 16 source-resolution frames are held together; retained clips use
    model dimensions. PyAV stops after ``max_frames`` rather than decoding the
    entire source clip, and timestamps stay available for alignment validation.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if path.suffix.lower() in _IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"))  # [H,W,3]
        frames = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        original_hw = (frames.shape[2], frames.shape[3])
        _validate_control_geometry(original_hw, geometry)
        resized = resize_transfer_frames(frames, geometry.target_hw, modality=modality)  # [1,3,H,W]
        return TransferMedia(resized.permute(1, 0, 2, 3), 1.0, original_hw, (0.0,), False)  # [3,1,H,W]
    if path.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {path.suffix}")

    batches: list[torch.Tensor] = []  # Each [T,3,H,W]
    pending: list[np.ndarray] = []  # Each [source_h,source_w,3]
    timestamps: list[float] = []
    original_hw: tuple[int, int] | None = None
    num_threads = host_threads_per_local_rank()
    with av.open(str(path), metadata_errors="ignore") as container, torch_host_threads(num_threads):
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        stream.thread_count = num_threads
        fps = float(stream.average_rate or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot determine a positive frame rate for {path}")
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts < 0:
                continue
            array = frame.to_rgb().to_ndarray()  # [source_h,source_w,3]
            frame_hw = (array.shape[0], array.shape[1])
            if original_hw is None:
                original_hw = frame_hw
                _validate_control_geometry(original_hw, geometry)
            elif frame_hw != original_hw:
                raise ValueError(f"Video changes resolution within the selected window: {path}")
            pending.append(array)
            timestamps.append(float(frame.time) if frame.time is not None else len(timestamps) / fps)
            if len(pending) == 16 or (max_frames is not None and len(timestamps) == max_frames):
                pixels = torch.from_numpy(np.stack(pending)).permute(0, 3, 1, 2)  # [T,3,source_h,source_w]
                batches.append(resize_transfer_frames(pixels, geometry.target_hw, modality=modality))  # [T,3,H,W]
                pending.clear()
            if max_frames is not None and len(timestamps) == max_frames:
                break
        if pending:
            pixels = torch.from_numpy(np.stack(pending)).permute(0, 3, 1, 2)  # [T,3,source_h,source_w]
            batches.append(resize_transfer_frames(pixels, geometry.target_hw, modality=modality))  # [T,3,H,W]
    if original_hw is None:
        raise ValueError(f"Decoded zero video frames from {path}")
    resized = torch.cat(batches, dim=0).permute(1, 0, 2, 3)  # [3,T,H,W]
    return TransferMedia(resized, fps, original_hw, tuple(timestamps), True)


def _validate_control_geometry(control_hw: tuple[int, int], geometry: TransferGeometry) -> None:
    """Reject mismatched source canvases; integer rounding may differ by one pixel.

    Equal geometry is necessary but cannot prove that two files depict the same
    crop or camera. That correspondence remains the input producer's contract.
    """
    ratio = _scale_ratio(control_hw, geometry.original_hw)
    rounding_tolerance = 1.0 / min(control_hw) + 1.0 / min(geometry.original_hw)
    if ratio > 1.0 + rounding_tolerance:
        raise ValueError(
            f"Transfer RGB/control source aspect ratios do not match: "
            f"reference={geometry.original_hw}, control={control_hw}. Supply matching fields of view."
        )
    if _scale_ratio(control_hw, geometry.target_hw) > 1.0 + _MAX_DISTORTION:
        raise ValueError(f"Transfer control {control_hw} exceeds the 3% distortion limit for {geometry.target_hw}")


def validate_transfer_media_alignment(reference: TransferMedia, control: TransferMedia) -> None:
    """Verify observable temporal correspondence before any temporal padding."""
    # A still RGB image can condition the first frame of a video, so it has no
    # video clock to compare. Multiple video controls must have matching clocks.
    if reference.is_video and not control.is_video:
        raise ValueError("A video Transfer reference requires video controls with matching frames")
    if not reference.is_video:
        return
    if len(reference.timestamps) != len(control.timestamps):
        raise ValueError(
            f"Transfer RGB/control frame counts differ within the selected window: "
            f"{len(reference.timestamps)} != {len(control.timestamps)}"
        )
    if not math.isclose(reference.fps, control.fps, rel_tol=1e-3):
        raise ValueError(f"Transfer RGB/control FPS mismatch: {reference.fps} != {control.fps}")
    reference_times = np.asarray(reference.timestamps) - reference.timestamps[0]  # [T]
    control_times = np.asarray(control.timestamps) - control.timestamps[0]  # [T]
    if not np.allclose(reference_times, control_times, rtol=0, atol=0.5 / reference.fps):
        raise ValueError("Transfer RGB/control presentation timestamps do not align")
