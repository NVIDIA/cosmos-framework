# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Copied from projects/cosmos/reason1/datasets/video_decoder_qwen.py
Changes:
1: remove hardcoded hyper-parameters for Qwen, now read it from processor
2: support skipping smart resize, since it may resize the video frames to be smaller than model input and frames will get resized up later in processor
"""

import hashlib
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cosmos_framework.utils import log
from cosmos_framework.data.generator.processors.qwen3vl_processor import VIDEO_BUDGET_RESIZED_KEY, Qwen3VLProcessor
from cosmos_framework.utils.generator.source_video_timing import (
    SOURCE_VIDEO_TIMING_KEY,
    build_source_video_timing,
    require_source_pts_processor,
    validate_video_timestamp_mode,
)
from cosmos_framework.utils.generator.video_frame_sampling import smart_nframes_with_factor
from cosmos_framework.utils.generator.video_source_metadata import VIDEO_METADATA_KEY, validate_source_video_metadata

Image.MAX_IMAGE_PIXELS = 933120000
_VIDEO_EXTENSIONS = "mp4 avi webm mov".split()

VIDEO_DECODER_OPTIONS = {}
VideoTemporalMode = Literal["native", "framewise"]


class VideoDecodeTimeoutError(TimeoutError):
    """A resource-dependent decode failure that must not become a fixed data drop."""


def get_effective_temporal_patch_size(temporal_patch_size: int, video_temporal_mode: VideoTemporalMode) -> int:
    """Return the temporal token-budget factor without changing the ViT tubelet kernel."""
    if video_temporal_mode == "native":
        return temporal_patch_size
    if video_temporal_mode == "framewise":
        return 1
    raise ValueError(f"Unsupported video_temporal_mode: {video_temporal_mode!r}")


def token_to_pixels(token_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2) -> int:
    """Convert token length to pixels based on patch size and temporal patch size.

    Args:
        token_length: Token length
        patch_size: Patch size
        temporal_patch_size: Temporal patch size,
            for Qwen it has 3D conv, temporal patch size is 2; for other models like internVL or eagle er, the temporal patch size is 1 since their VIT is image encoder;
        merge_size: Merge size, or called pixel shuffing factor;
            for Qwen and internVL it is 2; for eagle er it is 1;
    """
    merged_patch_size = patch_size * merge_size
    return token_length * merged_patch_size**2 * temporal_patch_size


def pixels_to_token(pixels: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2) -> int:
    """Convert pixels to token length based on patch size and temporal patch size."""
    merged_patch_size = patch_size * merge_size
    return pixels // merged_patch_size**2 // temporal_patch_size


def video_decoder_qwen(
    num_threads: int = 0,
    min_fps_thres: int = 4,
    max_fps_thres: float = 60.0,
    target_fps: float = 2.0,
    min_video_token_length: int = 16,
    max_video_token_length: int = 8192,
    max_video_frames: int | None = None,
    random_augmentation: bool = False,
    frame_count_random_range: Optional[list[int]] = None,
    video_timestamp_mode: str = "qwen_index",
    video_temporal_mode: VideoTemporalMode = "native",
    decoding_timeout: int = 60,
    raise_on_decode_timeout: bool = False,
    **kwargs,
) -> Callable:
    """
    Sampling video frames similar to Qwen. It prioritizes matching the target FPS first and then resizing the video frames.
    See https://github.com/kq-chen/qwen-vl-utils/blob/main/src/qwen_vl_utils/vision_process.py#L118 for more details.

    Args:
        key: Video file name/key
        data: Video binary data
        min_fps_thres: Minimum FPS threshold
        max_fps_thres: Maximum FPS threshold
        target_fps: Target FPS
        min_video_token_length: Minimum token length
        max_video_token_length: Maximum token length
        num_threads: Number of threads for the torchcodec video decoder
        random_augmentation: Whether to randomize the FPS and max_video_token_length
        frame_count_random_range: Random frame count range
        video_timestamp_mode: Qwen source index/FPS for supported processors, or legacy synthetic FPS.
            Source PTS requires the native SFT BytesToMedia path.

    Returns:
        dict with video frames tensor and target FPS
    """

    validate_video_timestamp_mode(video_timestamp_mode)
    if video_timestamp_mode == "source_pts":
        raise ValueError("source_pts requires the native SFT BytesToMedia path with its verified Edge processor")
    video_decoder_configured = partial(
        _video_decoder_qwen_func,
        min_fps_thres=min_fps_thres,
        max_fps_thres=max_fps_thres,
        num_threads=num_threads,
        target_fps=target_fps,
        min_video_token_length=min_video_token_length,
        max_video_token_length=max_video_token_length,
        max_video_frames=max_video_frames,
        random_augmentation=random_augmentation,
        frame_count_random_range=frame_count_random_range,
        video_timestamp_mode=video_timestamp_mode,
        video_temporal_mode=video_temporal_mode,
        decoding_timeout=decoding_timeout,
        raise_on_decode_timeout=raise_on_decode_timeout,
    )

    return video_decoder_configured


def _video_decoder_qwen_func(
    key: str,
    data: bytes,
    processor: Qwen3VLProcessor,
    min_fps_thres: int = 4,
    max_fps_thres: float = 60.0,
    target_fps: float = 2.0,
    min_video_token_length: int = 16,
    max_video_token_length: int = 8192,
    max_video_frames: int | None = None,
    num_threads: int = 0,
    random_augmentation: bool = False,
    fps_random_range: list[float] = [0.5, 1.5],
    max_video_token_length_random_range: list[float] = [0.75, 1.25],
    frame_count_random_range: Optional[list[int]] = None,
    video_temporal_mode: VideoTemporalMode = "native",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    decoding_timeout: int = 60,
    video_timestamp_mode: str = "qwen_index",
    raise_on_decode_timeout: bool = False,
    decode_chunk_size: int = 32,
    **kwargs,
) -> dict | None:
    """Actual video decoder function.

    Args:
        key (str): Video file name/key
        data (bytes): Video binary data
        min_fps_thres (int, optional): Minimum FPS threshold. Defaults to 4.
        max_fps_thres (int, optional): Maximum FPS threshold. Defaults to 60.
        target_fps (float, optional): Target FPS. Defaults to 2.0.
        min_video_token_length (int, optional): Minimum token length. Defaults to 16.
        max_video_token_length (int, optional): Maximum token length. Defaults to 8192.
        num_threads (int, optional): Number of threads for the torchcodec video decoder. Defaults to 0.
        random_augmentation (bool, optional): Whether to randomize the FPS and max_video_token_length. Defaults to False.
        fps_random_range (list[float], optional): Random FPS range. Defaults to [10.0, 24.0].
        max_video_token_length_random_range (list[float], optional): Random max_video_token_length range. Defaults to [0.75, 1.25].
        frame_count_random_range (list[int], optional): Random frame count range. If provided, take priority over fps_random_range.
        start_frame (Optional[int], optional): Start frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.
        end_frame (Optional[int], optional): End frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.
        decoding_timeout (int, optional): Timeout in seconds for decoding one chunk of frames. Defaults to 60.
        raise_on_decode_timeout: Raise instead of dropping a sample when decoding times out.
        video_timestamp_mode: Use Qwen source index/FPS by default for supported processors,
            legacy synthetic FPS, or actual selected FrameBatch presentation timestamps.
        decode_chunk_size (int, optional): Frames decoded (and resized) at a time, bounding native-resolution
            frames held in memory. Defaults to 32.
    Raises:
        ValueError: Video fps lower than 1, skipping
        ValueError: Video fps lower than min_fps_thres, skipping
        ValueError: Video fps higher than max_fps_thres, skipping

    Returns:
        dict | None: Dictionary with video frames, effective sampled FPS, and source video_metadata
            for processors that render Qwen-style timestamps. Selected source indices retain crop offsets.
    """
    validate_video_timestamp_mode(video_timestamp_mode)
    if video_timestamp_mode == "source_pts":
        require_source_pts_processor(processor)
        if (start_frame is None) != (end_frame is None):
            raise ValueError("source_pts requires both start_frame and end_frame, or neither")
    # Check video extension
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _VIDEO_EXTENSIONS:
        return None

    # Read video with torchcodec
    video_reader = VideoDecoder(data, num_ffmpeg_threads=num_threads)
    total_frames = video_reader.metadata.num_frames
    video_fps = video_reader.metadata.average_fps

    # torchcodec returns ``None`` for containers that don't store frame count
    # or average fps (e.g. some MKV/WebM streams).  Downstream arithmetic
    # (``total_frames - 1``, ``video_fps < 1``, ...) would TypeError on None;
    # surface a ValueError so the dataloader's skip path handles it uniformly.
    if total_frames is None or video_fps is None:
        raise ValueError(f"torchcodec missing metadata (num_frames={total_frames}, average_fps={video_fps}), skipping")

    source_frame_offset = 0
    source_total_frames = total_frames
    uses_source_timestamps = video_timestamp_mode == "qwen_index" and getattr(
        processor, "USES_SOURCE_VIDEO_TIMESTAMPS", False
    )
    if uses_source_timestamps and (start_frame is None) != (end_frame is None):
        raise ValueError("Source video timing requires both start_frame and end_frame, or neither")
    if start_frame is not None and end_frame is not None:
        if video_timestamp_mode == "source_pts" and not 0 <= start_frame < end_frame <= total_frames:
            raise ValueError("source_pts requires an in-bounds nonempty source frame interval [start_frame,end_frame)")
        if uses_source_timestamps and not 0 <= start_frame < end_frame <= source_total_frames:
            raise ValueError("Source video crop must be an in-bounds interval [start_frame, end_frame)")
        source_frame_offset = start_frame
        total_frames = end_frame - start_frame

    if video_fps < 1:
        raise ValueError("Video fps lower than 1, skipping")
    if video_fps < min_fps_thres and not math.isclose(video_fps, min_fps_thres, rel_tol=1e-12):
        raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
    if video_fps > max_fps_thres and not math.isclose(video_fps, max_fps_thres, rel_tol=1e-12):
        raise ValueError(f"Video fps {video_fps} higher than {max_fps_thres}, skipping")

    configured_max_video_token_length = max_video_token_length
    if random_augmentation:
        if frame_count_random_range is not None:
            # Random number of frames
            min_frames_range, max_frames_range = frame_count_random_range
            min_frames_range = min(min_frames_range, total_frames)
            max_frames_range = min(max_frames_range, total_frames)
            target_frames = random.uniform(min_frames_range, max_frames_range)
            target_fps = target_frames / total_frames * video_fps
        else:
            # randomize fps
            target_fps = (
                random.uniform(fps_random_range[0], fps_random_range[1]) * target_fps
                if random.random() < 0.5
                else target_fps
            )
        # randomize max_video_token_length
        max_video_token_length = int(
            random.uniform(max_video_token_length_random_range[0], max_video_token_length_random_range[1])
            * max_video_token_length
        )
        log.debug(f"random_augmentation: max_video_token_length: {max_video_token_length}, target_fps: {target_fps}")

    patch_size = processor.patch_size
    min_height_width = processor.min_height_width
    temporal_patch_size = get_effective_temporal_patch_size(processor.temporal_patch_size, video_temporal_mode)
    merge_size = processor.merge_size
    min_pixels: int = token_to_pixels(min_video_token_length, patch_size, temporal_patch_size, merge_size)
    max_pixels: int = token_to_pixels(max_video_token_length, patch_size, temporal_patch_size, merge_size)
    total_min_pixels = min_pixels
    total_max_pixels = max_pixels
    # A budget whose frame cap is below one temporal patch (under 128 tokens for Qwen3-VL in native mode, e.g. a
    # short clip sharing a budget with long videos) still decodes one temporal patch; the per-frame budget below
    # sets its resolution, under min_height_width when the budget cannot afford more.
    max_frames: int = max(max_pixels // (min_height_width) ** 2 // temporal_patch_size, temporal_patch_size)
    if max_video_frames is not None:
        if max_video_frames <= 0:
            raise ValueError(f"max_video_frames must be positive, got {max_video_frames}")
        aligned_max_video_frames = max_video_frames // temporal_patch_size * temporal_patch_size
        if aligned_max_video_frames < temporal_patch_size:
            raise ValueError(
                f"max_video_frames must allow at least {temporal_patch_size} temporally aligned frames, "
                f"got {max_video_frames}"
            )
        max_frames = min(max_frames, aligned_max_video_frames)

    nframes = smart_nframes_with_factor(
        {"fps": target_fps, "max_frames": max_frames},
        total_frames=total_frames,
        video_fps=video_fps,
        frame_factor=temporal_patch_size,
    )
    if start_frame is not None and end_frame is not None:
        idx = torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()  # [nframes]
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()  # [nframes]

    # Each frame gets an even share of the budget. min_pixels floors the frame size only where that share affords
    # it: below it (the single temporal patch a too-small budget still decodes, see max_frames) the floor is one
    # merged patch, so frames are only ever shrunk to fit and a small allocation never spends more tokens than it
    # was given.
    max_pixels_per_frame = max_pixels // nframes
    min_pixels_per_frame = min_pixels if max_pixels_per_frame >= min_pixels else (patch_size * merge_size) ** 2

    # Decode (and resize) chunk by chunk into one preallocated output. With a resize, only one native chunk is alive
    # at a time: decoding every sampled frame at native resolution before resizing peaks at ~37 MiB per 1080p frame,
    # tens of GiB for the frame counts of long-context budgets. Without one, the output itself is native; filling it
    # in place still avoids holding the decoded chunks and their concatenation at once.
    # Each chunk is decoded in a worker thread with a timeout; a stuck thread is abandoned immediately.
    executor = ThreadPoolExecutor(max_workers=1)
    video_frames: torch.Tensor | None = None  # [T,C,H',W'] uint8, allocated once the first chunk gives the size
    pts_chunks: list[torch.Tensor] = []
    duration_chunks: list[torch.Tensor] = []
    try:
        for chunk_start in range(0, nframes, decode_chunk_size):
            chunk_indices = idx[chunk_start : chunk_start + decode_chunk_size]
            frame_batch = executor.submit(video_reader.get_frames_at, indices=chunk_indices).result(
                timeout=decoding_timeout
            )  # pixels: [t,C,H,W]; times: [t]
            if video_timestamp_mode == "source_pts":
                if not isinstance(frame_batch, FrameBatch):
                    raise ValueError("source_pts requires TorchCodec FrameBatch presentation metadata")
                pts_chunks.append(frame_batch.pts_seconds)  # [t]
                duration_chunks.append(frame_batch.duration_seconds)  # [t]
            frames = frame_batch.data  # [t,C,H,W] uint8
            del frame_batch  # the native chunk must not outlive its resize while the next chunk decodes
            if frames.shape[0] != len(chunk_indices):
                raise ValueError(f"[{key}] decoder returned {frames.shape[0]} frames for {len(chunk_indices)} indices")
            if video_frames is None:
                native_height, native_width = int(frames.shape[2]), int(frames.shape[3])
                resized_height, resized_width = native_height, native_width
                if processor.use_smart_resize:
                    resized_height, resized_width = smart_resize(
                        native_height,
                        native_width,
                        factor=patch_size * merge_size,
                        min_pixels=min_pixels_per_frame,
                        max_pixels=max_pixels_per_frame,
                    )
                    log.debug(
                        f"resized_height: {resized_height}, resized_width: {resized_width} | original height: {native_height}, original width: {native_width}"
                    )
                video_frames = torch.empty(
                    (nframes, frames.shape[1], resized_height, resized_width), dtype=frames.dtype
                )  # [T,C,H',W']
            if processor.use_smart_resize:
                frames = transforms.functional.resize(
                    frames,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )  # [t,C,H',W'] uint8
            video_frames[chunk_start : chunk_start + len(chunk_indices)] = frames
            del frames  # copied; the chunk must not stay alive while the next one decodes
    except TimeoutError as e:
        log.warning(f"[{key}] Video decoding timed out after {decoding_timeout} seconds")
        if raise_on_decode_timeout:
            raise VideoDecodeTimeoutError(
                f"Video decoding exceeded {decoding_timeout} seconds for {key!r}; "
                "retry with sufficient CPU resources or a larger timeout instead of dropping this sample."
            ) from e
        return None
    finally:
        executor.shutdown(wait=False)

    if video_frames is None:
        raise ValueError(f"[{key}] no frames were sampled")
    if processor.use_smart_resize:
        video_frames = video_frames.float()  # [T,C,H',W']
    timing = None
    if video_timestamp_mode == "source_pts":
        timing = build_source_video_timing(
            idx,
            torch.cat(pts_chunks),  # [T]
            torch.cat(duration_chunks),  # [T]
            hashlib.sha256(data).hexdigest(),
        )

    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H',W']

    result = dict(
        videos=video_frames,
        fps=sample_fps,
        native_fps=float(video_fps),
        native_num_frames=int(total_frames),
        sampled_num_frames=int(nframes),
        native_height=native_height,
        native_width=native_width,
        native_duration_sec=float(total_frames) / video_fps,
        configured_max_video_token_length=int(configured_max_video_token_length),
        effective_max_video_token_length=int(max_video_token_length),
        budget_min_pixels=int(total_min_pixels),
        budget_max_pixels=int(total_max_pixels),
    )
    result[VIDEO_BUDGET_RESIZED_KEY] = bool(processor.use_smart_resize)
    if timing is not None:
        result[SOURCE_VIDEO_TIMING_KEY] = timing
    if uses_source_timestamps:
        # Match qwen-vl-utils: retain source indices and source FPS even after sampling/cropping.
        # The sampled FPS remains available for callers that do not render source timestamps.
        result[VIDEO_METADATA_KEY] = validate_source_video_metadata(
            {"fps": video_fps, "total_num_frames": source_total_frames, "frames_indices": idx},
            video_frames.shape[1],
        )
    if video_temporal_mode == "framewise":
        result.update(
            source_fps=video_fps,
            source_frames_indices=[frame_index - source_frame_offset for frame_index in idx],
            source_total_num_frames=total_frames,
        )
    return result
