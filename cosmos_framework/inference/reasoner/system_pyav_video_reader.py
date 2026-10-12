# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sparse, cached Qwen video reader for the release image's CPU PyAV stack.

qwen-vl-utils' torchvision reader decodes every frame before selecting the
requested samples. Long clips can repeat across multiple conversations, which
makes that behavior prohibitively expensive. This reader seeks only the requested
frames through the release image's checksum-pinned official PyAV wheel and
keeps a small, process-local single-flight cache for repeated clips. The image
build must prove that the ``h264`` and ``hevc`` descriptors resolve to software
decoders; CUDA/CUVID resolution is rejected before dataset loading.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from cosmos_framework.inference.reasoner.video_pixel_bounds import normalize_video_pixel_bounds

_CACHE_MAX_ITEMS = max(0, int(os.environ.get("COSMOS_VIDEO_CACHE_ITEMS", "8")))
_CACHE: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, dict[str, Any], float]] = OrderedDict()
_INFLIGHT: dict[tuple[Any, ...], threading.Event] = {}
_PROCESSED_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_PROCESSED_INFLIGHT: dict[tuple[Any, ...], threading.Event] = {}
_ORIGINAL_FETCH_VIDEO: Any = None
_LOCK = threading.RLock()
# PyAV codec contexts are process-local and cold seeks mutate decoder state.
# Cache hits remain concurrent; only cold software decodes are serialized.
_DECODE_LOCK = threading.Lock()


def _assert_software_video_decoders() -> dict[str, str]:
    """Fail before loading data if common codecs resolve to CUDA decoders."""
    resolved = {codec_name: av.Codec(codec_name, "r").name for codec_name in ("h264", "hevc")}
    unexpected = {
        codec_name: decoder_name for codec_name, decoder_name in resolved.items() if decoder_name != codec_name
    }
    if unexpected:
        raise RuntimeError(
            "System PyAV CPU decoding requires software h264/hevc codecs; "
            f"resolved={resolved}. Install the official PyAV wheel rather than "
            "the CUDA-only source build."
        )
    return resolved


def _cache_key(element: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        element.get(key)
        for key in (
            "video",
            "video_start",
            "video_end",
            "nframes",
            "fps",
            "min_frames",
            "max_frames",
        )
    )


def _processed_cache_key(
    element: dict[str, Any],
    image_patch_size: int,
    return_video_sample_fps: bool,
    return_video_metadata: bool,
) -> tuple[Any, ...]:
    """Include every fetch_video input that can change the resized result."""
    return (
        _cache_key(element)
        + tuple(
            element.get(key)
            for key in (
                "min_pixels",
                "max_pixels",
                "total_pixels",
                "resized_height",
                "resized_width",
            )
        )
        + (
            image_patch_size,
            return_video_sample_fps,
            return_video_metadata,
        )
    )


def _stream_frame_count(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    fps: float,
) -> int:
    if stream.frames and stream.frames > 0:
        return int(stream.frames)
    if stream.duration is not None:
        return max(1, int(round(float(stream.duration * stream.time_base) * fps)))
    if container.duration is not None:
        return max(1, int(round(container.duration / av.time_base * fps)))
    raise RuntimeError("video has no frame-count or duration metadata")


def _decode_sparse(
    element: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], float]:
    from qwen_vl_utils.vision_process import smart_nframes

    video_path = str(element["video"])
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if "://" not in video_path and not Path(video_path).is_file():
        raise FileNotFoundError(video_path)

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        if stream.codec_context is None:
            raise RuntimeError(f"system FFmpeg has no decoder for the primary video stream: {video_path}")
        if stream.average_rate is None:
            raise RuntimeError(f"video has no average frame rate: {video_path}")
        fps = float(stream.average_rate)
        total_frames = _stream_frame_count(container, stream, fps)
        start_index = max(0, int(round(float(element.get("video_start", 0.0)) * fps)))
        configured_end = element.get("video_end")
        end_index = total_frames - 1
        if configured_end is not None:
            end_index = min(end_index, int(round(float(configured_end) * fps)))
        if end_index < start_index:
            raise ValueError(f"invalid video interval: start={start_index}, end={end_index}")

        clip_frames = end_index - start_index + 1
        sample_count = smart_nframes(element, total_frames=clip_frames, video_fps=fps)
        target_indices = np.linspace(start_index, end_index, sample_count).round().astype(np.int64)
        decoded: list[torch.Tensor] = []
        decoded_indices: list[int] = []
        time_base = float(stream.time_base)

        for target_index in target_indices:
            target_pts = int((int(target_index) / fps) / time_base)
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            selected = None
            for frame in container.decode(stream):
                selected = frame
                if frame.pts is None or frame.pts >= target_pts:
                    break
            if selected is None:
                raise RuntimeError(f"could not decode frame {target_index} from {video_path}")
            decoded.append(torch.from_numpy(selected.to_ndarray(format="rgb24")).permute(2, 0, 1))
            decoded_indices.append(int(target_index))

        video = torch.stack(decoded)
        sample_fps = sample_count / max(clip_frames, 1) * fps
        metadata = {
            "fps": fps,
            "frames_indices": decoded_indices,
            "total_num_frames": total_frames,
            "video_backend": "cosmos_system_pyav_sparse",
        }
        return video, metadata, sample_fps
    finally:
        container.close()


def read_video_system_pyav(
    element: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], float]:
    """Decode requested frames once per distinct clip/sampling configuration."""
    key = _cache_key(element)
    while True:
        with _LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                _CACHE.move_to_end(key)
                return cached
            event = _INFLIGHT.get(key)
            if event is None:
                event = threading.Event()
                _INFLIGHT[key] = event
                owner = True
            else:
                owner = False
        if owner:
            break
        event.wait()

    try:
        with _DECODE_LOCK:
            result = _decode_sparse(element)
        with _LOCK:
            if _CACHE_MAX_ITEMS:
                _CACHE[key] = result
                _CACHE.move_to_end(key)
                while len(_CACHE) > _CACHE_MAX_ITEMS:
                    _CACHE.popitem(last=False)
        return result
    finally:
        with _LOCK:
            completed = _INFLIGHT.pop(key, None)
            if completed is not None:
                completed.set()


def fetch_video_system_pyav_cached(
    element: dict[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Any:
    """Cache qwen-vl-utils' fully resized video result with single-flight reads.

    Video-conversation data can contain many prompts for each clip. Caching only
    the decoded uint8 frames still makes every prompt repeat the expensive
    bicubic resize and float conversion. This wrapper moves the cache boundary
    past that work.
    """
    if _ORIGINAL_FETCH_VIDEO is None:
        raise RuntimeError("system PyAV video reader has not been registered")

    import qwen_vl_utils.vision_process as vision_process

    element = normalize_video_pixel_bounds(element, image_patch_size, vision_process)

    key = _processed_cache_key(
        element,
        image_patch_size,
        return_video_sample_fps,
        return_video_metadata,
    )
    while True:
        with _LOCK:
            cached = _PROCESSED_CACHE.get(key)
            if cached is not None:
                _PROCESSED_CACHE.move_to_end(key)
                return cached
            event = _PROCESSED_INFLIGHT.get(key)
            if event is None:
                event = threading.Event()
                _PROCESSED_INFLIGHT[key] = event
                owner = True
            else:
                owner = False
        if owner:
            break
        event.wait()

    try:
        result = _ORIGINAL_FETCH_VIDEO(
            element,
            image_patch_size=image_patch_size,
            return_video_sample_fps=return_video_sample_fps,
            return_video_metadata=return_video_metadata,
        )
        with _LOCK:
            if _CACHE_MAX_ITEMS:
                _PROCESSED_CACHE[key] = result
                _PROCESSED_CACHE.move_to_end(key)
                while len(_PROCESSED_CACHE) > _CACHE_MAX_ITEMS:
                    _PROCESSED_CACHE.popitem(last=False)
        return result
    finally:
        with _LOCK:
            completed = _PROCESSED_INFLIGHT.pop(key, None)
            if completed is not None:
                completed.set()


def register_system_pyav_video_reader() -> None:
    """Replace qwen-vl-utils' full-file torchvision decoder in this process."""
    global _ORIGINAL_FETCH_VIDEO

    import qwen_vl_utils.vision_process as vision_process

    _assert_software_video_decoders()

    if os.environ.get("FORCE_QWENVL_VIDEO_READER") not in (None, "torchvision"):
        raise RuntimeError("the system PyAV reader requires FORCE_QWENVL_VIDEO_READER=torchvision")
    vision_process.VIDEO_READER_BACKENDS["torchvision"] = read_video_system_pyav
    if vision_process.fetch_video is not fetch_video_system_pyav_cached:
        _ORIGINAL_FETCH_VIDEO = vision_process.fetch_video
        vision_process.fetch_video = fetch_video_system_pyav_cached


def clear_video_cache() -> None:
    """Clear cached clips; intended for tests and explicit memory recovery."""
    with _LOCK:
        _CACHE.clear()
        _PROCESSED_CACHE.clear()
