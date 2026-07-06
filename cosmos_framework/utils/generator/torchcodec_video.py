# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TorchCodec helpers for Cosmos3 video decoding."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

VideoSource = str | Path | bytes | io.BytesIO | BinaryIO


@dataclass(frozen=True)
class VideoMetadata:
    num_frames: int
    average_fps: float
    height: int | None = None
    width: int | None = None


def _normalize_source(source: VideoSource) -> VideoSource:
    if isinstance(source, Path):
        return source.as_posix()
    if isinstance(source, io.BytesIO):
        source.seek(0)
    return source


def _get_video_decoder_cls() -> Any:
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise ImportError("TorchCodec is required for Cosmos3 video decoding. Install the torchcodec package.") from e
    return VideoDecoder


def _build_decoder(
    source: VideoSource,
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
) -> Any:
    normalized_source = _normalize_source(source)
    # Preserve FFmpeg/TorchCodec's 0 sentinel so callers can request automatic thread selection.
    num_ffmpeg_threads = 0 if num_threads == 0 else max(num_threads, 1)
    kwargs: dict[str, Any] = {"seek_mode": seek_mode, "num_ffmpeg_threads": num_ffmpeg_threads}
    if device != "cpu":
        kwargs["device"] = device
    video_decoder_cls = _get_video_decoder_cls()
    return video_decoder_cls(normalized_source, **kwargs)


def _read_basic_metadata(decoder: Any) -> tuple[int, float]:
    metadata = decoder.metadata
    num_frames = metadata.num_frames
    average_fps = metadata.average_fps
    if num_frames is None or average_fps is None:
        raise ValueError(f"TorchCodec missing metadata (num_frames={num_frames}, average_fps={average_fps})")
    return int(num_frames), float(average_fps)


def _metadata_from_frame(
    decoder: Any,
    first_frame_tchw: torch.Tensor | None = None,
    *,
    include_dimensions: bool = True,
) -> VideoMetadata:
    num_frames, average_fps = _read_basic_metadata(decoder)
    if not include_dimensions:
        return VideoMetadata(num_frames=num_frames, average_fps=average_fps)
    if first_frame_tchw is None:
        first_frame_tchw = decoder.get_frames_at([0]).data.cpu()  # [1,C,H,W]
    _, _, height, width = first_frame_tchw.shape  # [T,C,H,W]
    return VideoMetadata(num_frames=num_frames, average_fps=average_fps, height=int(height), width=int(width))


def probe_video(
    source: VideoSource,
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
    include_dimensions: bool = False,
) -> VideoMetadata:
    """Read video metadata, optionally decoding frame 0 to get frame dimensions."""
    decoder = _build_decoder(source, num_threads=num_threads, seek_mode=seek_mode, device=device)
    return _metadata_from_frame(decoder, include_dimensions=include_dimensions)


class TorchCodecVideoReader:
    """Reusable indexed video reader backed by one TorchCodec decoder."""

    metadata: VideoMetadata

    def __init__(
        self,
        source: VideoSource,
        *,
        num_threads: int = 1,
        seek_mode: str = "exact",
        device: str = "cpu",
        include_dimensions: bool = False,
    ) -> None:
        self._decoder = _build_decoder(source, num_threads=num_threads, seek_mode=seek_mode, device=device)
        self.metadata = _metadata_from_frame(self._decoder, include_dimensions=include_dimensions)

    def __len__(self) -> int:
        return self.metadata.num_frames

    def __getitem__(self, index: int) -> np.ndarray:
        return self.get_frame_nhwc_uint8(index)  # [H,W,C]

    def get_avg_fps(self) -> float:
        return self.metadata.average_fps

    def get_frames_tchw_uint8(self, indices: list[int]) -> torch.Tensor:
        frames_tchw = self._decoder.get_frames_at(indices).data.cpu()  # [T,C,H,W]
        return frames_tchw  # [T,C,H,W]

    def get_frames_nhwc_uint8(self, indices: list[int]) -> np.ndarray:
        frames_tchw = self.get_frames_tchw_uint8(indices)  # [T,C,H,W]
        frames_nhwc = frames_tchw.permute(0, 2, 3, 1).contiguous().numpy()  # [T,H,W,C]
        return frames_nhwc  # [T,H,W,C]

    def get_frame_nhwc_uint8(self, index: int) -> np.ndarray:
        frames_nhwc = self.get_frames_nhwc_uint8([index])  # [1,H,W,C]
        return frames_nhwc[0]  # [H,W,C]


def decode_frames_tchw_uint8(
    source: VideoSource,
    indices: list[int],
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
) -> tuple[torch.Tensor, VideoMetadata]:
    decoder = _build_decoder(source, num_threads=num_threads, seek_mode=seek_mode, device=device)
    frames_tchw = decoder.get_frames_at(indices).data.cpu()  # [T,C,H,W]
    metadata = _metadata_from_frame(decoder, frames_tchw[:1])  # frames_tchw[:1]: [1,C,H,W]
    return frames_tchw, metadata


def decode_frames_cthw_uint8(
    source: VideoSource,
    indices: list[int],
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
) -> tuple[torch.Tensor, VideoMetadata]:
    frames_tchw, metadata = decode_frames_tchw_uint8(
        source, indices, num_threads=num_threads, seek_mode=seek_mode, device=device
    )
    frames_cthw = frames_tchw.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    return frames_cthw, metadata


def decode_frames_nhwc_uint8(
    source: VideoSource,
    indices: list[int],
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
) -> tuple[np.ndarray, VideoMetadata]:
    frames_tchw, metadata = decode_frames_tchw_uint8(
        source, indices, num_threads=num_threads, seek_mode=seek_mode, device=device
    )
    frames_nhwc = frames_tchw.permute(0, 2, 3, 1).contiguous().numpy()  # [T,H,W,C]
    return frames_nhwc, metadata


def decode_frame_nhwc_uint8(
    source: VideoSource,
    index: int,
    *,
    num_threads: int = 1,
    seek_mode: str = "exact",
    device: str = "cpu",
) -> tuple[np.ndarray, VideoMetadata]:
    frames_nhwc, metadata = decode_frames_nhwc_uint8(
        source, [index], num_threads=num_threads, seek_mode=seek_mode, device=device
    )
    return frames_nhwc[0], metadata  # frames_nhwc[0]: [H,W,C]
