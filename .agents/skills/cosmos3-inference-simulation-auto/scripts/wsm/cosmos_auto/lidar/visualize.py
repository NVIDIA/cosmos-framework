# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Colour-coded range-map strips for eyeballing results (toolkit ``--vis``).

Everything here works one frame at a time: a clip of a few hundred sweeps is
coloured and encoded through a streaming writer so the peak footprint is a
handful of RGB frames rather than the whole ``[T, H, W, 3]`` stack.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

EMPTY_CELL_RGB = (0.82, 0.82, 0.82)
"""Colour of cells with no return (range <= 0): light grey."""

ROW_UPSCALE = 4
"""Vertical nearest-neighbour upscale of the strips so each laser ring stays visible."""


def range_map_normalisation(reference: np.ndarray) -> tuple[float, float]:
    """``(near, far)`` log-range bounds for colour-coding, taken from one reference frame.

    ``far`` is the log of the 99th percentile of the frame's valid (``> 0``) ranges and
    ``near`` the log of the 1st percentile; a frame without valid cells yields
    ``(-1.0, 0.0)``. Callers that colour a clip frame by frame compute the pair once
    (typically from the middle frame) and pass it to :func:`colorcode_range_maps` so
    every frame shares the same scale.
    """
    frame = np.asarray(reference, dtype=np.float32)
    valid = frame[frame > 0]
    far = float(np.log(np.quantile(valid, 0.99))) if valid.size else 0.0
    near = float(np.log(np.quantile(valid, 0.01))) if valid.size else far - 1.0
    return near, far


def colorcode_range_maps(
    range_maps: np.ndarray,
    cmap: str = "Spectral",
    near: float | None = None,
    far: float | None = None,
) -> np.ndarray:
    """``[T, H, W]`` (or a single ``[H, W]``) ranges -> ``[..., 3]`` uint8, log-normalised; empty cells grey.

    When ``near``/``far`` are not given they come from :func:`range_map_normalisation`
    of the clip's middle frame (or of the frame itself for ``[H, W]`` input), so a
    single frame coloured with the bounds of a stack matches that stack's output.
    """
    from matplotlib import colormaps

    maps = np.asarray(range_maps, dtype=np.float32)
    if near is None or far is None:
        reference = maps[maps.shape[0] // 2] if maps.ndim == 3 else maps
        reference_near, reference_far = range_map_normalisation(reference)
        far = reference_far if far is None else far
        near = reference_near if near is None else near
    empty = maps <= 0
    span = far - near if far > near else 1.0
    normalised = 1.0 - (np.log(np.where(empty, 1.0, maps)) - near) / span
    rgb = colormaps[cmap](np.clip(normalised, 0.0, 1.0))[..., :3]
    rgb[empty] = EMPTY_CELL_RGB
    return (rgb * 255 + 0.5).astype(np.uint8)


def colorcode_intensity_maps(
    intensity_maps: np.ndarray, cmap: str = "coolwarm"
) -> np.ndarray:
    """``[..., H, W]`` intensities in ``[0, 1]`` -> ``[..., H, W, 3]`` uint8."""
    from matplotlib import colormaps

    rgb = colormaps[cmap](
        np.clip(np.asarray(intensity_maps, dtype=np.float32), 0.0, 1.0)
    )[..., :3]
    return (rgb * 255 + 0.5).astype(np.uint8)


def _aligned_indices(
    hdmap_frames: list[int], lidar_frames: list[int]
) -> tuple[list[int], list[int]]:
    """Per-source row indices of the frames present in both lists, ordered by frame id."""
    common = sorted(set(hdmap_frames) & set(lidar_frames))
    return [hdmap_frames.index(f) for f in common], [
        lidar_frames.index(f) for f in common
    ]


def _upscale_rows(frame: np.ndarray) -> np.ndarray:
    """Repeat rows ``ROW_UPSCALE``x and crop to even dimensions for the encoder."""
    frame = np.repeat(frame, ROW_UPSCALE, axis=0)
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def write_range_map_video(
    path: str | Path,
    hdmap_maps: np.ndarray | None = None,
    hdmap_frames: list[int] | None = None,
    lidar_maps: np.ndarray | None = None,
    intensity_maps: np.ndarray | None = None,
    lidar_frames: list[int] | None = None,
    cmap: str = "Spectral",
    fps: int = 10,
) -> None:
    """Stack (lidar range / hdmap range / lidar intensity) strips per frame and write an MP4.

    When both maps are present only frames present in both are written, aligned by
    pose-frame id; if no frame id is shared a warning is printed and the HD-map strip
    is written alone. Strips are upscaled 4x vertically with nearest-neighbour so each
    row stays one laser ring. Frames are coloured with one ``(near, far)`` scale taken
    from the middle frame and streamed to the encoder one at a time.
    """
    # Row-index lookups into each source for output frame t; None when that source is absent.
    h_idx: list[int] | None = None
    l_idx: list[int] | None = None
    if hdmap_maps is not None and lidar_maps is not None:
        assert hdmap_frames is not None and lidar_frames is not None
        h_idx, l_idx = _aligned_indices(hdmap_frames, lidar_frames)
        if not h_idx:
            print(
                f"[visualize] hdmap and lidar range maps share no frame ids "
                f"({len(hdmap_frames)} hdmap, {len(lidar_frames)} lidar); writing the hdmap strip alone"
            )
            lidar_maps, intensity_maps, l_idx = None, None, None
            h_idx = list(range(len(hdmap_maps)))
    elif hdmap_maps is not None:
        h_idx = list(range(len(hdmap_maps)))
    elif lidar_maps is not None:
        l_idx = list(range(len(lidar_maps)))
    num_frames = (
        len(h_idx) if h_idx is not None else len(l_idx) if l_idx is not None else 0
    )
    if num_frames == 0:
        print(f"[visualize] no range-map frames to write for {path}")
        return

    def range_strip(t: int) -> np.ndarray:
        """Float ranges of output frame ``t``: lidar over hdmap when both exist, else the one present."""
        parts: list[np.ndarray] = []
        if lidar_maps is not None and l_idx is not None:
            parts.append(np.asarray(lidar_maps[l_idx[t]], dtype=np.float32))
        if hdmap_maps is not None and h_idx is not None:
            parts.append(np.asarray(hdmap_maps[h_idx[t]], dtype=np.float32))
        return np.concatenate(parts, axis=0)

    # One colour scale for the whole clip, from the middle frame (same frame as the stacked path used).
    near, far = range_map_normalisation(range_strip(num_frames // 2))

    def rgb_frame(t: int) -> np.ndarray:
        rows: list[np.ndarray] = [
            colorcode_range_maps(range_strip(t), cmap, near=near, far=far)
        ]
        if intensity_maps is not None and l_idx is not None:
            rows.append(colorcode_intensity_maps(intensity_maps[l_idx[t]]))
        return _upscale_rows(np.concatenate(rows, axis=0))

    _stream_frames(Path(path), rgb_frame, num_frames, fps)


def _stream_frames(
    path: Path, frame_at: Callable[[int], np.ndarray], num_frames: int, fps: int
) -> None:
    """Encode ``num_frames`` RGB frames from ``frame_at`` to ``path`` with libx264, one frame at a time."""
    from imageio.v3 import imopen

    path.parent.mkdir(parents=True, exist_ok=True)
    # imageio-ffmpeg backend (in the `wm` extra); macro_block_size=1 keeps the exact strip size.
    # This is the typed equivalent of ``imageio.v2.get_writer(path, format="FFMPEG", ...)``.
    writer = imopen(
        str(path), "w", plugin="FFMPEG", legacy_mode=True
    ).legacy_get_writer(
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
    )
    with writer:
        for t in range(num_frames):
            writer.append_data(frame_at(t))
