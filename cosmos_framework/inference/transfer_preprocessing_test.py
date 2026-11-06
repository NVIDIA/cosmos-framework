# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Geometry, discrete-control, and media-alignment regressions for Transfer."""

from pathlib import Path

import av
import numpy as np
import pytest
import torch
from av.video.stream import VideoStream
from PIL import Image

from cosmos_framework.inference.transfer_preprocessing import (
    TransferGeometry,
    TransferMedia,
    TransferPixelModality,
    _validate_control_geometry,
    read_direct_transfer_media,
    resize_transfer_frames,
    select_transfer_geometry,
    validate_transfer_media_alignment,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    ("source_hw", "resolution", "expected_hw", "expected_aspect"),
    [
        ((1080, 1920), "480", (480, 832), "16,9"),
        ((1080, 1920), "720", (720, 1280), "16,9"),
        ((1920, 1080), "480", (832, 480), "9,16"),
        ((600, 800), "480", (544, 736), "4,3"),
        ((800, 600), "720", (1104, 832), "3,4"),
        ((1000, 1000), "480", (640, 640), "1,1"),
    ],
)
def test_direct_resize_selects_trained_bucket(
    source_hw: tuple[int, int], resolution: str, expected_hw: tuple[int, int], expected_aspect: str
) -> None:
    geometry = select_transfer_geometry(source_hw, resolution=resolution)
    assert geometry.original_hw == source_hw
    assert geometry.target_hw == expected_hw
    assert geometry.aspect_ratio == expected_aspect
    assert geometry.distortion <= 0.03


def test_exact_three_percent_boundary_and_explicit_override() -> None:
    accepted = select_transfer_geometry((100, 103), resolution="480", aspect_ratio="1,1")
    assert accepted.target_hw == (640, 640)
    with pytest.raises(ValueError, match="exceeding 3%"):
        select_transfer_geometry((10000, 10301), resolution="480", aspect_ratio="1,1")
    with pytest.raises(ValueError, match="exceeding 3%"):
        select_transfer_geometry((1080, 1920), resolution="480", aspect_ratio="4,3")


def test_forced_aspect_ratio_error_names_the_field_and_the_bucket_that_fits() -> None:
    with pytest.raises(ValueError) as excinfo:
        select_transfer_geometry((720, 1280), resolution="720", aspect_ratio="3,4")
    message = str(excinfo.value)
    assert "aspect_ratio='3,4' was requested" in message
    assert "without cropping" in message
    assert "Omit aspect_ratio: the closest trained bucket for this source is '16,9'." in message


def test_forced_aspect_ratio_on_a_source_no_bucket_fits_does_not_suggest_omitting_it() -> None:
    with pytest.raises(ValueError) as excinfo:
        select_transfer_geometry((1080, 2560), resolution="480", aspect_ratio="16,9")
    message = str(excinfo.value)
    assert "aspect_ratio='16,9' was requested" in message
    assert "Omit aspect_ratio" not in message
    assert "No trained bucket is within 3% of this source either" in message


def test_auto_bucket_error_says_no_bucket_fits() -> None:
    with pytest.raises(ValueError) as excinfo:
        select_transfer_geometry((1080, 2560), resolution="480")
    message = str(excinfo.value)
    assert "No trained aspect-ratio bucket is within 3%" in message
    assert "aspect_ratio=" not in message


@pytest.mark.parametrize("source_hw", [(1080, 2560), (720, 1080), (0, 1920)])
def test_unsupported_source_is_rejected_without_extra_buckets(source_hw: tuple[int, int]) -> None:
    # The global size table also has a 2:3 robotics bucket, but this teacher was
    # trained on the five standard buckets and must not silently adopt that one.
    with pytest.raises(ValueError):
        select_transfer_geometry(source_hw, resolution="480")


@pytest.mark.parametrize("resolution", ["256", "704", "1080"])
def test_direct_resize_rejects_untrained_tiers(resolution: str) -> None:
    with pytest.raises(ValueError, match="480 and 720"):
        select_transfer_geometry((1080, 1920), resolution=resolution)


@pytest.mark.parametrize("modality", ["seg", "edge"])
def test_discrete_controls_preserve_palette_and_boundaries(modality: TransferPixelModality) -> None:
    pixels = torch.tensor([[[[0, 255], [91, 173]]]], dtype=torch.uint8)  # [1,1,2,2]
    resized = resize_transfer_frames(pixels, (4, 4), modality=modality)  # [1,1,4,4]
    expected = pixels.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # [1,1,4,4]
    assert torch.equal(resized, expected)
    assert resized.dtype == torch.uint8


def test_depth_retains_fractional_bilinear_pixels() -> None:
    pixels = torch.tensor([[[[0, 255], [0, 255]]]], dtype=torch.uint8)  # [1,1,2,2]
    resized = resize_transfer_frames(pixels, (3, 3), modality="depth")  # [1,1,3,3]
    assert resized.dtype == torch.float32
    assert resized[0, 0, 1, 1].item() == pytest.approx(127.5)
    assert resized[0, 0, :, 0].eq(0).all()
    assert resized[0, 0, :, -1].eq(255).all()


def test_rgb_preserves_complete_coordinate_grid_and_uint8_training_dtype() -> None:
    horizontal = torch.linspace(0, 255, 128).round().to(torch.uint8)  # [128]
    vertical = torch.linspace(0, 255, 72).round().to(torch.uint8)  # [72]
    pixels = torch.zeros((1, 3, 72, 128), dtype=torch.uint8)  # [1,3,72,128]
    pixels[:, 0] = horizontal.unsqueeze(0).expand(72, 128)  # [1,72,128]
    pixels[:, 1] = vertical.unsqueeze(1).expand(72, 128)  # [1,72,128]
    pixels[:, 2, :, :4] = 255  # [1,72,4], left-edge marker that cropping would remove
    resized = resize_transfer_frames(pixels, (120, 208), modality="rgb")  # [1,3,120,208]
    assert resized.dtype == torch.uint8
    assert resized[0, 0, 60, 0].item() <= 1
    assert resized[0, 0, 60, -1].item() >= 254
    assert resized[0, 1, 0, 104].item() <= 1
    assert resized[0, 1, -1, 104].item() >= 254
    assert resized[0, 2, :, 0].eq(255).all()


def _media(*, fps: float = 24.0, frames: int = 3, timestamps: tuple[float, ...] | None = None) -> TransferMedia:
    pixels = torch.zeros((3, frames, 8, 8), dtype=torch.uint8)  # [3,T,8,8]
    return TransferMedia(pixels, fps, (8, 8), timestamps or tuple(index / fps for index in range(frames)), True)


def test_alignment_accepts_different_resolution_of_same_canvas() -> None:
    geometry = select_transfer_geometry((1080, 1920), resolution="480")
    _validate_control_geometry((540, 960), geometry)
    validate_transfer_media_alignment(_media(), _media(timestamps=(3.0, 3 + 1 / 24, 3 + 2 / 24)))


def test_alignment_rejects_incompatible_canvases() -> None:
    geometry = select_transfer_geometry((1080, 1920), resolution="480")
    with pytest.raises(ValueError, match="aspect ratios do not match"):
        _validate_control_geometry((720, 960), geometry)


@pytest.mark.parametrize(
    ("control", "message"),
    [
        (_media(frames=2), "frame counts"),
        (_media(fps=30.0), "FPS mismatch"),
        (_media(timestamps=(0.0, 1 / 24, 4 / 24)), "timestamps"),
    ],
)
def test_alignment_rejects_temporal_mismatch(control: TransferMedia, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_transfer_media_alignment(_media(), control)


def test_image_decode_preserves_segmentation_colors(tmp_path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)  # [8,8,3]
    pixels[:, :4] = [255, 17, 93]  # [8,4,3]
    pixels[:, 4:] = [2, 144, 65]  # [8,4,3]
    path = tmp_path / "seg.png"
    Image.fromarray(pixels).save(path)
    geometry = TransferGeometry((8, 8), (6, 6), "1,1", 0.0)
    media = read_direct_transfer_media(path, geometry=geometry, modality="seg", max_frames=1)
    assert media.original_hw == (8, 8)
    assert media.frames.shape == (3, 1, 6, 6)
    colors = media.frames.permute(1, 2, 3, 0).reshape(-1, 3).unique(dim=0).tolist()  # [2,3]
    assert sorted(colors) == [[2, 144, 65], [255, 17, 93]]


def test_video_decode_obeys_frame_limit_and_keeps_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cosmos_framework.inference.transfer_preprocessing.host_threads_per_local_rank", lambda: 1)
    path = tmp_path / "source.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        assert isinstance(stream, VideoStream)
        stream.width = 64
        stream.height = 36
        stream.pix_fmt = "bgr0"
        stream.thread_count = 1
        for index in range(7):
            pixels = np.full((36, 64, 3), index * 20, dtype=np.uint8)  # [36,64,3]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    geometry = TransferGeometry((36, 64), (18, 32), "16,9", 0.0)
    media = read_direct_transfer_media(path, geometry=geometry, modality="rgb", max_frames=3)
    assert media.original_hw == (36, 64)
    assert media.frames.shape == (3, 3, 18, 32)
    assert media.fps == 24.0
    assert len(media.timestamps) == 3
    assert media.timestamps[-1] == pytest.approx(2 / 24, abs=0.001)
    assert media.frames[:, -1].eq(40).all()
