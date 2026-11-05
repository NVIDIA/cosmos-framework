# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.inference import vision


def test_pyav_video_decoder_preserves_rgb_pixels_frame_order_and_fps(tmp_path: Path) -> None:
    path = tmp_path / "lossless.mkv"
    expected = np.arange(4 * 8 * 10 * 3, dtype=np.uint8).reshape(4, 8, 10, 3)  # [T,H,W,C]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=12)
        stream.width = 10
        stream.height = 8
        stream.pix_fmt = "bgr0"
        for pixels in expected:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    actual, fps = vision.decode_video_pyav_thwc_uint8(path)  # [T,H,W,C]

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.uint8
    assert fps == 12.0
    np.testing.assert_array_equal(actual.numpy(), expected)
    truncated, truncated_fps = vision.read_media_frames(path, max_frames=2, video_decoder="pyav")  # [C,T,H,W]
    np.testing.assert_array_equal(truncated.permute(1, 2, 3, 0).numpy(), expected[:2])  # [T,H,W,C]
    assert truncated_fps == fps


def test_probe_media_hw_matches_full_decode_shape(tmp_path: Path) -> None:
    """The geometry probe agrees with the full PyAV decode and with PIL for images."""
    path = tmp_path / "lossless.mkv"
    pixels = np.zeros((3, 8, 10, 3), dtype=np.uint8)  # [T,H,W,C]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=12)
        stream.width = 10
        stream.height = 8
        stream.pix_fmt = "bgr0"
        for frame_pixels in pixels:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame_pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    full, _ = vision.read_media_frames(path, max_frames=3, video_decoder="pyav")  # [C,T,H,W]
    assert vision.probe_media_hw(path, video_decoder="pyav") == (full.shape[2], full.shape[3]) == (8, 10)

    image_path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((6, 9, 3), dtype=np.uint8)).save(image_path)
    assert vision.probe_media_hw(image_path) == (6, 9)
    with pytest.raises(ValueError, match="Unsupported media extension"):
        vision.probe_media_hw(tmp_path / "clip.txt")


@pytest.mark.parametrize("decoder", [None, "torchcodec", "pyav"])
def test_media_reader_selects_explicit_decoder_and_preserves_torchcodec_default(
    monkeypatch: pytest.MonkeyPatch, decoder: str | None
) -> None:
    pixels = torch.arange(3 * 2 * 4 * 3, dtype=torch.uint8).reshape(3, 2, 4, 3)  # [T,H,W,C]
    torchcodec = Mock(return_value=(pixels, 30.0))
    pyav = Mock(return_value=(pixels, 24.0))
    monkeypatch.setattr(vision, "decode_video_thwc_uint8", torchcodec)
    monkeypatch.setattr(vision, "decode_video_pyav_thwc_uint8", pyav)
    kwargs = {} if decoder is None else {"video_decoder": decoder}

    actual, fps = vision.read_media_frames(Path("input.mp4"), max_frames=2, **kwargs)  # [C,T,H,W]

    assert torch.equal(actual, pixels[:2].permute(3, 0, 1, 2))
    selected, unused = (pyav, torchcodec) if decoder == "pyav" else (torchcodec, pyav)
    selected.assert_called_once_with(Path("input.mp4"))
    unused.assert_not_called()
    assert fps == (24.0 if decoder == "pyav" else 30.0)


def test_resize_reader_forwards_native_decoder_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    pixels = torch.zeros(3, 2, 4, 4, dtype=torch.uint8)  # [C,T,H,W]
    reader = Mock(return_value=(pixels, 30.0))
    monkeypatch.setattr(vision, "read_media_frames", reader)
    monkeypatch.setattr(vision, "_resize_and_center_crop", lambda frames, height, width, **kwargs: frames)

    vision.read_and_resize_media(
        Path("input.mp4"), resolution="480", aspect_ratio="1,1", max_frames=2, video_decoder="pyav"
    )

    reader.assert_called_once_with(Path("input.mp4"), max_frames=2, video_decoder="pyav")


# --- load_conditioning_video: range decode on the host thread budget ----------------------------

_CLIP_T, _CLIP_H, _CLIP_W = 7, 12, 20  # synthetic clip, [T,H,W,3]
_TARGET_H, _TARGET_W = 8, 10  # needs both a resize and a crop
_THREAD_BUDGET = 3
_TORCHCODEC_VIDEO = "cosmos_framework.utils.generator.torchcodec_video"


@pytest.fixture
def fake_torchcodec(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Install a ``torchcodec.decoders.VideoDecoder`` backed by a synthetic uint8 clip.

    It records its constructor arguments and every decode call. Its frames are a plain
    ``torch.empty`` allocation, so they land on whatever default device the caller has active,
    as the tensors a real decoder builds in Python do. The CPU test environment cannot load the
    real torchcodec, which is also why ``vision`` must import it lazily.
    """
    pixels = np.random.default_rng(0).integers(0, 256, size=(_CLIP_T, _CLIP_H, _CLIP_W, 3), dtype=np.uint8)
    record = SimpleNamespace(pixels=pixels, inits=[], calls=[])

    class VideoDecoder:
        def __init__(
            self,
            source: object,
            *,
            dimension_order: str = "NCHW",
            num_ffmpeg_threads: int = 1,
            device: str | None = None,
            seek_mode: str = "exact",
        ) -> None:
            record.inits.append(
                {
                    "source": source,
                    "dimension_order": dimension_order,
                    "num_ffmpeg_threads": num_ffmpeg_threads,
                    "device": device,
                    "seek_mode": seek_mode,
                }
            )
            self._dimension_order = dimension_order
            self.metadata = SimpleNamespace(num_frames=_CLIP_T, average_fps=30.0, height=_CLIP_H, width=_CLIP_W)

        def __len__(self) -> int:
            return _CLIP_T

        def _frames(self, indices: list[int]) -> SimpleNamespace:
            data = torch.empty((len(indices), _CLIP_H, _CLIP_W, 3), dtype=torch.uint8)  # [T,H,W,3]
            data.copy_(torch.from_numpy(pixels[indices]))
            return SimpleNamespace(data=data.permute(0, 3, 1, 2) if self._dimension_order == "NCHW" else data)

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            record.calls.append(("get_frames_at", list(indices)))
            return self._frames(list(indices))

        def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> SimpleNamespace:
            record.calls.append(("get_frames_in_range", start, stop))
            return self._frames(list(range(start, stop, step)))

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = VideoDecoder  # type: ignore[attr-defined]
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.decoders = decoders  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    # The mapped full-decode helper binds VideoDecoder when it is imported. Import it afresh
    # against this fake, and drop that copy afterwards, so no test sees another test's decoder.
    had_torchcodec_video = _TORCHCODEC_VIDEO in sys.modules
    monkeypatch.delitem(sys.modules, _TORCHCODEC_VIDEO, raising=False)
    # raising=False: a loader that does not take the budget from here fails the assertions below
    # rather than this setup.
    monkeypatch.setattr(vision, "host_threads_per_local_rank", lambda: _THREAD_BUDGET, raising=False)
    yield record
    if not had_torchcodec_video:
        sys.modules.pop(_TORCHCODEC_VIDEO, None)


def _decode_all_then_slice(pixels: np.ndarray, max_frames: int, keep: str) -> torch.Tensor:  # returns [3,T,H,W]
    """What ``load_conditioning_video`` returned when it decoded every frame and sliced afterwards."""
    frames = torch.from_numpy(pixels)  # [T,H,W,3]
    frames = frames[-max_frames:] if keep == "last" else frames[:max_frames]  # [T,H,W,3]
    resized = vision._resize_and_center_crop(frames.permute(0, 3, 1, 2).float(), _TARGET_H, _TARGET_W)  # [T,3,h,w]
    return (resized / 127.5 - 1.0).permute(1, 0, 2, 3)  # [3,T,h,w]


@pytest.mark.parametrize(("keep", "expected_call"), [("first", (0, 5)), ("last", (_CLIP_T - 5, _CLIP_T))])
def test_conditioning_video_decodes_only_the_kept_frames_on_the_thread_budget(
    fake_torchcodec: SimpleNamespace, keep: str, expected_call: tuple[int, int]
) -> None:
    """One range decode of the kept frames, with FFmpeg on this rank's share of the host CPUs.

    A WSM control clip is far longer than the rollout it conditions, and an i2v camera keeps
    only frame 0; decoding every frame and slicing afterwards, on one FFmpeg thread, was most of
    the media preparation.
    """
    vision.load_conditioning_video(Path("clip.mp4"), _TARGET_H, _TARGET_W, 5, keep=keep)  # type: ignore[arg-type]

    decoders = [(init["num_ffmpeg_threads"], init["device"]) for init in fake_torchcodec.inits]
    assert decoders == [(_THREAD_BUDGET, "cpu")]
    assert fake_torchcodec.calls == [("get_frames_in_range", *expected_call)]  # and never get_frames_at


def test_conditioning_video_resizes_on_the_thread_budget_and_restores_the_caller_s(
    fake_torchcodec: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torchrun leaves torch at OMP_NUM_THREADS=1; the antialiased resize gets the budget, then gives it back."""
    resize = vision._resize_and_center_crop
    threads_in_resize: list[int] = []

    def recording_resize(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        threads_in_resize.append(torch.get_num_threads())
        return resize(frames, target_h, target_w)

    monkeypatch.setattr(vision, "_resize_and_center_crop", recording_resize)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vision.load_conditioning_video(Path("clip.mp4"), _TARGET_H, _TARGET_W, _CLIP_T)
        threads_after = torch.get_num_threads()
    finally:
        torch.set_num_threads(previous)

    assert threads_in_resize == [_THREAD_BUDGET]
    assert threads_after == 1


@pytest.mark.parametrize(
    ("keep", "max_frames"),
    [(keep, max_frames) for keep in ("first", "last") for max_frames in (1, 5, _CLIP_T, _CLIP_T + 3)]
    + [("last", 0), ("first", 0)],
)
def test_conditioning_video_pixels_match_decoding_every_frame_then_slicing(
    fake_torchcodec: SimpleNamespace, keep: str, max_frames: int
) -> None:
    """Python slice semantics, edge cases included: ``keep="last"`` with 0 keeps every frame, as
    ``frames[-0:]`` did, and ``keep="first"`` with 0 keeps none."""
    actual = vision.load_conditioning_video(
        Path("clip.mp4"),
        _TARGET_H,
        _TARGET_W,
        max_frames,
        keep=keep,  # type: ignore[arg-type]
    )  # [3,T,h,w]

    expected = _decode_all_then_slice(fake_torchcodec.pixels, max_frames, keep)  # [3,T,h,w]
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("max_frames", [5, 0])
def test_conditioning_video_stays_on_the_host_under_a_default_device_context(
    fake_torchcodec: SimpleNamespace, max_frames: int
) -> None:
    """Generation calls this under a default-CUDA device context; the frames must still be host tensors.

    0 frames takes the path that allocates the empty selection itself.
    """
    with torch.device("meta"):
        actual = vision.load_conditioning_video(Path("clip.mp4"), _TARGET_H, _TARGET_W, max_frames)  # [3,T,h,w]

    assert actual.device.type == "cpu"
    assert torch.equal(actual, _decode_all_then_slice(fake_torchcodec.pixels, max_frames, "first"))


@pytest.mark.parametrize(("keep", "max_frames"), [("first", 1), ("first", 10), ("last", 7), ("first", 27), ("last", 0)])
def test_conditioning_video_range_decode_matches_the_full_decode_of_a_b_frame_h264_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: str, max_frames: int
) -> None:
    """The real decoder: a range decode on several FFmpeg threads gives the full decode's pixels.

    B-frames are what make decode order differ from presentation order, so they are where a
    range decode could plausibly return different frames. Skipped where torchcodec cannot load.
    """
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception as error:  # ImportError, or RuntimeError when FFmpeg's libraries do not load
        pytest.skip(f"torchcodec does not load here ({type(error).__name__})")

    path = tmp_path / "b_frames.mp4"
    num_frames, height, width = 24, 48, 64
    yy, xx = np.mgrid[0:height, 0:width]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"bf": "3", "g": "12"}
        for t in range(num_frames):
            pixels = np.stack([(xx * 4 + t * 7) % 256, (yy * 5 + t * 3) % 256, ((xx + yy) * 2 + t * 11) % 256], -1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels.astype(np.uint8), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        assert av.video.frame.PictureType.B in {frame.pict_type for frame in container.decode(video=0)}

    full, _ = vision.decode_video_thwc_uint8(path)  # [T,H,W,3]
    kept = full[-max_frames:] if keep == "last" else full[:max_frames]  # [T,H,W,3]
    resized = vision._resize_and_center_crop(kept.permute(0, 3, 1, 2).float(), 32, 40)  # [T,3,32,40]
    expected = (resized / 127.5 - 1.0).permute(1, 0, 2, 3)  # [3,T,32,40]

    monkeypatch.setattr(vision, "host_threads_per_local_rank", lambda: 4)
    actual = vision.load_conditioning_video(path, 32, 40, max_frames, keep=keep)  # type: ignore[arg-type]

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
