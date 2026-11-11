# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

vision_process = pytest.importorskip("qwen_vl_utils.vision_process")

from cosmos_framework.inference.reasoner import decoded_video_frames  # noqa: E402
from cosmos_framework.inference.reasoner.decoded_video_frames import (  # noqa: E402
    fetch_decoded_video,
    is_decoded_frames,
    process_vision_info,
)

_PATCH_SIZE = 16
_FRAMES = 8
# Per-frame bounds of an evaluation that mirrors an 8-frame training run with
# COSMOS_VIDEO_MAX_PIXELS = 8 * 147456.
_MIN_PIXELS = 4096
_MAX_PIXELS = 147456


def _frames(width: int, height: int, count: int = _FRAMES) -> list[Image.Image]:
    generator = np.random.default_rng(width * 10007 + height)
    return [Image.fromarray(generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)) for _ in range(count)]


def _element(frames: list[Image.Image], fps: float = 6.5625) -> dict:
    return {
        "type": "video",
        "video": frames,
        "fps": fps,
        "min_pixels": _MIN_PIXELS,
        "max_pixels": _MAX_PIXELS,
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((448, 256), (448, 256)),
        ((480, 288), (480, 288)),
        ((512, 320), (480, 288)),
        ((640, 360), (512, 288)),
        ((640, 480), (416, 320)),
        ((854, 480), (512, 256)),
        ((1280, 720), (512, 288)),
        ((1920, 1080), (512, 288)),
    ],
)
def test_decoded_frames_are_resized_once_on_the_model_grid(source, expected):
    video, _ = fetch_decoded_video(_element(_frames(*source)), _PATCH_SIZE, vision_process)

    assert tuple(video.shape) == (_FRAMES, 3, expected[1], expected[0])
    assert video.dtype == torch.float32


@pytest.mark.parametrize("source", [(480, 288), (640, 360), (640, 480), (1280, 720)])
def test_decoded_frames_get_the_size_of_a_video_that_qwen_vl_utils_reads_itself(monkeypatch, source):
    frames = _frames(*source)
    fps = 6.5625

    def read_video(element):
        video = torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in frames])  # [T,3,H,W]
        metadata = {"fps": fps, "frames_indices": list(range(len(frames))), "total_num_frames": len(frames)}
        return video, metadata, fps

    monkeypatch.setitem(vision_process.VIDEO_READER_BACKENDS, "cosmos-test", read_video)
    monkeypatch.setattr(vision_process, "get_video_reader_backend", lambda: "cosmos-test")
    path_element = {"type": "video", "video": "clip.mp4", "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS}
    expected_video, expected_metadata = vision_process.fetch_video(
        path_element, image_patch_size=_PATCH_SIZE, return_video_metadata=True
    )

    video, metadata = fetch_decoded_video(_element(frames, fps), _PATCH_SIZE, vision_process)

    assert video.shape == expected_video.shape
    assert video.dtype == expected_video.dtype
    assert metadata == expected_metadata


@pytest.mark.parametrize("source", [(448, 256), (480, 288), (640, 360), (640, 480), (854, 480), (1280, 720)])
def test_decoded_frames_match_the_training_video_processor(source):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLVideoProcessor"):
        pytest.skip("transformers has no Qwen3VLVideoProcessor")
    processor = transformers.Qwen3VLVideoProcessor(
        size={"shortest_edge": _MIN_PIXELS * _FRAMES, "longest_edge": _MAX_PIXELS * _FRAMES},
        patch_size=_PATCH_SIZE,
    )
    frames = _frames(*source)
    fps = 6.5625
    training_metadata = {"fps": fps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
    # VideoSFTProcessor: decoded frames, the processor's own resize, no frame sampling.
    training = processor(
        videos=[frames], video_metadata=[training_metadata], do_sample_frames=False, return_tensors="pt"
    )

    video, metadata = fetch_decoded_video(_element(frames, fps), _PATCH_SIZE, vision_process)
    # CosmosFrameworkRuntime._prepare_transformers_batch: frames resized before the processor.
    evaluation = processor(
        videos=[video], video_metadata=[metadata], do_resize=False, do_sample_frames=False, return_tensors="pt"
    )

    assert metadata == training_metadata
    assert torch.equal(evaluation["video_grid_thw"], training["video_grid_thw"])
    assert torch.equal(evaluation["pixel_values_videos"], training["pixel_values_videos"])


def test_decoded_frames_keep_the_measured_frame_rate_and_an_odd_count():
    video, metadata = fetch_decoded_video(_element(_frames(480, 288, count=5), fps=3.75), _PATCH_SIZE, vision_process)

    assert video.shape[0] == 5
    assert metadata == {"fps": 3.75, "frames_indices": [0, 1, 2, 3, 4], "total_num_frames": 5}


def test_decoded_frames_require_a_frame_rate():
    element = _element(_frames(480, 288))
    del element["fps"]

    with pytest.raises(ValueError, match="positive fps"):
        fetch_decoded_video(element, _PATCH_SIZE, vision_process)


def test_decoded_frames_preserve_an_explicit_maximum_below_the_default_minimum():
    element = _element(_frames(640, 360))
    del element["min_pixels"]
    element["max_pixels"] = 65536

    video, _ = fetch_decoded_video(element, _PATCH_SIZE, vision_process)

    assert video.shape[-2] * video.shape[-1] <= 65536


def test_is_decoded_frames_only_accepts_pil_frame_sequences():
    assert is_decoded_frames(_frames(64, 64, count=2))
    assert is_decoded_frames(tuple(_frames(64, 64, count=2)))
    assert not is_decoded_frames([])
    assert not is_decoded_frames("clip.mp4")
    assert not is_decoded_frames(["frame0.jpg", "frame1.jpg"])


def test_process_vision_info_only_reroutes_decoded_frames(monkeypatch):
    calls: list[tuple[str, dict]] = []
    path_video = (torch.zeros(2, 3, 32, 32), {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2})

    def fetch_video(element, **kwargs):
        calls.append(("video", kwargs))
        return path_video, 2.0

    def fetch_image(element, **kwargs):
        calls.append(("image", kwargs))
        return "image"

    def fetch_decoded(element, image_patch_size, module):
        calls.append(("decoded", {"image_patch_size": image_patch_size}))
        return "decoded"

    monkeypatch.setattr(vision_process, "fetch_video", fetch_video)
    monkeypatch.setattr(vision_process, "fetch_image", fetch_image)
    monkeypatch.setattr(decoded_video_frames, "fetch_decoded_video", fetch_decoded)
    conversations = [
        [{"role": "user", "content": [_element(_frames(64, 64, count=2)), {"type": "text", "text": "q"}]}],
        [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}, {"type": "image", "image": "a.jpg"}]}],
        [{"role": "user", "content": [{"type": "video", "video": ["frame0.jpg", "frame1.jpg"]}]}],
    ]

    images, videos, video_kwargs = process_vision_info(conversations, image_patch_size=_PATCH_SIZE)

    assert images == ["image"]
    assert videos == ["decoded", path_video, path_video]
    assert video_kwargs == {"do_sample_frames": False}
    assert calls == [
        ("decoded", {"image_patch_size": _PATCH_SIZE}),
        ("video", {"return_video_sample_fps": True, "image_patch_size": _PATCH_SIZE, "return_video_metadata": True}),
        ("image", {"image_patch_size": _PATCH_SIZE}),
        ("video", {"return_video_sample_fps": True, "image_patch_size": _PATCH_SIZE, "return_video_metadata": True}),
    ]


def test_process_vision_info_returns_none_without_media():
    conversation = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]

    assert process_vision_info(conversation, image_patch_size=_PATCH_SIZE) == (None, None, {"do_sample_frames": False})


def test_decoded_frame_rate_reaches_the_prompt_timestamps():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLProcessor"):
        pytest.skip("transformers has no Qwen3VLProcessor")
    # Eight frames drawn from 39 source frames at 32 fps: stride 39 / 8, 6.5641 fps.
    fps = 32.0 / (39 / 8)

    _, metadata = fetch_decoded_video(_element(_frames(480, 288), fps), _PATCH_SIZE, vision_process)
    timestamps = transformers.Qwen3VLProcessor._calculate_timestamps(
        None, metadata["frames_indices"], metadata["fps"], merge_size=2
    )

    assert [round(value, 1) for value in timestamps] == [0.1, 0.4, 0.7, 1.0]


def test_task_conversation_keeps_the_measured_rate_of_decoded_frames():
    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

    frames = _frames(64, 64, count=2)
    task = {
        "question": "q",
        "media_mode": "video",
        "media_paths": ["clip.mp4"],
        "_framework_decoded_media": [{"frames": frames, "fps": 6.5625}],
    }
    vision_config = {"fps": 2.0, "num_frames": 2, "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS}

    decoded_item = CosmosFrameworkRuntime._task_conversation(task, vision_config)[0]["content"][0]
    del task["_framework_decoded_media"]
    path_item = CosmosFrameworkRuntime._task_conversation(task, vision_config)[0]["content"][0]

    assert decoded_item["fps"] == 6.5625
    assert decoded_item["max_pixels"] == _MAX_PIXELS
    assert path_item["video"] == "clip.mp4"
    assert path_item["fps"] == 2.0
