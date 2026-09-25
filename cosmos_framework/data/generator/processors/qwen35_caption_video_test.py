# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.augmentors.reasoner.timestamp import TimeStamp
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_with_subject_tracking import (
    TimeStampWithSubjectTracking,
)
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_without_end_time import TimeStampWithoutEndTime
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data import (
    TokenizeData,
    expand_video_frames_for_framewise,
)
from cosmos_framework.data.generator.reasoner.video_decoder_qwen import _video_decoder_qwen_func
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor
from cosmos_framework.utils.generator.video_preprocess import tensor_to_pil_images

pytestmark = [pytest.mark.L1, pytest.mark.CPU]

_DEFAULT_MODEL_PATH = Path(
    "/lustre/fsw/portfolios/cosmos/users/ypang/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)


def _synthetic_frames(count: int) -> list[Image.Image]:
    return [
        Image.fromarray(np.full((64, 64, 3), fill_value=frame_index * 40, dtype=np.uint8))
        for frame_index in range(count)
    ]


def _messages(frames: list[Image.Image], fps: float, frames_indices: list[int] | None = None) -> list[dict[str, Any]]:
    video = {"type": "video", "video": frames, "fps": fps, "max_pixels": 64 * 64}
    if frames_indices is not None:
        video["frames_indices"] = frames_indices
        video["total_num_frames"] = max(frames_indices) + 1
    return [
        {"role": "user", "content": [video, {"type": "text", "text": "Describe the video."}]},
        {"role": "assistant", "content": "The colors get brighter."},
    ]


@pytest.fixture(scope="module")
def processor() -> Qwen3VLProcessor:
    model_path = Path(os.environ.get("QWEN35_MODEL_PATH", _DEFAULT_MODEL_PATH))
    if not model_path.is_dir():
        pytest.skip(f"Qwen3.5 processor snapshot is unavailable: {model_path}")
    return Qwen3VLProcessor(str(model_path))


def _timestamp_values(processor: Qwen3VLProcessor, input_ids: torch.Tensor) -> list[float]:
    decoded = processor.decode(input_ids, skip_special_tokens=False)
    return [float(value) for value in re.findall(r"<([0-9.]+) seconds>", decoded)]


def _video_run_count(mm_token_type_ids: torch.Tensor) -> int:
    is_video = mm_token_type_ids == 2
    starts = is_video & torch.cat((torch.tensor([True]), ~is_video[:-1]))
    return int(starts.sum().item())


def test_qwen35_native_video_contract(processor: Qwen3VLProcessor) -> None:
    frames = _synthetic_frames(4)
    inputs = processor.apply_chat_template(_messages(frames, fps=1.0))

    grid_t, grid_h, grid_w = inputs["video_grid_thw"][0].tolist()
    assert grid_t == 2
    assert inputs["pixel_values_videos"].shape == (grid_t * grid_h * grid_w, 1536)
    assert int((inputs["mm_token_type_ids"] == 2).sum()) == grid_t * grid_h * grid_w // 4
    assert _video_run_count(inputs["mm_token_type_ids"]) == grid_t
    assert _timestamp_values(processor, inputs["input_ids"]) == pytest.approx([0.5, 2.5])
    assert "second_per_grid_ts" not in inputs


def test_qwen35_framewise_video_matches_image_style_tubelets(processor: Qwen3VLProcessor) -> None:
    frames = _synthetic_frames(4)
    expanded_frames, expanded_indices = expand_video_frames_for_framewise(frames, [0, 1, 2, 3], 2)
    inputs = processor.apply_chat_template(_messages(expanded_frames, fps=1.0, frames_indices=expanded_indices))

    grid_t, grid_h, grid_w = inputs["video_grid_thw"][0].tolist()
    assert grid_t == len(frames)
    assert inputs["pixel_values_videos"].shape == (grid_t * grid_h * grid_w, 1536)
    assert int((inputs["mm_token_type_ids"] == 2).sum()) == grid_t * grid_h * grid_w // 4
    assert _video_run_count(inputs["mm_token_type_ids"]) == grid_t
    assert _timestamp_values(processor, inputs["input_ids"]) == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert "second_per_grid_ts" not in inputs

    temporal_patches = inputs["pixel_values_videos"].reshape(-1, 3, 2, 16, 16)
    torch.testing.assert_close(temporal_patches[:, :, 0], temporal_patches[:, :, 1])


def test_qwen35_framewise_mrope_matches_timestamp_separated_video_groups(processor: Qwen3VLProcessor) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    frames = _synthetic_frames(4)
    expanded_frames, expanded_indices = expand_video_frames_for_framewise(frames, [0, 1, 2, 3], 2)
    inputs = processor.apply_chat_template(_messages(expanded_frames, fps=1.0, frames_indices=expanded_indices))

    class _RopeHarness:
        config = type("Config", (), {"vision_config": type("VisionConfig", (), {"spatial_merge_size": 2})()})()
        get_vision_position_ids = Qwen3_5Model.get_vision_position_ids

    input_ids = inputs["input_ids"].unsqueeze(0)
    mm_token_type_ids = inputs["mm_token_type_ids"].unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    position_ids, rope_deltas = Qwen3_5Model.get_rope_index(
        _RopeHarness(),
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        video_grid_thw=inputs["video_grid_thw"],
        attention_mask=attention_mask,
    )

    assert position_ids.shape == (3, 1, input_ids.shape[1])
    assert rope_deltas.shape == (1, 1)
    assert rope_deltas.item() == position_ids.max().item() + 1 - input_ids.shape[1]

    video_mask = mm_token_type_ids[0] == 2
    run_starts = torch.nonzero(video_mask & torch.cat((torch.tensor([True]), ~video_mask[:-1]))).flatten()
    run_ends = torch.nonzero(video_mask & torch.cat((~video_mask[1:], torch.tensor([True])))).flatten() + 1
    assert len(run_starts) == inputs["video_grid_thw"][0, 0].item()
    temporal_bases = []
    for start, end in zip(run_starts.tolist(), run_ends.tolist()):
        temporal_positions = position_ids[0, 0, start:end]
        assert torch.unique(temporal_positions).numel() == 1
        temporal_bases.append(temporal_positions[0].item())
    assert temporal_bases == sorted(temporal_bases)
    assert len(set(temporal_bases)) == len(temporal_bases)


def test_qwen35_framewise_decoder_preserves_exact_source_timestamps(processor: Qwen3VLProcessor) -> None:
    video_path = Path(__file__).parents[4] / "packages/cosmos-rl/tests/data/test_data_packer.mp4"
    if not video_path.is_file():
        pytest.skip(f"Tracked video fixture is unavailable: {video_path}")

    decoded = _video_decoder_qwen_func(
        key=video_path.name,
        data=video_path.read_bytes(),
        processor=processor,
        min_fps_thres=1,
        max_fps_thres=120,
        target_fps=1.0,
        min_video_token_length=16,
        max_video_token_length=512,
        video_temporal_mode="framewise",
    )
    assert decoded is not None
    frames = tensor_to_pil_images(decoded["videos"])
    source_indices = decoded["source_frames_indices"]
    assert len(frames) == len(source_indices)

    expanded_frames, expanded_indices = expand_video_frames_for_framewise(frames, source_indices, 2)
    inputs = processor.apply_chat_template(
        _messages(expanded_frames, fps=decoded["source_fps"], frames_indices=expanded_indices)
    )

    assert inputs["video_grid_thw"][0, 0].item() == len(frames)
    expected_timestamps = [round(index / decoded["source_fps"], 1) for index in source_indices]
    assert _timestamp_values(processor, inputs["input_ids"]) == pytest.approx(expected_timestamps)


@pytest.mark.parametrize("mode", ["native", "framewise"])
@pytest.mark.parametrize(
    "augmentor_class,output_format",
    [
        (TimeStamp, "temporal_localization"),
        (TimeStampWithoutEndTime, "temporal_localization"),
        (TimeStampWithSubjectTracking, "temporal_location_subject"),
    ],
)
def test_temporal_targets_share_the_tokenized_video_clock(
    processor: Qwen3VLProcessor,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    augmentor_class: type[TimeStamp | TimeStampWithoutEndTime | TimeStampWithSubjectTracking],
    output_format: str,
) -> None:
    # A 60-frame, 30 FPS clip sampled at 2 FPS selects these four source frames.
    # Clip offsets have already been removed by the decoder, including for crops.
    source_indices = [0, 20, 39, 59]
    media = {"videos": _synthetic_frames(4), "fps": 2.0}
    if mode == "framewise":
        media.update(source_frames_indices=source_indices, source_fps=30.0, source_total_num_frames=60)
    sample = {
        "__key__": "localization",
        "__url__": SimpleNamespace(root="temporal-test", path="clip.mp4"),
        "media": {"video": media},
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "video"},
                    {"type": "text", "text": "When does the person wave?"},
                ],
            },
            {
                "role": "assistant",
                "content": '[{"start":0.65,"end":1.25,"caption":"A person waves.","subject_id":"0"}]',
            },
        ],
    }

    def first_choice(choices: list[str]) -> str:
        return choices[0]

    monkeypatch.setattr(random, "choice", first_choice)
    augmentor = augmentor_class(
        output_format=output_format, urls_needs_timestamp=["temporal-test"], processor=processor
    )
    augmented = augmentor(sample)
    assert augmented is not None
    start, end = (0.7, 1.3) if mode == "framewise" else (0.2, 1.2)
    answer = str(start) if augmentor_class is TimeStampWithoutEndTime else f"{start}, {end}"
    assert augmented["conversation"][-1]["content"] == answer
    result = TokenizeData(processor=processor, video_temporal_mode=mode)(augmented)
    assert result is not None
    expected_clock = [0.0, 0.7, 1.3, 2.0] if mode == "framewise" else [0.2, 1.2]
    assert _timestamp_values(processor, result["input_ids"]) == pytest.approx(expected_clock)
    supervised = result["labels"][result["labels"] != -100]
    assert answer in processor.decode(supervised, skip_special_tokens=False)
