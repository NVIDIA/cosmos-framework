# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transforms import (
    ActionTransformPipeline,
    add_action_mode_metadata,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo


@pytest.mark.L0
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "forward_dynamics",
            "Open the drawer. Predict the future video given the image and action.",
        ),
        (
            "inverse_dynamics",
            "Predict the underlying action given the video.",
        ),
        (
            "wam",
            "Open the drawer. Predict the future video and action given the image.",
        ),
    ],
)
def test_add_action_mode_metadata_supports_plain_captions(
    mode: str,
    expected: str,
) -> None:
    caption = "  Open the drawer.  "

    result = add_action_mode_metadata(caption, mode)

    assert result == expected


@pytest.mark.L0
def test_add_action_mode_metadata_supports_structured_captions() -> None:
    caption = {
        "actions": [{"description": "Open the drawer."}],
        "duration": "2s",
    }

    forward_dynamics = add_action_mode_metadata(caption, "forward_dynamics")
    inverse_dynamics = add_action_mode_metadata(caption, "inverse_dynamics")

    assert forward_dynamics == {
        "actions": caption["actions"],
        "mode_description": "Predict the future video given the image and action.",
        "duration": "2s",
    }
    assert inverse_dynamics == {
        "mode_description": "Predict the underlying action given the video.",
        "duration": "2s",
    }


@pytest.mark.L0
def test_action_prompt_json_formatter_builds_requested_structure() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    idle_frames = torch.tensor(2)  # []
    data_dict = {
        "ai_caption": "Pick up the cup",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "concat_view",
        "additional_view_description": "The top row is the wrist camera and the bottom row is the scene camera.",
        "idle_frames": idle_frames,
    }

    result = formatter(data_dict)

    prompt = result["ai_caption"]
    assert list(prompt.keys()) == ["cinematography", "actions", "duration", "fps", "resolution", "aspect_ratio"]
    assert list(prompt["actions"][0].keys()) == ["time", "description", "idle_frame"]
    assert prompt == {
        "cinematography": {
            "framing": (
                "This video contains concatenated views from multiple camera perspectives. "
                "The top row is the wrist camera and the bottom row is the scene camera."
            )
        },
        "actions": [
            {
                "time": "0:00-0:00",
                "description": "Pick up the cup.",
                "idle_frame": "2 out of 11.",
            }
        ],
        "duration": "0s",
        "fps": 24.0,
        "resolution": {"H": 480, "W": 640},
        "aspect_ratio": "4,3",
    }
    assert "additional_view_description" not in result


@pytest.mark.L0
def test_video_padding_round_trips_to_unpadded_region() -> None:
    video = torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5)  # [C,T,H,W]
    data_dict = {"video": video}

    padded = reflection_pad_to_target(
        data_dict,
        keys=["video"],
        keep_aspect_ratio=True,
        target_w=8,
        target_h=6,
    )
    round_tripped = remove_reflection_padding(padded["video"], padded["image_size"])  # [C,T,H,W]

    assert padded["video"].shape == (3, 2, 6, 8)
    torch.testing.assert_close(round_tripped, video)


@pytest.mark.L0
def test_action_prompt_json_formatter_drops_empty_fields() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    data_dict = {
        "ai_caption": "Pick up the cup.",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "third_person_view",
    }

    result = formatter(data_dict)

    assert result["ai_caption"]["actions"] == [
        {
            "time": "0:00-0:00",
            "description": "Pick up the cup.",
        }
    ]


@pytest.mark.L0
def test_action_prompt_json_formatter_drops_empty_viewpoint() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    data_dict = {
        "ai_caption": "Pick up the cup.",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
    }

    result = formatter(data_dict)

    assert "cinematography" not in result["ai_caption"]


@pytest.mark.L0
def test_action_transform_pipeline_disables_mode_specific_prompt_by_default() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": torch.zeros(3, 17, 256, 256),  # [C,T,H,W]
        "action": torch.zeros(16, 2),  # [T,D]
        "mode": "inverse_dynamics",
        "domain_id": torch.tensor(0),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == "Open the drawer."


@pytest.mark.L0
def test_action_transform_pipeline_json_prompt_toggle() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        format_prompt_as_json=True,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 192, 320)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "wam",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    prompt = result["ai_caption"]
    assert isinstance(prompt, dict)
    assert list(prompt.keys()) == [
        "actions",
        "mode_description",
        "cinematography",
        "duration",
        "fps",
        "resolution",
        "aspect_ratio",
    ]
    assert list(prompt["actions"][0].keys()) == ["time", "description", "idle_frame"]
    assert prompt["mode_description"] == "Predict the future video and action given the image."
    assert prompt["cinematography"] == {
        "framing": "This video is captured from a third-person perspective looking towards the agent from the front."
    }
    assert prompt["actions"] == [
        {
            "time": "0:00-0:02",
            "description": "Open the drawer.",
            "idle_frame": "3 out of 16.",
        }
    ]
    assert prompt["duration"] == "2s"
    assert prompt["fps"] == 8.0
    assert prompt["resolution"] == {"H": 192, "W": 320}
    assert prompt["aspect_ratio"] == "16,9"
    assert result["action"].shape == (16, 4)
    torch.testing.assert_close(result["action_raw"], action)


@pytest.mark.L0
def test_action_transform_pipeline_json_prompt_respects_duration_fps_toggle() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_duration_fps_timestamps=False,
        format_prompt_as_json=True,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 192, 320)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "mode": "wam",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    prompt = result["ai_caption"]
    assert isinstance(prompt, dict)
    assert list(prompt.keys()) == ["actions", "mode_description", "cinematography", "resolution", "aspect_ratio"]
    assert list(prompt["actions"][0].keys()) == ["description", "idle_frame"]
    assert prompt["mode_description"] == "Predict the future video and action given the image."
    assert prompt["actions"] == [
        {
            "description": "Open the drawer.",
            "idle_frame": "3 out of 16.",
        }
    ]
    assert result["action"].shape == (16, 4)
    torch.testing.assert_close(result["action_raw"], action)


@pytest.mark.L0
def test_action_transform_pipeline_preserves_explicit_action_valid_mask() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    data_dict = {
        "ai_caption": "Move.",
        "video": torch.zeros(3, 3, 256, 256),
        "action": torch.zeros(2, 2),
        "mode": "wam",
        "domain_id": torch.tensor(0),
    }
    valid_mask = torch.tensor([True, False])

    result = pipeline(data_dict, resolution="256", action_valid_mask=valid_mask)

    torch.testing.assert_close(result["action_valid_mask"], torch.tensor([True, False, False, False]))
    torch.testing.assert_close(result["action_processing_record"].action_valid_mask, valid_mask)


@pytest.mark.L0
def test_action_transform_pipeline_supports_compact_streaming_video() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        format_prompt_as_json=True,
    )
    action = torch.zeros(960, 2)
    data_dict = {
        "ai_caption": "Move the camera.",
        "video": torch.zeros(3, 1, 192, 320),
        "video_num_frames": 961,
        "action": action,
        "conditioning_fps": torch.tensor(24),
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),
        "viewpoint": "ego_view",
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"]["duration"] == "40s"
    assert result["sequence_plan"].condition_frame_indexes_vision == [0]
    assert len(result["sequence_plan"].condition_frame_indexes_action) == 960
    assert "video_num_frames" not in result
    assert result["video"].shape[1] == 1


@pytest.mark.L0
def test_action_transform_pipeline_plain_prompt_uses_logical_video_duration() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
    )
    data_dict = {
        "ai_caption": "Move the camera.",
        "video": torch.zeros(3, 1, 192, 320),
        "video_num_frames": 961,
        "action": torch.zeros(960, 2),
        "conditioning_fps": torch.tensor(24),
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),
        "viewpoint": "ego_view",
    }

    result = pipeline(data_dict, resolution="256")

    assert "The video is 40.0 seconds long and is of 24 FPS." in result["ai_caption"]
    assert result["video"].shape[1] == 1


@pytest.mark.L0
def test_action_transform_pipeline_keeps_ai_caption_string_path() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "wam",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == (
        "Open the drawer. "
        "Predict the future video and action given the image. "
        "This video is captured from a third-person perspective looking towards the agent from the front. "
        "The video is 2.0 seconds long and is of 8 FPS. "
        "This video is of 256x256 resolution. "
        "IdleFrames: 3 out of 16."
    )
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_groups_registered_caption_postfixes() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_action_caption_semantics=True,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer",
        "video": video,
        "action": action,
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "action_caption_attributes": {
            "domain_postfix": "Domain.",
            "embodiment_postfix": "Embodiment.",
            "view_postfix": "Dataset-specific view.",
            "subject_postfix": "Caption subject.",
        },
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == ("Open the drawer. Domain. Embodiment. Dataset-specific view. Caption subject.")
    assert "action_caption_attributes" not in result
    assert "third-person perspective" not in result["ai_caption"]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_avoids_generic_viewpoint_with_attribute_postfixes() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_action_caption_semantics=True,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": torch.zeros(3, 17, 256, 256),  # [C,T,H,W]
        "action": torch.zeros(16, 2),  # [T,D]
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == "Open the drawer."


@pytest.mark.L0
def test_action_transform_pipeline_preserves_empty_caption_with_attribute_postfixes() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_action_caption_semantics=True,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    data_dict = {
        "ai_caption": "",
        "video": torch.zeros(3, 17, 256, 256),  # [C,T,H,W]
        "action": torch.zeros(16, 2),  # [T,D]
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "action_caption_attributes": {},
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == ""


@pytest.mark.L0
def test_action_transform_pipeline_keeps_idle_frames_for_forward_dynamics() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"].startswith("Open the drawer. Predict the future video given the image and action.")
    assert "IdleFrames: 3 out of 16." in result["ai_caption"]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_skips_idle_frames_for_inverse_dynamics_string_path() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "inverse_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"].startswith(
        "Predict the underlying action given the video. "
        "This video is captured from a third-person perspective looking towards the agent from the front."
    )
    assert "Open the drawer" not in result["ai_caption"]
    assert "IdleFrames" not in result["ai_caption"]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_skips_idle_frames_for_inverse_dynamics_json_prompt() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        format_prompt_as_json=True,
        enable_mode_specific_prompt=True,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "inverse_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    prompt = result["ai_caption"]
    assert isinstance(prompt, dict)
    assert prompt["mode_description"] == "Predict the underlying action given the video."
    assert "actions" not in prompt
    assert "Open the drawer" not in str(prompt)
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_prompt_json_formatter_matches_video_json_common_metadata() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 23, 192, 320)  # [C,T,H,W]
    image_size = torch.tensor([192, 320, 192, 320])  # [4]
    fps = torch.tensor(8.0)  # []
    action_data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": torch.zeros(22, 2),  # [T,D]
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    action_prompt = formatter(action_data_dict)["ai_caption"]

    video_data_dict = {
        "ai_caption": {
            "cinematography": {
                "framing": "This video is captured from a third-person perspective looking towards the agent from the front."
            },
            "actions": [
                {
                    "time": "0:00-0:03",
                    "description": "Open the drawer.",
                }
            ],
        },
        "video": video,
        "conditioning_fps": fps,
        "image_size": image_size,
        "__url__": SimpleNamespace(meta=SimpleNamespace(opts={"aspect_ratio": "16,9"})),
    }
    duration_augmentor = DurationFPSTextTimeStamps(
        input_keys=["ai_caption", "video", "conditioning_fps"],
        args={"caption_key": "ai_caption", "video_key": "video", "fps_key": "conditioning_fps"},
    )
    resolution_augmentor = ResolutionTextInfo(
        input_keys=["ai_caption", "video", "image_size"],
        args={"caption_key": "ai_caption", "video_key": "video"},
    )
    duration_augmentor(video_data_dict)
    resolution_augmentor(video_data_dict)
    video_prompt = video_data_dict["ai_caption"]

    common_top_level_keys = ["cinematography", "duration", "fps", "resolution", "aspect_ratio"]
    assert {key: action_prompt[key] for key in common_top_level_keys} == {
        key: video_prompt[key] for key in common_top_level_keys
    }
    assert action_prompt["actions"][0]["time"] == video_prompt["actions"][0]["time"]
    assert action_prompt["actions"][0]["description"] == video_prompt["actions"][0]["description"]
    assert action_prompt["actions"][0]["idle_frame"] == "3 out of 22."
