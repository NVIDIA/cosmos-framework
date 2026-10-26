# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.data.generator.action.action_caption_attribute_adapter import (
    CaptionSubjectMappings,
    ViewCompositionMappings,
)
from cosmos_framework.data.generator.action.utils.camera_views import native_camera_caption_prefixes
from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.utils.transforms import (
    ActionTransformPipeline,
    add_action_mode_metadata,
    reflection_pad_to_target,
    remove_reflection_padding,
)
from cosmos_framework.data.generator.action.utils.unified_action_schema import UNIFIED_ACTION_DIM
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["policy", "wam", "inverse_dynamics"])
@pytest.mark.parametrize("num_views", [1, 2])
@pytest.mark.parametrize("json_prompt", [False, True])
@pytest.mark.parametrize("cfg_dropped", [False, True])
def test_action_camera_roles_preserve_camera_layout_and_task_prompt(
    monkeypatch: pytest.MonkeyPatch, mode: str, num_views: int, json_prompt: bool, cfg_dropped: bool
) -> None:
    class Processor:
        def tokenize_text(self, caption: str, **kwargs: object) -> list[int]:
            return list(caption.encode("utf-8")) or [0]

    monkeypatch.setattr(
        "cosmos_framework.data.generator.augmentors.text_tokenizer.lazy_instantiate", lambda _: Processor()
    )
    names = ("observation.images.right_wrist", "observation.images.left_wrist")[:num_views]
    videos = [torch.full((3, 17, 4, 6), float(index)) for index in range(num_views)]
    action = torch.zeros(16, UNIFIED_ACTION_DIM)
    # Identity rotations in each canonical pose slot; openness is valid in both wrists.
    action[:, [3, 7, 12, 16, 37, 41]] = 1.0
    action[:, [33, 58]] = 0.5
    data = {
        "ai_caption": "Move the camera.",
        "video": torch.cat(videos, dim=1),
        "camera_videos": videos,
        "camera_view_names": names,
        "camera_caption_prefixes": native_camera_caption_prefixes(names),
        "camera_action_pose_convention": "backward_chunk_anchored_16f",
        "camera_action_gripper_width_m": 1.0,
        "action": action,
        "conditioning_fps": torch.tensor(8),
        "mode": mode,
        "viewpoint": "separate_camera_views",
    }
    options: dict[str, Any] = dict(
        pad_keys=[],
        max_action_dim=64,
        append_viewpoint_info=True,
        append_resolution_info=False,
        format_prompt_as_json=json_prompt,
        enable_mode_specific_prompt=True,
    )
    valid_mask = torch.ones(UNIFIED_ACTION_DIM, dtype=torch.bool)
    baseline = ActionTransformPipeline(**options)(copy.deepcopy(data), "256", action_valid_mask=valid_mask)
    pipeline = ActionTransformPipeline(
        **options, tokenizer_config={}, cfg_dropout_rate=0.1, append_camera_role_to_caption=True
    )
    monkeypatch.setattr("random.random", lambda: 0.05 if cfg_dropped else 0.9)
    result = pipeline(data, "256", action_valid_mask=valid_mask)

    if cfg_dropped:
        assert result["ai_caption"] == [""] * num_views
    else:
        for name, caption in zip(names, result["ai_caption"]):
            assert caption.startswith(f"{native_camera_caption_prefixes(names)[name]}\n\n")
            assert "View:" not in caption
            assert all(camera_name not in caption for camera_name in names)
            assert ("Move the camera." in caption) is (mode != "inverse_dynamics")
            assert "2s" in caption if json_prompt else "2.0 seconds" in caption
            assert "separately" in caption
            assert "concatenated" not in caption
    assert result["text_token_lengths"] == [len(tokens) for tokens in result["text_token_ids"]]
    assert len(result["text_token_ids"]) == int(result["sample_n_views"]) == num_views
    torch.testing.assert_close(result["video"], baseline["video"])
    torch.testing.assert_close(result["action"], baseline["action"])
    assert result["sequence_plan"] == baseline["sequence_plan"]
    batch = custom_collate_fn([result])
    assert "camera_caption_prefixes" not in result
    assert batch["ai_caption"] == [result["ai_caption"]]
    assert batch["text_token_lengths"] == [result["text_token_lengths"]]
    # The model assigns local view IDs from these ragged caption boundaries.
    # Exercise the actual packer with that contract and camera-major VAE/action tensors.
    plan = copy.deepcopy(result["sequence_plan"])
    plan.text_view_ids = list(range(len(batch["text_token_lengths"][0])))
    latent_frames = 1 if mode == "policy" else 5
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 4, num_views * latent_frames, 1, 1)],
        num_vision_items_per_sample=[1],
        num_views_per_vision_item=[num_views],
        x0_tokens_action=[result["action"]],
        num_views_per_action_item=[int(result["num_views_per_action_item"])],
        fps_vision=torch.tensor([8.0]),
        fps_action=torch.tensor([8.0]),
    )
    packed = pack_input_sequence(
        [plan],
        [tokens.tolist() for tokens in result["text_token_ids"]],
        clean,
        torch.tensor([0.5]),
        {"eos_token_id": 256, "start_of_generation": 257, "end_of_generation": 258},
        action_dim=64,
    )
    assert packed.text_caption_view_ids == [list(range(num_views))]
    assert len(packed.text_caption_lens[0]) == num_views
    assert packed.num_views_per_vision_item == packed.num_views_per_action_item == [num_views]


@pytest.mark.L0
@pytest.mark.CPU
def test_action_camera_roles_reject_missing_separate_camera_metadata() -> None:
    pipeline = ActionTransformPipeline(
        pad_keys=[],
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_camera_role_to_caption=True,
    )
    with pytest.raises(ValueError, match="require separate camera clips"):
        pipeline({"ai_caption": "move", "video": torch.zeros(3, 17, 4, 6), "mode": "policy"}, "256")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["policy", "wam", "forward_dynamics", "inverse_dynamics"])
@pytest.mark.parametrize("head_mapped", [False, True])
@pytest.mark.parametrize("explicit_camera_poses", [False, True])
def test_per_view_action_subjects_match_projected_targets(
    mode: str, head_mapped: bool, explicit_camera_poses: bool
) -> None:
    # Exercise B1K's native head/left/right order, including an explicitly
    # unavailable optical head pose overriding an otherwise valid legacy head.
    names = (
        "observation.rgb.zed_link_camera_0",
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
    )
    videos = [torch.zeros(3, 17, 4, 6) for _ in names]  # 3 x [C,T,H,W]
    action = torch.zeros(16, UNIFIED_ACTION_DIM)  # [T,D]
    action[:, [3, 7, 12, 16, 37, 41]] = 1.0  # [T,6] identity rotations
    action[:, [33, 58]] = 0.5  # [T,2] gripper openness
    attributes = {
        "domain_postfix": "The video is captured from a synthetic environment.",
        "embodiment_postfix": "The video shows a dual-arm robot.",
        "view_postfix": ViewCompositionMappings["separate_camera_views"],
        "subject_postfix": CaptionSubjectMappings["ego_and_dual_wrist_cameras"],
    }
    data = {
        "ai_caption": "The goal is clean dining table and clear crumbs.",
        "video": torch.cat(videos, dim=1),  # [C,3*T,H,W]
        "camera_videos": videos,
        "camera_view_names": names,
        "camera_caption_prefixes": native_camera_caption_prefixes(names),
        "camera_action_pose_convention": "backward_chunk_anchored_16f",
        "camera_action_gripper_width_m": 0.1,
        "action": action,
        "action_caption_attributes": attributes,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": mode,
        "viewpoint": "separate_camera_views",
    }
    valid_mask = torch.ones(UNIFIED_ACTION_DIM, dtype=torch.bool)  # [D]
    if explicit_camera_poses:
        data["camera_pose_actions"] = {
            names[0]: action[:, :9].numpy() if head_mapped else None,  # [T,9] or None
            names[1]: action[:, 34:43].numpy(),  # [T,9]
            names[2]: action[:, 9:18].numpy(),  # [T,9]
        }
    else:
        valid_mask[:9] = head_mapped  # [9]
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        pad_keys=[],
        max_action_dim=64,
        append_action_caption_semantics=True,
        append_camera_role_to_caption=True,
        enable_mode_specific_prompt=True,
    )

    result = pipeline(data, "256", action_valid_mask=valid_mask)

    head, left, right = result["ai_caption"]
    assert head.startswith("The view is from the camera mounted on the robot head.\n\n")
    assert (
        "The action targets are the head ego motion, mapped to the ego-pose."
        if head_mapped
        else "There is no action target."
    ) in head
    assert "wrist" not in head
    for caption, side, other_side in ((left, "left", "right"), (right, "right", "left")):
        assert caption.startswith(f"The view is from the camera mounted on the {side} wrist.\n\n")
        assert (
            f"The action targets are the {side} wrist-mounted camera motion and gripper openness, "
            f"mapped to the {side} wrist-pose and {side}-openness components."
        ) in caption
        assert other_side not in caption and "ego-pose" not in caption
    for caption in (head, left, right):
        assert ("The goal is" in caption) is (mode != "inverse_dynamics")
        assert "2.0 seconds long" in caption
        assert attributes["subject_postfix"] not in caption
    assert bool(result["action_valid_mask"][:9].all()) is head_mapped
    if not head_mapped:
        assert not bool(result["action_raw"][:16].any())
        assert set(range(16)) <= set(result["sequence_plan"].condition_frame_indexes_action)
    assert "action_caption_attributes" not in result
    assert attributes["subject_postfix"] == CaptionSubjectMappings["ego_and_dual_wrist_cameras"]


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
@pytest.mark.CPU
@pytest.mark.parametrize("video_frames", [1, 17])
@pytest.mark.parametrize("history_steps", [0, 2])
def test_policy_preserves_action_horizon_history_and_prompt_duration(video_frames: int, history_steps: int) -> None:
    pipeline = ActionTransformPipeline(
        pad_keys=[],
        max_action_dim=4,
        append_viewpoint_info=False,
        append_resolution_info=False,
        format_prompt_as_json=True,
    )
    video = torch.arange(3 * video_frames * 4 * 6).reshape(3, video_frames, 4, 6).float()
    action = torch.arange(32).reshape(16, 2).float()
    history = torch.arange(history_steps * 2).reshape(history_steps, 2).float()
    data = {
        "ai_caption": "Move the camera.",
        "video": video,
        "video_num_frames": 17,
        "action": action,
        "history_action": history,
        "conditioning_fps": torch.tensor(8),
        "mode": "policy",
    }

    result = pipeline(data, resolution="256")

    assert result["video"].shape == (3, 1, 4, 6)
    torch.testing.assert_close(result["video"], video[:, :1])
    assert result["ai_caption"]["duration"] == "2s"
    assert result["action"].shape == (history_steps + 16, 4)
    torch.testing.assert_close(result["action_raw"], torch.cat((history, action)))
    assert result["sequence_plan"].condition_frame_indexes_vision == [0]
    assert result["sequence_plan"].condition_frame_indexes_action == list(range(history_steps))
    assert result["sequence_plan"].action_start_frame_offset == 1 - history_steps
    assert "video_num_frames" not in result


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
@pytest.mark.parametrize("caption", ["Open the drawer.", ""])
@pytest.mark.parametrize("format_prompt_as_json", [False, True])
def test_action_transform_pipeline_requires_attributes_when_semantics_enabled(
    caption: str, format_prompt_as_json: bool
) -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_action_caption_semantics=True,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        format_prompt_as_json=format_prompt_as_json,
    )
    data_dict = {
        "__key__": "missing-semantics-sample",
        "ai_caption": caption,
        "video": torch.zeros(3, 17, 256, 256),  # [C,T,H,W]
        "action": torch.zeros(16, 2),  # [T,D]
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
    }

    with pytest.raises(ValueError, match="requires action_caption_attributes.*missing-semantics-sample"):
        pipeline(data_dict, resolution="256")


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
