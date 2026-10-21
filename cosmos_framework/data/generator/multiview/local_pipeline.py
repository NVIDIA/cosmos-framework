# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the RGB/WSM/LiDAR training transforms without opening a dataset catalog.

The transforms are shared with the reference training implementation. Storage,
logical task mixtures, and stage checkpoint policy belong to the caller.
"""

from __future__ import annotations

import inspect
from typing import Any

from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.first_caption_paragraph import FirstCaptionParagraph
from cosmos_framework.data.generator.augmentors.image_editing_transform import RemoveKeys
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos_framework.data.generator.augmentors.sequence_plan import (
    MultiviewSequencePlanAugmentor,
    SequencePlanAugmentor,
)
from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    TextTokenizerTransform,
    TextTokenizerTransformForTransfer,
)
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_ATTRIBUTES, CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.caption_format import DEFAULT_CAPTION_PREFIXES
from cosmos_framework.data.generator.multiview.schema import (
    MADS_CAMERAS,
    MADS_V1_CAPTION_KEY_MAPPING,
    MADS_V1_CAPTION_PROBABILITY,
    MADS_V1_CONTROL_KEY_MAPPING,
    MADS_V1_VIDEO_KEY_MAPPING,
)
from cosmos_framework.data.generator.multiview.transforms import (
    AlignLidarClips,
    DescribeLidarTargetClip,
    EnablePerCameraVAEEncoding,
    ExtractFramesAndCaptions,
    ExtractHdmapRange,
    ExtractLidarRange,
    ExtractLidarTransferWindow,
    FilterMissingData,
    JointCamLidarToTrainingFormat,
    LidarTransferToTrainingFormat,
    MultiviewTransferToTrainingFormat,
    NormalizeNestedRangemapKeys,
)
from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor


def make_auto_augmentations(config: dict[str, Any]) -> list[Augmentor]:
    """Construct the local training pipeline; never resolve remote data sources.

    Captured legacy bundles may contain disabled radar/geometry configuration.
    Refuse those modes when enabled rather than silently changing their meaning.
    """
    for field in (
        "load_camera_geometry",
        "radar_tokenizer_training",
        "joint_cam_radar_transfer",
        "joint_cam_lidar_radar_transfer",
    ):
        if config.get(field):
            raise ValueError(f"Local RGB/LiDAR training does not implement {field}")
    if config.get("lidar_tokenizer_training") or config.get("rgb_view_completion_probability", 0):
        raise ValueError("Tokenizer training and view completion are not this curriculum's data path")
    joint = bool(config.get("joint_cam_lidar_transfer", False))
    lidar_only = bool(config.get("lidar_transfer", False))
    if joint and lidar_only:
        raise ValueError("Joint and LiDAR-only tasks are mutually exclusive")
    load_lidar = bool(config.get("load_lidar", False) or joint or lidar_only)
    cameras = tuple(config.get("camera_keys", MADS_CAMERAS))
    if lidar_only:
        cameras = ()
    if not cameras and not lidar_only:
        raise ValueError("Camera and joint samples require at least one camera")
    controls = config.get("camera_control_key_mapping", MADS_V1_CONTROL_KEY_MAPPING)
    if joint and controls is None:
        raise ValueError("Joint training requires paired WSM controls")
    resolution = tuple(config.get("resolution_hw", (480, 832)))
    frames = int(config.get("num_video_frames", 65))
    captions = dict(config.get("camera_caption_key_mapping") or MADS_V1_CAPTION_KEY_MAPPING)
    probabilities = dict(config.get("caption_probability") or MADS_V1_CAPTION_PROBABILITY)
    conditioning = config.get("conditioning_config", {0: 0.5, 1: 0.5})
    per_camera = bool(config.get("enable_per_camera_vae_encoding", True))
    separate_text = bool(config.get("separate_view_text_tokenization", True))
    result: list[Augmentor] = []
    range_keys = ["lidar_rangemap", "hdmap_rangemap"] if joint or lidar_only else ["lidar_rangemap"]
    if load_lidar:
        result.extend(
            (
                NormalizeNestedRangemapKeys(),
                FilterMissingData(range_keys, skip_missing=config.get("skip_missing_lidar", True)),
            )
        )
    if lidar_only:
        result.append(
            ExtractLidarTransferWindow(
                num_frames=frames,
                fps_downsample_factor=config.get("fps_downsample_factor", 1),
                caption_probability=probabilities,
                camera_caption_key_mapping=captions,
                caption_camera_name=config.get("single_caption_camera_name") or "camera_front_wide_120fov",
                passthrough_keys=range_keys + ["_rangemap_compat_clip_key"],
                native_caption_window=config.get("lidar_native_caption_window", True),
            )
        )
    else:
        # Use the original extractor, with exactly one common temporal/spatial
        # draw for every RGB/WSM view. No inference input schema is substituted.
        video_mapping = dict(config.get("camera_video_key_mapping") or MADS_V1_VIDEO_KEY_MAPPING)
        # A bundle may carry front-wide caption metadata only for LiDAR-only
        # stages, without declaring that camera as an available RGB view.
        rgb_captions = {key: member for key, member in captions.items() if key in video_mapping}
        extractor = {
            key: value for key, value in config.items() if key in inspect.signature(ExtractFramesAndCaptions).parameters
        }
        extractor.update(
            camera_order=cameras,
            num_frames=frames,
            resolution_hw=resolution,
            fps_downsample_factor=config.get("fps_downsample_factor", 1),
            caption_probability=probabilities,
            camera_view_mapping=dict(CAMERA_VIEW_MAPPING),
            camera_video_key_mapping=video_mapping,
            camera_caption_key_mapping=rgb_captions,
            camera_control_key_mapping=controls,
            camera_attribute_mapping=config.get("camera_attribute_mapping") or CAMERA_ATTRIBUTES,
            camera_prefix_mapping=config.get("camera_prefix_mapping", dict(DEFAULT_CAPTION_PREFIXES)),
            passthrough_keys=range_keys + ["_rangemap_compat_clip_key"] if load_lidar else None,
            defer_camera_alignment=joint,
            minimum_num_frames=config.get("joint_aligned_num_video_frames") if joint else None,
        )
        result.append(ExtractFramesAndCaptions(**extractor))
    if load_lidar:
        result.append(
            ExtractLidarRange(
                tokenizer_v1=config.get("lidar_tokenizer_v1", True),
                range_projection=config.get("lidar_range_projection"),
            )
        )
    if lidar_only or joint:
        result.extend(
            (
                ExtractHdmapRange(
                    tokenizer_v1=config.get("lidar_tokenizer_v1", True),
                    range_projection=config.get("lidar_range_projection"),
                ),
                AlignLidarClips(),
            )
        )
    if lidar_only:
        result.append(DescribeLidarTargetClip())
        if config.get("first_caption_paragraph_only", False):
            result.append(FirstCaptionParagraph())
    duration = DurationFPSTextTimeStamps(
        input_keys=["ai_caption", "video", "fps"],
        output_keys=["ai_caption", "conditioning_duration", "duration_fps_template"],
        args={
            "caption_key": "ai_caption",
            "video_key": "video",
            "fps_key": "fps",
            "num_frames_key": "num_video_frames_per_view",
            "enabled": config.get("append_duration_fps_timestamps", True),
        },
    )
    spatial = ResolutionTextInfo(
        input_keys=["ai_caption", "video", "image_size"],
        output_keys=["ai_caption"],
        args={"caption_key": "ai_caption", "image_key": "video", "enabled": config.get("append_resolution_info", True)},
    )
    if joint:
        result.append(
            JointCamLidarToTrainingFormat(
                camera_resolution_hw=resolution,
                lidar_fps=config.get("lidar_fps", 10.0),
                max_camera_frames=frames,
                fixed_camera_frames=config.get("joint_cam_lidar_fixed_camera_frames"),
                exact_camera_frames=config.get("joint_aligned_num_video_frames"),
                conditioning_config=conditioning,
                lidar_conditioning_probability=config.get("lidar_conditioning_probability", 0.0),
                enable_per_camera_vae_encoding=per_camera,
                lidar_dropout_rate=config.get("lidar_dropout_rate", 0.0),
                lidar_prefix_dropout_rate=config.get("lidar_prefix_dropout_rate", 0.0),
                control_dropout_rate=config.get("control_dropout_rate", 0.0),
                caption_augmentors=(duration, spatial),
            )
        )
    else:
        result.extend((duration, spatial))
        if lidar_only:
            result.append(LidarTransferToTrainingFormat(conditioning_config=conditioning))
        elif controls is not None:
            result.append(
                MultiviewTransferToTrainingFormat(
                    resolution_hw=resolution,
                    conditioning_config=conditioning,
                    share_vision_temporal_positions=config.get("share_vision_temporal_positions", True),
                    enable_per_camera_vae_encoding=per_camera,
                )
            )
        elif conditioning is not None:
            cls = MultiviewSequencePlanAugmentor if per_camera else SequencePlanAugmentor
            result.append(
                cls(input_keys=["video"], args={"conditioning_config": conditioning, "temporal_compression_factor": 4})
            )
            if per_camera:
                result.append(EnablePerCameraVAEEncoding())
    transfer_text = joint or lidar_only or controls is not None
    cls = TextTokenizerTransformForTransfer if transfer_text else TextTokenizerTransform
    if config.get("tokenizer_config") is not None:
        result.append(
            cls(
                input_keys=["ai_caption"],
                output_keys=["text_token_ids", "text_token_lengths"] if separate_text else ["text_token_ids"],
                args={
                    "tokenizer_config": config["tokenizer_config"],
                    "cfg_dropout_rate": config.get("cfg_dropout_rate", 0.1),
                    "use_system_prompt": config.get("use_system_prompt", True),
                    "tokenize_separately": separate_text,
                    "emit_system_prompt": transfer_text,
                    **(
                        {"task": "av_joint_camera_lidar_transfer"}
                        if joint
                        else {"task": "av_multiview_transfer"}
                        if transfer_text and len(cameras) > 1
                        else {}
                    ),
                },
            )
        )
    remove = ["padding_mask", "original_hw", "camera_keys_selection", "chunk_index", "frame_indices"]
    if not joint and not lidar_only and controls is not None:
        remove.append("control_input")
    result.append(RemoveKeys(input_keys=remove))
    return result
