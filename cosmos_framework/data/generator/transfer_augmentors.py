# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Augmentor builders for single-view Transfer (control + RGB) video training."""

from __future__ import annotations

import cosmos_framework.data.imaginaire.webdataset.augmentors.image.normalize as normalize
import cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize as resize
import cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps as duration_fps_text_timestamps
import cosmos_framework.data.generator.augmentors.interleaved_video_parsing as video_parsing_with_controlled
import cosmos_framework.data.generator.augmentors.merge_datadict as merge_datadict
import cosmos_framework.data.generator.augmentors.resolution_text_info as resolution_text_info
import cosmos_framework.data.generator.augmentors.text_transforms_for_video as text_transforms_for_video
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.data.generator.augmentors.image_editing_transform import RemoveKeys
from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransformForTransfer
from cosmos_framework.data.generator.augmentors.transfer_control_transform import (
    AddControlFromVideoComb,
    AddSelectedControlFromVideo,
    SampleResolution,
    TransferToTrainingFormat,
    VideoTransferSampleFrame,
)
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO

_TRANSFER_MAX_ASPECT_RATIO_DISTORTION: dict[str, float] = {
    resolution: 0.07 if resolution == "256" else 0.03 for resolution in VIDEO_RES_SIZE_INFO
}


def get_video_transfer_full_frames_augmentor(
    resolution: str = "480",
    resolutions: list[str] | None = None,
    resolutions_weights: list[float] | None = None,
    min_fps: float = 10.0,
    max_fps: float = 60.0,
    max_num_frames: int = 1000,
    control_input_type: str = "edge_blur_depth_seg",
    output_image: bool = False,
    mean: float = 0.5,
    std: float = 0.5,
    caption_config: dict | list | None = None,
    tokenizer_config: LazyDict | None = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    max_stride: int = 3,
    min_stride: int = 1,
    use_system_prompt: bool = True,
    resize_on_read: bool = False,
    conditioning_config: dict[int, float] | None = None,
    share_vision_temporal_positions: bool = True,
    dataset_resolution_type: str = "all",
    selected_control: bool = False,
    temporal_compression_factor: int = 4,
    max_aspect_ratio_distortion: float | None = None,
    **kwargs,
) -> dict:
    """Build variable-length full-video transfer augmentation.

    RGB, depth, and segmentation streams share the same sampled frame indices.
    Edge and blur controls are still computed online from the resized RGB video.
    ``caption_config`` selects which caption-source dict to read from the
    sample/metadata; the text transform then hardcodes reading the
    ``caption_structured`` field of that dict.
    With ``resize_on_read``, sample resolution before parsing and resize bounded
    decoded batches with the same modality-specific interpolation as the spatial stages.
    ``temporal_compression_factor`` determines the maximum conditioning latent-frame
    count while retaining at least one target latent frame for supervision.
    ``max_aspect_ratio_distortion`` overrides the default limit for every sampled
    resolution when provided.
    """

    if caption_config is None:
        raise ValueError("get_video_transfer_full_frames_augmentor requires a caption_config.")
    if selected_control and output_image:
        raise ValueError("selected_control currently supports video output only.")

    aspect_ratio_limits = _TRANSFER_MAX_ASPECT_RATIO_DISTORTION
    if max_aspect_ratio_distortion is not None:
        aspect_ratio_limits = dict.fromkeys(aspect_ratio_limits, max_aspect_ratio_distortion)

    output_media_key = "images" if output_image else "video"
    has_depth = "depth" in control_input_type
    has_seg = "seg" in control_input_type
    parse_input_keys = ["metas", "video"]
    merge_input_keys: list[str] = ["video"]
    merge_output_keys: list[str] = [
        "video",
        "fps",
        "num_frames",
        "frame_indices",
        "frame_start",
        "frame_end",
        "n_orig_video_frames",
        "num_multiplier",
        "conditioning_fps",
    ]
    resize_input_keys = ["video", "control_input"]
    sample_frame_input_keys = ["video"]
    remove_keys_input_keys = ["control_input", "frame_indices", "metas"]

    if has_depth and not selected_control:
        parse_input_keys.append("depth_pervideo_video_depth_anything")
        merge_input_keys.append("depth_pervideo_video_depth_anything")
        merge_output_keys.append("depth")
        sample_frame_input_keys.append("depth")
        remove_keys_input_keys.append("depth")
    if has_seg and not selected_control:
        parse_input_keys.append("segmentation_sam2_color_video_v2")
        merge_input_keys.append("segmentation_sam2_color_video_v2")
        merge_output_keys.append("segmentation")
        sample_frame_input_keys.append("segmentation")
        remove_keys_input_keys.append("segmentation")

    pipeline = {
        "text_transform": L(
            text_transforms_for_video.TextTransformForVideoTransferChunkedFrames
            if selected_control
            else text_transforms_for_video.TextTransformForVideoTransferFullFrames
        )(
            input_keys=["metas"],
            args={
                "caption_config": caption_config,
                "keep_metas": True,
            },
        ),
        "video_parser_with_controlled": L(
            video_parsing_with_controlled.VideoTransferAlignedSelectedControlParsing
            if selected_control
            else video_parsing_with_controlled.VideoTransferAlignedFullFramesParsing
        )(
            input_keys=parse_input_keys,
            args={
                "video_decode_num_threads": 4,
                "max_num_frames": max_num_frames,
                "max_stride": max_stride,
                "min_stride": min_stride,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "seek_mode": "exact",
                "dataset_resolution_type": dataset_resolution_type,
            },
        ),
        "merge_datadict": L(merge_datadict.DataDictMerger)(
            input_keys=merge_input_keys,
            output_keys=merge_output_keys,
        ),
    }

    if output_image:
        pipeline["video_transfer_sample_frame"] = L(VideoTransferSampleFrame)(
            input_keys=sample_frame_input_keys,
            random_frame=True,
        )

    if resolutions and len(resolutions) > 0:
        pipeline["sample_resolution"] = L(SampleResolution)(
            input_keys=["video"],
            resolutions=resolutions,
            resolutions_weights=resolutions_weights,
            max_aspect_ratio_distortion=aspect_ratio_limits,
        )
        remove_keys_input_keys.extend(["_res_size_map", "_res_max_aspect_ratio_distortion"])
    # Keep this stage with resize_on_read to set image_size even when pixels already have the target dimensions.
    pipeline["input_video_resize_to_target_size"] = L(resize.ResizeToSize)(
        input_keys=["video"],
        args={
            "size": VIDEO_RES_SIZE_INFO[resolution],
            "max_aspect_ratio_distortion": aspect_ratio_limits[resolution],
            "log_aspect_ratio_stats": not resize_on_read,
        },
    )
    if resize_on_read:
        pipeline["video_parser_with_controlled"]["args"]["direct_resize"] = pipeline[
            "input_video_resize_to_target_size"
        ]["args"]
        if "sample_resolution" in pipeline:
            pipeline = {"sample_resolution": pipeline.pop("sample_resolution"), **pipeline}
    if selected_control:
        pipeline["add_control_from_video"] = L(AddSelectedControlFromVideo)(input_keys=["video"])
    else:
        pipeline["add_control_from_video"] = L(AddControlFromVideoComb)(
            input_keys=["video"],
            control_input_type=control_input_type,
            num_control_inputs_prob=(1.0, 0.0, 0.0, 0.0),
            num_control_inputs=1,
        )
    pipeline["normalize"] = L(normalize.Normalize)(
        input_keys=resize_input_keys,
        args={"mean": mean, "std": std},
    )
    if not output_image:
        pipeline["duration_fps_timestamps"] = L(duration_fps_text_timestamps.DurationFPSTextTimeStamps)(
            input_keys=["ai_caption", "video", "conditioning_fps"],
            output_keys=["ai_caption", "conditioning_duration", "duration_fps_template"],
            args={
                "caption_key": "ai_caption",
                "video_key": "video",
                "fps_key": "conditioning_fps",
                "enabled": append_duration_fps_timestamps,
            },
        )

    pipeline["resolution_info"] = L(resolution_text_info.ResolutionTextInfo)(
        input_keys=["ai_caption", output_media_key, "image_size"],
        output_keys=["ai_caption"],
        args={
            "caption_key": "ai_caption",
            "image_key": output_media_key,
            "enabled": append_resolution_info,
        },
    )
    pipeline["transfer_to_training_format"] = L(TransferToTrainingFormat)(
        mean=mean,
        std=std,
        output_media_key=output_media_key,
        conditioning_config=conditioning_config,
        share_vision_temporal_positions=share_vision_temporal_positions,
        temporal_compression_factor=temporal_compression_factor,
    )
    pipeline["text_tokenization"] = L(TextTokenizerTransformForTransfer)(
        input_keys=["ai_caption"],
        output_keys=["text_token_ids"],
        args={
            "tokenizer_config": tokenizer_config,
            "cfg_dropout_rate": cfg_dropout_rate,
            "use_system_prompt": use_system_prompt,
            "emit_system_prompt": True,
        },
    )
    if output_image:
        remove_keys_input_keys.append("video")
    pipeline["remove_keys"] = L(RemoveKeys)(input_keys=remove_keys_input_keys)
    return pipeline


def get_video_transfer_chunked_frames_augmentor(
    resolution: str = "480",
    resolutions: list[str] | None = None,
    resolutions_weights: list[float] | None = None,
    min_fps: float = 10.0,
    max_fps: float = 60.0,
    max_num_frames: int = 1000,
    min_num_frames: int = 5,
    target_num_frames: int | None = None,
    teacher_forcing_frames_per_chunk: int = 1,
    control_input_type: str = "edge_blur_depth_seg",
    output_image: bool = False,
    mean: float = 0.5,
    std: float = 0.5,
    caption_config: dict | list | None = None,
    tokenizer_config: LazyDict | None = None,
    cfg_dropout_rate: float = 0.0,
    append_duration_fps_timestamps: bool = False,
    append_resolution_info: bool = False,
    max_stride: int = 3,
    min_stride: int = 1,
    use_system_prompt: bool = True,
    resize_on_read: bool = False,
    conditioning_config: dict[int, float] | None = None,
    share_vision_temporal_positions: bool = True,
    dataset_resolution_type: str = "all",
    selected_control: bool = False,
    **kwargs,
) -> dict:
    """Build chunk-aware variable-length transfer augmentation.

    The text transform samples a structured-caption chunk first, then the
    parser decodes RGB and any precomputed control streams from that same
    absolute frame range. An exact target resamples the complete caption window
    to that frame count, preserving its semantic content before tokenization.
    """
    pipeline = get_video_transfer_full_frames_augmentor(
        resolution=resolution,
        resolutions=resolutions,
        resolutions_weights=resolutions_weights,
        min_fps=min_fps,
        max_fps=max_fps,
        max_num_frames=max_num_frames,
        control_input_type=control_input_type,
        output_image=output_image,
        mean=mean,
        std=std,
        caption_config=caption_config,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        max_stride=max_stride,
        min_stride=min_stride,
        use_system_prompt=use_system_prompt,
        resize_on_read=resize_on_read,
        conditioning_config=conditioning_config,
        share_vision_temporal_positions=share_vision_temporal_positions,
        dataset_resolution_type=dataset_resolution_type,
        selected_control=selected_control,
        **kwargs,
    )

    has_depth = "depth" in control_input_type
    has_seg = "seg" in control_input_type
    parse_input_keys = ["metas", "video"]
    if has_depth and not selected_control:
        parse_input_keys.append("depth_pervideo_video_depth_anything")
    if has_seg and not selected_control:
        parse_input_keys.append("segmentation_sam2_color_video_v2")

    pipeline["text_transform"] = L(text_transforms_for_video.TextTransformForVideoTransferChunkedFrames)(
        input_keys=["metas"],
        args={
            "caption_config": caption_config,
            "keep_metas": True,
            "min_num_frames": min_num_frames,
            "target_num_frames": target_num_frames,
            "min_stride": min_stride,
            "max_stride": max_stride,
            "min_fps": min_fps,
            "max_fps": max_fps,
        },
    )
    parser_args = {
        "video_decode_num_threads": 4,
        "max_num_frames": max_num_frames,
        "min_num_frames": min_num_frames,
        "target_num_frames": target_num_frames,
        "teacher_forcing_frames_per_chunk": teacher_forcing_frames_per_chunk,
        "max_stride": max_stride,
        "min_stride": min_stride,
        "min_fps": min_fps,
        "max_fps": max_fps,
        "seek_mode": "exact",
        "dataset_resolution_type": dataset_resolution_type,
    }
    if resize_on_read:
        parser_args["direct_resize"] = pipeline["input_video_resize_to_target_size"]["args"]
    parser = (
        video_parsing_with_controlled.VideoTransferAlignedSelectedControlParsing
        if selected_control
        else video_parsing_with_controlled.VideoTransferAlignedChunkedFramesParsing
    )
    pipeline["video_parser_with_controlled"] = L(parser)(
        input_keys=parse_input_keys,
        args=parser_args,
    )
    if target_num_frames is not None and not output_image:
        # Small scheduled horizons must not acquire a zero-second caption from
        # the legacy whole-second duration conversion.
        pipeline["duration_fps_timestamps"]["args"]["fractional_duration"] = True
        if target_num_frames == 1:
            # The usual tenth-second precision rounds a 24/30/60 FPS still to
            # zero seconds. Retain its positive duration in this smallest bucket.
            pipeline["duration_fps_timestamps"]["args"]["template"] = (
                "The video is {duration:.3f} seconds long and is of {fps:.0f} FPS."
            )
    return pipeline
