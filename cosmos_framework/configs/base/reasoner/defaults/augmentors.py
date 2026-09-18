# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Storage-independent Reasoner augmentation and processor configuration."""

from typing import Any

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.data.generator.augmentors.reasoner.bytes_to_media import BytesToMedia
from cosmos_framework.data.generator.augmentors.reasoner.filter_output_key import FilterOutputKey
from cosmos_framework.data.generator.augmentors.reasoner.filter_seq_length import FilterSeqLength
from cosmos_framework.data.generator.augmentors.reasoner.floating_number_format import FloatingNumberFormat
from cosmos_framework.data.generator.augmentors.reasoner.format_describe_anything import FormatDescribeAnything
from cosmos_framework.data.generator.augmentors.reasoner.format_hot_fixes import FormatHotFixes
from cosmos_framework.data.generator.augmentors.reasoner.prompt_format import PromptFormat
from cosmos_framework.data.generator.augmentors.reasoner.shuffle_text_media_order import ShuffleTextMediaOrder
from cosmos_framework.data.generator.augmentors.reasoner.timestamp import TimeStamp
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_with_subject_tracking import (
    TimeStampWithSubjectTracking,
)
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_without_augment_message import (
    TimeStampWithoutAugmentMessage,
)
from cosmos_framework.data.generator.augmentors.reasoner.timestamp_without_end_time import TimeStampWithoutEndTime
from cosmos_framework.data.generator.augmentors.reasoner.tokenize_data import TokenizeData
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.generator.deterministic_rng import DeterministicAugmentor


def create_data_augmentor_config() -> dict[str, Any]:
    config = {
        "bytes_to_media": L(BytesToMedia)(
            input_key="media",
            output_key="media",
            min_fps_thres=2,
            max_fps_thres=60,
            target_fps="${data_setting.qwen_target_fps}",  # type: ignore
            video_temporal_mode="${data_setting.qwen_video_temporal_mode}",
            max_video_token_length="${data_setting.qwen_max_video_token_length}",  # type: ignore
            processor=processor,
            extract_audio="${model.config.sound_und}",
            is_input_pickle_byptes=False,  # If True, it means the input "media" is pickled bytes that needs to be unpickled first; if False, it means the input "media" is raw bytes that can be directly decoded to image/video. Set to False for most cases, and only set to True for some special datasets where media is stored as pickled bytes.
        ),  # takes "videos" and output "videos"
        "prompt_format": L(PromptFormat)(  # takes text_keys and output "conversation"
            input_keys=["texts"],
            text_chat_order="${data_setting.text_chat_order}",
        ),
        "shuffle_text_media_order": L(ShuffleTextMediaOrder)(),
        "format_hot_fixes": L(FormatHotFixes)(),
        # ============================
        # TL data augmentation
        # ============================
        "timestamp": L(TimeStamp)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="temporal_localization",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "av_reasoning_localization_20250627",
                "tl_activitynet_20250630",
                "tl_agibot_fisheye_20250630",
                "tl_2dvlm_20250627",
                "tl_2dvlm_20251121",
                "tl_youcook2_20250716",
                "tl_yt_cctv_warehouse_20250724",
            ],
            processor=processor,
        ),
        "TL_recaption": L(TimeStamp)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="caption",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_2dvlm_recaption_20251121",
                "tl_2dvlm_recaption_20250627",
            ],
            processor=processor,
        ),
        # Special augmentors:
        # timestamp_without_end_time: nexar data does not contain end time
        # timestamp_with_subject_trackig: plm data has subject id + mask, and it's video data
        # format_describe_anything: dam data has subject id + mask + category label, and it's image data (does not need timestampt)
        # timestamp_without_augment_message: rft tl data require timestamp augmentation to video, but keep original text
        "timestamp_without_end_time": L(TimeStampWithoutEndTime)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="temporal_localization",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_nexar_20250708",
                "mimicgen_temporal_localization",
            ],
            processor=processor,
        ),
        "timestamp_with_subject_trackig": L(TimeStampWithSubjectTracking)(
            input_key="media",
            output_format="temporal_location_subject",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_plm_sav_20250714",
            ],
            processor=processor,
        ),
        "floating_number_format": L(FloatingNumberFormat)(
            input_key="conversation",
            decimal_places=2,
            urls_needs_format=[
                "3d_grounding_av",
            ],
        ),
        "format_describe_anything": L(FormatDescribeAnything)(
            input_key="media",
            urls_needs_timestamp=[
                "describe-anything-dataset",
            ],
        ),
        "timestamp_without_augment_message": L(TimeStampWithoutAugmentMessage)(
            input_key="media",
            output_format="${data_setting.temporal_localization_output_format}",
            urls_needs_timestamp=[
                "rl_distill_tl_0729",
            ],
            processor=processor,
        ),
        # ============================
        # End of TL data augmentation
        # ============================
        "tokenize_data": L(TokenizeData)(
            processor=processor,
            max_video_token_length="${data_setting.qwen_max_video_token_length}",
            max_image_token_length="${data_setting.qwen_max_image_token_length}",
            custom_system_prompt="${data_setting.custom_system_prompt}",
            strip_original_system_prompt="${data_setting.strip_original_system_prompt}",
            video_temporal_mode="${data_setting.qwen_video_temporal_mode}",
            text_only=False,
            sound_und="${model.config.sound_und}",
            audio_encoder_type="${model.config.sound_und_config.audio_encoder_type}",
            audio_start_token="${model.config.sound_und_config.audio_start_token}",
            audio_pad_token="${model.config.sound_und_config.audio_pad_token}",
            audio_end_token="${model.config.sound_und_config.audio_end_token}",
            audio_timestamp_fps="${model.config.sound_und_config.audio_timestamp_fps}",
            audio_layout="${model.config.sound_und_config.audio_layout}",
        ),
        "filter_output_keys": L(FilterOutputKey)(
            text_only=False,
        ),
        "filter_seq_length": L(FilterSeqLength)(
            max_token_length="${data_setting.max_tokens}",
            drop_over_max_length="${data_setting.qwen_drop_over_max_length}",
            processor=processor,
            sound_und="${model.config.sound_und}",
            audio_start_token="${model.config.sound_und_config.audio_start_token}",
            audio_pad_token="${model.config.sound_und_config.audio_pad_token}",
            audio_end_token="${model.config.sound_und_config.audio_end_token}",
        ),
    }
    for name in (
        "prompt_format",
        "shuffle_text_media_order",
        "timestamp",
        "TL_recaption",
        "timestamp_without_end_time",
        "timestamp_with_subject_trackig",
        "format_describe_anything",
        "timestamp_without_augment_message",
    ):
        config[name] = L(DeterministicAugmentor)(augmentor=config[name])
    return config


processor = L(build_processor_lazy)(
    tokenizer_type="${model.config.policy.backbone.model_name}",
    credentials="${checkpoint.load_from_object_store.credentials}",
    bucket="${checkpoint.load_from_object_store.bucket}",
)
