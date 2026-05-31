# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any, List

import attrs

from cosmos_framework.utils import config
from cosmos_framework.configs.base.vlm.defaults.policy_config import PolicyConfig


@attrs.define(slots=False)
class DataSetting:
    """Configuration for data.

    Attributes:
        qwen_max_video_token_length: Maximum video token length.
        qwen_target_fps: Target fps for video sampling.
        text_chat_order: Order of text items in user messages.
        distributor_type: "with_replace" (WeightedShardlistBasic) or "no_replace" (NoReplaceShardlistBasic).
        distributor_seed: Seed for the distributor.
    """

    qwen_max_video_token_length: int = 8192
    qwen_max_image_token_length: int = 8192
    qwen_target_fps: float = 4.0
    text_chat_order: str = attrs.field(
        default="text_end",
        validator=attrs.validators.in_({"text_end", "text_start", "random"}),
    )
    temporal_localization_output_format: str = attrs.field(
        default="random",
        validator=attrs.validators.in_({"dense_video_caption", "temporal_localization", "temporal_caption", "random"}),
    )
    temporal_localization_fps: float = 1.0
    # For packed dataset
    max_batch_size: int = 1
    max_tokens: int = 16000
    # "with_replace" (WeightedShardlistBasic) or "no_replace" (NoReplaceShardlistBasic).
    distributor_type: str = attrs.field(
        default="with_replace",
        validator=attrs.validators.in_({"with_replace", "no_replace"}),
    )
    distributor_seed: int = 1993
    webdataset_detshuffle: bool = False
    num_data_workers: int = 8
    data_prefetch_factor: int = 1
    val_split_ratio: float = 0.0


@attrs.define(slots=False)
class Config(config.Config):
    policy: PolicyConfig = PolicyConfig()
    data_setting: DataSetting = DataSetting()
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "vlm_fsdp"},
            {"vlm_policy": None},
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdacosine"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dcp"},
            {"callbacks": ["basic_vlm"]},
            {"experiment": None},
        ]
    )
