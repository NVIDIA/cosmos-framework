# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any, List

import attrs

from cosmos_framework.utils import config
from cosmos_framework.configs.base.reasoner.defaults.policy_config import PolicyConfig


@attrs.define(slots=False)
class DataSetting:
    """Configuration for data.

    Attributes:
        qwen_min_video_token_length: Minimum video token length per video.
        qwen_max_video_token_length: Maximum video token length.
        qwen_max_video_frames: Optional deterministic decoded-frame ceiling per video.
        qwen_adaptive_video_token_budget: Shrink video tokens when a row exceeds max_tokens.
        qwen_adaptive_video_token_safety_margin: Tokens reserved after adaptive retokenization.
        qwen_max_fps_thres: Maximum accepted native video FPS.
        qwen_target_fps: Target fps for video sampling.
        keep_raw_media: Emit the debug-only raw_image / raw_video tensors; disable for long-context runs.
        text_chat_order: Order of text items in user messages.
        strip_thinking_prob: Per-sample probability of converting thinking data into non-thinking data.
        offline_pack_text_media_shuffle_ratio: Offline packing only. Probability of swapping a user turn's
            media and text, pinned to sample epoch 0 so measurement, restarts and training see the same order.
            0.01 matches the ordinary loader; 0 reproduces packs built without the swap. Part of the packing
            contract: changing it needs a new packing plan (the Lance cache is reused).
        custom_system_prompt: System prompt injected when a conversation has no leading system message.
        strip_original_system_prompt: Remove existing system messages before optional custom prompt injection.
        distributor_type: "with_replace" (WeightedShardlistBasic) or "no_replace" (NoReplaceShardlistBasic).
        distributor_seed: Seed for the distributor.
        max_batch_size: Hard cap on the number of samples in each dynamic batch.
        max_tokens: Per-sample token limit used by filtering.
        max_tokens_in_batch: Padded-token budget for dynamic batching.
        long_threshold: Single seed-sample length threshold that emits the sample alone.
        data_stats_writer_enabled: Write per-sample parquet rows at every checkpoint save; a checkpoint
            counts as complete only once its rows are flushed and merged.
        lance_cache_manifest: Recipe cache manifest reusing pinned local or remote Lance sources.
        lance_data_root: Absolute local root of a receipt-backed conversation Lance export.
    """

    qwen_min_video_token_length: int = 16
    qwen_max_video_token_length: int = 8192
    qwen_max_video_frames: int | None = None
    qwen_max_image_token_length: int = 8192
    qwen_adaptive_video_token_budget: bool = False
    qwen_adaptive_video_token_safety_margin: int = 128
    qwen_max_fps_thres: float = 60.0
    qwen_target_fps: float = 4.0
    keep_raw_media: bool = True
    use_native_edge_processor: bool = False
    video_timestamp_mode: str = attrs.field(
        default="qwen_index",
        validator=attrs.validators.in_({"qwen_index", "legacy_fps", "source_pts"}),
    )
    qwen_video_temporal_mode: str = attrs.field(
        default="native", validator=attrs.validators.in_({"native", "framewise"})
    )
    qwen_drop_over_max_length: bool = False
    text_chat_order: str = attrs.field(
        default="text_end",
        validator=attrs.validators.in_({"text_end", "text_start", "random"}),
    )
    strip_thinking_prob: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(
            attrs.validators.instance_of((int, float)),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ),
    )
    offline_pack_text_media_shuffle_ratio: float = attrs.field(
        default=0.01,
        validator=attrs.validators.and_(
            attrs.validators.instance_of((int, float)),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ),
    )
    # Preserve processor-specific supervision unless reproducing an older packed run.
    mask_inserted_reasoning_prefix: bool = False
    custom_system_prompt: str | None = "You are a helpful assistant."
    strip_original_system_prompt: bool = False
    temporal_localization_output_format: str = attrs.field(
        default="random",
        validator=attrs.validators.in_({"dense_video_caption", "temporal_localization", "temporal_caption", "random"}),
    )
    temporal_localization_fps: float = 1.0
    # For packed dataset
    max_batch_size: int = 1
    max_tokens: int = 16000
    max_tokens_in_batch: int = 16000
    long_threshold: int = 6400
    # "with_replace" (WeightedShardlistBasic) or "no_replace" (NoReplaceShardlistBasic).
    distributor_type: str = attrs.field(
        default="with_replace",
        validator=attrs.validators.in_({"with_replace", "no_replace"}),
    )
    distributor_seed: int = 1993
    webdataset_detshuffle: bool = False
    num_data_workers: int = 8
    data_prefetch_factor: int | None = 1
    val_split_ratio: float = 0.0
    recipe_name: str | None = None
    data_stats_writer_enabled: bool = False
    lance_cache_manifest: str | None = None
    lance_data_root: str | None = None
    # Safe opt-in switch used by ``lance_recipe_selectable``. The legacy explicit
    # ``lance_offline_pack_recipe`` config remains available for exact resume compatibility.
    offline_pack_enabled: bool = False
    # Complete planned packs per rank/forward; distinct from the unpacked sample cap.
    offline_pack_batch_size: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    # Inert unless an offline-pack data config is selected.
    offline_pack_recipe_manifest: str | None = None
    offline_pack_schedule_path: str | None = None
    offline_pack_validation_schedule_path: str | None = None
    # Used only by the opt-in mixed-pack loader groups; existing defaults are unchanged.
    offline_pack_mixture_schedule_paths: list[str] | None = None
    offline_pack_recipe_source_manifest: str | None = None
    offline_pack_recipe_output_root: str | None = None
    # Opt-in persistent publication root; catalogs must already have been published offline.
    offline_pack_artifact_root: str | None = None
    # Optional integrity pin for the local recipe manifest.
    # Required with an explicit published manifest; pin this alongside the URI on resume.
    offline_pack_recipe_manifest_sha256: str | None = None
    offline_pack_recipe_auto_prepare: bool = False
    # Per-epoch launch requirement; does not change a published plan's content identity.
    offline_pack_minimum_required_packs: int = attrs.field(default=0, validator=attrs.validators.ge(0))
    # The packed training resolver rejects setting both draw options before artifact I/O.
    # Coverage computes total draws from all catalog rows plus weighted extras.
    offline_pack_recipe_total_draws: int | None = None
    offline_pack_recipe_weighted_extra_draws: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    offline_pack_recipe_required_training_epochs: int = attrs.field(
        default=1,
        validator=attrs.validators.ge(1),
    )
    offline_pack_recipe_max_members_per_shard: int = attrs.field(
        default=262_144,
        validator=attrs.validators.ge(1),
    )
    offline_pack_recipe_packing_workers: int = attrs.field(
        default=1,
        validator=attrs.validators.ge(1),
    )
    offline_pack_recipe_global_ordering_mode: str = attrs.field(
        default="epoch_permuted",
        validator=attrs.validators.in_({"epoch_permuted", "legacy_direct"}),
    )
    offline_pack_skip_recoverable_drops: bool = False
    offline_pack_max_consecutive_skipped_packs: int = attrs.field(
        default=32,
        validator=attrs.validators.ge(1),
    )


@attrs.define(slots=False)
class Config(config.Config):
    policy: PolicyConfig = PolicyConfig()
    data_setting: DataSetting = DataSetting()
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "vlm_fsdp"},
            {"sound_und_config": None},
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
