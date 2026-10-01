# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataset-neutral Nano and Edge video SFT experiment registrations."""

import os
from copy import deepcopy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import (
    VideoSFTProcessor,
    VideoVLMCollator,
    VLMCollator,
)
from cosmos_framework.data.generator.dataflow import ContiguousBatcher, CosmosDataLoader, MapDistributor
from cosmos_framework.data.generator.dataflow.distributors import MediaGroupedMapDistributor
from cosmos_framework.data.generator.local_datasets.reasoning_qa import ReasoningQADataset, VideoConversationDataset
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX


def _video_conversation_dataloader(
    *,
    annotation_env: str,
    media_env: str,
    limit_env: str,
    shuffle: bool,
    frame_env: str = "COSMOS_VIDEO_NUM_FRAMES",
    cache_env: str = "COSMOS_VIDEO_CACHE_SIZE",
    max_pixels_env: str = "COSMOS_VIDEO_MAX_PIXELS",
    system_prompt_env: str = "COSMOS_VIDEO_SYSTEM_PROMPT",
) -> LazyDict:
    validation_grouped = (
        not shuffle and os.environ.get("COSMOS_FRAMEWORK_VALIDATION_SHARD_STRATEGY", "stride") == "media_grouped"
    )
    distributor_cls = MediaGroupedMapDistributor if validation_grouped else MapDistributor
    max_batch_size = int(os.environ.get("COSMOS_FRAMEWORK_VALIDATION_BATCH_SIZE", "1")) if not shuffle else 1
    return L(CosmosDataLoader)(
        distributor=L(distributor_cls)(
            dataset=L(VideoConversationDataset)(
                annotation_path=f"${{oc.env:{annotation_env}}}",
                media_path=f"${{oc.env:{media_env}}}",
                limit=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed="${oc.env:COSMOS_DATALOADER_SEED,42}",
            name="train" if shuffle else "val",
        ),
        processor=L(VideoSFTProcessor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
            num_video_frames=f"${{oc.env:{frame_env},8}}",
            video_cache_size=f"${{oc.env:{cache_env},8}}",
            video_device="${oc.env:COSMOS_VIDEO_DECODER_DEVICE,cuda}",
            video_num_threads="${oc.env:COSMOS_VIDEO_DECODER_THREADS,1}",
            # Training was pinned to 0, so every epoch re-decoded every video. That is
            # the dominant cost of a training step here: measured on a GB300, the wall
            # step splits 2.00s waiting on the dataloader against 0.62s of compute, so
            # decode is roughly three quarters of the run. A cache large enough to hold
            # the split turns epochs after the first into cache hits.
            #
            # Left at 0 by default because the cache is per dataloader worker and holds
            # decoded frames, so capacity has to be chosen against the dataset size and
            # available host memory rather than assumed.
            processed_video_cache_size=(
                "${oc.env:COSMOS_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_SIZE,0}"
                if not shuffle
                else "${oc.env:COSMOS_FRAMEWORK_TRAIN_PROCESSED_VIDEO_CACHE_SIZE,0}"
            ),
            video_max_pixels=f"${{oc.env:{max_pixels_env},81920}}",
            video_override_map="${oc.env:COSMOS_VIDEO_OVERRIDE_MAP,''}",
            system_prompt=f"${{oc.env:{system_prompt_env},''}}",
        ),
        batcher=L(ContiguousBatcher)(
            max_batch_size=max_batch_size,
            max_tokens=81920,
            drop_last=False,
        ),
        collator=L(VideoVLMCollator)(),
        num_workers="${oc.env:COSMOS_FRAMEWORK_DATALOADER_NUM_WORKERS,1}",
        prefetch_factor="${oc.env:COSMOS_FRAMEWORK_DATALOADER_PREFETCH_FACTOR,4}",
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context="spawn",
        processing_threads="${oc.env:COSMOS_FRAMEWORK_SFT_PROCESS_THREADS,8}",
    )


def _task_aware_video_dataloader(
    *,
    split: str,
    shuffle: bool,
    annotation_env: str | None = None,
    media_env: str | None = None,
    limit_env: str | None = None,
    frame_env: str = "COSMOS_VIDEO_NUM_FRAMES",
    cache_env: str = "COSMOS_VIDEO_CACHE_SIZE",
    max_pixels_env: str = "COSMOS_VIDEO_MAX_PIXELS",
    system_prompt_env: str = "COSMOS_VIDEO_SYSTEM_PROMPT",
) -> LazyDict:
    annotation_env = annotation_env or f"COSMOS_VIDEO_{split.upper()}_ANNOTATIONS"
    media_env = media_env or f"COSMOS_VIDEO_{split.upper()}_MEDIA_ROOTS"
    limit_env = limit_env or f"COSMOS_VIDEO_{split.upper()}_LIMIT"
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(ReasoningQADataset)(
                annotation_paths=f"${{oc.env:{annotation_env}}}",
                media_root=f"${{oc.env:{media_env}}}",
                response_mode="hybrid" if split == "train" else "answer",
                system_prompt=f"${{oc.env:{system_prompt_env},''}}",
                vision_kwargs={},
                max_samples=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed="${oc.env:COSMOS_DATALOADER_SEED,42}",
            name=split,
        ),
        processor=L(VideoSFTProcessor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
            num_video_frames=f"${{oc.env:{frame_env},8}}",
            video_cache_size=f"${{oc.env:{cache_env},8}}",
            video_device="${oc.env:COSMOS_VIDEO_DECODER_DEVICE,cuda}",
            video_num_threads="${oc.env:COSMOS_VIDEO_DECODER_THREADS,1}",
            video_max_pixels=f"${{oc.env:{max_pixels_env},81920}}",
            video_override_map="${oc.env:COSMOS_VIDEO_OVERRIDE_MAP,''}",
            system_prompt="",
            use_reasoning_chat_template=True,
        ),
        batcher=L(ContiguousBatcher)(
            max_batch_size=1,
            max_tokens=81920,
            drop_last=False,
        ),
        collator=L(VLMCollator)(),
        num_workers="${oc.env:COSMOS_FRAMEWORK_DATALOADER_NUM_WORKERS,1}",
        prefetch_factor="${oc.env:COSMOS_FRAMEWORK_DATALOADER_PREFETCH_FACTOR,2}",
        persistent_workers=True,
        pin_memory=False,
        multiprocessing_context="spawn",
        processing_threads="${oc.env:COSMOS_FRAMEWORK_SFT_PROCESS_THREADS,8}",
    )


cosmos_video_conversation = LazyDict(
    dict(
        defaults=[
            {"override /checkpoint": "local"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "vlm_fsdp"},
            {"override /vlm_policy": "qwen3_vl_8b_instruct"},
            {"override /callbacks": ["basic_vlm", "basic_log"]},
            "_self_",
        ],
        job=dict(
            project="cosmos3_reasoner",
            group="cosmos_video_conversation_sft",
            wandb_mode="disabled",
        ),
        trainer=dict(
            callbacks=dict(
                dataloader_state=L(CosmosDataLoaderStateCallback)(),
                workflow_status=dict(
                    enabled=True,
                    logging_interval=1,
                    validation_heartbeat_interval=1,
                ),
            ),
            max_iter=10,
            logging_iter=1,
            run_validation=True,
            validation_iter=10,
            max_val_iter=10,
            run_validation_on_start=False,
            grad_accum_iter=1,
        ),
        optimizer=dict(
            lr=1.0e-4,
            fused=True,
            weight_decay=0.01,
            betas=[0.9, 0.999],
            lr_multipliers={"model.visual": 1.0},
        ),
        model=dict(
            config=dict(
                policy=dict(
                    model_max_length=81920,
                    qwen_max_video_token_length=8192,
                ),
                freeze=dict(trainable_params=[".*"]),
                parallelism=dict(
                    data_parallel_shard_degree=4,
                    data_parallel_replicate_degree=1,
                ),
            ),
        ),
        data_setting=dict(
            max_tokens=81920,
            qwen_max_video_token_length=8192,
        ),
        checkpoint=dict(
            save_iter=100,
            load_from_object_store=dict(enabled=False, credentials="", bucket=""),
            save_to_object_store=dict(enabled=False, credentials="", bucket=""),
        ),
        dataloader_train=_video_conversation_dataloader(
            annotation_env="COSMOS_VIDEO_TRAIN_ANNOTATION",
            media_env="COSMOS_VIDEO_TRAIN_MEDIA",
            limit_env="COSMOS_VIDEO_TRAIN_LIMIT",
            shuffle=True,
        ),
        dataloader_val=_video_conversation_dataloader(
            annotation_env="COSMOS_VIDEO_VAL_ANNOTATION",
            media_env="COSMOS_VIDEO_VAL_MEDIA",
            limit_env="COSMOS_VIDEO_VAL_LIMIT",
            shuffle=False,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cosmos_task_aware_video_reasoning = deepcopy(cosmos_video_conversation)
cosmos_task_aware_video_reasoning["job"]["group"] = "cosmos_task_aware_video_reasoning_sft"
cosmos_task_aware_video_reasoning["dataloader_train"] = _task_aware_video_dataloader(split="train", shuffle=True)
cosmos_task_aware_video_reasoning["dataloader_val"] = _task_aware_video_dataloader(split="val", shuffle=False)


def _edge_recipe(recipe: LazyDict, group: str) -> LazyDict:
    edge = deepcopy(recipe)
    edge["defaults"][4] = {"override /vlm_policy": "cosmos3_edge_reasoner"}
    edge["job"]["group"] = group
    edge["optimizer"].pop("lr_multipliers", None)
    edge["model"]["config"]["policy"]["model_max_length"] = 16000
    return edge


cosmos_video_conversation_edge = _edge_recipe(cosmos_video_conversation, "cosmos_video_conversation_edge_sft")
cosmos_task_aware_video_reasoning_edge = _edge_recipe(
    cosmos_task_aware_video_reasoning, "cosmos_task_aware_video_reasoning_edge_sft"
)

for name, node in (
    ("cosmos_video_conversation", cosmos_video_conversation),
    ("cosmos_task_aware_video_reasoning", cosmos_task_aware_video_reasoning),
    ("cosmos_video_conversation_edge", cosmos_video_conversation_edge),
    ("cosmos_task_aware_video_reasoning_edge", cosmos_task_aware_video_reasoning_edge),
):
    ConfigStore.instance().store(group="experiment", package="_global_", name=name, node=node)
