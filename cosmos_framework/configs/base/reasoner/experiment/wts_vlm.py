# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Woven Traffic Safety (WTS) video-QA SFT on the Cosmos3-Nano VLM path."""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from PIL import Image
from torch.utils.data import Dataset

from cosmos_framework.callbacks.cosmos_dataloader_state import CosmosDataLoaderStateCallback
from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import VLMCollator, VLMProcessor
from cosmos_framework.data.generator.dataflow import CosmosDataLoader, MapDistributor, PoolPackingBatcher
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX


class WTSLlavaDataset(Dataset):
    """Map-style loader for WTS LLaVA JSON annotations."""

    def __init__(
        self,
        annotation_path: str,
        media_path: str,
        limit: int | str | None = None,
    ) -> None:
        self.annotation_path = os.path.abspath(os.path.expanduser(annotation_path))
        self.media_path = os.path.abspath(os.path.expanduser(media_path))
        if limit in ("", None):
            parsed_limit = None
        else:
            parsed_limit = int(limit)
            if parsed_limit < 1:
                parsed_limit = None

        with open(self.annotation_path, encoding="utf-8") as annotation_file:
            records = json.load(annotation_file)
        if not isinstance(records, list):
            raise TypeError(f"WTS annotations must be a JSON array, got {type(records).__name__}")
        self.records = records[:parsed_limit] if parsed_limit is not None else records
        if not self.records:
            raise ValueError(f"WTS annotation file contains no usable records: {self.annotation_path}")

        for index, record in enumerate(self.records):
            if not isinstance(record, dict) or not isinstance(record.get("video"), str):
                raise ValueError(f"WTS record {index} must contain a string 'video' field")
            conversations = record.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"WTS record {index} must contain at least two conversation turns")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        video_path = record["video"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.media_path, video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"WTS video does not exist: {video_path}")
        record["video"] = video_path
        return record


class WTSProcessor(VLMProcessor):
    """Convert WTS ShareGPT records and uniformly sample each video to PIL frames."""

    def __init__(
        self,
        processor: Any,
        ignore_index: int = IGNORE_INDEX,
        num_video_frames: int = 8,
        video_cache_size: int = 8,
        system_prompt: str = "",
    ) -> None:
        super().__init__(processor=processor, ignore_index=ignore_index)
        if num_video_frames < 1:
            raise ValueError("num_video_frames must be >= 1")
        if video_cache_size < 0:
            raise ValueError("video_cache_size must be >= 0")
        self.num_video_frames = num_video_frames
        self.video_cache_size = video_cache_size
        self.system_prompt = system_prompt
        self._video_cache: OrderedDict[str, tuple[list[Image.Image], float]] = OrderedDict()

    def _decode_video(self, video_path: str) -> tuple[list[Image.Image], float]:
        cached = self._video_cache.get(video_path)
        if cached is not None:
            self._video_cache.move_to_end(video_path)
            return cached

        reader = TorchCodecVideoReader(video_path, num_threads=2)
        total_frames = len(reader)
        if total_frames < 1:
            raise ValueError(f"WTS video has zero frames: {video_path}")
        sample_count = min(self.num_video_frames, total_frames)
        if sample_count == 1:
            indices = [0]
        else:
            indices = torch.linspace(0, total_frames - 1, steps=sample_count).round().to(dtype=torch.long).tolist()
        frames_np = reader.get_frames_nhwc_uint8(indices)
        frames = [Image.fromarray(frame) for frame in frames_np]

        source_fps = reader.get_avg_fps()
        average_stride = (indices[-1] - indices[0]) / max(len(indices) - 1, 1) if len(indices) > 1 else 1.0
        effective_fps = source_fps / max(average_stride, 1.0)
        decoded = (frames, float(effective_fps))
        if self.video_cache_size > 0:
            self._video_cache[video_path] = decoded
            self._video_cache.move_to_end(video_path)
            while len(self._video_cache) > self.video_cache_size:
                self._video_cache.popitem(last=False)
        return decoded

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        conversations = item.get("conversations", [])
        video_path = item.get("video")
        frames, fps = self._decode_video(video_path)
        messages: list[dict] = []
        video_inserted = False
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = re.sub(r"(\n)?</?(image|video)>(\n)?", "", turn["value"]).strip()
            if role == "user" and not video_inserted:
                content: Any = [
                    {"type": "video", "video": frames, "fps": fps},
                    {"type": "text", "text": text},
                ]
                video_inserted = True
            else:
                content = text
            messages.append({"role": role, "content": content})
        return messages


def _wts_dataloader(
    *,
    annotation_env: str,
    media_env: str,
    limit_env: str,
    shuffle: bool,
) -> LazyDict:
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(WTSLlavaDataset)(
                annotation_path=f"${{oc.env:{annotation_env}}}",
                media_path=f"${{oc.env:{media_env}}}",
                limit=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed=42,
            name="train" if shuffle else "val",
        ),
        processor=L(WTSProcessor)(
            processor=L(build_processor)(
                tokenizer_type="${model.config.policy.backbone.model_name}",
                config_variant="hf",
            ),
            ignore_index=IGNORE_INDEX,
            num_video_frames=8,
            video_cache_size=8,
            system_prompt=("You are a helpful assistant that can answer questions about a street-view CCTV footage."),
        ),
        batcher=L(PoolPackingBatcher)(
            max_tokens="${data_setting.max_tokens}",
            pool_size=16 if shuffle else 1,
            max_batch_size=1,
            long_threshold=6400,
        ),
        collator=L(VLMCollator)(),
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
    )


wts_vlm = LazyDict(
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
            group="wts_sft",
            wandb_mode="disabled",
        ),
        trainer=dict(
            callbacks=dict(
                dataloader_state=L(CosmosDataLoaderStateCallback)(),
                tao=dict(
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
        dataloader_train=_wts_dataloader(
            annotation_env="WTS_TRAIN_ANNOTATION",
            media_env="WTS_TRAIN_MEDIA",
            limit_env="WTS_TRAIN_LIMIT",
            shuffle=True,
        ),
        dataloader_val=_wts_dataloader(
            annotation_env="WTS_VAL_ANNOTATION",
            media_env="WTS_VAL_MEDIA",
            limit_env="WTS_VAL_LIMIT",
            shuffle=False,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="wts_vlm",
    node=wts_vlm,
)


# Edge uses the same WTS data contract but a different native HF backbone.
# Keep it in this module so the dataloader worker callables retain a single
# pickle identity when the reasoner config loader reloads experiment modules.
wts_vlm_edge = deepcopy(wts_vlm)
wts_vlm_edge["defaults"][4] = {"override /vlm_policy": "cosmos3_edge_reasoner"}
wts_vlm_edge["job"]["group"] = "wts_edge_sft"
wts_vlm_edge["optimizer"].pop("lr_multipliers", None)
wts_vlm_edge["model"]["config"]["policy"]["model_max_length"] = 16000

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="wts_vlm_edge",
    node=wts_vlm_edge,
)
