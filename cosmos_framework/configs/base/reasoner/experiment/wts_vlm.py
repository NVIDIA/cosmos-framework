# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generic conversation and task-aware video SFT recipes for Cosmos3."""

from __future__ import annotations

import json
import os
import re
import threading
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
from cosmos_framework.data.generator.local_datasets.tao_vl_reason import (
    TaoVlReasonDaftDataset,
    apply_daft_chat_template,
)
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX


class VideoConversationDataset(Dataset):
    """Map-style loader for ShareGPT/LLaVA-style video conversations."""

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
            raise TypeError(f"video-conversation annotations must be a JSON array, got {type(records).__name__}")
        self.records = records[:parsed_limit] if parsed_limit is not None else records
        if not self.records:
            raise ValueError(f"video-conversation annotation file contains no usable records: {self.annotation_path}")

        for index, record in enumerate(self.records):
            media_value = next(
                (record.get(field) for field in ("video", "video_id", "media", "media_path") if isinstance(record.get(field), str)),
                None,
            ) if isinstance(record, dict) else None
            if media_value is None:
                raise ValueError(f"video-conversation record {index} must contain a string media field")
            conversations = record.get("conversations") or record.get("messages")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"video-conversation record {index} must contain at least two conversation turns")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        video_path = next(
            record[field] for field in ("video", "video_id", "media", "media_path")
            if isinstance(record.get(field), str)
        )
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.media_path, video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video-conversation media does not exist: {video_path}")
        record["video"] = video_path
        return record


class VideoSFTProcessor(VLMProcessor):
    """Convert video-supervision records and uniformly sample media to PIL frames."""

    def __init__(
        self,
        processor: Any,
        ignore_index: int = IGNORE_INDEX,
        num_video_frames: int = 8,
        video_cache_size: int = 8,
        video_device: str = "cuda",
        video_num_threads: int = 1,
        video_max_pixels: int | str | None = None,
        video_override_map: str | None = None,
        system_prompt: str = "",
        use_daft_chat_template: bool = False,
    ) -> None:
        super().__init__(processor=processor, ignore_index=ignore_index)
        num_video_frames = int(num_video_frames)
        video_cache_size = int(video_cache_size)
        video_num_threads = int(video_num_threads)
        if num_video_frames < 1:
            raise ValueError("num_video_frames must be >= 1")
        if video_cache_size < 0:
            raise ValueError("video_cache_size must be >= 0")
        self.num_video_frames = num_video_frames
        self.video_cache_size = video_cache_size
        self.video_device = video_device
        self.video_num_threads = video_num_threads
        self.video_overrides: dict[str, str] = {}
        if video_override_map not in (None, ""):
            override_path = os.path.abspath(os.path.expanduser(str(video_override_map)))
            with open(override_path, encoding="utf-8") as override_file:
                overrides = json.load(override_file)
            if not isinstance(overrides, dict) or not all(
                isinstance(source, str) and isinstance(target, str)
                for source, target in overrides.items()
            ):
                raise ValueError("video_override_map must be a JSON object of string paths")
            self.video_overrides = overrides
        self.video_max_pixels: int | None = None
        if video_max_pixels not in (None, "", 0, "0"):
            parsed_video_max_pixels = int(video_max_pixels)
            if parsed_video_max_pixels < 1:
                raise ValueError("video_max_pixels must be >= 1")
            hf_processor = getattr(processor, "processor", processor)
            video_processor = getattr(hf_processor, "video_processor", None)
            size = getattr(video_processor, "size", None)
            if not isinstance(size, dict):
                raise ValueError("video_max_pixels requires a processor.video_processor.size mapping")
            shortest_edge = size.get("shortest_edge")
            if shortest_edge is not None and parsed_video_max_pixels < int(shortest_edge):
                raise ValueError(
                    f"video_max_pixels ({parsed_video_max_pixels}) must be >= shortest_edge ({shortest_edge})"
                )
            size["longest_edge"] = parsed_video_max_pixels
            self.video_max_pixels = parsed_video_max_pixels
        self.system_prompt = system_prompt
        self.use_daft_chat_template = use_daft_chat_template
        if self.use_daft_chat_template:
            apply_daft_chat_template(processor)
        self._video_cache: OrderedDict[str, tuple[list[Image.Image], float]] = OrderedDict()
        self._video_cache_lock = threading.Lock()
        self._video_inflight: dict[str, threading.Event] = {}
        self._video_runtime_attested = False

    def _decode_video(self, video_path: str) -> tuple[list[Image.Image], float]:
        video_path = self.video_overrides.get(video_path, video_path)
        video_path = os.path.abspath(os.path.expanduser(video_path))
        if self.video_cache_size > 0:
            # Concurrent processing can request the same source video in one
            # logical pool. Elect one decoder and let peers consume its cached
            # result, avoiding duplicate GPU decoder sessions without prewarm.
            while True:
                with self._video_cache_lock:
                    cached = self._video_cache.get(video_path)
                    if cached is not None:
                        self._video_cache.move_to_end(video_path)
                        return cached
                    inflight = self._video_inflight.get(video_path)
                    if inflight is None:
                        inflight = threading.Event()
                        self._video_inflight[video_path] = inflight
                        decode_owner = True
                    else:
                        decode_owner = False
                if decode_owner:
                    break
                inflight.wait()

        try:
            reader = TorchCodecVideoReader(
                video_path,
                num_threads=self.video_num_threads,
                device=self.video_device,
            )
            total_frames = len(reader)
            if total_frames < 1:
                raise ValueError(f"video-supervision media has zero frames: {video_path}")
            sample_count = min(self.num_video_frames, total_frames)
            if sample_count == 1:
                indices = [0]
            else:
                indices = torch.linspace(0, total_frames - 1, steps=sample_count).round().to(dtype=torch.long).tolist()
            frames_np = reader.get_frames_nhwc_uint8(indices)
            decoded_device = str(reader.last_output_device)
            if self.video_device.startswith("cuda") and not decoded_device.startswith("cuda"):
                raise RuntimeError(
                    "TorchCodec did not decode on the requested CUDA device: "
                    f"requested={self.video_device} actual={decoded_device}"
                )
            frames = [Image.fromarray(frame) for frame in frames_np]

            with self._video_cache_lock:
                if not self._video_runtime_attested:
                    print(
                        "TAO_FRAMEWORK_VIDEO_RUNTIME "
                        f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                        "backend=torchcodec "
                        f"requested_device={self.video_device} actual_device={decoded_device} "
                        f"video_cache_size={self.video_cache_size} "
                        f"decoder_threads={self.video_num_threads}",
                        flush=True,
                    )
                    self._video_runtime_attested = True

            source_fps = reader.get_avg_fps()
            average_stride = (indices[-1] - indices[0]) / max(len(indices) - 1, 1) if len(indices) > 1 else 1.0
            effective_fps = source_fps / max(average_stride, 1.0)
            decoded = (frames, float(effective_fps))
            if self.video_cache_size > 0:
                with self._video_cache_lock:
                    self._video_cache[video_path] = decoded
                    self._video_cache.move_to_end(video_path)
                    while len(self._video_cache) > self.video_cache_size:
                        self._video_cache.popitem(last=False)
            return decoded
        finally:
            if self.video_cache_size > 0:
                with self._video_cache_lock:
                    completed = self._video_inflight.pop(video_path, None)
                    if completed is not None:
                        completed.set()

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        if "messages" in item:
            messages = deepcopy(item["messages"])
            video_inserted = False
            for message in messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") != "video":
                        continue
                    video_path = part.get("video")
                    if not isinstance(video_path, str):
                        raise TypeError("task-aware video content must contain a string path")
                    frames, fps = self._decode_video(video_path)
                    part["video"] = frames
                    part["fps"] = fps
                    video_inserted = True
            if not video_inserted and isinstance(item.get("video"), str):
                frames, fps = self._decode_video(item["video"])
                for message in messages:
                    if message.get("role") != "user":
                        continue
                    content = message.get("content", "")
                    message["content"] = [
                        {"type": "video", "video": frames, "fps": fps},
                        {"type": "text", "text": content if isinstance(content, str) else ""},
                    ]
                    break
            return messages

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


def _video_conversation_dataloader(
    *,
    annotation_env: str,
    media_env: str,
    limit_env: str,
    shuffle: bool,
    frame_env: str = "WTS_NUM_VIDEO_FRAMES",
    cache_env: str = "WTS_VIDEO_CACHE_SIZE",
    max_pixels_env: str = "WTS_VIDEO_MAX_PIXELS",
    system_prompt_env: str = "WTS_SYSTEM_PROMPT",
) -> LazyDict:
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(VideoConversationDataset)(
                annotation_path=f"${{oc.env:{annotation_env}}}",
                media_path=f"${{oc.env:{media_env}}}",
                limit=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed="${oc.env:TAO_DATALOADER_SEED,42}",
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
            video_device="${oc.env:TAO_VIDEO_DECODER_DEVICE,cuda}",
            video_num_threads="${oc.env:TAO_VIDEO_DECODER_THREADS,1}",
            video_max_pixels=f"${{oc.env:{max_pixels_env},''}}",
            video_override_map="${oc.env:TAO_VIDEO_OVERRIDE_MAP,''}",
            system_prompt=f"${{oc.env:{system_prompt_env},''}}",
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
        processing_threads="${oc.env:TAO_FRAMEWORK_SFT_PROCESS_THREADS,1}",
    )


def _task_aware_video_dataloader(
    *,
    split: str,
    shuffle: bool,
    annotation_env: str | None = None,
    media_env: str | None = None,
    limit_env: str | None = None,
    frame_env: str = "AETC_NUM_VIDEO_FRAMES",
    cache_env: str = "AETC_VIDEO_CACHE_SIZE",
    max_pixels_env: str = "AETC_VIDEO_MAX_PIXELS",
    system_prompt_env: str = "AETC_SYSTEM_PROMPT",
) -> LazyDict:
    annotation_env = annotation_env or f"AETC_{split.upper()}_ANNOTATIONS"
    media_env = media_env or f"AETC_{split.upper()}_MEDIA"
    limit_env = limit_env or f"AETC_{split.upper()}_LIMIT"
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(TaoVlReasonDaftDataset)(
                annotation_paths=f"${{oc.env:{annotation_env}}}",
                media_root=f"${{oc.env:{media_env}}}",
                response_mode="hybrid" if split == "train" else "answer",
                system_prompt=f"${{oc.env:{system_prompt_env},''}}",
                vision_kwargs={},
                max_samples=f"${{oc.env:{limit_env},''}}",
            ),
            shuffle=shuffle,
            seed="${oc.env:TAO_DATALOADER_SEED,42}",
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
            video_device="${oc.env:TAO_VIDEO_DECODER_DEVICE,cuda}",
            video_num_threads="${oc.env:TAO_VIDEO_DECODER_THREADS,1}",
            video_max_pixels=f"${{oc.env:{max_pixels_env},''}}",
            video_override_map="${oc.env:TAO_VIDEO_OVERRIDE_MAP,''}",
            system_prompt="",
            use_daft_chat_template=True,
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
        processing_threads="${oc.env:TAO_FRAMEWORK_SFT_PROCESS_THREADS,1}",
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
        dataloader_train=_video_conversation_dataloader(
            annotation_env="WTS_TRAIN_ANNOTATION",
            media_env="WTS_TRAIN_MEDIA",
            limit_env="WTS_TRAIN_LIMIT",
            shuffle=True,
        ),
        dataloader_val=_video_conversation_dataloader(
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


# Internal TAO AETC path: keep Framework's trainer/model implementation while
# consuming the same DAFT dataset and Qwen chat-template contract as Cosmos-RL.
aetc_daft_vlm = deepcopy(wts_vlm)
aetc_daft_vlm["job"]["group"] = "aetc_daft_sft"
aetc_daft_vlm["dataloader_train"] = _task_aware_video_dataloader(split="train", shuffle=True)
aetc_daft_vlm["dataloader_val"] = _task_aware_video_dataloader(split="val", shuffle=False)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="aetc_daft_vlm",
    node=aetc_daft_vlm,
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


# Edge AETC uses the native Edge policy and the same DAFT dataset contract as
# the Nano AETC recipe. Runtime processor limits are supplied by TAO rather
# than encoded by modifying the public model checkpoint.
aetc_daft_vlm_edge = deepcopy(wts_vlm_edge)
aetc_daft_vlm_edge["job"]["group"] = "aetc_daft_edge_sft"
aetc_daft_vlm_edge["dataloader_train"] = _task_aware_video_dataloader(split="train", shuffle=True)
aetc_daft_vlm_edge["dataloader_val"] = _task_aware_video_dataloader(split="val", shuffle=False)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="aetc_daft_vlm_edge",
    node=aetc_daft_vlm_edge,
)


# Dataset-neutral TAO contracts. The older experiment registrations above are
# retained only so existing result configs remain loadable.
tao_video_conversation = deepcopy(wts_vlm)
tao_video_conversation["job"]["group"] = "tao_video_conversation_sft"
tao_video_conversation["dataloader_train"] = _video_conversation_dataloader(
    annotation_env="TAO_VIDEO_TRAIN_ANNOTATION",
    media_env="TAO_VIDEO_TRAIN_MEDIA",
    limit_env="TAO_VIDEO_TRAIN_LIMIT",
    shuffle=True,
    frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE",
    max_pixels_env="TAO_VIDEO_MAX_PIXELS",
    system_prompt_env="TAO_VIDEO_SYSTEM_PROMPT",
)
tao_video_conversation["dataloader_val"] = _video_conversation_dataloader(
    annotation_env="TAO_VIDEO_VAL_ANNOTATION",
    media_env="TAO_VIDEO_VAL_MEDIA",
    limit_env="TAO_VIDEO_VAL_LIMIT",
    shuffle=False,
    frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE",
    max_pixels_env="TAO_VIDEO_MAX_PIXELS",
    system_prompt_env="TAO_VIDEO_SYSTEM_PROMPT",
)

tao_task_aware_video_reasoning = deepcopy(tao_video_conversation)
tao_task_aware_video_reasoning["job"]["group"] = "tao_task_aware_video_reasoning_sft"
tao_task_aware_video_reasoning["dataloader_train"] = _task_aware_video_dataloader(
    split="train", shuffle=True,
    annotation_env="TAO_VIDEO_TRAIN_ANNOTATIONS", media_env="TAO_VIDEO_TRAIN_MEDIA_ROOTS",
    limit_env="TAO_VIDEO_TRAIN_LIMIT", frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE", max_pixels_env="TAO_VIDEO_MAX_PIXELS",
    system_prompt_env="TAO_VIDEO_SYSTEM_PROMPT",
)
tao_task_aware_video_reasoning["dataloader_val"] = _task_aware_video_dataloader(
    split="val", shuffle=False,
    annotation_env="TAO_VIDEO_VAL_ANNOTATIONS", media_env="TAO_VIDEO_VAL_MEDIA_ROOTS",
    limit_env="TAO_VIDEO_VAL_LIMIT", frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE", max_pixels_env="TAO_VIDEO_MAX_PIXELS",
    system_prompt_env="TAO_VIDEO_SYSTEM_PROMPT",
)

tao_video_conversation_edge = deepcopy(tao_video_conversation)
tao_video_conversation_edge["defaults"][4] = {"override /vlm_policy": "cosmos3_edge_reasoner"}
tao_video_conversation_edge["job"]["group"] = "tao_video_conversation_edge_sft"
tao_video_conversation_edge["optimizer"].pop("lr_multipliers", None)
tao_video_conversation_edge["model"]["config"]["policy"]["model_max_length"] = 16000

tao_task_aware_video_reasoning_edge = deepcopy(tao_task_aware_video_reasoning)
tao_task_aware_video_reasoning_edge["defaults"][4] = {"override /vlm_policy": "cosmos3_edge_reasoner"}
tao_task_aware_video_reasoning_edge["job"]["group"] = "tao_task_aware_video_reasoning_edge_sft"
tao_task_aware_video_reasoning_edge["optimizer"].pop("lr_multipliers", None)
tao_task_aware_video_reasoning_edge["model"]["config"]["policy"]["model_max_length"] = 16000

for name, node in (
    ("tao_video_conversation", tao_video_conversation),
    ("tao_task_aware_video_reasoning", tao_task_aware_video_reasoning),
    ("tao_video_conversation_edge", tao_video_conversation_edge),
    ("tao_task_aware_video_reasoning_edge", tao_task_aware_video_reasoning_edge),
):
    ConfigStore.instance().store(group="experiment", package="_global_", name=name, node=node)


# Import compatibility for callers that used the development-dataset symbols.
WTSLlavaDataset = VideoConversationDataset
WTSProcessor = VideoSFTProcessor
