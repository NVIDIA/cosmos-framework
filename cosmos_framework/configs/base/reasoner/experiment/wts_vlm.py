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
from cosmos_framework.data.generator.dataflow import ContiguousBatcher, CosmosDataLoader, MapDistributor
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
            media_value = (
                next(
                    (
                        record.get(field)
                        for field in ("video", "video_id", "media", "media_path")
                        if isinstance(record.get(field), str)
                    ),
                    None,
                )
                if isinstance(record, dict)
                else None
            )
            if media_value is None:
                raise ValueError(f"video-conversation record {index} must contain a string media field")
            conversations = record.get("conversations") or record.get("messages")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"video-conversation record {index} must contain at least two conversation turns")

    def __len__(self) -> int:
        return len(self.records)

    def media_identity(self, index: int) -> str:
        """Return the stable local-media identity used by validation caches."""
        record = self.records[index]
        video_path = next(
            record[field]
            for field in ("video", "video_id", "media", "media_path")
            if isinstance(record.get(field), str)
        )
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.media_path, video_path)
        return os.path.realpath(video_path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        video_path = next(
            record[field]
            for field in ("video", "video_id", "media", "media_path")
            if isinstance(record.get(field), str)
        )
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.media_path, video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video-conversation media does not exist: {video_path}")
        record["video"] = video_path
        return record


class _ProcessedVideoCacheProxy:
    """On-demand worker-local cache around the HF video preprocessor."""

    def __init__(self, processor: Any, capacity: int) -> None:
        self._processor = processor
        self.capacity = int(capacity)
        self._entries: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict[tuple, threading.Event] = {}
        self._hit_attested = False

    def __getattr__(self, name: str) -> Any:
        processor = self.__dict__.get("_processor")
        if processor is None:
            raise AttributeError(name)
        return getattr(processor, name)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state["_entries"] = OrderedDict()
        state["_inflight"] = {}
        state["_hit_attested"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._inflight = {}

    @staticmethod
    def _identity(videos: Any) -> tuple | None:
        frames: list[tuple[int, tuple[int, int], str]] = []

        def collect(value: Any) -> None:
            if isinstance(value, Image.Image):
                frames.append((id(value), value.size, value.mode))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(videos)
        return tuple(frames) if frames else None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        videos = kwargs.get("videos", args[0] if args else None)
        key = self._identity(videos)
        if key is None or self.capacity <= 0:
            return self._processor(*args, **kwargs)

        while True:
            with self._lock:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    if not self._hit_attested:
                        print(
                            "TAO_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_HIT_ATTESTATION "
                            f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                            f"capacity={self.capacity}",
                            flush=True,
                        )
                        self._hit_attested = True
                    return deepcopy(cached)
                inflight = self._inflight.get(key)
                if inflight is None:
                    inflight = threading.Event()
                    self._inflight[key] = inflight
                    owner = True
                else:
                    owner = False
            if owner:
                break
            inflight.wait()

        try:
            output = self._processor(*args, **kwargs)
            canonical = deepcopy(output)
            with self._lock:
                self._entries[key] = canonical
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
            return output
        finally:
            with self._lock:
                completed = self._inflight.pop(key, None)
                if completed is not None:
                    completed.set()


class VideoSFTProcessor(VLMProcessor):
    """Convert video-supervision records and uniformly sample media to PIL frames."""

    @staticmethod
    def _resolve_video_device(video_device: str) -> str:
        """Bind a generic CUDA request to this torchrun process's local rank."""
        requested = str(video_device)
        if requested != "cuda":
            return requested
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is None:
            return requested
        try:
            rank = int(local_rank)
        except ValueError as exc:
            raise ValueError(f"LOCAL_RANK must be an integer, found {local_rank!r}") from exc
        if rank < 0:
            raise ValueError(f"LOCAL_RANK must be non-negative, found {rank}")
        return f"cuda:{rank}"

    def __init__(
        self,
        processor: Any,
        ignore_index: int = IGNORE_INDEX,
        num_video_frames: int = 8,
        video_cache_size: int = 8,
        video_device: str = "cuda",
        video_num_threads: int = 1,
        processed_video_cache_size: int = 0,
        video_max_pixels: int | str | None = 81920,
        video_override_map: str | None = None,
        system_prompt: str = "",
        use_daft_chat_template: bool = False,
    ) -> None:
        super().__init__(processor=processor, ignore_index=ignore_index)
        num_video_frames = int(num_video_frames)
        video_cache_size = int(video_cache_size)
        video_num_threads = int(video_num_threads)
        processed_video_cache_size = int(processed_video_cache_size)
        if num_video_frames < 1:
            raise ValueError("num_video_frames must be >= 1")
        if video_cache_size < 0:
            raise ValueError("video_cache_size must be >= 0")
        if processed_video_cache_size < 0:
            raise ValueError("processed_video_cache_size must be >= 0")
        self.num_video_frames = num_video_frames
        self.video_cache_size = video_cache_size
        self.requested_video_device = str(video_device)
        self.video_device = self._resolve_video_device(self.requested_video_device)
        self.video_num_threads = video_num_threads
        self.processed_video_cache_size = processed_video_cache_size
        hf_processor = getattr(processor, "processor", None)
        video_processor = getattr(hf_processor, "video_processor", None)
        if processed_video_cache_size:
            if video_processor is None:
                raise RuntimeError(
                    "processed video caching requires processor.video_processor"
                )
            hf_processor.video_processor = _ProcessedVideoCacheProxy(
                video_processor,
                processed_video_cache_size,
            )
            print(
                "TAO_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_ENABLED_ATTESTATION "
                f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                f"capacity={processed_video_cache_size} population=on_demand",
                flush=True,
            )
        self.video_overrides: dict[str, str] = {}
        if video_override_map not in (None, ""):
            override_path = os.path.abspath(os.path.expanduser(str(video_override_map)))
            with open(override_path, encoding="utf-8") as override_file:
                overrides = json.load(override_file)
            if not isinstance(overrides, dict) or not all(
                isinstance(source, str) and isinstance(target, str) for source, target in overrides.items()
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

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local synchronization and cache state before spawn."""
        state = self.__dict__.copy()
        state.pop("_video_cache_lock", None)
        state["_video_cache"] = OrderedDict()
        state["_video_inflight"] = {}
        state["_video_runtime_attested"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Recreate rank-local cache synchronization in a spawned worker."""
        self.__dict__.update(state)
        self._video_cache_lock = threading.Lock()
        self._video_inflight = {}

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
            if self.video_device.startswith("cuda"):
                requested_device = torch.device(self.video_device)
                actual_device = torch.device(decoded_device)
                if actual_device.type != "cuda" or (
                    requested_device.index is not None and actual_device.index != requested_device.index
                ):
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
                        f"requested_device={self.requested_video_device} "
                        f"resolved_device={self.video_device} actual_device={decoded_device} "
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

    def process(self, item: dict) -> dict:
        sample = super().process(item)
        video_path = item.get("video")
        if isinstance(video_path, str):
            video_path = self.video_overrides.get(video_path, video_path)
            sample["tao_video_cache_key"] = os.path.realpath(
                os.path.abspath(os.path.expanduser(video_path))
            )
        return sample


class VideoVLMCollator(VLMCollator):
    """Preserve one stable video identity per sample for validation caching."""

    def collate(self, samples: list[dict]) -> dict:
        cache_keys = [sample.get("tao_video_cache_key") for sample in samples]
        batch = super().collate(samples)
        batch.pop("tao_video_cache_key", None)
        if all(isinstance(key, str) for key in cache_keys):
            batch["tao_video_cache_keys"] = cache_keys
        return batch


class MediaGroupedMapDistributor(MapDistributor):
    """Group repeated validation media while preserving the padded multiset.

    Every DP-rank/worker stream receives exactly ``ceil(N / streams)`` records.
    Whole media groups are assigned first and split only when required to fill
    equal stream capacities, so FSDP ranks execute the same number of forwards.
    The validation stream is finite so ``drop_last=False`` can emit one equal
    partial final batch per rank without pulling records from the next epoch.
    """

    finite_validation_stream = True

    @staticmethod
    def _staged_cache_frontload(
        rank_groups: OrderedDict[str, list[int]],
        batch_size: int,
        unique_per_batch: int,
    ) -> list[int]:
        """Bound unseen media per early batch while preserving every index."""
        if batch_size <= 0 or unique_per_batch <= 0 or unique_per_batch > batch_size:
            raise ValueError(
                "staged validation cache frontloading requires positive batch and "
                "unique counts with unique <= batch"
            )
        remaining = OrderedDict((key, list(group)) for key, group in rank_groups.items())
        group_keys = list(remaining)
        active_keys: list[str] = []
        ordered: list[int] = []

        for start in range(0, len(group_keys), unique_per_batch):
            new_keys = group_keys[start : start + unique_per_batch]
            active_keys.extend(new_keys)
            batch = [remaining[key].pop(0) for key in new_keys]
            while len(batch) < batch_size:
                made_progress = False
                for key in active_keys:
                    if remaining[key]:
                        batch.append(remaining[key].pop(0))
                        made_progress = True
                        if len(batch) == batch_size:
                            break
                if not made_progress:
                    break
            ordered.extend(batch)

        for group in remaining.values():
            ordered.extend(group)
        return ordered

    @staticmethod
    def _assign_groups(dataset: VideoConversationDataset, total_streams: int) -> list[list[int]]:
        n = len(dataset)
        per_stream = (n + total_streams - 1) // total_streams
        padded = list(range(n))
        padded.extend(padded[: per_stream * total_streams - n])

        groups: OrderedDict[str, list[int]] = OrderedDict()
        for index in padded:
            groups.setdefault(dataset.media_identity(index), []).append(index)

        assignments = [[] for _ in range(total_streams)]
        remaining = [per_stream] * total_streams
        ordered_groups = sorted(enumerate(groups.values()), key=lambda item: (-len(item[1]), item[0]))
        for _, original_group in ordered_groups:
            group = list(original_group)
            while group:
                fitting = [stream for stream, capacity in enumerate(remaining) if capacity >= len(group)]
                if fitting:
                    stream = max(fitting, key=lambda value: (remaining[value], -value))
                    take = len(group)
                else:
                    stream = max(range(total_streams), key=lambda value: (remaining[value], -value))
                    take = remaining[stream]
                if take <= 0:
                    raise RuntimeError("media-grouped validation sharding exhausted stream capacity")
                assignments[stream].extend(group[:take])
                remaining[stream] -= take
                del group[:take]

        if any(remaining) or any(len(indices) != per_stream for indices in assignments):
            raise RuntimeError("media-grouped validation sharding did not produce equal stream lengths")

        batch_size = int(os.environ.get("TAO_FRAMEWORK_VALIDATION_BATCH_SIZE", "1"))
        unique_per_batch = int(
            os.environ.get(
                "TAO_FRAMEWORK_VALIDATION_CACHE_FRONTLOAD_UNIQUE_PER_BATCH",
                str(max(1, batch_size // 2)),
            )
        )
        staged_assignments: list[list[int]] = []
        for assignment in assignments:
            rank_groups: OrderedDict[str, list[int]] = OrderedDict()
            for index in assignment:
                rank_groups.setdefault(dataset.media_identity(index), []).append(index)
            staged = MediaGroupedMapDistributor._staged_cache_frontload(
                rank_groups,
                batch_size,
                unique_per_batch,
            )
            if len(staged) != per_stream or sorted(staged) != sorted(assignment):
                raise RuntimeError(
                    "staged validation cache frontloading changed the rank-local multiset"
                )
            staged_assignments.append(staged)
        assignments = staged_assignments
        return assignments

    def stream(self, dp_rank, dp_world_size, worker_id, num_workers):
        if self._shuffle:
            raise ValueError("MediaGroupedMapDistributor is validation-only and requires shuffle=False")
        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        assignments = self._assign_groups(self._dataset, total_streams)
        if stream_id >= len(assignments):
            return
        for position, index in enumerate(assignments[stream_id]):
            item = self._dataset[index]
            if isinstance(item, dict):
                yield {"_dp_epoch": 0, "_dp_stream_pos": position, **item}
            else:
                yield item


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
    validation_grouped = (
        not shuffle
        and os.environ.get("TAO_FRAMEWORK_VALIDATION_SHARD_STRATEGY", "stride") == "media_grouped"
    )
    distributor_cls = MediaGroupedMapDistributor if validation_grouped else MapDistributor
    max_batch_size = (
        int(os.environ.get("TAO_FRAMEWORK_VALIDATION_BATCH_SIZE", "1")) if not shuffle else 1
    )
    return L(CosmosDataLoader)(
        distributor=L(distributor_cls)(
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
            processed_video_cache_size=(
                "${oc.env:TAO_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_SIZE,0}"
                if not shuffle
                else 0
            ),
            video_max_pixels=f"${{oc.env:{max_pixels_env},81920}}",
            video_override_map="${oc.env:TAO_VIDEO_OVERRIDE_MAP,''}",
            system_prompt=f"${{oc.env:{system_prompt_env},''}}",
        ),
        batcher=L(ContiguousBatcher)(
            max_batch_size=max_batch_size,
            max_tokens=81920,
            drop_last=False,
        ),
        collator=L(VideoVLMCollator)(),
        num_workers="${oc.env:TAO_FRAMEWORK_DATALOADER_NUM_WORKERS,1}",
        prefetch_factor="${oc.env:TAO_FRAMEWORK_DATALOADER_PREFETCH_FACTOR,4}",
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context="spawn",
        processing_threads="${oc.env:TAO_FRAMEWORK_SFT_PROCESS_THREADS,8}",
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
            video_max_pixels=f"${{oc.env:{max_pixels_env},81920}}",
            video_override_map="${oc.env:TAO_VIDEO_OVERRIDE_MAP,''}",
            system_prompt="",
            use_daft_chat_template=True,
        ),
        batcher=L(ContiguousBatcher)(
            max_batch_size=1,
            max_tokens=81920,
            drop_last=False,
        ),
        collator=L(VLMCollator)(),
        num_workers="${oc.env:TAO_FRAMEWORK_DATALOADER_NUM_WORKERS,1}",
        prefetch_factor="${oc.env:TAO_FRAMEWORK_DATALOADER_PREFETCH_FACTOR,2}",
        persistent_workers=True,
        pin_memory=False,
        multiprocessing_context="spawn",
        processing_threads="${oc.env:TAO_FRAMEWORK_SFT_PROCESS_THREADS,8}",
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
    split="train",
    shuffle=True,
    annotation_env="TAO_VIDEO_TRAIN_ANNOTATIONS",
    media_env="TAO_VIDEO_TRAIN_MEDIA_ROOTS",
    limit_env="TAO_VIDEO_TRAIN_LIMIT",
    frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE",
    max_pixels_env="TAO_VIDEO_MAX_PIXELS",
    system_prompt_env="TAO_VIDEO_SYSTEM_PROMPT",
)
tao_task_aware_video_reasoning["dataloader_val"] = _task_aware_video_dataloader(
    split="val",
    shuffle=False,
    annotation_env="TAO_VIDEO_VAL_ANNOTATIONS",
    media_env="TAO_VIDEO_VAL_MEDIA_ROOTS",
    limit_env="TAO_VIDEO_VAL_LIMIT",
    frame_env="TAO_VIDEO_NUM_FRAMES",
    cache_env="TAO_VIDEO_CACHE_SIZE",
    max_pixels_env="TAO_VIDEO_MAX_PIXELS",
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
