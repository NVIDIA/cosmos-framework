# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Framework-native CUDA TorchCodec preprocessing for evaluation."""

from __future__ import annotations

import copy
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class FrameworkTorchCodecVideoPreprocessor:
    """Decode evaluation videos with the same CUDA TorchCodec policy as SFT.

    The evaluator remains data parallel (one process and model replica per
    GPU).  This object therefore owns a rank-local LRU and a bounded,
    order-preserving preprocessing pool. The Framework evaluator owns this
    native TorchCodec adapter.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        cache_size: int,
        process_threads: int = 8,
        decoder_threads: int = 1,
        dataloader_num_workers: int = 1,
        dataloader_prefetch_factor: int = 2,
        dataloader_multiprocessing_context: str = "spawn",
        dataloader_persistent_workers: bool = True,
        device: str = "cuda",
        video_override_map: str | None = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.cache_size = int(cache_size)
        self.process_threads = int(process_threads)
        self.decoder_threads = int(decoder_threads)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.dataloader_prefetch_factor = int(dataloader_prefetch_factor)
        self.dataloader_multiprocessing_context = str(dataloader_multiprocessing_context)
        self.dataloader_persistent_workers = bool(dataloader_persistent_workers)
        if self.num_frames < 1:
            raise ValueError("Framework evaluation num_frames must be positive")
        if self.cache_size < 0:
            raise ValueError("Framework evaluation video_cache_size must be non-negative")
        if self.process_threads < 1:
            raise ValueError("Framework evaluation process_threads must be positive")
        if self.decoder_threads < 1:
            raise ValueError("Framework evaluation decoder_threads must be positive")
        if self.dataloader_num_workers not in (0, 1):
            raise ValueError("Framework evaluation supports zero or one DataLoader worker")
        if self.dataloader_num_workers == 0:
            if self.dataloader_prefetch_factor != 0:
                raise ValueError("Framework evaluation must disable DataLoader prefetch when workers are zero")
            if self.dataloader_persistent_workers:
                raise ValueError("Framework evaluation must disable persistent workers when workers are zero")
        else:
            if self.dataloader_prefetch_factor != 2:
                raise ValueError("Framework evaluation requires DataLoader prefetch factor two")
            if self.dataloader_multiprocessing_context != "spawn":
                raise ValueError("Framework evaluation requires spawned DataLoader workers")
            if not self.dataloader_persistent_workers:
                raise ValueError("Framework evaluation requires a persistent DataLoader worker")

        self.requested_device = str(device)
        self.device = self._resolve_device(self.requested_device)
        self.video_overrides: dict[str, str] = {}
        if video_override_map not in (None, ""):
            override_path = Path(str(video_override_map)).expanduser().resolve(strict=True)
            payload = json.loads(override_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                isinstance(source, str) and isinstance(target, str) for source, target in payload.items()
            ):
                raise ValueError("video_override_map must be a JSON object of string paths")
            self.video_overrides = payload

        self._cache: OrderedDict[str, tuple[list[Image.Image], float]] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._attested = False
        self._direct_prefetch_attested = False
        self._media_lookahead_attested = False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_inflight"] = {}
        state["_lock"] = None
        state["_attested"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache = OrderedDict()
        self._inflight = {}
        self._lock = threading.RLock()
        self._attested = False
        self._direct_prefetch_attested = False
        self._media_lookahead_attested = False

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "cuda":
            return requested
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0:
            raise ValueError(f"LOCAL_RANK must be non-negative, found {local_rank}")
        return f"cuda:{local_rank}"

    def _decode(self, source_path: str) -> tuple[list[Image.Image], float]:
        from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader

        resolved_path = self.video_overrides.get(source_path, source_path)
        resolved_path = str(Path(resolved_path).expanduser().resolve(strict=True))
        if self.cache_size > 0:
            while True:
                with self._lock:
                    cached = self._cache.get(resolved_path)
                    if cached is not None:
                        self._cache.move_to_end(resolved_path)
                        return cached
                    event = self._inflight.get(resolved_path)
                    if event is None:
                        event = threading.Event()
                        self._inflight[resolved_path] = event
                        owner = True
                    else:
                        owner = False
                if owner:
                    break
                event.wait()

        try:
            reader = TorchCodecVideoReader(
                resolved_path,
                num_threads=self.decoder_threads,
                device=self.device,
            )
            total_frames = len(reader)
            if total_frames < 1:
                raise ValueError(f"Framework evaluation video has zero frames: {resolved_path}")
            frame_count = min(self.num_frames, total_frames)
            indices = (
                [0]
                if frame_count == 1
                else torch.linspace(0, total_frames - 1, frame_count).round().to(dtype=torch.long).tolist()
            )
            frames_np = reader.get_frames_nhwc_uint8(indices)
            actual_device = str(reader.last_output_device)
            requested_device = torch.device(self.device)
            decoded_device = torch.device(actual_device)
            if requested_device.type == "cuda" and (
                decoded_device.type != "cuda"
                or (requested_device.index is not None and requested_device.index != decoded_device.index)
            ):
                raise RuntimeError(
                    f"Framework evaluation TorchCodec device mismatch: requested={self.device} actual={actual_device}"
                )

            frames = [Image.fromarray(frame) for frame in frames_np]
            source_fps = float(reader.get_avg_fps())
            average_stride = (indices[-1] - indices[0]) / max(len(indices) - 1, 1) if len(indices) > 1 else 1.0
            decoded = (frames, source_fps / max(average_stride, 1.0))
            with self._lock:
                if not self._attested:
                    print(
                        "COSMOS_FRAMEWORK_EVALUATION_VIDEO_RUNTIME "
                        f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                        "backend=torchcodec "
                        f"requested_device={self.requested_device} "
                        f"resolved_device={self.device} actual_device={actual_device} "
                        f"video_cache_size={self.cache_size} "
                        f"process_threads={self.process_threads} "
                        f"decoder_threads={self.decoder_threads} "
                        f"dataloader_workers={self.dataloader_num_workers} "
                        f"prefetch_factor={self.dataloader_prefetch_factor} "
                        f"multiprocessing_context={self.dataloader_multiprocessing_context} "
                        f"persistent_workers={self.dataloader_persistent_workers}",
                        flush=True,
                    )
                    self._attested = True
                if self.cache_size > 0:
                    self._cache[resolved_path] = decoded
                    self._cache.move_to_end(resolved_path)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
            return decoded
        finally:
            if self.cache_size > 0:
                with self._lock:
                    completed = self._inflight.pop(resolved_path, None)
                    if completed is not None:
                        completed.set()

    def prepare_task(self, task: dict[str, Any]) -> dict[str, Any]:
        prepared = copy.copy(task)
        if task.get("media_mode", "image") != "video":
            return prepared
        prepared["_framework_decoded_media"] = [
            {"frames": frames, "fps": fps}
            for frames, fps in (self._decode(path) for path in task.get("media_paths", []))
        ]
        return prepared

    def prepare_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(tasks) < 2 or self.process_threads == 1:
            return [self.prepare_task(task) for task in tasks]
        workers = min(self.process_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map preserves input order while keeping the pool bounded.
            return list(executor.map(self.prepare_task, tasks))

    @staticmethod
    def _media_key(task: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(task.get("media_mode", "image")),
            tuple(str(path) for path in task.get("media_paths", [])),
        )

    def _iter_media_lookahead_singletons(self, tasks: list[dict[str, Any]]):
        """Decode the next media group while the model consumes this group.

        Media-balanced evaluation keeps every question for one video adjacent.
        A task-by-task one-deep prefetch therefore spends nearly all of its
        lookahead on cache hits and starts the next cold decode only during the
        final forward for the current video.  With two in-memory decoded-video
        entries, this iterator starts the next distinct media group before it
        yields the first task in the current group.  The model's repeated
        forwards then hide that cold decode without changing task order,
        sampling, frame bytes, or the on-demand/no-disk-cache contract.
        """
        groups: list[tuple[int, int]] = []
        start = 0
        while start < len(tasks):
            key = self._media_key(tasks[start])
            end = start + 1
            while end < len(tasks) and self._media_key(tasks[end]) == key:
                end += 1
            groups.append((start, end))
            start = end

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cosmos-framework-media-lookahead",
        ) as executor:
            current_future = executor.submit(self.prepare_task, tasks[groups[0][0]])
            for group_index, (group_start, group_end) in enumerate(groups):
                first_prepared = current_future.result()
                if group_index + 1 < len(groups):
                    next_start = groups[group_index + 1][0]
                    current_future = executor.submit(self.prepare_task, tasks[next_start])
                if not self._media_lookahead_attested:
                    print(
                        "COSMOS_FRAMEWORK_MEDIA_LOOKAHEAD_ATTESTATION "
                        f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                        "mode=next-distinct-media depth=1 "
                        f"decoded_video_cache_size={self.cache_size} "
                        f"media_groups={len(groups)} source_order_preserved=true",
                        flush=True,
                    )
                    self._media_lookahead_attested = True
                yield [first_prepared]
                for task_index in range(group_start + 1, group_end):
                    yield [self.prepare_task(tasks[task_index])]

    def iter_prepared_batches(
        self,
        tasks: list[dict[str, Any]],
        *,
        batch_size: int,
    ):
        """Yield ordered batches from either direct or spawned-worker preprocessing."""
        if self.dataloader_num_workers == 0:
            if batch_size == 1 and self.cache_size >= 2 and tasks:
                yield from self._iter_media_lookahead_singletons(tasks)
                return
            batches = (tasks[index : index + batch_size] for index in range(0, len(tasks), batch_size))
            try:
                first_batch = next(batches)
            except StopIteration:
                return
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cosmos-framework-video-prefetch",
            ) as executor:
                future = executor.submit(self.prepare_tasks, first_batch)
                if not self._direct_prefetch_attested:
                    print(
                        "COSMOS_FRAMEWORK_DIRECT_PREFETCH_ATTESTATION "
                        f"rank={os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))} "
                        "mode=thread depth=1 dataloader_workers=0",
                        flush=True,
                    )
                    self._direct_prefetch_attested = True
                for next_batch in batches:
                    prepared = future.result()
                    future = executor.submit(self.prepare_tasks, next_batch)
                    yield prepared
                yield future.result()
            return

        from torch.utils.data import DataLoader

        loader = DataLoader(
            tasks,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=self.prepare_tasks,
            prefetch_factor=self.dataloader_prefetch_factor,
            multiprocessing_context=self.dataloader_multiprocessing_context,
            persistent_workers=self.dataloader_persistent_workers,
            drop_last=False,
        )
        yield from loader
