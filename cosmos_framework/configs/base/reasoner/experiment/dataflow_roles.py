# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM dataflow roles (RawItemProcessor + BatchCollator) extracted 1:1 from
VLMDataPacker (llava_ov_vlm.py). Behavior-preserving."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from cosmos_framework.data.generator.dataflow.base import BatchCollator, RawItemProcessor
from cosmos_framework.data.generator.local_datasets.reasoning_qa import apply_reasoning_chat_template
from cosmos_framework.utils.generator.torchcodec_video import TorchCodecVideoReader
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class VLMProcessor(RawItemProcessor):
    """ShareGPT image+conversation record -> VLM training tensors."""

    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        self._processor = processor
        self._ignore_index = ignore_index
        # Resolve pad token id once; VLMCollator uses it to right-pad input_ids.
        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "eos_token_id", None)
        if pad_id is None:
            raise ValueError(
                "VLMProcessor: tokenizer exposes neither pad_token_id nor "
                "eos_token_id; cannot determine a padding id for VLMCollator. "
                "Configure the tokenizer's pad/eos token."
            )
        self._pad_token_id = int(pad_id)

    @staticmethod
    def _decode_image(image: Any) -> Any:
        """Decode a HuggingFace streaming image to PIL.

        In streaming mode HuggingFace delivers images as
        ``{"bytes": bytes, "path": str}`` dicts rather than decoded PIL Images.
        """
        if isinstance(image, dict):
            import io

            from PIL import Image

            raw = image.get("bytes")
            if raw:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            path = image.get("path")
            if path:
                return Image.open(path).convert("RGB")
            return None
        return image

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        """Convert ShareGPT conversation to OpenAI message format.

        LLaVA-OneVision-Data records use ``from``/``value`` pairs where the
        human turn may contain a ``<image>`` placeholder.  We strip the
        placeholder and attach the PIL image as a separate content block.
        """
        conversations = item.get("conversations", [])
        image = self._decode_image(item.get("image"))  # PIL.Image or None
        messages: list[dict] = []
        image_inserted = False

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"].replace("<image>", "").strip()

            if role == "user" and not image_inserted and image is not None:
                content: Any = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ]
                image_inserted = True
            else:
                content = text

            messages.append({"role": role, "content": content})

        return messages

    def process(self, item: dict) -> dict:
        messages = self._sharegpt_to_openai(item)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": token_mask,
            "pad_token_id": self._pad_token_id,
            "ignore_index": self._ignore_index,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result


class VLMCollator(BatchCollator):
    """Pad-and-stack collation for any batch size: right-pads sequence tensors to
    a multiple of 16, flat-concatenates vision tensors on dim 0, and stamps resume
    meta (zeros — streaming source has no position)."""

    def collate(self, samples: list[dict]) -> dict:
        # Parity with i4 custom_collate: skip if already collated.
        if samples and samples[0].get("collated"):
            return samples[0]

        # All four sequence tensors must be present and 1-D on every sample
        # before padding/stacking (matches i4 custom_collate). A missing key
        # here would otherwise fall through to default_collate as a ragged list.
        for key in ("input_ids", "token_mask", "attention_mask", "labels"):
            assert all(key in s and s[key].ndim == 1 for s in samples), (
                f"VLMCollator: {key} must be present and 1-D on every sample"
            )

        # Right-pad target length, rounded up to a multiple of 16 (FP8 support).
        max_seq_length = max(s["input_ids"].shape[0] for s in samples)
        max_seq_length = (max_seq_length + 15) // 16 * 16

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        batch_size = len(samples)

        regular: dict = {}
        special: dict = {}

        def _pad_stack(key: str, fill, dtype) -> torch.Tensor:
            rows = []
            for s in samples:
                t = s[key]
                pad = torch.full((max_seq_length - t.shape[0],), fill, dtype=dtype)
                rows.append(torch.cat([t, pad]))
            return torch.stack(rows, dim=0)

        # input_ids: pad with each sample's pad_token_id.
        regular["input_ids"] = torch.stack(
            [
                torch.cat([
                    s["input_ids"],
                    torch.full((max_seq_length - s["input_ids"].shape[0],),
                               s["pad_token_id"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # token_mask / attention_mask: pad with False (guaranteed present by the
        # assertion above).
        for key in ("token_mask", "attention_mask"):
            regular[key] = _pad_stack(key, False, torch.bool)

        # labels: pad with each sample's ignore_index.
        regular["labels"] = torch.stack(
            [
                torch.cat([
                    s["labels"],
                    torch.full((max_seq_length - s["labels"].shape[0],),
                               s["ignore_index"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # raw_image / raw_video: keep per-sample, per-item boundaries (parity).
        if any("raw_image" in s for s in samples):
            ri: list = []
            for s in samples:
                img = s.get("raw_image", [])
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img = img[:, None]
                    img = [img[:, i:i + 1] for i in range(img.shape[1])]
                ri.append(img)
            regular["raw_image"] = ri
        if any("raw_video" in s for s in samples):
            rv: list = []
            for s in samples:
                vid = s.get("raw_video", [])
                if isinstance(vid, torch.Tensor):
                    vid = [vid]
                rv.append(vid)
            regular["raw_video"] = rv

        # Vision tensors: flat-concatenate on dim 0 (Qwen3-VL addresses them via
        # placeholder tokens in input_ids, not by batch position).
        vision_cat_keys = (
            "image_grid_thw", "video_grid_thw", "second_per_grid_ts",
            "pixel_values", "pixel_values_videos", "image_sizes",
        )
        all_keys = {k for s in samples for k in s}
        for key in all_keys:
            if key in regular:
                continue
            if key in vision_cat_keys:
                special[key] = torch.cat([s[key] for s in samples if key in s], dim=0)
            else:
                regular[key] = default_collate([s[key] for s in samples])

        batch = {**regular, **special, "collated": True}
        # Resume meta (streaming source has no position -> zeros), length-B.
        batch["sample_worker_id"] = torch.tensor([worker_id] * batch_size)
        batch["sample_epoch"] = torch.tensor([0] * batch_size)
        batch["sample_index"] = torch.tensor([0] * batch_size)
        return batch


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
                            "COSMOS_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_HIT_ATTESTATION "
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
        use_reasoning_chat_template: bool = False,
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
                raise RuntimeError("processed video caching requires processor.video_processor")
            hf_processor.video_processor = _ProcessedVideoCacheProxy(
                video_processor,
                processed_video_cache_size,
            )
            print(
                "COSMOS_FRAMEWORK_VALIDATION_PROCESSED_VIDEO_CACHE_ENABLED_ATTESTATION "
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
        self.use_reasoning_chat_template = use_reasoning_chat_template
        if self.use_reasoning_chat_template:
            apply_reasoning_chat_template(processor)
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
                        "COSMOS_FRAMEWORK_VIDEO_RUNTIME "
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
            sample["cosmos_video_cache_key"] = os.path.realpath(os.path.abspath(os.path.expanduser(video_path)))
        return sample


class VideoVLMCollator(VLMCollator):
    """Preserve one stable video identity per sample for validation caching."""

    def collate(self, samples: list[dict]) -> dict:
        cache_keys = [sample.get("cosmos_video_cache_key") for sample in samples]
        batch = super().collate(samples)
        batch.pop("cosmos_video_cache_key", None)
        if all(isinstance(key, str) for key in cache_keys):
            batch["cosmos_video_cache_keys"] = cache_keys
        return batch
