# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Task-aware video reasoning dataset adapter for Framework SFT."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from torch.utils.data import Dataset

ResponseMode = Literal["think", "answer", "hybrid"]


def parse_path_list(value: str | list[str]) -> list[str]:
    """Resolve a path or JSON-encoded path list supplied through Hydra/env."""
    if isinstance(value, list):
        paths = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            paths = []
        elif stripped.startswith("["):
            paths = json.loads(stripped)
        else:
            paths = [stripped]
    else:
        raise TypeError(f"annotation_paths must be a string or list, got {type(value).__name__}")
    if not paths or not all(isinstance(path, str) and path.strip() for path in paths):
        raise ValueError("annotation_paths must contain at least one non-empty path")
    return [path.strip() for path in paths]


def _optional_positive_int(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


class ReasoningQADataset(Dataset):
    """Expose task-aware conversations through Cosmos Framework's map dataflow.

    Hybrid mode interleaves answer and reasoning targets for every raw item.
    """

    def __init__(
        self,
        annotation_paths: str | list[str],
        media_root: str | list[str] | None = None,
        response_mode: ResponseMode = "answer",
        system_prompt: str = "",
        vision_kwargs: dict[str, Any] | None = None,
        max_samples: int | str | None = None,
        sample_stride: int = 1,
        sample_offset: int = 0,
    ) -> None:
        from cosmos_framework.data.reasoner.qa_dataset import ReasoningConversationDataset

        paths = parse_path_list(annotation_paths)
        if isinstance(media_root, str) and media_root.strip().startswith("["):
            media_root = parse_path_list(media_root)
        if response_mode not in {"think", "answer", "hybrid"}:
            raise ValueError(f"unsupported response_mode: {response_mode!r}")
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")

        self.dataset = ReasoningConversationDataset(
            annotation_paths=paths,
            media_roots=media_root,
            system_prompt=system_prompt,
            vision_kwargs=vision_kwargs or {},
            response_mode=response_mode,
        )
        self.response_mode = response_mode
        self.sample_stride = sample_stride
        self.sample_offset = sample_offset
        self.max_samples = _optional_positive_int(max_samples)
        self.raw_length = int(getattr(self.dataset, "_raw_length"))

    def _subsampled_raw_length(self) -> int:
        if self.sample_offset >= self.raw_length:
            return 0
        return ((self.raw_length - self.sample_offset - 1) // self.sample_stride) + 1

    def __len__(self) -> int:
        length = self._subsampled_raw_length()
        if self.response_mode == "hybrid":
            length *= 2
        if self.max_samples is not None:
            length = min(length, self.max_samples)
        return length

    def __getitem__(self, index: int) -> dict[str, list[dict[str, Any]]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.response_mode == "hybrid":
            raw_index = self.sample_offset + (index // 2) * self.sample_stride
            conversation_index = raw_index if index % 2 == 0 else self.raw_length + raw_index
        else:
            conversation_index = self.sample_offset + index * self.sample_stride
        return {"messages": self.dataset[conversation_index]}


def apply_reasoning_chat_template(processor: Any) -> None:
    """Apply the Qwen3-VL Instruct template to a Framework processor."""
    from cosmos_framework.data.reasoner.qa_dataset import apply_chat_template_override

    huggingface_processor = getattr(processor, "processor", processor)
    apply_chat_template_override(huggingface_processor)


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
