# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DAFT-backed ``tao-vl-reason-v1.0`` dataset adapter for Framework SFT."""

from __future__ import annotations

import json
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


class TaoVlReasonDaftDataset(Dataset):
    """Expose DAFT conversations through Cosmos Framework's map dataflow.

    The index mapping intentionally matches the internal Cosmos-RL DAFT hook:
    hybrid mode interleaves answer and reasoning targets for every raw item.
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
        try:
            from nvidia_tao_daft.datasets.tao_vl_reason_v1_0 import (
                TaoVlReasonV1_0CosmosRLConversationDataset,
            )
        except ImportError as exc:
            raise ImportError("nvidia-tao-daft>=2.9.1 is required for the AETC DAFT recipe") from exc

        paths = parse_path_list(annotation_paths)
        if isinstance(media_root, str) and media_root.strip().startswith("["):
            media_root = parse_path_list(media_root)
        if response_mode not in {"think", "answer", "hybrid"}:
            raise ValueError(f"unsupported response_mode: {response_mode!r}")
        if sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")

        self.dataset = TaoVlReasonV1_0CosmosRLConversationDataset(
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
            daft_index = raw_index if index % 2 == 0 else self.raw_length + raw_index
        else:
            daft_index = self.sample_offset + index * self.sample_stride
        return {"messages": self.dataset[daft_index]}


def apply_daft_chat_template(processor: Any) -> None:
    """Apply DAFT's Qwen3-VL Instruct template to a Framework processor."""
    try:
        from nvidia_tao_daft.datasets.tao_vl_reason_v1_0 import (
            apply_chat_template_override,
        )
    except ImportError as exc:
        raise ImportError("nvidia-tao-daft>=2.9.1 is required for the AETC DAFT recipe") from exc

    huggingface_processor = getattr(processor, "processor", processor)
    apply_chat_template_override(huggingface_processor)
