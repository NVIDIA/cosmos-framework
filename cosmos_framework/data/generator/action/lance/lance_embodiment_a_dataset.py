# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec_name
from cosmos_framework.data.generator.action.hand_pose_dataset_config import WRIST_FRAME_ALIGN_embodiment_a
from cosmos_framework.data.generator.action.lance.lance_vitra_ego4d_dataset import LanceVITRAEgo4DDataset
from cosmos_framework.data.generator.lance.lance_action_source import LanceActionDatasetSpec

_SOURCE_FAMILY = "embodiment_a"


_embodiment_a_METADATA_BLOCK_SIZE: int = 4


_embodiment_a_VIDEO_DECODE_THREADS: int = 1


class Lanceembodiment_aDataset(LanceVITRAEgo4DDataset):
    """Apply embodiment_a wrist-frame semantics to one canonical ACT2 collection."""

    def __init__(
        self,
        lance_dataset_spec: LanceActionDatasetSpec,
        creds_path: str | Path,
        chunk_length: int = 72,
        source_dataset_name: str | None = None,
        metadata_block_size: int = _embodiment_a_METADATA_BLOCK_SIZE,
        video_decode_threads: int = _embodiment_a_VIDEO_DECODE_THREADS,
        **kwargs: Any,
    ) -> None:
        resolved_source_dataset_name = self._resolve_source_dataset_name(lance_dataset_spec, source_dataset_name)
        if get_action_binary_spec_name(resolved_source_dataset_name) != _SOURCE_FAMILY:
            raise ValueError(f"Unsupported embodiment_a source_dataset_name={resolved_source_dataset_name!r}.")

        super().__init__(
            lance_dataset_spec=lance_dataset_spec,
            creds_path=creds_path,
            chunk_length=chunk_length,
            source_dataset_name=resolved_source_dataset_name,
            metadata_block_size=metadata_block_size,
            video_decode_threads=video_decode_threads,
            **kwargs,
        )
        self.wrist_frame_alignment = WRIST_FRAME_ALIGN_embodiment_a
