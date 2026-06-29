# SPDX-License-Identifier: OpenMDW-1.1
"""Benchmark standins over base Cosmos loaders.

Subclasses genuine Cosmos loaders to measure performance in storage regimes
not natively supported by the base classes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
from transformers import AutoTokenizer

from cosmos_framework.data.vfm.action.datasets.base_dataset import _MODE_CHOICES  # noqa: F401
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import (
    _IMAGE_FEATURES,
    DROIDLeRobotDataset,
)
from cosmos_framework.data.vfm.local_datasets.sft_dataset import (
    SFTDataset,
    _load_sft_metadata_from_s3,
)

_QWEN_TOKENIZER = "Qwen/Qwen2.5-7B"


class S3DROIDLeRobotDataset(DROIDLeRobotDataset):
    """DROIDLeRobotDataset that materializes mega-mp4s from S3 to local cache."""

    def __init__(
        self,
        root: str,
        s3_bucket: str,
        s3_prefix: str,
        *,
        region: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, **kwargs)
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix.strip("/")
        self._region = region
        key = self._s3_prefix.replace("/", "_")
        self._cache_root = Path(cache_dir or os.path.join(tempfile.gettempdir(), "_s3base_droid", key))
        self._materialize_from_s3()

    def _rel_for(self, episode: dict[str, Any], video_key: str) -> str:
        ci = int(episode.get(
            f"videos/{video_key}/chunk_index",
            episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0))
        ))
        fi = int(episode.get(
            f"videos/{video_key}/file_index",
            episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0))
        ))
        return self._info["video_path"].format(
            video_key=video_key,
            chunk_index=ci,
            file_index=fi,
            episode_chunk=ci,
            episode_file=fi
        )

    def _materialize_from_s3(self) -> None:
        rels = set()
        for episode in self._episodes.values():
            for video_key in _IMAGE_FEATURES.values():
                rels.add(self._rel_for(episode, video_key))

        if self._region:
            s3 = boto3.client("s3", region_name=self._region)
        else:
            s3 = boto3.client("s3")

        for rel in sorted(rels):
            dst = self._cache_root / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(
                self._s3_bucket,
                f"{self._s3_prefix}/{rel}",
                str(dst.with_suffix(dst.suffix + f".part{os.getpid()}"))
            )
            os.replace(dst.with_suffix(dst.suffix + f".part{os.getpid()}"), dst)

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        return self._cache_root / self._rel_for(episode, video_key)


def _qwen_tokenizer_config():
    return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(_QWEN_TOKENIZER))


def load_sft_metadata(
    jsonl_path: str,
    *,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    min_frames: int = 61
) -> list[dict]:
    meta = _load_sft_metadata_from_s3(None, jsonl_path, min_frames=min_frames)
    if s3_bucket and s3_prefix:
        base_dir = os.path.dirname(os.path.abspath(jsonl_path))
        pref = s3_prefix.strip("/")
        for m in meta:
            vp = m["vision_path"]
            if os.path.isabs(vp) or os.path.exists(vp):
                rel = os.path.relpath(vp, base_dir)
            else:
                rel = vp
            m["vision_path"] = f"s3://{s3_bucket}/{pref}/{rel}"
    return meta


class BenchSFTDataset(SFTDataset):
    """SFTDataset driver for throughput benchmarks."""
    def __init__(
        self,
        metadata: list[dict],
        *,
        num_video_frames: int = 16,
        resolution: str = "256",
        temporal_interval_mode: str = "entire_chunk",
        frame_selection_mode: str = "first",
        temporal_compression_factor: int = 4,
        skip_tokenize: bool = False
    ) -> None:
        super().__init__(
            metadata=metadata,
            num_video_frames=num_video_frames,
            resolution=resolution,
            s3_credentials={},
            temporal_interval_mode=temporal_interval_mode,
            frame_selection_mode=frame_selection_mode,
            tokenizer_config=_qwen_tokenizer_config(),
            cfg_dropout_rate=0.0,
            temporal_compression_factor=temporal_compression_factor
        )
        self.skip_tokenize = bool(skip_tokenize)
        self.shard_world_size = 1
        self.shard_rank = 0
        self.shard_id = 0

    def _tokenize_caption(self, caption: str):
        if self.skip_tokenize:
            return ([], caption)
        return super()._tokenize_caption(caption)

    def __iter__(self):
        if not hasattr(self, "_meta0"):
            self._meta0 = list(self.metadata)
        self.metadata = list(self._meta0)
        self.is_initialized = False
        return super().__iter__()

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str,
        *,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        **kw
    ) -> "BenchSFTDataset":
        return cls(load_sft_metadata(jsonl_path, s3_bucket=s3_bucket, s3_prefix=s3_prefix), **kw)


__all__ = ["S3DROIDLeRobotDataset", "BenchSFTDataset", "load_sft_metadata"]
