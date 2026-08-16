# SPDX-License-Identifier: OpenMDW-1.1
"""Benchmark standins over base Cosmos loaders.

Subclasses genuine Cosmos loaders to measure performance in storage regimes
not natively supported by the base classes.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
from transformers import AutoTokenizer

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    _load_sft_metadata_from_s3,
)

_QWEN_TOKENIZER = "Qwen/Qwen2.5-7B"


@contextmanager
def hf_online_preserved():
    """Constructing the action base flips HF Hub offline process-wide (env + constant);
    restore both so HF-dependent loaders (tokenizers, streaming) keep working."""
    import huggingface_hub.constants as hfc

    prev_const, prev_env = hfc.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")
    try:
        yield
    finally:
        hfc.HF_HUB_OFFLINE = prev_const
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env


class S3DROIDLeRobotDataset(DROIDLeRobotDataset):
    """DROIDLeRobotDataset that materializes the S3-hosted videos, then runs the genuine base.

    Builds a shadow root (same versioned dir name, so the base's version registry
    resolves): metadata/labels are symlinked from the local tree, the mega-mp4s
    under ``videos/`` are downloaded from ``s3://{bucket}/{prefix}/videos/...``.
    """

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
        src = Path(root)
        key = s3_prefix.strip("/").replace("/", "_")
        cache = Path(cache_dir or os.path.join(tempfile.gettempdir(), "_s3base_droid", key)) / src.name
        success = cache / "success"
        success.mkdir(parents=True, exist_ok=True)
        for sub in ("meta", "data"):
            link = success / sub
            if not (link.exists() or link.is_symlink()):
                link.symlink_to((src / "success" / sub).resolve())

        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        pref = s3_prefix.strip("/") + "/videos/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=pref):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(pref) - len("videos/") :]  # keep the videos/ prefix
                dst = success / rel
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = str(dst) + f".part{os.getpid()}"
                s3.download_file(s3_bucket, obj["Key"], tmp)
                os.replace(tmp, dst)

        super().__init__(root=str(cache), **kwargs)


def _qwen_tokenizer_config():
    return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(_QWEN_TOKENIZER))


def load_sft_metadata(
    jsonl_path: str, *, s3_bucket: str | None = None, s3_prefix: str | None = None, min_frames: int = 61
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
        skip_tokenize: bool = False,
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
            temporal_compression_factor=temporal_compression_factor,
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
        cls, jsonl_path: str, *, s3_bucket: str | None = None, s3_prefix: str | None = None, **kw
    ) -> "BenchSFTDataset":
        return cls(load_sft_metadata(jsonl_path, s3_bucket=s3_bucket, s3_prefix=s3_prefix), **kw)


__all__ = ["S3DROIDLeRobotDataset", "BenchSFTDataset", "load_sft_metadata", "hf_online_preserved"]
