# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sample a population-weighted Transfer length index using one worker pool."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from typing import Any, ClassVar

from torch.utils.data import DataLoader, Dataset, Sampler

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader, custom_collate_fn
from cosmos_framework.data.generator.lance.augment_timing import iter_augmented_with_timing
from cosmos_framework.data.generator.lance.sample_rng import LANCE_SAMPLE_RNG_SEED_KEY
from cosmos_framework.data.generator.transfer_length_index_cache import resolve_transfer_length_manifest



def _draw_seed(seed: int, rank: int, fetch_count: int, attempt: int) -> int:
    """Use a stable, worker-topology-independent seed for a rank's indexed draw."""
    identity = f"transfer-length-v1:{seed}:{rank}:{fetch_count}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")




class IndexedTransferDatasetBase(Dataset[dict[str, Any]]):
    """Map a global fetch count to one shared bucket and one rank-local video.

    The immutable index has one row per unique video, assigned to exactly one
    latent length. Sampling buckets proportional to their populations, followed
    by uniform sampling within a bucket, gives each eligible indexed video equal
    weight. A minimum frame count excludes shorter buckets before scheduling.
    Every retry remains inside the scheduled bucket and yields one sample for
    the requested fetch count, so media failures cannot desynchronize ranks.
    """

    index_columns: ClassVar[set[str]] = {"bucket", "slot", "video_hash"}

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer_config: Any,
        *,
        index_creds_path: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
        caption_key: str = "canonical_caption_structured",
        conditioning_config: dict[int, float] | None = None,
        resolution: str = "480",
        seed: int = 42,
        max_attempts: int = 100,
        min_num_frames: int = 1,
        cfg_dropout_rate: float = 0.0,
        dense_caption_probability: float = 0.0,
        resize_on_read: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path: Path = resolve_transfer_length_manifest(
            manifest_path, creds_path=index_creds_path, cache_dir=index_cache_dir
        )
        with self.manifest_path.open() as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported Transfer length-index format_version.")
        coverage = self.manifest.get("coverage", {})
        if (
            coverage.get("complete") is not True
            or int(coverage.get("scanned_source_rows", 0)) < 1
            or coverage.get("scanned_source_rows") != coverage.get("expected_source_rows")
        ):
            raise ValueError("The Transfer length index must describe a completed full source scan.")
        counts = self.manifest["bucket_counts"]
        if set(counts) != {str(bucket) for bucket in range(1, 52)}:
            raise ValueError("The Transfer index must describe all 51 latent-frame buckets.")
        self.bucket_counts: tuple[int, ...] = tuple(int(counts[str(bucket)]) for bucket in range(1, 52))
        self.total_videos: int = int(self.manifest["total_videos"])
        if any(count < 0 for count in self.bucket_counts) or sum(self.bucket_counts) != self.total_videos:
            raise ValueError("Transfer bucket counts do not match total_videos.")
        if self.total_videos < 1:
            raise ValueError("The Transfer length index contains no eligible videos.")
        self.policy: dict[str, Any] = self.manifest["policy"]
        if self.policy["max_latent_frames"] != 51 or self.policy["temporal_compression_factor"] != 4:
            raise ValueError("This Transfer loader requires latent lengths 1..51 and temporal compression 4.")
        if not 1 <= min_num_frames <= 201:
            raise ValueError("min_num_frames must lie between 1 and 201 RGB frames.")
        if not 0.0 <= cfg_dropout_rate <= 1.0 or not 0.0 <= dense_caption_probability <= 1.0:
            raise ValueError("Caption dropout and dense-caption probabilities must lie between 0 and 1.")
        self.min_num_frames: int = min_num_frames
        self.cfg_dropout_rate: float = cfg_dropout_rate
        self.dense_caption_probability: float = dense_caption_probability
        self.resize_on_read: bool = resize_on_read
        # Mask only the scheduled population. Keep the immutable index counts
        # unchanged so row-slot selection still addresses the original buckets.
        self.sampling_bucket_counts: tuple[int, ...] = tuple(
            count if 4 * (bucket - 1) + 1 >= min_num_frames else 0
            for bucket, count in enumerate(self.bucket_counts, start=1)
        )
        self.eligible_videos: int = sum(self.sampling_bucket_counts)
        if self.eligible_videos == 0:
            raise ValueError("The Transfer length index has no videos meeting min_num_frames.")
        if seed < 0 or max_attempts < 1:
            raise ValueError("max_attempts must be positive, and seed must be nonnegative.")
        self.index_path: Path = self.manifest_path.parent / self.manifest["index_file"]
        if not self.index_path.is_file():
            raise ValueError(f"The Transfer length index does not exist: {self.index_path}")
        self.seed: int = seed
        self.max_attempts: int = max_attempts
        # Resolve rank in the parent; spawned workers need no process group.
        from cosmos_framework.data.generator.worker_rank import get_worker_rank_world_size

        self.rank: int = get_worker_rank_world_size()[0]
        self.resolution: str = resolution
        self.caption_key: str = caption_key
        self.tokenizer_config: Any = tokenizer_config
        self.conditioning_config: dict[int, float] = dict(conditioning_config or {0: 0.7, 1: 0.2, 2: 0.1})
        self._connection: sqlite3.Connection | None = None
        self._augmentors: dict[int, list[Any]] = {}
        self._shared_augmentors: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.eligible_videos

    def bucket_for_fetch(self, fetch_count: int) -> int:
        """Choose the same population-weighted bucket on every training rank."""
        if fetch_count < 0:
            raise ValueError("fetch_count must be nonnegative.")
        return random.Random(self.seed + fetch_count).choices(range(1, 52), weights=self.sampling_bucket_counts, k=1)[0]

    def _entry_for_draw(self, bucket: int, draw_seed: int) -> dict[str, Any]:
        if self._connection is None:
            self._connection = sqlite3.connect(f"{self.index_path.as_uri()}?mode=ro&immutable=1", uri=True)
            self._connection.row_factory = sqlite3.Row
            columns = {row["name"]: row["pk"] for row in self._connection.execute("PRAGMA table_info(videos)")}
            required = self.index_columns
            if not required.issubset(columns) or columns.get("bucket") != 1 or columns.get("slot") != 2:
                raise ValueError("The Transfer index must have the expected columns and primary key (bucket, slot).")
        slot = random.Random(draw_seed).randrange(self.bucket_counts[bucket - 1])
        row = self._connection.execute("SELECT * FROM videos WHERE bucket=? AND slot=?", (bucket, slot)).fetchone()
        if row is None:
            raise ValueError(f"Transfer length index is missing bucket={bucket}, slot={slot}.")
        return dict(row)

    def _load_entry(self, entry: dict[str, Any], draw_seed: int) -> dict[str, Any] | None:
        """Read the immutable source selected by one index row."""
        raise NotImplementedError

    def _augmentors_for_bucket(self, bucket: int) -> list[Any]:
        if bucket not in self._augmentors:
            from cosmos_framework.data.generator.transfer_augmentors import (
                get_video_transfer_chunked_frames_augmentor,
            )

            num_frames = 4 * (bucket - 1) + 1
            conditioning = {count: weight for count, weight in self.conditioning_config.items() if count < bucket}
            if not conditioning or sum(conditioning.values()) <= 0:
                raise ValueError(f"No conditioning choice leaves a generated frame in bucket {bucket}.")
            total_weight = sum(conditioning.values())
            conditioning = {count: weight / total_weight for count, weight in conditioning.items()}
            configs = get_video_transfer_chunked_frames_augmentor(
                resolution=self.resolution,
                min_num_frames=self.min_num_frames,
                max_num_frames=num_frames,
                target_num_frames=num_frames,
                teacher_forcing_frames_per_chunk=1,
                min_fps=self.policy["min_fps"],
                max_fps=self.policy["max_fps"],
                min_stride=self.policy["min_stride"],
                max_stride=self.policy["max_stride"],
                caption_config={
                    self.caption_key: {"ratio": 1.0, "dense_caption_probability": self.dense_caption_probability}
                },
                tokenizer_config=self.tokenizer_config,
                conditioning_config=conditioning,
                selected_control=True,
                append_duration_fps_timestamps=True,
                append_resolution_info=True,
                cfg_dropout_rate=self.cfg_dropout_rate,
                resize_on_read=self.resize_on_read,
            )
            per_bucket_keys = {
                "text_transform",
                "video_parser_with_controlled",
                "transfer_to_training_format",
                "duration_fps_timestamps",
            }
            augmentors: list[Any] = []
            for name, config in configs.items():
                if name in per_bucket_keys:
                    augmentors.append(instantiate(config))
                else:
                    if name not in self._shared_augmentors:
                        self._shared_augmentors[name] = instantiate(config)
                    augmentors.append(self._shared_augmentors[name])
            self._augmentors[bucket] = augmentors
        return self._augmentors[bucket]

    def _augment_sample(self, sample: dict[str, Any], bucket: int, draw_seed: int) -> dict[str, Any] | None:
        sample[LANCE_SAMPLE_RNG_SEED_KEY] = draw_seed
        return next(iter_augmented_with_timing([sample], self._augmentors_for_bucket(bucket)), None)

    def __getitem__(self, fetch_count: int) -> dict[str, Any]:
        bucket = self.bucket_for_fetch(fetch_count)
        for attempt in range(self.max_attempts):
            draw_seed = _draw_seed(self.seed, self.rank, fetch_count, attempt)
            entry = self._entry_for_draw(bucket, draw_seed)
            sample = self._load_entry(entry, draw_seed)
            if sample is None:
                continue
            sample = self._augment_sample(sample, bucket, draw_seed)
            if sample is not None:
                sample["transfer_length_bucket"] = bucket
                sample["transfer_video_hash"] = entry["video_hash"]
                sample["transfer_fetch_count"] = fetch_count
                return sample
        raise RuntimeError(
            f"Transfer bucket {bucket} failed {self.max_attempts} indexed attempts at fetch {fetch_count}, "
            f"rank {self.rank}. Check media integrity and index/parser policy agreement."
        )




class _FetchCountSampler(Sampler[int]):
    """An infinite request sequence whose cursor is set before worker prefetch."""

    def __init__(self) -> None:
        self.start: int = 0

    def __iter__(self) -> Iterator[int]:
        return itertools.count(self.start)

    def __len__(self) -> int:
        return 2**63 - 1


class IndexedTransferLengthDataLoader(IterativeJointDataLoader):
    """Pack one indexed sample per fetch using a single resumable worker pool."""

    def __init__(
        self,
        dataset: IndexedTransferDatasetBase,
        tokenizer_spatial_compression_factor: int,
        tokenizer_temporal_compression_factor: int,
        patch_spatial: int,
        *,
        num_workers: int = 1,
        prefetch_factor: int = 1,
        pin_memory: bool = True,
    ) -> None:
        self.fetch_sampler: _FetchCountSampler = _FetchCountSampler()
        loader_kwargs: dict[str, Any] = {}
        if num_workers:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        inner = DataLoader(
            dataset,
            batch_size=1,
            sampler=self.fetch_sampler,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            persistent_workers=False,
            pin_memory=pin_memory,
            **loader_kwargs,
        )
        super().__init__(
            dataloaders={f"video_transfer_4modality_{dataset.resolution}_indexed": {"dataloader": inner, "ratio": 1}},
            tokenizer_spatial_compression_factor=tokenizer_spatial_compression_factor,
            tokenizer_temporal_compression_factor=tokenizer_temporal_compression_factor,
            patch_spatial=patch_spatial,
            max_samples_per_batch=1,
            prewarm=False,
            lazy_initialize_child_iterators=True,
        )

    def set_start_iteration(self, iteration: int) -> None:
        """Restore the consumed fetch count before any sampler request is issued."""
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        if self._child_iterators_initialized:
            raise RuntimeError("Set the indexed Transfer resume iteration before iterating the dataloader.")
        self.fetch_sampler.start = iteration
        super().set_start_iteration(iteration)

    def _update_output_batch(self, output_batch: dict[str, Any], output: dict[str, Any]) -> None:
        """Include the selected latent length in normal per-dataset training logs."""
        bucket = int(output["transfer_length_bucket"].item())
        output["dataset_name"] = f"{self.dataset_name_list[0]}_{bucket}l"
        super()._update_output_batch(output_batch, output)
