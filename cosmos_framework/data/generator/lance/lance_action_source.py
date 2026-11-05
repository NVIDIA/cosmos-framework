# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

from __future__ import annotations

from dataclasses import dataclass, field

from cosmos_framework.data.generator.lance.lance_base_source import LanceBaseDatasetSpec

ACTION_CANONICAL_CAPTION_COLUMN: str = "canonical_caption"


ACTION_SOURCE_CAPTION_COLUMNS: tuple[str, ...] = ("caption", "source_caption")


ACTION_DENSE_CAPTION_COLUMN: str = "caption_cosmos3p5_v1"


ACTION_SAMPLE_ROW_ID_COLUMN: str = "action_sample_row_id"


ACTION_DENSE_CAPTION_RANGES_COLUMN: str = "dense_caption_ranges"


@dataclass
class ActionSamplesColumns:
    """Physical column names in ``action_samples.lance``."""

    source_dataset: str = "source_dataset"
    row_ids: str = "row_ids"
    view_names: str = "view_names"
    source_lance_uri: str = "source_lance_uri"
    source_lance_version: str = "source_lance_version"
    shard_id: str = "shard_id"
    subtask_index: str = "subtask_index"
    keep_range: str = "keep_range"
    retention_quality_score: str = "retention_quality_score"
    dense_caption_chunks: str = ACTION_DENSE_CAPTION_COLUMN

    def names(self) -> tuple[str, ...]:
        """Return every configured ``action_samples.lance`` column name."""
        return (
            self.source_dataset,
            self.row_ids,
            self.view_names,
            self.source_lance_uri,
            self.source_lance_version,
            self.shard_id,
            self.subtask_index,
            self.keep_range,
            self.retention_quality_score,
            self.dense_caption_chunks,
        )

    def cache_builder_projection(
        self,
        *,
        read_keep_range: bool = True,
        read_retention_quality_score: bool = True,
        read_dense_caption_chunks: bool = True,
    ) -> list[str]:
        """Return ``action_samples.lance`` columns read by the row-index cache builder."""
        columns = [
            self.source_dataset,
            self.row_ids,
            self.view_names,
            self.source_lance_uri,
            self.source_lance_version,
            self.shard_id,
            self.subtask_index,
        ]
        if read_keep_range:
            columns.append(self.keep_range)
        if read_retention_quality_score:
            columns.append(self.retention_quality_score)
        if read_dense_caption_chunks:
            columns.append(self.dense_caption_chunks)
        return list(dict.fromkeys(columns))


@dataclass
class ActionSpansColumns:
    """Physical column names in ``action_spans.lance``."""

    source_dataset: str = "source_dataset"
    view_name: str = "view_name"
    frame_count: str = "nb_frames"
    source_fps: str = "framerate"
    episode_index: str = "episode_index"
    shard_id: str = "shard_id"
    clip_url: str = "clip_url"
    action_url: str = "action_data_url"
    nb_bytes: str = "nb_bytes"
    moov_offset: str = "moov_offset"
    moov_size: str = "moov_size"
    width: str = "width"
    height: str = "height"

    def names(self) -> tuple[str, ...]:
        """Return every configured ``action_spans.lance`` column name."""
        return (
            self.source_dataset,
            self.view_name,
            self.frame_count,
            self.source_fps,
            self.episode_index,
            self.shard_id,
            self.clip_url,
            self.action_url,
            self.nb_bytes,
            self.moov_offset,
            self.moov_size,
            self.width,
            self.height,
        )

    def cache_builder_projection(
        self,
        recorded_columns: tuple[str, ...],
        caption_columns: tuple[str, ...],
    ) -> list[str]:
        """Return ``action_spans.lance`` columns copied into row-index caches."""
        columns = [
            self.frame_count,
            self.source_fps,
            self.episode_index,
            *recorded_columns,
            *caption_columns,
        ]
        return list(dict.fromkeys(columns))

    def runtime_projection(self, *, include_video: bool, include_video_dimensions: bool = False) -> list[str]:
        """Return ``action_spans.lance`` columns read by training workers."""
        columns = [
            self.source_dataset,
            self.episode_index,
            self.view_name,
            self.frame_count,
            self.source_fps,
            self.action_url,
        ]
        if include_video:
            columns.extend([self.clip_url, self.nb_bytes, self.moov_offset, self.moov_size])
        if include_video_dimensions:
            columns.extend([self.width, self.height])
        return list(dict.fromkeys(columns))


@dataclass
class LanceActionDatasetSpec(LanceBaseDatasetSpec):
    """Static contract for building dataset-specific Action source-row caches.

    Action caches select logical subtask rows from ``source_index_uri`` for each
    entry in ``source_datasets``, then store the source index's parallel
    ``row_ids`` and ``view_names`` lists. Sources normally use separate cache
    tables; ``combine_source_datasets`` writes one table named after the spec
    instead.
    Training-specific window sizes, sampling FPS values, and train/validation
    policy are intentionally excluded so one cache can serve many recipes.
    """

    # Split policy belongs outside this source/cache spec. The cache records
    # enough source identity to split episodes globally across shards while
    # keeping every subtask from one shard-local source episode in the same split.
    source_index_uri: str
    action_samples: ActionSamplesColumns = field(default_factory=ActionSamplesColumns)
    action_spans: ActionSpansColumns = field(default_factory=ActionSpansColumns)
    use_native_gcs_for_cache_write: bool = field(default=True, kw_only=True)
    recorded_columns: tuple[str, ...] = ()
    caption_columns: tuple[str, ...] = ()
    source_datasets: tuple[str, ...] = ()
    combine_source_datasets: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Action Lance spec name must not be empty.")
        if not self.lance_uri:
            raise ValueError("Action Lance spec lance_uri must not be empty.")
        if not self.source_index_uri:
            raise ValueError("Action Lance spec source_index_uri must not be empty.")

        for table_name, columns in (
            ("action_samples", self.action_samples.names()),
            ("action_spans", self.action_spans.names()),
        ):
            if any(not column for column in columns):
                raise ValueError(f"Action Lance {table_name} column names must not be empty.")
            duplicates = sorted({column for column in columns if columns.count(column) > 1})
            if duplicates:
                raise ValueError(f"Action Lance {table_name} column names must not contain duplicates: {duplicates}.")
        if any(not column for column in self.recorded_columns):
            raise ValueError("Action Lance source column names must not be empty.")
        if any(not column for column in self.caption_columns):
            raise ValueError("Action Lance caption column names must not be empty.")
        if len(set(self.caption_columns)) != len(self.caption_columns):
            raise ValueError(f"Action Lance caption columns must not contain duplicates: {self.caption_columns!r}.")
        if len(set(self.source_datasets)) != len(self.source_datasets):
            raise ValueError(f"Action Lance source datasets must not contain duplicates: {self.source_datasets!r}.")
        if self.combine_source_datasets and not self.source_datasets:
            raise ValueError("Combined Action Lance specs must select at least one source dataset.")

    def row_index_cache_columns(self) -> list[str]:
        """Return the metadata column order written to each Action row-index cache."""
        columns = [
            self.action_spans.frame_count,
            self.action_spans.source_fps,
            self.action_spans.episode_index,
            ACTION_SAMPLE_ROW_ID_COLUMN,
            self.action_samples.shard_id,
            self.action_samples.subtask_index,
            self.action_samples.keep_range,
            self.action_samples.retention_quality_score,
            ACTION_DENSE_CAPTION_RANGES_COLUMN,
            *self.recorded_columns,
        ]
        return list(dict.fromkeys(columns))
