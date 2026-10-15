# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Write the exact conversation rows and columns needed by training.

The grouped conversation row-index cache stores physical Lance row IDs. This
module materializes the union of those IDs from one or more immutable source
snapshots into a new, projected Lance table. It deliberately does not resolve
recipes, rewrite row-index caches, or copy media shards.

Callers must provide a fresh output URI. A failed export can leave uncommitted
data files there, so production callers should export into a staging URI and
publish it only after this function returns successfully.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import monotonic

import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from lance.fragment import FragmentMetadata, LanceFragment

TRAINING_CONVERSATION_COLUMNS: tuple[str, ...] = (
    "conv_uuid",
    "media_uuid",
    "media_key",
    "source_id",
    "dataset_name",
    "conv_text",
    "media_type",
    "media_files",
)

_DEFAULT_BATCH_SIZE: int = 65_536
_MAX_SOURCE_READ_ROWS: int = 32_768
_WRITE_WORKERS: int = 2
_LANCE_DATA_STORAGE_VERSION: str = "2.1"
_LANCE_ROW_ID_COLUMN: str = "_rowid"
_LANCE_ROW_ID_FRAGMENT_SHIFT: int = 32
_MAX_UINT32: int = (1 << _LANCE_ROW_ID_FRAGMENT_SHIFT) - 1
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PinnedConversationSource:
    """One immutable source snapshot and its sorted unique physical row IDs."""

    source_uri: str
    source_version: int
    required_row_ids: npt.NDArray[np.uint64]
    storage_options: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ConversationSourceIdentity:
    """Stable identity passed to a projected-batch transform."""

    source_uri: str
    source_version: int


@dataclass(frozen=True, slots=True)
class ConversationRowIdMapping:
    """Sorted physical row-ID mapping for one source snapshot."""

    source_uri: str
    source_version: int
    old_row_ids: npt.NDArray[np.uint64]
    new_row_ids: npt.NDArray[np.uint64]


@dataclass(frozen=True, slots=True)
class ConversationTableExportResult:
    """Committed projected-table identity and snapshot-specific row mappings."""

    output_uri: str
    output_version: int
    row_count: int
    mappings: tuple[ConversationRowIdMapping, ...]


ConversationBatchTransform = Callable[
    [ConversationSourceIdentity, npt.NDArray[np.uint64], pa.Table],
    pa.Table,
]


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    identity: ConversationSourceIdentity
    row_ids: npt.NDArray[np.uint64]
    storage_options: dict[str, str] | None


def _prepare_sources(sources: Iterable[PinnedConversationSource]) -> tuple[_PreparedSource, ...]:
    prepared: list[_PreparedSource] = []
    snapshots: set[ConversationSourceIdentity] = set()
    for source_index, source in enumerate(sources):
        source_uri = source.source_uri.strip()
        if not source_uri:
            raise ValueError(f"sources[{source_index}].source_uri must not be empty")
        if source.source_version < 1:
            raise ValueError(f"sources[{source_index}].source_version must be positive")
        row_ids = np.asarray(source.required_row_ids)
        if row_ids.dtype != np.uint64 or row_ids.ndim != 1:
            raise ValueError(f"sources[{source_index}].required_row_ids must be a one-dimensional uint64 array")
        if len(row_ids) > 1 and bool(np.any(row_ids[1:] <= row_ids[:-1])):
            raise ValueError(f"sources[{source_index}].required_row_ids must be sorted and unique")
        if row_ids.flags.writeable:
            row_ids = row_ids.view()
            row_ids.setflags(write=False)
        identity = ConversationSourceIdentity(source_uri=source_uri, source_version=source.source_version)
        if identity in snapshots:
            raise ValueError(f"Source snapshot {source_uri!r} v{source.source_version} was provided more than once")
        snapshots.add(identity)
        prepared.append(
            _PreparedSource(
                identity=identity,
                row_ids=row_ids,
                storage_options=dict(source.storage_options) if source.storage_options else None,
            )
        )

    if not prepared:
        raise ValueError("At least one pinned conversation source is required")
    return tuple(sorted(prepared, key=lambda source: (source.identity.source_uri, source.identity.source_version)))


def _projected_schema(dataset: lance.LanceDataset, identity: ConversationSourceIdentity) -> pa.Schema:
    missing_columns = [column for column in TRAINING_CONVERSATION_COLUMNS if column not in dataset.schema.names]
    if missing_columns:
        raise ValueError(
            f"Conversation source {identity.source_uri!r} v{identity.source_version} is missing training columns "
            f"{missing_columns}; available columns: {list(dataset.schema.names)}"
        )
    return pa.schema([dataset.schema.field(column) for column in TRAINING_CONVERSATION_COLUMNS])


def _open_sources(
    prepared_sources: tuple[_PreparedSource, ...],
) -> tuple[dict[ConversationSourceIdentity, lance.LanceDataset], pa.Schema]:
    datasets: dict[ConversationSourceIdentity, lance.LanceDataset] = {}
    output_schema: pa.Schema | None = None
    for source in prepared_sources:
        identity = source.identity
        try:
            dataset = lance.dataset(
                identity.source_uri,
                version=identity.source_version,
                storage_options=source.storage_options,
            )
        except Exception as error:  # noqa: BLE001 - Lance raises backend-specific exceptions.
            raise RuntimeError(
                f"Could not open pinned conversation source {identity.source_uri!r} v{identity.source_version}"
            ) from error
        if int(dataset.version) != identity.source_version:
            raise RuntimeError(
                f"Conversation source {identity.source_uri!r} opened at v{dataset.version}, "
                f"expected pinned v{identity.source_version}"
            )
        source_schema = _projected_schema(dataset, identity)
        if output_schema is None:
            output_schema = source_schema
        elif not output_schema.equals(source_schema, check_metadata=False):
            raise ValueError(
                f"Conversation source {identity.source_uri!r} v{identity.source_version} has incompatible training "
                f"schema {source_schema}; expected {output_schema}"
            )
        datasets[identity] = dataset
    assert output_schema is not None
    return datasets, output_schema


def _require_output_table_absent(output_uri: str, storage_options: dict[str, str] | None) -> None:
    try:
        lance.dataset(output_uri, storage_options=storage_options)
    except FileNotFoundError:
        return
    except ValueError as error:
        message = str(error).lower()
        if "dataset at path" in message and "was not found" in message:
            return
        raise RuntimeError(f"Could not verify that output table {output_uri!r} is absent") from error
    except Exception as error:  # noqa: BLE001 - Lance raises backend-specific exceptions.
        raise RuntimeError(f"Could not verify that output table {output_uri!r} is absent") from error
    raise FileExistsError(f"Output URI {output_uri!r} already contains a Lance table")


def _take_source_rows(dataset: lance.LanceDataset, row_ids: list[int]) -> pa.Table:
    """Bound source reads and split Arrow overflows, preserving chunked columns and row order."""
    if len(row_ids) > _MAX_SOURCE_READ_ROWS:
        return pa.concat_tables(
            [
                _take_source_rows(dataset, row_ids[start : start + _MAX_SOURCE_READ_ROWS])
                for start in range(0, len(row_ids), _MAX_SOURCE_READ_ROWS)
            ]
        )
    try:
        return dataset._take_rows(row_ids, columns=[_LANCE_ROW_ID_COLUMN, *TRAINING_CONVERSATION_COLUMNS])
    except OSError as error:
        if "Offset overflow" not in str(error) or len(row_ids) < 2:
            raise
        midpoint = len(row_ids) // 2
        _LOGGER.warning(
            "phase=table_read splitting Arrow offset overflow rows=%d first_row=%d", len(row_ids), row_ids[0]
        )
        return pa.concat_tables(
            [
                _take_source_rows(dataset, row_ids[:midpoint]),
                _take_source_rows(dataset, row_ids[midpoint:]),
            ]
        )


def _take_projected_rows_exact(
    dataset: lance.LanceDataset,
    source: _PreparedSource,
    requested_row_ids: npt.NDArray[np.uint64],
    output_schema: pa.Schema,
    transform_batch: ConversationBatchTransform | None,
) -> pa.Table:
    requested_as_list = [int(row_id) for row_id in requested_row_ids]
    try:
        fetched = _take_source_rows(dataset, requested_as_list)
    except Exception as error:  # noqa: BLE001 - Lance raises backend-specific exceptions.
        raise RuntimeError(
            f"Could not fetch {len(requested_as_list)} rows from conversation source "
            f"{source.identity.source_uri!r} v{source.identity.source_version}"
        ) from error
    if not isinstance(fetched, pa.Table):
        raise TypeError(f"Lance _take_rows returned {type(fetched).__name__}, expected pyarrow.Table")
    if fetched.num_rows != len(requested_as_list):
        raise RuntimeError(
            f"Conversation source {source.identity.source_uri!r} v{source.identity.source_version} returned "
            f"{fetched.num_rows} rows for {len(requested_as_list)} requested physical row IDs"
        )
    if _LANCE_ROW_ID_COLUMN not in fetched.column_names:
        raise RuntimeError(
            f"Conversation source {source.identity.source_uri!r} v{source.identity.source_version} did not return "
            f"the {_LANCE_ROW_ID_COLUMN!r} verification column"
        )
    observed_column = fetched.column(_LANCE_ROW_ID_COLUMN).combine_chunks()
    if not pa.types.is_uint64(observed_column.type) or observed_column.null_count:
        raise RuntimeError(
            f"Conversation source {source.identity.source_uri!r} v{source.identity.source_version} returned an "
            f"invalid {_LANCE_ROW_ID_COLUMN!r} column of type {observed_column.type}"
        )
    observed_row_ids = observed_column.to_numpy(zero_copy_only=False)
    if not np.array_equal(observed_row_ids, requested_row_ids):
        raise RuntimeError(
            f"Conversation source {source.identity.source_uri!r} v{source.identity.source_version} returned "
            "physical row IDs that were missing, different, or reordered"
        )

    projected = fetched.select(list(TRAINING_CONVERSATION_COLUMNS))
    return _transform_projected_rows(source, requested_row_ids, projected, output_schema, transform_batch)


def _transform_projected_rows(
    source: _PreparedSource,
    requested_row_ids: npt.NDArray[np.uint64],
    projected: pa.Table,
    output_schema: pa.Schema,
    transform_batch: ConversationBatchTransform | None,
) -> pa.Table:
    if transform_batch is not None:
        projected = transform_batch(source.identity, requested_row_ids, projected)
        if not isinstance(projected, pa.Table):
            raise TypeError(f"transform_batch returned {type(projected).__name__}, expected pyarrow.Table")
    if projected.num_rows != len(requested_row_ids):
        raise RuntimeError(
            f"Projected-batch transform returned {projected.num_rows} rows for {len(requested_row_ids)} source rows"
        )
    if projected.column_names != list(TRAINING_CONVERSATION_COLUMNS):
        raise ValueError(
            "Projected conversation batch must contain exactly the training columns in canonical order; "
            f"got {projected.column_names}"
        )
    if not output_schema.equals(projected.schema, check_metadata=False):
        raise ValueError(f"Projected conversation batch schema {projected.schema} does not match {output_schema}")
    return projected


def _iter_projected_batches(
    dataset: lance.LanceDataset,
    source: _PreparedSource,
    output_schema: pa.Schema,
    batch_size: int,
    prefetch_batches: int,
) -> Iterator[tuple[int, pa.Table]]:
    starts = iter(range(0, len(source.row_ids), batch_size))

    def fetch(start: int) -> pa.Table:
        return _take_projected_rows_exact(
            dataset, source, source.row_ids[start : start + batch_size], output_schema, None
        )

    if prefetch_batches == 0:
        for start in starts:
            yield start, fetch(start)
        return
    with ThreadPoolExecutor(max_workers=prefetch_batches) as pool:
        pending = deque((start, pool.submit(fetch, start)) for start in islice(starts, prefetch_batches))
        while pending:
            start, future = pending.popleft()
            yield start, future.result()
            next_start = next(starts, None)
            if next_start is not None:
                pending.append((next_start, pool.submit(fetch, next_start)))


def _new_physical_row_ids(fragment_id: int, row_count: int) -> npt.NDArray[np.uint64]:
    fragment_base = np.uint64(fragment_id << _LANCE_ROW_ID_FRAGMENT_SHIFT)
    return fragment_base + np.arange(row_count, dtype=np.uint64)


def _validate_committed_output(
    dataset: lance.LanceDataset,
    *,
    output_schema: pa.Schema,
    expected_row_count: int,
    expected_fragment_count: int,
) -> None:
    if str(dataset.data_storage_version) != _LANCE_DATA_STORAGE_VERSION:
        raise RuntimeError(
            f"Projected conversation table uses data storage version {dataset.data_storage_version!r}, "
            f"expected {_LANCE_DATA_STORAGE_VERSION!r}"
        )
    if not output_schema.equals(dataset.schema, check_metadata=False):
        raise RuntimeError(f"Committed conversation schema {dataset.schema} does not match {output_schema}")
    observed_row_count = int(dataset.count_rows())
    if observed_row_count != expected_row_count:
        raise RuntimeError(f"Committed conversation table has {observed_row_count} rows, expected {expected_row_count}")
    fragments = list(dataset.get_fragments())
    fragment_ids = [int(fragment.fragment_id) for fragment in fragments]
    if fragment_ids != list(range(expected_fragment_count)):
        raise RuntimeError(
            f"Committed conversation table has fragment IDs {fragment_ids}, "
            f"expected {list(range(expected_fragment_count))}"
        )


def export_projected_conversation_table(
    sources: Iterable[PinnedConversationSource],
    output_uri: str | Path,
    *,
    output_storage_options: Mapping[str, str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    prefetch_batches: int = 2,
    transform_batch: ConversationBatchTransform | None = None,
) -> ConversationTableExportResult:
    """Create a training-column table containing exactly the requested source rows.

    Each ``(source_uri, source_version)`` snapshot must appear once with sorted
    unique ``uint64`` row IDs. Up to ``prefetch_batches`` batches are fetched
    ahead; zero disables prefetch. Transforms and committed fragments remain in
    source order, with at most two writes in flight. Rows are written as one
    explicit fragment per batch. The returned mappings retain read-only
    row-ID arrays.

    ``transform_batch`` may rewrite values such as absolute media shard paths.
    It runs only after the source ``_rowid`` sequence has been verified and must
    preserve row count, row order, column order, and schema.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1 or batch_size > _MAX_UINT32 + 1:
        raise ValueError(f"batch_size must be in [1, {_MAX_UINT32 + 1}], got {batch_size}")
    if isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int) or prefetch_batches < 0:
        raise ValueError("prefetch_batches must be a non-negative integer")
    output_uri_string = str(output_uri).strip()
    if not output_uri_string:
        raise ValueError("output_uri must not be empty")
    normalized_output_storage_options = dict(output_storage_options) if output_storage_options else None
    prepared_sources = _prepare_sources(sources)
    started = monotonic()
    _LOGGER.info("phase=table_open sources=%d", len(prepared_sources))
    datasets, output_schema = _open_sources(prepared_sources)
    _LOGGER.info("phase=table_open complete seconds=%.2f", monotonic() - started)
    _require_output_table_absent(output_uri_string, normalized_output_storage_options)

    expected_fragment_count = sum((len(source.row_ids) + batch_size - 1) // batch_size for source in prepared_sources)
    if expected_fragment_count > _MAX_UINT32 + 1:
        raise ValueError(
            f"Export needs {expected_fragment_count} fragments, exceeding the physical row-ID limit "
            f"of {_MAX_UINT32 + 1}"
        )

    fragments: list[FragmentMetadata] = []
    mappings: list[ConversationRowIdMapping] = []
    fragment_id = 0
    total_rows = sum(len(source.row_ids) for source in prepared_sources)
    written_rows = 0
    projected_bytes = 0
    started = last_report = monotonic()
    pending_writes: deque[tuple[Future[FragmentMetadata], int, int, bool]] = deque()

    def write_fragment(fragment_id: int, projected: pa.Table) -> FragmentMetadata:
        try:
            fragment = LanceFragment.create(
                dataset_uri=output_uri_string,
                data=projected,
                fragment_id=fragment_id,
                schema=output_schema,
                mode="create",
                data_storage_version=_LANCE_DATA_STORAGE_VERSION,
                storage_options=normalized_output_storage_options,
            )
        except Exception as error:  # noqa: BLE001 - Lance raises backend-specific exceptions.
            raise RuntimeError(
                f"Could not write projected conversation fragment {fragment_id} to {output_uri_string!r}"
            ) from error
        if int(fragment.physical_rows) != projected.num_rows:
            raise RuntimeError(
                f"Projected conversation fragment {fragment_id} reports {fragment.physical_rows} physical rows, "
                f"expected {projected.num_rows}"
            )
        return fragment

    def finish_write() -> None:
        nonlocal written_rows, projected_bytes, last_report
        future, finished_source_index, batch_bytes, source_complete = pending_writes.popleft()
        fragment = future.result()
        fragments.append(fragment)
        written_rows += int(fragment.physical_rows)
        projected_bytes += batch_bytes
        now = monotonic()
        if now - last_report >= 30 or source_complete:
            elapsed = max(now - started, 1e-9)
            _LOGGER.info(
                "phase=table_export sources=%d/%d rows=%d/%d rows_per_sec=%.1f projected_MiB_per_sec=%.1f seconds=%.1f",
                finished_source_index,
                len(prepared_sources),
                written_rows,
                total_rows,
                written_rows / elapsed,
                projected_bytes / elapsed / 2**20,
                elapsed,
            )
            last_report = now

    _LOGGER.info("phase=table_export rows=%d prefetch_batches=%d", total_rows, prefetch_batches)
    with ThreadPoolExecutor(max_workers=_WRITE_WORKERS) as writer:
        for source_index, source in enumerate(prepared_sources, 1):
            new_row_ids = np.empty(len(source.row_ids), dtype=np.uint64)
            dataset = datasets[source.identity]
            for start, fetched in _iter_projected_batches(dataset, source, output_schema, batch_size, prefetch_batches):
                requested_row_ids = source.row_ids[start : start + batch_size]
                projected = _transform_projected_rows(
                    source,
                    requested_row_ids,
                    fetched,
                    output_schema,
                    transform_batch,
                )
                pending_writes.append(
                    (
                        writer.submit(write_fragment, fragment_id, projected),
                        source_index,
                        projected.nbytes,
                        start + projected.num_rows == len(source.row_ids),
                    )
                )
                new_row_ids[start : start + len(requested_row_ids)] = _new_physical_row_ids(
                    fragment_id,
                    len(requested_row_ids),
                )
                fragment_id += 1
                if len(pending_writes) >= _WRITE_WORKERS:
                    finish_write()
            new_row_ids.setflags(write=False)
            mappings.append(
                ConversationRowIdMapping(
                    source_uri=source.identity.source_uri,
                    source_version=source.identity.source_version,
                    old_row_ids=source.row_ids,
                    new_row_ids=new_row_ids,
                )
            )
        while pending_writes:
            finish_write()

    row_count = sum(len(mapping.old_row_ids) for mapping in mappings)
    try:
        committed = lance.LanceDataset.commit(
            output_uri_string,
            lance.LanceOperation.Overwrite(output_schema, fragments),
            read_version=0,
            storage_options=normalized_output_storage_options,
            enable_v2_manifest_paths=True,
            enable_stable_row_ids=False,
            max_retries=0,
        )
    except Exception as error:  # noqa: BLE001 - Lance raises backend-specific exceptions.
        raise RuntimeError(
            f"Could not commit new projected conversation table at {output_uri_string!r}; "
            "the output URI must not contain an existing Lance table"
        ) from error
    _validate_committed_output(
        committed,
        output_schema=output_schema,
        expected_row_count=row_count,
        expected_fragment_count=expected_fragment_count,
    )
    return ConversationTableExportResult(
        output_uri=output_uri_string,
        output_version=int(committed.version),
        row_count=row_count,
        mappings=tuple(mappings),
    )
