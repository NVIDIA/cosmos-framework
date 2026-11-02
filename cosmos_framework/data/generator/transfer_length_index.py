# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a complete, version-pinned Transfer video census and disjoint length index.

Only source metadata rows are read: no video bytes are downloaded. The runtime
adapter supplies candidate selection so source/caption/control eligibility cannot
silently diverge. Every unique RGB URI belongs to one bucket, with its longest
fully available caption window. The full window is resampled, never prefix-cropped.
A manifest is published only after scanning every source row and closing SQLite.

Build an index from metadata rows with ``build_index_from_rows``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter

from collections.abc import Callable, Iterable

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LengthIndexPolicy:
    """Temporal policy shared with exact-length Transfer augmentation."""

    max_latent_frames: int = 51
    temporal_compression_factor: int = 4
    min_stride: int = 1
    max_stride: int = 3
    min_fps: float = 5.0
    max_fps: float = 60.0

    def __post_init__(self) -> None:
        if self.max_latent_frames < 1 or self.temporal_compression_factor < 1:
            raise ValueError("Frame count and temporal compression factor must be positive.")
        if not 1 <= self.min_stride <= self.max_stride:
            raise ValueError("Expected 1 <= min_stride <= max_stride.")
        if not 0 < self.min_fps <= self.max_fps or not math.isfinite(self.max_fps):
            raise ValueError("Expected finite 0 < min_fps <= max_fps.")


class TransferCandidate(Protocol):
    """Metadata-only interface of the canonical runtime adapter's candidate."""

    rgb: dict[str, Any]
    caption_chunks: dict[str, dict[str, Any]]
    persisted_by_modality: dict[str, dict[str, Any]]
    available_modalities: tuple[str, ...]


@dataclass(frozen=True)
class IndexedVideo:
    """One unique source video and its chosen complete caption window."""

    bucket: int
    video_hash: str
    row_id: str
    sample_id: str
    stream_id: str
    rgb_uri: str
    caption_key: str
    caption_start: int
    caption_end: int
    source_frames: int
    source_fps: float


def canonical_video_hash(rgb_uri: str) -> str:
    """Hash the canonical, case-sensitive object URI without caption/row identity."""
    if not rgb_uri or rgb_uri != rgb_uri.strip():
        raise ValueError("A canonical RGB URI must be nonempty without surrounding whitespace.")
    return hashlib.sha256(rgb_uri.encode("utf-8")).hexdigest()


def _positive_int(value: Any) -> int:
    number = int(value)
    if isinstance(value, bool) or number < 1 or float(value) != number:
        raise ValueError("Expected a positive integer.")
    return number


def _validate_control_metadata(candidate: TransferCandidate, source_frames: int, source_fps: float) -> int:
    """Match runtime geometry/FPS checks; return the common available frame limit."""
    available_frames = source_frames
    for modality in candidate.available_modalities:
        if modality not in candidate.persisted_by_modality:
            continue
        control = candidate.persisted_by_modality[modality]
        control_frames = _positive_int(control.get("total_frames"))
        control_fps = float(control.get("framerate"))
        if not math.isfinite(control_fps) or control_fps <= 0:
            raise ValueError("invalid_control_fps")
        for dimension in ("width_px", "height_px"):
            if _positive_int(candidate.rgb.get(dimension)) != _positive_int(control.get(dimension)):
                raise ValueError("control_geometry_mismatch")
        # VideoTransferAlignedSelectedControlParsing tolerates at most one frame
        # of accumulated duration drift over the common RGB/control length.
        comparable_frames = min(source_frames, control_frames)
        duration_delta = abs(comparable_frames / source_fps - comparable_frames / control_fps)
        one_frame_seconds = max(1 / source_fps, 1 / control_fps)
        if not math.isclose(source_fps, control_fps, rel_tol=0.0, abs_tol=1e-3) and (
            duration_delta > one_frame_seconds + 1e-9
        ):
            raise ValueError("control_fps_mismatch")
        available_frames = min(available_frames, control_frames)
    return available_frames


def select_indexed_video(
    candidate: TransferCandidate,
    row: dict[str, Any],
    policy: LengthIndexPolicy,
    exclusions: Counter[str],
) -> IndexedVideo | None:
    """Choose the longest eligible complete caption, breaking ties reproducibly."""
    try:
        source_frames = _positive_int(candidate.rgb.get("total_frames"))
        for dimension in ("width_px", "height_px"):
            _positive_int(candidate.rgb.get(dimension))
        source_fps = float(candidate.rgb.get("framerate"))
        if not math.isfinite(source_fps) or not policy.min_fps <= source_fps <= policy.max_fps:
            exclusions["source_fps_out_of_range"] += 1
            return None
        available_frames = _validate_control_metadata(candidate, source_frames, source_fps)
        rgb_uri = str(candidate.rgb["data_uri"])
        video_hash = canonical_video_hash(rgb_uri)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        reason = str(error)
        exclusions[
            reason
            if reason in ("control_geometry_mismatch", "control_fps_mismatch", "invalid_control_fps")
            else "invalid_asset_metadata"
        ] += 1
        return None

    eligible: list[tuple[int, int, int, str, int]] = []
    for caption_key, chunk in candidate.caption_chunks.items():
        try:
            start, end = int(chunk["start_frame"]), int(chunk["end_frame"])
            caption = json.loads(chunk["caption"])
            if not isinstance(caption, dict) or not caption:
                raise ValueError("Caption must be a nonempty structured dictionary.")
        except (KeyError, TypeError, ValueError, OverflowError):
            exclusions["invalid_structured_caption"] += 1
            continue
        if start < 0 or end <= start or end > available_frames:
            exclusions["caption_outside_aligned_video"] += 1
            continue
        window_frames = end - start
        bucket = min(policy.max_latent_frames, 1 + (window_frames - 1) // policy.temporal_compression_factor)
        target_frames = 1 + policy.temporal_compression_factor * (bucket - 1)
        if target_frames == 1 and window_frames != 1:
            exclusions["multi_frame_caption_in_single_frame_bucket"] += 1
            continue
        stride = (window_frames - 1) / (target_frames - 1) if target_frames > 1 else 1.0
        if not policy.min_stride <= stride <= policy.max_stride:
            exclusions["caption_resampling_stride_out_of_range"] += 1
            continue
        if not policy.min_fps <= source_fps / stride <= policy.max_fps:
            exclusions["caption_resampling_fps_out_of_range"] += 1
            continue
        eligible.append((-window_frames, start, end, caption_key, bucket))
    if not eligible:
        exclusions["no_eligible_caption_window"] += 1
        return None
    _, start, end, caption_key, bucket = min(eligible)
    return IndexedVideo(
        bucket=bucket,
        video_hash=video_hash,
        row_id=str(int(row["_rowid"])),
        sample_id=str(row.get("sample_id") or row.get("source_sample_id") or ""),
        stream_id=str(candidate.rgb["stream_id"]),
        rgb_uri=rgb_uri,
        caption_key=caption_key,
        caption_start=start,
        caption_end=end,
        source_frames=source_frames,
        source_fps=source_fps,
    )


_VIDEO_COLUMNS = "bucket, video_hash, row_id, sample_id, stream_id, rgb_uri, caption_key, caption_start, caption_end, source_frames, source_fps"
_VIDEO_SCHEMA = """
    bucket INTEGER NOT NULL,
    video_hash TEXT NOT NULL UNIQUE,
    row_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    rgb_uri TEXT NOT NULL,
    caption_key TEXT NOT NULL,
    caption_start INTEGER NOT NULL,
    caption_end INTEGER NOT NULL,
    source_frames INTEGER NOT NULL,
    source_fps REAL NOT NULL
"""


def _publish_index(source: Path, destination: Path) -> str:
    """Copy the closed local database sequentially, then atomically publish it."""
    temporary_index = destination.with_suffix(".sqlite.tmp")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary_index.open("xb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    temporary_index.chmod(0o444)
    os.replace(temporary_index, destination)
    return digest.hexdigest()


def build_index_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate_selector: Callable[[dict[str, Any]], list[TransferCandidate]],
    source_uri: str,
    source_version: int,
    source_filter: str,
    output_dir: str | Path,
    expected_source_rows: int,
    policy: LengthIndexPolicy = LengthIndexPolicy(),
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Stream a complete pinned population into SQLite; atomically publish its manifest.

    Duplicate RGB URIs retain the lowest numeric Lance row ID among eligible rows.
    Bucket positions are then dense and ordered by hash, independent of scan order.
    Failed/incomplete scans intentionally leave no manifest and cannot be consumed.
    SQLite runs on local /tmp storage; only the completed database is copied out.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    index_path = output / "index.sqlite"
    # Indexed insertion, external sorting, and VACUUM generate many small I/Os.
    # Keep them off Lustre; publish one sequential copy after SQLite is closed.
    with tempfile.TemporaryDirectory(prefix="transfer-length-index-", dir="/tmp") as scratch_dir:
        local_index = Path(scratch_dir) / "index.sqlite"
        connection = sqlite3.connect(local_index)
        exclusions: Counter[str] = Counter()
        scanned_rows, candidate_occurrences, eligible_occurrences = 0, 0, 0
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute(f"CREATE TABLE staging ({_VIDEO_SCHEMA})")
            placeholders = ",".join("?" for _ in range(11))
            assignments = ",".join(f"{column}=excluded.{column}" for column in _VIDEO_COLUMNS.split(", "))
            insert_sql = (
                f"INSERT INTO staging ({_VIDEO_COLUMNS}) VALUES ({placeholders}) "
                f"ON CONFLICT(video_hash) DO UPDATE SET {assignments} "
                "WHERE length(excluded.row_id) < length(staging.row_id) "
                "OR (length(excluded.row_id) = length(staging.row_id) AND excluded.row_id < staging.row_id)"
            )
            for row in rows:
                scanned_rows += 1
                candidates = candidate_selector(row)
                if not candidates:
                    exclusions["row_without_runtime_candidate"] += 1
                for candidate in candidates:
                    candidate_occurrences += 1
                    video = select_indexed_video(candidate, row, policy, exclusions)
                    if video is not None:
                        eligible_occurrences += 1
                        connection.execute(insert_sql, tuple(asdict(video).values()))
                if scanned_rows % 4096 == 0:
                    connection.commit()
                    if progress is not None:
                        progress(scanned_rows)
            if scanned_rows != expected_source_rows:
                raise ValueError(f"Incomplete source scan: read {scanned_rows} rows, expected {expected_source_rows}.")
            connection.execute(
                f"CREATE TABLE videos ({_VIDEO_SCHEMA}, slot INTEGER NOT NULL, PRIMARY KEY(bucket, slot))"
            )
            connection.execute(
                f"INSERT INTO videos ({_VIDEO_COLUMNS}, slot) SELECT {_VIDEO_COLUMNS}, "
                "ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY video_hash) - 1 FROM staging"
            )
            counts = {str(bucket): 0 for bucket in range(1, policy.max_latent_frames + 1)}
            counts.update(
                {
                    str(bucket): count
                    for bucket, count in connection.execute("SELECT bucket, COUNT(*) FROM videos GROUP BY bucket")
                }
            )
            total_videos = sum(counts.values())
            if total_videos == 0:
                raise ValueError("The complete source scan contained no eligible unique videos.")
            connection.execute("DROP TABLE staging")
            connection.commit()
            connection.execute("VACUUM")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError(f"SQLite integrity check failed: {integrity}.")
        finally:
            connection.close()
        index_sha256 = _publish_index(local_index, index_path)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "source_uri": source_uri,
        "source_version": source_version,
        "source_filter": source_filter,
        "index_file": index_path.name,
        "index_sha256": index_sha256,
        "bucket_counts": counts,
        "bucket_probabilities": {bucket: count / total_videos for bucket, count in counts.items()},
        "total_videos": total_videos,
        "policy": asdict(policy),
        "identity": "sha256(canonical_rgb_uri_utf8)",
        "caption_selection": "longest_complete_eligible_window_then_start_end_key",
        "duplicate_resolution": "lowest_eligible_numeric_lance_row_id",
        "measurement": "canonical_lance_metadata_no_media_download",
        "coverage": {
            "expected_source_rows": expected_source_rows,
            "scanned_source_rows": scanned_rows,
            "complete": True,
            "candidate_occurrences": candidate_occurrences,
            "eligible_occurrences": eligible_occurrences,
            "duplicate_eligible_occurrences": eligible_occurrences - total_videos,
        },
        "exclusions": dict(sorted(exclusions.items())),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_manifest = output / "manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_manifest, output / "manifest.json")
    (output / "manifest.json").chmod(0o444)
    return manifest


