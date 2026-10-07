# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Receipt contract for a training-scoped conversation Lance export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

EXPORT_MANIFEST_FILENAME = "export_manifest.json"
EXPORT_MANIFEST_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value deterministically."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_mapping(
    value: object, *, context: str, required: set[str] | None = None, optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys.")
    if required is not None and (not required <= value.keys() or value.keys() - required - set(optional)):
        raise ValueError(f"{context} requires {sorted(required)}, optionally {list(optional)}; got {sorted(value)}.")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _require_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer.")
    return value


def _require_positive_int(value: object, *, context: str) -> int:
    result = _require_nonnegative_int(value, context=context)
    if result < 1:
        raise ValueError(f"{context} must be at least one.")
    return result


def _require_sha256(value: object, *, context: str) -> str:
    result = _require_string(value, context=context)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return result


def _require_relative_path(value: object, *, context: str) -> str:
    result = _require_string(value, context=context)
    path = Path(result)
    if path.is_absolute() or result != path.as_posix() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{context} must be a normalized relative POSIX path, got {result!r}.")
    return result


@dataclass(frozen=True)
class RecipeSource:
    """One live Recipe DB collection-to-datasource edge and effective weight."""

    collection_name: str
    datasource_name: str
    effective_weight: float

    def __post_init__(self) -> None:
        _require_string(self.collection_name, context="Recipe collection name")
        _require_string(self.datasource_name, context="Recipe datasource name")
        if isinstance(self.effective_weight, bool) or not isinstance(self.effective_weight, (int, float)):
            raise ValueError("Recipe effective weight must be numeric.")
        if not math.isfinite(self.effective_weight) or self.effective_weight <= 0:
            raise ValueError("Recipe effective weights must be positive and finite.")

    @property
    def stream_name(self) -> str:
        """Return this source's stable cache identity."""
        return f"{self.collection_name}__{self.datasource_name}"

    def to_dict(self) -> dict[str, str | float]:
        """Return the canonical JSON representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, context: str) -> "RecipeSource":
        """Parse and validate one recipe source."""
        source = cls(**_require_mapping(value, context=context, required={item.name for item in fields(cls)}))
        return cls(source.collection_name, source.datasource_name, float(source.effective_weight))


def normalize_recipe_sources(
    sources: list[tuple[str, str, float]] | tuple[RecipeSource, ...],
) -> tuple[RecipeSource, ...]:
    """Return a sorted, duplicate-free Recipe DB source snapshot."""
    normalized = (
        tuple(RecipeSource(collection, datasource, weight) for collection, datasource, weight in sources)
        if isinstance(sources, list)
        else sources
    )
    ordered = tuple(sorted(normalized, key=lambda source: source.stream_name))
    names = [source.stream_name for source in ordered]
    if len(names) != len(set(names)):
        raise ValueError("Recipe source snapshot contains duplicate collection/datasource streams.")
    return ordered


def validate_recipe_subset(
    cache_names: set[str],
    excluded_streams: dict[str, str],
    recipe_names: set[str] | None = None,
) -> None:
    """Require explicit, disjoint exclusions and complete live recipe coverage."""
    _require_mapping(excluded_streams, context="excluded_streams")
    for stream_name, reason in excluded_streams.items():
        _require_string(stream_name, context="excluded_streams stream name")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"excluded_streams[{stream_name!r}] must be a non-empty reason.")
    overlap = cache_names & excluded_streams.keys()
    if overlap:
        raise ValueError(f"Caches and excluded_streams overlap: {sorted(overlap)}.")
    covered_names = cache_names | excluded_streams.keys()
    if recipe_names is not None and covered_names != recipe_names:
        raise ValueError(
            "Caches and excluded_streams must exactly cover the live Recipe DB streams: "
            f"missing caches={sorted(recipe_names - covered_names)}, "
            f"unknown streams={sorted(covered_names - recipe_names)}."
        )


@dataclass(frozen=True)
class ExportedRowIndexCache:
    """One immutable local grouped row-index cache in the export bundle."""

    stream_name: str
    path: str
    version: int
    num_groups: int
    num_indexed_rows: int
    ordered_group_selection_sha256: str
    source_cache_uri: str
    source_cache_version: int

    def __post_init__(self) -> None:
        _require_string(self.stream_name, context="row-index cache stream_name")
        _require_relative_path(self.path, context="row-index cache path")
        _require_positive_int(self.version, context="row-index cache version")
        _require_positive_int(self.num_groups, context="row-index cache num_groups")
        _require_positive_int(self.num_indexed_rows, context="row-index cache num_indexed_rows")
        _require_sha256(
            self.ordered_group_selection_sha256,
            context="row-index cache ordered_group_selection_sha256",
        )
        _require_string(self.source_cache_uri, context="row-index cache source_cache_uri")
        _require_positive_int(self.source_cache_version, context="row-index cache source_cache_version")

    def to_dict(self) -> dict[str, str | int]:
        """Return the canonical JSON representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object, *, context: str) -> "ExportedRowIndexCache":
        """Parse and validate one local cache receipt."""
        return cls(**_require_mapping(value, context=context, required={item.name for item in fields(cls)}))


@dataclass(frozen=True)
class ConversationTrainingExportManifest:
    """Validated contents of ``export_manifest.json``.

    Recipe weights are a recorded snapshot used to detect drift. Runtime still
    queries Recipe DB and compares that live result with this receipt before it
    constructs any local Lance source.
    """

    export_id: str
    created_at: str
    recipe_name: str
    recipe_storage_type: str
    recipe_data_type: str
    wandb_run_path: str | None
    recipe_sources: tuple[RecipeSource, ...]
    conversation_table_path: str
    conversation_table_version: int
    conversation_row_count: int
    conversation_columns: tuple[str, ...]
    row_index_caches: tuple[ExportedRowIndexCache, ...]
    media_root_path: str
    media_shard_count: int
    media_total_bytes: int
    media_inventory_sha256: str
    input_identity: dict[str, Any]
    git_commit: str
    is_production_database: bool = False
    excluded_streams: dict[str, str] = field(default_factory=dict)
    recipe_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_sha256(self.export_id, context="export_id")
        _require_string(self.created_at, context="created_at")
        _require_string(self.recipe_name, context="recipe_name")
        _require_string(self.recipe_storage_type, context="recipe_storage_type")
        _require_string(self.recipe_data_type, context="recipe_data_type")
        if not isinstance(self.is_production_database, bool):
            raise ValueError("is_production_database must be a boolean.")
        if self.wandb_run_path is not None:
            _require_string(self.wandb_run_path, context="wandb_run_path")
        normalized_sources = normalize_recipe_sources(self.recipe_sources)
        if normalized_sources != self.recipe_sources or not normalized_sources:
            raise ValueError("recipe_sources must be non-empty and sorted by stream name.")
        _require_relative_path(self.conversation_table_path, context="conversation_table_path")
        _require_positive_int(self.conversation_table_version, context="conversation_table_version")
        _require_positive_int(self.conversation_row_count, context="conversation_row_count")
        if not self.conversation_columns or any(not column for column in self.conversation_columns):
            raise ValueError("conversation_columns must be a non-empty tuple of column names.")
        if len(self.conversation_columns) != len(set(self.conversation_columns)):
            raise ValueError("conversation_columns must not contain duplicates.")
        cache_names = [cache.stream_name for cache in self.row_index_caches]
        if not cache_names or cache_names != sorted(cache_names) or len(cache_names) != len(set(cache_names)):
            raise ValueError("row_index_caches must be non-empty, sorted, and unique by stream_name.")
        if self.recipe_snapshots:
            included: set[str] = set()
            for name in self.recipe_snapshots:
                _require_string(name, context="recipe snapshot name")
                sources, excluded = self.recipe_selection(name)
                names = {source.stream_name for source in sources}
                validate_recipe_subset(names - set(excluded), excluded, names)
                included.update(names - set(excluded))
            if included != set(cache_names):
                raise ValueError("Shared bundle caches differ from the union of recipe selections.")
            if self.recipe_selection(self.recipe_name) != (self.recipe_sources, self.excluded_streams):
                raise ValueError("Primary recipe differs from its shared-bundle snapshot.")
        else:
            validate_recipe_subset(
                set(cache_names), self.excluded_streams, {source.stream_name for source in normalized_sources}
            )
        _require_relative_path(self.media_root_path, context="media_root_path")
        _require_nonnegative_int(self.media_shard_count, context="media_shard_count")
        _require_nonnegative_int(self.media_total_bytes, context="media_total_bytes")
        _require_sha256(self.media_inventory_sha256, context="media_inventory_sha256")
        _require_mapping(self.input_identity, context="input_identity")
        if canonical_json_sha256(self.input_identity) != self.export_id:
            raise ValueError("export_id does not match the canonical input_identity digest.")
        if self.input_identity.get("recipe_snapshots", {}) != self.recipe_snapshots:
            raise ValueError("Recipe snapshots differ from input_identity.")
        identity_recipe = _require_mapping(self.input_identity.get("recipe"), context="input_identity.recipe")
        if (
            identity_recipe.get("is_production_database", False) is not self.is_production_database
            or self.input_identity.get("excluded_streams", {}) != self.excluded_streams
        ):
            raise ValueError("Receipt is_production_database or excluded_streams differs from input_identity.")
        _require_string(self.git_commit, context="git_commit")

    def recipe_selection(self, name: str) -> tuple[tuple[RecipeSource, ...], dict[str, str]]:
        """Select one recipe without changing the shared table or cache identities."""
        if not self.recipe_snapshots:
            if name != self.recipe_name:
                raise ValueError(f"Conversation training export records recipe {self.recipe_name!r}, not {name!r}.")
            return self.recipe_sources, self.excluded_streams
        if name not in self.recipe_snapshots:
            raise ValueError(f"Shared bundle has no recipe snapshot for {name!r}.")
        snapshot = _require_mapping(
            self.recipe_snapshots[name], context=f"recipe snapshot {name}", required={"sources", "excluded_streams"}
        )
        if not isinstance(snapshot["sources"], list) or not snapshot["sources"]:
            raise ValueError("Recipe snapshot sources must be a non-empty array.")
        sources = tuple(RecipeSource.from_dict(source, context=name) for source in snapshot["sources"])
        if normalize_recipe_sources(sources) != sources:
            raise ValueError("Recipe snapshot sources must be sorted and unique.")
        return sources, _require_mapping(snapshot["excluded_streams"], context=f"{name} exclusions")

    def to_dict(self) -> dict[str, object]:
        """Return the complete canonical JSON payload."""
        return {
            "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            "export_id": self.export_id,
            "created_at": self.created_at,
            "excluded_streams": self.excluded_streams,
            **({"recipe_snapshots": self.recipe_snapshots} if self.recipe_snapshots else {}),
            "recipe": {
                "name": self.recipe_name,
                "storage_type": self.recipe_storage_type,
                "data_type": self.recipe_data_type,
                "is_production_database": self.is_production_database,
                "wandb_run_path": self.wandb_run_path,
                "sources": [source.to_dict() for source in self.recipe_sources],
            },
            "conversation_table": {
                "path": self.conversation_table_path,
                "version": self.conversation_table_version,
                "row_count": self.conversation_row_count,
                "columns": list(self.conversation_columns),
            },
            "row_index_caches": [cache.to_dict() for cache in self.row_index_caches],
            "media": {
                "path": self.media_root_path,
                "shard_count": self.media_shard_count,
                "total_bytes": self.media_total_bytes,
                "inventory_sha256": self.media_inventory_sha256,
            },
            "input_identity": self.input_identity,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConversationTrainingExportManifest":
        """Parse and validate an export receipt without accessing its artifacts."""
        expected_keys = {
            "schema_version",
            "export_id",
            "created_at",
            "recipe",
            "conversation_table",
            "row_index_caches",
            "media",
            "input_identity",
            "git_commit",
        }
        payload = _require_mapping(
            value, context="export manifest", required=expected_keys, optional=("excluded_streams", "recipe_snapshots")
        )
        if payload["schema_version"] != EXPORT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported export manifest schema version {payload['schema_version']!r}; "
                f"expected {EXPORT_MANIFEST_SCHEMA_VERSION}."
            )
        recipe = _require_mapping(
            payload["recipe"],
            context="recipe",
            required={"name", "storage_type", "data_type", "wandb_run_path", "sources"},
            optional=("is_production_database",),
        )
        raw_sources = recipe["sources"]
        if not isinstance(raw_sources, list):
            raise ValueError("recipe.sources must be a JSON array.")
        table = _require_mapping(
            payload["conversation_table"],
            context="conversation_table",
            required={"path", "version", "row_count", "columns"},
        )
        raw_columns = table["columns"]
        if not isinstance(raw_columns, list) or not all(isinstance(column, str) for column in raw_columns):
            raise ValueError("conversation_table.columns must be an array of strings.")
        raw_caches = payload["row_index_caches"]
        if not isinstance(raw_caches, list):
            raise ValueError("row_index_caches must be a JSON array.")
        media = _require_mapping(
            payload["media"], context="media", required={"path", "shard_count", "total_bytes", "inventory_sha256"}
        )
        return cls(
            export_id=payload["export_id"],
            created_at=payload["created_at"],
            recipe_name=recipe["name"],
            recipe_storage_type=recipe["storage_type"],
            recipe_data_type=recipe["data_type"],
            wandb_run_path=recipe["wandb_run_path"],
            recipe_sources=tuple(
                RecipeSource.from_dict(source, context=f"recipe.sources[{index}]")
                for index, source in enumerate(raw_sources)
            ),
            conversation_table_path=table["path"],
            conversation_table_version=table["version"],
            conversation_row_count=table["row_count"],
            conversation_columns=tuple(raw_columns),
            row_index_caches=tuple(
                ExportedRowIndexCache.from_dict(cache, context=f"row_index_caches[{index}]")
                for index, cache in enumerate(raw_caches)
            ),
            media_root_path=media["path"],
            media_shard_count=media["shard_count"],
            media_total_bytes=media["total_bytes"],
            media_inventory_sha256=media["inventory_sha256"],
            input_identity=_require_mapping(payload["input_identity"], context="input_identity"),
            git_commit=payload["git_commit"],
            is_production_database=recipe.get("is_production_database", False),
            excluded_streams=payload.get("excluded_streams", {}),
            recipe_snapshots=_require_mapping(payload.get("recipe_snapshots", {}), context="recipe_snapshots"),
        )


def resolve_export_artifact(export_root: str | Path, relative_path: str) -> Path:
    """Resolve one manifest path while preventing traversal outside the bundle."""
    normalized = _require_relative_path(relative_path, context="export artifact path")
    root = Path(export_root).expanduser().resolve()
    resolved = (root / normalized).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Export artifact {relative_path!r} escapes root {str(root)!r}.")
    return resolved


def read_export_manifest(export_root: str | Path) -> ConversationTrainingExportManifest:
    """Read and validate ``export_manifest.json`` from a local export root."""
    manifest_path = resolve_export_artifact(export_root, EXPORT_MANIFEST_FILENAME)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Conversation training export has no receipt at {manifest_path}.") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Conversation training export receipt {manifest_path} is not valid JSON.") from error
    return ConversationTrainingExportManifest.from_dict(payload)


def publish_export_manifest(
    export_root: str | Path,
    manifest: ConversationTrainingExportManifest,
) -> bool:
    """Publish the receipt last, returning ``False`` for an identical replay."""
    root = Path(export_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / EXPORT_MANIFEST_FILENAME
    contents = canonical_json_bytes(manifest.to_dict())
    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        if existing == contents:
            return False
        raise FileExistsError(
            f"Export root {root} already contains a different {EXPORT_MANIFEST_FILENAME}; use a new root."
        )
    temporary_path = root / f".{EXPORT_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    try:
        with temporary_path.open("xb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True
