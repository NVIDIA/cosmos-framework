# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Immutable, sharded on-disk cache for frozen Reasoner K/V features.

The cache is deliberately provider-neutral and training-safe:

* tensors are stored layer-major in multi-record safetensors shards;
* the manifest is published only after every shard is complete;
* model/tokenizer/framing identities and per-record fingerprints fail closed;
* shard checksums are verified before the first read; and
* :class:`OfflineReasonerFeatureProvider` implements the same completed-Future
  interface used by future asynchronous/remote providers.

The incremental writer/finalizer supports rank-local extraction onto a shared
POSIX filesystem without owning dataset enumeration or process launch. Those
orchestration concerns belong in the extraction CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import asdict, dataclass
from errno import EEXIST, ENOTEMPTY
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
    compute_reasoner_feature_fingerprint,
)

REASONER_FEATURE_CACHE_SCHEMA_VERSION = 1
REASONER_FEATURE_CACHE_MANIFEST = "manifest.json"
_FORMAT_NAME = "cosmos3-reasoner-kv"
_OFFSETS_KEY = "record_offsets"
_INCREMENTAL_SCHEMA_VERSION = 1
_INCREMENTAL_STAGING_FILE = "staging.json"
_INCREMENTAL_RANK_COMPLETE_FILE = "rank.complete.json"
_INCREMENTAL_SIDECAR_SUFFIX = ".index.json"

_CacheSignature = tuple[int, int, int, str]


ReasonerFeatureCacheIdentity = ReasonerFeatureIdentity


@dataclass(frozen=True)
class ReasonerFeatureCacheEntry:
    """One cache record; ``features`` must describe exactly one sample."""

    sample_key: str
    features: ReasonerFeatureBatch

    def __post_init__(self) -> None:
        if not self.sample_key:
            raise ValueError("sample_key must be non-empty")
        if self.features.num_samples != 1:
            raise ValueError(
                f"Cache entries must contain exactly one sample, got {self.features.num_samples} for {self.sample_key!r}"
            )
        if len(self.features.fingerprints) != 1:
            raise ValueError(f"Cache entry {self.sample_key!r} must carry exactly one non-empty fingerprint")


@dataclass(frozen=True)
class _Record:
    sample_key: str
    start: int
    end: int
    fingerprint: str

    @property
    def num_tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _Shard:
    path: str
    sha256: str
    num_tokens: int
    records: tuple[_Record, ...]


@dataclass(frozen=True)
class _Manifest:
    cache_fingerprint: str
    identity: ReasonerFeatureCacheIdentity
    dtype: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    shards: tuple[_Shard, ...]


@dataclass(frozen=True)
class _IncrementalShard:
    """One committed rank-local shard and its durable index sidecar."""

    rank: int
    world_size: int
    shard_index: int
    signature: _CacheSignature
    cache_fingerprint: str
    shard: _Shard
    shard_path: Path
    sidecar_path: Path


def _layer_key(kind: str, layer_idx: int) -> str:
    return f"{kind}.{layer_idx:03d}"


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in {"bfloat16", "float16", "float32"}:
        raise TypeError(f"Unsupported Reasoner cache dtype: {dtype}")
    return name


def _manifest_fingerprint(
    identity: ReasonerFeatureCacheIdentity,
    *,
    dtype: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    payload = {
        "schema_version": REASONER_FEATURE_CACHE_SCHEMA_VERSION,
        "format": _FORMAT_NAME,
        "identity": asdict(identity),
        "dtype": dtype,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on the shared POSIX filesystem."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON commit marker through a same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_json_file(temporary_path, payload)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish a JSON file only if no peer has published it first.

    The hard-link commit is the shared-POSIX equivalent of ``O_EXCL`` while
    retaining the requested temp-file + atomic-publish discipline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_json_file(temporary_path, payload)
        try:
            os.link(temporary_path, path)
            created = True
        except FileExistsError:
            created = False
        if created:
            _fsync_directory(path.parent)
        return created
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _incremental_staging_root(cache_root: Path) -> Path:
    return cache_root.parent / f".{cache_root.name}.reasoner-kv-staging"


def _signature_to_dict(signature: _CacheSignature) -> dict[str, int | str]:
    num_layers, num_kv_heads, head_dim, dtype = signature
    return {
        "dtype": dtype,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _signature_from_dict(value: Any, *, field: str) -> _CacheSignature:
    raw = _require_dict(value, field=field)
    dtype = raw.get("dtype")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Incremental cache field {field!r} has unsupported dtype {dtype!r}")
    return (
        _require_positive_int(raw.get("num_layers"), field=f"{field}.num_layers"),
        _require_positive_int(raw.get("num_kv_heads"), field=f"{field}.num_kv_heads"),
        _require_positive_int(raw.get("head_dim"), field=f"{field}.head_dim"),
        dtype,
    )


def build_reasoner_feature_requests(
    packed_sequence: PackedSequence,
    sample_keys: Sequence[str],
    identity: ReasonerFeatureCacheIdentity,
) -> tuple[ReasonerFeatureRequest, ...]:
    """Build exact per-sample requests from the finalized CPU training pack.

    Producers must call the same sequence packer first and use this function
    instead of reproducing BOS/EOS/start-of-generation framing or mRoPE
    positions independently.
    """
    packed_sequence.prepare_sequence_pack_metadata()
    metadata = packed_sequence.get_sequence_pack_metadata()
    if metadata is None:
        raise RuntimeError("PackedSequence failed to prepare Reasoner attention metadata")
    offsets = metadata.causal_seq_offsets.detach().to(device="cpu", dtype=torch.int64)
    if len(sample_keys) != offsets.numel() - 1:
        raise ValueError(
            f"Reasoner sample keys ({len(sample_keys)}) do not match packed causal segments ({offsets.numel() - 1})"
        )

    text_ids = packed_sequence.text_ids.detach().to(device="cpu")
    text_indexes = packed_sequence.text_indexes.detach().to(device="cpu", dtype=torch.int64)
    position_ids = packed_sequence.position_ids.detach().to(device="cpu").index_select(-1, text_indexes)
    if text_ids.numel() != int(offsets[-1]) or position_ids.shape[-1] != text_ids.numel():
        raise ValueError(
            "External Reasoner conditioning requires every causal token to be a framed text token; "
            f"text={text_ids.numel()} positions={position_ids.shape[-1]} causal={int(offsets[-1])}"
        )

    requests: list[ReasonerFeatureRequest] = []
    for sample_idx, sample_key in enumerate(sample_keys):
        start = int(offsets[sample_idx])
        end = int(offsets[sample_idx + 1])
        token_slice = text_ids[start:end]
        position_slice = position_ids[..., start:end]
        sample_offsets = torch.tensor([0, end - start], dtype=torch.int64)
        fingerprint = compute_reasoner_feature_fingerprint(
            token_slice,
            position_slice,
            sample_offsets,
            identity=identity,
        )
        requests.append(
            ReasonerFeatureRequest(
                sample_key=str(sample_key),
                token_ids=token_slice,
                position_ids=position_slice,
                causal_offsets=sample_offsets,
                fingerprint=fingerprint,
            )
        )
    return tuple(requests)


def build_reasoner_feature_request_from_text_tokens(
    *,
    sample_key: str,
    text_ids: Sequence[int] | torch.Tensor,
    special_tokens: Mapping[str, int],
    use_float_positions: bool,
    identity: ReasonerFeatureCacheIdentity,
) -> ReasonerFeatureRequest:
    """Frame one ordinary SFT caption through the canonical sequence builder.

    The helper is for deterministic offline producers. It deliberately covers
    only the currently supported single-caption, non-AR SFT layout and always
    includes the start-of-generation token used by a video sample.
    """
    raw_text_ids = (
        text_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        if isinstance(text_ids, torch.Tensor)
        else list(text_ids)
    )
    builder = PackedSequenceBuilder()
    builder.begin_sample(initial_mrope_temporal_offset=0)
    split_len = builder.pack_text_tokens(
        raw_text_ids,
        dict(special_tokens),
        has_generation=True,
        use_float_positions=use_float_positions,
    )
    framed_ids = torch.tensor(builder.text_ids, dtype=torch.int64)
    if len(builder.position_ids) != 1:
        raise RuntimeError(f"Expected one framed text position block, got {len(builder.position_ids)}")
    position_ids = builder.position_ids[0]
    causal_offsets = torch.tensor([0, split_len], dtype=torch.int64)
    fingerprint = compute_reasoner_feature_fingerprint(
        framed_ids,
        position_ids,
        causal_offsets,
        identity=identity,
    )
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=framed_ids,
        position_ids=position_ids,
        causal_offsets=causal_offsets,
        fingerprint=fingerprint,
    )


def _entry_bytes(entry: ReasonerFeatureCacheEntry) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in (*entry.features.cross_k, *entry.features.cross_v))


def _partition_entries(
    entries: Sequence[ReasonerFeatureCacheEntry],
    max_shard_bytes: int,
) -> list[list[ReasonerFeatureCacheEntry]]:
    if max_shard_bytes <= 0:
        raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}")
    shards: list[list[ReasonerFeatureCacheEntry]] = []
    current: list[ReasonerFeatureCacheEntry] = []
    current_bytes = 0
    for entry in entries:
        size = _entry_bytes(entry)
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def _validate_entries(entries: Sequence[ReasonerFeatureCacheEntry]) -> _CacheSignature:
    if not entries:
        raise ValueError("At least one Reasoner feature cache entry is required")
    identities = [(entry.sample_key, entry.features.fingerprints[0]) for entry in entries]
    duplicates = sorted(identity for identity in set(identities) if identities.count(identity) > 1)
    if duplicates:
        raise ValueError(f"Duplicate Reasoner feature sample/fingerprint pairs: {duplicates}")

    first = entries[0].features
    first_layer = first.layer(0)
    signature = (
        first.num_layers,
        first_layer.num_kv_heads,
        first_layer.head_dim,
        _dtype_name(first_layer.cross_k.dtype),
    )
    for entry in entries[1:]:
        features = entry.features
        layer = features.layer(0)
        candidate = (
            features.num_layers,
            layer.num_kv_heads,
            layer.head_dim,
            _dtype_name(layer.cross_k.dtype),
        )
        if candidate != signature:
            raise ValueError(
                f"Reasoner cache entry {entry.sample_key!r} has signature {candidate}, expected {signature}"
            )
    return signature


def _write_shard(
    path: Path,
    entries: Sequence[ReasonerFeatureCacheEntry],
    *,
    cache_fingerprint: str,
) -> _Shard:
    offsets = [0]
    records: list[_Record] = []
    for entry in entries:
        start = offsets[-1]
        end = start + entry.features.sequence_length
        offsets.append(end)
        records.append(
            _Record(
                sample_key=entry.sample_key,
                start=start,
                end=end,
                fingerprint=entry.features.fingerprints[0],
            )
        )

    num_layers = entries[0].features.num_layers
    tensors: dict[str, torch.Tensor] = {
        _OFFSETS_KEY: torch.tensor(offsets, dtype=torch.int64),
    }
    for layer_idx in range(num_layers):
        tensors[_layer_key("cross_k", layer_idx)] = torch.cat(
            [entry.features.cross_k[layer_idx].detach().to(device="cpu").contiguous() for entry in entries],
            dim=0,
        )
        tensors[_layer_key("cross_v", layer_idx)] = torch.cat(
            [entry.features.cross_v[layer_idx].detach().to(device="cpu").contiguous() for entry in entries],
            dim=0,
        )

    save_file(
        tensors,
        str(path),
        metadata={
            "format": _FORMAT_NAME,
            "schema_version": str(REASONER_FEATURE_CACHE_SCHEMA_VERSION),
            "cache_fingerprint": cache_fingerprint,
        },
    )
    return _Shard(
        path=path.name,
        sha256=_sha256_file(path),
        num_tokens=offsets[-1],
        records=tuple(records),
    )


def write_reasoner_feature_cache(
    cache_root: str | Path,
    entries: Sequence[ReasonerFeatureCacheEntry],
    *,
    identity: ReasonerFeatureCacheIdentity,
    max_shard_bytes: int = 4 * 1024**3,
) -> Path:
    """Atomically publish a new immutable Reasoner feature cache.

    ``cache_root`` must not already exist. The complete cache is first written
    to a sibling temporary directory and becomes visible through one atomic
    rename, so readers never observe a partial manifest or shard set.
    """
    root = Path(cache_root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Reasoner feature cache: {root}")
    num_layers, num_kv_heads, head_dim, dtype = _validate_entries(entries)
    cache_fingerprint = _manifest_fingerprint(
        identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    entry_shards = _partition_entries(entries, max_shard_bytes)

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        shards: list[_Shard] = []
        for shard_idx, shard_entries in enumerate(entry_shards):
            shard_name = f"reasoner-kv-{shard_idx:05d}-of-{len(entry_shards):05d}.safetensors"
            shards.append(
                _write_shard(
                    temporary_root / shard_name,
                    shard_entries,
                    cache_fingerprint=cache_fingerprint,
                )
            )

        manifest = {
            "schema_version": REASONER_FEATURE_CACHE_SCHEMA_VERSION,
            "format": _FORMAT_NAME,
            "cache_fingerprint": cache_fingerprint,
            "identity": asdict(identity),
            "dtype": dtype,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "shards": [
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "num_tokens": shard.num_tokens,
                    "records": [asdict(record) for record in shard.records],
                }
                for shard in shards
            ],
        }
        manifest_tmp = temporary_root / f"{REASONER_FEATURE_CACHE_MANIFEST}.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, temporary_root / REASONER_FEATURE_CACHE_MANIFEST)
        os.replace(temporary_root, root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return root / REASONER_FEATURE_CACHE_MANIFEST


def _require_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {field!r} must be an object")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Manifest field {field!r} must be a positive integer")
    return value


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest field {field!r} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"Manifest field {field!r} must be a simple relative filename, got {value!r}")
    return value


def _load_manifest(cache_root: Path) -> _Manifest:
    path = cache_root / REASONER_FEATURE_CACHE_MANIFEST
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Reasoner feature cache manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Reasoner feature cache manifest JSON: {path}") from error
    raw = _require_dict(raw, field="root")
    if raw.get("schema_version") != REASONER_FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Reasoner cache schema_version={raw.get('schema_version')!r}; "
            f"expected {REASONER_FEATURE_CACHE_SCHEMA_VERSION}"
        )
    if raw.get("format") != _FORMAT_NAME:
        raise ValueError(f"Unexpected Reasoner cache format={raw.get('format')!r}")

    identity_dict = _require_dict(raw.get("identity"), field="identity")
    try:
        identity = ReasonerFeatureCacheIdentity(**identity_dict)
    except TypeError as error:
        raise ValueError(f"Invalid Reasoner cache identity fields: {sorted(identity_dict)}") from error
    dtype = raw.get("dtype")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Manifest field 'dtype' is unsupported: {dtype!r}")
    num_layers = _require_positive_int(raw.get("num_layers"), field="num_layers")
    num_kv_heads = _require_positive_int(raw.get("num_kv_heads"), field="num_kv_heads")
    head_dim = _require_positive_int(raw.get("head_dim"), field="head_dim")
    expected_cache_fingerprint = _manifest_fingerprint(
        identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if raw.get("cache_fingerprint") != expected_cache_fingerprint:
        raise ValueError("Reasoner cache manifest fingerprint does not match its identity/shape fields")

    raw_shards = raw.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Manifest field 'shards' must be a non-empty list")
    shards: list[_Shard] = []
    seen_records: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for shard_idx, raw_shard_value in enumerate(raw_shards):
        raw_shard = _require_dict(raw_shard_value, field=f"shards[{shard_idx}]")
        shard_path = _safe_relative_path(raw_shard.get("path"), field=f"shards[{shard_idx}].path")
        if shard_path in seen_paths:
            raise ValueError(f"Duplicate shard path in manifest: {shard_path!r}")
        seen_paths.add(shard_path)
        checksum = raw_shard.get("sha256")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"Invalid SHA256 for shard {shard_path!r}")
        num_tokens = _require_positive_int(raw_shard.get("num_tokens"), field=f"{shard_path}.num_tokens")
        raw_records = raw_shard.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"Shard {shard_path!r} must contain a non-empty records list")
        records: list[_Record] = []
        expected_start = 0
        for record_idx, raw_record_value in enumerate(raw_records):
            raw_record = _require_dict(raw_record_value, field=f"{shard_path}.records[{record_idx}]")
            sample_key = raw_record.get("sample_key")
            fingerprint = raw_record.get("fingerprint")
            start = raw_record.get("start")
            end = raw_record.get("end")
            if not isinstance(sample_key, str) or not sample_key:
                raise ValueError(f"Shard {shard_path!r} contains an invalid sample key")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(f"Cache record {sample_key!r} has an invalid fingerprint")
            record_identity = (sample_key, fingerprint)
            if record_identity in seen_records:
                raise ValueError(f"Duplicate Reasoner feature record in manifest: {record_identity!r}")
            if not isinstance(start, int) or not isinstance(end, int) or start != expected_start or end <= start:
                raise ValueError(
                    f"Cache record {sample_key!r} has invalid/non-contiguous range [{start}, {end}); "
                    f"expected start {expected_start}"
                )
            record = _Record(sample_key, start, end, fingerprint)
            records.append(record)
            seen_records.add(record_identity)
            expected_start = end
        if expected_start != num_tokens:
            raise ValueError(
                f"Shard {shard_path!r} record ranges end at {expected_start}, expected num_tokens={num_tokens}"
            )
        shards.append(_Shard(shard_path, checksum, num_tokens, tuple(records)))

    return _Manifest(
        cache_fingerprint=expected_cache_fingerprint,
        identity=identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        shards=tuple(shards),
    )


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{description} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description} JSON: {path}") from error
    return _require_dict(raw, field=description)


def _incremental_shard_name(rank: int, shard_index: int) -> str:
    return f"reasoner-kv-r{rank:05d}-s{shard_index:05d}.safetensors"


def _incremental_sidecar_name(rank: int, shard_index: int) -> str:
    return f"{_incremental_shard_name(rank, shard_index)}{_INCREMENTAL_SIDECAR_SUFFIX}"


def _incremental_staging_payload(
    identity: ReasonerFeatureCacheIdentity,
    *,
    world_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": _INCREMENTAL_SCHEMA_VERSION,
        "format": _FORMAT_NAME,
        "kind": "distributed-staging",
        "identity": asdict(identity),
        "world_size": world_size,
    }


def _validate_incremental_staging(
    staging_root: Path,
    *,
    expected_identity: ReasonerFeatureCacheIdentity,
    expected_world_size: int,
) -> None:
    raw = _read_json_object(staging_root / _INCREMENTAL_STAGING_FILE, description="incremental staging metadata")
    if raw.get("schema_version") != _INCREMENTAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported incremental cache schema_version={raw.get('schema_version')!r}")
    if raw.get("format") != _FORMAT_NAME or raw.get("kind") != "distributed-staging":
        raise ValueError("Incremental staging metadata has an unexpected format or kind")
    identity_dict = _require_dict(raw.get("identity"), field="incremental identity")
    try:
        identity = ReasonerFeatureCacheIdentity(**identity_dict)
    except TypeError as error:
        raise ValueError(f"Invalid incremental cache identity fields: {sorted(identity_dict)}") from error
    if identity != expected_identity:
        raise ValueError(f"Incremental cache identity mismatch: staging={identity!r}, expected={expected_identity!r}")
    world_size = _require_positive_int(raw.get("world_size"), field="incremental world_size")
    if world_size != expected_world_size:
        raise ValueError(f"Incremental cache world-size mismatch: staging={world_size}, expected={expected_world_size}")


def _ensure_incremental_staging(
    cache_root: Path,
    *,
    identity: ReasonerFeatureCacheIdentity,
    world_size: int,
) -> Path:
    if cache_root.exists():
        raise FileExistsError(f"Reasoner feature cache is already published and immutable: {cache_root}")
    staging_root = _incremental_staging_root(cache_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(staging_root.parent)
    payload = _incremental_staging_payload(identity, world_size=world_size)
    _atomic_create_json(staging_root / _INCREMENTAL_STAGING_FILE, payload)
    _validate_incremental_staging(
        staging_root,
        expected_identity=identity,
        expected_world_size=world_size,
    )
    return staging_root


def _parse_incremental_records(
    value: Any,
    *,
    shard_name: str,
    num_tokens: int,
) -> tuple[_Record, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Incremental shard {shard_name!r} must contain a non-empty records list")
    records: list[_Record] = []
    expected_start = 0
    seen: set[tuple[str, str]] = set()
    for record_idx, raw_record_value in enumerate(value):
        raw_record = _require_dict(raw_record_value, field=f"{shard_name}.records[{record_idx}]")
        sample_key = raw_record.get("sample_key")
        fingerprint = raw_record.get("fingerprint")
        start = raw_record.get("start")
        end = raw_record.get("end")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError(f"Incremental shard {shard_name!r} contains an invalid sample key")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"Incremental cache record {sample_key!r} has an invalid fingerprint")
        identity = (sample_key, fingerprint)
        if identity in seen:
            raise ValueError(f"Duplicate Reasoner feature record inside shard {shard_name!r}: {identity!r}")
        if not isinstance(start, int) or not isinstance(end, int) or start != expected_start or end <= start:
            raise ValueError(
                f"Incremental record {sample_key!r} has invalid/non-contiguous range [{start}, {end}); "
                f"expected start {expected_start}"
            )
        records.append(_Record(sample_key, start, end, fingerprint))
        seen.add(identity)
        expected_start = end
    if expected_start != num_tokens:
        raise ValueError(
            f"Incremental shard {shard_name!r} record ranges end at {expected_start}, expected num_tokens={num_tokens}"
        )
    return tuple(records)


def _validate_incremental_shard_payload(
    path: Path,
    *,
    shard: _Shard,
    signature: _CacheSignature,
    cache_fingerprint: str,
) -> None:
    num_layers, num_kv_heads, head_dim, dtype = signature
    safe_dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != _FORMAT_NAME:
            raise ValueError(f"Incremental cache shard {path} has invalid format metadata")
        if metadata.get("schema_version") != str(REASONER_FEATURE_CACHE_SCHEMA_VERSION):
            raise ValueError(f"Incremental cache shard {path} has invalid schema metadata")
        if metadata.get("cache_fingerprint") != cache_fingerprint:
            raise ValueError(f"Incremental cache shard {path} belongs to a different cache identity/signature")

        expected_keys = {_OFFSETS_KEY}
        for layer_idx in range(num_layers):
            expected_keys.add(_layer_key("cross_k", layer_idx))
            expected_keys.add(_layer_key("cross_v", layer_idx))
        actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Incremental cache shard {path} tensor keys disagree with its sidecar: "
                f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}"
            )

        offsets = handle.get_tensor(_OFFSETS_KEY)
        expected_offsets = torch.tensor([0, *(record.end for record in shard.records)], dtype=torch.int64)
        if not torch.equal(offsets, expected_offsets):
            raise ValueError(f"Incremental cache shard {path} offsets disagree with its sidecar")
        expected_shape = [shard.num_tokens, num_kv_heads, head_dim]
        for layer_idx in range(num_layers):
            for kind in ("cross_k", "cross_v"):
                tensor_slice = handle.get_slice(_layer_key(kind, layer_idx))
                if tensor_slice.get_shape() != expected_shape:
                    raise ValueError(
                        f"Incremental cache shard {path} tensor {_layer_key(kind, layer_idx)!r} has shape "
                        f"{tensor_slice.get_shape()}, expected {expected_shape}"
                    )
                if tensor_slice.get_dtype() != safe_dtype:
                    raise ValueError(
                        f"Incremental cache shard {path} tensor {_layer_key(kind, layer_idx)!r} has dtype "
                        f"{tensor_slice.get_dtype()}, expected {safe_dtype}"
                    )


def _load_incremental_shard(
    sidecar_path: Path,
    *,
    identity: ReasonerFeatureCacheIdentity,
    expected_rank: int,
    expected_world_size: int,
    verify_checksum: bool,
) -> _IncrementalShard:
    raw = _read_json_object(sidecar_path, description="incremental shard sidecar")
    if raw.get("schema_version") != _INCREMENTAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported incremental sidecar schema_version={raw.get('schema_version')!r}")
    if raw.get("format") != _FORMAT_NAME or raw.get("kind") != "rank-shard":
        raise ValueError(f"Incremental sidecar {sidecar_path} has an unexpected format or kind")
    rank = raw.get("rank")
    world_size = raw.get("world_size")
    shard_index = raw.get("shard_index")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank != expected_rank:
        raise ValueError(f"Incremental sidecar {sidecar_path} has rank={rank!r}, expected {expected_rank}")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size != expected_world_size:
        raise ValueError(
            f"Incremental sidecar {sidecar_path} has world_size={world_size!r}, expected {expected_world_size}"
        )
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise ValueError(f"Incremental sidecar {sidecar_path} has invalid shard_index={shard_index!r}")
    expected_sidecar_name = _incremental_sidecar_name(rank, shard_index)
    if sidecar_path.name != expected_sidecar_name:
        raise ValueError(
            f"Incremental sidecar name {sidecar_path.name!r} does not match rank/index {expected_sidecar_name!r}"
        )

    signature = _signature_from_dict(raw.get("signature"), field=f"{sidecar_path.name}.signature")
    num_layers, num_kv_heads, head_dim, dtype = signature
    expected_cache_fingerprint = _manifest_fingerprint(
        identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if raw.get("cache_fingerprint") != expected_cache_fingerprint:
        raise ValueError(f"Incremental sidecar {sidecar_path} has a stale cache fingerprint")

    raw_shard = _require_dict(raw.get("shard"), field=f"{sidecar_path.name}.shard")
    shard_name = _safe_relative_path(raw_shard.get("path"), field=f"{sidecar_path.name}.shard.path")
    expected_shard_name = _incremental_shard_name(rank, shard_index)
    if shard_name != expected_shard_name:
        raise ValueError(f"Incremental sidecar points to {shard_name!r}, expected {expected_shard_name!r}")
    checksum = raw_shard.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError(f"Invalid SHA256 for incremental shard {shard_name!r}")
    num_tokens = _require_positive_int(raw_shard.get("num_tokens"), field=f"{shard_name}.num_tokens")
    records = _parse_incremental_records(
        raw_shard.get("records"),
        shard_name=shard_name,
        num_tokens=num_tokens,
    )
    shard = _Shard(shard_name, checksum, num_tokens, records)
    shard_path = sidecar_path.parent / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"Committed incremental cache shard not found: {shard_path}")
    if verify_checksum:
        actual_checksum = _sha256_file(shard_path)
        if actual_checksum != checksum:
            raise ValueError(
                f"Incremental cache shard checksum mismatch for {shard_path}: "
                f"expected {checksum}, got {actual_checksum}"
            )
    _validate_incremental_shard_payload(
        shard_path,
        shard=shard,
        signature=signature,
        cache_fingerprint=expected_cache_fingerprint,
    )
    return _IncrementalShard(
        rank=rank,
        world_size=world_size,
        shard_index=shard_index,
        signature=signature,
        cache_fingerprint=expected_cache_fingerprint,
        shard=shard,
        shard_path=shard_path,
        sidecar_path=sidecar_path,
    )


def _scan_incremental_rank(
    rank_dir: Path,
    *,
    identity: ReasonerFeatureCacheIdentity,
    rank: int,
    world_size: int,
    verify_checksums: bool,
) -> list[_IncrementalShard]:
    sidecars = sorted(rank_dir.glob(f"*{_INCREMENTAL_SIDECAR_SUFFIX}"))
    shards = [
        _load_incremental_shard(
            sidecar,
            identity=identity,
            expected_rank=rank,
            expected_world_size=world_size,
            verify_checksum=verify_checksums,
        )
        for sidecar in sidecars
    ]
    shards.sort(key=lambda item: item.shard_index)
    indices = [item.shard_index for item in shards]
    if indices != list(range(len(shards))):
        raise ValueError(f"Rank {rank} incremental shard indices must be contiguous from zero, got {indices}")
    signatures = {item.signature for item in shards}
    if len(signatures) > 1:
        raise ValueError(f"Rank {rank} incremental shards have inconsistent signatures: {sorted(signatures)}")
    seen: set[tuple[str, str]] = set()
    for item in shards:
        for record in item.shard.records:
            record_identity = (record.sample_key, record.fingerprint)
            if record_identity in seen:
                raise ValueError(f"Rank {rank} contains duplicate incremental record {record_identity!r}")
            seen.add(record_identity)
    return shards


def _rank_completion_payload(
    *,
    rank: int,
    world_size: int,
    identity: ReasonerFeatureCacheIdentity,
    signature: _CacheSignature | None,
    shards: Sequence[_IncrementalShard],
) -> dict[str, Any]:
    return {
        "schema_version": _INCREMENTAL_SCHEMA_VERSION,
        "format": _FORMAT_NAME,
        "kind": "rank-complete",
        "rank": rank,
        "world_size": world_size,
        "identity": asdict(identity),
        "signature": None if signature is None else _signature_to_dict(signature),
        "num_records": sum(len(item.shard.records) for item in shards),
        "num_tokens": sum(item.shard.num_tokens for item in shards),
        "sidecars": [
            {
                "path": item.sidecar_path.name,
                "sha256": _sha256_file(item.sidecar_path),
            }
            for item in shards
        ],
    }


def _validate_rank_completion(
    rank_dir: Path,
    *,
    identity: ReasonerFeatureCacheIdentity,
    rank: int,
    world_size: int,
    shards: Sequence[_IncrementalShard],
) -> None:
    completion_path = rank_dir / _INCREMENTAL_RANK_COMPLETE_FILE
    raw = _read_json_object(completion_path, description=f"rank {rank} completion marker")
    if raw.get("schema_version") != _INCREMENTAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rank completion schema_version={raw.get('schema_version')!r}")
    if raw.get("format") != _FORMAT_NAME or raw.get("kind") != "rank-complete":
        raise ValueError(f"Rank {rank} completion marker has an unexpected format or kind")
    completed_rank = raw.get("rank")
    completed_world_size = raw.get("world_size")
    if (
        not isinstance(completed_rank, int)
        or isinstance(completed_rank, bool)
        or completed_rank != rank
        or not isinstance(completed_world_size, int)
        or isinstance(completed_world_size, bool)
        or completed_world_size != world_size
    ):
        raise ValueError(
            f"Rank completion ownership mismatch: rank/world={completed_rank!r}/{completed_world_size!r}, "
            f"expected {rank}/{world_size}"
        )
    identity_dict = _require_dict(raw.get("identity"), field=f"rank {rank} completion identity")
    try:
        completed_identity = ReasonerFeatureCacheIdentity(**identity_dict)
    except TypeError as error:
        raise ValueError(f"Invalid rank {rank} completion identity fields: {sorted(identity_dict)}") from error
    if completed_identity != identity:
        raise ValueError(f"Rank {rank} completion identity does not match the staging identity")

    signatures = {item.signature for item in shards}
    expected_signature = next(iter(signatures)) if signatures else None
    raw_signature = raw.get("signature")
    actual_signature = (
        None if raw_signature is None else _signature_from_dict(raw_signature, field=f"rank {rank} signature")
    )
    if actual_signature != expected_signature:
        raise ValueError(
            f"Rank {rank} completion signature {actual_signature!r} does not match shards {expected_signature!r}"
        )

    expected_num_records = sum(len(item.shard.records) for item in shards)
    expected_num_tokens = sum(item.shard.num_tokens for item in shards)
    if raw.get("num_records") != expected_num_records or raw.get("num_tokens") != expected_num_tokens:
        raise ValueError(f"Rank {rank} completion counts do not match its committed shards")
    raw_sidecars = raw.get("sidecars")
    if not isinstance(raw_sidecars, list):
        raise ValueError(f"Rank {rank} completion sidecars must be a list")
    expected_sidecars = [{"path": item.sidecar_path.name, "sha256": _sha256_file(item.sidecar_path)} for item in shards]
    if raw_sidecars != expected_sidecars:
        raise ValueError(f"Rank {rank} completion marker does not match its committed sidecar set/checksums")


def _entry_on_cpu(entry: ReasonerFeatureCacheEntry) -> ReasonerFeatureCacheEntry:
    features = entry.features
    return ReasonerFeatureCacheEntry(
        sample_key=entry.sample_key,
        features=ReasonerFeatureBatch(
            cross_k=tuple(tensor.detach().to(device="cpu").contiguous() for tensor in features.cross_k),
            cross_v=tuple(tensor.detach().to(device="cpu").contiguous() for tensor in features.cross_v),
            causal_offsets=features.causal_offsets.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            fingerprints=features.fingerprints,
        ),
    )


class IncrementalReasonerFeatureCacheWriter:
    """Bounded-memory, resumable writer for one rank on a shared POSIX filesystem.

    A sidecar is the commit record for exactly one rank-local safetensors shard.
    Construction scans committed sidecars, validates their shards, and rebuilds
    the ``(sample_key, fingerprint)`` skip index. Uncommitted temporary files or
    orphan shard files are ignored and can be atomically replaced on retry.

    Only one live process may own a given ``rank``. Different ranks never write
    the same pathname and may append/flush concurrently.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        identity: ReasonerFeatureCacheIdentity,
        rank: int,
        world_size: int,
        max_shard_bytes: int = 4 * 1024**3,
        verify_checksums_on_resume: bool = True,
    ) -> None:
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
        if not isinstance(max_shard_bytes, int) or isinstance(max_shard_bytes, bool) or max_shard_bytes <= 0:
            raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}")
        self.cache_root = Path(cache_root)
        self.identity = identity
        self.rank = rank
        self.world_size = world_size
        self.max_shard_bytes = max_shard_bytes
        self.staging_root = _ensure_incremental_staging(
            self.cache_root,
            identity=identity,
            world_size=world_size,
        )
        self.rank_dir = self.staging_root / f"rank-{rank:05d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.staging_root)
        self._shards = _scan_incremental_rank(
            self.rank_dir,
            identity=identity,
            rank=rank,
            world_size=world_size,
            verify_checksums=verify_checksums_on_resume,
        )
        signatures = {item.signature for item in self._shards}
        self._signature: _CacheSignature | None = next(iter(signatures)) if signatures else None
        self._seen: set[tuple[str, str]] = {
            (record.sample_key, record.fingerprint) for item in self._shards for record in item.shard.records
        }
        self._buffer: list[ReasonerFeatureCacheEntry] = []
        self._buffered_bytes = 0
        completion_path = self.rank_dir / _INCREMENTAL_RANK_COMPLETE_FILE
        self._finalized = completion_path.exists()
        if self._finalized:
            _validate_rank_completion(
                self.rank_dir,
                identity=self.identity,
                rank=self.rank,
                world_size=self.world_size,
                shards=self._shards,
            )

    @classmethod
    def resume(
        cls,
        cache_root: str | Path,
        *,
        identity: ReasonerFeatureCacheIdentity,
        rank: int,
        world_size: int,
        max_shard_bytes: int = 4 * 1024**3,
        verify_checksums: bool = True,
    ) -> IncrementalReasonerFeatureCacheWriter:
        """Open a new or interrupted rank-local writer and validate its commits."""
        return cls(
            cache_root,
            identity=identity,
            rank=rank,
            world_size=world_size,
            max_shard_bytes=max_shard_bytes,
            verify_checksums_on_resume=verify_checksums,
        )

    @property
    def buffered_bytes(self) -> int:
        return self._buffered_bytes

    @property
    def committed_records(self) -> int:
        return len(self._seen) - len(self._buffer)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def contains(self, sample_key: str, fingerprint: str) -> bool:
        """Return whether an exact record is committed or buffered.

        Extraction loops should call this before running the frozen Reasoner so
        a resumed job does not recompute already committed features.
        """
        return (sample_key, fingerprint) in self._seen

    def append(self, entry: ReasonerFeatureCacheEntry) -> bool:
        """Buffer one entry; return ``False`` when its exact identity was committed/buffered."""
        record_identity = (entry.sample_key, entry.features.fingerprints[0])
        if record_identity in self._seen:
            return False
        if self._finalized:
            raise RuntimeError(f"Rank {self.rank} Reasoner cache writer has already been finalized")

        signature = _validate_entries([entry])
        if self._signature is not None and signature != self._signature:
            raise ValueError(
                f"Rank {self.rank} Reasoner cache entry has signature {signature}, expected {self._signature}"
            )
        entry = _entry_on_cpu(entry)
        if self._signature is None:
            self._signature = signature
        entry_bytes = _entry_bytes(entry)
        if self._buffer and self._buffered_bytes + entry_bytes > self.max_shard_bytes:
            self.flush()
        self._buffer.append(entry)
        self._buffered_bytes += entry_bytes
        self._seen.add(record_identity)
        # A single record may exceed the target; commit it immediately so the
        # retained feature payload never grows beyond that unavoidable record.
        if self._buffered_bytes >= self.max_shard_bytes:
            self.flush()
        return True

    def flush(self) -> Path | None:
        """Atomically commit the buffered entries as one shard + index sidecar."""
        if not self._buffer:
            return None
        if self._finalized:
            raise RuntimeError(f"Rank {self.rank} Reasoner cache writer has already been finalized")
        assert self._signature is not None
        num_layers, num_kv_heads, head_dim, dtype = self._signature
        cache_fingerprint = _manifest_fingerprint(
            self.identity,
            dtype=dtype,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        shard_index = len(self._shards)
        shard_name = _incremental_shard_name(self.rank, shard_index)
        sidecar_name = _incremental_sidecar_name(self.rank, shard_index)
        shard_path = self.rank_dir / shard_name
        sidecar_path = self.rank_dir / sidecar_name

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{shard_name}.tmp-", dir=self.rank_dir)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            temporary_shard = _write_shard(
                temporary_path,
                self._buffer,
                cache_fingerprint=cache_fingerprint,
            )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, shard_path)
            _fsync_directory(self.rank_dir)
        except BaseException:
            with suppress(FileNotFoundError):
                temporary_path.unlink()
            raise

        shard = _Shard(
            path=shard_name,
            sha256=temporary_shard.sha256,
            num_tokens=temporary_shard.num_tokens,
            records=temporary_shard.records,
        )
        sidecar_payload = {
            "schema_version": _INCREMENTAL_SCHEMA_VERSION,
            "format": _FORMAT_NAME,
            "kind": "rank-shard",
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_index": shard_index,
            "signature": _signature_to_dict(self._signature),
            "cache_fingerprint": cache_fingerprint,
            "shard": {
                "path": shard.path,
                "sha256": shard.sha256,
                "num_tokens": shard.num_tokens,
                "records": [asdict(record) for record in shard.records],
            },
        }
        _atomic_write_json(sidecar_path, sidecar_payload)
        committed = _IncrementalShard(
            rank=self.rank,
            world_size=self.world_size,
            shard_index=shard_index,
            signature=self._signature,
            cache_fingerprint=cache_fingerprint,
            shard=shard,
            shard_path=shard_path,
            sidecar_path=sidecar_path,
        )
        self._shards.append(committed)
        self._buffer.clear()
        self._buffered_bytes = 0
        return shard_path

    def finalize(self) -> Path:
        """Flush and atomically mark this rank complete; safe to call repeatedly."""
        if self._finalized:
            _validate_rank_completion(
                self.rank_dir,
                identity=self.identity,
                rank=self.rank,
                world_size=self.world_size,
                shards=self._shards,
            )
            return self.rank_dir / _INCREMENTAL_RANK_COMPLETE_FILE
        self.flush()
        payload = _rank_completion_payload(
            rank=self.rank,
            world_size=self.world_size,
            identity=self.identity,
            signature=self._signature,
            shards=self._shards,
        )
        completion_path = self.rank_dir / _INCREMENTAL_RANK_COMPLETE_FILE
        _atomic_write_json(completion_path, payload)
        self._finalized = True
        return completion_path


def _manifest_from_incremental_shards(
    shards: Sequence[_IncrementalShard],
    *,
    identity: ReasonerFeatureCacheIdentity,
) -> _Manifest:
    if not shards:
        raise ValueError("Cannot finalize an empty distributed Reasoner feature cache")
    signatures = {item.signature for item in shards}
    if len(signatures) != 1:
        raise ValueError(f"Distributed Reasoner cache ranks have inconsistent signatures: {sorted(signatures)}")
    num_layers, num_kv_heads, head_dim, dtype = next(iter(signatures))
    cache_fingerprint = _manifest_fingerprint(
        identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    seen: dict[tuple[str, str], tuple[int, int]] = {}
    for item in shards:
        if item.cache_fingerprint != cache_fingerprint:
            raise ValueError(f"Rank {item.rank} shard {item.shard_index} has an inconsistent cache fingerprint")
        for record in item.shard.records:
            record_identity = (record.sample_key, record.fingerprint)
            previous = seen.get(record_identity)
            if previous is not None:
                raise ValueError(
                    f"Duplicate Reasoner feature record across ranks: {record_identity!r} appears in "
                    f"rank/shard {previous[0]}/{previous[1]} and {item.rank}/{item.shard_index}"
                )
            seen[record_identity] = (item.rank, item.shard_index)
    return _Manifest(
        cache_fingerprint=cache_fingerprint,
        identity=identity,
        dtype=dtype,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        shards=tuple(item.shard for item in shards),
    )


def _manifest_payload(manifest: _Manifest) -> dict[str, Any]:
    return {
        "schema_version": REASONER_FEATURE_CACHE_SCHEMA_VERSION,
        "format": _FORMAT_NAME,
        "cache_fingerprint": manifest.cache_fingerprint,
        "identity": asdict(manifest.identity),
        "dtype": manifest.dtype,
        "num_layers": manifest.num_layers,
        "num_kv_heads": manifest.num_kv_heads,
        "head_dim": manifest.head_dim,
        "shards": [
            {
                "path": shard.path,
                "sha256": shard.sha256,
                "num_tokens": shard.num_tokens,
                "records": [asdict(record) for record in shard.records],
            }
            for shard in manifest.shards
        ],
    }


def finalize_incremental_reasoner_feature_cache(
    cache_root: str | Path,
    *,
    identity: ReasonerFeatureCacheIdentity,
    world_size: int,
    verify_checksums: bool = True,
) -> Path:
    """Publish all completed rank-local shards as one immutable provider cache.

    This MVP assumes every rank writes to the same POSIX filesystem. The caller
    is responsible for invoking this function on one coordinator after rank
    writers have called :meth:`IncrementalReasonerFeatureCacheWriter.finalize`.
    Concurrent coordinators are harmless: a single directory rename wins and a
    loser accepts the winner only when the complete manifest is identical.
    """
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    root = Path(cache_root)
    staging_root = _incremental_staging_root(root)
    _validate_incremental_staging(
        staging_root,
        expected_identity=identity,
        expected_world_size=world_size,
    )

    incremental_shards: list[_IncrementalShard] = []
    for rank in range(world_size):
        rank_dir = staging_root / f"rank-{rank:05d}"
        if not rank_dir.is_dir():
            raise FileNotFoundError(f"Incremental Reasoner cache rank directory not found: {rank_dir}")
        shards = _scan_incremental_rank(
            rank_dir,
            identity=identity,
            rank=rank,
            world_size=world_size,
            verify_checksums=verify_checksums,
        )
        _validate_rank_completion(
            rank_dir,
            identity=identity,
            rank=rank,
            world_size=world_size,
            shards=shards,
        )
        incremental_shards.extend(shards)
    incremental_shards.sort(key=lambda item: (item.rank, item.shard_index))
    expected_manifest = _manifest_from_incremental_shards(incremental_shards, identity=identity)

    if root.exists():
        published = _load_manifest(root)
        if published != expected_manifest:
            raise FileExistsError(f"Published cache {root} does not match the completed incremental staging data")
        return root / REASONER_FEATURE_CACHE_MANIFEST

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.publish-", dir=root.parent))
    try:
        for item in incremental_shards:
            destination = temporary_root / item.shard.path
            # Do not hard-link staging into the immutable publication: staging
            # is intentionally retained for audit/retry, and a later accidental
            # write through that pathname must not mutate the published cache.
            shutil.copy2(item.shard_path, destination)
            with destination.open("rb") as handle:
                os.fsync(handle.fileno())
            if _sha256_file(destination) != item.shard.sha256:
                raise ValueError(f"Published shard copy checksum mismatch: {destination}")
        _atomic_write_json(
            temporary_root / REASONER_FEATURE_CACHE_MANIFEST,
            _manifest_payload(expected_manifest),
        )
        _fsync_directory(temporary_root)
        try:
            os.replace(temporary_root, root)
        except OSError as error:
            if error.errno not in {EEXIST, ENOTEMPTY} or not root.exists():
                raise
            published = _load_manifest(root)
            if published != expected_manifest:
                raise FileExistsError(
                    f"Concurrent finalizer published a different Reasoner feature cache at {root}"
                ) from error
        _fsync_directory(root.parent)
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    if temporary_root.exists():
        shutil.rmtree(temporary_root, ignore_errors=True)
    return root / REASONER_FEATURE_CACHE_MANIFEST


class OfflineReasonerFeatureProvider:
    """Read immutable local shards and return completed feature futures."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_identity: ReasonerFeatureCacheIdentity | None,
        expected_dtype: torch.dtype | None = None,
        strict_fingerprint: bool = True,
        verify_checksums: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest = _load_manifest(self.cache_root)
        if strict_fingerprint and expected_identity is None:
            raise ValueError("strict_fingerprint=True requires an expected cache identity")
        if expected_identity is not None and expected_identity != self.manifest.identity:
            raise ValueError(
                f"Reasoner cache identity mismatch: manifest={self.manifest.identity!r}, expected={expected_identity!r}"
            )
        if expected_dtype is not None:
            dtype_name = _dtype_name(expected_dtype)
            if dtype_name != self.manifest.dtype:
                raise ValueError(
                    f"Reasoner cache dtype mismatch: manifest={self.manifest.dtype!r}, expected={dtype_name!r}"
                )
        self.strict_fingerprint = strict_fingerprint
        self.verify_checksums = verify_checksums
        self._verified_shards: set[str] = set()
        self._records: dict[tuple[str, str], tuple[_Shard, _Record]] = {
            (record.sample_key, record.fingerprint): (shard, record)
            for shard in self.manifest.shards
            for record in shard.records
        }
        self._records_by_key: dict[str, list[tuple[_Shard, _Record]]] = defaultdict(list)
        self._records_by_fingerprint: dict[str, list[tuple[_Shard, _Record]]] = defaultdict(list)
        for shard in self.manifest.shards:
            for record in shard.records:
                self._records_by_key[record.sample_key].append((shard, record))
                self._records_by_fingerprint[record.fingerprint].append((shard, record))

    @property
    def cache_fingerprint(self) -> str:
        return self.manifest.cache_fingerprint

    @property
    def signature(self) -> ReasonerFeatureSignature:
        return ReasonerFeatureSignature(
            num_layers=self.manifest.num_layers,
            num_kv_heads=self.manifest.num_kv_heads,
            head_dim=self.manifest.head_dim,
            dtype=getattr(torch, self.manifest.dtype),
        )

    def submit(self, requests: Sequence[ReasonerFeatureRequest]) -> Future[ReasonerFeatureBatch]:
        future: Future[ReasonerFeatureBatch] = Future()
        try:
            future.set_result(self.load(requests))
        except BaseException as error:
            future.set_exception(error)
        return future

    def close(self) -> None:
        """Match the provider lifecycle contract; immutable local caches own no live resources."""

    def _verify_shard(self, shard: _Shard) -> Path:
        path = self.cache_root / shard.path
        if not path.is_file():
            raise FileNotFoundError(f"Reasoner cache shard not found: {path}")
        if self.verify_checksums and shard.path not in self._verified_shards:
            actual = _sha256_file(path)
            if actual != shard.sha256:
                raise ValueError(
                    f"Reasoner cache shard checksum mismatch for {path}: expected {shard.sha256}, got {actual}"
                )
            self._verified_shards.add(shard.path)
        return path

    def verify_all_shards(self) -> None:
        """Stream-validate every published shard without loading feature payloads."""
        signature: _CacheSignature = (
            self.manifest.num_layers,
            self.manifest.num_kv_heads,
            self.manifest.head_dim,
            self.manifest.dtype,
        )
        for shard in self.manifest.shards:
            path = self._verify_shard(shard)
            _validate_incremental_shard_payload(
                path,
                shard=shard,
                signature=signature,
                cache_fingerprint=self.manifest.cache_fingerprint,
            )

    def _validate_request(self, request: ReasonerFeatureRequest, record: _Record) -> None:
        if request.causal_offsets.numel() != 2:
            raise ValueError(
                f"Offline cache request {request.sample_key!r} must describe one causal document, "
                f"got {request.causal_offsets.numel() - 1}"
            )
        if request.token_ids.numel() != record.num_tokens:
            raise ValueError(
                f"Offline cache token count mismatch for {request.sample_key!r}: "
                f"request={request.token_ids.numel()} cache={record.num_tokens}"
            )
        if self.strict_fingerprint and request.fingerprint != record.fingerprint:
            raise ValueError(
                f"Offline cache fingerprint mismatch for {request.sample_key!r}: "
                f"request={request.fingerprint!r} cache={record.fingerprint!r}"
            )

    def load(self, requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
        if not requests:
            raise ValueError("At least one Reasoner feature request is required")

        resolved: list[tuple[ReasonerFeatureRequest, _Shard, _Record]] = []
        for request in requests:
            if self.strict_fingerprint:
                match = self._records.get((request.sample_key, request.fingerprint))
                # Content-addressed fallback deduplicates shared prompts such as
                # CFG's null caption across otherwise unrelated sample keys.
                if match is None:
                    fingerprint_matches = self._records_by_fingerprint.get(request.fingerprint, [])
                    match = fingerprint_matches[0] if fingerprint_matches else None
                if match is None:
                    if request.sample_key in self._records_by_key:
                        available = sorted(
                            record.fingerprint for _shard, record in self._records_by_key[request.sample_key]
                        )
                        raise ValueError(
                            f"Offline cache fingerprint mismatch for {request.sample_key!r}: "
                            f"request={request.fingerprint!r}, available={available}"
                        )
                    raise KeyError(f"Reasoner feature cache miss for sample {request.sample_key!r}")
                shard, record = match
            else:
                matches = self._records_by_key.get(request.sample_key, [])
                if not matches:
                    raise KeyError(f"Reasoner feature cache miss for sample {request.sample_key!r}")
                if len(matches) != 1:
                    raise ValueError(
                        f"Non-strict cache lookup for {request.sample_key!r} is ambiguous across {len(matches)} records"
                    )
                shard, record = matches[0]
            self._validate_request(request, record)
            resolved.append((request, shard, record))

        by_shard: dict[str, list[tuple[int, _Record]]] = defaultdict(list)
        shard_by_path: dict[str, _Shard] = {}
        for output_idx, (_request, shard, record) in enumerate(resolved):
            by_shard[shard.path].append((output_idx, record))
            shard_by_path[shard.path] = shard

        layer_k: list[list[torch.Tensor | None]] = [[None] * len(requests) for _ in range(self.manifest.num_layers)]
        layer_v: list[list[torch.Tensor | None]] = [[None] * len(requests) for _ in range(self.manifest.num_layers)]
        for shard_path, indexed_records in by_shard.items():
            shard = shard_by_path[shard_path]
            path = self._verify_shard(shard)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("format") != _FORMAT_NAME:
                    raise ValueError(f"Reasoner cache shard {path} has invalid format metadata")
                if metadata.get("cache_fingerprint") != self.manifest.cache_fingerprint:
                    raise ValueError(f"Reasoner cache shard {path} belongs to a different manifest")
                offsets = handle.get_tensor(_OFFSETS_KEY)
                expected_offsets = torch.tensor(
                    [0, *(record.end for record in shard.records)],
                    dtype=torch.int64,
                )
                if not torch.equal(offsets, expected_offsets):
                    raise ValueError(f"Reasoner cache shard {path} offsets disagree with the manifest")
                for layer_idx in range(self.manifest.num_layers):
                    k_slice = handle.get_slice(_layer_key("cross_k", layer_idx))
                    v_slice = handle.get_slice(_layer_key("cross_v", layer_idx))
                    for output_idx, record in indexed_records:
                        layer_k[layer_idx][output_idx] = k_slice[record.start : record.end].contiguous()
                        layer_v[layer_idx][output_idx] = v_slice[record.start : record.end].contiguous()

        def _concatenate(parts: list[torch.Tensor | None], *, name: str) -> torch.Tensor:
            if any(part is None for part in parts):
                raise RuntimeError(f"Internal Reasoner cache read left an unresolved {name} slice")
            tensors = [part for part in parts if part is not None]
            return torch.cat(tensors, dim=0)

        cross_k = tuple(_concatenate(parts, name=f"K layer {idx}") for idx, parts in enumerate(layer_k))
        cross_v = tuple(_concatenate(parts, name=f"V layer {idx}") for idx, parts in enumerate(layer_v))
        lengths = [record.num_tokens for _request, _shard, record in resolved]
        causal_offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
        return ReasonerFeatureBatch(
            cross_k=cross_k,
            cross_v=cross_v,
            causal_offsets=causal_offsets,
            fingerprints=tuple(request.fingerprint for request in requests),
        )


__all__ = [
    "IncrementalReasonerFeatureCacheWriter",
    "OfflineReasonerFeatureProvider",
    "REASONER_FEATURE_CACHE_MANIFEST",
    "REASONER_FEATURE_CACHE_SCHEMA_VERSION",
    "ReasonerFeatureCacheEntry",
    "ReasonerFeatureCacheIdentity",
    "build_reasoner_feature_request_from_text_tokens",
    "build_reasoner_feature_requests",
    "compute_reasoner_feature_fingerprint",
    "finalize_incremental_reasoner_feature_cache",
    "write_reasoner_feature_cache",
]
