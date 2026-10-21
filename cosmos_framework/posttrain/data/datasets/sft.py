# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Map-style SFT conversations with local media and an exact source identity."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, TextIO

from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.data.datasets.factory import build_custom_dataset
from cosmos_framework.posttrain.schema import (
    SFT_OPENER_PREFILLED,
    SFT_OPENER_SELF,
    SFT_OPENER_VALUES,
    SFT_SUPERVISE_ALL,
    SFT_SUPERVISE_VALUES,
)
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256


@contextmanager
def _open_jsonl(path: str, credentials: str) -> Iterator[TextIO]:
    """Open a local file or one S3/GCS object; close the response even when parsing fails."""
    if path.startswith("s3://"):
        from cosmos_framework.posttrain.utils.object_store import _parse_s3_url, build_s3_client

        bucket, key = _parse_s3_url(path)
        response = build_s3_client(credentials).get_object(Bucket=bucket, Key=key)
        with closing(response["Body"]) as body, io.TextIOWrapper(body, encoding="utf-8") as stream:
            yield stream
    else:
        with open(path, encoding="utf-8") as stream:
            yield stream


def sha256_file(path: str) -> str:
    """sha256 of one file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_row(row: Any, media_root: Path | None, *, label: str) -> dict[str, Any]:
    """Validate message structure and resolve media without decoding it or changing the source."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
        raise ValueError(f"{label}: expected a non-empty messages list")
    result = copy.deepcopy(row)
    assistant_count = 0
    for index, message in enumerate(result["messages"]):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{label}: message {index} has an unsupported role")
        assistant_count += message["role"] == "assistant"
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            content = message["content"] = ""
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError(f"{label}: message {index} needs string or content-list content")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(f"{label}: content parts must be objects")
            kind = part.get("type")
            if kind == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError(f"{label}: text part requires text")
            elif kind in {"image", "video"}:
                if media_root is None:
                    raise ValueError(f"{label}: S3 JSONL media requires an explicit local media_root")
                value = part.get(kind)
                if not isinstance(value, str) or not value or "://" in value:
                    raise ValueError(f"{label}: {kind} requires a local file path in '{kind}'")
                path = Path(value).expanduser()
                path = (media_root / path).resolve() if not path.is_absolute() else path.resolve()
                if not path.is_file():
                    raise ValueError(f"{label}: missing {kind} file {path}")
                part[kind] = str(path)
            else:
                raise ValueError(f"{label}: unsupported content type {kind!r}")
    if not assistant_count:
        raise ValueError(f"{label}: no assistant turns to supervise")
    tools = result.get("tools")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)):
        raise ValueError(f"{label}: tools must be a list of objects")
    # The row's supervision rule (``SFTTarget.supervise``): validated when present, never added when absent, so the source
    # fingerprint of every existing file is unchanged. ``null`` is not a value: a row either names its rule or has none.
    if "supervise" in result and result["supervise"] not in SFT_SUPERVISE_VALUES:
        raise ValueError(
            f"{label}: supervise must be one of {list(SFT_SUPERVISE_VALUES)} when present, got {result['supervise']!r}"
        )
    # The row's opener rule (``SFTTarget.opener``, row rule R4): validated when present, never added when absent. The
    # exporter records its choice as ``meta.opener``; a record that disagrees with the field the packer reads (a
    # ``prefilled`` in ``meta`` with no field included) would train the prefilled slice as self rows, so it is refused.
    if "opener" in result and result["opener"] not in SFT_OPENER_VALUES:
        raise ValueError(
            f"{label}: opener must be one of {list(SFT_OPENER_VALUES)} when present, got {result['opener']!r}"
        )
    meta = result.get("meta")
    meta_opener = meta.get("opener") if isinstance(meta, dict) else None
    if meta_opener is not None and meta_opener != result.get("opener", SFT_OPENER_SELF):
        raise ValueError(
            f"{label}: meta.opener {meta_opener!r} (the exporter's record) disagrees with the row's opener field "
            f"{result.get('opener', SFT_OPENER_SELF)!r}, which is what the packer reads; write the same value in both"
        )
    supervise = result.get("supervise", SFT_SUPERVISE_ALL)
    meta_supervise = meta.get("supervise") if isinstance(meta, dict) else None
    if meta_supervise is not None and meta_supervise != supervise:
        raise ValueError(
            f"{label}: meta.supervise {meta_supervise!r} (the exporter's record) disagrees with the row's supervise field "
            f"{supervise!r}, which is what the packer reads; write the same value in both"
        )
    # A prefilled row needs a reasoning block in every SUPERVISED assistant turn (all of them under all_assistant, the last
    # under last_assistant), or the packer would find no opener to mask there and the row would train as a self row under
    # a prefilled contract string (review R149-01; the packer refuses the same rows by the mask it removed).
    if result.get("opener", SFT_OPENER_SELF) == SFT_OPENER_PREFILLED:
        assistant = [i for i, message in enumerate(result["messages"]) if message.get("role") == "assistant"]
        supervised = assistant if supervise == SFT_SUPERVISE_ALL else assistant[-1:]
        bare = [
            i
            for i in supervised
            if not (
                isinstance(result["messages"][i].get("reasoning_content"), str)
                and result["messages"][i]["reasoning_content"].strip()
            )
        ]
        if bare:
            raise ValueError(
                f"{label}: opener=prefilled requires a non-empty reasoning_content on every supervised assistant turn (all "
                f"of them under all_assistant, the last under last_assistant); message(s) {bare} carry none (review R149-01)"
            )
    return result


class JSONLSFTDataset:
    """Small/example JSONL source; fingerprints each referenced media file once at setup.

    S3/GCS JSONL is read once per instance into memory, without a persistent cache.
    Its media, if any, must already be staged under an explicit local media_root.
    Source identity covers normalized content, not the URI or credential path, so text-only
    data can move between clusters without changing its resume identity.
    Large corpora should use a map-style factory exposing its own immutable dataset identity.
    """

    def __init__(self, path: str, media_root: str = "", credentials: str = "credentials/<credential>.secret") -> None:
        remote = path.startswith("s3://")
        source = path if remote else str(Path(path).expanduser().resolve())
        root = Path(media_root).expanduser().resolve() if media_root else (None if remote else Path(source).parent)
        self.rows: list[dict[str, Any]] = []
        # Resolved media path -> sha256 of the file's bytes; one read per distinct file. Part of the source identity
        # below and the media side of ``row_identity`` (review R134-04).
        self.media_hashes: dict[str, str] = {}
        media_hashes = self.media_hashes
        with _open_jsonl(source, credentials) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                label = f"{source}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{label}: invalid JSON: {error}") from error
                normalized = normalize_row(row, root, label=label)
                self.rows.append(normalized)
                for message in normalized["messages"]:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        kind = part.get("type")
                        if kind in {"image", "video"} and part[kind] not in media_hashes:
                            media_hashes[part[kind]] = sha256_file(part[kind])
        if not self.rows:
            raise ValueError(f"{source}: no SFT conversations")
        self._fingerprint: str = canonical_json_sha256(
            {"rows": self.rows, "media": media_hashes}, context="SFT JSONL source"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[index])

    def cursor_fingerprint(self) -> str:
        return self._fingerprint


class FactorySFTDataset:
    """Lazy canonical view; the factory owns content identity, including referenced media."""

    def __init__(self, config: SFTDatasetConfig) -> None:
        self.dataset: Any = build_custom_dataset(config.factory, config.factory_kwargs)
        fingerprint = getattr(self.dataset, "cursor_fingerprint", None)
        if not callable(fingerprint) or not isinstance(identity := fingerprint(), str) or not identity:
            raise ValueError("SFT dataset factories must expose a non-empty cursor_fingerprint()")
        self.media_root: Path = Path(config.media_root or ".").expanduser().resolve()
        self._fingerprint: str = canonical_json_sha256(
            {
                "factory": config.factory,
                "kwargs": config.factory_kwargs,
                "media_root": str(self.media_root),
                "content": identity,
            },
            context="SFT factory source",
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return normalize_row(self.dataset[index], self.media_root, label=f"SFT factory row {index}")

    def cursor_fingerprint(self) -> str:
        return self._fingerprint

    def __getattr__(self, name: str) -> Any:
        if name in {"cursor_state_dict", "load_cursor_state_dict"}:
            return getattr(self.dataset, name)
        raise AttributeError(name)


def media_digest_for(*datasets: Any) -> Callable[[str], str]:
    """A ``resolved media path -> sha256(file bytes)`` lookup seeded from the datasets' own ``media_hashes`` (the JSONL
    source hashes every distinct file once at setup) and hashing on first sight otherwise (factory sources)."""
    cache: dict[str, str] = {}
    for dataset in datasets:
        known = getattr(dataset, "media_hashes", None)
        if isinstance(known, dict):
            cache.update(known)

    def digest(path: str) -> str:
        value = cache.get(path)
        if value is None:
            value = cache[path] = sha256_file(path)
        return value

    return digest


def row_identity(row: dict[str, Any], *, media_digest: Callable[[str], str] | None = None) -> tuple[str, str | None]:
    """``(content_sha256, item)`` of one normalized row.

    The hash covers ``messages`` and ``tools`` only (the same conversation re-exported under other metadata still
    matches), with every image/video part's path replaced by ``sha256:<digest of the file's bytes>``: a normalized row
    carries the absolute resolved path, which differs between two media roots holding COPIES of the same file, so a
    path-based hash let the same image row pass under another root (review R134-04; symlinks resolved and were caught,
    copies were not). ``item`` is ``row["meta"]["item"]`` when the exporter recorded one (the benchmark item the episode
    was run on), else None.
    """
    digest = media_digest or sha256_file
    messages: list[Any] = []
    for message in row.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            messages.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in {"image", "video"} and isinstance(part.get(kind), str):
                parts.append({**part, kind: f"sha256:{digest(part[kind])}"})
            else:
                parts.append(part)
        messages.append({**message, "content": parts})
    content_hash = canonical_json_sha256(
        {"messages": messages, "tools": row.get("tools")}, context="SFT row content identity"
    )
    meta = row.get("meta")
    item = meta.get("item") if isinstance(meta, dict) else None
    return content_hash, item if isinstance(item, str) and item else None


def source_overlap(training: Any, validation: Any, *, max_examples: int = 5) -> dict[str, Any]:
    """Census of the rows two SFT sources share: identical content (sha256 of messages+tools, media by file bytes) or
    the same item id, plus the item-id coverage of both sides.

    Both sources are map-style datasets (``len`` and ``[index]`` returning normalized rows). A validation source
    overlapping the training source by even one row or one item is not held out (review R125-D-02: ``validation_4.jsonl``
    was the training file's first four lines and the whole-source fingerprint guard let it pass). A validation row
    without an item id has only its content hash as identity, which one changed character defeats, so the census counts
    ``training_rows_with_item`` and ``validation_rows_without_item`` for the guard to refuse that combination (review
    R134-04).
    """

    def rows_of(dataset: Any) -> Any:
        rows = getattr(dataset, "rows", None)
        return rows if isinstance(rows, list) else (dataset[index] for index in range(len(dataset)))

    digest = media_digest_for(training, validation)
    train_rows: dict[str, int] = {}
    train_items: dict[str, int] = {}
    training_rows_with_item = 0
    for index, row in enumerate(rows_of(training)):
        content, item = row_identity(row, media_digest=digest)
        train_rows.setdefault(content, index)
        if item is not None:
            training_rows_with_item += 1
            train_items.setdefault(item, index)
    shared_rows: list[tuple[int, int]] = []
    shared_items: list[tuple[str, int, int]] = []
    validation_items = 0
    validation_rows_without_item = 0
    for index, row in enumerate(rows_of(validation)):
        content, item = row_identity(row, media_digest=digest)
        if content in train_rows:
            shared_rows.append((index, train_rows[content]))
        if item is None:
            validation_rows_without_item += 1
        else:
            validation_items += 1
            if item in train_items:
                shared_items.append((item, index, train_items[item]))
    return {
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_items": len(train_items),
        "training_rows_with_item": training_rows_with_item,
        "validation_items": validation_items,
        "validation_rows_without_item": validation_rows_without_item,
        "shared_rows": len(shared_rows),
        "shared_items": len(shared_items),
        "shared_row_examples": shared_rows[:max_examples],
        "shared_item_examples": shared_items[:max_examples],
    }


def refuse_source_overlap(census: Mapping[str, Any]) -> None:
    """The held-out contract on a :func:`source_overlap` census: refuse a validation source that shares a row (content
    hash) or an item id with the training source, and one whose rows carry no item id while the training rows do.

    Pure on the census dict, so one rank can compute the census and every rank apply the same refusal (the SFT setup,
    ``entrypoints/sft.py::held_out_census``), and the row validator applies the same rule as the setup. The whole-source
    fingerprint comparison catches a byte-identical copy only; a subset of the training file passed it (review
    R125-D-02), so the census is per row. The content hash covers media by file bytes (a copy of the training media
    under another root is the same row), and a validation row without ``meta.item`` is refused when the training rows
    carry items: its only identity would be the content hash, which one changed character defeats (review R134-04,
    bypasses 4 and 5).
    """
    if census["shared_rows"] or census["shared_items"]:
        raise ValueError(
            "SFT validation source is not held out from the training source: "
            f"{census['shared_rows']} shared row(s) (validation index, training index) {census['shared_row_examples']}, "
            f"{census['shared_items']} shared item(s) (item, validation index, training index) "
            f"{census['shared_item_examples']}; use rows whose content and item are absent from the training file"
        )
    if census["training_rows_with_item"] and census["validation_rows_without_item"]:
        raise ValueError(
            "SFT validation source is not verifiably held out from the training source: "
            f"{census['validation_rows_without_item']} of {census['validation_rows']} validation row(s) carry no "
            f"meta.item while {census['training_rows_with_item']} of {census['training_rows']} training rows do; a row "
            "without an item id has only its content hash as identity, and one changed character defeats that hash; "
            "record the item every validation row was run on"
        )


def require_disjoint_datasets(training: Any, validation: Any) -> dict[str, Any]:
    """Census plus refusal in one process (:func:`source_overlap` then :func:`refuse_source_overlap`); returns the
    census for the run record. The SFT setup and the row validator both call this contract, never a copy of it."""
    census = source_overlap(training, validation)
    refuse_source_overlap(census)
    return census


def build_sft_dataset(config: SFTDatasetConfig) -> JSONLSFTDataset | FactorySFTDataset:
    config.require_source()
    if config.path:
        return JSONLSFTDataset(config.path, config.media_root, config.credentials)
    return FactorySFTDataset(config)
