# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert disjoint ShareGPT video splits to native SFT JSONL using CPU file I/O.

All source rows and media are checked before optional deterministic subsetting.
The full media inventory remains attached to the accepted dataset, including media
outside the subset. Videos stay as real local payloads; this command never decodes,
resizes, copies, or replaces them with text. No training runtime is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

PROTOCOL = "cosmos.wts-sft-conversion.v1"
_CHUNK = 1024 * 1024
_MAX_ROW_CHARS = 16 * _CHUNK


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
            size += len(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"source changed while hashing: {path}")
    return {"size": size, "sha256": digest.hexdigest()}


def _array_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSON array with a bounded buffer, rejecting malformed/trailing input."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill() -> None:
            nonlocal buffer, position, eof
            remaining = buffer[position:]
            if len(remaining) > _MAX_ROW_CHARS:
                raise ValueError(f"annotation row exceeds {_MAX_ROW_CHARS} characters: {path}")
            chunk = stream.read(_CHUNK)
            buffer = remaining + chunk
            position = 0
            eof = not chunk

        def peek() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ""
                fill()

        if peek() != "[":
            raise ValueError(f"annotations must be a JSON array: {path}")
        position += 1
        if peek() == "]":
            position += 1
        else:
            while True:
                peek()
                while True:
                    try:
                        row, end = decoder.raw_decode(buffer, position)
                        position = end
                        break
                    except json.JSONDecodeError as error:
                        if eof:
                            raise ValueError(f"invalid annotation JSON in {path}: {error}") from error
                        fill()
                if not isinstance(row, dict):
                    raise ValueError(f"annotation rows must be objects: {path}")
                yield row
                separator = peek()
                position += 1
                if separator == "]":
                    break
                if separator != ",":
                    raise ValueError(f"expected ',' or ']' after annotation row: {path}")
        if peek():
            raise ValueError(f"trailing content after annotation array: {path}")


def _convert(row: dict[str, Any], media_root: Path, system_prompt: str | None) -> tuple[dict[str, Any], Path]:
    identifier = row.get("id")
    if type(identifier) not in (str, int) or not str(identifier).strip():
        raise ValueError("every source row requires a nonempty string or integer id")
    video = row.get("video")
    if not isinstance(video, str) or not video or Path(video).is_absolute():
        raise ValueError(f"row {identifier!r} requires a relative video path")
    resolved = (media_root / video).resolve()
    if not resolved.is_relative_to(media_root) or not resolved.is_file():
        raise ValueError(f"row {identifier!r} video must be an existing file inside its media root: {video}")
    turns = row.get("conversations")
    if not isinstance(turns, list) or not turns or len(turns) % 2:
        raise ValueError(f"row {identifier!r} needs complete human/gpt conversation pairs")
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    marker_count = 0
    for index, turn in enumerate(turns):
        role = "human" if index % 2 == 0 else "gpt"
        if not isinstance(turn, dict) or turn.get("from") != role or not isinstance(turn.get("value"), str):
            raise ValueError(f"row {identifier!r} conversation {index} must be a {role} text turn")
        value = turn["value"]
        marker_count += value.count("<video>")
        if index == 0:
            if value.count("<video>") != 1:
                raise ValueError(f"row {identifier!r} first human turn must contain exactly one <video> marker")
            content: Any = [
                {"type": "video", "video": str(resolved)},
                {"type": "text", "text": value.replace("<video>", "", 1)},
            ]
        else:
            content = value
        messages.append({"role": "user" if role == "human" else "assistant", "content": content})
    if marker_count != 1:
        raise ValueError(f"row {identifier!r} must contain exactly one <video> marker across all turns")
    return {"id": identifier, "messages": messages}, resolved


def _sync_write(path: Path, lines: Iterator[str]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for line in lines:
            stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_wts(
    *,
    train_annotations: Path,
    train_media: Path,
    validation_annotations: Path,
    validation_media: Path,
    output_dir: Path,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    seed: int = 42,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Publish a complete dataset directory atomically; accept identical reruns only."""
    for limit in (train_limit, validation_limit):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("split limits must be positive integers or omitted")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("an explicit system prompt must be nonempty")
    output_dir = output_dir.expanduser().absolute()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.parent / f".{output_dir.name}.prepare.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(f"another conversion or interrupted conversion owns {lock}") from error
    os.close(lock_fd)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.prepare-", dir=output_dir.parent) as temporary:
            work = Path(temporary)
            staging = work / "dataset"
            staging.mkdir()
            manifest: dict[str, Any] = {
                "protocol": PROTOCOL,
                "options": {
                    "train_limit": train_limit,
                    "validation_limit": validation_limit,
                    "seed": seed,
                    "system_prompt": system_prompt,
                    "selection": "lowest_sha256_of_json_seed_split_id_then_source_order",
                    "marker_policy": "remove_exactly_one_video_marker_preserve_other_text",
                },
                "splits": {},
            }
            with closing(sqlite3.connect(work / "inventory.sqlite")) as database:
                database.execute("PRAGMA temp_store=FILE")
                database.execute("PRAGMA cache_size=-4096")
                database.execute(
                    "CREATE TABLE rows (id TEXT PRIMARY KEY, split TEXT, ordinal INTEGER, sample_key TEXT, payload TEXT, media TEXT)"
                )
                database.execute("CREATE TABLE media (path TEXT PRIMARY KEY, sha256 TEXT, size INTEGER)")
                for split, annotations, media_root, limit in (
                    ("train", train_annotations, train_media, train_limit),
                    ("validation", validation_annotations, validation_media, validation_limit),
                ):
                    annotations, media_root = annotations.expanduser().resolve(), media_root.expanduser().resolve()
                    if not annotations.is_file() or not media_root.is_dir():
                        raise ValueError(f"{split} annotations and media root must exist")
                    annotation_identity = _hash(annotations)
                    count = 0
                    for ordinal, row in enumerate(_array_rows(annotations)):
                        converted, media = _convert(row, media_root, system_prompt)
                        identifier = str(converted["id"])
                        sample_key = hashlib.sha256(_json([seed, split, identifier]).encode()).hexdigest()
                        try:
                            database.execute(
                                "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)",
                                (identifier, split, ordinal, sample_key, _json(converted), str(media)),
                            )
                        except sqlite3.IntegrityError as error:
                            raise ValueError(
                                f"duplicate or overlapping row id across full splits: {identifier!r}"
                            ) from error
                        if database.execute("SELECT 1 FROM media WHERE path=?", (str(media),)).fetchone() is None:
                            identity = _hash(media)
                            if identity["size"] == 0:
                                raise ValueError(f"source video is empty: {media}")
                            database.execute(
                                "INSERT INTO media VALUES (?, ?, ?)", (str(media), identity["sha256"], identity["size"])
                            )
                        count += 1
                    if not count or (limit is not None and limit > count):
                        raise ValueError(f"{split} count={count} cannot satisfy requested limit={limit}")
                    if _hash(annotations) != annotation_identity:
                        raise ValueError(f"{split} annotations changed during conversion")
                    manifest["splits"][split] = {
                        "source_annotations": {"path": str(annotations), **annotation_identity},
                        "media_root": str(media_root),
                        "source_rows": count,
                        "selected_rows": count if limit is None else limit,
                    }
                database.execute("CREATE INDEX media_digest ON media(sha256)")
                overlap = database.execute(
                    "SELECT m.sha256 FROM rows r JOIN media m ON m.path=r.media GROUP BY m.sha256 HAVING COUNT(DISTINCT r.split)>1 LIMIT 1"
                ).fetchone()
                if overlap is not None:
                    raise ValueError(f"video content overlaps across full train/validation splits: {overlap[0]}")
                for split in ("train", "validation"):
                    target = staging / f"{split}.jsonl"
                    limit = manifest["splits"][split]["selected_rows"]
                    cursor = database.execute(
                        "SELECT payload FROM (SELECT payload,ordinal FROM rows WHERE split=? ORDER BY sample_key,id LIMIT ?) ORDER BY ordinal",
                        (split, limit),
                    )
                    _sync_write(target, (row[0] for row in cursor))
                    manifest["splits"][split]["output"] = {"file": target.name, **_hash(target)}
                inventory = staging / "media_inventory.jsonl"
                cursor = database.execute(
                    "SELECT r.split,r.id,m.path,m.sha256,m.size FROM rows r JOIN media m ON m.path=r.media ORDER BY r.split,r.ordinal"
                )
                _sync_write(
                    inventory,
                    (_json(dict(zip(("split", "id", "path", "sha256", "size"), row, strict=True))) for row in cursor),
                )
                manifest["media_inventory"] = {"file": inventory.name, **_hash(inventory)}
                manifest["full_split_disjointness"] = {"row_ids": True, "video_sha256": True}
                database.commit()
            _sync_write(staging / "manifest.json", iter([_json(manifest)]))
            _sync_directory(staging)
            if output_dir.exists() or output_dir.is_symlink():
                expected = {path.name for path in staging.iterdir()}
                if (
                    output_dir.is_symlink()
                    or not output_dir.is_dir()
                    or {path.name for path in output_dir.iterdir()} != expected
                ):
                    raise ValueError(f"refusing an existing different or incomplete output: {output_dir}")
                if any(
                    not (output_dir / name).is_file()
                    or (output_dir / name).is_symlink()
                    or _hash(output_dir / name) != _hash(staging / name)
                    for name in expected
                ):
                    raise ValueError(f"refusing an existing different output: {output_dir}")
            else:
                os.rename(staging, output_dir)
                _sync_directory(output_dir.parent)
            return manifest
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-annotations", "train-media", "validation-annotations", "validation-media", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", help="Explicit optional system message; no prompt is silently injected")
    args = parser.parse_args()
    manifest = prepare_wts(**vars(args))
    print(
        _json(
            {
                "output_dir": str(args.output_dir),
                "counts": {name: split["selected_rows"] for name, split in manifest["splits"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
