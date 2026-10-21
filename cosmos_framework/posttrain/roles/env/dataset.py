# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dataset-backed 1-step Envs (DESIGN §1.3, degenerate form).

v0 implements the 1-step dataset Env for VLM RL plus a synthetic arithmetic Env for
GRPO. The `step()` hook is kept so multi-step sims slot in later with no interface
change.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import random
import zlib
from collections.abc import Callable, Iterator, Mapping
from typing import IO, Any, NamedTuple

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.schema import ObsBundle


def split_prompts(prompts: list[Any], i: int, n: int) -> list[Any]:
    """Return replica ``i``'s deterministic contiguous share of ``prompts``.

    Identical ``(i, n)`` inputs must produce identical slices on every SPMD rank; the floor-division
    boundaries cover the input once without overlap.
    """
    length = len(prompts)
    return prompts[i * length // n : (i + 1) * length // n]


def _resolve_shard(shard_index: int | None, shard_count: int | None) -> tuple[int, int]:
    """Validate explicit shard coordinates, defaulting both omissions to no sharding.

    Placement coordinates cannot be inferred because a colocation's replica count may exceed the
    data role's and silently drop rows in fan-out topologies.
    """
    if (shard_index is None) != (shard_count is None):
        raise ValueError(
            f"shard_index={shard_index} and shard_count={shard_count} must be given together "
            f"(or both omitted for no sharding)"
        )
    if shard_index is None:
        return 0, 1  # no sharding -- see the docstring for why this is not the placement coordinates
    if shard_count < 1 or not (0 <= shard_index < shard_count):
        raise ValueError(
            f"shard_index={shard_index}, shard_count={shard_count}: need 0 <= index < count and count >= 1"
        )
    return shard_index, shard_count


def _stamp_prompt_id(shard_index: int, emitted: int) -> str:
    """Deterministic, shard-scoped, monotonic prompt id (the GRPO `group_id` namespace).

    Replaces `uuid.uuid4().hex`: a random id makes `sorted(key=(group_id, id))`
    (`workers/dispatch.py::select_chunk`) a canonical order WITHIN one run (every rank of that
    run sorts the same random values identically) but not ACROSS runs, so two seeded runs that
    should be bit-identical hand each DP rank a different subset of the batch and diverge from step
    2 onward. This id is a pure function of `(shard_index, emitted)`, so two runs with the same
    seed/shard layout reproduce the identical id sequence and therefore the identical sort order.

    Uniqueness across replicas comes from `shard_index`: `_resolve_shard` requires every replica
    sharding one source to carry a distinct index (see its docstring), so distinct replicas can
    never stamp the same prefix. Uniqueness within one replica's draws comes from `emitted`, a
    counter that only ever increments -- unlike the cursor position, it does NOT reset when
    `repeat=True` wraps the cursor back to epoch start, so a wrapped epoch does not reissue an id.

    Args:
        shard_index: this replica's shard index (`DatasetEnv._shard_index` / `ArithmeticEnv`'s).
        emitted: number of ids this instance has stamped before this one (0-based, monotonic).

    Returns:
        `f"{shard_index}-{emitted:010d}"`.
    """
    return f"{shard_index}-{emitted:010d}"


def require_readable_jsonl(path: str) -> None:
    """Fail before Ray placement/model construction when JSONL is unavailable in this environment."""
    try:
        with open(path, encoding="utf-8"):
            pass
    except OSError as exc:
        raise ValueError(f"prompt JSONL {path!r} is not readable: {exc}") from exc


def is_conversation_prompt(prompt: Any) -> bool:
    """Return whether ``prompt`` has the conversation-list shape the prompt sources accept.

    The ONE conversation grammar of the data roles: a non-empty list of objects, each with a string
    ``role`` and a ``content`` field. `jsonl_prompts` (the GRPO loader) and `project_prompt_items`
    (the PPO projection) both apply it, so a row one role serves the other serves too. What the
    content parts may hold (text / image / video blocks) is the DataPacker's contract
    (`data/packer/qwen3_vl.py::Qwen3VLDataPacker._messages_and_media`) and is not validated twice.
    """
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and all(
            isinstance(message, dict) and isinstance(message.get("role"), str) and "content" in message
            for message in prompt
        )
    )


_GZIP_MAGIC = b"\x1f\x8b"
_UTF8_BOM = b"\xef\xbb\xbf"


class _BadByte(NamedTuple):
    """The first byte outside UTF-8 of a leniently decoded text: the character index it lands on, its refusal
    (``<path>:<line>: not UTF-8 text (...)``, the line that holds the byte) and the decode error behind it."""

    index: int
    message: str
    error: UnicodeDecodeError


def _decoded_to_first_bad_byte(path: str, position: int, raw: bytes) -> tuple[str, _BadByte | None]:
    """``raw`` (its first byte on file line ``position``) as text, with ``None`` when every byte is UTF-8, else
    the first byte that is not: the text is decoded as written up to that byte and leniently (``surrogateescape``)
    from it on, so that the rows before it can be served and refused in file order before its refusal is raised."""
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        line = position + raw.count(b"\n", 0, exc.start)
        index = len(raw[: exc.start].decode("utf-8"))
        return raw.decode("utf-8", "surrogateescape"), _BadByte(index, f"{path}:{line}: not UTF-8 text ({exc})", exc)


def _decoded(path: str, position: int, raw: bytes) -> str:
    """``raw`` (its first byte on file line ``position``) as UTF-8 text; bytes outside UTF-8 raise `ValueError`
    with the line that holds the first bad byte."""
    text, bad_byte = _decoded_to_first_bad_byte(path, position, raw)
    if bad_byte is not None:
        raise ValueError(bad_byte.message) from bad_byte.error
    return text


def _array_objects(path: str, text: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """The elements of one top-level JSON array, positioned by their 1-based index."""
    try:
        elements = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not a valid JSON array ({exc})") from exc
    for index, obj in enumerate(elements, start=1):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{index}: expected a JSON object, got {type(obj).__name__}")
        yield index, obj


def _stream_objects(
    path: str, position: int, text: str, bad_byte: _BadByte | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """The objects of a whitespace-separated stream of JSON values -- ``text``, whose first character is on file
    line ``position`` -- each positioned by the line it opens on.

    The general shape of the local-rows contract, the one pyarrow's JSON reader (behind the Hugging Face
    ``load_dataset("json")`` the roles used before) read: one object pretty-printed over the document is the
    one-row case, several pretty-printed objects separated by whitespace (what ``json.dump(obj, f, indent=2)``
    in a loop writes) are one row each, and one-line rows may sit among them. Consecutive objects are separated
    by at least one whitespace character -- a newline, or a space when two objects share a line (two rows); an
    object glued to the one before it (``}{``) is outside the contract and raises `ValueError` with the line and
    the 1-based column where the second object starts, instead of serving both as rows -- after the object before
    the boundary was served, so that a caller's refusal of that row (a missing column) comes first, in file order.
    Any other text glued to an object (``}x``, ``}[1]``) is refused as what it is: not valid JSON, or a value that
    is not an object. `_objects_from` enters the stream from the first line that opens an object without closing
    it; a file of one object per line never gets here. A value that is not an object raises `ValueError` with the line it starts on; text that is not
    JSON raises `ValueError` with the line it starts on (the parenthesised detail counts lines and characters
    from the start of ``text``, i.e. from file line ``position``). ``bad_byte`` is the first byte outside UTF-8
    when ``text`` was decoded leniently (`_decoded_to_first_bad_byte`): the objects before it are served and
    refused in file order, and its own refusal is raised when the stream reaches it -- between values, inside
    a value that parses, or at the point where the parse of the value holding it fails.
    """
    decoder = json.JSONDecoder()
    offset, scanned, line, length = 0, 0, position, len(text)
    after_object = False
    while True:
        if after_object and offset < length and text[offset] == "{":
            # The next object starts where the previous one ended (`}{`): outside the whitespace-separated
            # contract (the one-object-per-line path refuses the same line as not valid JSON). Checked here, after
            # the previous object was served, so that a caller's refusal of THAT row comes first; other text glued
            # to an object is refused below as not valid JSON or as a non-object value.
            at_line = line + text.count("\n", scanned, offset)
            raise ValueError(
                f"{path}:{at_line}: objects must be separated by whitespace "
                f"(adjacent objects at column {offset - text.rfind(chr(10), 0, offset)})"
            )
        while offset < length and text[offset] in " \t\n\r":
            offset += 1
        if offset >= length:
            return
        line += text.count("\n", scanned, offset)
        scanned = offset
        if bad_byte is not None and bad_byte.index <= offset:
            raise ValueError(bad_byte.message) from bad_byte.error
        try:
            obj, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError as exc:
            if bad_byte is not None and bad_byte.index <= exc.pos:
                raise ValueError(bad_byte.message) from bad_byte.error
            raise ValueError(f"{path}:{line}: not valid JSON ({exc})") from exc
        if bad_byte is not None and bad_byte.index <= end:
            raise ValueError(bad_byte.message) from bad_byte.error
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line}: expected a JSON object, got {type(obj).__name__}")
        yield line, obj
        offset, after_object = end, True


def _objects_from(path: str, handle: IO[bytes]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(position, object)`` for the rows in ``handle`` (the bytes of ``path``), in file order.

    The local-rows contract of the data roles (`jsonl_prompts`, `JsonlRows`, PPO's ``rl.data.path``) -- the
    shapes Hugging Face ``load_dataset("json")`` (pyarrow's streaming JSON reader) read before the roles stopped
    using it: a whitespace-separated stream of JSON objects -- one object per line (JSONL; blank lines skipped;
    the position is the line number), one object pretty-printed over several lines (one row), or several
    pretty-printed objects one after the other (the position is the line each object opens on), consecutive
    objects separated by at least one whitespace character (two objects on one line need a space between them;
    an object glued to the one before it, ``}{``, is refused with the line and column where it starts) -- or one
    top-level JSON array of objects (a file whose first non-blank line starts with ``[``; the position is the
    1-based element index), any of them gzip-compressed: gzip only, detected by its magic bytes whatever the
    suffix (bz2 / xz / zst are not detected and are refused as non-UTF-8 text); a leading UTF-8 byte-order mark
    is dropped. The array shape is decided on the FIRST non-blank line only: a ``[`` line after a row is a
    non-object row of a JSONL file, refused with its line number. Every object is `json.loads` / `raw_decode`
    output: as written, no schema inference, no null-fill. Rows are yielded one at a time, so a caller's per-row
    check names the FIRST bad row of the file. Malformed rows, non-object rows and bytes outside UTF-8 raise
    `ValueError` with the position rather than being skipped, and in FILE ORDER -- a byte outside UTF-8 further
    down the stream never hides an earlier bad row: downstream a dropped prompt is indistinguishable from a bad
    sample.
    """
    try:
        compressed = handle.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
        handle.seek(0)
        # The gzip stream is closed when the rows are exhausted or the generator is closed; a plain handle
        # belongs to the caller (`_jsonl_objects` closes the file it opened, `JsonlRows` passes a BytesIO).
        with gzip.GzipFile(fileobj=handle, mode="rb") if compressed else contextlib.nullcontext(handle) as stream:
            position = 0
            rows_seen = False
            for raw in stream:
                position += 1
                if position == 1:
                    raw = raw.removeprefix(_UTF8_BOM)
                if not raw.strip():
                    continue
                if not rows_seen and raw.lstrip().startswith(b"["):
                    # The array shape, on the first non-blank line only; a `[` after a row is a non-object
                    # row of a JSONL file and is refused below with its line number.
                    yield from _array_objects(path, _decoded(path, position, raw + stream.read()))
                    return
                text = _decoded(path, position, raw)
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    if text.lstrip().startswith("{"):
                        # An object that opens on this line without closing it: from here on the file is read as
                        # a whitespace-separated stream of objects (one pretty-printed object = one row; several
                        # = one row each; one-line rows may follow), positioned by the line each object opens on.
                        # The rest is decoded as written up to its first byte outside UTF-8 (if any), so the
                        # objects before that byte are served and refused in file order; its refusal is raised
                        # when the stream reaches it, not before the rows that precede it.
                        rest, bad_byte = _decoded_to_first_bad_byte(path, position + 1, stream.read())
                        if bad_byte is not None:
                            bad_byte = bad_byte._replace(index=bad_byte.index + len(text))
                        yield from _stream_objects(path, position, text + rest, bad_byte)
                        return
                    raise ValueError(f"{path}:{position}: not valid JSON ({exc})") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{position}: expected a JSON object, got {type(obj).__name__}")
                rows_seen = True
                yield position, obj
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"prompt JSONL {path!r} is not readable: {exc}") from exc


def _jsonl_objects(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """`_objects_from` over the file at ``path``, opened when the first row is asked for."""
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise ValueError(f"prompt JSONL {path!r} is not readable: {exc}") from exc
    with handle:
        yield from _objects_from(path, handle)


def jsonl_prompts(path: str, prompt_column: str, response_column: str = "") -> list[dict[str, Any]]:
    """Read a JSONL prompt file into `DatasetEnv` rows, renaming `prompt_column` to `prompt` and
    `response_column` to `reference`. Malformed rows raise with the file position rather than being
    skipped: downstream a dropped prompt is indistinguishable from a bad sample."""
    rows: list[dict[str, Any]] = []
    for lineno, obj in _jsonl_objects(path):
        for column in (prompt_column, response_column):
            if column and column not in obj:
                raise ValueError(f"{path}:{lineno}: no {column!r} field (has {sorted(obj)})")
        prompt = obj[prompt_column]
        if isinstance(prompt, list):
            if not is_conversation_prompt(prompt):
                raise ValueError(
                    f"{path}:{lineno}: {prompt_column!r} conversation must be a non-empty list of "
                    "objects with string 'role' and a 'content' field"
                )
        elif not isinstance(prompt, str):
            raise ValueError(
                f"{path}:{lineno}: {prompt_column!r} must be a string or conversation list, got {type(prompt).__name__}"
            )
        row = {"prompt": prompt}
        if response_column:
            answer = obj[response_column]
            if answer is None:
                raise ValueError(f"{path}:{lineno}: {response_column!r} must not be null")
            row["reference"] = answer
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prompts")
    return rows


class JsonlRows:
    """The rows of one local file as a finite map-style source, exactly as written.

    `project_prompt_items` (the PPO data role) needs a map-style source with a stable content
    fingerprint for exact cursor resume. Hugging Face ``datasets`` (``load_dataset("json", ...)``)
    is NOT that source for nested rows: Arrow infers one struct type over every content block of
    the file and null-fills the keys a block does not carry, so a text block comes back as
    ``{"type": "text", "text": ..., "image": None, "video": None, ...}``, and a chat template that
    tests ``'image' in content`` (Qwen3-VL) renders it as an image placeholder with the text gone;
    a nested reference column gains null keys the same way; and a conversation column whose turns
    mix string and list content is refused outright. This source reads the file's bytes ONCE and
    derives both the rows (`_objects_from`: JSONL or any whitespace-separated stream of JSON objects,
    pretty-printed or not, one JSON array, or the same gzip-compressed -- gzip only) and
    the fingerprint from that one snapshot, so the fingerprint names the bytes the rows came from
    even when the file is replaced underneath. The fingerprint is the SHA-256 of those bytes plus
    the row count.
    """

    def __init__(self, path: str, *, fingerprint_prefix: str = "jsonl-rows") -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("JsonlRows needs a non-empty path")
        if not isinstance(fingerprint_prefix, str) or not fingerprint_prefix:
            raise ValueError("fingerprint_prefix must be a non-empty string")
        try:
            with open(path, "rb") as handle:
                snapshot = handle.read()
        except OSError as exc:
            raise ValueError(f"prompt JSONL {path!r} is not readable: {exc}") from exc
        self._rows: list[dict[str, Any]] = [obj for _position, obj in _objects_from(path, io.BytesIO(snapshot))]
        if not self._rows:
            raise ValueError(f"{path}: no rows")
        digest = hashlib.sha256(snapshot).hexdigest()
        self._fingerprint: str = f"{fingerprint_prefix}-sha256={digest}|len={len(self._rows)}"

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def cursor_fingerprint(self) -> str:
        """Content identity for exact cursor resume: the file bytes and the row count."""
        return self._fingerprint


class _ColumnMapped:
    """Lazy row view that applies ``row_map`` then renames prompt/reference columns.

    Unmapped columns pass through, and access stays lazy so Arrow-backed sources remain memory mapped.
    """

    def __init__(
        self,
        ds: Any,
        prompt_column: str,
        response_column: str = "",
        row_map: Any | None = None,
        source_identity: str = "",
        cursor_fingerprint: str | Callable[[], str] = "",
    ) -> None:
        if not prompt_column:
            raise ValueError("_ColumnMapped needs a prompt_column: it names the column renamed to 'prompt'")
        self._ds: Any = ds
        self._prompt: str = prompt_column
        self._response: str = response_column
        self._row_map: Any | None = row_map
        self._source_identity: str = source_identity
        self._cursor_fingerprint: str | Callable[[], str] = cursor_fingerprint

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> dict:
        row = self._ds[i]
        if self._row_map is not None:
            row = self._row_map(row)
            if not isinstance(row, dict):
                raise ValueError(
                    f"rl.data.items_fn must return a ROW dict (it maps a row to a row); got "
                    f"{type(row).__name__}. The prompt itself belongs at the prompt_column key."
                )
        item = dict(row)
        if self._prompt not in item:
            raise KeyError(
                f"prompt_column={self._prompt!r} is not in the row (columns: {sorted(item)}); "
                f"an items_fn must keep the schema it was given."
            )
        item["prompt"] = item.pop(self._prompt)
        if self._response:
            item["reference"] = item.pop(self._response)
        return item

    def cursor_fingerprint(self) -> str:
        """Return source identity for exact cursor resume.

        Public dataset metadata is combined with the optional Arrow fingerprint and length; the
        private fingerprint is read defensively so future datasets versions degrade gracefully.
        """
        if self._cursor_fingerprint:
            return self._cursor_fingerprint() if callable(self._cursor_fingerprint) else self._cursor_fingerprint
        fn = getattr(self._ds, "cursor_fingerprint", None)
        if self._source_identity:
            from cosmos_framework.posttrain.roles.env.identity import stable_value_digest

            refinement = str(fn()) if callable(fn) else None
            identity = stable_value_digest((self._source_identity, len(self._ds), refinement))
            return f"custom-prompt-cursor-sha256:{hashlib.sha256(identity).hexdigest()}"
        if callable(fn):
            return str(fn())
        parts = []
        info = getattr(self._ds, "info", None)
        for attr in ("dataset_name", "config_name", "version"):
            value = getattr(info, attr, None)
            if value is not None:
                parts.append(f"{attr}={value}")
        split = getattr(self._ds, "split", None)
        if split is not None:
            parts.append(f"split={split}")
        arrow = getattr(self._ds, "_fingerprint", None)  # private on purpose; see the docstring
        if arrow:
            parts.append(f"arrow={arrow}")
        parts.append(f"len={len(self._ds)}")
        return "|".join(parts)

    def raw_source_fingerprint(self) -> str:
        """Content identity before this view's column mapping and ``row_map`` transform."""
        return self.cursor_fingerprint()


def map_prompt_items(
    dataset: Any,
    prompt_column: str,
    response_column: str = "",
    row_map: Any | None = None,
    source_identity: str = "",
    cursor_fingerprint: str | Callable[[], str] = "",
) -> _ColumnMapped:
    """Expose raw map-style rows through the canonical prompt/reference vocabulary."""
    return _ColumnMapped(
        dataset,
        prompt_column,
        response_column,
        row_map,
        source_identity,
        cursor_fingerprint,
    )


#: The projection's prompt contract, named once so the string and the list refusals read the same.
_PROMPT_SHAPE_ERROR = (
    "dataset prompt values must be non-empty strings or conversation lists (a non-empty list of objects, "
    "each with a non-empty string 'role' and a non-empty string or list 'content') for single-turn training"
)

#: The null-fill signature. A schema-inferring reader (Hugging Face ``datasets`` over Arrow) gives every content
#: part of a conversation column every key of every other part, ``None`` where it did not carry one, so a part
#: ends up with ``None`` under a PAYLOAD key (``image: None`` on a text part, ``text: None`` on an image part),
#: and the Qwen chat template -- which tests ``'image' in content`` / ``'image_url' in content``
#: / ``'video' in content`` before ``'text' in content`` -- renders such a text part as a media placeholder with
#: the text gone; a null handle can likewise shadow a valid alternate handle in the packer. Only that signature
#: (and a ``None`` type) is refused, whatever source produced the row. A
#: ``None`` under the part's OWN optional key (``fps: None`` on a video part, as a pool may write it) is the
#: DataPacker's contract and passes through (`cosmos_framework/processors/base.py::
#: maybe_parse_video_content` reads ``fps`` with ``.get`` and carries the ``None`` on). A local path is read
#: as written through `JsonlRows`, so its parts carry exactly the keys the file wrote.
_PART_PAYLOAD_KEYS = frozenset({"text", "image", "image_url", "video"})

_NULL_FILLED_PART_ERROR = (
    "dataset prompt content parts must not carry null values under a payload key "
    "(text / image / image_url / video) or a null type: a part such as {'type': 'text', 'text': ..., "
    "'image': None} is the null-filled struct a schema-inferring reader makes of a conversation column, and "
    "can render incorrectly or let a null handle shadow a valid alternate handle; read the rows as written "
    "(a local rl.data.path through JsonlRows) instead of through Hugging Face datasets"
)


def is_null_filled_part(part: Any) -> bool:
    """Return whether ``part`` (one entry of a list-typed turn content) carries the null-fill signature: a
    ``None`` under one of the payload keys (``text`` / ``image`` / ``image_url`` / ``video``), or a ``None``
    ``type``. A ``None`` under any other key (``fps``) is not the signature."""
    if not isinstance(part, dict):
        return False
    if "type" in part and part["type"] is None:
        return True
    return any(part.get(key, "") is None for key in _PART_PAYLOAD_KEYS)


def project_prompt_items(
    dataset: Any,
    *,
    source_name: str,
    revision: str = "",
    subset: str = "",
    split: str = "train",
    prompt_column: str,
    response_column: str = "",
    image_column: str = "",
    video_column: str = "",
    id_column: str = "",
    metadata_columns: tuple[str, ...] = (),
    modality: str = "text",
    video_fps: float = 24.0,
    seed: int = 0,
    row_map: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    row_map_identity: str = "",
    fingerprint_prefix: str = "prompt-projection",
) -> _ColumnMapped:
    """Project one finite map dataset onto an allowlisted multimodal prompt schema.

    The prompt column holds either a non-empty string (one user turn; the DataPacker wraps it) or a
    conversation list (`is_conversation_prompt`, the shape the GRPO loader `jsonl_prompts` accepts)
    whose turns each carry a non-empty ``role`` and a non-empty string or list ``content`` — the
    rows multimodal JSONL pools carry, with the image or video inside the content parts. Both pass
    through as the item's ``prompt`` untouched. Content parts that carry the null-fill signature
    (`is_null_filled_part`: ``None`` under a payload key, or a ``None`` type) are
    refused (`_NULL_FILLED_PART_ERROR`): they are what a schema-inferring reader makes of a
    conversation column, and they render wrong; a ``None`` under a part's own optional key (``fps``
    on a video part) is the DataPacker's contract and passes through. Row 0 is projected here, at
    construction: a defect that is a property of the whole source (null-fill is column-wide) is
    refused when the data role is built, before placement, not at the first fetch.
    """
    from cosmos_framework.posttrain.config.rl import normalize_prompt_metadata_columns, normalize_video_fps
    from cosmos_framework.posttrain.utils.identity import normalize_json_dict

    if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
        raise TypeError("prompt projection requires a finite map-style dataset")
    if len(dataset) < 1:
        raise ValueError("prompt projection requires at least one row")
    for name, value in (
        ("source_name", source_name),
        ("revision", revision),
        ("subset", subset),
        ("split", split),
        ("prompt_column", prompt_column),
        ("response_column", response_column),
        ("image_column", image_column),
        ("video_column", video_column),
        ("id_column", id_column),
        ("fingerprint_prefix", fingerprint_prefix),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not source_name or not split or not prompt_column or not fingerprint_prefix:
        raise ValueError("source_name, split, prompt_column, and fingerprint_prefix must be non-empty")
    if modality not in ("text", "image", "video"):
        raise ValueError(f"modality must be 'text', 'image', or 'video', got {modality!r}")
    configured_media = image_column if modality == "image" else video_column
    if modality != "text" and not configured_media:
        raise ValueError(f"{modality}_column must be non-empty for {modality} prompts")
    if type(metadata_columns) is not tuple:
        raise TypeError(f"metadata_columns must be a tuple, got {type(metadata_columns).__name__}")
    normalized_metadata_columns = normalize_prompt_metadata_columns(
        metadata_columns,
        configured_columns=(prompt_column, response_column, image_column, video_column, id_column),
    )
    if type(seed) is not int:
        raise ValueError(f"seed must be an exact integer, got {seed!r}")
    normalized_video_fps = normalize_video_fps(video_fps)

    def column(row: Mapping[str, Any], name: str, role: str) -> Any:
        if name not in row:
            raise ValueError(f"dataset row is missing configured {role} column {name!r}")
        return row[name]

    def project(raw_row: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"dataset row must be a mapping, got {type(raw_row).__name__}")
        row = raw_row if row_map is None else row_map(raw_row)
        if not isinstance(row, Mapping):
            raise TypeError(f"dataset row map must return a mapping, got {type(row).__name__}")
        prompt = column(row, prompt_column, "prompt")
        if isinstance(prompt, list):
            if not is_conversation_prompt(prompt) or any(
                not message["role"] or not isinstance(message["content"], (str, list)) or not message["content"]
                for message in prompt
            ):
                raise ValueError(_PROMPT_SHAPE_ERROR)
            if any(
                is_null_filled_part(part)
                for message in prompt
                if isinstance(message["content"], list)
                for part in message["content"]
            ):
                raise ValueError(_NULL_FILLED_PART_ERROR)
        elif not isinstance(prompt, str) or not prompt:
            raise ValueError(_PROMPT_SHAPE_ERROR)
        projected: dict[str, Any] = {"prompt": prompt}
        if response_column:
            response = column(row, response_column, "response")
            if response is None:
                raise ValueError("dataset response values must not be None")
            projected["reference"] = response
        if id_column:
            source_id = column(row, id_column, "id")
            if source_id is None:
                raise ValueError("dataset source ID values must not be None")
            projected["source_id"] = source_id
        if normalized_metadata_columns:
            selected_metadata = {name: column(row, name, "metadata") for name in normalized_metadata_columns}
            projected["dataset_row"] = normalize_json_dict(selected_metadata, context="dataset metadata")
        if modality != "text":
            media_column = image_column if modality == "image" else video_column
            media = column(row, media_column, modality)
            if media is None:
                raise ValueError(f"dataset {modality} values must not be None")
            projected[modality] = media
            if modality == "video":
                projected["video_fps"] = normalized_video_fps
        return projected

    def projection_fingerprint() -> str:
        source_fingerprint_fn = getattr(dataset, "cursor_fingerprint", None)
        if callable(source_fingerprint_fn):
            source_fingerprint = str(source_fingerprint_fn())
        else:
            arrow_fingerprint = getattr(dataset, "_fingerprint", None)
            source_fingerprint = "" if arrow_fingerprint is None else str(arrow_fingerprint)
        if not source_fingerprint:
            raise ValueError("prompt projection exact resume requires a stable source fingerprint")
        identity = {
            "source_fingerprint": source_fingerprint,
            "source_name": source_name,
            "revision": revision,
            "subset": subset,
            "split": split,
            "prompt_column": prompt_column,
            "response_column": response_column,
            "image_column": image_column,
            "video_column": video_column,
            "id_column": id_column,
            "metadata_columns": normalized_metadata_columns,
            "modality": modality,
            "video_fps": normalized_video_fps,
            "seed": seed,
            "row_map": row_map_identity,
            "length": len(dataset),
        }
        serialized = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return f"{fingerprint_prefix}-sha256=" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Row 0 now, not at the first fetch: Arrow's null-fill gives every part of the column the signature, so the
    # first row carries it whenever any row does, and a null-filled `rl.data.name` source is refused while the
    # data role is built (before placement and model construction). A defect confined to a later row is still
    # refused at that row's fetch.
    project(dataset[0])
    return map_prompt_items(
        dataset,
        "prompt",
        "reference" if response_column else "",
        project,
        cursor_fingerprint=projection_fingerprint,
    )


def hf_items(cfg: Any, row_map: Any | None = None) -> _ColumnMapped:
    """Load a HuggingFace dataset named by `cfg` and rename its columns onto the item contract.

    The sibling of `jsonl_prompts`: same rows, different source. Neither needs a tokenizer — a
    recipe that wants a chat prompt has its `row_map` put a CONVERSATION in the prompt column, and
    the DataPacker renders it.

    Args:
        cfg: a `DatasetConfig` supplying `name`/`subset`/`split`/`prompt_column`/`response_column`.
        row_map: optional `row -> row` applied before the rename; see `_ColumnMapped`.

    Returns:
        A lazy map-style view. Nothing is materialized.
    """
    from datasets import load_dataset  # lazy: requires the `datasets` dep

    ds = load_dataset(cfg.name, cfg.subset, split=cfg.split)
    return map_prompt_items(ds, cfg.prompt_column, cfg.response_column, row_map)


class DatasetEnv:
    """v0: a 1-step Env over a prompt dataset (DESIGN §1.3, degenerate form).

    `reset()` yields the next prompt as an ObsBundle; `step(completion)` ends the episode
    immediately (done=True). Reward is NOT computed here in v0 (a separate RewardProvider scores
    the trajectory, §1.5).

    Traversal — order, shuffle, epoch wrap, shard stride — is `cursor.IndexedCursor`, shared with
    `DatasetPreferenceEnv`. This class only turns an item into an `ObsBundle`. The two used to
    implement traversal separately and had drifted: this one reshuffled every epoch, `GSM8KEnv`
    shuffled once at construction and never again.
    """

    def __init__(
        self,
        dataset: Any,
        shuffle: bool = False,
        seed: int = 0,
        repeat: bool = False,
        shard_index: int | None = None,
        shard_count: int | None = None,
        modality_tags: tuple[str, ...] = ("text",),
        cursor_offset: int = 0,
        cursor_shards: int = 0,
    ) -> None:
        """
        Args:
            dataset: a MAP-STYLE source (`__len__` + `__getitem__`) of rows already in the
                canonical vocabulary -- `prompt`, and optionally `reference` (what `hf_items` /
                `jsonl_prompts` produce by renaming the configured columns); any other keys pass
                through to `representations`. A one-shot iterable is materialized once for backward
                compatibility; generators are not a supported source, because an opaque cursor
                cannot be checkpointed.
            shuffle: reshuffle at every epoch boundary (seeded, so replicas agree).
            seed: RNG seed for the shuffle.
            repeat: wrap at the end of an epoch instead of raising StopIteration.
            modality_tags: what the served rows actually are. Defaults to `("text",)` for the
                prompt datasets this env was written for; a non-text source (e.g. AlpaGym's scene
                ids, which name a simulator scene rather than carrying text) declares its own.
            shard_index: this data replica's index; see `shard_count`.
            shard_count: number of data replicas reading this source. Each keeps the strided slice
                `items[shard_index::shard_count]`, so replicas see DISJOINT prompts. Omit BOTH for
                NO sharding. They are not defaulted from placement — the published coordinates are
                the COLOCATION's, whose replica count can exceed this role's, which would silently
                DROP rows in a fan-out topology (see `_resolve_shard`).
            cursor_offset: prompt draws THIS SHARD skips before serving its first prompt -- the exact
                continuation of a run restarted from a model-only artifact (`rl.data.cursor_offset`,
                `checkpoint/continuation.py::ContinuationPlan.overrides`). With the source run's
                `seed`, `shuffle` and shard coordinates, the first prompt served is the source shard's
                draw `cursor_offset + 1`, not its draw 1; the prompt-id clock starts at the same count,
                as a full-checkpoint resume sets it. `0` serves the sequence from its start.
            cursor_shards: the shard count `cursor_offset` was computed for; a non-zero value that
                differs from this env's `shard_count` is refused, because the per-shard stride
                `permutation[shard_index::shard_count]` changes with the count and no offset into a
                different stride reproduces the source order. `0` asserts nothing.

        Raises:
            ValueError: inconsistent shard coordinates, an empty source, a negative or non-integer
                `cursor_offset`, or a `cursor_shards` that disagrees with the shard count.
        """
        self._shard_index, self._shard_count = _resolve_shard(shard_index, shard_count)
        if type(cursor_offset) is not int or cursor_offset < 0:
            raise ValueError(f"DatasetEnv: cursor_offset must be a non-negative integer, got {cursor_offset!r}")
        if type(cursor_shards) is not int or cursor_shards < 0:
            raise ValueError(f"DatasetEnv: cursor_shards must be a non-negative integer, got {cursor_shards!r}")
        if cursor_shards and cursor_shards != self._shard_count:
            raise ValueError(
                f"DatasetEnv: cursor_offset={cursor_offset} was computed for {cursor_shards} data shard(s) but this "
                f"source is sharded {self._shard_count}-way; the per-shard draw order differs, so the offset would not "
                "continue the source run. Run the continuation with the source run's data-replica count."
            )
        if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            # Materialize a one-shot iterable ONCE so a list-producing helper still works, while
            # generators stop being a supported source (no stable item identity => no cursor).
            dataset = list(dataset)
        if len(dataset) == 0:
            raise ValueError("empty prompt source")
        self._repeat = repeat
        self._modality_tags = modality_tags
        self._dataset = dataset  # the env owns the DATA; the cursor owns only the position
        # One epoch of THIS shard, so `repeat=False` can stop after a single pass.
        if len(dataset) < self._shard_count:
            # Hoisted from `IndexedCursor._rebuild_order`, which only runs on the first `reset()`.
            # Deferred, a 4-row smoke source with 8 data replicas builds all 8 actors and loads
            # every vLLM engine before dying on the first batch; both the pre-cursor `_shard` and
            # `GSM8KEnv` refused it at construction. The cursor keeps its own copy of this check --
            # it is reachable through other constructors -- but the env has `len(dataset)` and the
            # shard count right here, so the expensive path never has to start.
            raise ValueError(
                f"DatasetEnv: dataset has {len(dataset)} item(s) but shard_count="
                f"{self._shard_count} — shard_index {self._shard_index} would get an empty shard. "
                f"Provide >= {self._shard_count} items, or reduce the data-parallel degree."
            )
        self._epoch_len = len(range(self._shard_index, len(dataset), self._shard_count))
        self._cursor = IndexedCursor(
            dataset,
            shard_index=self._shard_index,
            shard_count=self._shard_count,
            shuffle=shuffle,
            seed=seed,
            label="DatasetEnv",
        )
        # Exact continuation: skip the draws the source shard consumed, so `reset()` serves its next
        # prompt. The id clock follows the cursor (`load_state_dict` enforces emitted == consumed), so
        # the continued prompts carry the ids the source run would have stamped.
        self._cursor.advance(cursor_offset)
        # Monotonic; never reset within this instance's lifetime -- see `_stamp_prompt_id` and the
        # resume caveat in `reset()`'s docstring.
        self._emitted = cursor_offset
        self._loaded_checkpoint_provenance: dict[str, int | str] | None = None

    def reset(self) -> ObsBundle:
        """The next prompt.

        Returns:
            An `ObsBundle` carrying the item's fields, tagged with `modality_tags`, with a fresh id:
            deterministic and shard-scoped (`_stamp_prompt_id`), so two identically-configured envs
            (same seed, same shard coordinates) emit the SAME id sequence -- the property that makes
            `sorted(key=(group_id, id))` (`workers/dispatch.py::select_chunk`) a canonical
            order ACROSS runs, not just within one.

            Resume caveat: a fresh process starts `self._emitted` at `cursor_offset` (0 unless the run
            is an exact continuation, see `__init__`), and a full-checkpoint resume restores it through
            `load_state_dict`; without either, ids repeat across a restart boundary. That is safe today
            because the experience buffer is drained every step (no id from before the restart is ever
            compared against one from after); it constrains any future off-policy/staleness work that
            keeps trajectories alive across a restart.

        Raises:
            StopIteration: the shard is exhausted and `repeat=False`.
        """
        if not self._repeat and self._cursor.consumed >= self._epoch_len:
            raise StopIteration
        item = self._dataset[self._cursor.next_index()]
        # Stamp a deterministic, shard-scoped, monotonic id (GRPO group_id namespace): the rollout
        # keys group_id off obs.id so distinct prompts never collide across replicas at fan-in, and
        # determinism (not a random uuid) is what makes the DP partition's sort canonical across
        # runs -- see `_stamp_prompt_id`.
        prompt_id = _stamp_prompt_id(self._shard_index, self._emitted)
        self._emitted += 1
        return ObsBundle(representations=dict(item), modality_tags=self._modality_tags, id=prompt_id)

    def step(self, action: Any) -> tuple[None, None, bool, dict[str, Any]]:
        # 1-step Env: the completion ends the episode. No env-native reward in v0.
        return None, None, True, {}

    # --- convenience used by the v0 runner / rollout ---
    def get_samples(self, n: int) -> list[ObsBundle]:
        """Pull n initial observations (prompts). v0 helper for batch rollout."""
        return [self.reset() for _ in range(n)]

    def state_dict(self) -> dict[str, Any]:
        """Return the shared cursor plus the prompt-ID clock for exact resume."""
        cursor = self._cursor.state_dict()
        if cursor is None:
            raise RuntimeError("DatasetEnv cursor unexpectedly declined an enabled snapshot")
        return {"version": 1, "cursor": cursor, "emitted": self._emitted}

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore the shared cursor and its matching prompt-ID clock."""
        if not isinstance(state, dict) or set(state) != {"version", "cursor", "emitted"}:
            raise ValueError("DatasetEnv checkpoint must contain version, cursor, and emitted")
        if state["version"] != 1:
            raise ValueError(f"unsupported DatasetEnv checkpoint version {state['version']!r}")
        emitted = state["emitted"]
        if type(emitted) is not int or emitted < 0:
            raise ValueError(f"DatasetEnv emitted count must be a non-negative integer, got {emitted!r}")
        if not isinstance(state["cursor"], dict):
            raise ValueError("DatasetEnv cursor checkpoint must be a dictionary")
        self._cursor.load_state_dict(state["cursor"])
        if emitted != self._cursor.consumed:
            raise ValueError(
                f"DatasetEnv emitted count {emitted} disagrees with cursor consumption {self._cursor.consumed}"
            )
        self._emitted = emitted
        return True

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        """Record the immutable checkpoint generation used to restore this source."""
        self._loaded_checkpoint_provenance = dict(provenance)

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        """Return the restored checkpoint generation, when any."""
        return None if self._loaded_checkpoint_provenance is None else dict(self._loaded_checkpoint_provenance)


class TrainValidationEnv:
    """One existing data-role body with isolated training and held-out cursors.

    Training methods delegate to ``training_env`` unchanged. Validation owns a restartable finite
    pass over an explicit map-style source. ``max_samples`` is applied to the global stable order
    before replica sharding, so the cap retains the same meaning at every replica count.
    """

    def __init__(
        self,
        training_env: Any,
        validation_dataset: Any,
        *,
        seed: int = 0,
        max_samples: int | None = None,
        shard_index: int | None = None,
        shard_count: int | None = None,
        stream_fingerprint: str = "",
    ) -> None:
        """Bind the training Env and one explicit finite held-out source.

        Args:
            training_env: Existing training data-role body.
            validation_dataset: Map-style held-out rows in the canonical prompt/reference vocabulary.
            seed: Validation-run seed retained as part of the stable prompt identity.
            max_samples: Optional cap applied before sharding.
            shard_index: This data replica's validation shard.
            shard_count: Number of data replicas sharing the held-out pass.
            stream_fingerprint: Optional resolved logical stream identity. When omitted, derive one
                from ``validation_dataset``, including framework-side mapping transforms.

        Raises:
            TypeError: Validation data is not map-style.
            ValueError: Validation data/cap cannot produce a non-empty global pass.
        """
        if not (hasattr(validation_dataset, "__len__") and hasattr(validation_dataset, "__getitem__")):
            raise TypeError("TrainValidationEnv requires a map-style validation dataset (__len__ + __getitem__)")
        if len(validation_dataset) == 0:
            raise ValueError("empty validation prompt source")
        self._shard_index, self._shard_count = _resolve_shard(shard_index, shard_count)
        limit = len(validation_dataset) if max_samples is None else min(int(max_samples), len(validation_dataset))
        if limit < 1:
            raise ValueError(f"validation max_samples must retain at least one row, got {max_samples}")
        self._training_env: Any = training_env
        self._validation_dataset: Any = validation_dataset
        self._validation_indices: list[int] = list(range(limit))[self._shard_index :: self._shard_count]
        self._validation_position: int = 0
        self._validation_seed: int = int(seed)
        from cosmos_framework.posttrain.roles.env.identity import dataset_fingerprint

        self._validation_fingerprint: str = stream_fingerprint or dataset_fingerprint(validation_dataset)
        if not self._validation_fingerprint:
            raise ValueError("validation source fingerprint must be non-empty")

    def reset(self) -> ObsBundle:
        """Delegate one training reset without touching the held-out cursor."""
        return self._training_env.reset()

    def step(self, action: Any) -> tuple[Any, Any, bool, dict[str, Any]]:
        """Delegate the training Env transition."""
        return self._training_env.step(action)

    def get_samples(self, n: int) -> list[ObsBundle]:
        """Delegate a training batch without touching the held-out cursor."""
        return self._training_env.get_samples(n)

    def begin_validation(self, batch_size: int = 1) -> int:
        """Restart this replica's finite validation pass and return its logical sample count."""
        if batch_size < 1:
            raise ValueError(f"validation batch_size must be >= 1, got {batch_size}")
        self._validation_position = 0
        return len(self._validation_indices)

    def get_validation_samples(self, n: int, *, expected_count: int | None = None) -> list[ObsBundle]:
        """Return up to ``n`` held-out observations and enforce the controller's metadata contract.

        ``expected_count`` is scattered as one integer per data replica. Checking it here keeps the
        prompt payload inside the data plane; the controller can forward the returned handle to
        rollout without resolving every replica's observations merely to inspect their lengths.
        """
        if n < 1:
            raise ValueError(f"validation batch size must be >= 1, got {n}")
        if expected_count is not None and (
            isinstance(expected_count, bool) or not isinstance(expected_count, int) or not 0 <= expected_count <= n
        ):
            raise ValueError(f"validation expected_count must be an integer in [0, {n}], got {expected_count!r}")
        stop = min(self._validation_position + n, len(self._validation_indices))
        indices = self._validation_indices[self._validation_position : stop]
        self._validation_position = stop
        observations: list[ObsBundle] = []
        for index in indices:
            item = self._validation_dataset[index]
            observations.append(
                ObsBundle(
                    representations=dict(item),
                    modality_tags=("text",),
                    id=f"validation:{self._validation_fingerprint}:{self._validation_seed}:{index}",
                )
            )
        if expected_count is not None and len(observations) != expected_count:
            raise RuntimeError(
                f"validation source shard {self._shard_index} returned {len(observations)} samples, "
                f"expected {expected_count}"
            )
        return observations

    def validation_source_fingerprint(self) -> str:
        """Return the held-out source identity for the validation report."""
        return self._validation_fingerprint


class ArithmeticEnv:
    """v0 synthetic 1-step Env for GRPO: reset() -> an addition prompt + ground-truth
    answer; step() ends the episode. Difficulty via max_operand (tune so the model is
    sometimes right, sometimes wrong -> reward has variance -> nonzero advantage).
    Task is decoupled from the reward fn (swap to GSM8K later by changing both)."""

    def __init__(
        self,
        seed: int = 0,
        max_operand: int = 50,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ):
        """
        Args:
            seed: base RNG seed.
            max_operand: difficulty knob.
            shard_index: this data replica's index; see `shard_count`.
            shard_count: number of data replicas. This env SYNTHESIZES prompts rather than reading a
                fixed corpus, so "disjoint" means independent streams: the seed is offset per
                replica. Every replica sharing one seed emits the IDENTICAL prompt sequence, which
                makes the data-parallel axis pure duplication — R times the compute for one
                replica's data, with gradients averaging copies of themselves.

        Raises:
            ValueError: on inconsistent shard coordinates.
        """
        shard_index, _ = _resolve_shard(shard_index, shard_count)
        self._rng = random.Random(seed + shard_index)
        self.max_operand = max_operand
        self._shard_index = shard_index
        # Monotonic; never reset within this instance's lifetime -- see `_stamp_prompt_id` and the
        # resume caveat below.
        self._emitted = 0

    def reset(self) -> ObsBundle:
        """The next synthetic addition prompt.

        Returns:
            An `ObsBundle` with a fresh id: deterministic and shard-scoped (`_stamp_prompt_id`),
            NOT drawn from `self._rng` -- it is a per-prompt identity (GRPO `group_id` namespace)
            that must stay unique across replicas even when two seeds happen to emit identical
            prompt text, which `self._rng` alone could not guarantee. This used to be
            `uuid.uuid4().hex`; that was globally unique but random, which made the DP partition's
            `sorted(key=(group_id, id))` order canonical only WITHIN a run, not reproducible ACROSS
            two seeded runs. The deterministic id fixes that while keeping the same uniqueness
            guarantee: `shard_index` (distinct per replica by construction, see `_resolve_shard`)
            separates replicas, `self._emitted` (monotonic, survives an epoch wrap) separates draws
            within one replica.

            Resume caveat: `self._emitted` restarts at 0 in a fresh process, so ids repeat across a
            resume boundary. Safe today because the experience buffer is drained every step; a
            constraint for any future off-policy/staleness work.
        """
        a = self._rng.randint(0, self.max_operand)
        b = self._rng.randint(0, self.max_operand)
        prompt_id = _stamp_prompt_id(self._shard_index, self._emitted)
        self._emitted += 1
        return ObsBundle(
            representations={"prompt": f"{a} + {b} =", "reference": a + b}, modality_tags=("text",), id=prompt_id
        )

    def step(self, action: Any) -> tuple[None, None, bool, dict[str, Any]]:
        return None, None, True, {}

    def get_samples(self, n: int) -> list[ObsBundle]:
        return [self.reset() for _ in range(n)]
