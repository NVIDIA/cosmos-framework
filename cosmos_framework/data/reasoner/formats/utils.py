# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-format primitives used by every per-format helper module."""

import json
from pathlib import Path
from typing import Optional, Union


class FormatError(Exception):
    """Raised when a format-aware helper cannot proceed (e.g. missing directory,
    malformed input, unrecognizable contents)."""


def read_json_object(path: Path) -> dict:
    """Parse a JSON file and require the top-level value to be an object.

    Raises ``FormatError`` when the file is missing, cannot be parsed as
    JSON, or parses to something other than a JSON object (e.g. a top-level
    array, number, or string). Returning ``None`` for these cases was the
    previous behavior; it conflated three distinct failure modes (and a
    top-level array silently masquerading as "not a dict") into a single
    sentinel, forcing callers to write "if data is None" with no way to
    diagnose. Callers now ``try``/``except FormatError`` and surface the
    specific message.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FormatError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise FormatError(f"Invalid JSON in {path.name}: {e.msg} (line {e.lineno})") from e
    except OSError as e:
        raise FormatError(f"Cannot read {path.name}: {e}") from e
    if not isinstance(data, dict):
        raise FormatError(f"{path.name}: top-level JSON value is {type(data).__name__}, expected object")
    return data


def get_metadata_type(data: dict) -> Optional[str]:
    """Extract ``metadata.type`` from a parsed JSON document."""
    meta = data.get("metadata")
    if isinstance(meta, dict):
        return meta.get("type")
    return None


def media_dest_basename(
    media_src: Union[str, Path],
    dataset_root: Union[str, Path],
) -> str:
    """Flat, source-traceable basename for *media_src* under *dataset_root*.

    Path-relative-to-root with ``"/"`` replaced by ``"--"`` so two scenes
    sharing a basename (e.g. ``camera_01.mp4``) land at distinct files when
    copied into the output dataset.
    """
    rel = Path(media_src).resolve().relative_to(Path(dataset_root).resolve())
    return rel.as_posix().replace("/", "--")
