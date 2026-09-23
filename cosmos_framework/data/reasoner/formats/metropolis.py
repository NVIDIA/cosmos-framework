# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the metropolis-v3.0 annotation format.

Used by both the validator (which iterates ``contextual/``) and the converter
(which iterates ``task/``). Free functions; no class state.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from cosmos_framework.data.reasoner.formats.utils import FormatError, get_metadata_type, read_json_object


class RawType(Enum):
    """Raw data type (media type) for metropolis scenes."""

    IMAGE = "image"
    VIDEO = "video"
    AUTO = "auto"


def detect_raw_type(scene_path: Path) -> RawType:
    """Detect raw data type from a metropolis scene's contextual directory.

    Reads ``metadata.type`` from each JSON file under ``contextual/``. Returns
    ``RawType.VIDEO`` if any video-typed file is present, ``RawType.IMAGE`` if
    only image-typed files are present.

    Raises ``FormatError`` if no recognizable file is found.
    """
    contextual_path = scene_path / "contextual"
    if not contextual_path.exists():
        raise FormatError(f"Cannot detect raw type: contextual/ directory not found in {scene_path}")

    has_video = False
    has_image = False

    for json_file in contextual_path.glob("*.json"):
        try:
            data = read_json_object(json_file)
        except FormatError:
            continue  # discovery-pass: ignore unparseable / non-object JSON
        meta_type = get_metadata_type(data)
        if meta_type == "video":
            has_video = True
        elif meta_type == "image":
            has_image = True

    if has_video:
        return RawType.VIDEO
    if has_image:
        return RawType.IMAGE
    raise FormatError(
        f"Cannot detect raw type: no file with metadata.type 'video' or 'image' found in {contextual_path}"
    )


def find_scenes(root: Path, marker_dir: str = "contextual") -> List[Path]:
    """Recursively find scene directories that contain ``marker_dir``.

    Validators key on ``contextual/`` (the default); the converter passes
    ``marker_dir="task"`` because it only cares about scenes that have task
    items to convert.
    """
    if (root / marker_dir).is_dir():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        scene for child in sorted(root.iterdir()) if child.is_dir() for scene in find_scenes(child, marker_dir)
    )


def find_media_file(
    scene_path: Path,
    media_id: str,
    fmt: str,
) -> Optional[Path]:
    """Resolve the media file declared by a contextual entry.

    The metropolis-v3.0 contextual schema requires both ``image_id`` /
    ``video_id`` AND a sibling ``format`` field, so the file in ``raw/``
    lives at exactly ``{media_id}.{fmt}``. We trust that contract directly:
    no multi-extension probing, no stem-match fallback. If a caller
    violates the convention (e.g. embeds an extension in the id), the
    natural "file not found" tells them so via the converter's
    skip-and-report path.
    """
    candidate = scene_path / "raw" / f"{media_id}.{fmt}"
    return candidate if candidate.exists() else None


def build_media_format_index(
    scene_path: Path,
    warnings: List[str],
) -> Dict[str, str]:
    """Return ``{media_id: format}`` for every video/image contextual entry
    in this scene.

    Reads each ``contextual/*.json`` once. If two files declare the same
    ``media_id`` with conflicting ``format`` values, the first wins and a
    warning is appended to ``warnings``.

    Defensive ``isinstance`` checks here because callers (the converters)
    do not gate on prior schema validation; users wanting type-strict
    guarantees should run ``cosmos-dataset validate metropolis-v3.0`` first.
    """
    index: Dict[str, str] = {}
    contextual_path = scene_path / "contextual"
    if not contextual_path.exists():
        return index

    for json_file in sorted(contextual_path.glob("*.json")):
        try:
            data = read_json_object(json_file)
        except FormatError:
            continue
        meta_type = get_metadata_type(data)
        if meta_type == "video":
            id_key = "video_id"
        elif meta_type == "image":
            id_key = "image_id"
        else:
            continue
        media_id = data.get(id_key)
        fmt_ext = data.get("format")
        if not (isinstance(media_id, str) and isinstance(fmt_ext, str)):
            continue
        existing = index.get(media_id)
        if existing is not None and existing != fmt_ext:
            warnings.append(
                f"contextual/: media_id '{media_id}' declared with "
                f"conflicting formats ('{existing}' vs '{fmt_ext}'); "
                f"using the first."
            )
            continue
        index[media_id] = fmt_ext
    return index


def iter_task_items(
    scene_path: Path,
    supported_tasks: frozenset,
    task_filter: Optional[List[str]],
    warnings: List[str],
) -> Iterator[Tuple[Path, str, int, dict]]:
    """Yield ``(task_file, task_type, idx, item)`` for each item in each task file.

    Applies two filters:
    - ``supported_tasks``: skips files whose task type is not supported (a
      warning is appended for explicitly typed but unsupported files).
    - ``task_filter``: skips files whose task type is not in this user-supplied
      list (silent).

    ``warnings`` is appended to in place.
    """
    task_dir = scene_path / "task"
    if not task_dir.is_dir():
        return

    for task_file in sorted(task_dir.glob("*.json")):
        try:
            task_data = read_json_object(task_file)
        except FormatError as e:
            warnings.append(f"Skipping {task_file.name}: {e}")
            continue

        task_type = get_metadata_type(task_data) or task_data.get("task_type")
        if not task_type or task_type not in supported_tasks:
            if task_type:
                warnings.append(f"Skipping {task_file.name}: unsupported task type '{task_type}'")
            continue

        if task_filter is not None and task_type not in task_filter:
            continue

        for idx, item in enumerate(task_data.get("items", [])):
            yield task_file, task_type, idx, item


class ContextualRequirements:
    """Valid contextual annotation types per raw type.

    Used to validate ``--contextual`` choices and to expand the ``all`` /
    ``complete`` shorthand into the default set for a given raw type. In
    metropolis-v3.0 schemas are matched by ``metadata.type``, not by filename,
    so no filename list is needed here.
    """

    VALID: Dict[RawType, Set[str]] = {
        RawType.IMAGE: {"objects", "tracking"},
        RawType.VIDEO: {"objects", "events", "tracking", "video", "chunks", "msted"},
    }

    COMPLETE: Dict[RawType, List[str]] = {
        RawType.IMAGE: ["objects", "tracking"],
        RawType.VIDEO: ["objects", "events", "tracking"],
    }

    @classmethod
    def is_valid_combination(cls, raw_type: RawType, contextual_type: str) -> bool:
        """Check whether ``contextual_type`` is valid for ``raw_type``."""
        return contextual_type in cls.VALID.get(raw_type, set())
