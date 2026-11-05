# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Subject-bound layout conditioning for the unified multi-reference annotation.

The unified annotation prompt has three sections::

    [Entities]
    <subject 1>: the woman from <image 1>, wearing the dress from <image 2>
    <subject 2>: a small brown dog (text-only, no reference)

    [Scene] The overall scene is ... from <image 3>, ...
    [Pose] ...      (optional)
    [Style] ...     (optional)

    [Composition]
    <subject 1> sits beside <subject 2> ...

Boxes are bound to subjects, not to reference images: ``subject_bboxes`` is
``[{"subject": k, "boxes": [[x1, y1, x2, y2], ...]}]`` in target-normalized xyxy.
Subjects without a box (text-only or ungrounded) are absent from the layout.

The condition adds one ``[Layout]`` line right after ``[Scene]`` (the position
the annotation format renders it at). ``rendered`` mode also produces a white
layout image with one numbered outline per subject box; it becomes reference
``<image N+1>`` and is appended after reference shuffling, so its tag is outside
the shuffled range and never renumbered. ``rendered_paste`` mode additionally
pastes each subject's content and worn-item references into its boxes (tiled in
order, aspect ratio kept) under the outlines. Badges show subject numbers, which shuffling never
changes. References themselves are not modified.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageColor, ImageDraw, ImageFont

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.augmentors.multi_reference_bbox_condition import (
    REFERENCE_BBOX_COLORS,
    BBoxCoordinateFormat,
    format_coordinate_box,
)

BoundingBox = tuple[float, float, float, float]
SubjectLayoutMode = Literal["none", "rendered", "rendered_paste", "coordinates"]
SUBJECT_LAYOUT_MODES: tuple[str, ...] = ("none", "rendered", "rendered_paste", "coordinates")
# Modes that append a layout image as the last reference.
IMAGE_LAYOUT_MODES: tuple[str, ...] = ("rendered", "rendered_paste")

# Same palette as reference-numbered bbox conditioning, indexed by subject number.
SUBJECT_COLORS: tuple[str, ...] = REFERENCE_BBOX_COLORS
SUBJECT_LAYOUT_IMAGE_KEY = "subject_layout_image"

RENDERED_LAYOUT_NOTE = "Each numbered box marks where the subject with that number appears."
PASTED_LAYOUT_NOTE = (
    "Each numbered box marks where the subject with that number appears and shows the reference images of that subject."
)
_SCENE_PREFIX = "[Scene]"
# Lines rendered after [Scene] in the unified format; the [Layout] line goes before the first of them.
_AFTER_SCENE_MARKERS: tuple[str, ...] = ("\n[Pose] ", "\n[Style] ", "\n\n[Composition]")


@dataclass(frozen=True)
class SubjectBoxes:
    subject: int
    boxes: tuple[BoundingBox, ...]


def parse_subject_bboxes(value: object) -> tuple[SubjectBoxes, ...]:
    """Validate the ``subject_bboxes`` payload (JSON text or decoded list).

    ``None`` and ``[]`` both mean no layout. Subjects with no boxes are dropped.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("subject_bboxes must be a list.")
    seen: set[int] = set()
    result: list[SubjectBoxes] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Each subject bbox entry must be an object.")
        subject = entry.get("subject")
        if isinstance(subject, bool) or not isinstance(subject, int) or subject < 1:
            raise ValueError(f"Subject number must be a positive integer, got {subject!r}.")
        if subject in seen:
            raise ValueError(f"Duplicate subject {subject} in subject_bboxes.")
        seen.add(subject)
        raw_boxes = entry.get("boxes")
        if not isinstance(raw_boxes, list):
            raise ValueError(f"Boxes for subject {subject} must be a list.")
        boxes: list[BoundingBox] = []
        for raw_box in raw_boxes:
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise ValueError(f"Each box for subject {subject} must contain four xyxy coordinates.")
            if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in raw_box):
                raise ValueError(f"Coordinates for subject {subject} must be finite numbers.")
            x1, y1, x2, y2 = (float(c) for c in raw_box)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Boxes for subject {subject} must be nonempty xyxy rectangles in [0, 1].")
            boxes.append((x1, y1, x2, y2))
        if boxes:
            result.append(SubjectBoxes(subject=subject, boxes=tuple(boxes)))
    return tuple(result)


def drop_shared_boxes(subjects: tuple[SubjectBoxes, ...]) -> tuple[SubjectBoxes, ...]:
    """Remove boxes assigned to more than one subject; subjects left without boxes are dropped.

    Boxes are copied from reference geometry, so two subjects get the same box when they
    come from one reference image (several things in one crop, or repeated copies of one
    object) or when upstream gave two references the same box. Such a box does not say
    where each subject is.
    """
    counts = Counter(box for item in subjects for box in set(item.boxes))
    result: list[SubjectBoxes] = []
    for item in subjects:
        boxes = tuple(box for box in item.boxes if counts[box] == 1)
        if boxes:
            result.append(SubjectBoxes(subject=item.subject, boxes=boxes))
    return tuple(result)


_SUBJECT_TAG_RE = re.compile(r"<subject (\d+)>")


def defined_subjects(prompt: str) -> set[int]:
    """Subject numbers defined in the ``[Entities]`` section (``<subject k>:`` line starts)."""
    entities = prompt.split("\n\n", 1)[0]
    return {
        int(match.group(1))
        for line in entities.splitlines()
        if (match := _SUBJECT_TAG_RE.match(line)) is not None and line[match.end() :].startswith(":")
    }


def insert_layout_line(prompt: str, line: str) -> str:
    """Insert a ``[Layout]`` line directly after the ``[Scene]`` line of a unified annotation."""
    scene_start = prompt.find(_SCENE_PREFIX)
    if scene_start < 0:
        raise ValueError("Prompt has no [Scene] line; it is not a unified annotation.")
    if "\n[Layout] " in prompt:
        raise ValueError("Prompt already has a [Layout] line.")
    positions = [position for marker in _AFTER_SCENE_MARKERS if (position := prompt.find(marker, scene_start)) >= 0]
    if not positions:
        raise ValueError("Prompt has no [Composition] section after [Scene].")
    insert_at = min(positions)
    return prompt[:insert_at] + "\n" + line + prompt[insert_at:]


def rendered_layout_line(layout_image_index: int, note: str = RENDERED_LAYOUT_NOTE) -> str:
    return (
        f"[Layout] The overall layout of all the mentioned entities is provided in <image {layout_image_index}>. {note}"
    )


def coordinates_layout_line(
    subjects: tuple[SubjectBoxes, ...], coordinate_format: BBoxCoordinateFormat = "integer_1000"
) -> str:
    """Boxes as text, formatted like reference-numbered bbox coordinates (integers in [0, 1000] by default)."""
    parts = []
    for item in subjects:
        boxes = ", ".join(format_coordinate_box(box, coordinate_format) for box in item.boxes)
        parts.append(f"<subject {item.subject}> {boxes}")
    scale = "integers in [0, 1000]" if coordinate_format == "integer_1000" else "values in [0, 1]"
    return (
        f"[Layout] The positions of the subjects are given as [x1, y1, x2, y2] boxes normalized to {scale}: "
        + "; ".join(parts)
        + "."
    )


Rect = tuple[int, int, int, int]


def _overlap(a: Rect, b: Rect) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _label_candidates(box: Rect, width: int, height: int) -> list[tuple[int, int]]:
    """Badge anchors around a box, in preference order: corners inside, then just outside."""
    x1, y1, x2, y2 = box
    return [
        (x1, y1),
        (x2 - width, y1),
        (x1, y1 - height),
        (x2 - width, y1 - height),
        (x1, y2 - height),
        (x2 - width, y2 - height),
        (x1, y2),
        ((x1 + x2 - width) // 2, (y1 + y2 - height) // 2),
    ]


def place_labels(boxes: list[Rect], size: tuple[int, int], label_size: tuple[int, int]) -> list[Rect]:
    """Greedy deterministic badge placement: the first on-canvas anchor that overlaps no earlier badge,
    else the anchor with the least overlap."""
    canvas_w, canvas_h = size
    width, height = label_size
    placed: list[Rect] = []
    for box in boxes:
        best: Rect | None = None
        best_cost: int | None = None
        for anchor_x, anchor_y in _label_candidates(box, width, height):
            x = max(0, min(anchor_x, canvas_w - width))
            y = max(0, min(anchor_y, canvas_h - height))
            rect = (x, y, x + width, y + height)
            cost = sum(_overlap(rect, other) for other in placed)
            if cost == 0:
                best = rect
                break
            if best_cost is None or cost < best_cost:
                best, best_cost = rect, cost
        assert best is not None
        placed.append(best)
    return placed


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    return image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BICUBIC
    )


def _grid_columns(images: Sequence[Image.Image], width: int, height: int) -> int:
    """Column count whose grid shows the largest total pasted area (ties prefer fewer columns)."""
    best_columns, best_area = 1, -1.0
    for columns in range(1, len(images) + 1):
        rows = math.ceil(len(images) / columns)
        cell_width, cell_height = width / columns, height / rows
        area = sum(
            image.width * image.height * min(cell_width / image.width, cell_height / image.height) ** 2
            for image in images
        )
        if area > best_area:
            best_columns, best_area = columns, area
    return best_columns


# A subject's references avoid boxes pasted over them unless the free part is smaller than this share of the box.
MIN_FREE_PASTE_AREA = 0.25


def _largest_free_rect(box: Rect, obstacles: list[Rect]) -> Rect:
    """Largest axis-aligned rectangle inside ``box`` that overlaps none of ``obstacles``."""
    x1, y1, x2, y2 = box
    xs = sorted({x1, x2, *(min(max(x, x1), x2) for o in obstacles for x in (o[0], o[2]))})
    best, best_area = (x1, y1, x1, y1), 0
    for i, left in enumerate(xs):
        for right in xs[i + 1 :]:
            spans = sorted(
                (max(o[1], y1), min(o[3], y2))
                for o in obstacles
                if o[0] < right and o[2] > left and o[1] < y2 and o[3] > y1
            )
            cursor = y1
            for top, bottom in [*spans, (y2, y2)]:
                if (right - left) * (top - cursor) > best_area:
                    best, best_area = (left, cursor, right, top), (right - left) * (top - cursor)
                cursor = max(cursor, bottom)
    return best


def _tile(canvas: Image.Image, images: Sequence[Image.Image], area: Rect) -> None:
    """Tile ``images`` in order, row by row, inside ``area``; each is fitted and centered in its cell."""
    left, top = area[0], area[1]
    area_width, area_height = max(1, area[2] - area[0]), max(1, area[3] - area[1])
    columns = _grid_columns(images, area_width, area_height)
    rows = math.ceil(len(images) / columns)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        cell_left = left + column * area_width // columns
        cell_top = top + row * area_height // rows
        cell_width = left + (column + 1) * area_width // columns - cell_left
        cell_height = top + (row + 1) * area_height // rows - cell_top
        fitted = _fit(image, max(1, cell_width), max(1, cell_height))
        canvas.paste(
            fitted, (cell_left + (cell_width - fitted.width) // 2, cell_top + (cell_height - fitted.height) // 2)
        )


def paste_subject_references(
    canvas: Image.Image, subjects: tuple[SubjectBoxes, ...], subject_images: Mapping[int, Sequence[Image.Image]]
) -> None:
    """Paste each subject's references into its boxes.

    A subject's references (content images first, then worn-item references) are tiled
    in order, row by row, in the grid that shows them largest; each keeps its aspect ratio
    and is centered in its cell. Larger boxes are pasted first, so smaller ones (e.g. held
    objects) stay visible on top. The tiles use the largest part of the box that no later
    (smaller) box of another subject covers, so no reference is hidden; when that part is
    under ``MIN_FREE_PASTE_AREA`` of the box, the whole box is used. Subjects without images
    keep an empty box.
    """
    width, height = canvas.size
    items: list[tuple[int, Rect]] = [
        (item.subject, (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))
        for item in subjects
        if subject_images.get(item.subject)
        for x1, y1, x2, y2 in item.boxes
    ]
    items.sort(key=lambda item: -(item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
    for position, (subject, box) in enumerate(items):
        later = [other for other_subject, other in items[position + 1 :] if other_subject != subject]
        free = _largest_free_rect(box, later)
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        use_free = (free[2] - free[0]) * (free[3] - free[1]) >= MIN_FREE_PASTE_AREA * box_area
        _tile(canvas, subject_images[subject], free if use_free else box)


def render_subject_layout(
    subjects: tuple[SubjectBoxes, ...],
    size: tuple[int, int],
    subject_images: Mapping[int, Sequence[Image.Image]] | None = None,
) -> Image.Image:
    """White canvas at the target size with one numbered outline per subject box.

    With ``subject_images`` (subject -> its references), the references are pasted into
    their boxes first and the outlines and badges are drawn on top.
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Layout dimensions must be positive.")
    canvas = Image.new("RGB", size, "white")
    if subject_images:
        paste_subject_references(canvas, subjects, subject_images)
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(size) / 256))
    font_size = max(12, round(min(size) / 28))
    font = ImageFont.load_default(size=font_size)
    margin = max(2, font_size // 5)

    items: list[tuple[int, Rect]] = []
    for item in subjects:
        for x1, y1, x2, y2 in item.boxes:
            rect = (
                min(width - 1, round(x1 * width)),
                min(height - 1, round(y1 * height)),
                min(width - 1, round(x2 * width)),
                min(height - 1, round(y2 * height)),
            )
            items.append((item.subject, rect))
    # Every badge has the size of the widest label.
    widest = max((str(subject) for subject, _ in items), key=len, default="0")
    left, top, right, bottom = draw.textbbox((0, 0), widest, font=font)
    label_size = (math.ceil(right - left) + 2 * margin, math.ceil(bottom - top) + 2 * margin)

    for subject, rect in items:
        draw.rectangle(rect, outline=SUBJECT_COLORS[(subject - 1) % len(SUBJECT_COLORS)], width=line_width)
    for (subject, _), badge in zip(items, place_labels([rect for _, rect in items], size, label_size), strict=True):
        color = SUBJECT_COLORS[(subject - 1) % len(SUBJECT_COLORS)]
        draw.rectangle((badge[0], badge[1], badge[2] - 1, badge[3] - 1), fill=color)
        red, green, blue = ImageColor.getrgb(color)[:3]
        text_color = "white" if 299 * red + 587 * green + 114 * blue < 140000 else "black"
        text = str(subject)
        t_left, t_top, t_right, _ = draw.textbbox((0, 0), text, font=font)
        x = badge[0] + (label_size[0] - (t_right - t_left)) // 2 - t_left
        draw.text((x, badge[1] + margin - t_top), text, fill=text_color, font=font)
    return canvas


def build_subject_layout_condition(
    prompt: str,
    subject_bboxes: object,
    num_references: int,
    target_size: tuple[int, int],
    mode: SubjectLayoutMode,
    *,
    coordinate_format: BBoxCoordinateFormat = "integer_1000",
    subject_images: Mapping[int, Sequence[Image.Image]] | None = None,
) -> tuple[str, Image.Image | None]:
    """Return the conditioned prompt and, in image modes, the layout image for ``<image N+1>``.

    ``rendered_paste`` pastes ``subject_images`` (subject -> its content references, then
    its worn-item references) into the boxes; subjects missing from it keep an empty box.

    Without boxes (NULL or empty geometry, or only boxes shared by several subjects)
    the prompt is returned unchanged, so a layout recipe can mix in populations that
    have no geometry as plain samples.
    """
    if mode not in SUBJECT_LAYOUT_MODES:
        raise ValueError(f"Unknown subject layout mode: {mode!r}.")
    if coordinate_format not in ("integer_1000", "normalized_01"):
        raise ValueError(f"Unknown bbox coordinate format: {coordinate_format!r}.")
    if mode == "none":
        return prompt, None
    subjects = parse_subject_bboxes(subject_bboxes)
    undefined = {item.subject for item in subjects} - defined_subjects(prompt)
    if undefined:
        raise ValueError(f"Boxes refer to subjects {sorted(undefined)} that the prompt does not define.")
    subjects = drop_shared_boxes(subjects)
    if not subjects:
        return prompt, None
    if mode == "coordinates":
        return insert_layout_line(prompt, coordinates_layout_line(subjects, coordinate_format)), None
    if mode == "rendered_paste":
        layout_image = render_subject_layout(subjects, target_size, subject_images or {})
        return insert_layout_line(prompt, rendered_layout_line(num_references + 1, PASTED_LAYOUT_NOTE)), layout_image
    layout_image = render_subject_layout(subjects, target_size)
    return insert_layout_line(prompt, rendered_layout_line(num_references + 1)), layout_image


def subject_reference_images(
    subject_paste_images: Mapping[int, Sequence[int]], references: list[Image.Image]
) -> dict[int, list[Image.Image]]:
    """Subject -> its reference images, from 1-based reference indices."""
    images: dict[int, list[Image.Image]] = {}
    for subject, indices in subject_paste_images.items():
        for index in indices:
            if not 1 <= index <= len(references):
                raise ValueError(
                    f"Subject {subject} refers to <image {index}> but there are {len(references)} references."
                )
        images[subject] = [references[index - 1] for index in indices]
    return images


class BuildSubjectLayoutCondition(Augmentor):
    """Add the subject layout to the prompt before references are shuffled.

    Reads ``annotations["subject_bboxes"]``, ``source_image`` and ``target_image``
    (after resize); ``rendered_paste`` also reads ``annotations["subject_paste_images"]``
    (subject -> 1-based indices of its content and worn-item references, in the unshuffled order).
    The layout image is kept under ``subject_layout_image`` and appended by
    :class:`AppendSubjectLayoutImage` after the shuffle.
    """

    def __init__(
        self,
        mode: SubjectLayoutMode,
        coordinate_format: BBoxCoordinateFormat = "integer_1000",
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__(input_keys or [])
        if mode not in ("rendered", "rendered_paste", "coordinates"):
            raise ValueError("The subject layout augmentor requires rendered, rendered_paste or coordinates mode.")
        if coordinate_format not in ("integer_1000", "normalized_01"):
            raise ValueError(f"Unknown bbox coordinate format: {coordinate_format!r}.")
        self.mode: SubjectLayoutMode = mode
        self.coordinate_format: BBoxCoordinateFormat = coordinate_format

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            references = data_dict["source_image"]
            subject_images = None
            if self.mode == "rendered_paste":
                subject_images = subject_reference_images(data_dict["annotations"]["subject_paste_images"], references)
            prompt, layout_image = build_subject_layout_condition(
                prompt=data_dict["editing_instruction"],
                subject_bboxes=data_dict["annotations"].get("subject_bboxes"),
                num_references=len(references),
                target_size=data_dict["target_image"].size,
                mode=self.mode,
                coordinate_format=self.coordinate_format,
                subject_images=subject_images,
            )
        except (KeyError, ValueError) as error:
            log.warning(
                f"Skipping subject layout sample {data_dict.get('__key__', 'unknown')!r}: {error}", rank0_only=False
            )
            return None
        data_dict["editing_instruction"] = prompt
        if layout_image is not None:
            data_dict[SUBJECT_LAYOUT_IMAGE_KEY] = layout_image
        return data_dict


class AppendSubjectLayoutImage(Augmentor):
    """Append the rendered layout as the last reference, after reference shuffling."""

    def __init__(self, input_keys: list[str] | None = None) -> None:
        super().__init__(input_keys or [])

    def __call__(self, data_dict: dict) -> dict | None:
        layout_image = data_dict.pop(SUBJECT_LAYOUT_IMAGE_KEY, None)
        if layout_image is not None:
            data_dict["source_image"] = [*data_dict["source_image"], layout_image]
        return data_dict
