# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared target-layout conditioning for multi-reference training and inference.

Boxes use target-normalized xyxy coordinates and logical ``input_NN`` groups.
References without boxes are retained, including pose, style, and attribute
donors. A known empty bbox list leaves the images and prompt unchanged.
"""

from __future__ import annotations

import math
from typing import Literal

from PIL import Image, ImageColor, ImageDraw, ImageFont

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log

BoundingBox = tuple[float, float, float, float]
ReferenceBBoxes = dict[str, list[BoundingBox]]
BBoxConditionMode = Literal["none", "rendered", "coordinates"]

# Colors are indexed by the final reference position, including unboxed control
# inputs. Layout is an additional reference and does not consume a palette slot.
REFERENCE_BBOX_COLORS: tuple[str, ...] = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#000000",
    "#0088cc",
    "#dd4477",
    "#66aa00",
    "#aa5500",
)
MAX_BBOX_REFERENCE_IMAGES = len(REFERENCE_BBOX_COLORS)


def parse_reference_bboxes(value: object, reference_keys: list[str]) -> ReferenceBBoxes:
    """Validate the Lance bbox column without requiring boxes for every input."""
    if not isinstance(value, list):
        raise ValueError("reference_bboxes must be a list; NULL represents unknown geometry.")
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("Reference keys must be unique.")
    available = set(reference_keys)
    seen: set[str] = set()
    result: ReferenceBBoxes = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Each reference bbox entry must be an object.")
        group = entry.get("reference_group")
        if not isinstance(group, str) or group not in available:
            raise ValueError(f"BBox reference group {group!r} is absent from the input images.")
        if group in seen:
            raise ValueError(f"Duplicate bbox reference group {group!r}.")
        seen.add(group)
        raw_boxes = entry.get("boxes")
        if not isinstance(raw_boxes, list):
            raise ValueError(f"Boxes for {group!r} must be a list.")
        boxes: list[BoundingBox] = []
        for raw_box in raw_boxes:
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise ValueError(f"Each box for {group!r} must contain four xyxy coordinates.")
            coordinates: list[float] = []
            for coordinate in raw_box:
                if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                    raise ValueError(f"Coordinates for {group!r} must be numeric.")
                if not math.isfinite(coordinate):
                    raise ValueError(f"Coordinates for {group!r} must be finite.")
                coordinates.append(float(coordinate))
            x1, y1, x2, y2 = coordinates
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Boxes for {group!r} must be nonempty xyxy rectangles normalized to [0, 1].")
            boxes.append((x1, y1, x2, y2))
        if boxes:
            result[group] = boxes
    return result


def _draw_reference_number(image: Image.Image, position: tuple[int, int], index: int, color: str) -> None:
    """Draw a readable colored index badge inside the canvas boundary."""
    font_size = max(10, round(min(image.size) / 32))
    font = ImageFont.load_default(size=font_size)
    draw = ImageDraw.Draw(image)
    text = str(index)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    margin = max(2, font_size // 5)
    badge_w = math.ceil(right - left) + 2 * margin
    badge_h = math.ceil(bottom - top) + 2 * margin
    x = max(0, min(position[0], image.width - badge_w))
    y = max(0, min(position[1], image.height - badge_h))
    draw.rectangle((x, y, x + badge_w - 1, y + badge_h - 1), fill=color)
    red, green, blue = ImageColor.getrgb(color)[:3]
    text_color = "white" if 299 * red + 587 * green + 114 * blue < 140000 else "black"
    draw.text((x + margin - left, y + margin - top), text, fill=text_color, font=font)


def build_bbox_condition(
    reference_images: list[Image.Image],
    prompt: str,
    target_size: tuple[int, int],
    reference_keys: list[str],
    reference_bboxes: ReferenceBBoxes,
    mode: BBoxConditionMode,
) -> tuple[list[Image.Image], str]:
    """Append a layout or coordinate section after resizing and reference shuffle.

    ``reference_bboxes`` is the validated result of ``parse_reference_bboxes``.
    ``reference_keys`` follows the current image order, including unboxed inputs.
    Target size is ``(width, height)``; target pixels are never used for rendering.
    """
    if mode not in ("none", "rendered", "coordinates"):
        raise ValueError(f"Unknown bbox conditioning mode: {mode!r}.")
    if mode == "none":
        return reference_images, prompt
    if len(reference_images) != len(reference_keys) or len(set(reference_keys)) != len(reference_keys):
        raise ValueError("Reference images and unique reference keys must have the same length.")
    if not reference_images or len(reference_images) > MAX_BBOX_REFERENCE_IMAGES:
        raise ValueError(f"BBox conditioning supports 1..{MAX_BBOX_REFERENCE_IMAGES} original reference images.")
    if set(reference_bboxes) - set(reference_keys):
        raise ValueError("Bbox groups must refer to the supplied reference images.")
    # A known empty annotation is valid for all-abstract-control samples. Do not
    # invent object locations or add a blank layout with a misleading instruction.
    if not reference_bboxes:
        return reference_images, prompt

    if mode == "coordinates":
        lines = ["The reference coordinates are as follows:"]
        for index, key in enumerate(reference_keys, start=1):
            boxes = reference_bboxes.get(key, [])
            if not boxes:
                continue
            formatted = ["[" + ", ".join(f"{coordinate:.6f}" for coordinate in box) + "]" for box in boxes]
            coordinates = formatted[0] if len(formatted) == 1 else "[" + ", ".join(formatted) + "]"
            lines.append(f"<img-{index}>: {coordinates}")
        return reference_images, prompt + "\n\n" + "\n".join(lines)

    width, height = target_size
    if width <= 0 or height <= 0:
        raise ValueError("Layout dimensions must be positive.")
    layout = Image.new("RGB", target_size, "white")
    draw = ImageDraw.Draw(layout)
    line_width = max(2, round(min(target_size) / 256))
    numbered_references: list[Image.Image] = []
    for index, (key, image) in enumerate(zip(reference_keys, reference_images, strict=True), start=1):
        color = REFERENCE_BBOX_COLORS[index - 1]
        numbered = image.copy()
        _draw_reference_number(numbered, (0, 0), index, color)
        numbered_references.append(numbered)
        for x1, y1, x2, y2 in reference_bboxes.get(key, []):
            pixels = (
                min(width - 1, round(x1 * width)),
                min(height - 1, round(y1 * height)),
                min(width - 1, round(x2 * width)),
                min(height - 1, round(y2 * height)),
            )
            draw.rectangle(pixels, outline=color, width=line_width)
            _draw_reference_number(layout, (pixels[0], pixels[1]), index, color)
    numbered_references.append(layout)
    section = (
        "The layout of all the references are specified by the bounding box layout shown in "
        f"<img-{len(reference_images) + 1}>"
    )
    return numbered_references, prompt + "\n\n" + section


class InjectMultiReferenceBBoxCondition(Augmentor):
    """Inject bbox conditioning after images, group keys, and prompt are shuffled."""

    mode: BBoxConditionMode

    def __init__(self, mode: BBoxConditionMode, input_keys: list[str] | None = None) -> None:
        super().__init__(input_keys or [])
        if mode not in ("rendered", "coordinates"):
            raise ValueError("The bbox augmentor requires rendered or coordinates mode.")
        self.mode = mode

    def __call__(self, data_dict: dict) -> dict | None:
        try:
            target = data_dict["target_image"]
            sources, prompt = build_bbox_condition(
                reference_images=data_dict["source_image"],
                prompt=data_dict["editing_instruction"],
                target_size=target.size,
                reference_keys=data_dict["source_reference_keys"],
                reference_bboxes=data_dict["reference_bboxes"],
                mode=self.mode,
            )
        except (KeyError, ValueError) as error:
            log.warning(f"Skipping bbox sample {data_dict.get('__key__', 'unknown')!r}: {error}", rank0_only=False)
            return None
        data_dict["source_image"] = sources
        data_dict["editing_instruction"] = prompt
        return data_dict
