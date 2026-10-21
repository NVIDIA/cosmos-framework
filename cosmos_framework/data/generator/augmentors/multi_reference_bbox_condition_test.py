# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Coordinate text migration and compatibility with existing bbox checkpoints."""

import copy

import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.multi_reference_bbox_condition import (
    BBoxConditionMode,
    BoundingBox,
    InjectMultiReferenceBBoxCondition,
    build_bbox_condition,
    parse_reference_bboxes,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_training_coordinates_use_thousandths_after_reference_reordering() -> None:
    references = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    annotations = [{"reference_group": "input_01", "boxes": [[0.08, 0.18, 0.26, 0.68]]}]
    original = copy.deepcopy(annotations)
    keys = ["input_03", "input_01", "input_02"]
    data = {
        "source_image": references,
        "source_reference_keys": keys,
        "editing_instruction": "Use <img-1>, <img-2> and <img-3>.",
        "target_image": Image.new("RGB", (64, 48)),
        "reference_bboxes": parse_reference_bboxes(annotations, keys),
    }
    result = InjectMultiReferenceBBoxCondition("coordinates")(data)
    assert result is not None
    assert result["source_image"] is references
    assert result["editing_instruction"].endswith("<img-2>: [80, 180, 260, 680]")
    assert "<img-1>:" not in result["editing_instruction"]
    assert "<img-3>:" not in result["editing_instruction"]
    assert annotations == original


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((0.0, 0.0, 1.0, 1.0), "[0, 0, 1000, 1000]"),
        ((0.10049, 0.2005, 0.70049, 0.9005), "[100, 201, 700, 901]"),
        ((0.5001, 0.5001, 0.5002, 0.5002), "[500, 500, 501, 501]"),
        ((0.9998, 0.9998, 0.9999, 0.9999), "[999, 999, 1000, 1000]"),
    ],
)
def test_integer_coordinates_round_and_preserve_nonempty_boxes(box: BoundingBox, expected: str) -> None:
    _, prompt = build_bbox_condition(
        [Image.new("RGB", (32, 32))],
        "Create an image.",
        (64, 64),
        ["input_01"],
        {"input_01": [box]},
        "coordinates",
    )
    assert prompt.endswith(f"<img-1>: {expected}")


def test_multiple_boxes_keep_nested_lists_and_unboxed_controls() -> None:
    references = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    images, prompt = build_bbox_condition(
        references,
        "Use <img-1> with the style of <img-2>.",
        (64, 64),
        ["input_01", "input_02"],
        {"input_01": [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)]},
        "coordinates",
    )
    assert images is references
    assert prompt.endswith("<img-1>: [[100, 200, 300, 400], [500, 600, 700, 800]]")
    assert "<img-2>:" not in prompt


def test_legacy_checkpoint_prompt_retains_exact_six_decimal_coordinates() -> None:
    _, prompt = build_bbox_condition(
        [Image.new("RGB", (32, 32))],
        "Base prompt.",
        (64, 64),
        ["input_01"],
        {"input_01": [(0.08, 0.18, 0.26, 0.68)]},
        "coordinates",
        coordinate_format="normalized_01",
    )
    assert prompt == (
        "Base prompt.\n\nThe reference coordinates are as follows:\n<img-1>: [0.080000, 0.180000, 0.260000, 0.680000]"
    )


def test_coordinate_format_does_not_change_rendered_inputs() -> None:
    references = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    args = (
        references,
        "Use both inputs.",
        (64, 64),
        ["input_01", "input_02"],
        {"input_01": [(0.1, 0.2, 0.6, 0.8)]},
        "rendered",
    )
    integer_images, integer_prompt = build_bbox_condition(*args, coordinate_format="integer_1000")
    legacy_images, legacy_prompt = build_bbox_condition(*args, coordinate_format="normalized_01")
    assert integer_prompt == legacy_prompt
    assert [image.tobytes() for image in integer_images] == [image.tobytes() for image in legacy_images]
    assert references[0].getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("mode", ["coordinates", "rendered"])
def test_known_empty_annotations_keep_original_prompt_and_images(mode: BBoxConditionMode) -> None:
    references = [Image.new("RGB", (32, 32))]
    images, prompt = build_bbox_condition(references, "Base prompt.", (64, 64), ["input_01"], {}, mode)
    assert images is references
    assert prompt == "Base prompt."
