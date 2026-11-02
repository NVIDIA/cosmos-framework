# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for subject-bound layout conditioning of unified multi-reference prompts."""

import json
import random

import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.multi_reference_subject_layout import (
    SUBJECT_COLORS,
    AppendSubjectLayoutImage,
    BuildSubjectLayoutCondition,
    build_subject_layout_condition,
    defined_subjects,
    insert_layout_line,
    parse_subject_bboxes,
    place_labels,
)
from cosmos_framework.data.generator.augmentors.multi_reference_transform import ReorderReferenceImages

PROMPT = (
    "[Entities]\n"
    "<subject 1>: the woman from <image 1>, wearing the red dress from <image 2>.\n"
    "<subject 2>: a small brown dog.\n\n"
    "[Scene] The overall scene is a reading nook from <image 3>, with warm light.\n"
    "[Style] The overall style of the image is provided in <image 4>, a watercolor.\n\n"
    "[Composition]\n"
    "<subject 1> sits on the rug beside <subject 2>."
)
BOXES = json.dumps([{"subject": 1, "boxes": [[0.1, 0.2, 0.5, 0.9]]}, {"subject": 2, "boxes": [[0.6, 0.6, 0.9, 0.95]]}])


def _sample(prompt: str = PROMPT, boxes: str | None = BOXES, num_references: int = 4) -> dict:
    return {
        "__key__": "view-0",
        "annotations": {"subject_bboxes": boxes, "subject_paste_images": {1: [1]}},
        "source_image": [Image.new("RGB", (32, 32), (index * 60, 0, 0)) for index in range(num_references)],
        "target_image": Image.new("RGB", (200, 100)),
        "editing_instruction": prompt,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_layout_line_goes_right_after_scene() -> None:
    text = insert_layout_line(PROMPT, "[Layout] X.")
    assert "[Scene] The overall scene is a reading nook from <image 3>, with warm light.\n[Layout] X.\n[Style]" in text
    without_style = PROMPT.replace(
        "\n[Style] The overall style of the image is provided in <image 4>, a watercolor.", ""
    )
    assert "warm light.\n[Layout] X.\n\n[Composition]" in insert_layout_line(without_style, "[Layout] X.")
    with pytest.raises(ValueError, match="no \\[Scene\\]"):
        insert_layout_line("<img-1> a cat", "[Layout] X.")
    with pytest.raises(ValueError, match="already"):
        insert_layout_line(text, "[Layout] Y.")


@pytest.mark.L0
@pytest.mark.CPU
def test_rendered_mode_adds_layout_image_as_next_reference() -> None:
    prompt, layout = build_subject_layout_condition(PROMPT, BOXES, 4, (200, 100), "rendered")
    assert "\n[Layout] The overall layout of all the mentioned entities is provided in <image 5>." in prompt
    assert layout is not None and layout.size == (200, 100)
    # Subject 1's box outline uses the first palette color.
    assert layout.getpixel((20, 60)) == Image.new("RGB", (1, 1), SUBJECT_COLORS[0]).getpixel((0, 0))
    assert layout.getpixel((70, 60)) == (255, 255, 255)


@pytest.mark.L0
@pytest.mark.CPU
def test_rendered_paste_mode_pastes_subject_reference_under_outlines() -> None:
    reference = Image.new("RGB", (40, 80), (0, 0, 255))
    prompt, layout = build_subject_layout_condition(
        PROMPT, BOXES, 4, (200, 100), "rendered_paste", subject_images={1: [reference]}
    )
    assert (
        "provided in <image 5>. Each numbered box marks where the subject with that number appears and shows "
        "the reference images of that subject." in prompt
    )
    assert layout is not None
    # Subject 1's box is 80x70 px; the 40x80 reference is fitted to 35x70 and centered in it.
    assert layout.getpixel((60, 55)) == (0, 0, 255)
    assert layout.getpixel((30, 55)) == (255, 255, 255)
    assert layout.getpixel((20, 60)) == Image.new("RGB", (1, 1), SUBJECT_COLORS[0]).getpixel((0, 0))
    # Subject 2 has no content image: an empty outlined box.
    assert layout.getpixel((150, 80)) == (255, 255, 255)


@pytest.mark.L0
@pytest.mark.CPU
def test_rendered_paste_tiles_a_subjects_references_in_order() -> None:
    person, dress = Image.new("RGB", (40, 80), (0, 0, 255)), Image.new("RGB", (40, 80), (0, 255, 0))
    boxes = json.dumps([{"subject": 1, "boxes": [[0.1, 0.2, 0.9, 0.9]]}])
    _, layout = build_subject_layout_condition(
        PROMPT, boxes, 4, (200, 100), "rendered_paste", subject_images={1: [person, dress]}
    )
    assert layout is not None
    # A wide 160x70 box becomes two 80x70 cells; the tall references sit side by side, content first.
    assert layout.getpixel((60, 55)) == (0, 0, 255)
    assert layout.getpixel((140, 55)) == (0, 255, 0)
    assert layout.getpixel((100, 55)) == (255, 255, 255)


@pytest.mark.L0
@pytest.mark.CPU
def test_rendered_paste_keeps_references_clear_of_smaller_boxes_pasted_on_top() -> None:
    person, bag = Image.new("RGB", (40, 40), (0, 0, 255)), Image.new("RGB", (40, 40), (0, 255, 0))
    # The bag box covers the left half of the person box, so the person goes into the right half.
    boxes = json.dumps(
        [{"subject": 1, "boxes": [[0.0, 0.0, 1.0, 1.0]]}, {"subject": 2, "boxes": [[0.0, 0.0, 0.5, 1.0]]}]
    )
    _, layout = build_subject_layout_condition(
        PROMPT, boxes, 4, (200, 100), "rendered_paste", subject_images={1: [person], 2: [bag]}
    )
    assert layout is not None
    assert layout.getpixel((150, 50)) == (0, 0, 255)
    assert layout.getpixel((50, 50)) == (0, 255, 0)
    # When less than a quarter of the box is free, the whole box is used.
    covered = json.dumps(
        [{"subject": 1, "boxes": [[0.0, 0.0, 1.0, 1.0]]}, {"subject": 2, "boxes": [[0.0, 0.0, 0.9, 1.0]]}]
    )
    _, layout = build_subject_layout_condition(
        PROMPT, covered, 4, (200, 100), "rendered_paste", subject_images={1: [person], 2: [bag]}
    )
    assert layout is not None
    # The person fills the whole box (x 50..150); the bag (x 40..140) covers all but its right edge.
    assert layout.getpixel((147, 50)) == (0, 0, 255)
    assert layout.getpixel((190, 50)) == (255, 255, 255)


@pytest.mark.L0
@pytest.mark.CPU
def test_coordinates_mode_lists_integer_boxes_per_subject_without_image() -> None:
    prompt, layout = build_subject_layout_condition(PROMPT, BOXES, 4, (200, 100), "coordinates")
    assert layout is None
    assert (
        "[Layout] The positions of the subjects are given as [x1, y1, x2, y2] boxes normalized to integers in "
        "[0, 1000]: <subject 1> [100, 200, 500, 900]; <subject 2> [600, 600, 900, 950]." in prompt
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_coordinates_mode_keeps_legacy_normalized_format() -> None:
    prompt, _ = build_subject_layout_condition(
        PROMPT, BOXES, 4, (200, 100), "coordinates", coordinate_format="normalized_01"
    )
    assert "boxes normalized to values in [0, 1]: <subject 1> [0.100000, 0.200000, 0.500000, 0.900000];" in prompt


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("boxes", [None, "[]", json.dumps([{"subject": 1, "boxes": []}])])
def test_missing_geometry_keeps_plain_prompt(boxes: str | None) -> None:
    assert build_subject_layout_condition(PROMPT, boxes, 4, (200, 100), "rendered") == (PROMPT, None)


@pytest.mark.L0
@pytest.mark.CPU
def test_boxes_shared_by_several_subjects_are_dropped() -> None:
    prompt = PROMPT.replace(
        "<subject 2>: a small brown dog.", "<subject 2>: a small brown dog.\n<subject 3>: a red ball."
    )
    shared, own = [0.1, 0.2, 0.5, 0.9], [0.6, 0.6, 0.9, 0.95]
    boxes = json.dumps(
        [{"subject": 1, "boxes": [shared]}, {"subject": 2, "boxes": [shared, own]}, {"subject": 3, "boxes": [shared]}]
    )
    text, _ = build_subject_layout_condition(prompt, boxes, 4, (200, 100), "coordinates")
    assert "[0, 1000]: <subject 2> [600, 600, 900, 950]." in text
    only_shared = json.dumps([{"subject": 1, "boxes": [shared]}, {"subject": 2, "boxes": [shared]}])
    assert build_subject_layout_condition(PROMPT, only_shared, 4, (200, 100), "rendered") == (PROMPT, None)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "boxes",
    [
        json.dumps([{"subject": 3, "boxes": [[0.1, 0.1, 0.2, 0.2]]}]),
        json.dumps([{"subject": 1, "boxes": [[0.5, 0.1, 0.2, 0.2]]}]),
        json.dumps([{"subject": 1, "boxes": [[0.1, 0.1, 0.2]]}]),
        json.dumps([{"subject": 1, "boxes": []}, {"subject": 1, "boxes": []}]),
    ],
)
def test_invalid_geometry_skips_sample(boxes: str) -> None:
    assert BuildSubjectLayoutCondition("rendered")(_sample(boxes=boxes)) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_defined_subjects_reads_entity_lines_only() -> None:
    assert defined_subjects(PROMPT) == {1, 2}
    assert parse_subject_bboxes(None) == ()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["rendered", "rendered_paste"])
def test_shuffle_renumbers_references_but_not_the_appended_layout(mode: str) -> None:
    random.seed(3)
    data = BuildSubjectLayoutCondition(mode)(_sample())
    assert data is not None
    originals = list(data["source_image"])
    data = ReorderReferenceImages(shuffle_prob=1.0)(data)
    data = AppendSubjectLayoutImage()(data)
    sources = data["source_image"]
    assert len(sources) == 5 and sources[-1].size == (200, 100)
    new_position = {id(image): index + 1 for index, image in enumerate(sources[:4])}
    prompt = data["editing_instruction"]
    assert f"the woman from <image {new_position[id(originals[0])]}>" in prompt
    assert f"red dress from <image {new_position[id(originals[1])]}>" in prompt
    assert f"reading nook from <image {new_position[id(originals[2])]}>" in prompt
    assert "is provided in <image 5>. Each numbered box" in prompt
    assert "subject_layout_image" not in data
    # The pasted reference is subject 1's own image, whatever its position after the shuffle.
    assert sources[-1].getpixel((60, 55)) == ((0, 0, 0) if mode == "rendered_paste" else (255, 255, 255))


@pytest.mark.L0
@pytest.mark.CPU
def test_rendered_paste_skips_samples_with_out_of_range_content_images() -> None:
    sample = _sample()
    sample["annotations"]["subject_paste_images"] = {1: [1, 9]}
    assert BuildSubjectLayoutCondition("rendered_paste")(sample) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_img_markers_still_shuffle() -> None:
    random.seed(0)
    data = {"source_image": ["a", "b", "c"], "editing_instruction": "<img-1> and <img-2> in <img-3>"}
    out = ReorderReferenceImages(shuffle_prob=1.0)(data)
    order = {name: index + 1 for index, name in enumerate(out["source_image"])}
    assert out["editing_instruction"] == f"<img-{order['a']}> and <img-{order['b']}> in <img-{order['c']}>"


@pytest.mark.L0
@pytest.mark.CPU
def test_labels_never_overlap_when_space_allows() -> None:
    boxes = [(10, 10, 60, 60), (12, 12, 62, 62), (14, 14, 64, 64)]
    badges = place_labels(boxes, (200, 200), (12, 12))
    for index, first in enumerate(badges):
        for second in badges[index + 1 :]:
            assert min(first[2], second[2]) <= max(first[0], second[0]) or min(first[3], second[3]) <= max(
                first[1], second[1]
            )
