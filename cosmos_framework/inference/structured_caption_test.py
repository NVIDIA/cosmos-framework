# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the structured-JSON caption schema, parsing, and assembly."""

import json

import pytest

from cosmos_framework.inference.structured_caption import (
    CAPTION_JSON_KEY,
    aspect_ratio_str,
    assemble_caption_json,
    caption_json_to_prompt,
    extract_xml_tag,
    media_fields_from_metadata,
    parse_structured_caption,
)


def test_caption_json_key_is_stable():
    assert CAPTION_JSON_KEY == "caption_json"


@pytest.mark.parametrize(
    "w,h,expected",
    [(256, 256, "1,1"), (1920, 1080, "16,9"), (1080, 1920, "9,16"), (640, 480, "4,3"), (0, 10, "")],
)
def test_aspect_ratio_str(w, h, expected):
    assert aspect_ratio_str(w, h) == expected


def test_extract_xml_tag_multiline():
    text = "<final_prompt>\nA cat\nsits.\n</final_prompt>"
    assert extract_xml_tag(text, "final_prompt") == "A cat\nsits."
    assert extract_xml_tag(text, "missing") is None


def test_parse_scene_draft_in_tags_with_fences():
    resp = (
        "preamble\n<scene_draft>\n```json\n"
        '{"subjects": [{"description": "arm"}], "background_setting": "kitchen"}\n'
        "```\n</scene_draft>\n<final_prompt>An arm.</final_prompt>"
    )
    sd = parse_structured_caption(resp)
    assert sd["background_setting"] == "kitchen"
    assert sd["subjects"][0]["description"] == "arm"


def test_parse_raw_object_without_tags():
    assert parse_structured_caption('{"background_setting": "x"}')["background_setting"] == "x"


def test_parse_object_embedded_in_prose_with_brace_in_string():
    # The brace-matcher must ignore braces inside quoted strings.
    sd = parse_structured_caption('Result: {"background_setting": "a } brace", "fps": 5} end')
    assert sd["background_setting"] == "a } brace"
    assert sd["fps"] == 5


def test_parse_returns_none_for_garbage():
    assert parse_structured_caption("there is no json here") is None
    assert parse_structured_caption("") is None


def test_media_fields_from_metadata_uses_actual_values():
    media = media_fields_from_metadata({"width": 256, "height": 256, "duration": 17.0, "fps": 5})
    assert media == {"resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1", "duration": "17s", "fps": 5}


def test_assemble_sets_temporal_caption_and_media():
    scene_draft = {"subjects": [{"description": "arm"}], "background_setting": "kitchen"}
    media = media_fields_from_metadata({"width": 256, "height": 256, "duration": 17.0, "fps": 5})
    cj = assemble_caption_json(scene_draft, "  An arm moves.  ", media)
    assert cj["temporal_caption"] == "An arm moves."  # stripped, overrides any draft value
    assert cj["resolution"] == {"H": 256, "W": 256}
    assert cj["duration"] == "17s" and cj["fps"] == 5
    assert cj["background_setting"] == "kitchen"


def test_assemble_overrides_draft_temporal_caption():
    scene_draft = {"temporal_caption": "draft timeline", "background_setting": "x"}
    cj = assemble_caption_json(scene_draft, "final dense", {})
    assert cj["temporal_caption"] == "final dense"


def test_assemble_drops_none_and_preserves_extras():
    # extra (non-schema) keys must survive; None-valued fields are dropped.
    scene_draft = {"background_setting": "x", "short_description": "extra field", "lighting": None}
    cj = assemble_caption_json(scene_draft, "d", {})
    assert cj["short_description"] == "extra field"
    assert "lighting" not in cj


def test_caption_json_to_prompt_is_compact_and_roundtrips():
    cj = {"background_setting": "café", "fps": 5}
    prompt = caption_json_to_prompt(cj)
    assert ", " not in prompt or ": " in prompt  # compact separators
    assert "café" in prompt  # ensure_ascii=False keeps unicode
    assert json.loads(prompt) == cj
