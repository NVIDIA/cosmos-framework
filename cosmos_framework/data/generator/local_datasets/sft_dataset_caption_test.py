# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the SFT loader's caption selection / JSON-vs-dense normalization."""

import json

from cosmos_framework.data.generator.local_datasets.sft_dataset import _select_caption
from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY


def test_caption_json_dict_serialized_verbatim_no_trailing_period():
    cj = {"background_setting": "kitchen", "fps": 5}
    key, text, used_json = _select_caption({CAPTION_JSON_KEY: cj})
    assert key == CAPTION_JSON_KEY and used_json is True
    assert not text.endswith(".")  # MUST NOT append a stray '.' after '}'
    assert text.endswith("}")
    assert json.loads(text) == cj


def test_caption_json_priority_over_dense():
    cj = {"background_setting": "x"}
    key, text, used_json = _select_caption({CAPTION_JSON_KEY: cj, "caption": "dense backup"})
    assert key == CAPTION_JSON_KEY and used_json is True


def test_caption_json_as_preserialized_string():
    key, text, used_json = _select_caption({CAPTION_JSON_KEY: '{"a": 1}  '})
    assert key == CAPTION_JSON_KEY and used_json is True
    assert text == '{"a": 1}'  # stripped, no period


def test_dense_caption_gets_terminal_period():
    key, text, used_json = _select_caption({"caption": "a robot arm moves"})
    assert key == "caption" and used_json is False
    assert text == "a robot arm moves."


def test_dense_caption_period_not_doubled():
    _, text, _ = _select_caption({"caption": "ends with period."})
    assert text == "ends with period."


def test_rewrite_dense_key_priority_over_generic_caption():
    key, _, used_json = _select_caption({"qwen3_32b_rewrite-dense": "x", "caption": "y"})
    assert key == "qwen3_32b_rewrite-dense" and used_json is False


def test_weighted_caption_types_fallback():
    key, text, used_json = _select_caption({"qwen3_235b_dense": "some dense caption"})
    assert key == "qwen3_235b_dense" and used_json is False
    assert text.endswith(".")


def test_no_known_caption_key_returns_none():
    assert _select_caption({"start_frame": 0, "end_frame": 84}) is None
