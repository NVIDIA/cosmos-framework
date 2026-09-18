# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Preserve training metadata precedence while supporting inference captions."""

from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("source_ratio", ["16,9", None])
@pytest.mark.parametrize("inference_ratio", ["4,3", None])
@pytest.mark.parametrize("per_view", [False, True])
def test_training_json_caption_uses_source_ratio(
    source_ratio: str | None, inference_ratio: str | None, per_view: bool
) -> None:
    caption = {"caption": "A car.", "aspect_ratio": "old"}
    data: dict[str, Any] = {
        "ai_caption": [caption.copy(), caption.copy()] if per_view else caption,
        "image_size": [480, 640],
        "__url__": SimpleNamespace(meta=SimpleNamespace(opts={"aspect_ratio": source_ratio})),
        "aspect_ratio": inference_ratio,
    }
    result = ResolutionTextInfo()(data)
    assert result is not None
    captions = result["ai_caption"] if per_view else [result["ai_caption"]]
    for item in captions:
        assert item == {
            "caption": "A car.",
            "aspect_ratio": source_ratio,
            "resolution": {"H": 480, "W": 640},
        }


@pytest.mark.parametrize("inference_ratio,expected_ratio", [("4,3", "4,3"), (None, "old")])
def test_inference_json_caption_without_source_url(inference_ratio: str | None, expected_ratio: str) -> None:
    data = {
        "ai_caption": {"caption": "A car.", "aspect_ratio": "old"},
        "image_size": [480, 640],
        "aspect_ratio": inference_ratio,
    }
    result = ResolutionTextInfo()(data)
    assert result is not None
    assert result["ai_caption"] == {
        "caption": "A car.",
        "aspect_ratio": expected_ratio,
        "resolution": {"H": 480, "W": 640},
    }
