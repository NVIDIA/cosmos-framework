# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Temporal-caption trimming must not retain events outside the short training window."""

import pytest

from cosmos_framework.data.generator.augmentors.first_caption_paragraph import FirstCaptionParagraph

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("[0-2s] The car approaches.\n\n[2-5s] It turns.", "[0-2s] The car approaches."),
        ("[0-1.8s] The car approaches.\n[1.8-5s] It turns.", "[0-1.8s] The car approaches."),
        ("[0s – 2.5s] The car approaches, [2.5s – 5s] It turns.", "[0s – 2.5s] The car approaches"),
        ("Overview.\n\n[0-3s] The car\napproaches.\n\n[3-5s] It turns.", "[0-3s] The car\napproaches."),
        ("  A car approaches.\n\nIt turns later.  ", "A car approaches."),
        ("  [0-2s] A car approaches.  ", "[0-2s] A car approaches."),
        ("", ""),
    ],
)
def test_keeps_only_the_opening_caption_paragraph(caption: str, expected: str) -> None:
    data = {"ai_caption": caption, "__key__": "clip"}
    result = FirstCaptionParagraph()(data)
    assert result is not None
    assert result["ai_caption"] == expected
    assert result["__key__"] == "clip"


@pytest.mark.parametrize("caption", [{"scene": "not a Tier-1 paragraph"}, None, ["caption"]])
def test_drops_nontext_captions(caption: object) -> None:
    assert FirstCaptionParagraph()({"ai_caption": caption, "__key__": "clip"}) is None
