# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from unittest.mock import Mock

import pytest

from cosmos_framework.data.generator.augmentors import text_transforms_for_video
from cosmos_framework.data.generator.augmentors.text_transforms_for_video import (
    TextTransformForVideoTransferChunkedFrames,
    TextTransformForVideoTransferFullFrames,
)
from cosmos_framework.utils.lazy_config import LazyCall, instantiate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize(
    "transform_class", [TextTransformForVideoTransferFullFrames, TextTransformForVideoTransferChunkedFrames]
)
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_lazy_caption_config_preserves_source_options(
    monkeypatch: pytest.MonkeyPatch,
    transform_class: type[TextTransformForVideoTransferFullFrames],
    as_list: bool,
    probability: float,
) -> None:
    option = {"ratio": 0.25, "dense_caption_probability": probability, "min_stride": 2}
    caption_config = [{"key": "captions", **option}] if as_list else {"captions": option}
    transform = instantiate(LazyCall(transform_class)(input_keys=["metas"], args={"caption_config": caption_config}))
    dense_caption = "A robot opens the drawer, picks up the cup, and closes the drawer."
    structured = {"temporal_caption": dense_caption}
    sample = {
        "metas": {"nb_frames": 600, "framerate": 30.0},
        "captions": {
            "caption_structured": json.dumps(
                {
                    "chunk_200_400": {
                        "start_frame": 200,
                        "end_frame": 400,
                        "caption": json.dumps(structured),
                    }
                }
            )
        },
    }
    sample_source = Mock(wraps=text_transforms_for_video.random.choices)
    monkeypatch.setattr(text_transforms_for_video.random, "choices", sample_source)

    result = transform(sample)

    assert result is not None
    assert result["ai_caption"] == (dense_caption if probability else json.dumps(structured))
    assert result["_full_frames_min_stride"] == 2
    sample_source.assert_called_once_with(["captions"], weights=[0.25], k=1)
