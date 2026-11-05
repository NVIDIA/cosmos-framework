# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest
import torch

from cosmos_framework.data.generator.augmentors.video_editing_transform import PairedVideoEditingToTrainingFormat
from cosmos_framework.data.generator.sequence_packing.packers import resolve_item_condition_frames
from cosmos_framework.utils.generator.torchcodec_video import VideoMetadata

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_clean_prefix_changes_only_final_target_mask(monkeypatch: pytest.MonkeyPatch, probability: float) -> None:
    transform = PairedVideoEditingToTrainingFormat(target_first_frame_probability=probability)
    source = torch.zeros(3, 9, 16, 32)  # [C,T,H,W]
    target = torch.arange(source.numel()).reshape_as(source).float()  # [C,T,H,W]
    original = target.clone()
    metadata = VideoMetadata(num_frames=9, average_fps=16.0, height=16, width=32)
    monkeypatch.setattr(transform, "_decode_pair", lambda *_: (source, target, metadata, 32, 16))
    conversation = json.dumps(
        [
            [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "source"}, {"type": "text", "text": "Add a cup."}],
                },
                {"role": "assistant", "content": [{"type": "video", "video": "target"}]},
            ]
        ]
    )
    result = transform({"texts": conversation, "media": {"source": b"source", "target": b"target"}})
    assert result is not None
    assert result["video"][0] is source
    assert result["video"][1] is target
    torch.testing.assert_close(target, original)
    indexes = result["sequence_plan"].condition_frame_indexes_vision
    assert indexes == ([0] if probability else [])
    assert resolve_item_condition_frames(indexes, item_idx=0, num_items=2, latent_t=3) == [0, 1, 2]
    assert resolve_item_condition_frames(indexes, item_idx=1, num_items=2, latent_t=3) == indexes


@pytest.mark.parametrize("probability", [-1, 2, float("nan")])
def test_invalid_probability_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        PairedVideoEditingToTrainingFormat(target_first_frame_probability=probability)
