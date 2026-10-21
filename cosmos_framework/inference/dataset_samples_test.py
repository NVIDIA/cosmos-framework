# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.dataset_samples import MapDatasetSamples, _normalize_caption


class _SampleDataset(Dataset):
    def __init__(self, sample: dict[str, Any]) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx != 0:
            raise IndexError(idx)
        return dict(self.sample)


@pytest.mark.L0
def test_normalize_caption_string_unchanged() -> None:
    sample = {"ai_caption": "Pick up the cup."}
    assert _normalize_caption(sample) == "Pick up the cup."
    assert sample["ai_caption"] == "Pick up the cup."


@pytest.mark.L0
def test_normalize_caption_dict_serialized_and_written_back() -> None:
    caption = {
        "cinematography": {"framing": "third-person view"},
        "actions": [{"time": "0:00-0:02", "description": "Open the drawer."}],
        "duration": "2s",
        "fps": 8.0,
        "resolution": {"H": 192, "W": 320},
        "aspect_ratio": "16,9",
    }
    sample = {"ai_caption": caption}
    result = _normalize_caption(sample)
    assert json.loads(result) == caption
    assert sample["ai_caption"] == result


@pytest.mark.L0
def test_normalize_caption_missing_returns_empty() -> None:
    assert _normalize_caption({}) == ""


@pytest.mark.L0
def test_normalize_caption_raises_on_non_str_non_dict() -> None:
    with pytest.raises(TypeError, match="ai_caption must be str or dict"):
        _normalize_caption({"ai_caption": None})
    with pytest.raises(TypeError, match="ai_caption must be str or dict"):
        _normalize_caption({"ai_caption": 42})


@pytest.mark.L0
@pytest.mark.parametrize("mode", ["forward_dynamics", "inverse_dynamics", "wam"])
def test_dataset_samples_preserve_action_model_mode(mode: str) -> None:
    dataset = _SampleDataset(
        {
            "ai_caption": "hello",
            "video": torch.zeros(3, 1, 8, 8, dtype=torch.uint8),  # [C,T,H,W]
            "raw_action_dim": 10,
        }
    )
    samples = MapDatasetSamples(
        dataset=dataset,
        modes=[mode],
        sample_ids=[0],
        transform=None,
        resolution="256",
        dataset_name="test",
    )

    sample_args, sample_data = next(iter(samples))

    assert sample_args.name == f"test/{mode}/0"
    assert sample_args.model_mode == ModelMode(mode)
    assert sample_args.sample_meta.model_mode == ModelMode(mode)
    assert sample_args.raw_action_dim == 10
    assert sample_data["video"].shape == (1, 3, 1, 8, 8)
