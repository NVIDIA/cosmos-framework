# SPDX-License-Identifier: OpenMDW-1.1
"""The LanceDB vision-SFT loader must match the local base loader.

Video is near-identical (H.264 re-encode tolerance) and caption/token-ids are
exact. Run after building the JSONL + Lance table:

    BRIDGE_JSONL=.../sft_dataset_bridge/train/video_dataset_file.jsonl \
    VISION_SFT_LANCE_URI=.../lance/vision_sft \
    pytest tests/data/lance/test_vision_sft_equivalence.py
"""
from __future__ import annotations

import os

import pytest
import torch

JSONL = os.environ.get("BRIDGE_JSONL")
URI = os.environ.get("VISION_SFT_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (JSONL and URI and os.path.isfile(JSONL)),
    reason="set BRIDGE_JSONL and VISION_SFT_LANCE_URI to the prepared fixtures",
)

_KW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


@pytest.fixture(scope="module")
def loaders():
    from cosmos_framework.data.lance import LanceVisionSFTDataset
    from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import LocalSFTDataset

    base = LocalSFTDataset(JSONL, **_KW)
    lance = LanceVisionSFTDataset(URI, table="vision_sft", decode_device="cpu", **_KW)
    return base, lance


def test_same_length(loaders):
    base, lance = loaders
    assert len(base) == len(lance)


@pytest.mark.parametrize("idx", [0, 1, 17, 50, 123, 199])
def test_sample_equivalent(loaders, idx):
    base, lance = loaders
    b, l = base[idx], lance[idx]
    # token ids + caption must be EXACT
    assert b["ai_caption"] == l["ai_caption"], "caption differs"
    assert b["sampled_caption_style"] == l["sampled_caption_style"]
    assert torch.equal(b["text_token_ids"], l["text_token_ids"]), "token ids differ"
    # video shape exact; pixels near-identical (H.264 re-encode of the resize)
    assert b["video"].shape == l["video"].shape, f"shape {b['video'].shape} != {l['video'].shape}"
    assert b["num_frames"] == l["num_frames"]
    mad = (b["video"].float() - l["video"].float()).abs().mean().item() / 255.0
    assert mad < 0.05, f"mean|Δ|/255 = {mad:.4f} too large"
