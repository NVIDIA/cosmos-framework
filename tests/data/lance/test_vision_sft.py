# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence test for the Vision-SFT loader vs the genuine SFTDataset."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from cosmos_framework.data.lance import LanceVisionSFTDataset
from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.vfm.local_datasets.sft_dataset import SFTDataset

JSONL = os.environ.get("BRIDGE_JSONL")
URI = os.environ.get("VISION_SFT_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (JSONL and URI and os.path.isfile(JSONL)), reason="set BRIDGE_JSONL and VISION_SFT_LANCE_URI")

_VKW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


@pytest.fixture(scope="module")
def base_and_metas():
    base_dir = os.path.dirname(os.path.abspath(JSONL))
    metas = []
    with open(JSONL) as f:
        for line in f:
            rec = json.loads(line)
            vp = rec["vision_path"]
            vp = vp if ("://" in vp or vp.startswith("/")) else os.path.join(base_dir, vp)
            for wi, w in enumerate(rec["t2w_windows"]):
                metas.append({
                    "uuid": f"{rec['uuid']}_w{wi}", "vision_path": vp,
                    "width": rec["width"], "height": rec["height"],
                    "aspect_ratio": get_aspect_ratio(rec["width"], rec["height"]),
                    "t2w_windows": [w],
                })
    tok_cfg = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"))
    ds = SFTDataset(metadata=metas, num_video_frames=16, resolution="256", s3_credentials={},
                    frame_selection_mode="first", temporal_interval_mode="entire_chunk",
                    tokenizer_config=tok_cfg, cfg_dropout_rate=0.0)
    ds.s3_client = None
    return ds, metas


def test_vision_sft(base_and_metas):
    base, metas = base_and_metas
    lance = LanceVisionSFTDataset(URI, table="vision_sft", decode_device="cpu", **_VKW)
    assert len(metas) == len(lance)
    idxs = [i for i in [0, 1, 17, 50, 123] if i < len(lance)]
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        ref, l = base.process_one_sample(metas[i]), batch[j]
        assert torch.equal(ref["text_token_ids"], l["text_token_ids"])
        assert ref["ai_caption"] == l["ai_caption"]
        mad = (ref["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        assert mad < 0.02
