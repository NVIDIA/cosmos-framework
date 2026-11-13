# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence test for the Vision-SFT loader vs the genuine SFTDataset."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    _flatten_metadata_by_window,
    _load_sft_metadata_from_s3,
)
from cosmos_framework.data.lance import LanceVisionSFTDataset

JSONL = os.environ.get("BRIDGE_JSONL")
URI = os.environ.get("VISION_SFT_LANCE_URI")

pytestmark = pytest.mark.skipif(
    not (JSONL and URI and os.path.isfile(JSONL)), reason="set BRIDGE_JSONL and VISION_SFT_LANCE_URI"
)

_VKW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


@pytest.fixture(scope="module", autouse=True)
def _hf_online():
    # the action base flips HF Hub offline process-wide; this module loads a tokenizer from the hub cache
    import huggingface_hub.constants as hfc

    mp = pytest.MonkeyPatch()
    mp.setattr(hfc, "HF_HUB_OFFLINE", False)
    mp.delenv("HF_HUB_OFFLINE", raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def base_and_metas():
    # the same metadata load + per-window flattening the converter uses (min_frames=61)
    metas = _flatten_metadata_by_window(_load_sft_metadata_from_s3(None, JSONL, min_frames=61))
    base_dir = os.path.dirname(os.path.abspath(JSONL))
    for m in metas:
        vp = m["vision_path"]
        m["vision_path"] = vp if ("://" in vp or vp.startswith("/")) else os.path.join(base_dir, vp)
    tok_cfg = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"))
    ds = SFTDataset(
        metadata=metas,
        num_video_frames=16,
        resolution="256",
        s3_credentials={},
        frame_selection_mode="first",
        temporal_interval_mode="entire_chunk",
        tokenizer_config=tok_cfg,
        cfg_dropout_rate=0.0,
    )
    ds.s3_client = None
    return ds, metas


def test_vision_sft(base_and_metas):
    base, metas = base_and_metas
    lance = LanceVisionSFTDataset(URI, table="vision_sft", decode_device="cpu", **_VKW)
    assert len(metas) == len(lance)
    idxs = [i for i in [50, 1, 123, 0, 17] if i < len(lance)]  # unsorted: batched take must map back to the right clip
    batch = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        ref, l = base.process_one_sample(metas[i]), batch[j]
        assert torch.equal(ref["text_token_ids"], l["text_token_ids"])
        assert ref["ai_caption"] == l["ai_caption"]
        mad = (ref["video"].float() - l["video"].float()).abs().mean().item() / 255.0
        assert mad < 0.02


def test_vision_sft_dense_caption(base_and_metas):
    """Dense (non-structured) captions get the base's duration/resolution suffixes."""
    base, metas = base_and_metas
    lance = LanceVisionSFTDataset(URI, table="vision_sft", decode_device="cpu", **_VKW)
    lance._ensure_open()
    for i in [0, 7]:
        meta = {
            **metas[i],
            "t2w_windows": [{k: v for k, v in metas[i]["t2w_windows"][0].items() if k != "caption_json"}],
        }
        lance._rows[i] = {**lance._rows[i], "caption_json": ""}
        ref, l = base.process_one_sample(meta), lance[i]
        assert "seconds long" in ref["ai_caption"]  # the dense path really appends the suffixes
        assert ref["ai_caption"] == l["ai_caption"]
        assert torch.equal(ref["text_token_ids"], l["text_token_ids"])
