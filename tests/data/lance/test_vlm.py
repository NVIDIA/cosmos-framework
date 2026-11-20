# SPDX-License-Identifier: OpenMDW-1.1
"""Equivalence test for the VLM (LLaVA-OneVision) loader vs the base HF stream."""

from __future__ import annotations

import io
import os
import tempfile

import pytest
from datasets import load_dataset

from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset, convert_llava_to_lance

pytestmark = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")), reason="set HF_TOKEN"
)


@pytest.fixture(scope="module", autouse=True)
def _hf_online():
    # the action base flips HF Hub offline process-wide; this module streams from the hub
    import huggingface_hub.constants as hfc

    mp = pytest.MonkeyPatch()
    mp.setattr(hfc, "HF_HUB_OFFLINE", False)
    mp.delenv("HF_HUB_OFFLINE", raising=False)
    yield
    mp.undo()


def _norm_image_bytes(rec):
    img = rec.get("image")
    if isinstance(img, dict):
        return img.get("bytes") or b""
    if img is not None:
        buf = io.BytesIO()
        img.save(buf, format=img.format or "PNG")
        return buf.getvalue()
    return b""


def test_vlm():
    subset = os.environ.get("LLAVA_SUBSET", "figureqa(cauldron,llava_format)")
    stream = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split="train", streaming=True)
    stream = stream.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)
    base = []
    for rec in stream:
        base.append(rec)
        if len(base) >= 8:
            break
    with tempfile.TemporaryDirectory() as tmp:
        convert_llava_to_lance(iter(base), tmp, table_name="llava")
        lance = LanceVLMDataset(tmp, table_name="llava")
        assert len(lance) == len(base)
        # unsorted + duplicate indices: batched take must map back to the requested order
        idxs = [5, 1, 5, 0, 7, 3]
        batch = lance.__getitems__(idxs)
        assert len(batch) == len(idxs)
        for i, l in zip(idxs, batch):
            assert l["conversations"] == (base[i].get("conversations") or [])
            assert l["image"]["bytes"] == _norm_image_bytes(base[i])
