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
        batch = lance.__getitems__(list(range(len(base))))
        for i, l in enumerate(batch):
            assert l["conversations"] == (base[i].get("conversations") or [])
            assert l["image"]["bytes"] == _norm_image_bytes(base[i])
