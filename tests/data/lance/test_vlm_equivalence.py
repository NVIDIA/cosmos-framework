# SPDX-License-Identifier: OpenMDW-1.1
"""The LanceDB VLM loader must yield the SAME raw records as the base HF stream.

The base VLM path (``get_llava_ov_streaming``) yields ``{id, image, conversations}``
dicts that the VLMProcessor tokenizes. ``LanceVLMDataset`` must reproduce those
records byte-for-byte (image bytes) and value-for-value (id, conversations) so the
downstream tokenizer produces identical tensors.

Self-contained: streams the first N records from the HF Hub, builds a temp Lance
table from exactly those, then asserts the Lance loader reproduces each one.

    HF_TOKEN=... pytest tests/data/lance/test_vlm_equivalence.py
"""
from __future__ import annotations

import os
import tempfile

import pytest

SUBSET = os.environ.get("LLAVA_SUBSET", "figureqa(cauldron,llava_format)")
N = int(os.environ.get("LLAVA_EQUIV_N", "64"))

pytestmark = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="set HF_TOKEN to stream the LLaVA-OneVision base records",
)


def _norm_image_bytes(rec):
    """Mirror convert_llava_to_lance: dict-image -> .bytes, PIL -> re-save."""
    import io

    img = rec.get("image")
    if isinstance(img, dict):
        return img.get("bytes") or b""
    if img is not None:
        buf = io.BytesIO()
        img.save(buf, format=img.format or "PNG")
        return buf.getvalue()
    return b""


@pytest.fixture(scope="module")
def base_and_lance():
    from datasets import load_dataset

    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset, convert_llava_to_lance

    stream = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=SUBSET, split="train", streaming=True)
    stream = stream.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)
    base = []
    for rec in stream:
        base.append(rec)
        if len(base) >= N:
            break

    tmp = tempfile.mkdtemp()
    convert_llava_to_lance(iter(base), tmp, table_name="llava")
    lance_ds = LanceVLMDataset(tmp, table_name="llava")
    return base, lance_ds


def test_same_length(base_and_lance):
    base, lance = base_and_lance
    assert len(lance) == len(base)


def test_records_identical(base_and_lance):
    base, lance = base_and_lance
    # the converter preserves input order, so row i corresponds to base[i].
    for i in range(len(base)):
        b, l = base[i], lance[i]
        assert str(b.get("id", i)) == str(l["id"]), f"id mismatch at {i}"
        assert l["image"]["bytes"] == _norm_image_bytes(b), f"image bytes differ at {i}"
        assert l["conversations"] == (b.get("conversations") or []), f"conversations differ at {i}"


def test_batched_matches_single(base_and_lance):
    _, lance = base_and_lance
    idxs = list(range(min(8, len(lance))))
    batched = lance.__getitems__(idxs)
    for j, i in enumerate(idxs):
        assert batched[j]["id"] == lance[i]["id"]
        assert batched[j]["image"]["bytes"] == lance[i]["image"]["bytes"]
