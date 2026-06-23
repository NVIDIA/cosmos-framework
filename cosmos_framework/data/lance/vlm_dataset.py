# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed VLM (LLaVA-OneVision) dataset.

The base VLM path streams ``lmms-lab/LLaVA-OneVision-Data`` as a HuggingFace
``IterableDataset`` (``streaming=True``): sequential shard reads, a bounded
shuffle buffer (no true global shuffle), and re-decode every epoch. This module
stores the same raw records in a single Lance table and serves them via the
**Permutation API** — true O(1) random access + global shuffle, columnar batched
reads, no streaming-iterator overhead.

It is a drop-in source for the *same* downstream processor (``VLMProcessor``):
``__getitem__`` yields the identical raw dict (``{"id", "image", "conversations"}``)
that ``get_llava_ov_streaming`` yields, so tokenization/image-processing — and thus
the produced training tensors — are unchanged. Only the access layer differs.
"""
from __future__ import annotations

import json
from typing import Any

import lance
import lancedb
import pyarrow as pa
import torch
from lancedb.permutation import Permutation

_COLS = ["sample_id", "image_bytes", "conversations"]


# ── conversion ────────────────────────────────────────────────────────
def _record_batches(hf_dataset, batch_rows: int = 512):
    """Yield RecordBatches of (sample_id, image_bytes, conversations-json).

    Stores the *encoded* image bytes (PNG/JPEG as shipped) — same pixels, no
    re-encode, columnar. Conversations are kept as a JSON string."""
    import io

    schema = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("image_bytes", pa.large_binary()),
            pa.field("conversations", pa.string()),
        ]
    )
    ids, imgs, convs = [], [], []
    for i, rec in enumerate(hf_dataset):
        img = rec.get("image")
        if isinstance(img, dict):
            raw = img.get("bytes") or b""
        elif img is not None:
            buf = io.BytesIO()
            img.save(buf, format=img.format or "PNG")
            raw = buf.getvalue()
        else:
            raw = b""
        ids.append(str(rec.get("id", i)))
        imgs.append(raw)
        convs.append(json.dumps(rec.get("conversations") or []))
        if len(ids) >= batch_rows:
            yield pa.RecordBatch.from_arrays(
                [pa.array(ids, pa.string()), pa.array(imgs, pa.large_binary()), pa.array(convs, pa.string())],
                schema=schema,
            )
            ids, imgs, convs = [], [], []
    if ids:
        yield pa.RecordBatch.from_arrays(
            [pa.array(ids, pa.string()), pa.array(imgs, pa.large_binary()), pa.array(convs, pa.string())],
            schema=schema,
        )


def convert_llava_to_lance(hf_dataset, uri: str, table_name: str = "llava") -> str:
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("image_bytes", pa.large_binary()),
            pa.field("conversations", pa.string()),
        ]
    )
    reader = pa.RecordBatchReader.from_batches(schema, _record_batches(hf_dataset))
    db = lancedb.connect(uri)
    if table_name in [t for t in db.table_names()]:
        db.drop_table(table_name)
    db.create_table(table_name, data=reader, schema=schema)
    return table_name


# ── dataset (map-style, Permutation API) ───────────────────────────────
class LanceVLMDataset(torch.utils.data.Dataset):
    """Map-style LLaVA-OneVision source backed by a Lance table.

    Yields the same raw dict shape as ``get_llava_ov_streaming`` so a downstream
    ``VLMProcessor`` produces identical tensors. Worker-safe: only conn params
    are pickled; each worker reopens its own Permutation handle."""

    def __init__(self, uri: str, table_name: str = "llava", storage_options: dict | None = None):
        self.uri = uri
        self.table_name = table_name
        self.storage_options = storage_options
        self._perm = None
        db = self._connect()
        self.length = db.open_table(table_name).count_rows()

    def _connect(self):
        if self.storage_options:
            return lancedb.connect(self.uri, storage_options=self.storage_options)
        return lancedb.connect(self.uri)

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self) -> None:
        if self._perm is None:
            db = self._connect()
            self._perm = (
                Permutation.identity(db.open_table(self.table_name))
                .select_columns(_COLS)
                .with_format("arrow")
            )

    def _row_to_item(self, batch: pa.RecordBatch, i: int) -> dict[str, Any]:
        return {
            "id": batch.column("sample_id")[i].as_py(),
            "image": {"bytes": batch.column("image_bytes")[i].as_py()},
            "conversations": json.loads(batch.column("conversations")[i].as_py()),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_open()
        return self._row_to_item(self._perm.__getitems__([int(idx)]), 0)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self._ensure_open()
        batch = self._perm.__getitems__([int(i) for i in indices])
        return [self._row_to_item(batch, i) for i in range(batch.num_rows)]


class LanceVLMShuffleScan(torch.utils.data.IterableDataset):
    """Chunked-shuffle scan over a Lance table — the right pattern for shuffled
    training reads from object storage.

    Naive random point-lookups are latency-bound on S3. Instead we shuffle the
    *fragment order* + buffer-shuffle rows within a sequential scan: bandwidth-
    bound reads (fast on S3) with shuffle quality on par with a WebDataset
    shuffle buffer — but lance's columnar scan is materially faster than tar
    streaming, and (unlike webdataset) true random access remains available.
    Fragments are sharded across DataLoader workers and DDP ranks.
    """

    def __init__(
        self,
        uri: str,
        table_name: str = "llava",
        storage_options: dict | None = None,
        buffer_size: int = 1000,
        batch_size: int = 256,
        seed: int = 42,
    ):
        self.uri = uri
        self.table_name = table_name
        self.storage_options = storage_options
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        db = lancedb.connect(uri, storage_options=storage_options) if storage_options else lancedb.connect(uri)
        self.length = db.open_table(table_name).count_rows()

    def __len__(self) -> int:
        return self.length

    def _dataset(self):
        path = f"{self.uri}/{self.table_name}.lance"
        return lance.dataset(path, storage_options=self.storage_options)

    def __iter__(self):
        import random as _random

        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        ds = self._dataset()
        frags = ds.get_fragments()
        rng = _random.Random(self.seed)
        rng.shuffle(frags)
        my_frags = frags[wid::nw]
        buf: list[dict] = []
        for frag in my_frags:
            # batch_readahead prefetches the next batches' IO (matters on S3); falls back
            # gracefully if an older lance build doesn't accept the kwarg.
            try:
                batches = frag.to_batches(columns=_COLS, batch_size=self.batch_size, batch_readahead=8)
            except TypeError:
                batches = frag.to_batches(columns=_COLS, batch_size=self.batch_size)
            for batch in batches:
                ids = batch.column("sample_id").to_pylist()
                imgs = batch.column("image_bytes").to_pylist()
                convs = batch.column("conversations").to_pylist()
                for sid, raw, cv in zip(ids, imgs, convs):
                    buf.append({"id": sid, "image": {"bytes": raw}, "conversations": json.loads(cv)})
                    if len(buf) >= self.buffer_size:
                        yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


__all__ = ["LanceVLMDataset", "LanceVLMShuffleScan", "convert_llava_to_lance"]
