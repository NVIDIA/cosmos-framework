# SPDX-License-Identifier: OpenMDW-1.1
"""LanceDB-backed VLM (LLaVA-OneVision) dataset.

Row-level random access and global shuffle for VLM datasets.
Drop-in replacement for HF streaming or WebDataset sources.
"""

from __future__ import annotations

import io
import json
import random
from typing import Any

import lancedb
import pyarrow as pa
import torch
from lancedb.permutation import Permutation

_COLS = ["sample_id", "image_bytes", "conversations"]
_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("image_bytes", pa.large_binary()),
        pa.field("conversations", pa.string()),
    ]
)


def _record_batches(hf_dataset, batch_rows: int = 512):
    schema = _SCHEMA
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
            [pa.array(ids, pa.string()), pa.array(imgs, pa.large_binary()), pa.array(convs, pa.string())], schema=schema
        )


def convert_llava_to_lance(hf_dataset, uri: str, table_name: str = "llava") -> str:
    reader = pa.RecordBatchReader.from_batches(_SCHEMA, _record_batches(hf_dataset))
    db = lancedb.connect(uri)
    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=reader, schema=_SCHEMA)
    return table_name


class LanceVLMDataset(torch.utils.data.Dataset):
    """Map-style LLaVA-OneVision source backed by LanceDB."""

    def __init__(self, uri: str, table_name: str = "llava", storage_options: dict | None = None):
        self.uri = uri
        self.table_name = table_name
        self.storage_options = storage_options
        self._perm = None
        self.length = self._connect().open_table(table_name).count_rows()

    def _connect(self):
        return lancedb.connect(self.uri, storage_options=self.storage_options)

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self) -> None:
        if self._perm is None:
            db = self._connect()
            table = db.open_table(self.table_name)
            self._perm = Permutation.identity(table).select_columns(_COLS).with_format("arrow")

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
        # take returns rows sorted by offset (and deduplicated), so key results by
        # row and map back to the requested order — never zip positionally.
        rows = sorted({int(i) for i in indices})
        batch = self._perm.__getitems__(rows)
        by_row = {r: self._row_to_item(batch, i) for i, r in enumerate(rows)}
        return [dict(by_row[int(i)]) for i in indices]


class LanceVLMShuffleScan(torch.utils.data.IterableDataset):
    """Chunked-shuffle scan over a Lance table for efficient S3 training.

    Permutation API only (no pylance): shuffle the order of contiguous row-chunks,
    read each chunk as a columnar range (sequential -> S3-friendly), and emit
    through a local shuffle buffer.
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
        self._perm = None
        self._epoch = 0
        # Set by RankPartitionedDataLoader; None falls back to torch.distributed.
        self.shard_world_size = None
        self.shard_rank = None
        self.length = self._open_table().count_rows()

    def _open_table(self):
        return lancedb.connect(self.uri, storage_options=self.storage_options).open_table(self.table_name)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_perm(self):
        if self._perm is None:
            self._perm = Permutation.identity(self._open_table()).select_columns(_COLS).with_format("arrow")
        return self._perm

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        ws, rk = self.shard_world_size, self.shard_rank
        if ws is None or rk is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws, rk = torch.distributed.get_world_size(), torch.distributed.get_rank()
            else:
                ws, rk = 1, 0
        shard = int(rk) * nw + wid
        total = max(1, int(ws) * nw)
        perm = self._ensure_perm()
        chunks = [(s, min(s + self.batch_size, self.length)) for s in range(0, self.length, self.batch_size)]
        epoch, self._epoch = self._epoch, self._epoch + 1  # reshuffle each pass
        rng = random.Random(self.seed + epoch)
        rng.shuffle(chunks)
        buf = []
        for start, end in chunks[shard::total]:
            batch = perm.__getitems__(list(range(start, end)))
            ids = batch.column("sample_id").to_pylist()
            imgs = batch.column("image_bytes").to_pylist()
            convs = batch.column("conversations").to_pylist()
            for sid, raw, cv in zip(ids, imgs, convs):
                buf.append({"id": sid, "image": {"bytes": raw}, "conversations": json.loads(cv)})
                if len(buf) >= self.buffer_size:
                    yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


def get_lance_vlm_dataset(
    *,
    uri: str,
    table_name: str = "llava",
    storage_options: dict | None = None,
    subset: str | None = None,
    split: str | None = None,
    n: int | None = None,
):
    """Lance drop-in for ``get_llava_ov_map``: the same map-style image+conversation
    records, read from LanceDB. ``n`` caps the dataset to the first ``n`` rows like
    the base's ``.select(range(n))``; ``subset``/``split`` are accepted for
    signature-compatibility but must match what the table was built from (the
    conversion bakes them in)."""
    ds = LanceVLMDataset(uri, table_name=table_name, storage_options=storage_options)
    if n is not None:
        ds.length = min(ds.length, int(n))
    return ds


__all__ = [
    "LanceVLMDataset",
    "LanceVLMShuffleScan",
    "convert_llava_to_lance",
    "get_lance_vlm_dataset",
]
