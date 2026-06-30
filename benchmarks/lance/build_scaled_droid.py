# SPDX-License-Identifier: OpenMDW-1.1
"""Build an N×-scaled DROID dataset (parquet index + meta/episodes + composed Lance table)
for the memory-SCALING benchmark — see bench_memory.py.

The per-worker memory that scales (and OOMs the base) is the index the loader materializes
at __init__ from the DROID ``data/`` parquet (``self._rows`` + compact arrays). To exercise
it at real-DROID scale without downloading the full multi-TB dataset, this replicates the
327-episode subset N×:

  * data/ parquet: rows replicated with shifted ``index`` / ``episode_index`` (kept sorted),
  * meta/episodes: replicated with shifted ``episode_index`` but the SAME video pointers, so
    each duplicated episode decodes the same frames from the original mega-mp4 (base path),
  * the composed Lance table: rows appended with shifted ``episode_index`` pointing at the
    same clip bytes (Lance path).

Then ``ln -s <orig>/videos <out_root>/videos`` so the base can decode. Usage:

    python build_scaled_droid.py --src-root <droid>/success --src-lance <lance>/droid_composed327_plain \
        --out-root /tmp/x16 --out-lance /tmp/lance_x16 --table droid_composed --n 16
    ln -sfn <droid>/success/videos /tmp/x16/videos
    python bench_memory.py --side base  --root /tmp/x16 --uri /tmp/lance_x16 --random
    python bench_memory.py --side lance --root /tmp/x16 --uri /tmp/lance_x16 --random
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import lance
import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _replicate_data(table, n, n_ep, n_rows):
    cols, names = [], table.column_names
    for name in names:
        parts = []
        for k in range(n):
            if name == "index":
                parts.append(table.column(name).to_numpy() + k * n_rows)
            elif name == "episode_index":
                parts.append(table.column(name).to_numpy() + k * n_ep)
            else:
                parts.append(table.column(name).combine_chunks())
        cols.append(pa.array(np.concatenate(parts)) if name in ("index", "episode_index") else pa.concat_arrays(parts))
    return pa.table(cols, names=names)


def _scale_root(src, out, n):
    data = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{src}/data/chunk-*/file-*.parquet"))])
    n_rows = data.num_rows
    n_ep = int(data.column("episode_index").to_numpy().max()) + 1
    os.makedirs(f"{out}/data/chunk-000", exist_ok=True)
    pq.write_table(_replicate_data(data, n, n_ep, n_rows), f"{out}/data/chunk-000/file-000.parquet")

    ep = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{src}/meta/episodes/chunk-*/file-*.parquet"))])
    ep_cols = []
    for name in ep.column_names:
        parts = [
            (ep.column(name).to_numpy() + k * n_ep) if name == "episode_index" else ep.column(name).combine_chunks()
            for k in range(n)
        ]
        ep_cols.append(pa.array(np.concatenate(parts)) if name == "episode_index" else pa.concat_arrays(parts))
    os.makedirs(f"{out}/meta/episodes/chunk-000", exist_ok=True)
    pq.write_table(pa.table(ep_cols, names=ep.column_names), f"{out}/meta/episodes/chunk-000/file-000.parquet")
    shutil.copy(f"{src}/meta/info.json", f"{out}/meta/info.json")
    shutil.copy(f"{src}/meta/tasks.parquet", f"{out}/meta/tasks.parquet")
    print(f"root {out}: {n_rows * n} frames, {n_ep * n} episodes ({n}x)")


def _scale_lance(src, out, table, n):
    t = lance.dataset(f"{src}/{table}.lance").to_table()
    n_ep = int(t.column("episode_index").to_numpy().max()) + 1

    def batches():
        for k in range(n):
            cols = [
                pa.array(t.column(nm).to_numpy() + k * n_ep) if nm == "episode_index" else t.column(nm).combine_chunks()
                for nm in t.column_names
            ]
            yield pa.RecordBatch.from_arrays(cols, names=t.column_names)

    db = lancedb.connect(out)
    if table in db.table_names():
        db.drop_table(table)
    db.create_table(table, data=pa.RecordBatchReader.from_batches(t.schema, batches()), schema=t.schema)
    print(f"lance {out}/{table}.lance: {db.open_table(table).count_rows()} clips ({n}x {t.num_rows})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--src-lance", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-lance", required=True)
    ap.add_argument("--table", default="droid_composed")
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()
    _scale_root(args.src_root, args.out_root, args.n)
    _scale_lance(args.src_lance, args.out_lance, args.table, args.n)
    print(f"now: ln -sfn {args.src_root}/videos {args.out_root}/videos")


if __name__ == "__main__":
    main()
