# SPDX-License-Identifier: OpenMDW-1.1
"""Is take_blobs()+readall-loop the S3 bottleneck? Compare against a columnar take.

The composed loaders read mp4 bytes via `take_blobs(col, indices)` then loop
`blob.readall()` per row. On S3 that serializes the per-row GETs (latency-bound).
For small/medium blobs (~1-2 MB mp4s) a plain columnar read of the binary column
(`to_table(columns=[col])` over a fragment-take) lets Lance parallelize the read
across LANCE_IO_THREADS. This measures both for identical index batches.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import lance


def via_take_blobs(ds, batches, col):
    nbytes = 0
    for b in batches:
        for blob in ds.take_blobs(col, indices=b):
            nbytes += len(blob.readall())
            blob.close()
    return nbytes


def via_take(ds, batches, col):
    nbytes = 0
    for b in batches:
        tbl = ds.take(b, columns=[col])
        arr = tbl.column(col)
        for v in arr:
            nbytes += len(v.as_py())
    return nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument("--col", default="video_bytes")
    ap.add_argument("--n", type=int, default=654)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    so = {"region": args.region} if args.region else None
    ds = lance.dataset(args.uri, storage_options=so)
    total = ds.count_rows()
    rng = random.Random(0)
    pool = [i % total for i in range(args.n)]
    rng.shuffle(pool)
    batches = [pool[i : i + args.batch] for i in range(0, args.n, args.batch)]

    regime = "S3" if args.region else "LOCAL"
    print(f"[{regime}] n={args.n} batch={args.batch} IO_THREADS={os.environ.get('LANCE_IO_THREADS','default')}")
    print(f"{'method':<16}{'clips/s':>12}{'MB/s':>10}{'sec':>8}")
    for name, fn in [("take_blobs", via_take_blobs), ("take(column)", via_take)]:
        fn(ds, batches[:1], args.col)  # warmup
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            nbytes = fn(ds, batches, args.col)
            dt = time.perf_counter() - t0
            if best is None or dt < best[2]:
                best = (args.n / dt, nbytes / 1e6 / dt, dt)
        print(f"{name:<16}{best[0]:>12.1f}{best[1]:>10.1f}{best[2]:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
