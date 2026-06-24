# SPDX-License-Identifier: OpenMDW-1.1
"""Measure which LanceDB blob-read levers actually move throughput, local + S3.

Reads the per-episode composed-DROID mp4 blobs (blob-v2) and reports MB/s and
clips/s for take_blobs under varying:
  * LANCE_IO_THREADS (set in the env BEFORE launching — printed for the record)
  * io_buffer_size (storage_options)
  * sorted vs shuffled indices (coalescing of byte-range GETs)
  * batch size of the take_blobs index list
This isolates the data-access layer (no decode) so the levers are visible.
"""
from __future__ import annotations

import argparse
import os
import time

import lance


def _read_blobs(ds, indices, col):
    blobs = ds.take_blobs(col, indices=indices)
    nbytes = 0
    for b in blobs:
        data = b.readall()
        nbytes += len(data)
        b.close()
    return nbytes


def run(uri, *, region, col, n, batch, sort, buffer_mb, repeats):
    so = {}
    if region:
        so["region"] = region
    if buffer_mb:
        so["io_buffer_size"] = str(buffer_mb * 1024 * 1024)
    ds = lance.dataset(uri, storage_options=so or None)
    total = ds.count_rows()
    import random

    rng = random.Random(0)
    # cycle through rows to reach n reads
    idx_pool = [i % total for i in range(n)]
    rng.shuffle(idx_pool)
    if sort:
        # sort within each batch -> adjacent rows coalesce into fewer GETs
        batches = [sorted(idx_pool[i : i + batch]) for i in range(0, n, batch)]
    else:
        batches = [idx_pool[i : i + batch] for i in range(0, n, batch)]

    # warmup one batch
    _read_blobs(ds, batches[0], col)
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        nbytes = 0
        nread = 0
        for b in batches:
            nbytes += _read_blobs(ds, b, col)
            nread += len(b)
        dt = time.perf_counter() - t0
        mbps = nbytes / 1e6 / dt
        cps = nread / dt
        if best is None or cps > best[0]:
            best = (cps, mbps, dt)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument("--col", default="video_bytes")
    ap.add_argument("--n", type=int, default=2000, help="total blob reads")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--buffer-mb", type=int, nargs="+", default=[0], help="io_buffer_size variants (0=default)")
    ap.add_argument("--sorts", nargs="+", type=int, default=[0, 1], help="0=shuffled 1=sorted-per-batch")
    args = ap.parse_args()

    regime = "S3" if args.region else "LOCAL"
    print(
        f"[{regime}] uri={args.uri}\n"
        f"n={args.n} batch={args.batch} repeats={args.repeats} "
        f"LANCE_IO_THREADS={os.environ.get('LANCE_IO_THREADS','default')}\n"
    )
    print(f"{'sorted':>7}{'buf_mb':>8}{'clips/s':>12}{'MB/s':>10}{'sec':>8}")
    for buf in args.buffer_mb:
        for sort in args.sorts:
            cps, mbps, dt = run(
                args.uri, region=args.region, col=args.col, n=args.n,
                batch=args.batch, sort=bool(sort), buffer_mb=buf, repeats=args.repeats,
            )
            print(f"{sort:>7}{buf:>8}{cps:>12.1f}{mbps:>10.1f}{dt:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
