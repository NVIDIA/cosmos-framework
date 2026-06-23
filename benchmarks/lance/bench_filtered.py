# SPDX-License-Identifier: OpenMDW-1.1
"""Filtered / curriculum sampling: LanceDB predicate pushdown vs WebDataset.

Real training often samples a SUBSET (curriculum, quality filter, task/domain balance).
LanceDB pushes the predicate into the scan and reads ONLY matching rows' blobs. A
WebDataset tar is sequential + opaque: it must stream + parse EVERY sample and discard the
misses — it cannot skip. So Lance's filtered-read throughput scales ~1/selectivity while
webdataset stays flat at full-stream cost.

Both sides apply the SAME selectivity fraction (the win is reading only that fraction from
storage, regardless of which rows). Lance selects by last-digit of sample_id (uniform 10%
buckets); webdataset by __key__ index mod 10. Measured at the storage level (yield bytes,
no decode) since decode is identical per kept sample and not the point.
"""
from __future__ import annotations

import time

import lance
import webdataset as wds

LANCE = "/home/ubuntu/work/data/lance/llava_figureqa/llava.lance"
SHARDS = "/home/ubuntu/work/data/wds/llava_figureqa/shard-{00000..00019}.tar"

# selectivity % -> allowed last digits
SEL = {100: list("0123456789"), 50: list("01234"), 30: list("012"), 10: list("0")}


def lance_filtered(digits):
    ds = lance.dataset(LANCE)
    if len(digits) == 10:
        flt = None
    else:
        flt = " OR ".join(f"sample_id LIKE '%{d}.png'" for d in digits)
    t0 = time.perf_counter()
    kept = 0
    nbytes = 0
    scanner = ds.scanner(columns=["image_bytes"], filter=flt, batch_size=512)
    for b in scanner.to_batches():
        kept += b.num_rows
        nbytes += sum(len(x.as_py()) for x in b.column("image_bytes"))
    dt = time.perf_counter() - t0
    return kept, nbytes, dt


def wds_filtered(digits):
    keep = set(int(d) for d in digits)
    ds = wds.WebDataset(SHARDS, shardshuffle=False, empty_check=False)
    t0 = time.perf_counter()
    kept = 0
    kept_bytes = 0
    read_bytes = 0  # webdataset must read EVERY sample
    for s in ds:
        png = s["png"]
        read_bytes += len(png)
        if int(s["__key__"][6:]) % 10 in keep:
            kept += 1
            kept_bytes += len(png)
    dt = time.perf_counter() - t0
    return kept, kept_bytes, read_bytes, dt


def main():
    # warm OS cache for both
    lance_filtered(list("0123456789"))
    print(f"{'sel%':>5}{'lance kept/s':>14}{'wds kept/s':>12}{'speedup':>9}"
          f"{'lance MB read':>15}{'wds MB read':>13}{'bytes ratio':>13}")
    for pct in (100, 50, 30, 10):
        digits = SEL[pct]
        lk, lb, ldt = lance_filtered(digits)
        wk, wkb, wrb, wdt = wds_filtered(digits)
        lsps, wsps = lk / ldt, wk / wdt
        print(f"{pct:>5}{lsps:>14.0f}{wsps:>12.0f}{lsps/wsps:>8.2f}x"
              f"{lb/1e6:>15.0f}{wrb/1e6:>13.0f}{lb/wrb:>12.2f}x")


if __name__ == "__main__":
    main()
