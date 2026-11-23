# SPDX-License-Identifier: OpenMDW-1.1
"""Throughput benchmark: the GENUINE vision-SFT base vs the LanceDB loader.

Base is the shipped ``SFTDataset`` (driven by :class:`BenchSFTDataset`, which only adds
single-shard setup + a direct Qwen tokenizer + a raw-mode flag — the per-sample hot path
``process_one_sample`` is unchanged). The SAME class is the base for both regimes:

  LOCAL — vision_path points at local mp4s; SFTDataset reads them (download_from_s3
          falls back to Path.read_bytes), spawns ffmpeg to decode+resize per sample.
  S3    — vision_path is rewritten to s3://; SFTDataset downloads each sample's mp4 via
          boto3 (genuine per-sample remote GET, no amortization) then decodes.

  lance — LanceVisionSFTDataset: decode a pre-resized, short-GOP per-clip blob; on S3 a
          columnar take of plain-binary clips (parallel IO).

``--mode raw`` skips tokenization on both sides to isolate the video I/O (the win is in
video I/O, not the storage-independent tokenize compute). Token-ids are otherwise exact.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import torch
from base_standins import BenchSFTDataset

from cosmos_framework.data.lance import LanceVisionSFTDataset

_KW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


def _collate(samples):
    out = {}
    for k in samples[0]:
        v = samples[0][k]
        if torch.is_tensor(v):
            try:
                out[k] = torch.stack([s[k] for s in samples])
            except Exception:
                out[k] = [s[k] for s in samples]  # ragged (e.g. text_token_ids)
        else:
            out[k] = [s[k] for s in samples]
    return out


def build_base(jsonl, tokenize, *, s3_bucket=None, s3_prefix=None):
    """Genuine SFTDataset (iterable) over local or s3:// vision paths."""
    return BenchSFTDataset.from_jsonl(
        jsonl, s3_bucket=s3_bucket, s3_prefix=s3_prefix, skip_tokenize=not tokenize, **_KW
    )


def build_lance(uri, tokenize, *, region=None, table="vision_sft"):
    so = {"region": region} if (region and str(uri).startswith("s3://")) else None
    ds = LanceVisionSFTDataset(uri, table=table, decode_device="cpu", storage_options=so, **_KW)
    ds.skip_tokenize = not tokenize
    return ds


def _measure_iter(ds, *, batch_size, num_workers, num_batches, warmup):
    """Steady-state samples/s for an IterableDataset (genuine SFT base)."""
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    return seen * batch_size / (time.perf_counter() - t0)


def _measure_map(ds, *, batch_size, num_workers, num_batches, warmup):
    """Steady-state samples/s for the map-style Lance loader (global-shuffle RandomSampler)."""
    n_total = (num_batches + warmup + 4) * batch_size
    g = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=n_total, generator=g)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    return seen * batch_size / (time.perf_counter() - t0)


def _side_entry(side, workers, a, q):
    """Subprocess entrypoint: build+measure one (side, workers) cell. Isolated per process
    so the ffmpeg/torchcodec/lance teardown can't SIGABRT a later cell."""
    tokenize = a["mode"] == "e2e"
    if side == "base":
        ds = build_base(a["jsonl"], tokenize, s3_bucket=a["s3_bucket"], s3_prefix=a["s3_prefix"])
        sps = _measure_iter(
            ds, batch_size=a["batch_size"], num_workers=workers, num_batches=a["num_batches"], warmup=a["warmup"]
        )
    else:
        ds = build_lance(a["uri"], tokenize, region=a["region"], table=a["table"])
        sps = _measure_map(
            ds, batch_size=a["batch_size"], num_workers=workers, num_batches=a["num_batches"], warmup=a["warmup"]
        )
    q.put(sps)
    q.close()
    q.join_thread()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--table", default="vision_sft")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--num-batches", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument(
        "--mode", choices=["raw", "e2e"], default="e2e", help="raw = video only (no tokenize); e2e = video + tokenize"
    )
    ap.add_argument("--modes", nargs="+", default=["base", "lance"])
    ap.add_argument("--region", default=None, help="storage_options region for an s3:// --uri")
    ap.add_argument(
        "--s3-bucket", default=None, help="if set, base reads each sample's mp4 from s3://bucket/<prefix>/<vision_path>"
    )
    ap.add_argument("--s3-prefix", default=None, help="key prefix the jsonl-relative vision_path lives under")
    args = ap.parse_args()

    a = vars(args)
    regime = "S3" if (args.s3_bucket and args.s3_prefix) else "LOCAL"
    print(
        f"mode={args.mode} regime={regime} batch_size={args.batch_size} "
        f"num_batches={args.num_batches} warmup={args.warmup}\n"
    )
    print(f"{'workers':>8}{'base sps':>12}{'lance sps':>12}{'speedup':>10}")
    ctx = mp.get_context("spawn")
    for workers in args.num_workers:
        sps = {}
        for side in ("base", "lance"):
            if side not in args.modes:
                continue
            q = ctx.Queue()
            p = ctx.Process(target=_side_entry, args=(side, workers, a, q))
            p.start()
            sps[side] = q.get()
            p.join()
        spd = sps["lance"] / sps["base"] if sps.get("base") else float("nan")
        print(
            f"{workers:>8}{sps.get('base', float('nan')):>12.1f}{sps.get('lance', float('nan')):>12.1f}{spd:>9.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
