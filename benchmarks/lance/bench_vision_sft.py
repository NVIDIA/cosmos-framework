# SPDX-License-Identifier: OpenMDW-1.1
"""Throughput benchmark: local vision-SFT loader vs the LanceDB loader.

Measures steady-state samples/sec through a torch ``DataLoader`` (warmup
excluded), same methodology as ``bench_action.py``. Both paths read the same
clips by index and feed the same tokenize step, so only the video-I/O differs:

  base   — LocalSFTDataset: seek source mp4 on disk, decode + resize per sample.
  lance  — LanceVisionSFTDataset: decode a pre-resized, short-GOP per-clip blob.

Shuffled (RandomSampler), CPU decode, LOCAL. ``--mode raw`` skips tokenization to
isolate the video path (the win is in video I/O, not the storage-independent
tokenize compute).
"""
from __future__ import annotations

import argparse
import time

import torch

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


def _build(mode, jsonl, uri, tokenize):
    from cosmos_framework.data.lance import LanceVisionSFTDataset
    from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import LocalSFTDataset

    # raw mode: skip tokenization by pointing both at a no-op tokenizer path.
    if mode == "base":
        ds = LocalSFTDataset(jsonl, **_KW)
    else:
        ds = LanceVisionSFTDataset(uri, table="vision_sft", decode_device="cpu", **_KW)
    ds.skip_tokenize = not tokenize  # raw mode: skip the storage-independent tokenize compute
    return ds


def _measure(ds, *, batch_size, num_workers, num_batches, warmup, n_total):
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
    dt = time.perf_counter() - t0
    return seen * batch_size / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--num-batches", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--mode", choices=["raw", "e2e"], default="e2e",
                    help="raw = video only (no tokenize); e2e = video + tokenize")
    ap.add_argument("--modes", nargs="+", default=["base", "lance"])
    args = ap.parse_args()

    tokenize = args.mode == "e2e"
    n_total = (args.num_batches + args.warmup + 4) * args.batch_size
    print(f"mode={args.mode} batch_size={args.batch_size} num_batches={args.num_batches} warmup={args.warmup}\n")
    print(f"{'workers':>8}{'base sps':>12}{'lance sps':>12}{'speedup':>10}")
    for workers in args.num_workers:
        sps = {}
        for m in args.modes:
            ds = _build(m, args.jsonl, args.uri, tokenize)
            sps[m] = _measure(
                ds, batch_size=args.batch_size, num_workers=workers,
                num_batches=args.num_batches, warmup=args.warmup, n_total=n_total,
            )
        spd = sps["lance"] / sps["base"] if "base" in sps and sps["base"] else float("nan")
        print(f"{workers:>8}{sps.get('base', float('nan')):>12.1f}{sps.get('lance', float('nan')):>12.1f}{spd:>9.2f}x")


if __name__ == "__main__":
    main()
