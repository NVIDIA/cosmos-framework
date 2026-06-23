# SPDX-License-Identifier: OpenMDW-1.1
"""Validate the episode-shuffle borrow: RandomSampler vs LanceDROIDComposedIterable.

Episode-shuffle streams windows within an episode consecutively, so the per-episode clip
decoder is built once and reused. RandomSampler jumps episodes -> rebuilds the decoder
(take_blobs + VideoDecoder) whenever the per-worker LRU cache misses. The gap grows as the
cache covers a smaller fraction of episodes (i.e. the real many-episode regime), which we
emulate with --cache-size.
"""
from __future__ import annotations

import argparse
import time

import torch

KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


def _collate(items):
    return torch.stack([s["video"] for s in items])


def _measure(loader, num_batches, warmup, bs):
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    return seen * bs / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/work/data/droid_cosmos/success")
    ap.add_argument("--uri", default="/home/ubuntu/work/data/lance/droid_composed")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--cache-size", type=int, default=4, help="per-worker decoder LRU (small = many-episode regime)")
    ap.add_argument("--region", default=None, help="storage_options region for s3:// uri")
    args = ap.parse_args()

    from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceDROIDComposedIterable

    so = {"region": args.region} if args.region else None
    ds = LanceDROIDComposedDataset(root=args.root, lance_uri=args.uri, decode_device="cpu",
                                   decoder_cache_size=args.cache_size, storage_options=so, **KW)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=_collate,
                  persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
                  multiprocessing_context="spawn" if args.num_workers > 0 else None)

    g = torch.Generator(); g.manual_seed(0)
    rand_loader = torch.utils.data.DataLoader(ds, sampler=torch.utils.data.RandomSampler(ds, generator=g), **common)
    rand_sps = _measure(rand_loader, args.num_batches, args.warmup, args.batch_size)

    epi_loader = torch.utils.data.DataLoader(LanceDROIDComposedIterable(ds, seed=0), **common)
    epi_sps = _measure(epi_loader, args.num_batches, args.warmup, args.batch_size)

    print(f"decoder_cache_size={args.cache_size} workers={args.num_workers} batch={args.batch_size}")
    print(f"{'sampler':<22}{'samples/s':>12}{'speedup':>10}")
    print(f"{'RandomSampler':<22}{rand_sps:>12.1f}{'1.00x':>10}")
    print(f"{'episode-shuffle':<22}{epi_sps:>12.1f}{epi_sps/rand_sps:>9.2f}x")


if __name__ == "__main__":
    main()
