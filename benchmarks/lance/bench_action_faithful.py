# SPDX-License-Identifier: OpenMDW-1.1
"""Faithful action-loader dataloader-throughput benchmark.

The production base loader uses EPISODE-SHUFFLE (`ActionIterableShuffleDataset`,
`iterable_shuffle=True`), not RandomSampler. So the apples-to-apples comparison is
episode-shuffle on BOTH sides. We also include lance-random to show that batched
take_blobs + concurrency (LANCE_IO_THREADS) makes random S3 reads competitive too.

Pure dataloader throughput (no model). Stressful config: many episodes (decoder cache
<< episodes), 8+ workers, batch 16, long steady-state. Set LANCE_IO_THREADS=256 for S3.

  modes: base-episode | lance-episode | lance-random
"""
from __future__ import annotations

import argparse
import time

import torch

_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


def _collate(items):
    return torch.stack([s["video"] for s in items])


class _EpisodeShuffle(torch.utils.data.IterableDataset):
    """Generic episode-shuffle stream (mirrors base ActionIterableShuffleDataset):
    shuffle per-episode block order, stream windows within a block sequentially,
    shard disjointly across (rank, worker). Works on any dataset exposing
    get_shuffle_blocks() + __getitem__ (base DROIDLeRobotDataset and lance composed)."""

    def __init__(self, ds, seed: int = 42):
        self.ds = ds
        self.seed = seed
        self.shard_rank = 0
        self.shard_world_size = 1

    def __iter__(self):
        blocks = self.ds.get_shuffle_blocks()
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        shard = self.shard_rank * nw + wid
        total = max(1, self.shard_world_size * nw)
        ep = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + ep)
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[shard::total]:
                s, length = blocks[b]
                for i in range(s, s + length):
                    yield self.ds[i]
            ep += 1


def _build(mode, root, uri, region, cache):
    from cosmos_framework.data.lance import LanceDROIDComposedDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    so = {"region": region} if region else None
    if mode == "base-episode":
        return _EpisodeShuffle(DROIDLeRobotDataset(root=root, **_KW)), None
    comp = LanceDROIDComposedDataset(root=root, lance_uri=uri, decode_device="cpu",
                                     decoder_cache_size=cache, storage_options=so, **_KW)
    if mode == "lance-episode":
        return _EpisodeShuffle(comp), None
    return comp, "random"  # lance-random -> RandomSampler


def _measure(ds, sampler_kind, *, batch_size, num_workers, num_batches, warmup):
    kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
              persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
              multiprocessing_context="spawn" if num_workers > 0 else None)
    if sampler_kind == "random":
        g = torch.Generator(); g.manual_seed(0)
        loader = torch.utils.data.DataLoader(ds, sampler=torch.utils.data.RandomSampler(ds, generator=g), **kw)
    else:
        loader = torch.utils.data.DataLoader(ds, **kw)  # IterableDataset (episode-shuffle)
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    return seen * batch_size / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument("--cache-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["base-episode", "lance-episode", "lance-random"])
    args = ap.parse_args()

    import os
    print(f"batch={args.batch_size} workers={args.num_workers} cache={args.cache_size} "
          f"num_batches={args.num_batches} LANCE_IO_THREADS={os.environ.get('LANCE_IO_THREADS','default')}\n")
    print(f"{'mode':<16}{'samples/s':>12}{'vs base':>10}")
    base = None
    for mode in args.modes:
        ds, sk = _build(mode, args.root, args.uri, args.region, args.cache_size)
        sps = _measure(ds, sk, batch_size=args.batch_size, num_workers=args.num_workers,
                       num_batches=args.num_batches, warmup=args.warmup)
        if mode == "base-episode":
            base = sps
        spd = f"{sps/base:.2f}x" if base else "-"
        print(f"{mode:<16}{sps:>12.1f}{spd:>10}", flush=True)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)  # skip torchcodec/lance C++ teardown SIGABRT (results already printed)