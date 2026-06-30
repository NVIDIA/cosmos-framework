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
import multiprocessing as mp
import os
import time

import torch
from base_standins import S3DROIDLeRobotDataset

from cosmos_framework.data.lance import LanceDROIDComposedDataset
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

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


def _build(mode, root, uri, region, cache, s3_bucket=None, s3_prefix=None):
    so = {"region": region} if region else None

    def _base():
        # genuine DROIDLeRobotDataset; for S3 the standin materializes the mega-mp4s first.
        if s3_bucket and s3_prefix:
            return S3DROIDLeRobotDataset(root=root, s3_bucket=s3_bucket, s3_prefix=s3_prefix, region=region, **_KW)
        return DROIDLeRobotDataset(root=root, **_KW)

    if mode == "base-random":
        return _base(), "random"
    if mode == "base-episode":
        return _EpisodeShuffle(_base()), None
    comp = LanceDROIDComposedDataset(
        root=root, lance_uri=uri, decode_device="cpu", decoder_cache_size=cache, storage_options=so, **_KW
    )
    if mode == "lance-episode":
        return _EpisodeShuffle(comp), None
    return comp, "random"  # lance-random -> RandomSampler


def _measure(ds, sampler_kind, *, batch_size, num_workers, num_batches, warmup):
    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    if sampler_kind == "random":
        g = torch.Generator()
        g.manual_seed(0)
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


def _mode_entry(mode, a, q):
    """Subprocess entrypoint: build+measure one mode, return its samples/s. Each mode runs
    in its own process so the torchcodec/lance C++ teardown can't SIGABRT a later mode."""
    ds, sk = _build(
        mode, a["root"], a["uri"], a["region"], a["cache_size"], s3_bucket=a["s3_bucket"], s3_prefix=a["s3_prefix"]
    )
    sps = _measure(
        ds,
        sk,
        batch_size=a["batch_size"],
        num_workers=a["num_workers"],
        num_batches=a["num_batches"],
        warmup=a["warmup"],
    )
    q.put(sps)
    q.close()
    q.join_thread()
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument(
        "--s3-bucket", default=None, help="if set, base materializes mega-mp4s from this bucket (S3 regime)"
    )
    ap.add_argument("--s3-prefix", default=None, help="key prefix the DROID videos/ tree lives under")
    ap.add_argument("--cache-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["base-episode", "lance-episode", "lance-random"])
    args = ap.parse_args()

    a = vars(args)
    print(
        f"batch={args.batch_size} workers={args.num_workers} cache={args.cache_size} "
        f"num_batches={args.num_batches} LANCE_IO_THREADS={os.environ.get('LANCE_IO_THREADS', 'default')}\n"
    )
    print(f"{'mode':<16}{'samples/s':>12}{'vs base':>10}")
    ctx = mp.get_context("spawn")
    base = None
    for mode in args.modes:
        q = ctx.Queue()
        p = ctx.Process(target=_mode_entry, args=(mode, a, q))
        p.start()
        sps = q.get()
        p.join()
        if mode == "base-episode":
            base = sps
        spd = f"{sps / base:.2f}x" if base else "-"
        print(f"{mode:<16}{sps:>12.1f}{spd:>10}", flush=True)


if __name__ == "__main__":
    main()
