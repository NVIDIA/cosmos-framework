# SPDX-License-Identifier: OpenMDW-1.1
"""Cold-cache action-loader benchmark: does the warm benchmark hide a base I/O cost?

The standard LOCAL benchmark reads a few OS-page-cached files, so file I/O is free —
the base loader's best case. Real cosmos-scale data does NOT fit in RAM, so reads are
cold. This script measures base vs lance throughput with the OS page cache dropped
before the measured pass (``--drop-caches`` needs sudo), optionally per epoch.

To simulate a dataset *larger than RAM* on a big-memory box, run this whole script
inside a memory-capped cgroup so the page cache is bounded and evicts during the run:

    sudo systemd-run --scope -p MemoryMax=3G -p MemorySwapMax=0 -- \
        python benchmarks/lance/bench_cold_cache.py --root ... --uri ... --drop-caches

Reports samples/s for base-episode and lance-episode (same episode-shuffle both sides).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import torch

_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


def _collate(items):
    return torch.stack([s["video"] for s in items])


def _drop_caches():
    subprocess.run(["sync"], check=False)
    r = subprocess.run(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _build(mode, root, uri):
    from bench_action_faithful import _EpisodeShuffle

    from cosmos_framework.data.lance import LanceDROIDComposedDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    if mode == "base":
        return _EpisodeShuffle(DROIDLeRobotDataset(root=root, **_KW))
    comp = LanceDROIDComposedDataset(root=root, lance_uri=uri, decode_device="cpu", decoder_cache_size=16, **_KW)
    return _EpisodeShuffle(comp)


def _epoch_sps(ds, *, batch_size, num_workers, batches):
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
        drop_last=True, persistent_workers=False,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    t0 = time.perf_counter()
    seen = 0
    for i, _ in enumerate(loader):
        seen += 1
        if seen >= batches:
            break
    return seen * batch_size / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--drop-caches", action="store_true", help="sudo drop page cache before each measured pass")
    ap.add_argument("--modes", nargs="+", default=["base", "lance"])
    args = ap.parse_args()

    mem = "?"
    try:  # show the cgroup memory cap if we're in a capped scope
        with open(f"/sys/fs/cgroup/{open('/proc/self/cgroup').read().strip().split(':')[-1]}/memory.max") as f:
            mem = f.read().strip()
    except Exception:
        pass
    print(f"COLD-CACHE action bench  drop_caches={args.drop_caches}  cgroup memory.max={mem}  "
          f"batch={args.batch_size} workers={args.num_workers} batches={args.batches}")
    print(f"{'mode':<14}{'cold sps':>12}{'warm sps':>12}{'cold penalty':>14}")
    for mode in args.modes:
        if args.drop_caches and not _drop_caches():
            print(f"  ({mode}) WARN: could not drop caches (need passwordless sudo)")
        cold = _epoch_sps(_build(mode, args.root, args.uri),
                          batch_size=args.batch_size, num_workers=args.num_workers, batches=args.batches)
        warm = _epoch_sps(_build(mode, args.root, args.uri),
                          batch_size=args.batch_size, num_workers=args.num_workers, batches=args.batches)
        pen = f"{(1 - cold / warm) * 100:.0f}%" if warm else "-"
        print(f"{mode:<14}{cold:>12.1f}{warm:>12.1f}{pen:>14}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
