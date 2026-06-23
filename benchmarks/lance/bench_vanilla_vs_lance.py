# SPDX-License-Identifier: OpenMDW-1.1
"""Reproduce lerobot-lancedb's benchmark methodology on DROID, to attribute the
speedup. Compares three loaders on the SAME data + read pattern (delta windows,
CPU decode, N workers):

  vanilla-lerobot  — upstream LeRobotDataset (parquet+mp4) — lerobot-lancedb's baseline
  lance-video-cpu  — LeRobotLanceVideoDataset (blob-v2), CPU torchcodec
  lance-video-gpu  — LeRobotLanceVideoDataset (blob-v2), NVDEC (num_workers=0)

The point: lerobot-lancedb's 3-5x is vs *vanilla* LeRobotDataset. Cosmos's
DROIDLeRobotDataset is already optimized (cached batched torchcodec), so it is a
much harder baseline — see bench_decode.py for lance-vs-cosmos-base.
"""
from __future__ import annotations

import argparse
import time

import torch


def _measure(ds, *, batch_size, num_workers, num_batches, warmup):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
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
    ap.add_argument("--root", required=True, help="cosmos-format success dir (parquet+mp4)")
    ap.add_argument("--lance-root", required=True, help="lance dir from convert_to_lance_video")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_lancedb import LeRobotLanceVideoDataset

    cams = [
        "observation.image.wrist_image_left",
        "observation.image.exterior_image_1_left",
        "observation.image.exterior_image_2_left",
    ]
    fps = 15
    dts = {c: [i / fps for i in range(args.frames)] for c in cams}

    print(f"frames/sample={args.frames} batch={args.batch_size} workers={args.num_workers}\n")
    print(f"{'loader':<20}{'workers':>8}{'samples/s':>12}{'frames/s':>12}{'speedup':>10}")

    base_sps = None
    # vanilla LeRobotDataset
    v = LeRobotDataset("local/droid", root=args.root, delta_timestamps=dts)
    sps = _measure(v, batch_size=args.batch_size, num_workers=args.num_workers,
                   num_batches=args.num_batches, warmup=args.warmup)
    base_sps = sps
    print(f"{'vanilla-lerobot':<20}{args.num_workers:>8}{sps:>12.1f}{sps*args.frames*3:>12.0f}{'1.00x':>10}")

    # cosmos optimized base (DROIDLeRobotDataset) — already cached+batched torchcodec
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    def _tol_collate(samples):
        out = {}
        for k in samples[0]:
            vv = samples[0][k]
            out[k] = torch.stack([s[k] for s in samples]) if torch.is_tensor(vv) else [s[k] for s in samples]
        return out

    cb = DROIDLeRobotDataset(root=args.root, action_space="joint_pos", use_state=True,
                             mode="policy", chunk_length=args.frames)
    loader = torch.utils.data.DataLoader(cb, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True,
                                         persistent_workers=True, prefetch_factor=2,
                                         collate_fn=_tol_collate)
    seen, t0 = 0, None
    for i, _ in enumerate(loader):
        if i == args.warmup:
            t0 = time.perf_counter()
        if i >= args.warmup:
            seen += 1
        if seen >= args.num_batches:
            break
    sps = seen * args.batch_size / (time.perf_counter() - t0)
    print(f"{'cosmos-base (opt)':<20}{args.num_workers:>8}{sps:>12.1f}{sps*args.frames*3:>12.0f}{sps/base_sps:>9.2f}x")

    # lance video, CPU (lerobot-lancedb's video-blob path is CPU-only)
    lc = LeRobotLanceVideoDataset(root=args.lance_root, return_uint8=True, delta_timestamps=dts)
    sps = _measure(lc, batch_size=args.batch_size, num_workers=args.num_workers,
                   num_batches=args.num_batches, warmup=args.warmup)
    print(f"{'lance-video-cpu':<20}{args.num_workers:>8}{sps:>12.1f}{sps*args.frames*3:>12.0f}{sps/base_sps:>9.2f}x")


if __name__ == "__main__":
    main()
