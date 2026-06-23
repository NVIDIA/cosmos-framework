# SPDX-License-Identifier: OpenMDW-1.1
"""Throughput benchmark: base DROID action loader vs the LanceDB loader.

Measures steady-state samples/sec (and decoded video-frames/sec) through a
torch ``DataLoader``, warmup excluded — same methodology as
``lerobot_lancedb.benchmark``.

Modes:
  base       — DROIDLeRobotDataset (mp4 files, CPU torchcodec), N workers
  lance-cpu  — LanceDROIDDataset, blob-v2 + CPU torchcodec, N workers
  lance-gpu  — LanceDROIDDataset, blob-v2 + NVDEC, main process (num_workers=0)
"""
from __future__ import annotations

import argparse
import time

import torch

_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


def _collate(samples):
    out = {}
    for k in samples[0]:
        v = samples[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([s[k] for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


def _build(mode, root, uri, region=None):
    from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceDROIDDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    if mode == "base":
        return DROIDLeRobotDataset(root=root, **_KW)
    so = {"region": region} if region else None
    if mode.startswith("lance-composed"):
        dev = "cuda" if mode.endswith("gpu") else "cpu"
        return LanceDROIDComposedDataset(root=root, lance_uri=uri, decode_device=dev, storage_options=so, **_KW)
    dev = "cuda" if mode == "lance-gpu" else "cpu"
    return LanceDROIDDataset(root=root, lance_uri=uri, decode_device=dev, storage_options=so, **_KW)


def _measure(ds, *, batch_size, num_workers, num_batches, warmup):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    seen = 0
    t0 = None
    for i, batch in enumerate(loader):
        if mode_needs_sync(batch):
            torch.cuda.synchronize()
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += 1
        if seen >= num_batches:
            break
    dt = time.perf_counter() - t0
    sps = seen * batch_size / dt
    return sps, sps * (_KW["chunk_length"] + 1) * 3  # samples/s, decoded frames/s


def mode_needs_sync(batch):
    v = batch.get("video")
    return torch.is_tensor(v) and v.is_cuda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--modes", nargs="+", default=["base", "lance-cpu", "lance-gpu"])
    ap.add_argument("--region", default=None, help="storage_options region for s3:// lance uri")
    args = ap.parse_args()

    print(f"batch_size={args.batch_size} num_batches={args.num_batches} warmup={args.warmup}\n")
    print(f"{'mode':<12}{'workers':>8}{'samples/s':>14}{'videoframes/s':>16}{'speedup':>10}")
    base_sps = None
    for mode in args.modes:
        workers = 0 if mode == "lance-gpu" else args.num_workers
        ds = _build(mode, args.root, args.uri, region=args.region)
        sps, fps = _measure(
            ds,
            batch_size=args.batch_size,
            num_workers=workers,
            num_batches=args.num_batches,
            warmup=args.warmup,
        )
        if mode == "base":
            base_sps = sps
        spd = f"{sps / base_sps:.2f}x" if base_sps else "-"
        print(f"{mode:<12}{workers:>8}{sps:>14.1f}{fps:>16.0f}{spd:>10}")


if __name__ == "__main__":
    main()
