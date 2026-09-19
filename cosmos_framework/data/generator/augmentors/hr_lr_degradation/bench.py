# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""E1: throughput and peak-memory benchmark for the HR-to-LR degradation operators.

Measures seconds per clip and peak memory for each profile on synthetic clips, in three settings:

- ``cpu``: one process, ``--threads`` torch threads (what one dataloader worker sees).
- ``cpu-pool``: ``--workers`` processes degrading clips concurrently (aggregate clips/s, like a
  dataloader with that many workers on one node).
- ``cuda``: one GPU, batched by ``--chunk-frames``.

Example::

    PYTHONPATH=. python -m cosmos_framework.data.generator.augmentors.hr_lr_degradation.bench \
        --sizes 720x1280 1080x1920 --frames 121 --profiles p0_clean_bicubic p1_first_order p1_second_order \
        p3_video_codec --workers 6 --out e1_results.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import resource
import sys
import time

import torch

from cosmos_framework.data.generator.augmentors.hr_lr_degradation.degrade import degrade_hr_to_lr
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.diffjpeg import DiffJPEG


def _synthetic_clip(frames: int, h: int, w: int, seed: int) -> torch.Tensor:  # returns [3,T,H,W] uint8
    """Smooth content plus texture so JPEG/codec have realistic work (pure noise compresses unrealistically)."""
    # Built frame by frame into a uint8 buffer so the benchmark's own footprint stays at the uint8 clip
    # size; peak RSS then reflects the degradation operators rather than clip synthesis.
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0, 1, h).view(1, h, 1)  # [1,H,1]
    xs = torch.linspace(0, 1, w).view(1, 1, w)  # [1,1,W]
    texture = torch.nn.functional.interpolate(
        torch.rand(1, 3, h // 8, w // 8, generator=g), size=(h, w), mode="bilinear", align_corners=False
    )[0]  # [3,H,W]
    clip = torch.empty(3, frames, h, w, dtype=torch.uint8)  # [3,T,H,W]
    for t in range(frames):
        blue = torch.full((1, h, w), 0.5 + 0.5 * t / max(1, frames - 1))  # [1,H,W]
        base = torch.cat([ys.expand(1, h, w), xs.expand(1, h, w), blue])  # [3,H,W]
        frame = 0.7 * base + 0.3 * texture  # [3,H,W]
        clip[:, t] = (frame.clamp(0, 1) * 255).round().to(torch.uint8)
    return clip


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _run_one(hr: torch.Tensor, profile: str, seed: int, chunk_frames: int, jpeger: DiffJPEG) -> float:
    if hr.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    degrade_hr_to_lr(hr, profile, seed=seed, chunk_frames=chunk_frames, jpeger=jpeger)
    if hr.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_single(
    device: str, size: tuple[int, int], frames: int, profile: str, reps: int, chunk_frames: int, threads: int
):
    torch.set_num_threads(threads)
    hr = _synthetic_clip(frames, *size, seed=0).to(device)
    jpeger = DiffJPEG().to(device)
    _run_one(hr, profile, 0, chunk_frames, jpeger)  # warm-up
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = [_run_one(hr, profile, 1 + r, chunk_frames, jpeger) for r in range(reps)]
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else _peak_rss_gb()
    return sum(times) / len(times), mem


def _pool_worker(args):
    size, frames, profile, seed, chunk_frames, threads = args
    torch.set_num_threads(threads)
    hr = _synthetic_clip(frames, *size, seed=seed)
    t0 = time.perf_counter()
    degrade_hr_to_lr(hr, profile, seed=seed, chunk_frames=chunk_frames)
    return time.perf_counter() - t0, _peak_rss_gb()


def bench_pool(
    size: tuple[int, int], frames: int, profile: str, workers: int, clips: int, chunk_frames: int, threads: int
):
    ctx = mp.get_context("forkserver")
    jobs = [(size, frames, profile, s, chunk_frames, threads) for s in range(clips)]
    with ctx.Pool(workers) as pool:
        # Warm every worker first (torch import, kernel caches) so the timing reflects steady state,
        # as in a long-running dataloader, rather than process start-up.
        pool.map(_pool_worker, [((64, 64), 4, profile, 10_000 + w, chunk_frames, threads) for w in range(workers)])
        t0 = time.perf_counter()
        results = pool.map(_pool_worker, jobs)
        wall = time.perf_counter() - t0
    per_clip = sum(r[0] for r in results) / len(results)
    peak_rss_per_worker = max(r[1] for r in results)
    return wall / clips, per_clip, peak_rss_per_worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["720x1280", "1080x1920"], help="HxW of the HR clip")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--profiles", nargs="+", default=["p0_clean_bicubic", "p1_first_order", "p1_second_order"])
    parser.add_argument(
        "--settings", nargs="+", default=["cpu", "cpu-pool", "cuda"], choices=["cpu", "cpu-pool", "cuda"]
    )
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pool-clips", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4, help="torch threads per process")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--out", type=str, default=None, help="write a markdown table here")
    args = parser.parse_args(argv)

    rows: list[str] = [
        "| HR size | frames | setting | profile | s/clip | ms/frame | peak mem |",
        "|---|---|---|---|---|---|---|",
    ]
    for size_str in args.sizes:
        h, w = (int(v) for v in size_str.split("x"))
        for profile in args.profiles:
            for setting in args.settings:
                if setting == "cuda" and not torch.cuda.is_available():
                    continue
                if setting == "cpu-pool":
                    wall_per_clip, per_clip, rss = bench_pool(
                        (h, w), args.frames, profile, args.workers, args.pool_clips, args.chunk_frames, args.threads
                    )
                    label = f"cpu x{args.workers} workers"
                    row = (
                        f"| {h}x{w} | {args.frames} | {label} | {profile} | {wall_per_clip:.2f} (aggregate), {per_clip:.1f} (per worker) "
                        f"| {wall_per_clip * 1000 / args.frames:.1f} (aggregate) | {rss:.2f} GB RSS / worker |"
                    )
                else:
                    per_clip, mem = bench_single(
                        setting, (h, w), args.frames, profile, args.reps, args.chunk_frames, args.threads
                    )
                    unit = "GB GPU" if setting == "cuda" else "GB RSS"
                    row = f"| {h}x{w} | {args.frames} | {setting} | {profile} | {per_clip:.2f} | {per_clip * 1000 / args.frames:.1f} | {mem:.2f} {unit} |"
                print(row, flush=True)
                rows.append(row)
    table = "\n".join(rows)
    if args.out:
        with open(args.out, "w") as f:
            f.write(
                f"# E1 results ({os.uname().nodename}, torch {torch.__version__}, threads={args.threads})\n\n{table}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
