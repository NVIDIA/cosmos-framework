# SPDX-License-Identifier: OpenMDW-1.1
"""Memory-footprint benchmark for the action / DROID loader: base vs LanceDB.

Throughput is only half the scaling story — the base loaders are also memory-heavy, and
that is what caps worker count (and OOMs) at full-DROID scale. This measures three axes,
one side per process (``--side base|lance``) so RSS is clean:

  1. INDEX memory — RSS after constructing the dataset (before any iteration). The base
     ``ActionBaseDataset.__init__`` materializes ``self._rows`` = one Python dict PER FRAME
     (~18M frames at full DROID = tens of GB, per its own code comment). For the DROID
     loader this is dead weight (it reads windows from compact numpy arrays via
     ``_window_rows`` and overrides ``__len__``), so the Lance loader frees it.
  2. ``_rows`` size — measured directly via del + gc (the redundant index materialization).
  3. RUNTIME memory — peak total RSS (main + all DataLoader workers) during steady-state
     iteration. The base decodes 3 full-resolution mega-mp4 views/sample; Lance decodes one
     small pre-composed 270x320 clip with a bounded per-worker decoder cache.

Reports per-worker RSS too — that is what multiplies by ``num_workers`` and decides how
many workers fit in RAM (the real scaling limit). Extrapolate index memory linearly in
frame count for full-dataset estimates.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import time

import psutil
import torch

from base_standins import S3DROIDLeRobotDataset
from cosmos_framework.data.lance import LanceDROIDComposedDataset
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_MB = 1024 * 1024


def _collate(items):
    return torch.stack([s["video"] for s in items])


def _build(side, root, uri, cache, s3_bucket=None, s3_prefix=None, region=None):
    if side == "base":
        if s3_bucket and s3_prefix:
            return S3DROIDLeRobotDataset(root=root, s3_bucket=s3_bucket, s3_prefix=s3_prefix, region=region, **_KW)
        return DROIDLeRobotDataset(root=root, **_KW)
    so = {"region": region} if (region and str(uri).startswith("s3://")) else None
    return LanceDROIDComposedDataset(root=root, lance_uri=uri, decode_device="cpu",
                                     decoder_cache_size=cache, storage_options=so, **_KW)


def _mem_tree(proc):
    """Return (total_rss, total_pss, per_worker_rss, per_worker_pss) for proc + children.

    PSS (proportional set size) splits each shared page across the procs mapping it, so it
    is the fair physical-RAM metric when fork shares pages copy-on-write; RSS double-counts
    those shared pages."""
    def _pss(p):
        try:
            return p.memory_full_info().pss
        except (psutil.Error, AttributeError):
            return p.memory_info().rss  # fallback if PSS unavailable

    total_rss = proc.memory_info().rss
    total_pss = _pss(proc)
    per_rss, per_pss = [], []
    for c in proc.children(recursive=True):
        try:
            per_rss.append(c.memory_info().rss)
            per_pss.append(_pss(c))
            total_rss += per_rss[-1]
            total_pss += per_pss[-1]
        except psutil.Error:
            pass
    return total_rss, total_pss, per_rss, per_pss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["base", "lance"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--cache-size", type=int, default=16)
    ap.add_argument("--mp-context", choices=["spawn", "fork"], default="spawn",
                    help="DataLoader worker start method. fork shares the parent's index via copy-on-write "
                    "(measure with PSS); lance fork support is experimental.")
    ap.add_argument("--free-base-rows", action="store_true",
                    help="(base only) free self._rows before iterating — isolates the per-worker _rows cost")
    ap.add_argument("--random", action="store_true",
                    help="iterate with a RandomSampler (touches all episodes across a scaled table) "
                    "instead of sequentially")
    ap.add_argument("--skip-iterate", action="store_true",
                    help="measure index/__init__ + spawn-payload memory only (no decode) — for scaled "
                    "parquet roots without matching video; the index is the term that scales/OOMs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    ds = _build(args.side, args.root, args.uri, args.cache_size,
                s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix, region=args.region)
    gc.collect()
    rss_after_init = proc.memory_info().rss
    n_frames = len(ds._row_episode)
    n_samples = len(ds)

    if args.free_base_rows and getattr(ds, "_rows", None) is not None:
        ds._rows = None
        gc.collect()

    # spawn per-worker payload: the bytes each spawn worker receives (pickle applies the
    # loader's __getstate__, so this is exactly what is shipped). With spawn this duplicates
    # into every worker; with fork the parent's pages are COW-shared instead.
    spawn_payload_mb = len(pickle.dumps(ds, protocol=pickle.HIGHEST_PROTOCOL)) / _MB

    if args.skip_iterate:
        rows_mb = float("nan")
        if getattr(ds, "_rows", None) is not None:
            gc.collect()
            before = proc.memory_info().rss
            ds._rows = None
            gc.collect()
            rows_mb = (before - proc.memory_info().rss) / _MB
        print(
            f"MEM_RESULT side={args.side} ctx=index-only workers=0 frames={n_frames} "
            f"init_index_mb={(rss_after_init - rss_before) / _MB:.0f} dead_rows_mb={rows_mb:.0f} "
            f"spawn_payload_mb={spawn_payload_mb:.1f} per_frame_payload_bytes={spawn_payload_mb * _MB / max(1, n_frames):.0f}",
            flush=True,
        )
        return

    # steady-state runtime RSS (main + workers) — measured with the dataset AS IT RUNS
    # (base keeps self._rows unless --free-base-rows; the Lance loaders free it in __init__),
    # so spawn workers carry exactly what the real loader would pickle to them.
    sampler = None
    if args.random:
        g = torch.Generator().manual_seed(0)
        sampler = torch.utils.data.RandomSampler(
            ds, replacement=True, num_samples=(args.num_batches + args.warmup + 4) * args.batch_size, generator=g)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=_collate,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None,
        multiprocessing_context=args.mp_context if args.num_workers > 0 else None,
    )
    peak_rss, peak_pss, rss_s, pss_s = 0, 0, [], []
    for i, _ in enumerate(loader):
        if i >= args.warmup:
            t_rss, t_pss, per_rss, per_pss = _mem_tree(proc)
            peak_rss = max(peak_rss, t_rss)
            peak_pss = max(peak_pss, t_pss)
            if per_rss:
                rss_s.append(sum(per_rss) / len(per_rss))
                pss_s.append(sum(per_pss) / len(per_pss))
        if i >= args.warmup + args.num_batches:
            break
    per_worker_mb = (sum(rss_s) / len(rss_s) / _MB) if rss_s else float("nan")
    per_worker_pss_mb = (sum(pss_s) / len(pss_s) / _MB) if pss_s else float("nan")
    peak_total = peak_rss

    # AFTER the runtime measurement, probe the size of self._rows (the per-frame dict list
    # the base ships to every spawn worker; the Lance loaders free it). Doing this last so it
    # can't perturb the runtime numbers above.
    rows_mb = float("nan")
    if getattr(ds, "_rows", None) is not None:
        gc.collect()
        before = proc.memory_info().rss
        ds._rows = None
        gc.collect()
        rows_mb = (before - proc.memory_info().rss) / _MB

    print(
        f"MEM_RESULT side={args.side} ctx={args.mp_context} workers={args.num_workers} frames={n_frames} "
        f"init_index_mb={(rss_after_init - rss_before) / _MB:.0f} dead_rows_mb={rows_mb:.0f} "
        f"peak_rss_mb={peak_rss / _MB:.0f} peak_pss_mb={peak_pss / _MB:.0f} "
        f"per_worker_rss_mb={per_worker_mb:.0f} per_worker_pss_mb={per_worker_pss_mb:.0f} "
        f"spawn_payload_mb={spawn_payload_mb:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    os._exit(0)  # skip torchcodec/lance teardown SIGABRT
