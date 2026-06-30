# SPDX-License-Identifier: OpenMDW-1.1
"""Faithful combined 3-dataloader throughput benchmark: base-trio vs lance-trio.

Every base side is a GENUINE shipped Cosmos loader (no reconstructions):

  * ACTION   — DROIDLeRobotDataset (LOCAL) / S3DROIDLeRobotDataset (S3 standin, which
               just materializes the mega-mp4s from S3 then runs the identical base
               decode). EPISODE-SHUFFLE on both sides (the production shuffle).
  * VLM      — get_llava_ov_streaming: the shipped HF-Hub streaming factory, imported
               and called directly. Cosmos has no local/S3 VLM base, so this is the base
               in every regime (sequential shards + shuffle buffer, no random access).
  * VISION-SFT — the shipped SFTDataset (via BenchSFTDataset). LOCAL reads local mp4s;
               S3 rewrites vision_path to s3:// so SFTDataset downloads each sample's mp4
               via boto3 — the genuine per-sample remote path.

Two regimes, reported separately:
  LOCAL — all loaders local (cosmos's pre-download-then-train workflow).
  S3    — action via the S3 standin, vision-SFT via genuine boto3 per-sample, VLM via HF
          streaming (its only mode); Lance reads s3:// natively.

RAW mode (no Qwen image-processor — that is model work, not the dataloader's job). The
1:1:1 mixer aggregate is gated by the SLOWEST loader, so report the combined number WITH
the per-loader breakdown, never as a bare multiple. Run ``--trios base`` and
``--trios lance`` in SEPARATE processes (one process hits the torchcodec/lance teardown
SIGABRT between trios).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bench_vision_sft  # noqa: E402  (kept loader benches)
import bench_vlm  # noqa: E402
from base_standins import S3DROIDLeRobotDataset  # noqa: E402
from bench_action_faithful import _EpisodeShuffle  # noqa: E402

from cosmos_framework.data.lance import (  # noqa: E402
    LanceDROIDComposedDataset,
    LanceVisionSFTDataset,
)
from cosmos_framework.data.lance.vlm_dataset import LanceVLMShuffleScan  # noqa: E402
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset  # noqa: E402

_ACTION_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_VSFT_KW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


# ── runner helpers ──
def _action_collate(samples):
    out = {}
    for k in samples[0]:
        v = samples[0][k]
        out[k] = torch.stack([s[k] for s in samples]) if torch.is_tensor(v) else [s[k] for s in samples]
    return out


def _batch_count(batch, batch_size):
    if isinstance(batch, (list, tuple)):
        return len(batch)
    if isinstance(batch, dict):
        for v in batch.values():
            try:
                return len(v)
            except TypeError:
                continue
        return batch_size
    if torch.is_tensor(batch):
        return batch.shape[0]
    try:
        return len(batch)
    except TypeError:
        return batch_size


class _InfiniteLoader:
    def __init__(self, loader, name):
        self.loader, self.name, self.it = loader, name, iter(loader)

    def next_batch(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def _standalone_sps(loader, *, batch_size, rounds, warmup):
    inf = _InfiniteLoader(loader, "standalone")
    seen, t0 = 0, None
    for i in range(rounds + warmup):
        b = inf.next_batch()
        n = _batch_count(b, batch_size)
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += n
    return seen / (time.perf_counter() - t0)


def _combined_sps(loaders, names, *, batch_size, rounds, warmup):
    infs = [_InfiniteLoader(ld, nm) for ld, nm in zip(loaders, names)]
    seen, t0 = 0, None
    for r in range(rounds + warmup):
        if r == warmup:
            t0 = time.perf_counter()
        for inf in infs:
            n = _batch_count(inf.next_batch(), batch_size)
            if r >= warmup:
                seen += n
    return seen / (time.perf_counter() - t0)


def _so(region, uri):
    """storage_options only for s3:// uris — lets one run mix local + S3 loaders."""
    return {"region": region} if (region and str(uri).startswith("s3://")) else None


# ── per-loader builders (genuine bases) ──
def build_action_loader(which, root, uri, region, cache, batch_size, num_workers, s3_bucket=None, s3_prefix=None):
    if which == "base":
        if s3_bucket and s3_prefix:  # genuine base + S3 materialization standin
            base = S3DROIDLeRobotDataset(
                root=root, s3_bucket=s3_bucket, s3_prefix=s3_prefix, region=region, **_ACTION_KW
            )
        else:
            base = DROIDLeRobotDataset(root=root, **_ACTION_KW)
        ds = _EpisodeShuffle(base)
    else:
        comp = LanceDROIDComposedDataset(
            root=root,
            lance_uri=uri,
            decode_device="cpu",
            decoder_cache_size=cache,
            storage_options=_so(region, uri),
            **_ACTION_KW,
        )
        ds = _EpisodeShuffle(comp)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_action_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def build_vlm_loader(which, uri, region, batch_size, num_workers, hf_subset):
    collate = bench_vlm.Collate("raw")
    if which == "base":
        return torch.utils.data.DataLoader(
            bench_vlm.GenuineVLMBase(hf_subset),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
    ds = LanceVLMShuffleScan(uri, "llava", buffer_size=1000, storage_options=_so(region, uri))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )  # lance not fork-safe


def build_vsft_loader(which, jsonl, uri, region, batch_size, num_workers, n_total, s3_bucket, s3_prefix):
    if which == "base":
        # genuine SFTDataset (iterable): local mp4s, or boto3 per-sample for s3://
        ds = bench_vision_sft.build_base(jsonl, tokenize=False, s3_bucket=s3_bucket, s3_prefix=s3_prefix)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=bench_vision_sft._collate,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
    ds = LanceVisionSFTDataset(
        uri, table="vision_sft", decode_device="cpu", storage_options=_so(region, uri), **_VSFT_KW
    )
    ds.skip_tokenize = True
    g = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=n_total, generator=g)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=bench_vision_sft._collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def run_trio(
    which,
    paths,
    *,
    region,
    cache,
    batch_size,
    workers,
    rounds,
    warmup,
    vsft_n_total,
    action_s3_bucket,
    action_s3_prefix,
    vsft_s3_bucket,
    vsft_s3_prefix,
    vlm_hf_subset,
):
    aw, vw, sw = workers["action"], workers["vlm"], workers["vision-sft"]
    print(f"\n========== {which.upper()}-TRIO (faithful) workers a={aw}/v={vw}/s={sw} ==========", flush=True)
    a = build_action_loader(
        which,
        paths["action_root"],
        paths["action_uri"],
        region,
        cache,
        batch_size,
        aw,
        s3_bucket=action_s3_bucket if which == "base" else None,
        s3_prefix=action_s3_prefix if which == "base" else None,
    )
    v = build_vlm_loader(which, paths["vlm_uri"], region, batch_size, vw, vlm_hf_subset)
    s = build_vsft_loader(
        which,
        paths["vsft_jsonl"],
        paths["vsft_uri"],
        region,
        batch_size,
        sw,
        vsft_n_total,
        vsft_s3_bucket if which == "base" else None,
        vsft_s3_prefix if which == "base" else None,
    )
    loaders, names = [a, v, s], ["action", "vlm", "vision-sft"]
    standalone = {}
    for ld, nm in zip(loaders, names):
        standalone[nm] = _standalone_sps(ld, batch_size=batch_size, rounds=rounds, warmup=warmup)
        print(f"    [{which}] standalone {nm:<12} {standalone[nm]:10.1f} samples/s", flush=True)
    agg = _combined_sps(loaders, names, batch_size=batch_size, rounds=rounds, warmup=warmup)
    print(f"    [{which}] combined mixer (1:1:1) {agg:10.1f} samples/s", flush=True)
    return standalone, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--action-root", required=True, help="local DROID root (parquet/meta index; videos local or via S3 standin)"
    )
    ap.add_argument("--action-uri", required=True)
    ap.add_argument(
        "--action-s3-bucket",
        default=None,
        help="if set, base action materializes mega-mp4s from this bucket (S3 regime)",
    )
    ap.add_argument("--action-s3-prefix", default=None, help="key prefix the DROID videos/ tree lives under")
    ap.add_argument("--vlm-uri", required=True)
    ap.add_argument(
        "--vlm-hf-subset",
        default="figureqa(cauldron,llava_format)",
        help="lmms-lab/LLaVA-OneVision-Data subset the base streams from HF (cosmos default)",
    )
    ap.add_argument("--vsft-jsonl", required=True)
    ap.add_argument("--vsft-uri", required=True)
    ap.add_argument(
        "--vsft-s3-bucket", default=None, help="if set, base vsft downloads each mp4 via boto3 (genuine S3 path)"
    )
    ap.add_argument("--vsft-s3-prefix", default=None, help="key prefix the jsonl-relative vision_path lives under")
    ap.add_argument("--region", default=None)
    ap.add_argument("--cache-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6, help="default per-loader worker count")
    ap.add_argument("--action-workers", type=int, default=None)
    ap.add_argument("--vlm-workers", type=int, default=None)
    ap.add_argument("--vsft-workers", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trios", nargs="+", default=["base", "lance"])
    args = ap.parse_args()

    paths = dict(
        action_root=args.action_root,
        action_uri=args.action_uri,
        vlm_uri=args.vlm_uri,
        vsft_jsonl=args.vsft_jsonl,
        vsft_uri=args.vsft_uri,
    )
    vsft_n_total = (args.rounds + args.warmup + 8) * args.batch_size
    regime = "S3" if args.region else "LOCAL"
    print(
        f"FAITHFUL COMBINED RAW [{regime}] — genuine bases; action=EPISODE-SHUFFLE both sides\n"
        f"batch={args.batch_size} workers={args.num_workers}/loader rounds={args.rounds} "
        f"LANCE_IO_THREADS={os.environ.get('LANCE_IO_THREADS', 'default')}",
        flush=True,
    )

    workers = {
        "action": args.action_workers or args.num_workers,
        "vlm": args.vlm_workers or args.num_workers,
        "vision-sft": args.vsft_workers or args.num_workers,
    }
    results = {}
    for which in args.trios:
        results[which] = run_trio(
            which,
            paths,
            region=args.region,
            cache=args.cache_size,
            batch_size=args.batch_size,
            workers=workers,
            rounds=args.rounds,
            warmup=args.warmup,
            vsft_n_total=vsft_n_total,
            action_s3_bucket=args.action_s3_bucket,
            action_s3_prefix=args.action_s3_prefix,
            vsft_s3_bucket=args.vsft_s3_bucket,
            vsft_s3_prefix=args.vsft_s3_prefix,
            vlm_hf_subset=args.vlm_hf_subset,
        )

    if "base" in results and "lance" in results:
        print("\n--- per-loader RAW samples/s ---")
        print(f"{'loader':<14}{'base':>12}{'lance':>12}{'speedup':>10}")
        for nm in ["action", "vlm", "vision-sft"]:
            b, l = results["base"][0].get(nm), results["lance"][0].get(nm)
            print(f"{nm:<14}{b:>12.1f}{l:>12.1f}{l / b:>9.2f}x")
        ba, la = results["base"][1], results["lance"][1]
        print(f"\ncombined (1:1:1)  base={ba:.1f}  lance={la:.1f}  speedup={la / ba:.2f}x")


if __name__ == "__main__":
    main()
    os._exit(0)
