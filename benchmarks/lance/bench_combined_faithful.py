# SPDX-License-Identifier: OpenMDW-1.1
"""Faithful combined 3-dataloader throughput benchmark: base-trio vs lance-trio.

HONEST by construction:
  1. ACTION uses the production base shuffle = EPISODE-SHUFFLE on BOTH sides
     (base DROIDLeRobotDataset and lance composed) — not RandomSampler.
  2. Two storage regimes, reported separately (cosmos trains from LOCAL DISK per its
     docs; S3 is Lance's object-store-native value-add):
       - LOCAL: all loaders read local disk (apples-to-apples, cosmos's real workflow).
       - S3: Lance reads natively from s3://. The base loaders have NO native S3 reader
         except vision-SFT, so for S3 the base accesses each dataset the way the stock
         loader actually would: action/VLM via the s3fs FUSE mount (the only option —
         see WHY in the README), vision-SFT via boto3 download-per-sample (what the stock
         `SFTDataset` does) when --vsft-s3-bucket/--vsft-s3-prefix are given.
  3. RAW mode (no Qwen image-processor — that is model work, not the dataloader's job).

The 1:1:1 mixer aggregate is gated by the SLOWEST loader (aggregate ≈ 3×slowest), so the
combined "speedup" tracks whichever loader bottlenecks each trio — report it WITH the
per-loader breakdown, never as a bare multiple. Run `--trios base` and `--trios lance`
in SEPARATE processes (a single process hits the torchcodec/lance teardown SIGABRT
between trios).
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
from bench_action_faithful import _EpisodeShuffle  # noqa: E402
from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import LocalSFTDataset  # noqa: E402

_ACTION_KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)
_VSFT_KW = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")


# ── self-contained helpers (formerly imported from bench_combined / bench_action) ──
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


# ── vision-SFT base: stock boto3 download-per-sample (mirrors SFTDataset) ──
class _Boto3SFTBase(LocalSFTDataset):
    """Stock-faithful S3 vision-SFT base: identical to LocalSFTDataset except each
    video is fetched via boto3 download-per-sample (what cosmos `SFTDataset` does via
    `download_from_s3` in sft_dataset.py). JSONL/metadata loads locally; only the
    per-sample video bytes come over boto3 — isolating the stock S3 access cost.
    Module-level subclass with real methods + __getstate__ so it pickles to spawn workers."""

    def __init__(self, jsonl, bucket, prefix, **kw):
        super().__init__(jsonl, **kw)
        self.skip_tokenize = True
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3 = None  # lazy, per-worker (never pickled)
        self._tmp = None

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_s3"] = None
        st["_tmp"] = None
        return st

    def _resolve_path(self, vision_path: str) -> str:
        if self._s3 is None:
            import boto3
            self._s3 = boto3.Session(
                profile_name=os.environ.get("AWS_PROFILE", "cosmosbench"),
                region_name=os.environ.get("AWS_REGION", "us-east-2"),
            ).client("s3")
            self._tmp = f"/tmp/_vsft_boto3_{os.getpid()}.mp4"
        self._s3.download_file(self._bucket, f"{self._prefix}/{vision_path}", self._tmp)
        return self._tmp


# ── per-loader builders ──
def build_action_loader(which, root, uri, region, cache, batch_size, num_workers):
    from cosmos_framework.data.lance import LanceDROIDComposedDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    if which == "base":
        ds = _EpisodeShuffle(DROIDLeRobotDataset(root=root, **_ACTION_KW))
    else:
        comp = LanceDROIDComposedDataset(root=root, lance_uri=uri, decode_device="cpu",
                                         decoder_cache_size=cache, storage_options=_so(region, uri), **_ACTION_KW)
        ds = _EpisodeShuffle(comp)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_action_collate,
        drop_last=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


class _HFStreamVLM(torch.utils.data.IterableDataset):
    """Cosmos's actual default VLM base: lmms-lab/LLaVA-OneVision-Data streamed from the
    HF Hub (`get_llava_ov_streaming`). Builds the stream fresh in __iter__ (the HF filter
    lambda isn't picklable for spawn), yields the raw {id, image(PIL), conversations} dict."""

    def __init__(self, subset):
        self.subset = subset

    def __iter__(self):
        # Inlined verbatim from cosmos_framework/.../llava_ov_vlm.py::get_llava_ov_streaming
        # (importing that module pulls the cosmos VLM processor chain). Same load_dataset call.
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=self.subset, split="train", streaming=True)
        ds = ds.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)
        yield from ds


def _so(region, uri):
    """storage_options only for s3:// uris — lets one run mix local + S3 loaders."""
    return {"region": region} if (region and str(uri).startswith("s3://")) else None


def build_vlm_loader(which, wds, uri, region, batch_size, num_workers, hf_subset=None):
    collate = bench_vlm.Collate("raw")
    if which == "base":
        if hf_subset:  # cosmos default: HF-Hub streaming
            return torch.utils.data.DataLoader(
                _HFStreamVLM(hf_subset), batch_size=batch_size, num_workers=num_workers,
                collate_fn=collate, persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
                multiprocessing_context="spawn" if num_workers > 0 else None)
        ds = bench_vlm.build_base_wds(wds)  # webdataset-tar alternative
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
            persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    from cosmos_framework.data.lance.vlm_dataset import LanceVLMShuffleScan

    ds = LanceVLMShuffleScan(uri, "llava", buffer_size=1000, storage_options=_so(region, uri))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None)  # lance not fork-safe


def build_vsft_loader(which, jsonl, uri, region, batch_size, num_workers, n_total, s3_bucket, s3_prefix):
    from cosmos_framework.data.lance import LanceVisionSFTDataset
    from cosmos_framework.data.vfm.local_datasets.sft_local_dataset import LocalSFTDataset

    if which == "base":
        if s3_bucket and s3_prefix:  # stock boto3 download-per-sample (fair S3 base)
            ds = _Boto3SFTBase(jsonl, s3_bucket, s3_prefix, **_VSFT_KW)
        else:
            ds = LocalSFTDataset(jsonl, **_VSFT_KW)
            ds.skip_tokenize = True
    else:
        ds = LanceVisionSFTDataset(uri, table="vision_sft", decode_device="cpu",
                                   storage_options=_so(region, uri), **_VSFT_KW)
        ds.skip_tokenize = True
    g = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=n_total, generator=g)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        collate_fn=bench_vision_sft._collate, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None)


def run_trio(which, paths, *, region, cache, batch_size, num_workers, rounds, warmup, vsft_n_total,
             vsft_s3_bucket, vsft_s3_prefix, vlm_hf_subset):
    print(f"\n========== {which.upper()}-TRIO (faithful) ==========", flush=True)
    a = build_action_loader(which, paths["action_root"], paths["action_uri"], region, cache, batch_size, num_workers)
    v = build_vlm_loader(which, paths["vlm_wds"], paths["vlm_uri"], region, batch_size, num_workers,
                         hf_subset=vlm_hf_subset if which == "base" else None)
    s = build_vsft_loader(which, paths["vsft_jsonl"], paths["vsft_uri"], region, batch_size, num_workers,
                          vsft_n_total, vsft_s3_bucket, vsft_s3_prefix)
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
    ap.add_argument("--action-root", required=True)
    ap.add_argument("--action-uri", required=True)
    ap.add_argument("--vlm-wds", required=True)
    ap.add_argument("--vlm-uri", required=True)
    ap.add_argument("--vsft-jsonl", required=True)
    ap.add_argument("--vsft-uri", required=True)
    ap.add_argument("--vsft-s3-bucket", default=None, help="if set, base vsft downloads videos via boto3 (stock S3 path)")
    ap.add_argument("--vsft-s3-prefix", default=None, help="key prefix under which <vision_path> lives")
    ap.add_argument("--vlm-hf-subset", default=None,
                    help="if set, base VLM streams this lmms-lab/LLaVA-OneVision-Data subset from HF Hub (cosmos default)")
    ap.add_argument("--region", default=None)
    ap.add_argument("--cache-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trios", nargs="+", default=["base", "lance"])
    args = ap.parse_args()

    paths = dict(action_root=args.action_root, action_uri=args.action_uri,
                 vlm_wds=args.vlm_wds, vlm_uri=args.vlm_uri,
                 vsft_jsonl=args.vsft_jsonl, vsft_uri=args.vsft_uri)
    vsft_n_total = (args.rounds + args.warmup + 8) * args.batch_size
    regime = "S3" if args.region else "LOCAL"
    vmode = "boto3-per-sample" if (args.vsft_s3_bucket and args.vsft_s3_prefix) else ("FUSE/local")
    print(f"FAITHFUL COMBINED RAW [{regime}] — action=EPISODE-SHUFFLE both sides; vsft-base={vmode}\n"
          f"batch={args.batch_size} workers={args.num_workers}/loader rounds={args.rounds} "
          f"LANCE_IO_THREADS={os.environ.get('LANCE_IO_THREADS','default')}", flush=True)

    results = {}
    for which in args.trios:
        results[which] = run_trio(which, paths, region=args.region, cache=args.cache_size,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  rounds=args.rounds, warmup=args.warmup, vsft_n_total=vsft_n_total,
                                  vsft_s3_bucket=args.vsft_s3_bucket, vsft_s3_prefix=args.vsft_s3_prefix,
                                  vlm_hf_subset=args.vlm_hf_subset)

    if "base" in results and "lance" in results:
        print("\n--- per-loader RAW samples/s ---")
        print(f"{'loader':<14}{'base':>12}{'lance':>12}{'speedup':>10}")
        for nm in ["action", "vlm", "vision-sft"]:
            b, l = results["base"][0].get(nm), results["lance"][0].get(nm)
            print(f"{nm:<14}{b:>12.1f}{l:>12.1f}{l/b:>9.2f}x")
        ba, la = results["base"][1], results["lance"][1]
        print(f"\ncombined (1:1:1)  base={ba:.1f}  lance={la:.1f}  speedup={la/ba:.2f}x")


if __name__ == "__main__":
    main()
    os._exit(0)
