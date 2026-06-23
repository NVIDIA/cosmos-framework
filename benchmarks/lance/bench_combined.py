# SPDX-License-Identifier: OpenMDW-1.1
"""Combined 3-dataloader throughput benchmark: base-trio vs lance-trio.

Mimics real cosmos training that mixes three loaders concurrently:

  ACTION      — DROID action (video+action sample)            base / lance-composed
  VLM         — LLaVA figureqa image+convo                    wds-tar / lance-scan
  VISION-SFT  — bridge vision-SFT video clips                 base / lance

A round-robin MIXER drives the 3 loaders at EQUAL ratio (1:1:1): three torch
``DataLoader``s, each with its own worker pool (num_workers=4 -> 12 total,
persistent workers). One round = pull one batch from each of the 3 loaders;
re-create an iterator on ``StopIteration`` (treat as infinite for steady-state).
Aggregate samples/s = (sum of batch sizes pulled) / elapsed, warmup excluded.

RAW mode only (each loader does its data-access + decode — the dataloader's
actual storage job — WITHOUT the model-side Qwen image-processor, which is not
the dataloader's work and would dominate the VLM path).

Reuses the exact builders / paths / _KW from the per-loader bench scripts; it
does NOT reinvent the loaders.
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

import bench_action  # noqa: E402
import bench_vision_sft  # noqa: E402
import bench_vlm  # noqa: E402

# ── dataset paths (from the per-loader bench scripts) ───────────────────────
ACTION_ROOT = "/home/ubuntu/work/data/droid_cosmos/success"
ACTION_URI = "/home/ubuntu/work/data/lance/droid_composed"

VLM_WDS = "/home/ubuntu/work/data/wds/llava_figureqa/shard-{00000..00019}.tar"
VLM_URI = "/home/ubuntu/work/data/lance/llava_figureqa"

VSFT_JSONL = "/home/ubuntu/work/data/bridge_src/sft_dataset_bridge/train/video_dataset_file.jsonl"
VSFT_URI = "/home/ubuntu/work/data/lance/vision_sft"


# ── per-loader DataLoader builders (RAW mode) ───────────────────────────────
def build_action_loader(which, batch_size, num_workers):
    """which: 'base' or 'lance'. Full training sample (video+action); e2e==raw."""
    mode = "base" if which == "base" else "lance-composed"
    ds = bench_action._build(mode, ACTION_ROOT, ACTION_URI)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=bench_action._collate,  # tensor-stacking collate
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def build_vlm_loader(which, batch_size, num_workers):
    """which: 'base' (wds tar) or 'lance' (chunked-shuffle scan). RAW collate."""
    collate = bench_vlm.Collate("raw")  # raw -> ids only, no image-processor
    if which == "base":
        ds = bench_vlm.build_base_wds(VLM_WDS)
        # wds is an IterableDataset: no spawn ctx (matches bench_vlm base path)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    from cosmos_framework.data.lance.vlm_dataset import LanceVLMShuffleScan

    ds = LanceVLMShuffleScan(VLM_URI, "llava", buffer_size=1000)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_vsft_loader(which, batch_size, num_workers, n_total):
    """which: 'base' or 'lance'. RAW (tokenize=False). Tensor-stacking collate."""
    mode = "base" if which == "base" else "lance"
    ds = bench_vision_sft._build(mode, VSFT_JSONL, VSFT_URI, tokenize=False)
    g = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(
        ds, replacement=True, num_samples=n_total, generator=g
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=bench_vision_sft._collate,  # tensor-stacking collate
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


# ── helpers ─────────────────────────────────────────────────────────────────
def _batch_count(batch, batch_size):
    """Count samples in a pulled batch regardless of its dict/list/tensor shape."""
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


class InfiniteLoader:
    """Wrap a DataLoader so StopIteration just restarts the iterator."""

    def __init__(self, loader, name):
        self.loader = loader
        self.name = name
        self.it = iter(loader)

    def next_batch(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def standalone_sps(loader, *, batch_size, rounds, warmup):
    """Per-loader steady-state samples/s through its own DataLoader."""
    inf = InfiniteLoader(loader, "standalone")
    seen, t0 = 0, None
    for i in range(rounds + warmup):
        b = inf.next_batch()
        n = _batch_count(b, batch_size)
        if i == warmup:
            t0 = time.perf_counter()
        if i >= warmup:
            seen += n
    dt = time.perf_counter() - t0
    return seen / dt


def combined_sps(loaders, names, *, batch_size, rounds, warmup):
    """Round-robin mixer at 1:1:1. One round = one batch from EACH loader.
    Aggregate samples/s = (sum batch sizes pulled, post-warmup) / elapsed."""
    infs = [InfiniteLoader(ld, nm) for ld, nm in zip(loaders, names)]
    seen, t0 = 0, None
    per_loader_seen = {nm: 0 for nm in names}
    for r in range(rounds + warmup):
        if r == warmup:
            t0 = time.perf_counter()
        for inf in infs:
            b = inf.next_batch()
            n = _batch_count(b, batch_size)
            if r >= warmup:
                seen += n
                per_loader_seen[inf.name] += n
    dt = time.perf_counter() - t0
    return seen / dt, dt, per_loader_seen


# ── main ─────────────────────────────────────────────────────────────────────
def run_trio(which, *, batch_size, num_workers, rounds, warmup, vsft_n_total):
    print(f"\n========== building {which.upper()}-TRIO loaders ==========", flush=True)
    a = build_action_loader(which, batch_size, num_workers)
    v = build_vlm_loader(which, batch_size, num_workers)
    s = build_vsft_loader(which, batch_size, num_workers, vsft_n_total)
    loaders = [a, v, s]
    names = ["action", "vlm", "vision-sft"]

    # per-loader standalone (for reference)
    standalone = {}
    for ld, nm in zip(loaders, names):
        print(f"  [{which}] standalone {nm} ...", flush=True)
        standalone[nm] = standalone_sps(ld, batch_size=batch_size, rounds=rounds, warmup=warmup)
        print(f"    {nm:<12} {standalone[nm]:8.1f} samples/s", flush=True)

    # combined aggregate (mixer)
    print(f"  [{which}] combined mixer (1:1:1) ...", flush=True)
    agg, dt, per = combined_sps(loaders, names, batch_size=batch_size, rounds=rounds, warmup=warmup)
    print(f"    combined aggregate {agg:8.1f} samples/s  ({dt:.1f}s, {rounds} rounds)", flush=True)
    return standalone, agg, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4, help="per loader (x3 = total)")
    ap.add_argument("--rounds", type=int, default=40, help="measured rounds")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--trios", nargs="+", default=["base", "lance"])
    # modeled combined-e2e per-loader e2e samples/s (overridable). Defaults:
    #   action e2e == raw (decode-bound) -> use measured raw
    #   vlm e2e == image-processor-bound -> measured raw is irrelevant; e2e ~1x base
    #   vision-sft e2e from bench_vision_sft
    ap.add_argument("--vsft-e2e-base", type=float, default=None)
    ap.add_argument("--vsft-e2e-lance", type=float, default=None)
    ap.add_argument("--vlm-e2e-base", type=float, default=None)
    ap.add_argument("--vlm-e2e-lance", type=float, default=None)
    args = ap.parse_args()

    bs = args.batch_size
    nw = args.num_workers
    vsft_n_total = (args.rounds + args.warmup + 8) * bs

    print(
        f"COMBINED RAW (data+decode) throughput — 3-loader 1:1:1 mixer\n"
        f"batch_size={bs} num_workers={nw}/loader ({nw*3} total) "
        f"rounds={args.rounds} warmup={args.warmup}",
        flush=True,
    )

    results = {}
    for which in args.trios:
        results[which] = run_trio(
            which, batch_size=bs, num_workers=nw, rounds=args.rounds,
            warmup=args.warmup, vsft_n_total=vsft_n_total,
        )

    # ── report ──
    print("\n\n################## RESULTS ##################")
    print("\n--- per-loader STANDALONE samples/s (RAW) ---")
    print(f"{'loader':<14}{'base':>12}{'lance':>12}{'speedup':>10}")
    for nm in ["action", "vlm", "vision-sft"]:
        b = results.get("base", ({}, 0, {}))[0].get(nm)
        l = results.get("lance", ({}, 0, {}))[0].get(nm)
        spd = f"{l / b:.2f}x" if (b and l) else "-"
        bs_ = f"{b:.1f}" if b else "-"
        ls_ = f"{l:.1f}" if l else "-"
        print(f"{nm:<14}{bs_:>12}{ls_:>12}{spd:>10}")

    print("\n--- COMBINED RAW aggregate samples/s (1:1:1 mixer) ---")
    base_agg = results.get("base", (None, None, None))[1]
    lance_agg = results.get("lance", (None, None, None))[1]
    if base_agg:
        print(f"  base-trio   {base_agg:8.1f} samples/s")
    if lance_agg:
        print(f"  lance-trio  {lance_agg:8.1f} samples/s")
    if base_agg and lance_agg:
        print(f"  speedup     {lance_agg / base_agg:.2f}x  (lance-trio / base-trio)")

    # ── modeled combined e2e ──
    # The mixer feeds a single training step; combined e2e throughput at a fixed
    # 1:1:1 ratio is harmonic-mean-like: to produce N samples from each loader,
    # wall time = N*(1/r_action + 1/r_vlm + 1/r_vsft); aggregate sps for 3N
    # samples = 3N / wall = 3 / (1/r_a + 1/r_v + 1/r_s). Bottleneck = slowest.
    def _agg_model(r_a, r_v, r_s):
        return 3.0 / (1.0 / r_a + 1.0 / r_v + 1.0 / r_s)

    print("\n--- MODELED combined END-TO-END (data+decode+model-side) ---")
    print("  MODEL ASSUMPTION: fixed 1:1:1 ratio; combined sps = 3 / (1/r_action + 1/r_vlm + 1/r_vsft)")
    print("  (per-loader e2e inputs):")
    for trio in ["base", "lance"]:
        std = results.get(trio, ({}, 0, {}))[0]
        if not std:
            continue
        # action e2e == raw (decode-bound; video+action is the full sample)
        r_a = std.get("action")
        # vlm e2e: image-processor-bound -> raw access win does NOT surface;
        # if not provided, model e2e ~= base raw for BOTH trios (processor dominates)
        if trio == "base":
            r_v = args.vlm_e2e_base if args.vlm_e2e_base else std.get("vlm")
            r_s = args.vsft_e2e_base if args.vsft_e2e_base else None
        else:
            r_v = args.vlm_e2e_lance if args.vlm_e2e_lance else std.get("vlm")
            r_s = args.vsft_e2e_lance if args.vsft_e2e_lance else None
        if r_s is None:
            print(f"  [{trio}] vision-sft e2e not supplied -> using RAW vision-sft as proxy")
            r_s = std.get("vision-sft")
        if r_a and r_v and r_s:
            model_agg = _agg_model(r_a, r_v, r_s)
            print(
                f"  [{trio}] action={r_a:.1f} vlm={r_v:.1f} vsft={r_s:.1f} "
                f"-> modeled combined e2e {model_agg:.1f} samples/s"
            )


if __name__ == "__main__":
    main()
