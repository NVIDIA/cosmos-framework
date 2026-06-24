# SPDX-License-Identifier: OpenMDW-1.1
"""End-to-end TRAINING throughput with the COMBINED (3-modality) dataloader.

Unlike the dataloader-only benches, this runs a real GPU train step (transformer
forward+backward) fed by the real combined mixer over the three Cosmos sub-loaders
(action / VLM / vision-SFT), base-trio vs lance-trio, and reports training-step
throughput + the GPU data-wait fraction.

Why a sized transformer and not the exact Cosmos model: Cosmos's combined path
(`IterativeJointDataLoader` → omni Mixture-of-Transformers) packs every modality into
one token sequence and trains a transformer over it. The omni model is an 8B FSDP job;
running it would only re-confirm "compute-bound on this GPU". Instead we keep the DATA
path 100% real (the actual base/lance sub-loaders + ratio mixing) and make the per-step
COMPUTE a transformer over a fixed packed-token budget, sized by --layers/--dim/--seq.
Sweeping --layers traces the data-bound → compute-bound crossover: where the dataloader
gates training (Lance wins) vs where model compute hides it (Lance frees CPU, wall-clock equal).

  # realistic MIXED regime, optimal workers, sweep compute:
  for L in 2 8 24; do
    python benchmarks/lance/train_combined_e2e.py --trio base  --regime mixed --layers $L \
       --action-workers 18 --vlm-workers 4 --vsft-workers 18 --steps 80 --warmup 20
    python benchmarks/lance/train_combined_e2e.py --trio lance --regime mixed --layers $L \
       --action-workers 18 --vlm-workers 4 --vsft-workers 18 --steps 80 --warmup 20
  done
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bench_combined_faithful as C  # reuse the exact sub-loader builders + InfiniteLoader

_D = "/home/ubuntu/work/data"
_S = "s3://lancedb-datasets-dev-us-east-2-devrel/cosmos"
_FUSE = "/home/ubuntu/s3mnt/cosmos"
_BUCKET = "lancedb-datasets-dev-us-east-2-devrel"
_JSONL = f"{_D}/bridge_src/sft_dataset_bridge/train/video_dataset_file.jsonl"


def _paths(regime, trio):
    """(paths-dict, region, vsft_s3_bucket, vsft_s3_prefix, vlm_hf_subset) for a regime."""
    if regime == "local":
        return (dict(action_root=f"{_D}/droid327/success", action_uri=f"{_D}/lance/droid_composed327_plain",
                     vlm_wds=f"{_D}/wds/llava_figureqa/shard-{{00000..00019}}.tar", vlm_uri=f"{_D}/lance/llava_figureqa",
                     vsft_jsonl=_JSONL, vsft_uri=f"{_D}/lance/vision_sft_plain"),
                None, None, None, None)
    if regime == "s3":
        return (dict(action_root=f"{_FUSE}/droid327/base/success", action_uri=f"{_S}/droid327/lance/droid_composed327_plain",
                     vlm_wds=f"{_FUSE}/llava/wds/shard-{{00000..00019}}.tar", vlm_uri=f"{_S}/llava/lance/llava_figureqa",
                     vsft_jsonl=_JSONL, vsft_uri=f"{_S}/vision_sft/lance/vision_sft_plain"),
                "us-east-2", _BUCKET, "cosmos/vision_sft/base/sft_dataset_bridge/train", None)
    # mixed: action local, vsft S3, VLM HF-stream(base)/S3(lance)
    return (dict(action_root=f"{_D}/droid327/success", action_uri=f"{_D}/lance/droid_composed327_plain",
                 vlm_wds=f"{_D}/wds/llava_figureqa/shard-{{00000..00019}}.tar", vlm_uri=f"{_S}/llava/lance/llava_figureqa",
                 vsft_jsonl=_JSONL, vsft_uri=f"{_S}/vision_sft/lance/vision_sft_plain"),
            "us-east-2", _BUCKET, "cosmos/vision_sft/base/sft_dataset_bridge/train", "figureqa(cauldron,llava_format)")


class PackedTransformer(nn.Module):
    """A transformer over a packed token sequence — stand-in for the omni MoT per-step
    compute. seq = packed-token budget, dim/heads/layers set the FLOPs/step."""

    def __init__(self, dim, heads, layers, vocab=4096):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):
        return self.head(self.enc(self.emb(tokens)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trio", choices=["base", "lance"], required=True)
    ap.add_argument("--regime", choices=["local", "s3", "mixed"], default="mixed")
    ap.add_argument("--ratios", default="1,1,1", help="action,vlm,vsft mixing ratios")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--action-workers", type=int, default=18)
    ap.add_argument("--vlm-workers", type=int, default=4)
    ap.add_argument("--vsft-workers", type=int, default=18)
    ap.add_argument("--cache-size", type=int, default=16)
    # compute knobs (per-step transformer over the packed token budget)
    ap.add_argument("--seq", type=int, default=2048, help="packed-token budget per step")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8, help="sweep this for the data/compute crossover")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device("cuda")
    paths, region, vb, vp, vhf = _paths(args.regime, args.trio)
    ratios = [int(x) for x in args.ratios.split(",")]
    which = args.trio

    a = C.build_action_loader(which, paths["action_root"], paths["action_uri"], region, args.cache_size,
                              args.batch_size, args.action_workers)
    v = C.build_vlm_loader(which, paths["vlm_wds"], paths["vlm_uri"], region, args.batch_size, args.vlm_workers,
                           hf_subset=vhf if which == "base" else None)
    vsft_n = (args.steps + args.warmup + 8) * args.batch_size
    s = C.build_vsft_loader(which, paths["vsft_jsonl"], paths["vsft_uri"], region, args.batch_size,
                            args.vsft_workers, vsft_n, vb, vp)
    loaders = [C._InfiniteLoader(a, "action"), C._InfiniteLoader(v, "vlm"), C._InfiniteLoader(s, "vsft")]

    # ratio-weighted round-robin selection (mirrors IterativeJointDataLoader modality pick)
    sched = []
    for i, r in enumerate(ratios):
        sched += [i] * r

    model = PackedTransformer(args.dim, args.heads, args.layers).to(dev).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cpu").manual_seed(0)

    print(f"[{which}|{args.regime}] workers a/v/s={args.action_workers}/{args.vlm_workers}/{args.vsft_workers} "
          f"compute: dim={args.dim} layers={args.layers} seq={args.seq} batch={args.batch_size}", flush=True)

    seen = 0
    t_data = 0.0
    t0 = None
    last = None
    for step in range(args.steps + args.warmup):
        if step == args.warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter(); t_data = 0.0; seen = 0
        sel = sched[step % len(sched)]
        if last is not None:
            pass
        t_d0 = time.perf_counter()
        batch = loaders[sel].next_batch()          # REAL data: blocks here if loader can't keep up
        n = C._batch_count(batch, args.batch_size)
        if step >= args.warmup:
            t_data += time.perf_counter() - t_d0
            seen += n
        # real train step on a packed-token sequence (compute independent of modality)
        tokens = torch.randint(0, 4096, (args.batch_size, args.seq), generator=g).to(dev)
        out = model(tokens)
        loss = out.float().log_softmax(-1).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"    steps/s={args.steps / wall:6.2f}  samples/s={seen / wall:8.1f}  "
          f"data-wait={100 * t_data / wall:5.1f}%  ({wall:.1f}s for {args.steps} steps, {seen} samples)", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
