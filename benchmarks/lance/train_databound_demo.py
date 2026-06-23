# SPDX-License-Identifier: OpenMDW-1.1
"""Data-bound regime demo (proxy for fast/many GPUs) reading from S3.

The heavy-model multi-GPU run was compute-bound (1.1% data-wait) so the loader was
hidden. Here we make the compute step TINY (a small pooled-linear head) so the GPU is
effectively "infinitely fast" — the loader becomes the bottleneck, exactly the regime
that fast/many GPUs (H100, 8x) approach. Reading from S3 (both loaders) makes the data
cost realistic. The per-epoch time then reflects the loader's real throughput.

NOTE: the tiny head is a PROXY for "GPU compute ~ 0", not the real Cosmos model. It
shows the upper bound of the training-time benefit when training is data-bound.

  torchrun --nproc-per-node=4 benchmarks/lance/train_databound_demo.py --loader base
  torchrun --nproc-per-node=4 benchmarks/lance/train_databound_demo.py --loader lance
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

S3_ROOT = "/home/ubuntu/work/s3mnt/cosmos/droid/base/success"          # base mp4 via s3fs
S3_LANCE = "s3://lancedb-datasets-dev-us-east-2-devrel/cosmos/droid/lance/droid_composed"
KW = dict(action_space="joint_pos", use_state=True, mode="policy", chunk_length=16)


def collate(items):
    return (torch.stack([s["video"] for s in items]),
            torch.stack([s["action"] for s in items]))


class TinyHead(nn.Module):
    """Pooled-linear head: compute ~ 0 so the loader is the bottleneck."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 8 * 8, 17 * 8)

    def forward(self, video):  # video: (B,3,17,270,320) uint8
        x = video.float().div_(255.0)
        x = torch.nn.functional.adaptive_avg_pool3d(x, (4, 8, 8)).flatten(1)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", choices=["base", "lance", "lance-episode"], required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    import math
    from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceDROIDComposedIterable
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    so = {"region": "us-east-2"}
    sampler = None
    max_steps = math.ceil(args.n / world / args.bs)  # samples/rank/epoch budget (caps the infinite episode stream)
    if args.loader == "lance-episode":
        composed = LanceDROIDComposedDataset(root=S3_ROOT, lance_uri=S3_LANCE,
                                             decode_device="cpu", storage_options=so, **KW)
        ds = LanceDROIDComposedIterable(composed, seed=0)
        ds.shard_rank = rank; ds.shard_world_size = world  # disjoint episode shards per rank
    else:
        if args.loader == "base":
            ds = DROIDLeRobotDataset(root=S3_ROOT, **KW)
        else:
            ds = LanceDROIDComposedDataset(root=S3_ROOT, lance_uri=S3_LANCE,
                                           decode_device="cpu", storage_options=so, **KW)
        ds = torch.utils.data.Subset(ds, list(range(args.n)))
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=args.workers,
                                         collate_fn=collate, persistent_workers=True, prefetch_factor=4,
                                         multiprocessing_context="spawn")

    model = DDP(TinyHead().to(dev), device_ids=[local])
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        torch.cuda.synchronize(); t0 = time.perf_counter(); t_data = 0.0; last = time.perf_counter(); n = 0; step = 0
        for video, action in loader:
            t_data += time.perf_counter() - last
            video = video.to(dev, non_blocking=True); action = action.to(dev, non_blocking=True)
            loss = ((model(video) - action.flatten(1)) ** 2).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            n += video.shape[0]; step += 1; last = time.perf_counter()
            if step >= max_steps:  # cap (sampler modes end naturally ~here; episode stream is infinite)
                break
        torch.cuda.synchronize()
        ep_t = time.perf_counter() - t0
        stats = torch.tensor([ep_t, t_data, n], device=dev); dist.all_reduce(stats)
        ept = stats[0].item() / world
        if rank == 0:
            tag = "WARMUP" if ep == 0 else "STEADY"
            print(f"[{args.loader}] epoch {ep} {tag}: {ept:6.1f}s/epoch | data-wait {100*stats[1].item()/world/ept:4.1f}% "
                  f"| {int(stats[2].item())} samples ({stats[2].item()/ept:6.1f} samp/s global)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
