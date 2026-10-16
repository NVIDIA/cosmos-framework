#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal NCCL all-reduce probe — runs IN the container as a cheap 2-node/2-GPU
job to prove the cluster's NCCL rendezvous works BEFORE a real multi-node run
burns GPU-hours hanging on the first collective.

It uses torchrun's WORLD_SIZE, RANK, LOCAL_WORLD_SIZE, and LOCAL_RANK, does
one all-reduce, and prints NCCL_PROBE_OK. For offline template checks it can
also derive these values from NNODES, NPROC_PER_NODE, and NODE_RANK.
If NCCL is misconfigured (e.g. the CS-OCI-ORD
intra-node P2P hang) it HANGS on all_reduce — the orchestrating skill wraps this
with a timeout and, on timeout, sets the cluster's NCCL knob (NCCL_P2P_DISABLE=1,
NCCL_SOCKET_IFNAME, ...) and re-probes, caching the working env per cluster.

`--dry-run` prints the computed rendezvous config WITHOUT importing torch or
touching a GPU — the node-count->global-rank math (the easy thing to get wrong
given NNODES is the node count, not the global rank count) is testable
offline. torch is imported lazily so this file loads on a CPU host.
"""

from __future__ import annotations

import argparse
import json
import os


def rendezvous_config(env: dict | None = None) -> dict:
    """Read standard torchrun ranks, with launcher-template fallbacks."""
    e = os.environ if env is None else env
    gpus_per_node = int(e.get("LOCAL_WORLD_SIZE", e.get("NPROC_PER_NODE", "1")))
    local_rank = int(e.get("LOCAL_RANK", "0"))
    world_size = int(e.get("WORLD_SIZE", int(e.get("NNODES", "1")) * gpus_per_node))
    rank = int(e.get("RANK", int(e.get("NODE_RANK", "0")) * gpus_per_node + local_rank))
    if gpus_per_node < 1 or world_size < 1 or world_size % gpus_per_node:
        raise ValueError("world size must be a positive multiple of local world size")
    if not 0 <= local_rank < gpus_per_node or not 0 <= rank < world_size:
        raise ValueError("global and local ranks must be within their world sizes")
    return {
        "global_world_size": world_size,
        "global_rank": rank,
        "local_rank": local_rank,
        "node_count": world_size // gpus_per_node,
        "gpus_per_node": gpus_per_node,
        "master_addr": e.get("MASTER_ADDR", ""),
        "master_port": e.get("MASTER_PORT", "29500"),
    }


def _run(cfg: dict) -> int:
    import torch  # lazy — only when actually probing on a GPU node
    import torch.distributed as dist

    torch.cuda.set_device(cfg["local_rank"])
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=cfg["global_world_size"],
        rank=cfg["global_rank"],
    )
    t = torch.ones(1, device=f"cuda:{cfg['local_rank']}")
    dist.all_reduce(t)  # HANGS here if NCCL rendezvous is broken
    ok = int(t.item()) == cfg["global_world_size"]
    dist.barrier()
    if cfg["global_rank"] == 0:
        print("NCCL_PROBE_OK" if ok else f"NCCL_PROBE_BAD sum={int(t.item())} expected={cfg['global_world_size']}")
    dist.destroy_process_group()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dry-run", action="store_true", help="print the computed rendezvous config and exit (no torch/GPU)"
    )
    args = p.parse_args(argv)
    cfg = rendezvous_config()
    if args.dry_run:
        print(json.dumps(cfg))
        return 0
    return _run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
