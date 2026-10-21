# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Branch B — separate-then-MERGE ablation ("which combination method"). Task-arithmetic (Ilharco 2023):
# W_merged = W_base + λ · Σ_t (W_expert_t − W_base). Merge single-task experts (ctrl_ifeval@400 +
# ctrl_chartqa@400) into ONE model in weight space; sweep λ. Compare (eval) vs the EXPOSURE-MATCHED joint
# (combined-long 800-step) — NOT the 400-step joint (experts each got 400 task-steps). garla 07-07.
# Streams shard-by-shard via safetensors to bound RAM. Copies tokenizer/config from base.

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RL = "/path/to/shared-storage"
BASE = f"{RL}/checkpoints/cosmos3-nano-reasoner"


def load_index(directory: str) -> dict[str, str]:
    with open(f"{directory}/model.safetensors.index.json", encoding="utf-8") as handle:
        index: dict[str, Any] = json.load(handle)
    weight_map: dict[str, str] = index["weight_map"]
    return weight_map  # key -> shard filename


def all_keys(directory: str) -> set[str]:
    return set(load_index(directory).keys())


def get_tensor(
    directory: str,
    weight_map: dict[str, str],
    key: str,
    cache: dict[str, Any],
) -> torch.Tensor:  # returns [*]
    shard = weight_map[key]
    if shard not in cache:
        cache[shard] = safe_open(f"{directory}/{shard}", framework="pt", device="cpu")
    tensor: torch.Tensor = cache[shard].get_tensor(key)  # [*]
    return tensor  # [*]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", required=True, help="comma list of expert ckpt dirs")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--lams",
        default=None,
        help="comma per-expert weights (overrides --lam); enables per-task λ ablation",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    experts = args.experts.split(",")
    lams = [float(value) for value in args.lams.split(",")] if args.lams else [args.lam] * len(experts)
    assert len(lams) == len(experts), f"lams {len(lams)} != experts {len(experts)}"
    base_map = load_index(BASE)
    expert_maps = [load_index(expert) for expert in experts]

    # sanity: identical key sets
    base_keys = set(base_map)
    for expert, expert_map in zip(experts, expert_maps):
        assert set(expert_map) == base_keys, f"key mismatch base vs {expert}: {len(set(expert_map) ^ base_keys)} differ"

    os.makedirs(args.out, exist_ok=True)

    # merge key-by-key, regroup into original base shards
    shard_of = base_map  # write into the SAME shard layout as base
    out_shards: dict[str, dict[str, torch.Tensor]] = {}  # shard filename -> {key: tensor}
    base_cache: dict[str, Any] = {}
    expert_caches: list[dict[str, Any]] = [{} for _ in experts]
    num_merged = 0
    for key in base_keys:
        base_weight = get_tensor(BASE, base_map, key, base_cache).float()  # [*]
        delta = torch.zeros_like(base_weight)  # [*]
        for expert, expert_map, expert_cache, task_lambda in zip(experts, expert_maps, expert_caches, lams):
            expert_weight = get_tensor(expert, expert_map, key, expert_cache).float()  # [*]
            delta += task_lambda * (expert_weight - base_weight)  # [*]
        merged = (base_weight + delta).to(torch.bfloat16)  # [*]
        out_shards.setdefault(shard_of[key], {})[key] = merged
        num_merged += 1
        if num_merged % 200 == 0:
            print(f"  merged {num_merged}/{len(base_keys)} tensors", flush=True)

    # write shards + index
    total_size = 0
    weight_map: dict[str, str] = {}
    for shard, tensors in out_shards.items():
        save_file(tensors, f"{args.out}/{shard}", metadata={"format": "pt"})
        for key, tensor in tensors.items():  # tensor: [*]
            weight_map[key] = shard
            total_size += tensor.numel() * tensor.element_size()
    with open(f"{args.out}/model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, handle, indent=2)

    # copy config/tokenizer from base (arch identical)
    for filename in os.listdir(BASE):
        if filename.endswith(".json") and filename != "model.safetensors.index.json":
            shutil.copy(f"{BASE}/{filename}", f"{args.out}/{filename}")
    print(
        f"[merge] λ={args.lam} experts={len(experts)} -> {args.out} ({num_merged} tensors, {total_size / 1e9:.1f}GB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
