# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Distributed rank resolution that also works inside spawned DataLoader workers."""

import os

import torch.distributed as dist


def get_worker_rank_world_size() -> tuple[int, int]:
    """Resolve distributed coordinates inside main or spawned worker processes.

    Spawned DataLoader workers inherit ``RANK`` and ``WORLD_SIZE`` but do not
    inherit the initialized Torch process group. Falling back to the environment
    preserves disjoint Lance row-index slots across training ranks.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        try:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError("RANK and WORLD_SIZE must be valid integers in spawned Lance workers.") from error

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be at least 1, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE, got rank={rank}, world_size={world_size}.")
    return rank, world_size
