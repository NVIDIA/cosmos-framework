# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Host CPU thread budget for the CPU-bound media work an inference rank does itself.

``torchrun`` starts every worker with ``OMP_NUM_THREADS=1`` when the variable is unset, so a
rank's CPU tensor ops (antialiased resize, dtype conversion) run on one thread no matter how
many cores the node has. For conditioning video that is most of the cost: on a GB200 node a
297-frame 1924x1084 control clip took ~6 s per rank to decode and resize at one thread each
for FFmpeg and torch, and ~1 s with the rank's share of the cores, with bitwise-identical
pixels (H.264 decoding is exact under threading, and the resize computes each output element
independently of how the work is split).
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pynvml
import torch


def host_threads_per_local_rank() -> int:
    """This rank's share of the CPUs it may run on, assuming the node's ranks split its CPUs evenly.

    A rank may run on its affinity mask: the whole node under plain ``torchrun``, or its GPU's
    local CPUs once ``cosmos_framework.utils.distributed.init`` has pinned it (on a GB200 node, 70 of
    140 CPUs, shared with the neighbouring GPU's rank). The ranks sharing the mask are estimated
    as the node's ranks spread evenly over the node's CPUs, so both cases give 140 / 4 = 35.
    """
    allowed_cpus = len(os.sched_getaffinity(0))
    node_cpus = max(os.cpu_count() or allowed_cpus, allowed_cpus)
    ranks_sharing_mask = max(1, math.ceil(_ranks_per_node() * allowed_cpus / node_cpus))
    return max(1, allowed_cpus // ranks_sharing_mask)


@functools.cache
def _ranks_per_node() -> int:
    """How many ranks this node runs: ``LOCAL_WORLD_SIZE``, or one per GPU if that is more.

    Ray serving runs one single-GPU worker per GPU and reports ``LOCAL_WORLD_SIZE=1`` to each, so
    the variable alone would let every worker claim the whole node. NVML counts the node's GPUs
    regardless of ``CUDA_VISIBLE_DEVICES``.
    """
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    try:
        pynvml.nvmlInit()
        node_gpus = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        node_gpus = 1
    return max(local_world_size, node_gpus)


@contextmanager
def torch_host_threads(num_threads: int) -> Iterator[None]:
    """Run torch's CPU intra-op work on ``num_threads`` threads, restoring the setting after."""
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
