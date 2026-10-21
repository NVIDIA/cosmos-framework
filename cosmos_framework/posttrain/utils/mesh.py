# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ray-grid mesh sizing from the multinode launcher's environment contract.

`utils/launch_multinode.sh` exports `NNODES` and `GPUS_PER_NODE` before starting the Ray cluster;
this module is the reader half of that contract, so an entry can size `(replicas, ranks)` without
duplicating the env parsing. Imports of torch are function-local so this module stays importable on
login nodes.
"""

from __future__ import annotations


def detect_mesh() -> tuple[int, int]:
    """Return the launcher's ``(nodes, GPUs-per-node)`` mesh.

    Explicit env wins: `launch_multinode.sh` sets `NNODES`/`GPUS_PER_NODE`, and it is the reliable
    source because an inner ``srun --nodes=1`` would otherwise reset the SLURM node counts. Falls
    back to the locally visible GPU count on a single node so a standalone run still works.

    Returns:
        ``(nodes, gpus_per_node)``, both at least 1.

    Raises:
        ValueError: Either dimension resolved to a non-positive value.
    """
    import os

    import torch

    gpus_per_node = int(os.environ.get("GPUS_PER_NODE") or torch.cuda.device_count() or 1)
    nodes = int(os.environ.get("NNODES") or 1)
    if nodes < 1 or gpus_per_node < 1:
        raise ValueError(f"invalid mesh: nodes={nodes}, gpus_per_node={gpus_per_node} (both must be >= 1)")
    return nodes, gpus_per_node
