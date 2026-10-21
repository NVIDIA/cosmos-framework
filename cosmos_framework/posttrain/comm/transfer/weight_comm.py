# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""WeightComm — the backend-agnostic weight-plane transport (design §2.2).

A WeightComm is a NCCL/gloo group spanning policy+rollout ranks that is INDEPENDENT of each
actor's default process group (already the FSDP or vLLM-TP group). We build a standalone
ProcessGroup from a fresh TCPStore and drive it with the WORK-based c10d API
(pg.broadcast([t], opts).wait()) — which, unlike dist.broadcast(group=pg), does not rely on the
global `_world` registry that only init_process_group/new_group populate. This is engine-agnostic
(no vLLM import); NcclMover holds one. A future PyncclComm implements the same surface (§2.2b).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist


@runtime_checkable
class WeightComm(Protocol):
    rank: int
    world_size: int

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        ...

    def send(self, tensor: torch.Tensor, dst: int) -> None:  # tensor: [*wire]
        ...

    def recv(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        ...

    def close(self) -> None: ...


class TorchDistComm:
    """Standalone c10d ProcessGroup over a fresh TCPStore. `backend='auto'` picks nccl on CUDA,
    gloo on CPU. Ranks are group-local (0..world-1); the store's rank-0 is the master.

    Constructor forms are torch-2.8-specific: ProcessGroupGloo(store, rank, size, timeout) — the
    `.Options()` path is broken on 2.8 (Options requires a backend string); ProcessGroupNCCL takes
    an Options() carrying the timeout. Collectives use the work-based API so this stays independent
    of the default-group registry. ``submit_broadcast`` exposes asynchronous completion while the
    original ``broadcast`` method remains the synchronous weight-mover compatibility surface."""

    def __init__(
        self, store_host: str, store_port: int, rank: int, world_size: int, backend: str = "auto", timeout_s: int = 300
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        self._backend = backend
        # PrefixStore isolates this plane's keys from any other TCPStore on the same host:port.
        base = dist.TCPStore(
            store_host, int(store_port), world_size, is_master=(rank == 0), timeout=timedelta(seconds=timeout_s)
        )
        self._store = dist.PrefixStore("weight_plane", base)
        to = timedelta(seconds=timeout_s)
        if backend == "nccl":
            from torch.distributed import ProcessGroupNCCL

            torch.cuda.set_device(torch.cuda.current_device())
            opts = ProcessGroupNCCL.Options()
            opts._timeout = to
            self._pg = ProcessGroupNCCL(self._store, rank, world_size, opts)
            self._dev = torch.device("cuda", torch.cuda.current_device())
        elif backend == "gloo":
            from torch.distributed import ProcessGroupGloo

            self._pg = ProcessGroupGloo(self._store, rank, world_size, to)
            self._dev = torch.device("cpu")
        else:
            raise ValueError(f"unsupported weight-plane backend {backend!r} (want nccl|gloo)")

    def submit_broadcast(self, tensor: torch.Tensor, src: int) -> dist.Work:  # tensor: [*wire]
        from torch.distributed import BroadcastOptions

        opts = BroadcastOptions()
        opts.rootRank = int(src)
        opts.rootTensor = 0
        return self._pg.broadcast([tensor], opts)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        self.submit_broadcast(tensor, src).wait()

    def all_reduce_max(self, tensor: torch.Tensor) -> None:  # tensor: [*control]
        """Reduce ``tensor`` in place so every rank observes the group-wide maximum."""
        from torch.distributed import AllreduceOptions

        opts = AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MAX
        self._pg.allreduce([tensor], opts).wait()

    def send(self, tensor: torch.Tensor, dst: int) -> None:  # tensor: [*wire]
        self._pg.send([tensor], int(dst), 0).wait()

    def recv(self, tensor: torch.Tensor, src: int) -> None:  # tensor: [*wire]
        self._pg.recv([tensor], int(src), 0).wait()

    def close(self) -> None:
        # Best-effort: drop references so the underlying comm is destroyed on GC. torch has no
        # public per-PG destroy for a standalone group across versions; dropping is the portable path.
        self._pg = None
        self._store = None


def build_weight_comm(host: str, port: int, rank: int, world: int, backend: str = "auto") -> TorchDistComm:
    return TorchDistComm(host, port, rank, world, backend=backend)
