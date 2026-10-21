# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NIXL and NCCL bandwidth check with no Ray in the path, for P producer/consumer pairs.

``WORLD_SIZE = 2P``: ranks ``0..P-1`` are producers, rank ``i+P`` is the consumer paired with
producer ``i``. Every consumer pulls its producer's CUDA buffer with a one-sided NIXL READ (the
operation Ray's RDT performs) and receives the same buffer over NCCL ``send``/``recv``; all pulls
start from one barrier, so with P consumers on one node the aggregate is the node-to-node figure.
torch.distributed over gloo is only the out-of-band channel for NIXL's metadata exchange and for
barriers. The pairs may span two GPUs of one node, two nodes of one NVL72 rack, or two racks; the
test infers which from the hostnames and applies the matching per-pair bandwidth floor. NIXL is timed twice per size: over plain
``cudaMalloc`` memory, whose legacy IPC handles are node-local so cross-node traffic falls back to
RDMA, and over fabric-handle VMM memory (what the weight-sync staging buffer uses), which cross-node
NVLink (MNNVL) can map.

Launch ``2P`` processes with ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT`` and one GPU
each (``CUDA_VISIBLE_DEVICES``); without that environment the test is skipped. Floors can be
overridden with ``NIXL_RAW_FLOOR_GBPS`` / ``NCCL_RAW_FLOOR_GBPS``; sizes with ``RAW_SIZES_MIB``.
"""

from __future__ import annotations

import os
import re
import statistics
import time
from dataclasses import dataclass

import pytest

_LAUNCHED = (
    os.environ.get("WORLD_SIZE", "0").isdigit()
    and int(os.environ.get("WORLD_SIZE", "0")) >= 2
    and int(os.environ.get("WORLD_SIZE", "0")) % 2 == 0
    and "RANK" in os.environ
    and "MASTER_ADDR" in os.environ
)
pytestmark = pytest.mark.skipif(not _LAUNCHED, reason="needs 2P launched ranks (RANK/WORLD_SIZE/MASTER_ADDR)")

WARMUP = 3
ITERS = 10
# Wall-clock floors (GB/s) at the largest size, per topology: NVLink between two GPUs of a node and
# NVL72 MNNVL between nodes of one rack for both transports; between racks NIXL stripes one-sided
# RDMA over all four NICs (UCX_MAX_RMA_RAILS) while NCCL point-to-point uses the GPU's nearest NIC.
NIXL_FLOORS_GBPS = {"cross_card": 200.0, "cross_node": 200.0, "cross_rack": 40.0}
NCCL_FLOORS_GBPS = {"cross_card": 200.0, "cross_node": 200.0, "cross_rack": 25.0}
# With several pairs in flight the NICs and the inter-rack fabric are shared, so the aggregate is
# gated at this fraction of PAIRS times the single-pair floor.
CONCURRENT_FLOOR_FRACTION = 0.6


@dataclass(frozen=True)
class Topology:
    mode: str
    hosts: tuple[str, str]


def _rack(host: str) -> str:
    """``nvl72D167-T15`` -> ``nvl72D167``; hosts without a tray suffix are their own rack."""
    match = re.match(r"^(.*?)-T\d+", host)
    return match.group(1) if match else host


def _topology(pairs: int) -> Topology:
    import torch.distributed as dist

    hosts: list = [None] * (2 * pairs)
    dist.all_gather_object(hosts, os.uname().nodename)
    producer, consumer = hosts[0], hosts[pairs]
    if producer == consumer:
        mode = "cross_card"
    elif _rack(producer) == _rack(consumer):
        mode = "cross_node"
    else:
        mode = "cross_rack"
    return Topology(mode, (producer, consumer))


def _median_seconds(times: list[float]) -> float:
    return statistics.median(times) if times else float("nan")


def _bench_nccl(group, rank: int, pairs: int, nbytes: int) -> float:
    """Each producer sends to its consumer, all pairs at once; timed on the receiver, median over ITERS."""
    import torch
    import torch.distributed as dist

    producer = rank < pairs
    partner = rank + pairs if producer else rank - pairs
    buf = torch.ones(nbytes, dtype=torch.uint8, device="cuda")  # [N]
    times: list[float] = []
    for i in range(WARMUP + ITERS):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if producer:
            dist.send(buf, dst=partner, group=group)
        else:
            dist.recv(buf, src=partner, group=group)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.perf_counter() - started)
    return _median_seconds(times)


class _NixlPeer:
    """One registered CUDA buffer per rank plus the peer's descriptors, exchanged once over gloo."""

    def __init__(self, rank: int, pairs: int, nbytes: int, *, fabric: bool) -> None:
        import torch
        import torch.distributed as dist
        from nixl._api import nixl_agent, nixl_agent_config

        from cosmos_framework.posttrain.workers.nixl import _fabric_buffer

        self.rank = rank
        self.producer = rank < pairs
        partner = rank + pairs if self.producer else rank - pairs
        self.agent = nixl_agent(f"rank{rank}", nixl_agent_config(backends=["UCX"]))
        self._keepalive = None
        if fabric:
            self.buf, self._keepalive = _fabric_buffer(nbytes)  # [N] (padded allocation, exact view)
        else:
            self.buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")  # [N]
        self.buf.fill_(0xA5 if self.producer else 0x00)
        torch.cuda.synchronize()
        self.reg_descs = self.agent.register_memory(
            [(int(self.buf.data_ptr()), int(self.buf.numel()), torch.cuda.current_device(), "")], mem_type="cuda"
        )
        self.local_descs = self.agent.get_xfer_descs(
            [(int(self.buf.data_ptr()), int(self.buf.numel()), torch.cuda.current_device())], mem_type="cuda"
        )
        payload = [self.agent.get_agent_metadata(), self.agent.get_serialized_descs(self.local_descs), self.agent.name]
        gathered: list = [None] * (2 * pairs)
        dist.all_gather_object(gathered, payload)
        peer_meta, peer_descs, peer_name = gathered[partner]
        self.agent.add_remote_agent(peer_meta)
        self.remote_descs = self.agent.deserialize_descs(peer_descs)
        self.peer_name = peer_name.decode() if isinstance(peer_name, bytes) else peer_name

    def read_once(self) -> None:
        """One-sided READ of the peer's buffer into ours, polled to completion."""
        xfer = self.agent.initialize_xfer("READ", self.local_descs, self.remote_descs, self.peer_name, b"RAW")
        state = self.agent.transfer(xfer)
        while state != "DONE":
            if state == "ERR":
                raise RuntimeError("NIXL transfer entered ERR state")
            state = self.agent.check_xfer_state(xfer)
        self.agent.release_xfer_handle(xfer)

    def close(self) -> None:
        from cosmos_framework.posttrain.workers.nixl import _release_fabric

        self.agent.deregister_memory(self.reg_descs)
        del self.buf
        if self._keepalive is not None:
            _release_fabric(self._keepalive)


def _bench_nixl(peer: _NixlPeer, nbytes: int) -> float:
    """Every consumer pulls its producer's buffer from one barrier; timed on the initiator, median
    over ITERS; values verified."""
    import torch
    import torch.distributed as dist

    times: list[float] = []
    for i in range(WARMUP + ITERS):
        dist.barrier()
        if not peer.producer:
            torch.cuda.synchronize()
            started = time.perf_counter()
            peer.read_once()
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append(time.perf_counter() - started)
    dist.barrier()
    if not peer.producer and not bool(torch.all(peer.buf == 0xA5).item()):
        raise RuntimeError("NIXL READ did not deliver the producer's bytes")
    return _median_seconds(times)


@pytest.mark.L1
@pytest.mark.GPU
def test_nixl_and_nccl_reach_the_topology_floor() -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    pairs = world // 2
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.zeros(1, device="cuda")
    dist.init_process_group(backend="gloo")
    try:
        nccl_group = dist.new_group(backend="nccl")
        topo = _topology(pairs)
        sizes_mib = [int(x) for x in os.environ.get("RAW_SIZES_MIB", "64,256,1024").split(",")]
        # Per size: list over consumers of (nccl_s, nixl_plain_s, nixl_fabric_s).
        rows: list[tuple[int, list[tuple[float, float, float]]]] = []
        for mib in sizes_mib:
            nbytes = mib * 2**20
            nccl_s = _bench_nccl(nccl_group, rank, pairs, nbytes)
            nixl_s: dict[bool, float] = {}
            for fabric in (False, True):
                peer = _NixlPeer(rank, pairs, nbytes, fabric=fabric)
                try:
                    nixl_s[fabric] = _bench_nixl(peer, nbytes)
                finally:
                    peer.close()
                    del peer
            # Consumers hold the timings; publish them so every rank asserts on the same numbers.
            timings: list = [None] * world
            dist.all_gather_object(timings, (nccl_s, nixl_s[False], nixl_s[True]))
            rows.append((mib, timings[pairs:]))

        def _gbps(nbytes: int, seconds: float) -> float:
            return nbytes / seconds / 1e9

        if rank == 0:
            print(
                f"\n{topo.mode}: {pairs} pair(s) {topo.hosts[0]} -> {topo.hosts[1]}  "
                f"UCX_TLS={os.environ.get('UCX_TLS', '(unset)')}"
            )
            print(
                f"{'size':>8} {'nccl ms':>9} {'nccl GB/s':>10} {'nixl ms':>9} {'nixl GB/s':>10} {'fabric ms':>10} {'fabric GB/s':>12}"
            )
            for mib, per_consumer in rows:
                nbytes = mib * 2**20
                for c, (nccl_s, plain_s, fabric_s) in enumerate(per_consumer):
                    label = f"{mib:>5} MiB" if pairs == 1 else f"{mib:>5} MiB p{c}"
                    print(
                        f"{label:>8} {nccl_s * 1e3:>9.2f} {_gbps(nbytes, nccl_s):>10.1f} "
                        f"{plain_s * 1e3:>9.2f} {_gbps(nbytes, plain_s):>10.1f} "
                        f"{fabric_s * 1e3:>10.2f} {_gbps(nbytes, fabric_s):>12.1f}",
                        flush=True,
                    )
                if pairs > 1:
                    # Aggregate: every consumer started from the same barrier, so the slowest one bounds
                    # the time in which all P buffers landed.
                    slowest = [max(t[k] for t in per_consumer) for k in range(3)]
                    print(
                        f"{mib:>5} MiB all {slowest[0] * 1e3:>9.2f} {_gbps(pairs * nbytes, slowest[0]):>10.1f} "
                        f"{slowest[1] * 1e3:>9.2f} {_gbps(pairs * nbytes, slowest[1]):>10.1f} "
                        f"{slowest[2] * 1e3:>10.2f} {_gbps(pairs * nbytes, slowest[2]):>12.1f}   (aggregate)",
                        flush=True,
                    )
        mib, per_consumer = rows[-1]
        nbytes = mib * 2**20
        nixl_floor = float(os.environ.get("NIXL_RAW_FLOOR_GBPS", NIXL_FLOORS_GBPS[topo.mode]))
        nccl_floor = float(os.environ.get("NCCL_RAW_FLOOR_GBPS", NCCL_FLOORS_GBPS[topo.mode]))
        if pairs == 1:
            nccl_s, _plain_s, fabric_s = per_consumer[0]
            assert _gbps(nbytes, fabric_s) >= nixl_floor, (
                f"NIXL {topo.mode}: {_gbps(nbytes, fabric_s):.1f} GB/s < {nixl_floor}"
            )
            assert _gbps(nbytes, nccl_s) >= nccl_floor, (
                f"NCCL {topo.mode}: {_gbps(nbytes, nccl_s):.1f} GB/s < {nccl_floor}"
            )
        else:
            nixl_agg = _gbps(pairs * nbytes, max(t[2] for t in per_consumer))
            nccl_agg = _gbps(pairs * nbytes, max(t[0] for t in per_consumer))
            scale = pairs * CONCURRENT_FLOOR_FRACTION
            assert nixl_agg >= nixl_floor * scale, (
                f"NIXL {topo.mode} x{pairs}: {nixl_agg:.1f} GB/s < {nixl_floor * scale:.1f}"
            )
            assert nccl_agg >= nccl_floor * scale, (
                f"NCCL {topo.mode} x{pairs}: {nccl_agg:.1f} GB/s < {nccl_floor * scale:.1f}"
            )
    finally:
        dist.destroy_process_group()
