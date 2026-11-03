# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NIXL weight mover end-to-end, through a real Ray actor on GPU.

The producer MUST be an actor: `ray.put(_tensor_transport="nixl")` reaches
`ray.get_runtime_context().current_actor` (rdt_manager.py:949), which raises
"This method is only available in an actor" on the driver. So these exercise the mover
inside `Peer`, not from the test process — the same shape production uses, where `stage`
runs inside `_Host`.

Skipped wholesale where ray/nixl are absent, which includes the training container today.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from cosmos_framework.posttrain.workers.nixl import nixl_available

_OK, _WHY = nixl_available()
pytestmark = pytest.mark.skipif(not _OK, reason=f"NIXL stack unavailable: {_WHY}")

if _OK:  # import only when the stack exists, so collection succeeds without ray
    import ray
    import torch

    @ray.remote(num_gpus=1, num_cpus=4, enable_tensor_transport=True)
    class _Peer:
        def warm(self):
            from cosmos_framework.posttrain.workers.nixl import NixlTransport

            NixlTransport.get().warmup()
            return True

        def _state(self, seed):
            g = torch.Generator(device="cuda").manual_seed(seed)
            return {
                "a.weight": torch.randn(64, 128, dtype=torch.bfloat16, device="cuda", generator=g),
                # fp32 alongside bf16 on purpose: a bucket is not uniform-dtype, which is why
                # the flat layout uses byte offsets (comm/transfer/flat.py).
                "a.bias": torch.randn(128, dtype=torch.float32, device="cuda", generator=g),
            }

        def round_trip(self):
            from cosmos_framework.posttrain.workers.nixl import NixlWeightMover

            m = NixlWeightMover()
            src = self._state(1)
            got = m.fetch(m.stage(src))
            assert set(got) == set(src), f"{set(got)} != {set(src)}"
            for k, v in src.items():
                assert got[k].shape == v.shape and got[k].dtype == v.dtype, k
                assert torch.equal(got[k], v), f"{k} differs after round trip"
            return True

        def two_syncs(self):
            """Two consecutive syncs through one reused staging buffer deliver the SECOND
            sync's weights, not the first's.

            Scope, established by experiment rather than assumed: deleting `_wait_free()`
            leaves this test GREEN. In a single-actor sequential run the previous
            `MoveHandle` is a temporary already dropped before the next `stage`, so the
            write-after-read window never opens. This therefore demonstrates correct reuse;
            it does NOT falsify the §4 invariant.

            What actually guards that invariant is
            `test_workers_nixl.py::test_stage_waits_before_reassigning_views`, which is
            structural and does fail when `_wait_free()` is removed. Keep both: this one
            catches a broken round trip, that one catches a removed barrier."""
            from cosmos_framework.posttrain.workers.nixl import NixlWeightMover

            m = NixlWeightMover()
            first = self._state(1)
            got1 = m.fetch(m.stage(first))
            assert torch.equal(got1["a.weight"], first["a.weight"])

            second = self._state(2)
            assert not torch.equal(second["a.weight"], first["a.weight"]), "seeds must differ"
            got2 = m.fetch(m.stage(second))
            assert torch.equal(got2["a.weight"], second["a.weight"]), "stale weights from sync 1"
            assert torch.equal(got2["a.bias"], second["a.bias"])
            return True

        def one_descriptor_per_bucket(self):
            """Publishing is per-descriptor-priced, so a bucket must travel as ONE tensor
            however many parameters it holds (design §3.2)."""
            from cosmos_framework.posttrain.comm.transfer.flat import flat_plan
            from cosmos_framework.posttrain.workers.nixl import NixlWeightMover

            src = self._state(3)
            handle = NixlWeightMover().stage(src)
            n_params = len(flat_plan(src))
            published = ray.get(handle.payload)
            assert n_params > 1, "fixture must hold several parameters to be meaningful"
            assert len(published) == 1, f"published {len(published)} tensors, expected 1 flat"
            return n_params

    @ray.remote(num_gpus=1, num_cpus=2, enable_tensor_transport=True)
    class _ReallocationPeer:
        def run(self) -> bool:
            import gc

            from cosmos_framework.posttrain.workers.nixl import NixlTransport, NixlWeightMover

            transport = NixlTransport.get()
            transport.warmup()
            small_bytes = 8 * 1024 * 1024
            large_bytes = 16 * 1024 * 1024
            transport.reserve(small_bytes)
            first = NixlWeightMover()
            small = torch.full((small_bytes,), 3, dtype=torch.uint8, device="cuda")  # [N_small]
            first_handle = first.stage({"probe": small})
            first_got = first.fetch(first_handle)["probe"]  # [N_small]
            assert bool(torch.all(first_got == 3).item())
            first_handle = None
            gc.collect()
            first.recycle_staging_buffer()
            assert transport._spare_buffer is not None

            # This must deregister and unmap the insufficient spare before a larger allocation.
            transport.reserve(large_bytes)
            assert transport._spare_buffer is None
            second = NixlWeightMover()
            large = torch.full((large_bytes,), 7, dtype=torch.uint8, device="cuda")  # [N_large]
            second_handle = second.stage({"probe": large})
            second_got = second.fetch(second_handle)["probe"]  # [N_large]
            assert bool(torch.all(second_got == 7).item())
            return True


@pytest.fixture(scope="module")
def peer() -> Any:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    p = _Peer.remote()
    ray.get(p.warm.remote())
    try:
        yield p
    finally:
        # Qualification runs the 8-GPU routed case immediately after this module. Explicitly release
        # the fixture's GPU rather than relying on asynchronous actor-handle garbage collection.
        ray.kill(p)


@pytest.mark.L1
@pytest.mark.GPU
def test_round_trip_preserves_every_parameter(peer):
    assert ray.get(peer.round_trip.remote()) is True


@pytest.mark.L1
@pytest.mark.GPU
def test_second_sync_delivers_new_weights_not_stale_ones(peer):
    assert ray.get(peer.two_syncs.remote()) is True


@pytest.mark.L1
@pytest.mark.GPU
def test_bucket_travels_as_a_single_descriptor(peer):
    assert ray.get(peer.one_descriptor_per_bucket.remote()) > 1


@pytest.mark.L1
@pytest.mark.GPU
def test_probe_buffer_can_be_deregistered_and_reallocated() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    if ray.available_resources().get("GPU", 0) < 1:
        pytest.skip("fabric-buffer reallocation gate needs one GPU")
    actor = _ReallocationPeer.remote()
    try:
        assert ray.get(actor.run.remote()) is True
    finally:
        ray.kill(actor)


@pytest.mark.L1
@pytest.mark.GPU
def test_real_actor_to_actor_value_probe() -> None:
    from cosmos_framework.posttrain.control.placement.colocate import _run_nixl_probe
    from cosmos_framework.posttrain.workers.ray import _Host

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    if ray.available_resources().get("GPU", 0) < 2:
        pytest.skip("real actor-to-actor NIXL probe needs two GPUs")
    producer = _Host.options(num_gpus=1, enable_tensor_transport=True).remote()
    consumer = _Host.options(num_gpus=1, enable_tensor_transport=True).remote()
    try:
        report = _run_nixl_probe(producer, consumer, 256 * 1024 * 1024, 20.0)
        assert report["bytes"] == 256 * 1024 * 1024
        assert report["gbps"] >= 20.0
        print(f"NIXL default probe: {report['gbps']:.2f} GB/s, {report['bytes']} bytes")
    finally:
        ray.kill(producer)
        ray.kill(consumer)


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.skipif(
    os.environ.get("POSTTRAIN_NIXL_CROSS_NODE") != "1",
    reason="set POSTTRAIN_NIXL_CROSS_NODE=1 inside a two-node Ray allocation",
)
def test_forced_cross_node_actor_to_actor_probe() -> None:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from cosmos_framework.posttrain.control.placement.colocate import _run_nixl_probe
    from cosmos_framework.posttrain.workers.ray import _Host

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    nodes = [node for node in ray.nodes() if node.get("Alive") and node["Resources"].get("GPU", 0)]
    if len(nodes) < 2:
        pytest.fail(f"forced cross-node NIXL gate requires two GPU nodes, found {len(nodes)}")
    producer = _Host.options(
        num_gpus=1,
        enable_tensor_transport=True,
        scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0]["NodeID"], soft=False),
    ).remote()
    consumer = _Host.options(
        num_gpus=1,
        enable_tensor_transport=True,
        scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[1]["NodeID"], soft=False),
    ).remote()
    try:
        report = _run_nixl_probe(producer, consumer, 3 * 1024 * 1024 * 1024, 200.0)
        assert report["source_node_id"] != report["node_id"]
        print(
            f"NIXL forced-cross-node probe: {report['gbps']:.2f} GB/s, {report['bytes']} bytes, "
            f"source={report['source_node_id']}, target={report['node_id']}"
        )
    finally:
        ray.kill(producer)
        ray.kill(consumer)
