# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cross-node coverage for the Ray/NIXL trajectory data path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from cosmos_framework.posttrain.comm.bulk import BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_payload, pack_payload
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory, Transition
from cosmos_framework.posttrain.workers.data_pool import (
    FIELD_ALIGN_BYTES,
    ConsumerDataPool,
    ProducerDataPool,
    release_pool_step_barrier,
)

PAGE_BYTES = 4 * 1024 * 1024
POOL_BYTES = 4 * PAGE_BYTES
REPO_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_ENV = {"env_vars": {"PYTHONPATH": os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", "")))}}
# (tier, shape, dtype, fill base) in staging order. Per tier: two uint8 fields sharing a page, a
# 1-byte field that leaves the next offset unaligned, a float32 field that must still reinterpret,
# a 0-D scalar, then an oversize field.
FIELDS: dict[str, tuple[str, tuple[int, ...], torch.dtype, int]] = {
    "cuda_first": ("cuda", (PAGE_BYTES // 4,), torch.uint8, 11),
    "cuda_second": ("cuda", (PAGE_BYTES // 2,), torch.uint8, 12),
    "cuda_odd": ("cuda", (1,), torch.uint8, 14),
    "cuda_f32": ("cuda", (1000,), torch.float32, 15),
    "cuda_scalar": ("cuda", (), torch.float32, 16),
    "cuda_oversize": ("cuda", (PAGE_BYTES + 257,), torch.uint8, 13),
    "cpu_first": ("cpu", (PAGE_BYTES // 4,), torch.uint8, 21),
    "cpu_second": ("cpu", (PAGE_BYTES // 2,), torch.uint8, 22),
    "cpu_odd": ("cpu", (1,), torch.uint8, 24),
    "cpu_f32": ("cpu", (1000,), torch.float32, 25),
    "cpu_scalar": ("cpu", (), torch.float32, 26),
    "cpu_oversize": ("cpu", (PAGE_BYTES + 257,), torch.uint8, 23),
}
EVICTION_FIELDS = {
    "first": (3 * PAGE_BYTES // 4, 31),
    "second": (3 * PAGE_BYTES // 4, 32),
}


def _expected(base: int, step: int) -> int:
    return (base + step) % 251


def _matches(tensor: torch.Tensor, name: str, step: int) -> bool:
    _kind, shape, dtype, base = FIELDS[name]
    return (
        tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and bool(torch.all(tensor == _expected(base, step)).item())
    )


@ray.remote(num_gpus=1, num_cpus=2, enable_tensor_transport=True)
class _Producer:
    def __init__(self) -> None:
        self.pool = ProducerDataPool(
            "producer",
            cuda_capacity_bytes=POOL_BYTES,
            cpu_capacity_bytes=POOL_BYTES,
            page_bytes=PAGE_BYTES,
        )

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def warmup(self) -> None:
        self.pool.warmup()

    def publish(self, step: int) -> Trajectory:
        lease = f"step-{step}"
        fields = {
            name: torch.full(shape, _expected(base, step), dtype=dtype, device="cuda" if kind == "cuda" else "cpu")
            for name, (kind, shape, dtype, base) in FIELDS.items()
        }
        trajectory = Trajectory(transitions=[Transition(obs=ObsBundle(), action=fields)])
        packed = pack_payload(
            trajectory,
            BulkStore(),
            threshold=0,
            pool_stage=lambda tensor: self.pool.stage_tensor(lease, tensor),
        )
        return self.pool.publish_step(lease, [packed])[0]

    def verify_local_views(self, packed: Trajectory, step: int) -> bool:
        """Same-actor path: every published field reinterprets correctly as a direct page view."""
        return all(
            _matches(self.pool.view_tensor(slice_), name, step) for name, slice_ in packed.transitions[0].action.items()
        )

    def publish_dropped(self, step: int) -> None:
        lease = f"dropped-{step}"
        trajectory = Trajectory(
            transitions=[
                Transition(
                    obs=ObsBundle(),
                    action={"dropped": torch.full((PAGE_BYTES // 4,), step, dtype=torch.uint8)},
                )
            ]
        )
        packed = pack_payload(
            trajectory,
            BulkStore(),
            threshold=0,
            pool_stage=lambda tensor: self.pool.stage_tensor(lease, tensor),
        )
        self.pool.publish_step(lease, [packed])

    def publish_eviction_pair(self) -> dict[str, Any]:
        lease = "eviction-pair"
        # Stage directly: pack_payload only externalizes Trajectory/ObsBundle payloads.
        packed = {
            name: self.pool.stage_tensor(lease, torch.full((nbytes,), value, dtype=torch.uint8, device="cuda"))
            for name, (nbytes, value) in EVICTION_FIELDS.items()
        }
        return self.pool.publish_step(lease, [packed])[0]

    def release_all_pool_leases(self) -> tuple[str, ...]:
        return self.pool.release_all_leases()

    def close(self) -> None:
        self.pool.close()


@ray.remote(num_gpus=1, num_cpus=2, enable_tensor_transport=True)
class _Consumer:
    def __init__(self) -> None:
        self.pool = ConsumerDataPool(
            cuda_cache_bytes=PAGE_BYTES,
            cpu_cache_bytes=PAGE_BYTES,
            page_bytes=PAGE_BYTES,
        )

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def warmup(self) -> None:
        self.pool.warmup()

    def consume(self, packed: Trajectory, step: int) -> tuple[int, int]:
        outcomes: list[bool] = []
        pool = self.pool

        class _Transport:
            @staticmethod
            def fetch_pool_slice(slice_: Any) -> torch.Tensor:
                tensor, hit = pool.fetch_tensor(slice_)
                outcomes.append(hit)
                return tensor

        trajectory = materialize_payload(packed, _Transport())
        for name, tensor in trajectory.transitions[0].action.items():
            assert _matches(tensor, name, step), name
        return outcomes.count(True), outcomes.count(False)

    def retain_across_eviction(self, packed: dict[str, Any]) -> tuple[bool, bool]:
        """Hold A while fetching B through the one-slot cache; both must stay correct."""
        first, second = packed["first"], packed["second"]
        assert first.page_ref != second.page_ref
        retained_first, _ = self.pool.fetch_tensor(first)
        fetched_second, _ = self.pool.fetch_tensor(second)
        return (
            bool(torch.all(retained_first == EVICTION_FIELDS["first"][1]).item()),
            bool(torch.all(fetched_second == EVICTION_FIELDS["second"][1]).item()),
        )

    def close(self) -> None:
        self.pool.close()


@ray.remote(num_gpus=0, num_cpus=2, enable_tensor_transport=True)
class _CpuConsumer:
    """CPU-placed actor on a GPU node: it inherits CUDA_VISIBLE_DEVICES yet must never touch CUDA."""

    def __init__(self) -> None:
        self.pool = ConsumerDataPool(cuda_cache_bytes=PAGE_BYTES, cpu_cache_bytes=PAGE_BYTES, page_bytes=PAGE_BYTES)

    def warmup(self) -> tuple[list[str], bool, int, int]:
        self.pool.warmup()
        return (
            list(ray.get_gpu_ids()),
            torch.cuda.is_initialized(),
            len(self.pool._slots["cuda"]),
            len(self.pool._slots["cpu"]),
        )

    def consume_cpu_fields(self, packed: Trajectory, step: int) -> tuple[bool, str, bool]:
        """Receive every CPU field cross-node; a CUDA slice must fail clearly; CUDA stays uninitialized."""
        slices = packed.transitions[0].action
        cpu_ok = all(
            _matches(self.pool.fetch_tensor(slice_)[0], name, step)
            for name, slice_ in slices.items()
            if name.startswith("cpu_")
        )
        try:
            self.pool.fetch_tensor(slices["cuda_first"])
            cuda_error = ""
        except RuntimeError as error:
            cuda_error = str(error)
        return cpu_ok, cuda_error, torch.cuda.is_initialized()

    def close(self) -> None:
        self.pool.close()


def _gpu_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) >= 1]
    if len(nodes) < 2:
        raise RuntimeError("need two live GPU nodes")
    return nodes[:2]


@pytest.mark.L2
@pytest.mark.GPU
def test_cross_node_nixl_trajectory_roundtrip() -> None:
    """Exercise packing, mixed pools, cache hits, oversize fields, reuse, and release."""
    if os.environ.get("POSTTRAIN_RUN_MULTINODE_NIXL") != "1":
        pytest.skip("set POSTTRAIN_RUN_MULTINODE_NIXL=1 on a two-node Ray/NIXL cluster")

    ray.init(address="auto")
    producer_node, consumer_node = _gpu_nodes()
    producer = _Producer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=producer_node["NodeID"], soft=False),
        runtime_env=RUNTIME_ENV,
    ).remote()
    consumer = _Consumer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=consumer_node["NodeID"], soft=False),
        runtime_env=RUNTIME_ENV,
    ).remote()
    cpu_consumer = _CpuConsumer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=consumer_node["NodeID"], soft=False),
        runtime_env=RUNTIME_ENV,
    ).remote()
    try:
        ray.get([producer.warmup.remote(), consumer.warmup.remote()])
        assert ray.get(producer.node_id.remote()) != ray.get(consumer.node_id.remote())
        # CPU-placed actor: no Ray GPU assignment, CPU tier only, no CUDA context.
        assert ray.get(cpu_consumer.warmup.remote()) == ([], False, 0, 1)
        eviction_payload = ray.get(producer.publish_eviction_pair.remote())
        assert ray.get(consumer.retain_across_eviction.remote(eviction_payload)) == (True, True)
        assert release_pool_step_barrier(eviction_payload, {"producer": producer}) == 1
        for step in range(2):
            ray.get(producer.publish_dropped.remote(step))
            payload = ray.get(producer.publish.remote(step))
            slices = payload.transitions[0].action
            assert {slice_.cache_key() for slice_ in payload.pool_leases} == {
                slice_.cache_key() for slice_ in slices.values()
            }
            for tier in ("cuda", "cpu"):
                shared = [slices[f"{tier}_{f}"] for f in ("first", "second", "odd", "f32", "scalar")]
                assert len({s.page_ref for s in shared}) == 1, tier
                assert slices[f"{tier}_f32"].offset % FIELD_ALIGN_BYTES == 0, slices[f"{tier}_f32"].offset
                assert slices[f"{tier}_scalar"].shape == () and slices[f"{tier}_scalar"].nbytes == 4
            assert slices["cuda_oversize"].oversize and slices["cpu_oversize"].oversize
            assert ray.get(producer.verify_local_views.remote(payload, step))
            distinct_pages = len({s.cache_key() for s in slices.values()})
            assert ray.get(consumer.consume.remote(payload, step)) == (len(slices) - distinct_pages, distinct_pages)
            cpu_ok, cuda_error, cuda_initialized = ray.get(cpu_consumer.consume_cpu_fields.remote(payload, step))
            assert cpu_ok and "without CUDA access" in cuda_error and not cuda_initialized
            assert release_pool_step_barrier(payload, {"producer": producer}) == 2
            assert all(slice_.page_ref is None for slice_ in slices.values())
        ray.get([cpu_consumer.close.remote(), consumer.close.remote(), producer.close.remote()])
    finally:
        ray.shutdown()
