# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public role binding preserves synchronous peers and their bulk ownership."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
from cosmos_framework.posttrain.roles import roles
from cosmos_framework.posttrain.schema import ObsBundle, Trajectory, Transition
from cosmos_framework.posttrain.utils.entry import resolve_all


def _bulk_state() -> tuple[int, int]:
    from cosmos_framework.posttrain.comm.bulk import _STORE_REGISTRY

    return sum(map(len, _STORE_REGISTRY.values())), sum(store.fetch_count for store in _STORE_REGISTRY.values())


class _Producer:
    metadata_only_input_methods = ("reject",)

    def __init__(self) -> None:
        from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV

        self.replica = int(os.environ.get(REPLICA_INDEX_ENV, "0"))

    def transform(self, observation: ObsBundle) -> list[Trajectory]:
        values = observation.representations["values"]
        assert isinstance(values, torch.Tensor)
        return [Trajectory(transitions=[Transition(obs=observation, action=values + self.replica + 1)])]

    def fail(self, observation: ObsBundle) -> None:
        assert isinstance(observation.representations["values"], torch.Tensor)
        raise RuntimeError("injected peer failure")

    def reject(self, observation: ObsBundle) -> None:
        from cosmos_framework.posttrain.comm.transfer.mover import BulkRef

        assert isinstance(observation.representations["values"], BulkRef)
        raise RuntimeError("injected rejection before bulk consumption")

    def identity(self) -> tuple[int, int]:
        return os.getpid(), self.replica

    def read(self, samples: list[Trajectory]) -> float:
        return float(samples[0].transitions[0].action.sum())

    def bulk_state(self) -> tuple[int, int]:
        return _bulk_state()


class _Consumer:
    metadata_only_input_methods = ("forward",)

    def bind_peer(self, peer) -> None:
        self.peer = peer

    def forward(self, samples, peer) -> float:
        from cosmos_framework.posttrain.comm.codec import collect_bulk_refs

        assert collect_bulk_refs(samples), "the forwarding role must receive metadata without consuming bulk"
        return peer.read(samples)

    def exercise(self, *, asynchronous: bool) -> dict:
        before = _bulk_state()
        values = torch.arange(1024, dtype=torch.float32)
        observation = ObsBundle(representations={"values": values})
        if asynchronous:
            future = self.peer.call_async("transform", observation)
            samples = future.result()
            assert future.result() is samples
        else:
            samples = self.peer.transform(observation)
        producer_pid, replica = self.peer.identity()
        step = samples[0].transitions[0]
        torch.testing.assert_close(step.action, values + replica + 1)
        torch.testing.assert_close(step.obs.representations["values"], values)
        return {
            "consumer_pid": os.getpid(),
            "producer_pid": producer_pid,
            "replica": replica,
            "same_storage": step.obs.representations["values"] is values,
            "live_bulk": _bulk_state()[0] - before[0],
        }

    def fail(self, *, asynchronous: bool) -> None:
        observation = ObsBundle(representations={"values": torch.ones(1024)})
        if asynchronous:
            self.peer.call_async("fail", observation).result()
        else:
            self.peer.fail(observation)

    def reject_and_retain(self, *, asynchronous: bool) -> dict:
        import gc
        import weakref

        before = _bulk_state()[0]
        values = torch.ones(1024)
        source = weakref.ref(values)
        observation = ObsBundle(representations={"values": values})
        try:
            if asynchronous:
                self.peer.call_async("reject", observation).result()
            else:
                self.peer.reject(observation)
        except Exception as error:
            self.retained_error = error
        finally:
            values, observation = None, None
        gc.collect()
        return {
            "error": str(self.retained_error),
            "source_released": source() is None,
            "live_bulk": _bulk_state()[0] - before,
        }


def _setup(names: tuple[str, ...]) -> dict:
    return {name: _Producer() if name == "producer" else _Consumer() for name in names}


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("asynchronous", "topology"),
    [(False, "local"), (True, "colocated"), (False, "separate"), (True, "separate")],
)
def test_bound_peers_preserve_bulk_and_actor_alignment(
    topology: str, asynchronous: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ray

    replicas = 1 if topology == "local" else 2
    producer = roles.rollout("producer", replicas=replicas)
    consumer = roles.env("consumer", replicas=replicas, needs_bulk=True)
    if topology == "separate":
        actors = (
            ActorGroup((producer,), partial(_setup, ("producer",))),
            ActorGroup((consumer,), partial(_setup, ("consumer",))),
        )
    else:
        actors = (ActorGroup((producer, consumer), partial(_setup, ("producer", "consumer"))),)
    # Keep Ray's Unix socket names short while storing every artifact under pytest's task directory.
    monkeypatch.chdir(tmp_path)
    ray_directory = f"/proc/{os.getpid()}/cwd/ray"
    try:
        bind_device_groups(
            [DeviceGroup(actors, device="cpu")],
            backend="local" if topology == "local" else "ray",
            temp_dir=ray_directory,
            threshold=64,
        )
        resolve_all(consumer.bind_peer(producer))
        initial_states = resolve_all(producer.bulk_state())
        result = consumer.exercise(asynchronous=asynchronous)
        if topology != "local":
            ray.get(list(result._refs.values()), timeout=30)
        rows = resolve_all(result)
        assert {row["replica"] for row in rows} == set(range(replicas))
        assert all((row["producer_pid"] == row["consumer_pid"]) == (topology != "separate") for row in rows)
        assert all(row["same_storage"] == (topology != "separate") for row in rows)
        assert all(row["live_bulk"] == 0 for row in rows)
        states = resolve_all(producer.bulk_state())
        for (initial_live, initial_fetches), (live, fetches) in zip(initial_states, states, strict=True):
            assert live == initial_live
            assert (fetches > initial_fetches) == (topology == "separate")
        if topology != "local":
            values = torch.arange(1024, dtype=torch.float32)
            payload = producer.transform(ObsBundle(representations={"values": values}))
            forwarded = resolve_all(consumer.forward(payload, producer))
            assert forwarded == [float((values + replica + 1).sum()) for replica in range(replicas)]
        with pytest.raises(Exception, match="injected peer failure"):
            resolve_all(consumer.fail(asynchronous=asynchronous))
        assert [live for live, _ in resolve_all(producer.bulk_state())] == [live for live, _ in initial_states]
        if topology == "separate":
            failures = resolve_all(consumer.reject_and_retain(asynchronous=asynchronous))
            assert all("injected rejection before bulk consumption" in row["error"] for row in failures)
            assert all(row["source_released"] and row["live_bulk"] == 0 for row in failures)
    finally:
        if topology != "local":
            ray.shutdown()
