# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared model-free helpers for preference-cursor tests on a two-rank Ray grid."""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Callable

from cosmos_framework.posttrain.control.checkpointing import save_policy_checkpoint
from cosmos_framework.posttrain.roles.env.preference import DatasetPreferenceEnv
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
from cosmos_framework.posttrain.utils.rank_state import (
    RankStateWorldSizeMismatch,
    collect_rank_state,
    state_for_current_rank,
)
from cosmos_framework.posttrain.workers.grid import GridBackend, spawn_grid
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

_DATASET_SIZE = 8
_DATASET_SEED = 17
_SHUFFLE_SEED = 7
_WORLD_SIZE = 2


class CursorDataset:
    """Small stateful map dataset with deterministic identity and stochastic samples."""

    def __init__(self, size: int = _DATASET_SIZE, seed: int = _DATASET_SEED) -> None:
        self._items: list[str] = [f"pair-{index}" for index in range(size)]
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, str], dict[str, str]]:
        item = self._items[index]
        prompt = f"{item}:random={self._rng.random()}"
        return prompt, {"item": item}, {"item": item}

    def cursor_fingerprint(self) -> str:
        return "|".join(self._items)

    def cursor_state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_cursor_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class CursorlessDataset:
    """Map-style source that intentionally cannot make an exact cursor claim."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[str, dict[str, str], dict[str, str]]:
        return str(index), {"item": "winner"}, {"item": "loser"}


class CursorPreferenceEnv(DatasetPreferenceEnv):
    """Preference source with process identity exposed only for the Ray test."""

    def process_id(self) -> int:
        return os.getpid()


def _prompts(trajectories: list[Trajectory]) -> list[str]:
    return [str(trajectory.transitions[0].obs.representations["prompt"]) for trajectory in trajectories]


class CursorCheckpointProbe:
    """Policy-side probe for the real rank-state collection and restore contracts."""

    def __init__(self) -> None:
        self._component: dict[str, Any] | None = None

    def save(
        self,
        path: str,
        data_state: dict[str, Any] | None = None,
        *,
        publish: bool = True,
    ) -> None:
        del path
        assert publish
        self._component = collect_rank_state(data_state)

    def component(self) -> dict[str, Any]:
        if self._component is None:
            raise RuntimeError("cursor component has not been collected")
        return copy.deepcopy(self._component)

    def process_id(self) -> int:
        return os.getpid()

    def restore_next_prompt(self, component: dict[str, Any]) -> str:
        state = state_for_current_rank(component)
        if state is None:
            raise RuntimeError("cursor component unexpectedly has no rank state")
        env = DatasetPreferenceEnv(CursorDataset(seed=999), shuffle=True, seed=_SHUFFLE_SEED)
        if not env.load_state_dict(state):
            raise RuntimeError("cursor state unexpectedly degraded during exact restore")
        return _prompts(env.get_samples(1))[0]

    def converge_injected_failure(self, failing_rank: int) -> str:
        import torch.distributed as dist

        local_error = ValueError("injected cursor failure") if dist.get_rank() == failing_rank else None
        try:
            raise_if_any_rank_failed(local_error, "preference cursor test")
        except RuntimeError as error:
            return str(error)
        return "no distributed failure"

    def collect_mixed_presence(self, missing_rank: int) -> str:
        import torch.distributed as dist

        state = None if dist.get_rank() == missing_rank else {"rank": dist.get_rank()}
        try:
            collect_rank_state(state)
        except RuntimeError as error:
            return str(error)
        return "mixed rank-local state unexpectedly succeeded"

    def collect_unpickleable_state(self, failing_rank: int) -> str:
        import torch.distributed as dist

        state: dict[str, Any] = {"rank": dist.get_rank()}
        if dist.get_rank() == failing_rank:
            state["unpickleable"] = lambda: None
        try:
            collect_rank_state(state)
        except RuntimeError as error:
            return str(error)
        return "unpickleable rank-local state unexpectedly succeeded"

    def snapshot_unsupported_auto(self) -> bool:
        env = DatasetPreferenceEnv(CursorlessDataset(), cursor_resume="auto")
        return env.state_dict() is None

    def snapshot_mixed_capability(self, unavailable_rank: int) -> str:
        import torch.distributed as dist

        dataset = CursorlessDataset() if dist.get_rank() == unavailable_rank else CursorDataset()
        env = DatasetPreferenceEnv(dataset, cursor_resume="auto")
        try:
            env.state_dict()
        except RuntimeError as error:
            return str(error)
        return "mixed cursor capability unexpectedly succeeded"

    def world_size_policy(self) -> tuple[bool, str]:
        component = {
            "version": 1,
            "world_size": 3,
            "states": {
                0: {"position": 0},
                1: {"position": 0},
                2: {"position": 0},
            },
        }
        skipped = state_for_current_rank(
            component,
            on_world_size_mismatch="skip",
        )
        try:
            state_for_current_rank(component, on_world_size_mismatch="raise")
        except RankStateWorldSizeMismatch as error:
            return skipped is None, str(error)
        return skipped is None, "strict world-size mismatch unexpectedly succeeded"

    def converge_asymmetric_restore_failure(self, failing_rank: int) -> str:
        import torch.distributed as dist

        env = DatasetPreferenceEnv(CursorDataset(), shuffle=True, seed=_SHUFFLE_SEED)
        state = env.state_dict()
        if state is None:
            raise RuntimeError("cursor state unexpectedly unavailable")
        if dist.get_rank() == failing_rank:
            state["dataset_fingerprint"] = "corrupt-on-one-rank"
        try:
            env.load_state_dict(state)
        except RuntimeError as error:
            return str(error)
        return "asymmetric cursor restore unexpectedly succeeded"


def setup_cursor_bodies() -> dict[str, Any]:
    """Build rank-aware source and policy bodies after the actor process group exists."""
    return {
        "data": CursorPreferenceEnv(CursorDataset(), shuffle=True, seed=_SHUFFLE_SEED),
        "policy": CursorCheckpointProbe(),
    }


def setup_cursor_data_body() -> dict[str, Any]:
    """Build only the data role for a per-role actor grid."""
    return {
        "data": CursorPreferenceEnv(CursorDataset(), shuffle=True, seed=_SHUFFLE_SEED),
    }


def setup_cursor_policy_body() -> dict[str, Any]:
    """Build only the policy-side checkpoint probe for a per-role actor grid."""
    return {"policy": CursorCheckpointProbe()}


def _spawn_cursor_grid_without_mesh(
    group_id: str,
    setup: Callable[[], dict[str, Any]] = setup_cursor_bodies,
) -> dict[tuple[int, int], tuple[str, Any]]:
    """Create the two-rank Gloo grid without importing optional model-mesh dependencies."""
    import ray

    from cosmos_framework.posttrain.workers.ray import _Host

    grid: dict[tuple[int, int], tuple[str, Any]] = {}
    for rank in range(_WORLD_SIZE):
        store_id = f"{group_id}_r0_k{rank}"
        grid[(0, rank)] = (store_id, _Host.remote())

    master_addr, master_port = ray.get(grid[(0, 0)][1].rendezvous_endpoint.remote())
    ray.get(
        [
            grid[(0, rank)][1].init_distributed.remote(master_addr, master_port, rank, _WORLD_SIZE)
            for rank in range(_WORLD_SIZE)
        ]
    )
    ray.get([host.run_setup.remote(setup, True, 1 << 20, (), store_id, 1, 1) for store_id, host in grid.values()])
    peers = {store_id: host for store_id, host in grid.values()}
    ray.get([host.set_peers.remote(peers) for host in peers.values()])
    return grid


def make_cursor_grid(
    group_id: str, gpu_request: float, *, inject_device_mesh: bool = True
) -> tuple[WorkerGroup, WorkerGroup]:
    """Create a two-rank production GridBackend using Gloo on CPU or NCCL on GPU."""
    data = WorkerGroup("data", role="env", replicas=1, ranks=_WORLD_SIZE)
    policy = WorkerGroup("policy", role="trainer", replicas=1, ranks=_WORLD_SIZE)
    if inject_device_mesh:
        grid = spawn_grid(
            setup_cursor_bodies,
            replicas=1,
            ranks=_WORLD_SIZE,
            group_id=group_id,
            gpu_request=gpu_request,
            threshold=1 << 20,
            materialize_names=(),
            init_dist=True,
        )
    else:
        grid = _spawn_cursor_grid_without_mesh(group_id)
    backend = GridBackend(grid, shape=(1, _WORLD_SIZE))
    data._bind_backend(backend)
    policy._bind_backend(backend)
    return data, policy


def make_cursor_per_role_grids(group_id: str) -> tuple[WorkerGroup, WorkerGroup]:
    """Create separate two-rank data and policy Ray/Gloo actor meshes."""
    data = WorkerGroup("data", role="env", replicas=1, ranks=_WORLD_SIZE)
    policy = WorkerGroup("policy", role="trainer", replicas=1, ranks=_WORLD_SIZE)
    data_grid = _spawn_cursor_grid_without_mesh(
        f"{group_id}_data",
        setup_cursor_data_body,
    )
    policy_grid = _spawn_cursor_grid_without_mesh(
        f"{group_id}_policy",
        setup_cursor_policy_body,
    )
    data._bind_backend(GridBackend(data_grid, shape=(1, _WORLD_SIZE)))
    policy._bind_backend(GridBackend(policy_grid, shape=(1, _WORLD_SIZE)))
    return data, policy


def assert_cursor_checkpoint_round_trip(
    data: WorkerGroup,
    policy: WorkerGroup,
    *,
    expect_separate_roles: bool = False,
) -> None:
    """Exercise rank alignment, checkpoint collection, exact restore, and failure convergence."""
    worker_pids = set(policy.process_id().all().values())
    assert len(worker_pids) == _WORLD_SIZE
    assert os.getpid() not in worker_pids
    data_pids = set(data.process_id().all().values())
    if expect_separate_roles:
        assert data_pids.isdisjoint(worker_pids)
    else:
        assert data_pids == worker_pids

    first = data.get_samples(1).all()
    first_prompts = {rank: _prompts(first[(0, rank)])[0] for rank in range(_WORLD_SIZE)}
    assert first_prompts[0].split(":", 1)[0] != first_prompts[1].split(":", 1)[0]

    save_policy_checkpoint(policy, "unused-by-cursor-probe", data=data)
    components = policy.component().all()
    component = components[(0, 0)]
    assert components[(0, 1)] == component
    assert component["version"] == 1
    assert component["world_size"] == _WORLD_SIZE
    assert set(component["states"]) == {0, 1}
    assert component["states"][0]["rank"] == 0
    assert component["states"][1]["rank"] == 1
    for state in component["states"].values():
        assert {"order", "next_index", "samples_emitted"}.isdisjoint(state)

    expected = data.get_samples(1).all()
    expected_prompts = {rank: _prompts(expected[(0, rank)])[0] for rank in range(_WORLD_SIZE)}
    restored = policy.restore_next_prompt(component).all()
    restored_prompts = {rank: restored[(0, rank)] for rank in range(_WORLD_SIZE)}
    assert restored_prompts == expected_prompts

    failures = policy.converge_injected_failure(1).all()
    for message in failures.values():
        assert "preference cursor test failed on one or more ranks" in message
        assert "rank 1: ValueError: injected cursor failure" in message

    mixed = policy.collect_mixed_presence(1).all()
    for message in mixed.values():
        assert "rank-local checkpoint state must be present on every rank or none" in message

    unpickleable = policy.collect_unpickleable_state(1).all()
    for message in unpickleable.values():
        assert "rank-local checkpoint state preparation failed on one or more ranks" in message
        assert "rank 1:" in message

    unsupported = policy.snapshot_unsupported_auto().all()
    assert all(omitted for omitted in unsupported.values())

    mixed_capability = policy.snapshot_mixed_capability(1).all()
    for message in mixed_capability.values():
        assert "preference source cursor capability differs across ranks" in message
        assert "unavailable on ranks [1]" in message

    world_size = policy.world_size_policy().all()
    for skipped, message in world_size.values():
        assert skipped
        assert "world_size=3" in message
        assert "current world_size=2" in message

    restore_failures = policy.converge_asymmetric_restore_failure(1).all()
    for message in restore_failures.values():
        assert "preference source restore failed on one or more ranks" in message
        assert "rank 1: ValueError: DatasetPreferenceEnv cursor dataset_fingerprint" in message
