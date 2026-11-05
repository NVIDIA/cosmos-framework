# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CollectiveHandle + _resolve passthrough unit tests (no ray)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.workers.grid import GridBackend
from cosmos_framework.posttrain.workers.worker_group import (
    CollectiveHandle,
    Handle,
    RankAligned,
    Scatter,
    WorkerGroup,
    _local_select,
    _resolve,
    release_result_handles,
    scatter,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_collective_handle_indexing_and_release():
    refs = {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}
    ch = CollectiveHandle(refs, (2, 2), getter=lambda x: x.upper())
    assert ch.shape == (2, 2)
    assert ch.ref_at(1, 0) == "c"
    assert ch.all() == {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}
    assert ch.result() == "A"  # (0,0)
    assert ch.wait() == "A"  # wait alias must resolve via CollectiveHandle.result, not Handle.result
    plain = Handle(ref=object(), getter=lambda value: value)
    release_result_handles(plain, ch, "concrete")
    assert ch._refs == {}
    with pytest.raises(RuntimeError, match="released"):
        plain.result()
    with pytest.raises(RuntimeError, match="released"):
        ch.result()


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_passes_collective_handle_through_but_resolves_plain():
    plain = Handle(value=7)
    ch = CollectiveHandle({(0, 0): 1}, (1, 1), getter=lambda x: x)
    assert _resolve(plain) == 7  # plain Handle resolved
    assert _resolve(ch) is ch  # CollectiveHandle passed through untouched


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_passes_scatter_through():
    s = scatter([1, 2, 3])
    assert isinstance(s, Scatter) and s.seq == [1, 2, 3]
    assert _resolve(s) is s  # dispatch marker passed through to the backend untouched


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_aligned_marker_unwraps_one_eager_cell() -> None:
    class _Body:
        def echo(self, value: list[str]) -> list[str]:
            return value

    source = Handle(value=["rank-local"])
    marker = RankAligned(source)
    assert _resolve(marker) is marker
    worker = WorkerGroup("w", "test", body=_Body())
    assert worker.echo(marker).result() == ["rank-local"]


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_aligned_routes_exact_rank_and_rejects_shape_drift() -> None:
    # This path is pure alignment arithmetic; bypass the constructor because it
    # intentionally imports the Ray-backed replica runtime.
    backend = object.__new__(GridBackend)
    handle = CollectiveHandle(
        {
            (0, 0): "replica-0-rank-0",
            (0, 1): "replica-0-rank-1",
            (1, 0): "replica-1-rank-0",
            (1, 1): "replica-1-rank-1",
        },
        (2, 2),
        getter=lambda value: value,
    )
    marker = RankAligned(handle)
    assert backend._align_one(marker, 0, 0, (2, 2)) == "replica-0-rank-0"
    assert backend._align_one(marker, 0, 1, (2, 2)) == "replica-0-rank-1"
    assert backend._align_one(marker, 1, 0, (2, 2)) == "replica-1-rank-0"
    assert backend._align_one(marker, 1, 1, (2, 2)) == "replica-1-rank-1"
    with pytest.raises(ValueError, match="identical producer and consumer shapes"):
        backend._align_one(marker, 0, 0, (1, 4))
    with pytest.raises(ValueError, match="identical producer and consumer shapes"):
        backend._align_one(marker, 0, 0, (2, 1))


@pytest.mark.L0
@pytest.mark.CPU
def test_collective_handle_as_completed_fallback_and_error_passthrough():
    # non-ObjectRef refs (identity getter) -> insertion-order fallback; a getter that
    # raises for one cell yields the Exception as that cell's value (drain not aborted).
    def _get(x):
        if x == "boom":
            raise RuntimeError("kaboom")
        return x.upper()

    ch = CollectiveHandle({(0, 0): "a", (1, 0): "boom", (2, 0): "c"}, (3, 1), getter=_get)
    out = dict(ch.as_completed())
    assert out[(0, 0)] == "A" and out[(2, 0)] == "C"
    assert isinstance(out[(1, 0)], Exception)


@pytest.mark.L0
@pytest.mark.CPU
def test_worker_group_hosts_gpus_empty_on_local_backend():
    wg = WorkerGroup("w", "test")  # unbound (_backend is None) -> local/eager
    assert wg.hosts() == {}  # real methods, not routed through __getattr__ to a body
    assert wg.gpus() == {}
    assert _resolve(5) == 5  # non-handle passthrough


@pytest.mark.L0
@pytest.mark.CPU
def test_worker_group_has_ranks_field():
    wg = WorkerGroup("policy", "rollout", replicas=2, ranks=2)
    assert wg.replicas == 2 and wg.ranks == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_worker_group_dispatch_profile_is_explicit_contract_not_placement() -> None:
    wg = WorkerGroup("policy", "trainer", dispatch_profile="nft", pin_to="learner")
    assert wg.dispatch_profile == "nft"
    assert wg.placement == {"pin_to": "learner"}
    with pytest.raises(ValueError, match="dispatch_profile"):
        WorkerGroup("policy", "trainer", dispatch_profile=" nft ")


@pytest.mark.L0
@pytest.mark.CPU
def test_trainer_model_parallelism_is_not_placement_metadata() -> None:
    from cosmos_framework.posttrain.roles.core.factory import trainer

    wg = trainer("policy", replicas=1, ranks=8, model_parallelism=(2, 4), pin_to="learner")
    assert wg.model_parallelism == (2, 4)
    assert wg.placement == {"pin_to": "learner"}


@pytest.mark.L0
@pytest.mark.CPU
def test_local_train_selection_uses_live_policy_dp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    items = [SimpleNamespace(group_id="g", id=str(index)) for index in range(4)]

    args, kwargs = _local_select(
        "trainer",
        "train",
        (items,),
        {"mini": 0, "micro": 0, "n_mini": 1, "n_micro": 1},
    )

    assert [item.id for item in args[0]] == ["1", "3"]
    assert kwargs == {}
