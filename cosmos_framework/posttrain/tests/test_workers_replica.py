# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `workers/replica.py::Communicator` — cross-replica reduction of driver-gathered values."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest


class _FakeRay:
    """Stands in for Ray: `all_reduce`'s only use of it is `ray.get(refs)`, and the behavior under
    test is the REDUCTION, not the transport. Starting a cluster would make this an L1."""

    @staticmethod
    def get(refs):
        return refs  # already-gathered values stand in for ObjectRefs

    @staticmethod
    def remote(cls):
        return cls  # `@ray.remote` at class-definition time; identity is enough to import the module


def _comm(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    """The `Communicator` with Ray stubbed out, importable WITHOUT ray installed.

    Two substitutions, and both are needed. `workers/replica.py` does `import ray` at module top and
    decorates `_Replica` with `@ray.remote`, so merely monkeypatching the module attribute is too
    late -- the import itself raises ModuleNotFoundError on an image without ray, which is exactly
    what the CPU CI runner is. So a stub goes into `sys.modules` FIRST (same trick as
    `test_entrypoints_grpo.py` uses for `workers.ray`), and only then is the module imported.

    The setattr afterwards is not redundant: when ray IS installed, or when another test already
    imported this module in the same process, the import returns the cached module holding the REAL
    ray, and the reduction under test would try to `ray.get` driver-side values.

    This test is @CPU on purpose -- `Communicator.all_reduce`'s reduction is pure Python over
    already-gathered values, so requiring a cluster to test it would be testing the wrong thing.
    """
    try:
        __import__("ray")
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=_FakeRay.get, remote=_FakeRay.remote))
    from cosmos_framework.posttrain.workers import replica as mod

    monkeypatch.setattr(mod, "ray", _FakeRay)
    return mod, mod.Communicator.__new__(mod.Communicator)


@pytest.mark.L0
@pytest.mark.CPU
def test_metric_dicts_require_explicit_reduction_contributions(monkeypatch):
    """A finalized dictionary cannot be safely reduced by numeric type or key spelling."""
    mod, comm = _comm(monkeypatch)
    gathered = [{"loss": 1.0, "n": 3, "tag": "a"}, {"loss": 3.0, "n": 5, "tag": "b"}]

    with pytest.raises(TypeError, match="MetricContribution"):
        mod.Communicator.all_reduce(comm, gathered, op="mean")


@pytest.mark.L0
@pytest.mark.CPU
def test_metric_contributions_merge_counts_weighted_means_and_missing_keys(monkeypatch):
    """Reducer semantics ride on each value; missing means unobserved rather than zero."""
    from cosmos_framework.posttrain.schema import MetricContribution

    mod, comm = _comm(monkeypatch)
    gathered = [
        {
            "train/loss_avg": MetricContribution.weighted_mean(1.0, 1),
            "train/valid_token_count": MetricContribution.sum(3),
        },
        {
            "train/loss_avg": MetricContribution.weighted_mean(3.0, 3),
            "train/valid_token_count": MetricContribution.sum(5),
            "train/loss_max": MetricContribution.maximum(4.0),
        },
    ]
    got = mod.Communicator.all_reduce(comm, gathered, op="mean")
    assert got == {"train/loss_avg": 2.5, "train/valid_token_count": 8, "train/loss_max": 4.0}
    assert type(got["train/valid_token_count"]) is int


@pytest.mark.L0
@pytest.mark.CPU
def test_already_global_metric_reports_are_identity_checked_without_type_erasure(monkeypatch):
    """HSDP already reduced the report; the grid boundary only proves agreement."""
    torch = pytest.importorskip("torch")
    mod, comm = _comm(monkeypatch)
    first = {"train/loss_avg": torch.tensor(1.5), "train/valid_token_count": 7}  # loss: []
    second = {"train/loss_avg": torch.tensor(1.5), "train/valid_token_count": 7}  # loss: []

    got = mod.Communicator.all_reduce(comm, [first, second], op="identical")
    assert got is first
    assert type(got["train/valid_token_count"]) is int

    with pytest.raises(RuntimeError, match="differs between replica 0 and replica 1"):
        mod.Communicator.all_reduce(
            comm,
            [first, {"train/loss_avg": torch.tensor(2.0), "train/valid_token_count": 7}],  # loss: []
            op="identical",
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_explicit_unsynchronized_demo_selects_replica_zero_without_fake_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A divergent flow demo reports one honest local diagnostic, never a mean-of-means."""
    mod, comm = _comm(monkeypatch)
    first = {"train/loss_avg": 1.5, "train/valid_token_count": 3}
    second = {"train/loss_avg": 4.0, "train/valid_token_count": 9}

    got = mod.Communicator.all_reduce(comm, [first, second], op="replica0")

    assert got is first


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_awaits_one_cell_but_resolve_all_awaits_every_cell():
    """A write barrier must await EVERY cell; `resolve` awaits replica 0 alone.

    `resolve(replay.put_samples(...))` looks like an await and is one — for replica 0. At R>1 a
    failed write on any other replica is neither raised nor noticed: the samples are simply absent,
    and the next `get_samples` under-fills so the trainer steps on a short batch without complaint.
    Awaiting one cell of N catches exactly as much as awaiting none.
    """
    from cosmos_framework.posttrain.utils.entry import resolve, resolve_all

    awaited: list = []

    class _Cell:
        def __init__(self, rk):
            self.rk = rk

    class _CollectiveHandle:
        """Mirrors the real contract: `result()` is replica-0 rank-0, `all()` is every cell."""

        def __init__(self, cells):
            self._cells = cells

        def result(self):
            awaited.append((0, 0))
            return "r0"

        def all(self):
            for rk in self._cells:
                awaited.append(rk)
            return {rk: f"r{rk[0]}" for rk in self._cells}

    cells = [(0, 0), (1, 0), (2, 0)]

    awaited.clear()
    resolve(_CollectiveHandle(cells))
    assert awaited == [(0, 0)], f"resolve must await replica 0 only, got {awaited}"

    awaited.clear()
    got = resolve_all(_CollectiveHandle(cells))
    assert awaited == cells, f"resolve_all must await EVERY cell, got {awaited}"
    assert got == ["r0", "r1", "r2"]

    # Non-handles and single Handles stay trivially correct.
    assert resolve_all("plain") == ["plain"]

    class _Handle:
        def result(self):
            return 7

    assert resolve_all(_Handle()) == [7]


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_all_drains_late_cells_before_raising_aggregate_failure() -> None:
    """An early failed cell cannot release the control plane while a later write is unsettled."""
    from cosmos_framework.posttrain.utils.entry import resolve_all

    settled: list[tuple[int, int]] = []

    class _CollectiveHandle:
        @staticmethod
        def as_completed() -> Iterator[tuple[tuple[int, int], object]]:
            settled.append((0, 0))
            yield (0, 0), RuntimeError("early write failed")
            settled.append((1, 0))
            yield (1, 0), "late write completed"

    with pytest.raises(RuntimeError, match=r"cell \(0, 0\).*early write failed"):
        resolve_all(_CollectiveHandle())

    assert settled == [(0, 0), (1, 0)]


@pytest.mark.L0
@pytest.mark.CPU
def test_replica_lead_values_drains_every_rank_before_raising() -> None:
    """Selecting replica leaders is still a barrier over every submitted mesh cell."""
    from cosmos_framework.posttrain.utils.entry import replica_lead_values

    settled: list[tuple[int, int]] = []

    class _CollectiveHandle:
        @staticmethod
        def as_completed() -> Iterator[tuple[tuple[int, int], object]]:
            for coordinate, value in (
                ((0, 0), RuntimeError("leader failed")),
                ((0, 1), "nonlead finished"),
                ((1, 0), "late leader finished"),
            ):
                settled.append(coordinate)
                yield coordinate, value

    with pytest.raises(RuntimeError, match=r"cell \(0, 0\).*leader failed"):
        replica_lead_values(_CollectiveHandle())

    assert settled == [(0, 0), (0, 1), (1, 0)]

    class _SuccessHandle:
        @staticmethod
        def as_completed() -> Iterator[tuple[tuple[int, int], object]]:
            yield (1, 0), "replica-1"
            yield (0, 1), "ignored-nonlead"
            yield (0, 0), "replica-0"

    assert replica_lead_values(_SuccessHandle()) == ["replica-0", "replica-1"]
