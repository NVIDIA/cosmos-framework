# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for ExperienceBuffer.select (DESIGN §5).

Guards two bugs that a full-drain e2e test cannot catch:
- partial drain must NOT duplicate the non-batched items;
- select must not invoke Trajectory.__eq__ (tensor actions make `t in list` raise).
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.control import ExperienceBuffer, MetadataGate
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


def _traj(tid: str, action=None) -> Trajectory:
    tr = Transition(obs=ObsBundle(), action=action, done=True)
    return Trajectory(transitions=[tr], id=tid, status=Status.COMPLETED)


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_drain_does_not_duplicate():
    buf = ExperienceBuffer()
    buf.put_samples([_traj("a"), _traj("b"), _traj("c"), _traj("d"), _traj("e")])
    batch = buf.select(2)
    assert [t.id for t in batch] == ["a", "b"]
    # remaining must be exactly the 3 non-batched items, no duplication
    assert [t.id for t in buf._items] == ["c", "d", "e"]
    assert len(buf) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_drain_with_predicate_keeps_filtered_out():
    """Filtering reads a declared metadata gate only; a bare callable is refused at the public surface."""
    buf = ExperienceBuffer()
    items = [_traj("a"), _traj("b"), _traj("c"), _traj("d"), _traj("e")]
    for t in items:
        t.transitions[0].algo_extra["keep"] = float(t.id in {"a", "c", "e"})
    buf.put_samples(items)
    # gate passes a, c, e; take 1 -> batch=[a]; remaining keeps b,c,d,e once each
    batch = buf.select(1, where=MetadataGate(key="keep"))
    assert [t.id for t in batch] == ["a"]
    assert sorted(t.id for t in buf._items) == ["b", "c", "d", "e"]
    for filtering in (
        lambda: buf.select(1, where=lambda t: t.id == "b"),
        lambda: buf.evict(lambda t: t.id == "b"),
        lambda: buf.count(lambda t: True),
        lambda: buf.select_prioritized(1, lambda t: True),
    ):
        with pytest.raises(TypeError, match="must be a MetadataGate"):
            filtering()
    assert sorted(t.id for t in buf._items) == ["b", "c", "d", "e"]  # a refused filter changes nothing
    buf.evict(MetadataGate(key="keep"))
    assert sorted(t.id for t in buf._items) == ["b", "d"]


@pytest.mark.L0
@pytest.mark.CPU
def test_select_with_tensor_actions_does_not_raise():
    torch = pytest.importorskip("torch")
    buf = ExperienceBuffer()
    buf.put_samples(
        [
            _traj("a", action=torch.tensor([1, 2, 3])),
            _traj("b", action=torch.tensor([4, 5, 6])),
            _traj("c", action=torch.tensor([7, 8, 9])),
        ]
    )
    batch = buf.select(1)  # partial drain over tensor-carrying trajectories
    assert [t.id for t in batch] == ["a"]
    assert [t.id for t in buf._items] == ["b", "c"]


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_select_rejects_an_underfilled_batch_without_consuming_it() -> None:
    buf = ExperienceBuffer()
    buf.put_samples([_traj("a")])

    with pytest.raises(RuntimeError, match="returned 1 samples, expected exactly 2"):
        buf.get_exact_samples(2)

    assert [trajectory.id for trajectory in buf._items] == ["a"]


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_select_uses_role_bound_batch_size() -> None:
    buf = ExperienceBuffer(exact_batch_size=2)
    buf.put_samples([_traj("a"), _traj("b"), _traj("c")])

    assert [trajectory.id for trajectory in buf.get_exact_samples()] == ["a", "b"]
    assert [trajectory.id for trajectory in buf._items] == ["c"]


@pytest.mark.L0
@pytest.mark.CPU
def test_exact_select_rejects_override_of_role_bound_batch_size() -> None:
    buf = ExperienceBuffer(exact_batch_size=2)
    buf.put_samples([_traj("a"), _traj("b")])

    with pytest.raises(ValueError, match="cannot override the role-bound"):
        buf.get_exact_samples(1)

    assert [trajectory.id for trajectory in buf._items] == ["a", "b"]


def _stamped_group(group_id: str, spread: float, size: int = 2) -> list[Trajectory]:
    """One complete group whose members carry the reward stage's spread stamp."""
    members = []
    for k in range(size):
        tr = Transition(obs=ObsBundle(), done=True)
        tr.algo_extra["spread"] = spread
        members.append(Trajectory(transitions=[tr], id=f"{group_id}_k{k}", group_id=group_id))
    return members


@pytest.mark.L0
@pytest.mark.CPU
def test_select_prioritized_is_group_atomic_and_drain_returns_the_rest() -> None:
    """Gate-passing groups first, first-seen filler after, never a split group; drain hands back the rest."""
    gate = MetadataGate(key="spread", threshold=0.0)
    buf = ExperienceBuffer()
    a0, a1 = _stamped_group("a", 0.0)  # tied
    b0, b1 = _stamped_group("b", 1.0)  # informative
    c0, c1 = _stamped_group("c", 0.0)  # tied
    buf.put_samples([a0, b0, c0, a1, b1, c1])  # interleaved arrival, as replica fan-in produces
    assert buf.count(gate) == 2
    with pytest.raises(ValueError, match="cuts through group"):
        buf.select_prioritized(1, gate)
    assert [t.id for t in buf.select_prioritized(4, gate)] == ["b_k0", "b_k1", "a_k0", "a_k1"]
    assert [t.id for t in buf.drain()] == ["c_k0", "c_k1"]
    assert len(buf) == 0
