# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experience buffer — a metadata index, not a data store (DESIGN §5).

Final form: holds only minimal inline metadata (id/status/version/group/reward/
advantage/size/lineage + bulk locator+lease); filter/dispatch/advantage run on
metadata; bulk stays where produced; trainer pulls point-to-point after dispatch.

v0: single process ⇒ no metadata/bulk split needed; the buffer just holds whole
Trajectories in memory. The interface (insert/select/evict) is the real one so the
metadata-index + on-demand-pull implementation drops in later (§3.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from cosmos_framework.posttrain.schema import Status, Trajectory


@dataclass(frozen=True)
class MetadataGate:
    """Declared buffer predicate: terminal ``algo_extra[key] > threshold`` (DESIGN §5).

    The gating scalar must be precomputed by the stage that owns it (for GRPO, the reward-time
    advantage estimator stamps its group facts); the buffer only compares it. Declaring the predicate
    as data rather than a callable keeps it serializable to a remote replay cell and makes it
    structurally impossible to peek at action bulk. An absent or non-numeric stamp is a broken
    upstream contract and raises rather than silently classifying the trajectory as failing.
    """

    key: str
    threshold: float = 0.0

    def __call__(self, trajectory: Trajectory) -> bool:
        value = trajectory.transitions[-1].algo_extra.get(self.key) if trajectory.transitions else None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"buffer gate needs terminal algo_extra[{self.key!r}] on trajectory {trajectory.id!r}, "
                f"got {value!r}; the owning stage should have stamped it before replay"
            )
        return float(value) > self.threshold


def _declared_gate(predicate: object, *, argument: str) -> MetadataGate:
    """Reject anything but a `MetadataGate` at the buffer's public filtering surface.

    The buffer is a metadata index (DESIGN §5): a filter may read one precomputed scalar stamp and
    nothing else. A bare callable could inspect the full `Trajectory`, including bulk, so it is
    refused rather than trusted by convention.
    """
    if not isinstance(predicate, MetadataGate):
        raise TypeError(
            f"{argument} must be a MetadataGate over a precomputed algo_extra scalar, got "
            f"{type(predicate).__name__}; content-dependent filtering belongs in a reward-stage provider"
        )
    return predicate


class ExperienceBuffer:
    # Methods whose returned items leave the data plane for good: the actor host releases every bulk
    # lease they carry at its owning store (a refcount decrement, no fetch) before the metadata
    # returns — DESIGN §5 "evict releases the bulk lease", declared by the buffer, run by the host.
    lease_releasing_output_methods: frozenset[str] = frozenset({"drain"})

    def __init__(self, staleness: int = 0, exact_batch_size: int | None = None) -> None:
        self.staleness = staleness  # async knob (§2.2/§5); v0 sync ⇒ unused
        if exact_batch_size is not None and (
            isinstance(exact_batch_size, bool) or not isinstance(exact_batch_size, int) or exact_batch_size < 1
        ):
            raise ValueError(f"exact_batch_size must be a positive integer when set, got {exact_batch_size!r}")
        self.exact_batch_size = exact_batch_size
        self._items: list[Trajectory] = []

    def insert(self, trajectories: list[Trajectory]) -> None:
        # v0: store whole trajectories. TODO(§5): store metadata only + bulk lease.
        self._items.extend(t for t in trajectories if t.status != Status.ABORTED)

    # convenience alias matching the §2.7 naive surface (buffer.put_samples)
    def put_samples(self, trajectories: list[Trajectory]) -> None:
        self.insert(trajectories)

    def select(self, size: int, where: MetadataGate | None = None, staleness: int | None = None) -> list[Trajectory]:
        """Assemble a training batch. v0: FIFO + optional declared gate.

        ``where`` is a `MetadataGate` or nothing: the only filtering the buffer performs compares a
        precomputed metadata scalar, so no caller can make it peek at trajectory content or bulk.
        TODO(v1+, §5/§2.2): drop entries older than `staleness` weight versions (async/off-policy);
        v0 is sync/on-policy so `staleness` is accepted but ignored. Also curriculum, on metadata.
        """
        if where is not None:
            where = _declared_gate(where, argument="where")
        pool = [t for t in self._items if (where is None or where(t))]
        batch = pool[:size]
        # Drop exactly the batched items, keeping everything else (filtered-out items
        # AND filtered-but-not-batched items) in original order. Identity, not __eq__:
        # a Trajectory holds tensors, and tensor __eq__ returns a tensor (ambiguous as
        # bool). DESIGN §5: the buffer holds object references — identity is the right
        # equality predicate here.
        batch_ids = {id(t) for t in batch}
        self._items = [t for t in self._items if id(t) not in batch_ids]
        return batch

    # alias matching §2.7 (buffer.get_samples)
    def get_samples(self, size: int) -> list[Trajectory]:
        return self.select(size)

    def count(self, gate: MetadataGate) -> int:
        """Number of buffered trajectories whose stamped metadata passes ``gate`` (a `MetadataGate`)."""
        gate = _declared_gate(gate, argument="gate")
        return sum(1 for t in self._items if gate(t))

    def select_prioritized(self, size: int, gate: MetadataGate) -> list[Trajectory]:
        """Assemble ``size`` items: whole groups passing ``gate`` first, then whole groups as filler.

        Groups keep first-seen order inside each class and members keep FIFO order, so a pool that
        passes entirely and arrived group-contiguously yields exactly ``select(size)``. A group's
        verdict is its members' identical stamp, checked rather than assumed. Filler is taken only
        when the passing groups cannot fill ``size``; the caller decides whether that shortfall is
        acceptable.

        Raises:
            TypeError: ``gate`` is not a `MetadataGate`.
            ValueError: a trajectory lacks ``group_id``, a group's members disagree under ``gate``, or
                ``size`` would cut through a group — the batch must hold whole groups.
        """
        gate = _declared_gate(gate, argument="gate")
        groups: dict[str, list[Trajectory]] = {}  # first-seen group order, FIFO members
        for trajectory in self._items:
            if trajectory.group_id is None:
                raise ValueError(f"select_prioritized needs group_id on trajectory {trajectory.id!r}")
            groups.setdefault(trajectory.group_id, []).append(trajectory)
        passing: list[Trajectory] = []
        filler: list[Trajectory] = []
        for group_id, members in groups.items():
            verdicts = {gate(t) for t in members}
            if len(verdicts) != 1:
                raise ValueError(f"group {group_id!r} members disagree under {gate}; its stamp is a group fact")
            (passing if verdicts.pop() else filler).extend(members)
        pool = passing + filler
        batch, rest = pool[:size], pool[size:]
        if batch and rest and batch[-1].group_id == rest[0].group_id:
            raise ValueError(f"size {size} cuts through group {batch[-1].group_id!r}; the batch must hold whole groups")
        batch_ids = {id(t) for t in batch}
        self._items = [t for t in self._items if id(t) not in batch_ids]
        return batch

    def drain(self) -> list[Trajectory]:
        """Remove and return everything left in the buffer.

        The sync on-policy loop calls this once per step so over-collected trajectories never reach
        the next step's batch, where they would be trained against a policy that has moved. The
        returned items are terminal: ``lease_releasing_output_methods`` makes the actor host release
        their bulk leases at the owning store without moving any bulk, so the caller only awaits the
        result and drops its handle. Clearing the index alone would leak the leases.
        """
        dropped = self._items
        self._items = []
        return dropped

    def get_exact_samples(self, size: int | None = None) -> list[Trajectory]:
        """Select exactly ``size`` trajectories without draining an underfilled buffer."""
        if size is None:
            size = self.exact_batch_size
        if size is None:
            raise ValueError("get_exact_samples requires size or a bound exact_batch_size")
        if self.exact_batch_size is not None and size != self.exact_batch_size:
            raise ValueError(
                "get_exact_samples cannot override the role-bound exact_batch_size: "
                f"requested {size}, bound {self.exact_batch_size}"
            )
        if size < 0:
            raise ValueError(f"experience buffer size must be non-negative, got {size}")
        available = len(self._items)
        if available < size:
            raise RuntimeError(f"experience buffer returned {available} samples, expected exactly {size}")
        return self.select(size)

    def evict(self, predicate: MetadataGate) -> None:
        """Drop every buffered trajectory whose stamped metadata passes ``predicate`` (a `MetadataGate`)."""
        # TODO(§5): release bulk lease on evict; spill HBM→host→NVMe for kept-but-cold
        predicate = _declared_gate(predicate, argument="predicate")
        self._items = [t for t in self._items if not predicate(t)]

    def __len__(self) -> int:
        return len(self._items)
