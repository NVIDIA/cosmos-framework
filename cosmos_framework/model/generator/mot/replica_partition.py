# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Split work that every rank of an inference replica would otherwise repeat.

An inference replica is the ``cfgp x cp`` block of ranks that generates ONE sample together.
Every rank of it holds the same inputs, so anything outside the sharded transformer forward
-- decoding control media, VAE encode, VAE decode -- is by default computed identically on
all of them. This module turns that replicated work into partitioned work: the work is cut
into independent :class:`WorkUnit` s, each unit runs on exactly one rank, and the results are
either broadcast to every rank or collected on one.

This is the counterpart of ``vae_load_balance.offload_encode`` rather than a use of it.
``offload_encode`` balances ranks that hold DIFFERENT samples (training), by shipping raw
pixels from busy ranks to idle ones. Here every rank already holds every input, so nothing
but results ever crosses the wire, and ``plan_rebalance`` would find nothing to move: it sees
identical per-rank totals.

Planning needs no communication. Units are assigned round-robin in their shared input order,
so every rank computes the same owners and issues the same collectives in the same order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils import log


@dataclasses.dataclass(frozen=True)
class WorkUnit:
    """One independent piece of replicated work.

    Attributes:
        run: Produces the unit's tensor. Called only on the rank that owns the unit, and
            must depend on nothing that differs between ranks.
    """

    run: Callable[[], torch.Tensor]


def plan_partition(num_units: int, group_size: int) -> list[int]:
    """Assign units round-robin in their shared input order.

    Args:
        num_units: Number of units to assign.
        group_size: Ranks to spread the units over.

    Returns:
        The owning group rank of each unit, in input order.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if num_units < 0:
        raise ValueError(f"num_units must be nonnegative, got {num_units}.")
    return [index % group_size for index in range(num_units)]


def run_units_locally(
    units: Sequence[WorkUnit],
    on_collected: Callable[[int, torch.Tensor], None] | None = None,
) -> list[torch.Tensor | None]:
    """Run every unit on this rank: the fallback when there is no replica group to share with.

    Mirrors :meth:`ReplicaPartitioner.run`'s return contract, so callers need one code path.
    """
    results: list[torch.Tensor | None] = []
    for index, unit in enumerate(units):
        result = unit.run()  # [...]
        if on_collected is not None:
            on_collected(index, result)
            results.append(None)
        else:
            results.append(result)
    return results


class ReplicaPartitioner:
    """Runs a list of :class:`WorkUnit` s once across an inference replica instead of on every rank.

    Every rank of ``group`` must call :meth:`run` with the same units in the same order: the
    method enters collectives unconditionally.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        group_rank: int,
        group_size: int,
        device: torch.device | str,
    ) -> None:
        """
        Args:
            group: Process group spanning exactly the ranks that hold identical inputs.
            group_rank: This rank's index within ``group``.
            group_size: Ranks in ``group``.
            device: Where received results are allocated. Must be a device the group's
                backend can communicate (CUDA for NCCL, CPU for gloo).
        """
        self.group = group
        self.group_rank = group_rank
        self.group_size = group_size
        self.device = torch.device(device)

    @classmethod
    def from_parallel_dims(cls, parallel_dims: Any, device: torch.device | str) -> ReplicaPartitioner | None:
        """Build the partitioner for an inference replica, or ``None`` when there is none to share.

        The replica group is the ``lb`` mesh. ``OmniInference`` sizes it to ``cp * cfgp`` (see
        ``OmniInference._get_parallelism_config``), and because both the ``lb`` mesh and the
        ``(rest, cfgp, cp)`` overlay are row-major over contiguous rank blocks, an ``lb`` group of
        that size is exactly one replica and ``lb_rank`` equals the rank's index in its replica.
        Training never gets one: its ``lb`` mesh groups ranks holding DIFFERENT samples. Nor does
        an inference model whose ``lb`` group is any other size, e.g. one loaded directly with the
        training experiment's value: its ranks do not hold one shared sample.

        When the replica spans the whole world, the world group is reused instead of the ``lb``
        mesh's own, which saves initializing a second communicator over the same ranks.
        """
        if parallel_dims is None or not parallel_dims.enable_inference_mode or not parallel_dims.lb_enabled:
            return None
        replica_size = parallel_dims.cp_size * parallel_dims.cfgp_size
        if parallel_dims.lb_size != replica_size:
            log.info(
                f"Not sharing per-sample work across ranks: the lb group has {parallel_dims.lb_size} ranks, "
                f"but an inference replica has cp*cfgp={replica_size}."
            )
            return None
        group = (
            dist.group.WORLD if parallel_dims.lb_size == dist.get_world_size() else parallel_dims.lb_mesh.get_group()
        )
        return cls(
            group=group,
            group_rank=parallel_dims.lb_rank,
            group_size=parallel_dims.lb_size,
            device=device,
        )

    def run(
        self,
        units: Sequence[WorkUnit],
        *,
        collect_on: int | None = None,
        on_collected: Callable[[int, torch.Tensor], None] | None = None,
        label: str = "",
    ) -> list[torch.Tensor | None]:
        """Run each unit on one rank and deliver the results.

        Args:
            units: The work, identical on every rank.
            collect_on: ``None`` delivers every result to every rank. A group rank delivers
                them to that rank only; the other ranks get ``None`` entries.
            on_collected: Called as ``on_collected(index, tensor)`` in unit order on each rank
                that receives results, instead of retaining them. Lets a collector stream
                large results (e.g. decoded video) into host memory one at a time. Those
                entries of the returned list are ``None``.
            label: Names the run in the plan log line.

        Returns:
            One entry per unit: the result tensor where this rank receives it and
            ``on_collected`` is not given, otherwise ``None``. Results that crossed ranks, and
            every broadcast result, are on :attr:`device`; a collector's own results are
            returned as its units produced them.

        Raises:
            The owning rank's own exception if a unit fails there. Every other rank raises a
            ``RuntimeError`` naming the failed rank, so no rank is left waiting in a collective.
        """
        if collect_on is not None and not 0 <= collect_on < self.group_size:
            raise ValueError(f"collect_on must be a group rank in [0, {self.group_size}), got {collect_on}.")
        if self.group_size == 1:
            return run_units_locally(units, on_collected)

        owners = plan_partition(len(units), self.group_size)
        # A result this rank sends must be a contiguous tensor on the communication device. Making
        # it one right after its unit runs lets every owner's host-to-device copy overlap with the
        # other owners' instead of queueing behind the per-unit delivery. A collector's own results
        # never travel, so they stay as made.
        keeps_own_results = collect_on == self.group_rank

        local_results: dict[int, torch.Tensor] = {}
        local_error: Exception | None = None
        for index, unit in enumerate(units):
            if owners[index] != self.group_rank:
                continue
            try:
                result = unit.run()  # [...]
                if not keeps_own_results:
                    result = result.to(device=self.device).contiguous()  # [...]
                local_results[index] = result
            except Exception as error:  # Re-raised below, after every rank knows.
                local_error = error
                break

        # One small collective carries the error state and every result's shape and dtype:
        # whether to proceed and how to size receive buffers.
        local_report = {
            "error": None if local_error is None else f"{type(local_error).__name__}: {local_error}",
            "units": {index: (tuple(result.shape), result.dtype) for index, result in local_results.items()},
        }
        reports: list[Any] = [None] * self.group_size
        dist.all_gather_object(reports, local_report, group=self.group)
        failures = [(rank, report["error"]) for rank, report in enumerate(reports) if report["error"] is not None]
        if failures:
            if local_error is not None:
                raise local_error
            rank, message = failures[0]
            raise RuntimeError(f"Replica rank {rank} failed while running {label or 'partitioned'} work: {message}")

        unit_meta: dict[int, tuple[tuple[int, ...], torch.dtype]] = {}
        for report in reports:
            unit_meta.update(report["units"])
        log.debug(
            f"[replica partition{f' {label}' if label else ''}] {len(units)} units over {self.group_size} ranks; "
            f"round-robin units/rank {[owners.count(rank) for rank in range(self.group_size)]}"
        )

        delivery = _Delivery(len(units), on_collected)
        if collect_on is None:
            self._broadcast_all(owners, unit_meta, local_results, delivery)
        else:
            self._collect(collect_on, owners, unit_meta, local_results, delivery)
        if on_collected is not None:
            # A callback can fail on one rank only (e.g. a host allocation on the collector).
            # Delivery kept every transfer matched regardless; now every rank learns the outcome,
            # so none returns normally while a peer raises.
            self._raise_if_any_failed(delivery.error, label)
        return delivery.results

    def _raise_if_any_failed(self, local_error: Exception | None, label: str) -> None:
        errors: list[Any] = [None] * self.group_size
        dist.all_gather_object(
            errors, None if local_error is None else f"{type(local_error).__name__}: {local_error}", group=self.group
        )
        if local_error is not None:
            raise local_error
        for rank, message in enumerate(errors):
            if message is not None:
                raise RuntimeError(
                    f"Replica rank {rank} failed while collecting {label or 'partitioned'} results: {message}"
                )

    def _receive_buffer(self, meta: tuple[tuple[int, ...], torch.dtype]) -> torch.Tensor:  # returns [...]
        shape, dtype = meta
        return torch.empty(shape, dtype=dtype, device=self.device)  # [...]

    def _broadcast_all(
        self,
        owners: list[int],
        unit_meta: dict[int, tuple[tuple[int, ...], torch.dtype]],
        local_results: dict[int, torch.Tensor],
        delivery: _Delivery,
    ) -> None:
        for index, owner in enumerate(owners):
            tensor = local_results.pop(index) if owner == self.group_rank else self._receive_buffer(unit_meta[index])
            dist.broadcast(tensor, group=self.group, group_src=owner)
            delivery.deliver(index, tensor)

    def _collect(
        self,
        collect_on: int,
        owners: list[int],
        unit_meta: dict[int, tuple[tuple[int, ...], torch.dtype]],
        local_results: dict[int, torch.Tensor],
        delivery: _Delivery,
    ) -> None:
        # Owners send and the collector receives in unit order, so every pending send is matched
        # by a receive the collector will post without first waiting on a later unit.
        for index, owner in enumerate(owners):
            if owner == collect_on:
                if self.group_rank == collect_on:
                    delivery.deliver(index, local_results.pop(index))
            elif self.group_rank == owner:
                dist.send(local_results.pop(index), group=self.group, group_dst=collect_on)
            elif self.group_rank == collect_on:
                tensor = self._receive_buffer(unit_meta[index])
                dist.recv(tensor, group=self.group, group_src=owner)
                delivery.deliver(index, tensor)


class _Delivery:
    """Where a run's results go on one rank: retained, or streamed through ``on_collected``.

    A failing callback must not stop the transfers, since every peer's send or broadcast still
    needs its match, so the first error is recorded, later callbacks are skipped, and the caller
    raises once delivery is complete.
    """

    def __init__(self, num_units: int, on_collected: Callable[[int, torch.Tensor], None] | None) -> None:
        self.results: list[torch.Tensor | None] = [None] * num_units
        self.error: Exception | None = None
        self._on_collected = on_collected

    def deliver(self, index: int, tensor: torch.Tensor) -> None:
        if self._on_collected is None:
            self.results[index] = tensor
        elif self.error is None:
            try:
                self._on_collected(index, tensor)
            except Exception as error:  # Re-raised by ReplicaPartitioner.run once delivery completes.
                self.error = error
