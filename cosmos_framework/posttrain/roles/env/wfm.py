# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transactional source adapter for synchronous online WFM training."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from cosmos_framework.posttrain.schema import WFM_CONDITIONING_KEY, ObsBundle, WFMConditioningInput
from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0, rank_identical
from cosmos_framework.posttrain.utils.wfm import make_t2i_conditioning_input, make_t2v_conditioning_input

_NO_PENDING_BATCH = object()
_INT64_MAX = (1 << 63) - 1


def _non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _INT64_MAX:
        raise ValueError(f"{name} must be a non-negative int64, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class WFMSeedContract:
    """Immutable identity for the rollout seed sequence bound to an exact resume."""

    base_seed: int
    clock: Literal["source_row", "optimizer_step"]
    group_size: int
    stride: int = 10_000
    version: int = 1

    def __post_init__(self) -> None:
        _non_negative_int("WFM seed-contract base_seed", self.base_seed)
        if self.clock not in ("source_row", "optimizer_step"):
            raise ValueError(f"WFM seed-contract clock must be 'source_row' or 'optimizer_step', got {self.clock!r}")
        if not isinstance(self.group_size, int) or isinstance(self.group_size, bool) or self.group_size < 1:
            raise ValueError(f"WFM seed-contract group_size must be an integer >= 1, got {self.group_size!r}")
        if self.group_size > _INT64_MAX:
            raise ValueError(f"WFM seed-contract group_size exceeds int64, got {self.group_size!r}")
        if (
            not isinstance(self.stride, int)
            or isinstance(self.stride, bool)
            or self.stride < self.group_size
            or self.stride > _INT64_MAX
        ):
            raise ValueError(
                f"WFM seed-contract stride must be an int64 >= group_size ({self.group_size}), got {self.stride!r}"
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version != 1:
            raise ValueError(f"unsupported WFM seed-contract version {self.version!r}")
        if self.base_seed + self.group_size - 1 > _INT64_MAX:
            raise ValueError("WFM seed-contract initial group exceeds the non-negative int64 seed range")

    def state_dict(self) -> dict[str, int | str]:
        """Return the canonical identity stored inside version-2 source state."""
        return {
            "version": self.version,
            "base_seed": self.base_seed,
            "clock": self.clock,
            "stride": self.stride,
            "group_size": self.group_size,
        }


def _seed_range(
    contract: WFMSeedContract,
    *,
    source_row: int,
    optimizer_step: int,
) -> range:
    source_row = _non_negative_int("WFM seed source_row", source_row)
    optimizer_step = _non_negative_int("WFM seed optimizer_step", optimizer_step)
    clock_value = source_row if contract.clock == "source_row" else optimizer_step
    first_seed = contract.base_seed + clock_value * contract.stride
    last_seed = first_seed + contract.group_size - 1
    if last_seed > _INT64_MAX:
        raise ValueError(
            "WFM rollout seed group exceeds the non-negative int64 range: "
            f"first={first_seed}, group_size={contract.group_size}"
        )
    return range(first_seed, last_seed + 1)


def wfm_seed_set(
    contract: WFMSeedContract,
    *,
    source_row: int,
    optimizer_step: int,
) -> range:
    """Return the complete, ordered seed set for one global rollout group."""
    return _seed_range(contract, source_row=source_row, optimizer_step=optimizer_step)


def wfm_first_seed(
    contract: WFMSeedContract,
    *,
    source_row: int,
    optimizer_step: int,
    process_rank: int,
    world_size: int,
) -> int:
    """Return the first seed in one rank's contiguous partition of the global group."""
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError(f"WFM seed world_size must be an integer >= 1, got {world_size!r}")
    if not isinstance(process_rank, int) or isinstance(process_rank, bool) or not 0 <= process_rank < world_size:
        raise ValueError(f"WFM seed process_rank must be in [0, {world_size}), got {process_rank!r}")
    if contract.group_size % world_size:
        raise ValueError(f"WFM seed group_size={contract.group_size} must be divisible by world_size={world_size}")
    seeds = _seed_range(contract, source_row=source_row, optimizer_step=optimizer_step)
    local_group_size = contract.group_size // world_size
    return seeds.start + process_rank * local_group_size


class WFMDataEnv:
    """Expose a streaming i4 video batch behind a caller-owned commit boundary.

    ``get_samples`` peeks one row and ``stage`` seals it into the transient
    accumulation window. ``commit_window`` is the only API that advances the durable
    boundary. The later control loop must call it only after an applied optimizer update.

    This is deliberately not the generic map-style ``IndexedCursor`` used by
    ``DatasetEnv`` and preference data. WFM loaders are streaming/prefetched, so
    the persisted cursor is a structural iteration, not an exact WebDataset-worker
    snapshot; exact numerical parity uses frozen source artifacts.
    """

    def __init__(
        self,
        dataloader: Any,
        *,
        start_iteration: int = 0,
        grad_accum: int = 1,
        modality: Literal["image", "video"] = "video",
        seed_contract: WFMSeedContract | None = None,
    ) -> None:
        if not isinstance(start_iteration, int) or isinstance(start_iteration, bool) or start_iteration < 0:
            raise ValueError(f"WFM source start_iteration must be >= 0, got {start_iteration}")
        if not isinstance(grad_accum, int) or isinstance(grad_accum, bool) or grad_accum < 1:
            raise ValueError(f"WFM source grad_accum must be an integer >= 1, got {grad_accum!r}")
        if modality not in ("image", "video"):
            raise ValueError(f"WFM source modality must be 'image' or 'video', got {modality!r}")
        if seed_contract is not None and not isinstance(seed_contract, WFMSeedContract):
            raise TypeError(f"WFM source seed_contract must be WFMSeedContract or None, got {seed_contract!r}")
        if start_iteration % grad_accum:
            raise ValueError(
                f"WFM source start_iteration={start_iteration} is not an optimizer boundary for grad_accum={grad_accum}"
            )
        self.dataloader = dataloader
        self.grad_accum = grad_accum
        self.modality = modality
        self.seed_contract = seed_contract
        self._committed_rows = start_iteration
        self._iterator: Iterator[dict[str, Any]] | None = None
        self._pending_source_id: str | None = None
        self._pending_batch: object = _NO_PENDING_BATCH
        self._pending: list[ObsBundle] | None = None
        self._staged_rows = 0
        fingerprint = getattr(self.dataloader, "cursor_fingerprint", None)
        source_fingerprint = fingerprint() if callable(fingerprint) else ""
        if not isinstance(source_fingerprint, str) or (callable(fingerprint) and not source_fingerprint):
            raise ValueError("WFM dataloader cursor_fingerprint() must return a non-empty string")
        self._source_fingerprint = source_fingerprint
        self._set_start_iteration(self._committed_rows)

    def _set_start_iteration(self, iteration: int) -> None:
        setter = getattr(self.dataloader, "set_start_iteration", None)
        if callable(setter):
            setter(iteration)
        elif iteration:
            raise ValueError(
                "the selected WFM dataloader cannot restore a structural iteration; "
                "resume from step 0 or provide set_start_iteration()"
            )
        self._iterator = None
        self._pending_source_id = None
        self._pending_batch = _NO_PENDING_BATCH
        self._pending = None
        self._staged_rows = 0

    def _next_batch(self) -> dict[str, Any]:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        # Acquisition opens the live-process transaction. Cache even a malformed object so a
        # validation/conversion retry cannot advance the iterator under the same source ID.
        if self._pending_batch is _NO_PENDING_BATCH:
            self._pending_batch = next(self._iterator)
        batch = self._pending_batch
        if not isinstance(batch, dict):
            raise TypeError(f"WFM dataloader must yield dict batches, got {type(batch).__name__}")
        media_key = "images" if self.modality == "image" else "video"
        media = batch.get(media_key)
        if not isinstance(media, list) or len(media) != 1:
            size = len(media) if isinstance(media, list) else "n/a"
            raise ValueError(
                f"online WFM requires exactly one {self.modality} source row per microbatch; "
                f"got {media_key} batch {type(media).__name__} with size {size}"
            )
        return batch

    def get_samples(self, size: int) -> list[ObsBundle]:
        """Peek one pending source batch; ``size`` is fixed to one in v1."""
        if not isinstance(size, int) or isinstance(size, bool) or size != 1:
            raise ValueError(f"online WFM v1 expects one conditioning input per rank, got size={size}")
        rank_identical(
            (
                self._committed_rows,
                self._staged_rows,
                self.grad_accum,
                self._source_fingerprint,
                self.seed_contract,
                self._pending_source_id,
                self._pending is not None,
            ),
            "online WFM source transaction signature",
        )
        if self._pending is None:
            if self._pending_source_id is None:
                # Reserve on every rank before rank 0 fetches. This also makes abort/restore
                # behavior rank-consistent after a converged rank-0 preparation failure.
                row = self._committed_rows + self._staged_rows
                self._pending_source_id = f"wfm_source_{row:08d}"
            source_id = self._pending_source_id

            def prepare() -> WFMConditioningInput:
                if self.modality == "image":
                    return make_t2i_conditioning_input(
                        self._next_batch(),
                        source_id=source_id,
                        source_fingerprint=self._source_fingerprint,
                    )
                return make_t2v_conditioning_input(
                    self._next_batch(),
                    source_id=source_id,
                    source_fingerprint=self._source_fingerprint,
                )

            conditioning = broadcast_from_rank0(
                prepare,
                context="online WFM source fetch",
            )
            self._pending = [
                ObsBundle(
                    representations={WFM_CONDITIONING_KEY: conditioning},
                    modality_tags=("text", self.modality),
                    id=conditioning.id,
                )
            ]
            # The typed conditioning owns a recursive snapshot, so the raw loader row need not
            # remain live after successful conversion and broadcast.
            self._pending_batch = _NO_PENDING_BATCH
        return self._pending

    def stage(self) -> None:
        """Seal the current row after one successful microbatch backward."""
        if self._pending is None:
            raise RuntimeError("cannot stage the WFM source without a pending row")
        if self._staged_rows >= self.grad_accum:
            raise RuntimeError(f"WFM source already staged its complete {self.grad_accum}-row window")
        self._staged_rows += 1
        self._pending_source_id = None
        self._pending_batch = _NO_PENDING_BATCH
        self._pending = None

    def commit_window(self, expected_rows: int) -> None:
        """Atomically advance one complete optimizer window."""
        if not isinstance(expected_rows, int) or isinstance(expected_rows, bool):
            raise ValueError(f"WFM source expected_rows must be an integer, got {expected_rows!r}")
        if self._pending is not None or self._pending_source_id is not None:
            raise RuntimeError("cannot commit a WFM source window with a pending row")
        if expected_rows != self.grad_accum or self._staged_rows != expected_rows:
            raise RuntimeError(
                f"WFM source commit needs exactly grad_accum={self.grad_accum} staged rows, "
                f"got expected_rows={expected_rows}, staged={self._staged_rows}"
            )
        self._committed_rows += expected_rows
        self._staged_rows = 0

    def commit(self) -> None:
        """Advance the one-row compatibility path after an applied optimizer update."""
        if self.grad_accum != 1:
            raise RuntimeError("commit() is only valid for grad_accum=1; use stage() and commit_window()")
        self.stage()
        self.commit_window(1)

    def abort(self) -> None:
        """Retain the reserved source transaction so an explicit retry sees the same row."""
        if self._pending_source_id is None:
            raise RuntimeError("cannot abort the WFM source without a pending source transaction")

    def abort_window(self) -> None:
        """Discard transient rows and rewind to the last durable source boundary."""
        if self._pending_source_id is None and not self._staged_rows:
            raise RuntimeError("cannot abort the WFM source without a pending or staged window")
        self._set_start_iteration(self._committed_rows)

    def state_dict(self) -> dict[str, Any]:
        """Return the quiescent optimizer-window cursor; transient work is excluded."""
        if self._pending_source_id is not None or self._pending is not None or self._staged_rows:
            raise RuntimeError("cannot checkpoint the WFM source inside an accumulation window")
        state: dict[str, Any] = {
            "version": 2 if self.seed_contract is not None else 1,
            "resume_kind": "optimizer_window_rows",
            "committed_rows": self._committed_rows,
            "optimizer_step": self._committed_rows // self.grad_accum,
            "grad_accum": self.grad_accum,
            "source_fingerprint": self._source_fingerprint,
        }
        if self.seed_contract is not None:
            state["seed_contract"] = self.seed_contract.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore a structural cursor before any batch is requested."""
        if (
            self._pending_source_id is not None
            or self._pending_batch is not _NO_PENDING_BATCH
            or self._pending is not None
            or self._staged_rows
        ):
            raise RuntimeError("cannot restore the WFM source inside an accumulation window")
        version = state.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"unsupported WFM source state {state!r}")
        expected = {
            "version",
            "resume_kind",
            "committed_rows",
            "optimizer_step",
            "grad_accum",
            "source_fingerprint",
        }
        if version == 1:
            if self.seed_contract is not None:
                raise ValueError("WFM source version-1 state has no seed contract required by this runtime")
        elif version == 2:
            if self.seed_contract is None:
                raise ValueError("WFM source version-2 seed contract is not configured by this runtime")
            expected.add("seed_contract")
        else:
            raise ValueError(f"unsupported WFM source state {state!r}")
        if set(state) != expected:
            raise ValueError(f"invalid WFM source state fields: {sorted(state)}")
        if state["resume_kind"] != "optimizer_window_rows":
            raise ValueError(f"unsupported WFM source state {state!r}")
        if version == 2:
            contract_state = state["seed_contract"]
            if not isinstance(contract_state, dict) or set(contract_state) != {
                "version",
                "base_seed",
                "clock",
                "stride",
                "group_size",
            }:
                raise ValueError(f"invalid WFM source seed contract {contract_state!r}")
            checkpoint_contract = WFMSeedContract(
                version=contract_state["version"],
                base_seed=contract_state["base_seed"],
                clock=contract_state["clock"],
                stride=contract_state["stride"],
                group_size=contract_state["group_size"],
            )
            if checkpoint_contract != self.seed_contract:
                raise ValueError(
                    "WFM source seed contract differs from the checkpoint: "
                    f"checkpoint={checkpoint_contract!r}, runtime={self.seed_contract!r}"
                )
        if not isinstance(state["grad_accum"], int) or isinstance(state["grad_accum"], bool):
            raise ValueError(f"invalid WFM source grad_accum {state['grad_accum']!r}")
        if state["grad_accum"] != self.grad_accum:
            raise ValueError(
                f"WFM source grad_accum changed: checkpoint={state['grad_accum']!r}, runtime={self.grad_accum}"
            )
        if not isinstance(state["source_fingerprint"], str):
            raise ValueError(f"invalid WFM source fingerprint {state['source_fingerprint']!r}")
        if state["source_fingerprint"] != self._source_fingerprint:
            raise ValueError("WFM source fingerprint differs from the checkpoint")
        committed_rows = state["committed_rows"]
        if not isinstance(committed_rows, int) or isinstance(committed_rows, bool) or committed_rows < 0:
            raise ValueError(f"invalid WFM committed_rows {committed_rows!r}")
        if committed_rows % self.grad_accum:
            raise ValueError(
                f"WFM committed_rows={committed_rows} is not an optimizer boundary for grad_accum={self.grad_accum}"
            )
        optimizer_step = state["optimizer_step"]
        if not isinstance(optimizer_step, int) or isinstance(optimizer_step, bool) or optimizer_step < 0:
            raise ValueError(f"invalid WFM optimizer_step {optimizer_step!r}")
        if optimizer_step != committed_rows // self.grad_accum:
            raise ValueError("WFM source optimizer_step is inconsistent with committed_rows")
        # Reset the loader first. A loader that cannot restore must leave the durable
        # source boundary unchanged rather than publishing the rejected cursor.
        self._set_start_iteration(committed_rows)
        self._committed_rows = committed_rows
        return True
