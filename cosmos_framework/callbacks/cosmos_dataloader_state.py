# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint / resume callbacks for ``CosmosDataLoader``.

Two public classes:

* ``CosmosDataLoaderStateCallback`` — for a single ``CosmosDataLoader`` whose
  distributor is a ``MapDistributor``.  Saves per-worker ``(epoch, index)`` to
  the DCP checkpoint and, on resume, sets ``COSMOS_DL_STATE_*`` env vars so
  that ``MapDistributor.stream`` fast-forwards each worker to the saved
  position.

* ``JointCosmosDataLoaderStateCallback`` — for ``JointCosmosDataLoader``.
  Persists the outer ``global_id`` (dataset-selection sequence cursor) plus
  inner per-dataset per-worker state via one ``CosmosDataLoaderStateCallback``
  per inner loader.

Usage (single loader)::

    exp["trainer"]["callbacks"]["dataloader_state"] = CosmosDataLoaderStateCallback()

Usage (joint loader)::

    joint_loader = JointCosmosDataLoader(dataloaders={...}, seed=42)
    exp["dataloader_train"] = joint_loader
    exp["trainer"]["callbacks"]["dataloader_state"] = JointCosmosDataLoaderStateCallback(
        outer_loader=joint_loader,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


@dataclass
class _WorkerState:
    epoch: int = 0
    index: int = 0


class CosmosDataLoaderStateCallback(Callback):
    """Checkpoint/resume for a single ``CosmosDataLoader(MapDistributor)``.

    Tracks the highest-seen ``(epoch, index)`` per worker from batch metadata
    fields ``sample_worker_id``, ``sample_epoch``, ``sample_index`` (injected
    by ``MapDistributor``).

    On ``state_dict()`` the per-worker positions are serialised into the DCP
    checkpoint (``checkpoint_component = "dataloader"``).

    On ``load_state_dict()`` the positions are written to env vars::

        COSMOS_DL_STATE_WORKER_{id}_EPOCH
        COSMOS_DL_STATE_WORKER_{id}_INDEX

    (or ``COSMOS_DL_STATE_{name}_WORKER_{id}_*`` when ``name`` is set, for
    multi-loader namespacing).  ``MapDistributor.stream`` pops these on first
    iteration and resumes from ``index + 1``.
    """

    checkpoint_component: str = "dataloader"

    def __init__(self, name: str = "", distributor_type: str | None = None) -> None:
        # distributor_type is accepted but unused — it exists only so that Hydra
        # struct-merging over the legacy DataLoaderStateCallback entry (which
        # carries distributor_type="${data_setting.distributor_type}") does not
        # raise an unexpected-keyword-argument error at instantiation time.
        super().__init__()
        self.name = name
        self.config: Any = None
        self.state: dict[int, _WorkerState] = {}

    @property
    def _env_prefix(self) -> str:
        return f"COSMOS_DL_STATE_{self.name}_" if self.name else "COSMOS_DL_STATE_"

    def _update_state_from_batch(self, data_batch: dict[str, torch.Tensor]) -> None:
        if "sample_worker_id" not in data_batch:
            return  # IterableDistributor / no position metadata
        worker_ids = data_batch["sample_worker_id"].tolist()
        epochs = data_batch["sample_epoch"].tolist()
        indices = data_batch["sample_index"].tolist()
        for worker_id, epoch, index in zip(worker_ids, epochs, indices, strict=True):
            cur = self.state.get(worker_id)
            if cur is None:
                self.state[worker_id] = _WorkerState(epoch=epoch, index=index)
            elif epoch > cur.epoch or (epoch == cur.epoch and index > cur.index):
                self.state[worker_id] = _WorkerState(epoch=epoch, index=index)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.config and iteration % self.config.trainer.logging_iter == 0:
            msg = "\n"
            for wid, s in self.state.items():
                msg += f"worker {wid}: epoch={s.epoch}, index={s.index}\n"
            log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[int, dict[str, int]]:
        result: dict[int, dict[str, int]] = {}
        for worker_id, s in self.state.items():
            result[worker_id] = {"epoch": s.epoch, "index": s.index}
            log.info(f"Saved CosmosDataLoader state for worker {worker_id}: epoch={s.epoch}, index={s.index}")
        return result

    def load_state_dict(self, state_dict: dict[int, dict[str, int]]) -> None:
        if not state_dict:
            log.info("No CosmosDataLoader state found in checkpoint")
            return

        pfx = self._env_prefix
        self.state = {}
        for worker_id, per_worker in state_dict.items():
            epoch = per_worker["epoch"]
            index = per_worker["index"]
            self.state[worker_id] = _WorkerState(epoch=epoch, index=index)
            os.environ[f"{pfx}WORKER_{worker_id}_EPOCH"] = str(epoch)
            os.environ[f"{pfx}WORKER_{worker_id}_INDEX"] = str(index)
            log.info(f"Loaded CosmosDataLoader state for worker {worker_id}: epoch={epoch}, index={index}")


class JointCosmosDataLoaderStateCallback(Callback):
    """Checkpoint/resume for ``JointCosmosDataLoader``.

    Manages two levels of state in a single DCP checkpoint entry:

    1. **Outer** ``global_id`` — how many batches the outer loader has yielded.
       Restored via ``outer_loader.set_start_iteration(global_id)`` so the
       deterministic dataset-selection sequence resumes from the right step.

    2. **Inner** per-dataset, per-worker ``(epoch, index)`` — one
       ``CosmosDataLoaderStateCallback`` per inner loader, keyed by name.

    The ``checkpoint_component = "dataloader"`` class attribute ensures the DCP
    checkpointer's ``_DataloaderWrapper`` discovers exactly this callback.  Do
    **not** also register standalone ``CosmosDataLoaderStateCallback`` instances
    for the inner loaders — this class already handles them all.
    """

    checkpoint_component: str = "dataloader"

    def __init__(self, outer_loader: Any) -> None:
        super().__init__()
        self._outer = outer_loader
        self._inner: dict[str, CosmosDataLoaderStateCallback] = {
            name: CosmosDataLoaderStateCallback(name=name)
            for name in outer_loader._names
        }
        self.config: Any = None

    def _update_state_from_batch(self, batch: dict) -> None:
        name = batch.get("dataset_name")
        if name in self._inner:
            self._inner[name]._update_state_from_batch(batch)

    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict,
        output_batch: dict,
        loss: Any,
        iteration: int = 0,
    ) -> None:
        self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict,
        output_batch: dict,
        loss: Any,
        iteration: int = 0,
    ) -> None:
        if self.config and iteration % self.config.trainer.logging_iter == 0:
            msg = f"\nJointCosmosDataLoader global_id={self._outer._global_id}\n"
            for name, cb in self._inner.items():
                for wid, s in cb.state.items():
                    msg += f"  [{name}] worker {wid}: epoch={s.epoch}, index={s.index}\n"
            log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict:
        return {
            "global_id": self._outer._global_id,
            **{name: cb.state_dict() for name, cb in self._inner.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        global_id = state.get("global_id", 0)
        self._outer.set_start_iteration(global_id)
        log.info(f"JointCosmosDataLoaderStateCallback: resumed outer global_id={global_id}")
        for name, cb in self._inner.items():
            if name in state:
                cb.load_state_dict(state[name])
