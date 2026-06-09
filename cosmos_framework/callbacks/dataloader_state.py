# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


@dataclass
class NoReplaceShardlistState:
    epoch: int = 0
    index: int = 0


class DataLoaderStateCallback(Callback):
    checkpoint_component: str = "dataloader"

    def __init__(
        self,
        distributor_type: str | None = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.distributor_type = distributor_type
        self.name = name
        self.config: Any = None
        self.state: dict[int, NoReplaceShardlistState] = {}
        self.verbose = True

    def _update_state_from_batch(self, data_batch: dict[str, torch.Tensor]) -> None:
        if "sample_worker_id" not in data_batch:
            return  # batch has no position metadata (shuffle=False or iterable data_source)
        worker_ids = data_batch["sample_worker_id"].tolist()  # [B]
        epochs = data_batch["sample_epoch"].tolist()  # [B]
        indices = data_batch["sample_index"].tolist()  # [B]
        for worker_id, epoch, index in zip(worker_ids, epochs, indices, strict=True):
            if worker_id not in self.state:
                self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

            elif self.state[worker_id].epoch < epoch or (
                self.state[worker_id].index < index and self.state[worker_id].epoch == epoch
            ):
                self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

    _ACTIVE_DISTRIBUTOR_TYPES = ("no_replace", "data_packer")

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type in self._ACTIVE_DISTRIBUTOR_TYPES:
            self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type in self._ACTIVE_DISTRIBUTOR_TYPES:
            if self.verbose:
                if iteration % self.config.trainer.logging_iter == 0:
                    msg = "\n"
                    for wid, state in self.state.items():
                        msg += f"worker {wid}: epoch={state.epoch}, index={state.index}\n"
                    log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return self.distributor_type in self._ACTIVE_DISTRIBUTOR_TYPES

    def state_dict(self) -> dict[int, dict[str, int]]:
        if self.distributor_type not in self._ACTIVE_DISTRIBUTOR_TYPES:
            return {}

        state_dict: dict[int, dict[str, int]] = {}
        for worker_id, per_worker_state in self.state.items():
            state_dict[worker_id] = {"epoch": per_worker_state.epoch, "index": per_worker_state.index}
            log.info(
                f"Saved dataloader state for worker {worker_id}: "
                f"epoch={per_worker_state.epoch}, index={per_worker_state.index}"
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[int, dict[str, int]]) -> None:
        if self.distributor_type not in self._ACTIVE_DISTRIBUTOR_TYPES:
            return

        if not state_dict:
            log.info("No dataloader state found in checkpoint")
            return

        self.state = {}
        # Build env var prefix. For data_packer, namespacing avoids conflicts
        # when multiple DataPackerDataLoader instances share the same process
        # (e.g. inside JointDataPackerDataLoader). name="" → original format.
        _dp_pfx = f"DP_STATE_{self.name}_" if self.name else "DP_STATE_"
        for worker_id, per_worker_state in state_dict.items():
            epoch = per_worker_state["epoch"]
            index = per_worker_state["index"]
            self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)
            if self.distributor_type == "data_packer":
                os.environ[f"{_dp_pfx}WORKER_{worker_id}_EPOCH"] = str(epoch)
                os.environ[f"{_dp_pfx}WORKER_{worker_id}_INDEX"] = str(index)
                log.info(f"Loaded data_packer dataloader state for worker {worker_id}: epoch={epoch}, index={index}")
            else:
                os.environ[f"NSL_STATE_WORKER_{worker_id}_EPOCH"] = str(epoch)
                os.environ[f"NSL_STATE_WORKER_{worker_id}_INDEX"] = str(index)
                log.info(f"Loaded no_replace dataloader state for worker {worker_id}: epoch={epoch}, index={index}")


class JointDataLoaderStateCallback(Callback):
    """Checkpoint/resume state for ``JointDataPackerDataLoader``.

    Manages two levels of state in a single DCP checkpoint entry
    (``checkpoint_component = "dataloader"``):

    1. **Outer** ``global_id`` — the number of batches the outer loader has
       yielded.  Restored via ``outer_loader.set_start_iteration(global_id)``
       so the deterministic dataset-selection sequence resumes from the correct
       step.

    2. **Inner** per-dataset, per-worker ``(epoch, index)`` — one
       ``DataLoaderStateCallback`` per inner loader, keyed by the dataset name.
       Each inner callback sets namespaced env vars on ``load_state_dict`` so
       workers fast-forward to the saved sample position.

    Usage in experiment configs::

        joint_loader = JointDataPackerDataLoader(dataloaders={...}, seed=42)
        exp["dataloader_train"] = joint_loader
        exp["trainer"]["callbacks"]["dataloader_state"] = JointDataLoaderStateCallback(
            outer_loader=joint_loader,
            distributor_type="data_packer",
        )

    The ``checkpoint_component = "dataloader"`` class attribute ensures the DCP
    checkpointer's ``_DataloaderWrapper`` discovers exactly this callback (it
    picks the first matching callback).  Do **not** also register standalone
    ``DataLoaderStateCallback`` instances for the inner loaders — this class
    already handles them all.
    """

    checkpoint_component: str = "dataloader"

    def __init__(
        self,
        outer_loader: Any,
        distributor_type: str = "data_packer",
    ) -> None:
        super().__init__()
        self._outer = outer_loader
        self._inner: dict[str, DataLoaderStateCallback] = {
            name: DataLoaderStateCallback(distributor_type=distributor_type, name=name)
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
            msg = f"\nJointDataPackerDataLoader global_id={self._outer._global_id}\n"
            for name, cb in self._inner.items():
                for wid, state in cb.state.items():
                    msg += f"  [{name}] worker {wid}: epoch={state.epoch}, index={state.index}\n"
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
        log.info(f"JointDataLoaderStateCallback: resumed outer global_id={global_id}")
        for name, cb in self._inner.items():
            if name in state:
                cb.load_state_dict(state[name])
