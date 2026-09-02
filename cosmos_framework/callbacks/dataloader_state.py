# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback

LANCE_VLM_RESUME_FORMAT = "lance_vlm_cursor_v1"
LANCE_VLM_RESUME_STATE_KEY = "_lance_vlm_resume_state"
LANCE_VLM_RESUME_WORKER_ENV_PREFIX = "LANCE_VLM_RESUME_STATE_WORKER_"


@dataclass
class NoReplaceShardlistState:
    epoch: int = 0
    index: int = 0


class DataLoaderStateCallback(Callback):
    checkpoint_component: str = "dataloader"

    def __init__(
        self,
        distributor_type: str | None = None,
    ) -> None:
        super().__init__()
        self.distributor_type = distributor_type
        self.config: Any = None
        self.state: dict[int, NoReplaceShardlistState] = {}
        self.lance_state: dict[int, dict[str, Any]] = {}
        self._pending_lance_state: dict[str, Any] | None = None
        self.verbose = True

    def on_training_step_batch_start(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        """Capture and remove the Lance worker cursor for no-replacement sampling."""
        del model, iteration
        if self.distributor_type != "no_replace":
            return
        lance_state = data_batch.pop(LANCE_VLM_RESUME_STATE_KEY, None)
        if lance_state is not None and not isinstance(lance_state, dict):
            raise TypeError(f"{LANCE_VLM_RESUME_STATE_KEY} must be a dict, got {type(lance_state).__name__}.")
        self._pending_lance_state = lance_state

    def _update_lance_state(self, update: dict[str, Any]) -> None:
        if update.get("format") != LANCE_VLM_RESUME_FORMAT:
            raise ValueError(f"Unsupported Lance VLM resume state format {update.get('format')!r}.")
        worker_id = update.get("worker_id")
        draw_count = update.get("draw_count")
        fingerprint = update.get("fingerprint")
        source_cursors = update.get("source_cursors")
        if not isinstance(worker_id, int) or worker_id < 0:
            raise ValueError(f"Lance VLM resume worker_id must be non-negative, got {worker_id!r}.")
        if not isinstance(draw_count, int) or draw_count < 0:
            raise ValueError(f"Lance VLM resume draw_count must be non-negative, got {draw_count!r}.")
        if not isinstance(fingerprint, str) or not fingerprint or not isinstance(source_cursors, dict):
            raise TypeError("Lance VLM resume fingerprint must be non-empty and source_cursors must be a dictionary.")
        if any(
            not isinstance(name, str) or not isinstance(cursor, int) or cursor < 0
            for name, cursor in source_cursors.items()
        ):
            raise ValueError("Lance VLM source cursors must map source names to non-negative integers.")

        saved = self.lance_state.get(worker_id)
        previous_current = saved["current"] if saved is not None else None
        if previous_current is not None:
            if previous_current["fingerprint"] != fingerprint:
                raise ValueError(f"Lance VLM resume identity changed for worker {worker_id}.")
            if draw_count < previous_current["draw_count"]:
                raise ValueError(f"Lance VLM draw count moved backwards for worker {worker_id}.")
            previous_cursors = previous_current["source_cursors"]
            for source_name, cursor in source_cursors.items():
                previous_cursor = previous_cursors.get(source_name)
                if previous_cursor is not None and cursor < previous_cursor:
                    raise ValueError(f"Lance VLM source cursor {source_name!r} moved backwards for worker {worker_id}.")
            merged_cursors = {**previous_cursors, **source_cursors}
        else:
            merged_cursors = dict(source_cursors)
        current = {
            "format": LANCE_VLM_RESUME_FORMAT,
            "worker_id": worker_id,
            "draw_count": draw_count,
            "fingerprint": fingerprint,
            "source_cursors": merged_cursors,
        }
        self.lance_state[worker_id] = {"previous": previous_current, "current": current}

    def _update_state_from_batch(self, data_batch: dict[str, torch.Tensor]) -> None:
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

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type == "no_replace":
            if self._pending_lance_state is not None:
                self._update_lance_state(self._pending_lance_state)
                self._pending_lance_state = None
            else:
                self._update_state_from_batch(data_batch)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.distributor_type == "no_replace":
            if self.verbose:
                if iteration % self.config.trainer.logging_iter == 0:
                    msg = "\n"
                    if self.lance_state:
                        for worker_id, state_pair in self.lance_state.items():
                            msg += f"worker {worker_id}: draw_count={state_pair['current']['draw_count']}\n"
                    else:
                        for wid, state in self.state.items():
                            msg += f"worker {wid}: epoch={state.epoch}, index={state.index}\n"
                    log.info(msg)

    def has_checkpoint_state(self) -> bool:
        return self.distributor_type == "no_replace"

    def state_dict(self) -> dict[Any, Any]:
        if self.distributor_type != "no_replace":
            return {}

        if self.lance_state:
            return {"format": LANCE_VLM_RESUME_FORMAT, "workers": self.lance_state}

        state_dict: dict[int, dict[str, int]] = {}
        for worker_id, per_worker_state in self.state.items():
            state_dict[worker_id] = {"epoch": per_worker_state.epoch, "index": per_worker_state.index}
            log.info(
                f"Saved dataloader state for worker {worker_id}: "
                f"epoch={per_worker_state.epoch}, index={per_worker_state.index}"
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        if self.distributor_type != "no_replace":
            return

        if not state_dict:
            log.info("No dataloader state found in checkpoint")
            return

        if state_dict.get("format") == LANCE_VLM_RESUME_FORMAT:
            workers = state_dict.get("workers")
            if not isinstance(workers, dict):
                raise TypeError("Lance VLM checkpoint workers state must be a dictionary.")
            self.lance_state = {int(worker_id): state_pair for worker_id, state_pair in workers.items()}
            for worker_id, state_pair in self.lance_state.items():
                if not isinstance(state_pair, dict) or not isinstance(state_pair.get("current"), dict):
                    raise ValueError(f"Invalid Lance VLM resume state for worker {worker_id}.")
                os.environ[f"{LANCE_VLM_RESUME_WORKER_ENV_PREFIX}{worker_id}"] = json.dumps(
                    state_pair,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                log.info(
                    f"Loaded Lance VLM dataloader state for worker {worker_id}: "
                    f"draw_count={state_pair['current']['draw_count']}"
                )
            return

        self.state = {}
        for worker_id, per_worker_state in state_dict.items():
            epoch = per_worker_state["epoch"]
            index = per_worker_state["index"]
            self.state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)
            os.environ[f"NSL_STATE_WORKER_{worker_id}_EPOCH"] = str(epoch)
            os.environ[f"NSL_STATE_WORKER_{worker_id}_INDEX"] = str(index)
            log.info(f"Loaded no replace dataloader state for worker {worker_id}: epoch={epoch}, index={index}")
