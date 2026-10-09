# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import persistent_validation_enabled
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback

LANCE_VLM_RESUME_FORMAT = "lance_vlm_schedule_v2"
LANCE_VLM_RESUME_STATE_KEY = "_lance_vlm_resume_state"
LANCE_VLM_RESUME_WORKER_ENV_PREFIX = "LANCE_VLM_RESUME_STATE_WORKER_"
LANCE_VLM_VAL_RESUME_WORKER_ENV_PREFIX = "LANCE_VLM_VAL_RESUME_STATE_WORKER_"
LANCE_VLM_VAL_RESUME_BATCH_COUNT_ENV = "LANCE_VLM_VAL_RESUME_BATCH_COUNT"
VAL_STATE_KEY = "val"
_STATE_LABELS = {"train": "dataloader", "val": "validation dataloader"}


def lance_resume_worker_env_prefix(split: str) -> str:
    return LANCE_VLM_RESUME_WORKER_ENV_PREFIX if split == "train" else LANCE_VLM_VAL_RESUME_WORKER_ENV_PREFIX


def no_replace_resume_env_names(split: str, worker_id: int) -> tuple[str, str]:
    """Return the env vars that carry one worker's last consumed (epoch, index) for a split."""
    prefix = "NSL_STATE_WORKER" if split == "train" else "NSL_STATE_VAL_WORKER"
    return f"{prefix}_{worker_id}_EPOCH", f"{prefix}_{worker_id}_INDEX"


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
        self.val_state: dict[int, NoReplaceShardlistState] = {}
        self.lance_state: dict[int, dict[str, Any]] = {}
        self.val_lance_state: dict[int, dict[str, Any]] = {}
        self.val_lance_batches: int = 0
        self._pending_lance_state: dict[str, Any] | None = None
        self._pending_val_lance_state: dict[str, Any] | None = None
        self.verbose = True

    def _tracks_validation(self) -> bool:
        """Validation cursors only carry over when the trainer keeps one validation iterator alive."""
        return self.distributor_type == "no_replace" and persistent_validation_enabled(self.config.trainer)

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

    def _update_lance_state(self, update: dict[str, Any], split: str = "train") -> None:
        if update.get("format") != LANCE_VLM_RESUME_FORMAT:
            raise ValueError(f"Unsupported Lance VLM resume state format {update.get('format')!r}.")
        worker_id = update.get("worker_id")
        draw_count = update.get("draw_count")
        fingerprint = update.get("fingerprint")
        source_cursors = update.get("source_cursors")
        source_drops = update.get("source_drops")
        pool = update.get("pool")
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
        if not isinstance(source_drops, dict) or any(
            not isinstance(name, str) or type(count) is not int or count < 0 for name, count in source_drops.items()
        ):
            raise ValueError("Lance VLM source drops must map names to non-negative counts")
        if not isinstance(pool, list) or any(
            not isinstance(entry, list)
            or len(entry) != 5
            or not isinstance(entry[0], str)
            or any(type(value) is not int or value < 0 for value in entry[1:])
            or entry[4] < 1
            for entry in pool
        ):
            raise ValueError("Lance VLM pool entries must contain source, group, epoch, position and source cursor")

        states = self.lance_state if split == "train" else self.val_lance_state
        saved = states.get(worker_id)
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
            previous_drops = previous_current["source_drops"]
            if any(count < previous_drops.get(name, 0) for name, count in source_drops.items()):
                raise ValueError("Lance VLM source drop count moved backwards")
            merged_drops = {**previous_drops, **source_drops}
        else:
            merged_cursors = dict(source_cursors)
            merged_drops = dict(source_drops)
        current = {
            "format": LANCE_VLM_RESUME_FORMAT,
            "worker_id": worker_id,
            "draw_count": draw_count,
            "fingerprint": fingerprint,
            "source_cursors": merged_cursors,
            "source_drops": merged_drops,
            "pool": pool,
        }
        states[worker_id] = {"previous": previous_current, "current": current}

    @staticmethod
    def _update_state_from_batch(
        state: dict[int, NoReplaceShardlistState], data_batch: dict[str, torch.Tensor]
    ) -> None:
        worker_ids = data_batch["sample_worker_id"].tolist()  # [B]
        epochs = data_batch["sample_epoch"].tolist()  # [B]
        indices = data_batch["sample_index"].tolist()  # [B]
        for worker_id, epoch, index in zip(worker_ids, epochs, indices, strict=True):
            if worker_id not in state:
                state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

            elif state[worker_id].epoch < epoch or (state[worker_id].index < index and state[worker_id].epoch == epoch):
                state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)

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
                self._update_state_from_batch(self.state, data_batch)

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

    def on_validation_step_start(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        if self.distributor_type != "no_replace":
            return
        lance_state = data_batch.pop(LANCE_VLM_RESUME_STATE_KEY, None)
        if lance_state is not None and not isinstance(lance_state, dict):
            raise TypeError(f"{LANCE_VLM_RESUME_STATE_KEY} must be a dict, got {type(lance_state).__name__}.")
        self._pending_val_lance_state = lance_state if self._tracks_validation() else None

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self._tracks_validation():
            return
        # Lance batches carry a group-schedule cursor instead of a shardlist cursor.
        if self._pending_val_lance_state is not None:
            self._update_lance_state(self._pending_val_lance_state, "val")
            self.val_lance_batches += 1
            self._pending_val_lance_state = None
        elif "sample_worker_id" in data_batch:
            self._update_state_from_batch(self.val_state, data_batch)

    def has_checkpoint_state(self) -> bool:
        return self.distributor_type == "no_replace"

    def state_dict(self) -> dict[Any, Any]:
        if self.distributor_type != "no_replace":
            return {}

        if self.lance_state:
            state_dict: dict[Any, Any] = {"format": LANCE_VLM_RESUME_FORMAT, "workers": self.lance_state}
        else:
            state_dict = self._save_shardlist_state(self.state, "train")
        if self.val_lance_state:
            state_dict[VAL_STATE_KEY] = {
                "format": LANCE_VLM_RESUME_FORMAT,
                "workers": self.val_lance_state,
                "batch_count": self.val_lance_batches,
            }
        elif self.val_state:
            state_dict[VAL_STATE_KEY] = self._save_shardlist_state(self.val_state, "val")
        return state_dict

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        if self.distributor_type != "no_replace":
            return

        if not state_dict:
            log.info("No dataloader state found in checkpoint")
            return

        state_dict = dict(state_dict)
        val_state_dict = state_dict.pop(VAL_STATE_KEY, None)
        if val_state_dict is not None:
            if self._tracks_validation():
                if val_state_dict.get("format") == LANCE_VLM_RESUME_FORMAT:
                    batch_count = val_state_dict.get("batch_count")
                    if type(batch_count) is not int or batch_count < 0:
                        raise ValueError("Lance validation checkpoint requires a non-negative batch_count")
                    self.val_lance_state = self._load_lance_state(val_state_dict, "val")
                    self.val_lance_batches = batch_count
                    os.environ[LANCE_VLM_VAL_RESUME_BATCH_COUNT_ENV] = str(batch_count)
                elif "format" in val_state_dict:
                    raise ValueError(f"Unsupported validation sampler state {val_state_dict['format']!r}")
                else:
                    self.val_state = self._load_shardlist_state(val_state_dict, "val")
            else:
                log.info("Ignoring validation dataloader state because the validation iterator is not persistent")

        if state_dict.get("format") == LANCE_VLM_RESUME_FORMAT:
            self.lance_state = self._load_lance_state(state_dict, "train")
            if val_state_dict is None and self.config is not None and self._tracks_validation():
                log.warning("Checkpoint has no validation dataloader state; validation will start from epoch zero")
            return

        if "format" in state_dict:
            raise ValueError(f"Unsupported sampler state {state_dict['format']!r}; finite sampling requires a new run")
        self.state = self._load_shardlist_state(state_dict, "train")

    @staticmethod
    def _load_lance_state(state_dict: dict[str, Any], split: str) -> dict[int, dict[str, Any]]:
        workers = state_dict.get("workers")
        if not isinstance(workers, dict):
            raise TypeError("Lance VLM checkpoint workers state must be a dictionary.")
        states = {int(worker_id): state_pair for worker_id, state_pair in workers.items()}
        prefix = lance_resume_worker_env_prefix(split)
        for worker_id, state_pair in states.items():
            if not isinstance(state_pair, dict) or not isinstance(state_pair.get("current"), dict):
                raise ValueError(f"Invalid Lance VLM resume state for worker {worker_id}.")
            os.environ[f"{prefix}{worker_id}"] = json.dumps(state_pair, separators=(",", ":"), sort_keys=True)
            log.info(
                f"Loaded Lance VLM {_STATE_LABELS[split]} state for worker {worker_id}: "
                f"draw_count={state_pair['current']['draw_count']}"
            )
        return states

    @staticmethod
    def _save_shardlist_state(state: dict[int, NoReplaceShardlistState], split: str) -> dict[int, dict[str, int]]:
        state_dict: dict[int, dict[str, int]] = {}
        for worker_id, per_worker_state in state.items():
            state_dict[worker_id] = {"epoch": per_worker_state.epoch, "index": per_worker_state.index}
            log.info(
                f"Saved {_STATE_LABELS[split]} state for worker {worker_id}: "
                f"epoch={per_worker_state.epoch}, index={per_worker_state.index}"
            )
        return state_dict

    @staticmethod
    def _load_shardlist_state(state_dict: dict[int, dict[str, int]], split: str) -> dict[int, NoReplaceShardlistState]:
        """Restore per-worker cursors and export them to the split's dataloader workers, which start after loading."""
        state: dict[int, NoReplaceShardlistState] = {}
        for worker_id, per_worker_state in state_dict.items():
            epoch = per_worker_state["epoch"]
            index = per_worker_state["index"]
            state[worker_id] = NoReplaceShardlistState(epoch=epoch, index=index)
            epoch_env, index_env = no_replace_resume_env_names(split, worker_id)
            os.environ[epoch_env] = str(epoch)
            os.environ[index_env] = str(index)
            log.info(
                f"Loaded no replace {_STATE_LABELS[split]} state for worker {worker_id}: epoch={epoch}, index={index}"
            )
        return state
