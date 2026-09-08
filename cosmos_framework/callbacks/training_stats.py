# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch
import torch.distributed as dist
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback
from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
    ACTION_SLOT_SAMPLE_COUNT_KEY,
    ACTION_SLOT_SAMPLE_LOSS_KEY,
)
from cosmos_framework.callbacks.wandb_log import _LossRecord
from cosmos_framework.data.generator.action.utils.domain_utils import EMBODIMENT_TO_DOMAIN_ID
from cosmos_framework.data.generator.action.utils.unified_action_schema import UNIFIED_ACTION_SLOT_GROUPS

# Build inverse mapping: domain_id -> embodiment_type. First occurrence wins when multiple embodiment names share the
# same domain id.
DOMAIN_ID_TO_EMBODIMENT: dict[int, str] = {}
for _k, _v in EMBODIMENT_TO_DOMAIN_ID.items():
    DOMAIN_ID_TO_EMBODIMENT.setdefault(_v, _k)
NUM_ACTION_DOMAINS = max(DOMAIN_ID_TO_EMBODIMENT) + 1


class TrainingStatsCallback(Callback):
    """Callback for tracking and logging training mode and embodiment statistics to wandb."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self._mode_counts: dict[str, int] = {}
        self._mode_total_count: int = 0
        self._embodiment_counts: dict[str, int] = {}
        self._embodiment_total_count: int = 0
        self._per_embodiment_loss: dict[str, _LossRecord] = {}
        self._per_embodiment_sub_loss: dict[str, dict[str, _LossRecord]] = {}
        self._action_slot_family_stats: dict[str, torch.Tensor] = {}
        self._action_slot_rank_stats: torch.Tensor | None = None

    def _accumulate_mode_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        modes = data_batch.get("mode", None)
        if modes is None:
            return

        if isinstance(modes, str):
            modes_list = [modes]
        elif isinstance(modes, (list, tuple)):
            modes_list = [str(m) for m in modes]
        elif isinstance(modes, torch.Tensor):
            # Defensive: support cases where mode might be encoded numerically.
            modes_list = [str(m) for m in modes.detach().cpu().tolist()]
        else:
            modes_list = [str(modes)]

        for mode in modes_list:
            self._mode_total_count += 1
            self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

    def _accumulate_embodiment_counts(self, data_batch: dict[str, torch.Tensor]) -> None:
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return

        if isinstance(domain_ids, int):
            domain_id_list = [domain_ids]
        elif isinstance(domain_ids, (list, tuple)):
            domain_id_list = [int(d) for d in domain_ids if d is not None]
        elif isinstance(domain_ids, torch.Tensor):
            # Flatten to handle any shape (scalar, 1D, or 2D with trailing dim)
            domain_id_list = [int(d) for d in domain_ids.detach().cpu().flatten().tolist()]
        else:
            domain_id_list = [int(domain_ids)]

        for domain_id in domain_id_list:
            embodiment = DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")
            self._embodiment_total_count += 1
            self._embodiment_counts[embodiment] = self._embodiment_counts.get(embodiment, 0) + 1

    def _gather_global_mode_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_mode_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._mode_counts)
        local["__total__"] = int(self._mode_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _gather_global_embodiment_counts(self) -> tuple[int, dict[str, int]]:
        """
        Returns (global_total, global_embodiment_counts) aggregated across all ranks.
        """
        local: dict[str, int] = dict(self._embodiment_counts)
        local["__total__"] = int(self._embodiment_total_count)

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
            gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        global_total = 0
        global_counts: dict[str, int] = {}
        for item in gathered:
            if not item:
                continue
            global_total += int(item.get("__total__", 0))
            for k, v in item.items():
                if k == "__total__":
                    continue
                global_counts[k] = global_counts.get(k, 0) + int(v)
        return global_total, global_counts

    def _build_mode_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for mode in sorted(global_counts.keys()):
            count = float(global_counts.get(mode, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_mode/{mode}"] = pct

        return info

    def _build_embodiment_log_dict(
        self, *, log_prefix: str, global_total: int, global_counts: dict[str, int]
    ) -> dict[str, float]:
        info: dict[str, float] = {}

        denom = float(global_total) if global_total > 0 else 0.0
        for embodiment in sorted(global_counts.keys()):
            count = float(global_counts.get(embodiment, 0))
            pct = (100.0 * count / denom) if denom > 0 else 0.0
            info[f"{log_prefix}_stats_embodiment/{embodiment}"] = pct

        return info

    def _get_batch_embodiment(self, data_batch: dict[str, torch.Tensor]) -> str | None:
        """Extract the embodiment name from the first non-None sample's domain_id."""
        domain_ids = data_batch.get("domain_id", None)
        if domain_ids is None:
            return None
        if isinstance(domain_ids, torch.Tensor):
            if domain_ids.numel() == 0:
                return None
            domain_id = int(domain_ids.flatten()[0].item())
        elif isinstance(domain_ids, (list, tuple)):
            first = next((d for d in domain_ids if d is not None), None)
            if first is None:
                return None
            domain_id = int(first)
        else:
            domain_id = int(domain_ids)
        return DOMAIN_ID_TO_EMBODIMENT.get(domain_id, f"unknown_{domain_id}")

    def _accumulate_per_embodiment_loss(
        self,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        embodiment = self._get_batch_embodiment(data_batch)
        if embodiment is None:
            return

        if embodiment not in self._per_embodiment_loss:
            self._per_embodiment_loss[embodiment] = _LossRecord()
        self._per_embodiment_loss[embodiment].loss += loss.detach().float()
        self._per_embodiment_loss[embodiment].iter_count += 1

        if embodiment not in self._per_embodiment_sub_loss:
            self._per_embodiment_sub_loss[embodiment] = {}
        for key in output_batch:
            if key.startswith("_"):
                continue
            if "loss" in key and "per_instance" not in key:
                if key not in self._per_embodiment_sub_loss[embodiment]:
                    self._per_embodiment_sub_loss[embodiment][key] = _LossRecord()
                self._per_embodiment_sub_loss[embodiment][key].loss += output_batch[key].detach().float()
                self._per_embodiment_sub_loss[embodiment][key].iter_count += 1

    def _accumulate_rank_action_slot_stats(self, sample_loss: torch.Tensor, sample_count: torch.Tensor) -> None:
        """Accumulate optimizer-aligned rank/microbatch means."""
        # Match the optimizer reduction: every contributing rank-local
        # microbatch mean has equal weight, independent of its sample count.
        rank_sample_count = sample_count.sum(dim=0)
        rank_has_slot = rank_sample_count.gt(0)
        if self._action_slot_rank_stats is None:
            self._action_slot_rank_stats = sample_loss.new_zeros(
                (2, len(UNIFIED_ACTION_SLOT_GROUPS)), dtype=torch.float32
            )
        self._action_slot_rank_stats[0].add_(sample_loss.sum(dim=0) / rank_sample_count.clamp(min=1))
        self._action_slot_rank_stats[1].add_(rank_has_slot.to(dtype=torch.float32))

    def _accumulate_family_action_slot_stats(
        self, sample_loss: torch.Tensor, sample_count: torch.Tensor, dataset_names: list[str]
    ) -> None:
        """Accumulate sample-weighted statistics for each action dataset."""
        for dataset_name in sorted(set(dataset_names)):
            row_indices = [index for index, name in enumerate(dataset_names) if name == dataset_name]
            index = torch.tensor(row_indices, device=sample_loss.device)
            stats = self._action_slot_family_stats.get(dataset_name)
            if stats is None:
                stats = sample_loss.new_zeros((2, len(UNIFIED_ACTION_SLOT_GROUPS)), dtype=torch.float32)
                self._action_slot_family_stats[dataset_name] = stats
            stats[0].add_(sample_loss.index_select(0, index).sum(dim=0))
            stats[1].add_(sample_count.index_select(0, index).sum(dim=0))

    def _accumulate_action_slot_stats(self, output_batch: dict[str, torch.Tensor]) -> None:
        """Accumulate global and per-dataset action-slot diagnostics."""
        sample_loss = output_batch.get(ACTION_SLOT_SAMPLE_LOSS_KEY)
        sample_count = output_batch.get(ACTION_SLOT_SAMPLE_COUNT_KEY)
        if (
            not isinstance(sample_loss, torch.Tensor)
            or not isinstance(sample_count, torch.Tensor)
            or sample_loss.ndim != 2
            or sample_loss.shape != sample_count.shape
            or sample_loss.shape[1] != len(UNIFIED_ACTION_SLOT_GROUPS)
        ):
            return

        sample_loss = sample_loss.detach().float()
        sample_count = sample_count.detach().float()
        self._accumulate_rank_action_slot_stats(sample_loss, sample_count)

        dataset_names = output_batch.get("_action_family")
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if (
            not isinstance(dataset_names, (list, tuple))
            or len(dataset_names) != sample_loss.shape[0]
            or any(not isinstance(name, str) or not name for name in dataset_names)
        ):
            return
        self._accumulate_family_action_slot_stats(sample_loss, sample_count, list(dataset_names))

    def _compute_action_slot_loss_stats(self, log_prefix: str) -> dict[str, float]:
        """Aggregate rank-weighted global and sample-weighted per-dataset slot losses."""
        local_dataset_stats = {
            dataset_name: stats.cpu().tolist() for dataset_name, stats in self._action_slot_family_stats.items()
        }
        local_rank_stats = self._action_slot_rank_stats
        if local_rank_stats is None:
            local_rank_stats = torch.zeros(2, len(UNIFIED_ACTION_SLOT_GROUPS), dtype=torch.float32)
        local_payload = (local_dataset_stats, local_rank_stats.cpu().tolist())
        if dist.is_available() and dist.is_initialized():
            gathered: list[tuple[dict[str, list[list[float]]], list[list[float]]] | None] = [
                None for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(gathered, local_payload)
        else:
            gathered = [local_payload]

        family_totals: dict[str, torch.Tensor] = {}
        rank_totals = torch.zeros(2, len(UNIFIED_ACTION_SLOT_GROUPS), dtype=torch.float64)
        for payload in gathered:
            if payload is None:
                continue
            dataset_stats, rank_stats = payload
            rank_totals.add_(torch.tensor(rank_stats, dtype=torch.float64))
            for dataset_name, stats in dataset_stats.items():
                family_totals.setdefault(
                    dataset_name,
                    torch.zeros(2, len(UNIFIED_ACTION_SLOT_GROUPS), dtype=torch.float64),
                ).add_(torch.tensor(stats, dtype=torch.float64))

        result: dict[str, float] = {}
        for slot_index, (slot_name, _) in enumerate(UNIFIED_ACTION_SLOT_GROUPS):
            count = rank_totals[1, slot_index].item()
            if count > 0:
                result[f"{log_prefix}_stats_loss_action_slot/{slot_name}"] = rank_totals[0, slot_index].item() / count
        for dataset_name, stats in sorted(family_totals.items()):
            sample_loss_sum = stats[0]
            sample_count = stats[1]
            for slot_index, (slot_name, _) in enumerate(UNIFIED_ACTION_SLOT_GROUPS):
                count = sample_count[slot_index].item()
                if count > 0:
                    result[f"{log_prefix}_stats_action_slot_loss/{dataset_name}/{slot_name}"] = (
                        sample_loss_sum[slot_index].item() / count
                    )
                    result[f"{log_prefix}_stats_action_slot_count/{dataset_name}/{slot_name}"] = count
        self._action_slot_family_stats = {}
        self._action_slot_rank_stats = None
        return result

    @torch.no_grad()
    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        parallel_dims = getattr(model, "parallel_dims", None)
        if (
            parallel_dims is not None
            and getattr(parallel_dims, "cp_enabled", False)
            and getattr(parallel_dims, "cp_rank", 0) != 0
        ):
            return
        self._accumulate_action_slot_stats(output_batch)

    def _compute_per_embodiment_loss_stats(self, log_prefix: str) -> dict[str, float]:
        """Compute per-embodiment loss averages across all ranks.

        All ranks must call this method (contains collective operations).
        Returns the log dict (only meaningful on rank 0).
        """
        dist_available = dist.is_available() and dist.is_initialized()
        world_size = int(dist.get_world_size()) if dist_available else 1

        # Step 1: gather union of embodiment names across ranks
        local_embodiments = sorted(self._per_embodiment_loss.keys())
        if dist_available:
            all_embodiments: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(all_embodiments, local_embodiments)
        else:
            all_embodiments = [local_embodiments]
        union_embodiments = sorted({e for el in all_embodiments for e in el})

        # Step 2: gather union of sub-loss keys across ranks
        local_sub_keys = sorted({k for d in self._per_embodiment_sub_loss.values() for k in d})
        if dist_available:
            all_sub_keys: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(all_sub_keys, local_sub_keys)
        else:
            all_sub_keys = [local_sub_keys]
        union_sub_keys = sorted({k for kl in all_sub_keys for k in kl})

        # Step 3: insert NaN dummy _LossRecord for missing embodiment/key combos
        for emb in union_embodiments:
            if emb not in self._per_embodiment_loss:
                dummy = _LossRecord()
                dummy.loss += torch.tensor([float("nan")], device="cuda")
                dummy.iter_count += 1
                self._per_embodiment_loss[emb] = dummy
            if emb not in self._per_embodiment_sub_loss:
                self._per_embodiment_sub_loss[emb] = {}
            for key in union_sub_keys:
                if key not in self._per_embodiment_sub_loss[emb]:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")
                    dummy.iter_count += 1
                    self._per_embodiment_sub_loss[emb][key] = dummy

        # Step 4: compute distributed averages (all ranks participate in all_reduce)
        log_dict: dict[str, float] = {}
        for emb in union_embodiments:
            avg, valid = self._per_embodiment_loss[emb].get_stat(return_valid_mask_sum=True)
            if valid > 0:
                log_dict[f"{log_prefix}_stats_loss/{emb}"] = avg

        for emb in union_embodiments:
            for key in union_sub_keys:
                avg, valid = self._per_embodiment_sub_loss[emb][key].get_stat(return_valid_mask_sum=True)
                if valid > 0:
                    log_dict[f"{log_prefix}_stats_loss_detail/{emb}_{key}"] = avg

        # Step 5: reset accumulators
        self._per_embodiment_loss = {}
        self._per_embodiment_sub_loss = {}

        return log_dict

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self._accumulate_mode_counts(data_batch)
        self._accumulate_embodiment_counts(data_batch)
        self._accumulate_per_embodiment_loss(data_batch, output_batch, loss)

        if iteration % self.log_freq != 0:
            return

        # All ranks must participate in collective operations below.
        mode_total, mode_counts = self._gather_global_mode_counts()
        embodiment_total, embodiment_counts = self._gather_global_embodiment_counts()
        per_embodiment_loss_dict = self._compute_per_embodiment_loss_stats(log_prefix="train")
        action_slot_loss_dict = self._compute_action_slot_loss_stats(log_prefix="train")

        if not distributed.is_rank0():
            return

        if wandb.run is None:
            return

        log_dict: dict[str, float] = {}
        log_dict.update(
            self._build_mode_log_dict(log_prefix="train", global_total=mode_total, global_counts=mode_counts)
        )
        log_dict.update(
            self._build_embodiment_log_dict(
                log_prefix="train", global_total=embodiment_total, global_counts=embodiment_counts
            )
        )
        log_dict.update(per_embodiment_loss_dict)
        log_dict.update(action_slot_loss_dict)

        wandb.log({k: float(v) for k, v in log_dict.items()}, step=iteration)
