# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time
from collections import defaultdict

import psutil
import torch
import torch.distributed as dist
import wandb

from imaginaire.datasets.webdataset.utils.stream import (
    ENABLE_STREAM_WANDB,
    WATCHDOG_ENABLED,
    collect_throughput_ipc_stats,
)
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.data.vfm.joint_dataloader import _PackingMetrics

_AGG_COUNT, _AGG_SUM, _AGG_MIN, _AGG_MAX = 0, 1, 2, 3
_AGG_COLS = 4


class DetailedDataLoadingSpeedMonitor(Callback):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        save_s3: bool = False,
        action_family_names: list[str] | None = None,
    ):
        """Detailed dataloading + memory diagnostics callback.

        Args:
            every_n: How often (in training iterations) to flush stats to W&B.
            step_size: Trainer step granularity (passed-through; usually 1).
            save_s3: If True, snapshot ``wandb_info`` to ``s3://rundir/...`` on each flush.
            action_family_names: Set of dataset names that are *action families*
                (e.g. ``["umi_20260501", "fractal_20260501", ...]``).  When
                supplied, every flush also emits per-rank labeled metrics
                ``action_dl_per_rank/{family}_rank_{NNN:03d}_{mean,max}`` —
                which is exactly the per-rank dataloading wait under
                Hare-Niemeyer (each rank serves one family) restricted to
                **action iterations only** (i.e. iterations where the
                ``JointDataLoader`` selected ``action_data``, not VFM).
                Useful to disambiguate per-rank stalls in VFM+action mixed
                runs without scrolling through 2048 ``dataloading_NNN``
                metrics whose rank ↔ family mapping is implicit.
                If ``None`` (default), the per-rank labeled metrics are
                skipped — the existing global ``dl_wait_time_per_dataset/{ds}_*``
                cross-rank aggregates remain available for back-compat.
        """
        self.every_n = every_n
        self.step_size = step_size
        self.should_run = False
        self.start_dataloading_time = None
        self.dataloading_time = None
        self.name = self.__class__.__name__
        self.save_s3 = save_s3
        self.time_delta_list = []
        # Parallel to ``time_delta_list``: the dataset name attached to the
        # batch consumed at iter N.  Populated at on_training_step_end (when
        # ``data_batch["dataset_name"]`` is available), so its length tracks
        # ``time_delta_list`` exactly under normal flow.  Used to filter the
        # per-rank dataloading metric to action-only iterations when
        # ``action_family_names`` is supplied.
        self._iter_dataset_names: list[str] = []
        # Node-wide RAM stats (psutil.virtual_memory()): the same value for all
        # local ranks on the same host, so the per-rank suffix indexes nodes
        # not ranks.  Useful for tracking absolute host pressure / page-cache
        # headroom but cannot attribute memory to a specific action family.
        self.memory_consumption_list = []
        self.memory_consumption_percentage_list = []
        self._action_family_set: set[str] | None = set(action_family_names) if action_family_names else None
        # Deterministic sorted list of action family names — same on every
        # rank.  Used to encode each rank's family identity as an integer
        # index that travels alongside (mean, max) inside a single
        # all_gather_tensor call, avoiding the slower all_gather_object that
        # the per-rank label metric used to need.
        self._action_family_list: list[str] = sorted(self._action_family_set) if self._action_family_set else []
        self._action_family_index: dict[str, int] = {f: i for i, f in enumerate(self._action_family_list)}
        self._pending_time_delta = None
        self.dataloading_time_per_dataset = {}
        self._worker_batch_times = []
        self._worker_aug_times = []
        self._worker_io_times = []
        self._worker_aug_step_times: dict[str, list[float]] = defaultdict(list)
        self._worker_times_by_ds_wid: dict[tuple[str, int], list[float]] = defaultdict(list)
        self._dataset_scalar_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._dataset_list_stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    def on_before_dataloading(self, iteration: int = 0) -> None:
        # We want to run it one iteration before on_training_step_start should_run is set to True.
        global_step = iteration // self.step_size
        self.should_run = (global_step + 1) % self.every_n == 0
        self.start_dataloading_time = time.time()

    def on_after_dataloading(self, iteration: int = 0) -> None:
        self._pending_time_delta = time.time() - self.start_dataloading_time
        self.time_delta_list.append(self._pending_time_delta)
        memory = psutil.virtual_memory()
        self.memory_consumption_list.append(memory.used / (1024**3))
        self.memory_consumption_percentage_list.append(memory.percent)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        dataset_name = data_batch.get("dataset_name", ["default"])[0]
        # ``_action_family`` is stamped per-sample by
        # :class:`~projects.cosmos3.vfm.datasets.action.unified_dataset.ActionUnifiedIterableDataset`
        # and survives the outer ``JointDataLoader`` (which overwrites
        # ``dataset_name`` with the *stream* name e.g. ``"action_data"``).
        # When present, it is the per-rank action family identity needed by
        # the ``action_dl_per_rank/{family}_rank_{NNN:03d}_{mean,max}``
        # filter below.  Falls back to ``dataset_name`` for VFM streams that
        # have no inner family stamp.
        action_family = data_batch["_action_family"][0] if "_action_family" in data_batch else None
        family_or_stream = action_family if action_family is not None else dataset_name
        # Always tag this iteration with its family-or-stream name (even
        # when _pending_time_delta is None due to a missed
        # on_after_dataloading call) so that ``_iter_dataset_names`` and
        # ``time_delta_list`` stay the same length for the action-only
        # filter below.
        if len(self._iter_dataset_names) < len(self.time_delta_list):
            self._iter_dataset_names.append(family_or_stream)
        if self._pending_time_delta is not None:
            if dataset_name not in self.dataloading_time_per_dataset:
                self.dataloading_time_per_dataset[dataset_name] = []
            self.dataloading_time_per_dataset[dataset_name].append(self._pending_time_delta)
            self._pending_time_delta = None

        for batch_key, _, agg_type in _PackingMetrics.STATS_SPEC:
            if batch_key not in data_batch:
                continue
            val = int(data_batch[batch_key])
            if agg_type == "scalar":
                self._dataset_scalar_stats[batch_key][dataset_name] += val
            else:
                self._dataset_list_stats[batch_key][dataset_name].append(val)

        if "_worker_batch_time" in data_batch:
            bt = float(data_batch["_worker_batch_time"])
            self._worker_batch_times.append(bt)
            wid = int(data_batch.get("_worker_id", 0))
            self._worker_times_by_ds_wid[(dataset_name, wid)].append(bt)
        if "_worker_aug_time" in data_batch:
            self._worker_aug_times.append(float(data_batch["_worker_aug_time"]))
        if "_worker_io_time" in data_batch:
            self._worker_io_times.append(float(data_batch["_worker_io_time"]))
        if "_worker_aug_step_times" in data_batch:
            for step_name, t in data_batch["_worker_aug_step_times"].items():
                self._worker_aug_step_times[step_name].append(float(t))

        if self.should_run:
            # Convert list to tensor on GPU for gathering
            local_times = torch.tensor(self.time_delta_list).cuda()  # [num_iters]
            local_memory_consumption = torch.tensor(self.memory_consumption_list).cuda()  # [num_iters]
            local_memory_consumption_percentage = torch.tensor(
                self.memory_consumption_percentage_list
            ).cuda()  # [num_iters]
            iteration_count = len(self.time_delta_list)
            # Compute action-only per-iter mask BEFORE clearing buffers — paired
            # with ``time_delta_list`` 1:1 so we can derive each rank's action
            # iteration mean/max in the per-rank labeled metric below.
            iter_dataset_names_local = list(self._iter_dataset_names[:iteration_count])
            self.time_delta_list = []  # Reset the list
            self.memory_consumption_list = []
            self.memory_consumption_percentage_list = []
            self._iter_dataset_names = self._iter_dataset_names[iteration_count:]

            # Gather all times from all ranks
            # Each tensor in the list has shape (num_iterations,)
            gathered_times_list = distributed.all_gather_tensor(local_times)  # list of [num_iters], len=world_size

            # Stack to get shape (world_size, num_iterations)
            all_times = torch.stack(gathered_times_list)  # [world_size,num_iters]

            # Calculate per-rank statistics
            # dim=1 is across iterations
            rank_means = torch.mean(all_times, dim=1)  # [world_size]
            rank_maxes = torch.max(all_times, dim=1).values  # [world_size]

            wandb_info = {f"{self.name}_mean/dataloading_{k:03d}": v.item() for k, v in enumerate(rank_means)}
            wandb_info.update({f"{self.name}_max/dataloading_{k:03d}": v.item() for k, v in enumerate(rank_maxes)})

            # Per-rank action-only labeled dataloading metric.  Unlike the
            # generic ``dataloading_NNN`` above (which mixes action + VFM in
            # joint runs), this filters ``time_delta_list`` to iterations
            # whose batch came from one of ``self._action_family_set``, then
            # publishes each rank's mean/max under a name that includes the
            # family it served — e.g.
            # ``action_dl_per_rank/umi_20260501_rank_000_mean``.  Because
            # under Hare-Niemeyer each rank serves exactly one action family,
            # a rank's filtered-times all share a single dataset name, and
            # we use the most recent observed name as the family label.
            #
            # Family identity is encoded as an integer index into the
            # deterministic ``_action_family_list`` (same on every rank,
            # built once at __init__ from the constructor-supplied
            # ``action_family_names``) so it can travel inside the same
            # ``all_gather_tensor`` call as (mean, max) — no
            # ``all_gather_object`` needed.
            if self._action_family_set is not None and iter_dataset_names_local:
                action_mask_local = torch.tensor(
                    [(ds in self._action_family_set) for ds in iter_dataset_names_local],
                    dtype=torch.bool,
                ).cuda()  # [num_iters]
                # Pad to iteration_count if dataset_names were missed (shouldn't
                # happen with normal flow but guards against edge cases).
                if action_mask_local.numel() < iteration_count:
                    pad = torch.zeros(
                        iteration_count - action_mask_local.numel(),
                        dtype=torch.bool,
                    ).cuda()
                    action_mask_local = torch.cat([action_mask_local, pad])
                # Most recent action family observed locally — under
                # Hare-Niemeyer this is the SAME family on every action
                # iter for this rank, so picking the last one is exact.
                local_family_label = next(
                    (ds for ds in reversed(iter_dataset_names_local) if ds in self._action_family_set),
                    None,
                )
                local_family_idx = (
                    self._action_family_index.get(local_family_label, -1) if local_family_label is not None else -1
                )
                if action_mask_local.any():
                    action_times_local = all_times[distributed.get_rank()][action_mask_local]  # [num_action_iters]
                    local_action_mean = action_times_local.mean().to(torch.float64)
                    local_action_max = action_times_local.max().to(torch.float64)
                else:
                    local_action_mean = torch.tensor(float("nan"), dtype=torch.float64).cuda()
                    local_action_max = torch.tensor(float("nan"), dtype=torch.float64).cuda()
                # Pack [mean, max, family_idx] into one tensor and gather
                # once — single all_gather_tensor, no object exchange.
                local_action_stats = torch.stack(
                    [
                        local_action_mean,
                        local_action_max,
                        torch.tensor(float(local_family_idx), dtype=torch.float64).cuda(),
                    ]
                )  # [3]
                all_action_stats = self._gather_list_stats(local_action_stats)  # [world_size, 3]
                num_families = len(self._action_family_list)
                for rank_id in range(all_action_stats.shape[0]):
                    fam_idx = int(all_action_stats[rank_id, 2].item())
                    if fam_idx < 0 or fam_idx >= num_families:
                        continue  # rank served no action data this window
                    m = all_action_stats[rank_id, 0].item()
                    mx = all_action_stats[rank_id, 1].item()
                    if m != m or mx != mx:  # NaN check
                        continue
                    fam = self._action_family_list[fam_idx]
                    wandb_info[f"action_dl_per_rank/{fam}_rank_{rank_id:03d}_mean"] = m
                    wandb_info[f"action_dl_per_rank/{fam}_rank_{rank_id:03d}_max"] = mx

            gathered_memory_consumption = distributed.all_gather_tensor(
                local_memory_consumption
            )  # list of [num_iters], len=world_size
            gathered_memory_consumption_percentage = distributed.all_gather_tensor(
                local_memory_consumption_percentage
            )  # list of [num_iters], len=world_size

            wandb_info.update(
                {
                    f"{self.name}_mean/memory_consumption_gb_{k:03d}": v.mean().item()
                    for k, v in enumerate(gathered_memory_consumption)
                }
            )
            wandb_info.update(
                {
                    f"{self.name}_mean/memory_consumption_percentage_{k:03d}": v.mean().item()
                    for k, v in enumerate(gathered_memory_consumption_percentage)
                }
            )
            # Per-rank max over the window — surfaces transient peaks (heavy
            # families on lonely ranks can spike between mean-sampled steps).
            wandb_info.update(
                {
                    f"{self.name}_max/memory_consumption_gb_{k:03d}": v.max().item()
                    for k, v in enumerate(gathered_memory_consumption)
                }
            )
            wandb_info.update(
                {
                    f"{self.name}_max/memory_consumption_percentage_{k:03d}": v.max().item()
                    for k, v in enumerate(gathered_memory_consumption_percentage)
                }
            )

            wandb_info[f"{self.name}_mean/memory_consumption_gb_mean"] = (
                torch.stack(gathered_memory_consumption).mean().item()  # [world_size,num_iters]
            )
            wandb_info[f"{self.name}_mean/memory_consumption_percentage_mean"] = (
                torch.stack(gathered_memory_consumption_percentage).mean().item()  # [world_size,num_iters]
            )
            wandb_info[f"{self.name}_max/memory_consumption_gb_max"] = (
                torch.stack(gathered_memory_consumption).max().item()  # [world_size,num_iters]
            )
            wandb_info[f"{self.name}_max/memory_consumption_percentage_max"] = (
                torch.stack(gathered_memory_consumption_percentage).max().item()  # [world_size,num_iters]
            )

            # Identify slowest rank based on mean time
            slowest_dataloading_rank_id = torch.argmax(rank_means)  # []
            max_dataloading = torch.max(rank_means)  # []

            # Calculate sum of max times across all iterations (new metric)
            # Max across ranks for each iteration (dim=0)
            max_per_iter = torch.max(all_times, dim=0).values  # [num_iters]
            sum_of_max_times = torch.sum(max_per_iter).item() / iteration_count

            wandb_info.update(
                {
                    "slowest_rank/slowest_dataloading_rank": slowest_dataloading_rank_id.item(),
                    "slowest_rank/slowest_dataloading_time": max_dataloading.item(),
                    "slowest_rank/sum_of_max_dataloading_time_per_iteration": sum_of_max_times,
                }
            )

            # 1. Gather and log stream throughput and watchdog reconnect stats for `stream_throughput` metrics
            self._gather_and_log_stream_throughput(wandb_info)

            # Only all_gather_object to get name indices (dataset names, aug-step names, worker-balance keys) across all ranks
            # Later methods 2-4 use efficient all_gather_tensor to gather tensor data, then compute statistics and log metrics
            ds_index, aug_index, dswid_index = self._discover_name_indices()

            # 2.Gather and log per-dataset dataloading wait times for `dl_wait_time_per_dataset` metrics
            self._gather_and_log_per_dataset_time(wandb_info, ds_index)

            # 3. Gather and log per-dataset sampling stats for `dl_packing_stats` metrics
            self._gather_and_log_packing_stats(wandb_info, ds_index)

            # 4. Gather and log worker timing metrics for `dl_worker_batch_time`, `dl_worker_balance_per_dataset`, `dl_worker_augmentation` metrics
            self._gather_and_log_worker_timing(wandb_info, dswid_index, aug_index)

            if wandb.run:
                wandb.log(wandb_info, step=iteration)

            if self.save_s3 and distributed.is_rank0():
                easy_io.dump(
                    wandb_info,
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )

    def _discover_name_indices(
        self,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Discover the global union of dataset, aug-step, and worker-balance names.

        Performs a single ``all_gather_object`` call to exchange short string
        lists across all ranks and returns deterministic index mappings.

        Returns:
            ds_index: ``{dataset_name: col_idx}`` for per-dataset tensors.
            aug_index: ``{step_name: col_idx}`` for augmentation step tensors.
            dswid_index: ``{"ds|wid": col_idx}`` for worker-balance tensors.
        """
        local_ds_names: set[str] = set(self.dataloading_time_per_dataset.keys())
        for key_dict in self._dataset_scalar_stats.values():
            local_ds_names.update(key_dict.keys())
        for key_dict in self._dataset_list_stats.values():
            local_ds_names.update(key_dict.keys())

        local_names = {
            "datasets": sorted(local_ds_names),
            "aug_steps": sorted(self._worker_aug_step_times.keys()),
            "ds_wid": sorted(f"{ds}|{wid}" for ds, wid in self._worker_times_by_ds_wid.keys()),
        }
        all_names: list[dict] = [{} for _ in range(dist.get_world_size())]  # len=world_size
        dist.all_gather_object(all_names, local_names)

        union_ds = sorted({n for r in all_names for n in r.get("datasets", [])})
        ds_index = {name: i for i, name in enumerate(union_ds)}

        union_aug = sorted({n for r in all_names for n in r.get("aug_steps", [])})
        aug_index = {name: i for i, name in enumerate(union_aug)}

        union_dswid = sorted({n for r in all_names for n in r.get("ds_wid", [])})
        dswid_index = {name: i for i, name in enumerate(union_dswid)}

        return ds_index, aug_index, dswid_index

    def _gather_and_log_per_dataset_time(self, wandb_info: dict, ds_index: dict[str, int]) -> None:
        """Gather per-dataset dataloading wait times via ``all_gather_tensor``."""
        N = len(ds_index)
        if N == 0:
            self.dataloading_time_per_dataset = {}
            return

        local_ds_time = torch.full((N,), float("nan"), dtype=torch.float64).cuda()  # [num_datasets]
        for ds, times in self.dataloading_time_per_dataset.items():
            if ds in ds_index:
                local_ds_time[ds_index[ds]] = sum(times) / len(times)

        all_ds_time = self._gather_list_stats(local_ds_time)  # [world_size, num_datasets]
        for ds, i in ds_index.items():
            col = all_ds_time[:, i]  # [world_size]
            valid = col[~col.isnan()]  # [<=world_size]
            if len(valid) > 0:
                wandb_info[f"dl_wait_time_per_dataset/{ds}_mean"] = valid.mean().item()
                wandb_info[f"dl_wait_time_per_dataset/{ds}_max"] = valid.max().item()

        self.dataloading_time_per_dataset = {}

    def _gather_and_log_packing_stats(self, wandb_info: dict, ds_index: dict[str, int]) -> None:
        """Gather packing diagnostics via ``all_gather_tensor``, driven by ``_PackingMetrics.STATS_SPEC``."""
        _STATS = "dl_packing_stats"
        N = len(ds_index)
        if N == 0:
            self._dataset_scalar_stats = defaultdict(lambda: defaultdict(int))
            self._dataset_list_stats = defaultdict(lambda: defaultdict(list))
            return

        for batch_key, wandb_suffix, _ in _PackingMetrics.STATS_SPEC:
            if batch_key == "_num_tokens":
                # Token fraction: gather per-rank token sums, compute each dataset's share of total
                local_v = torch.zeros(N, dtype=torch.float64).cuda()  # [num_datasets]
                for ds, i in ds_index.items():
                    local_v[i] = self._dataset_scalar_stats.get(batch_key, {}).get(ds, 0)
                all_v = self._gather_list_stats(local_v)  # [world_size, num_datasets]
                global_tokens = all_v.sum(dim=0)  # [num_datasets]
                total = global_tokens.sum().item()
                for ds, i in ds_index.items():
                    wandb_info[f"{_STATS}/{ds}_{wandb_suffix}"] = global_tokens[i].item() / total if total > 0 else 0.0

            elif batch_key == "_dropped_count":
                # Dropped samples: gather per-rank counts, report global total per dataset
                local_v = torch.zeros(N, dtype=torch.float64).cuda()  # [num_datasets]
                for ds, i in ds_index.items():
                    local_v[i] = self._dataset_scalar_stats.get(batch_key, {}).get(ds, 0)
                all_v = self._gather_list_stats(local_v)  # [world_size, num_datasets]
                for ds, i in ds_index.items():
                    wandb_info[f"{_STATS}/{ds}_{wandb_suffix}_total"] = int(all_v[:, i].sum().item())

            else:
                # Per-batch distributions (_num_samples, _from_buffer, _from_workers, _buffer_size).
                # Each rank packs [count, sum, min, max]; reduce to weighted global mean/min/max.
                local_t = torch.full(
                    (N, _AGG_COLS), float("nan"), dtype=torch.float64
                ).cuda()  # [num_datasets, _AGG_COLS]
                for ds, i in ds_index.items():
                    vals = self._dataset_list_stats.get(batch_key, {}).get(ds, [])
                    if vals:
                        local_t[i] = torch.tensor([len(vals), sum(vals), min(vals), max(vals)], dtype=torch.float64)
                all_t = self._gather_list_stats(local_t)  # [world_size, num_datasets, _AGG_COLS]
                for ds, i in ds_index.items():
                    result = self._reduce_agg_column(all_t[:, i, :])
                    if result:
                        mean_val, min_val, max_val = result
                        wandb_info[f"{_STATS}/{ds}_{wandb_suffix}_mean"] = mean_val
                        wandb_info[f"{_STATS}/{ds}_{wandb_suffix}_min"] = min_val
                        wandb_info[f"{_STATS}/{ds}_{wandb_suffix}_max"] = max_val

        self._dataset_scalar_stats = defaultdict(lambda: defaultdict(int))
        self._dataset_list_stats = defaultdict(lambda: defaultdict(list))

    def _gather_and_log_stream_throughput(self, wandb_info: dict) -> None:
        """Gather stream throughput and watchdog reconnect stats via IPC files."""
        if not ENABLE_STREAM_WANDB:
            return

        tp_keys = ["MBps"]
        if WATCHDOG_ENABLED:
            tp_keys.append("watchdog_reconnects")
        tp_stats = collect_throughput_ipc_stats()
        local_tp = torch.tensor([tp_stats.get(k, 0.0) for k in tp_keys]).cuda()  # [num_metrics]
        gathered_tp = distributed.all_gather_tensor(local_tp)  # list of [num_metrics], len=world_size
        all_tp = torch.stack(gathered_tp)  # [world_size, num_metrics]

        for ki, k in enumerate(tp_keys):
            col = all_tp[:, ki]  # [world_size]
            wandb_info[f"stream_throughput/{k}_mean"] = col.mean().item()
            wandb_info[f"stream_throughput/{k}_min"] = col.min().item()
            wandb_info[f"stream_throughput/{k}_max"] = col.max().item()
            if k == "watchdog_reconnects":
                wandb_info[f"stream_throughput/{k}_sum"] = col.sum().item()

        mbps_col = all_tp[:, 0]  # [world_size]
        slowest_throughput_rank = mbps_col.argmin().item()
        wandb_info["slowest_rank/slowest_stream_throughput_rank"] = slowest_throughput_rank

    @staticmethod
    def _gather_list_stats(local: torch.Tensor) -> torch.Tensor:
        """all_gather_tensor + stack, returning [world_size, *local.shape]."""
        return torch.stack(distributed.all_gather_tensor(local))

    @staticmethod
    def _reduce_agg_column(col: torch.Tensor) -> tuple[float, float, float] | None:
        """From a [world_size, _AGG_COLS] slice, return (mean, min, max) or None if empty.

        Each row is [count, sum, min, max] from one rank. Rows with NaN count
        are ranks that had no data for this key.

        Used for metrics where each rank accumulates a variable-length list of
        values (e.g. samples_per_batch, buffer_size, per-aug-step times) and we
        need a correct weighted global mean rather than a simple average of
        per-rank means. The sum/count columns make this possible.

        Callers: ``_gather_and_log_packing_stats`` (list-type metrics) and
        ``_gather_and_log_worker_timing`` (per-aug-step breakdown).
        """
        valid = col[~col[:, _AGG_COUNT].isnan()]
        if len(valid) == 0:
            return None
        total_count = valid[:, _AGG_COUNT].sum().item()
        total_sum = valid[:, _AGG_SUM].sum().item()
        if total_count == 0:
            return None
        return (
            total_sum / total_count,
            valid[:, _AGG_MIN].min().item(),
            valid[:, _AGG_MAX].max().item(),
        )

    def _gather_and_log_worker_timing(
        self, wandb_info: dict, dswid_index: dict[str, int], aug_index: dict[str, int]
    ) -> None:
        """Gather worker timing from all ranks and log percentile metrics.

        All metrics here are worker-side measurements — time spent inside
        DataLoader worker processes producing batches. This is different from
        DetailedDataLoadingSpeedMonitor or dl_wait_time_per_dataset/ metrics which measure main-process wall-clock time,
        This can help identify if the bottleneck is in the dataloader worker processes or in the main process,
        for example waiting for a packed output batch from the JointDataLoader

        Logged metrics:
        Section 1 – dl_worker_batch_time/
            Every individual batch time from every worker from every rank, all
            thrown into one pool. One data point = one batch produced by one
            worker at one step. Computes p50/p90/p99/max/mean of that pool.
            Answers: What is the tail latency to produce a batch?

        Section 2 – dl_worker_balance_per_dataset/
            First computes each worker's average batch time over the logging
            window. One data point = one worker's mean over several batches.
            Then gathers these per-worker averages across all ranks, grouped by
            dataset. Computes mean/std/min/max of those averages.
            Answers: Are some workers consistently slower than others within
            each sub-dataloader?

        Section 3 – dl_worker_augmentation/
            Unified augmentation profiling.  Contains:
            - total_aug_mean|min|max – total augmentation time per batch
            - total_io_mean|min|max – I/O time per batch (batch_time minus aug_time)
            - aug_fraction_mean, io_fraction_mean – what fraction of batch time is spent in augmentation vs I/O
            - aug_steps/{StepName}_mean|min|max – per-augmentor-step breakdown
              (e.g. VideoParsingWithFullFrames for video decode,
              TextTokenizerTransform for text tokenization).
            All use mean/min/max globally across all ranks.
            Answers: Is the bottleneck in augmentations or downloads, and
            which augmentor step dominates?

        Note: dl_packing_stats/ is logged from on_training_step_end (not here).
        It reports token_fraction, samples_per_batch, from_buffer, from_workers, buffer_size, and dropped_total per dataset — useful for tuning num_workers/batch_size/prefetch per dataloader.
        """
        if not self._worker_batch_times:
            self._worker_aug_times = []
            self._worker_io_times = []
            self._worker_aug_step_times = defaultdict(list)
            self._worker_times_by_ds_wid = defaultdict(list)
            return

        _PERCENTILES = [0.50, 0.90, 0.99]
        _PNAMES = ["p50", "p90", "p99"]

        # Gather raw batch times across all ranks
        local_bt = torch.tensor(self._worker_batch_times, dtype=torch.float32).cuda()  # [num_batches_local]
        gathered_bt = distributed.all_gather_tensor(local_bt)  # list of [num_batches_local], len=world_size
        all_bt = torch.cat(gathered_bt)  # [num_batches_all_ranks]

        # Section 1: global batch time percentiles
        _BATCH_PREFIX = "dl_worker_batch_time"
        for pval, pname in zip(_PERCENTILES, _PNAMES):
            wandb_info[f"{_BATCH_PREFIX}/{pname}"] = all_bt.quantile(pval).item()
        wandb_info[f"{_BATCH_PREFIX}/max"] = all_bt.max().item()
        wandb_info[f"{_BATCH_PREFIX}/mean"] = all_bt.mean().item()

        # Section 2: per-dataloader worker balance
        # Each rank fills its (dataset, worker_id) slots with that worker's
        # mean batch time; NaN marks absent slots. After all_gather we group
        # by dataset and compute cross-rank statistics.

        _BALANCE_PREFIX = "dl_worker_balance_per_dataset"
        if dswid_index:
            N_dswid = len(dswid_index)
            local_pw = torch.full((N_dswid,), float("nan"), dtype=torch.float64).cuda()  # [num_ds_worker_pairs]
            for (ds_name, wid), ts in self._worker_times_by_ds_wid.items():
                key = f"{ds_name}|{wid}"
                if key in dswid_index:
                    local_pw[dswid_index[key]] = sum(ts) / len(ts)

            all_pw = self._gather_list_stats(local_pw)  # [world_size, num_ds_worker_pairs]

            # Pass 1: collect all valid per-worker means, grouped by dataset
            ds_worker_vals: dict[str, list[float]] = defaultdict(list)
            for key, idx in dswid_index.items():
                ds_name = key.rsplit("|", 1)[0]
                col = all_pw[:, idx]  # [world_size]
                valid = col[~col.isnan()]  # [<=world_size]
                ds_worker_vals[ds_name].extend(valid.tolist())

            # Pass 2: log per-dataset worker balance statistics
            for ds_name in sorted(ds_worker_vals):
                pw_means = ds_worker_vals[ds_name]
                if not pw_means:
                    continue
                pw_t = torch.tensor(pw_means, dtype=torch.float32).cuda()  # [num_workers_for_ds]
                wandb_info[f"{_BALANCE_PREFIX}/{ds_name}_mean"] = pw_t.mean().item()
                wandb_info[f"{_BALANCE_PREFIX}/{ds_name}_std"] = pw_t.std().item()
                wandb_info[f"{_BALANCE_PREFIX}/{ds_name}_min"] = pw_t.min().item()
                wandb_info[f"{_BALANCE_PREFIX}/{ds_name}_max"] = pw_t.max().item()

        # Section 3: augmentation profiling (total aug/io + per-step breakdown)
        _AUG_PREFIX = "dl_worker_augmentation"

        if self._worker_aug_times:
            local_aug = torch.tensor(self._worker_aug_times, dtype=torch.float32).cuda()  # [num_batches_local]
            all_aug = torch.cat(distributed.all_gather_tensor(local_aug))  # [num_batches_all_ranks]
            wandb_info[f"{_AUG_PREFIX}/total_aug_mean"] = all_aug.mean().item()
            wandb_info[f"{_AUG_PREFIX}/total_aug_min"] = all_aug.min().item()
            wandb_info[f"{_AUG_PREFIX}/total_aug_max"] = all_aug.max().item()

        if self._worker_io_times:
            local_io = torch.tensor(self._worker_io_times, dtype=torch.float32).cuda()  # [num_batches_local]
            all_io = torch.cat(distributed.all_gather_tensor(local_io))  # [num_batches_all_ranks]
            wandb_info[f"{_AUG_PREFIX}/total_io_mean"] = all_io.mean().item()
            wandb_info[f"{_AUG_PREFIX}/total_io_min"] = all_io.min().item()
            wandb_info[f"{_AUG_PREFIX}/total_io_max"] = all_io.max().item()

        if self._worker_aug_times and self._worker_batch_times:
            aug_fracs = [
                a / b for a, b in zip(self._worker_aug_times, self._worker_batch_times) if b > 0
            ]  # [num_valid_batches_local]
            if aug_fracs:
                local_fracs = torch.tensor(aug_fracs, dtype=torch.float32).cuda()  # [num_valid_batches_local]
                all_fracs = torch.cat(distributed.all_gather_tensor(local_fracs))  # [num_valid_batches_all_ranks]
                wandb_info[f"{_AUG_PREFIX}/aug_fraction_mean"] = all_fracs.mean().item()
                wandb_info[f"{_AUG_PREFIX}/io_fraction_mean"] = 1.0 - all_fracs.mean().item()

        # Per-augmentor-step breakdown (converted to all_gather_tensor)
        if aug_index:
            N_aug = len(aug_index)
            local_aug_steps = torch.full(
                (N_aug, _AGG_COLS), float("nan"), dtype=torch.float64
            ).cuda()  # [num_aug_steps, _AGG_COLS]
            for step_name, ts in self._worker_aug_step_times.items():
                if step_name in aug_index and ts:
                    local_aug_steps[aug_index[step_name]] = torch.tensor(
                        [len(ts), sum(ts), min(ts), max(ts)], dtype=torch.float64
                    )

            all_aug_steps = self._gather_list_stats(local_aug_steps)  # [world_size, num_aug_steps, _AGG_COLS]
            for step_name, idx in aug_index.items():
                result = self._reduce_agg_column(all_aug_steps[:, idx, :])
                if result:
                    mean_val, min_val, max_val = result
                    wandb_info[f"{_AUG_PREFIX}/aug_steps/{step_name}_mean"] = mean_val
                    wandb_info[f"{_AUG_PREFIX}/aug_steps/{step_name}_min"] = min_val
                    wandb_info[f"{_AUG_PREFIX}/aug_steps/{step_name}_max"] = max_val

        self._worker_batch_times = []
        self._worker_aug_times = []
        self._worker_io_times = []
        self._worker_aug_step_times = defaultdict(list)
        self._worker_times_by_ds_wid = defaultdict(list)
