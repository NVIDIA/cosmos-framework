# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io


def _debug_enabled() -> bool:
    """Whether ``log.debug`` will emit, so callers can skip building expensive messages."""
    return log.LEVEL in ("TRACE", "DEBUG")


@dataclass
class _LossRecord:
    loss: torch.Tensor | float = 0
    iter_count: int = 0
    name: str | None = None

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def local_average(self) -> torch.Tensor:
        """This rank's mean loss as a 0-dim float32 tensor, before any cross-rank reduction.

        Returns NaN when nothing was recorded, which is how the caller masks this rank out
        of the average. Deliberately free of ``.item()`` so that a caller reducing many
        records pays one device synchronization for the whole batch instead of one per
        record; see ``_reduce_batched``. The float32 cast makes records stackable even when
        an accumulator kept the model's own dtype, as the per-sequence losses do.
        """
        if self.iter_count == 0:
            return torch.tensor(float("nan"), device="cuda")
        return (self.loss.mean() / self.iter_count).to(torch.float32)

    def get_stat(self, return_valid_mask_sum: bool = False) -> Tuple[float, float]:
        if self.iter_count == 0:
            self.loss = torch.tensor([float("nan")], device="cuda")
            self.iter_count = 1
        self.loss = self.loss.mean()
        avg_loss_tensor = self.loss / self.iter_count
        # Create a mask (1 if valid, 0 if NaN or Inf)
        valid_mask = torch.isfinite(avg_loss_tensor).float()

        # Each interpolation below reads a device tensor and so blocks on the GPU. The
        # message is discarded unless DEBUG is on, so build it only when it will be seen.
        debug = self.name is not None and _debug_enabled()
        msg_str = ""
        if debug:
            msg_str = f"{self.name}: sum_loss={self.loss.item()}/iter_count={self.iter_count}="
            msg_str += f"avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}, "

        # Replace NaN/Inf with 0 to avoid affecting sum
        avg_loss_tensor = torch.where(
            torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.zeros_like(avg_loss_tensor)
        )

        # Reduce across all ranks
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
        dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses
        if debug:
            msg_str += f" | all_reduce: avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}"
        # Compute final average, avoiding division by zero
        valid_mask_sum = valid_mask.item()
        if valid_mask_sum > 0:
            final_avg_loss = (avg_loss_tensor / valid_mask).item()
        else:
            final_avg_loss = 0.0  # Default to zero if all values were invalid
            valid_mask_sum = 0

        avg_loss = final_avg_loss
        if debug:
            msg_str += f" | final: avg_loss={final_avg_loss}"
            log.debug(msg_str, rank0_only=False)
        self.reset()
        if return_valid_mask_sum:
            return avg_loss, valid_mask_sum
        else:
            return avg_loss


def _reduce_batched(
    local_averages: list[torch.Tensor], extra_sums: torch.Tensor
) -> Tuple[list[Tuple[float, float]], list[float]]:
    """Sum-reduce every entry of ``local_averages`` and ``extra_sums`` in one collective.

    ``_LossRecord.get_stat`` issues two all-reduces and several device synchronizations per
    record. Recipes keep one record per dataset and can have hundreds, so calling it in a
    loop costs O(datasets) four-byte collectives per log step; at large world sizes the
    launch latency of those collectives, not the data, dominates the step. Packing the
    batch into a single tensor leaves the arithmetic unchanged while making the cost
    independent of how many records there are.

    Every rank must pass the entries in the same order for the reduction to line up.

    Returns one ``(avg_loss, valid_rank_count)`` pair per entry, matching what
    ``get_stat(return_valid_mask_sum=True)`` would have produced, plus the reduced
    ``extra_sums``. Records are not reset; the caller owns their lifetime.
    """
    count = len(local_averages)
    if count:
        averages = torch.stack(local_averages)
    else:
        averages = torch.empty(0, dtype=torch.float32, device=extra_sums.device)
    # A rank that recorded nothing for an entry reports NaN; masking it out keeps it from
    # dragging the average toward zero, and the mask doubles as the divisor.
    finite = torch.isfinite(averages)
    packed = torch.cat(
        [
            torch.where(finite, averages, torch.zeros_like(averages)),
            finite.to(torch.float32),
            extra_sums.to(torch.float32),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    # The single device-to-host copy for the whole batch. Every read below hits host
    # memory and so does not synchronize.
    host = packed.cpu()
    stats: list[Tuple[float, float]] = []
    for index in range(count):
        valid_rank_count = host[count + index].item()
        stats.append((host[index].item() / valid_rank_count if valid_rank_count > 0 else 0.0, valid_rank_count))
    return stats, host[2 * count :].tolist()


class WandbCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}
        self.final_all_loss_log = {}
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag
        self.unstable_count = torch.zeros(1, device="cuda")
        self._objective_numerator: torch.Tensor | None = None
        self._objective_denominator: torch.Tensor | None = None
        self._union_names: set[str] | None = None
        self._union_names_sorted: list[str] = []

    def _union_dataset_names(self) -> list[str]:
        """Dataset names seen by any rank, in an order every rank agrees on.

        The union only changes when some rank meets a dataset for the first time, which
        after warm-up is rare, so the ``all_gather_object`` that exchanges the names is
        skipped unless it would actually learn something. Whether to refresh has to be
        agreed on collectively: if ranks decided locally they would disagree about which
        collectives to run and deadlock.
        """
        local_names = set(self.final_loss_log_per_dataset)
        distributed_run = dist.is_available() and dist.is_initialized()
        stale = self._union_names is None or not local_names.issubset(self._union_names)
        if distributed_run:
            vote = torch.tensor([float(stale)], device="cuda")
            dist.all_reduce(vote, op=dist.ReduceOp.MAX)
            stale = bool(vote.item())
        if stale:
            if distributed_run:
                gathered: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, sorted(local_names))
            else:
                gathered = [sorted(local_names)]
            self._union_names = {name for names in gathered for name in names}
            self._union_names_sorted = sorted(self._union_names)  # This is very important!
        return self._union_names_sorted

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Accumulate exact VLM objective statistics from every GA microbatch."""
        objective_numerator = output_batch.get("train_objective_numerator")
        objective_denominator = output_batch.get("train_objective_denominator")
        if objective_numerator is not None and objective_denominator is not None:
            if self._objective_numerator is None:
                self._objective_numerator = objective_numerator.detach().clone()
                self._objective_denominator = objective_denominator.detach().clone()
            else:
                self._objective_numerator.add_(objective_numerator)
                assert self._objective_denominator is not None
                self._objective_denominator.add_(objective_denominator)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        dataset_name = data_batch.get("dataset_name", "default")

        # Handle case where dataset_name gets batched into a list
        if isinstance(dataset_name, list):
            # For reasoner, dataset_name will be a list of different datasets
            # For generator, dataset_name is a list of size larger than one,
            #   assume they are all the same
            # Dedup using set to extract identical dataset names
            dataset_name = list(set(dataset_name))

        if dataset_name == "default" and "__url__" in data_batch:
            # try to get the name from url
            dataset_name = ["/".join(data_batch["__url__"][0].split("/")[:-1])]

        for single_dataset_name in dataset_name:
            if single_dataset_name not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[single_dataset_name] = _LossRecord()
                self.final_loss_log_per_dataset[single_dataset_name].name = single_dataset_name
            self.final_loss_log_per_dataset[single_dataset_name].loss += loss.detach().float()
            self.final_loss_log_per_dataset[single_dataset_name].iter_count += 1

        # VLM: per-sequence loss normalization using token counts when available
        if "avg_num_assistant_tokens" in output_batch:
            per_seq_loss = (
                loss
                * output_batch["avg_num_assistant_tokens"]
                * output_batch["batch_size_local"]
                / output_batch["current_num_assistant_tokens"]
            )
            per_seq_key = f"per_seq/{dataset_name}"
            if per_seq_key not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[per_seq_key] = _LossRecord()
                self.final_loss_log_per_dataset[per_seq_key].name = per_seq_key
            self.final_loss_log_per_dataset[per_seq_key].loss += per_seq_loss
            self.final_loss_log_per_dataset[per_seq_key].iter_count += 1

        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        for key in output_batch.keys():
            # Curve can be plotted only on aggregated loss, not per-instance loss
            if not key.startswith("_") and "loss" in key and "per_instance" not in key:
                if key not in self.final_all_loss_log:
                    self.final_all_loss_log[key] = _LossRecord()
                self.final_all_loss_log[key].loss += output_batch[key].detach().float()
                self.final_all_loss_log[key].iter_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            # Step 1: agree on the dataset names, so every rank packs the same entries in
            # the same slots below.
            union_dataset_names = self._union_dataset_names()
            detail_keys = sorted(self.final_all_loss_log)

            # Step 2: collect every per-rank average without touching the host. Ranks that
            # never saw a dataset contribute NaN and are masked out by the reduction.
            missing = torch.tensor(float("nan"), device="cuda")
            local_averages = [self.final_loss_log.local_average()]
            local_averages += [self.final_all_loss_log[key].local_average() for key in detail_keys]
            for dataset_name in union_dataset_names:
                record = self.final_loss_log_per_dataset.get(dataset_name)
                local_averages.append(missing if record is None else record.local_average())

            # Step 3: the scalars ride along in the same collective rather than paying for
            # three more of their own.
            has_objective = self._objective_numerator is not None
            zero = torch.zeros((), device="cuda")
            extra_sums = torch.stack(
                [
                    self.unstable_count.reshape(()).to(torch.float32),
                    self._objective_numerator.sum().to(torch.float32) if has_objective else zero,
                    self._objective_denominator.sum().to(torch.float32) if has_objective else zero,
                ]
            )

            stats, (unstable_count, objective_numerator, objective_denominator) = _reduce_batched(
                local_averages, extra_sums
            )
            self.final_loss_log.reset()
            for key in detail_keys:
                self.final_all_loss_log[key].reset()

            avg_final_loss = stats[0][0]
            avg_final_all_loss = {key: stats[1 + offset][0] for offset, key in enumerate(detail_keys)}

            avg_final_loss_per_dataset = {}
            dataset_offset = 1 + len(detail_keys)
            for offset, dataset_name in enumerate(union_dataset_names):
                avg_loss, valid_mask_sum = stats[dataset_offset + offset]
                if valid_mask_sum > 0:
                    avg_final_loss_per_dataset[dataset_name] = avg_loss

            exact_objective: float | None = None
            if has_objective:
                exact_objective = objective_numerator / max(objective_denominator, 1e-8)

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/loss": (
                            exact_objective if exact_objective is not None else avg_final_loss
                        ),
                        f"train{self.wandb_extra_tag}/unstable_count": unstable_count,
                        "iteration": iteration,
                    }
                )
                if exact_objective is not None:
                    info[f"train{self.wandb_extra_tag}/loss_avg"] = exact_objective
                for key, loss in avg_final_all_loss.items():
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_detail/{key}": loss,
                        }
                    )
                for dataset_name, loss in avg_final_loss_per_dataset.items():
                    tag = ""
                    if "per_seq" in dataset_name:
                        tag = "_per_seq"
                        dataset_name = dataset_name.replace("per_seq/", "")
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_per_data{tag}/{dataset_name}": loss,
                        }
                    )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)

            # reset unstable count
            self.unstable_count.zero_()
            self._objective_numerator = None
            self._objective_denominator = None
            self.final_loss_log_per_dataset = {}
